"""Regression tests for the DTM redis queue.

Covers the three production incidents:
1. out-of-range priorities vanishing into unscanned queue buckets,
2. retry backoff not being honored by dequeue,
3. processing tasks being lost forever after a worker crash,
plus the get_pending_tasks(limit) total-count semantics.
"""

import unittest

from app.services.dtm_queue import DTMQueue, InvalidPriorityError


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRedis:
    """Minimal in-memory redis stub with lazy TTL expiry."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.strings = {}
        self.expiries = {}
        self.lists = {}
        self.zsets = {}

    def _purge(self, key):
        expiry = self.expiries.get(key)
        if expiry is not None and expiry <= self.clock.now:
            self.strings.pop(key, None)
            self.expiries.pop(key, None)

    # strings -----------------------------------------------------------
    def set(self, key, value):
        self._purge(key)
        self.strings[key] = value

    def get(self, key):
        self._purge(key)
        return self.strings.get(key)

    def setex(self, key, ttl, value):
        self.strings[key] = value
        self.expiries[key] = self.clock.now + ttl

    def delete(self, key):
        self._purge(key)
        existed = key in self.strings
        self.strings.pop(key, None)
        self.expiries.pop(key, None)
        return 1 if existed else 0

    # lists -------------------------------------------------------------
    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def lpop(self, key):
        items = self.lists.get(key) or []
        if not items:
            return None
        value = items.pop(0)
        if not items:
            self.lists.pop(key, None)
        return value

    def llen(self, key):
        return len(self.lists.get(key) or [])

    def lrange(self, key, start, stop):
        items = self.lists.get(key) or []
        if stop == -1:
            return list(items[start:])
        return list(items[start : stop + 1])

    # sorted sets -------------------------------------------------------
    def zadd(self, key, mapping):
        zset = self.zsets.setdefault(key, {})
        added = 0
        for member, score in mapping.items():
            if member not in zset:
                added += 1
            zset[member] = score
        return added

    def zrem(self, key, member):
        zset = self.zsets.get(key) or {}
        if member in zset:
            del zset[member]
            return 1
        return 0

    def zcard(self, key):
        return len(self.zsets.get(key) or {})

    def zrangebyscore(self, key, min_score, max_score):
        zset = self.zsets.get(key) or {}
        lo = float("-inf") if min_score == "-inf" else float(min_score)
        hi = float("inf") if max_score == "+inf" else float(max_score)
        return sorted(m for m, s in zset.items() if lo <= s <= hi)


class DTMQueueTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.redis = FakeRedis(self.clock)
        self.queue = DTMQueue(self.redis, time_func=self.clock)


class TestPriorityRange(DTMQueueTestCase):
    def test_priority_zero_is_rejected(self):
        with self.assertRaises(InvalidPriorityError):
            self.queue.enqueue("t0", {"x": 1}, priority=0)
        self.assertIsNone(self.queue.get_task("t0"))
        self.assertEqual(self.redis.lists, {})
        self.assertEqual(self.queue.get_queue_status()["pending"], 0)

    def test_priority_eleven_is_rejected(self):
        with self.assertRaises(InvalidPriorityError):
            self.queue.enqueue("t11", {"x": 1}, priority=11)
        self.assertIsNone(self.queue.get_task("t11"))
        self.assertNotIn("dtm:queue:11", self.redis.lists)

    def test_non_integer_priority_is_rejected(self):
        for bad in ("3", 2.5, None, True):
            with self.assertRaises(InvalidPriorityError, msg=f"priority={bad!r}"):
                self.queue.enqueue(f"t-{bad}", {}, priority=bad)

    def test_boundary_priorities_are_consumable(self):
        self.queue.enqueue("low", {}, priority=1)
        self.queue.enqueue("high", {}, priority=10)
        status = self.queue.get_queue_status()
        self.assertEqual(status["pending"], 2)
        self.assertEqual(status["pending_by_priority"][1], 1)
        self.assertEqual(status["pending_by_priority"][10], 1)
        self.assertEqual(self.queue.dequeue()["task_id"], "high")
        self.assertEqual(self.queue.dequeue()["task_id"], "low")
        self.assertIsNone(self.queue.dequeue())

    def test_every_enqueued_task_is_visible_and_consumable(self):
        for priority in range(1, 11):
            self.queue.enqueue(f"t{priority}", {}, priority=priority)
        self.assertEqual(self.queue.get_queue_status()["pending"], 10)
        seen = set()
        while True:
            task = self.queue.dequeue()
            if task is None:
                break
            seen.add(task["task_id"])
        self.assertEqual(seen, {f"t{p}" for p in range(1, 11)})


class TestRetryBackoff(DTMQueueTestCase):
    def test_failed_task_is_not_immediately_dequeueable(self):
        self.queue.enqueue("t1", {}, priority=5)
        self.queue.dequeue()
        task = self.queue.mark_failed("t1", error="link down")
        self.assertEqual(task["status"], "retry_wait")
        self.assertEqual(task["retry_count"], 1)
        self.assertEqual(task["next_retry_at"], self.clock.now + 60)

        # The backoff must actually block consumption.
        self.assertIsNone(self.queue.dequeue())
        status = self.queue.get_queue_status()
        self.assertEqual(status["retry_wait"], 1)
        self.assertEqual(status["pending"], 0)
        self.assertEqual(self.queue.get_task("t1")["status"], "retry_wait")

    def test_task_becomes_consumable_after_next_retry_at(self):
        self.queue.enqueue("t1", {}, priority=5)
        self.queue.dequeue()
        self.queue.mark_failed("t1")

        self.clock.advance(59)
        self.assertIsNone(self.queue.dequeue())

        self.clock.advance(1)
        task = self.queue.dequeue()
        self.assertIsNotNone(task)
        self.assertEqual(task["task_id"], "t1")
        self.assertEqual(task["attempts"], 2)
        self.assertEqual(self.queue.get_queue_status()["retry_wait"], 0)

    def test_retry_count_exhaustion_marks_task_failed(self):
        self.queue.enqueue("t1", {}, priority=5, max_retry_attempts=2)
        for expected_retry_count in (1, 2, 3):
            self.assertIsNotNone(self.queue.dequeue())
            task = self.queue.mark_failed("t1")
            self.assertEqual(task["retry_count"], expected_retry_count)
            if expected_retry_count <= 2:
                self.assertEqual(task["status"], "retry_wait")
                self.clock.advance(3600)
        self.assertEqual(task["status"], "failed")
        self.assertIsNone(task["next_retry_at"])
        self.assertIsNone(self.queue.dequeue())

    def test_retry_failed_task_requeues(self):
        self.queue.enqueue("t1", {}, priority=7, max_retry_attempts=0)
        self.queue.dequeue()
        self.queue.mark_failed("t1")
        self.assertEqual(self.queue.get_task("t1")["status"], "failed")

        task = self.queue.retry_failed_task("t1")
        self.assertEqual(task["status"], "pending")
        self.assertEqual(task["retry_count"], 0)
        self.assertEqual(self.queue.get_queue_status()["pending"], 1)
        self.assertEqual(self.queue.dequeue()["task_id"], "t1")


class TestReclaim(DTMQueueTestCase):
    def test_crashed_worker_task_is_reclaimed(self):
        self.queue.enqueue("t1", {}, priority=5)
        self.queue.dequeue()
        # Simulate a crash: no mark_completed / mark_failed ever happens.
        self.assertEqual(self.queue.get_queue_status()["processing"], 1)
        self.assertIsNone(self.queue.dequeue())

        self.clock.advance(3601)
        reclaimed = self.queue.reclaim_expired_processing()
        self.assertEqual(reclaimed, ["t1"])

        status = self.queue.get_queue_status()
        self.assertEqual(status["processing"], 0)
        self.assertEqual(status["pending"], 1)

        task = self.queue.dequeue()
        self.assertEqual(task["task_id"], "t1")
        self.assertEqual(task["attempts"], 2)

    def test_fresh_processing_task_is_not_reclaimed(self):
        self.queue.enqueue("t1", {}, priority=5)
        self.queue.dequeue()
        self.clock.advance(3599)
        self.assertEqual(self.queue.reclaim_expired_processing(), [])
        self.assertEqual(self.queue.get_queue_status()["processing"], 1)

    def test_completed_task_is_not_reclaimed(self):
        self.queue.enqueue("t1", {}, priority=5)
        self.queue.dequeue()
        self.queue.mark_completed("t1")
        self.clock.advance(7200)
        self.assertEqual(self.queue.reclaim_expired_processing(), [])
        self.assertEqual(self.queue.get_task("t1")["status"], "completed")

    def test_reclaim_preserves_priority(self):
        self.queue.enqueue("t1", {}, priority=9)
        self.queue.enqueue("t2", {}, priority=2)
        self.queue.dequeue()  # t1
        self.queue.dequeue()  # t2
        self.clock.advance(3601)
        reclaimed = self.queue.reclaim_expired_processing()
        self.assertEqual(sorted(reclaimed), ["t1", "t2"])
        self.assertEqual(self.queue.dequeue()["task_id"], "t1")
        self.assertEqual(self.queue.dequeue()["task_id"], "t2")


class TestGetPendingTasksLimit(DTMQueueTestCase):
    def test_limit_is_total_across_priorities(self):
        self.queue.enqueue("p1", {}, priority=1)
        self.queue.enqueue("p5", {}, priority=5)
        self.queue.enqueue("p10", {}, priority=10)
        tasks = self.queue.get_pending_tasks(limit=1)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "p10")

    def test_merges_across_buckets_high_priority_first(self):
        for priority in (1, 1, 5, 5, 10):
            self.queue.enqueue(f"t{priority}-{len(self.redis.lists)}", {}, priority=priority)
        tasks = self.queue.get_pending_tasks(limit=3)
        self.assertEqual(len(tasks), 3)
        self.assertEqual([t["priority"] for t in tasks], [10, 5, 5])
        tasks = self.queue.get_pending_tasks(limit=100)
        self.assertEqual([t["priority"] for t in tasks], [10, 5, 5, 1, 1])

    def test_zero_and_negative_limit_return_nothing(self):
        self.queue.enqueue("t1", {}, priority=5)
        self.assertEqual(self.queue.get_pending_tasks(limit=0), [])
        self.assertEqual(self.queue.get_pending_tasks(limit=-3), [])


class TestStableApiSurface(DTMQueueTestCase):
    def test_task_fields_and_return_structures(self):
        task = self.queue.enqueue("t1", {"data": 1}, priority=5)
        expected_fields = {
            "task_id", "payload", "priority", "status", "retry_count",
            "max_retry_attempts", "attempts", "next_retry_at",
            "created_at", "updated_at", "completed_at", "last_error",
        }
        self.assertEqual(set(task.keys()), expected_fields)

        dequeued = self.queue.dequeue()
        self.assertEqual(set(dequeued.keys()), expected_fields)

        completed = self.queue.mark_completed("t1")
        self.assertEqual(set(completed.keys()), expected_fields)
        self.assertEqual(completed["status"], "completed")
        self.assertIsNotNone(completed["completed_at"])

        fetched = self.queue.get_task("t1")
        self.assertEqual(set(fetched.keys()), expected_fields)
        self.assertIsNone(self.queue.get_task("missing"))
        self.assertIsNone(self.queue.mark_completed("missing"))
        self.assertIsNone(self.queue.mark_failed("missing"))
        self.assertIsNone(self.queue.retry_failed_task("missing"))


if __name__ == "__main__":
    unittest.main()
