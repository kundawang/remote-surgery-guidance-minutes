"""Regression tests for the DTM redis queue.

The incidents that motivated these tests:
1. priorities 0 / 11 were accepted and landed in unscanned buckets
   (dtm:queue:0 / dtm:queue:11), invisible to dequeue and status.
2. mark_failed computed next_retry_at but pushed the task straight back
   onto its ready queue, so the 60s backoff never applied.
3. dequeued tasks only lived under a 1h-TTL processing key; a worker
   crash lost the task forever and nothing could reclaim it.

A small in-memory FakeRedis stub stands in for redis-py.
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.dtm_queue import (  # noqa: E402
    DTMQueue,
    InvalidPriorityError,
)


class FakeRedis:
    """Minimal redis-py surface used by DTMQueue."""

    def __init__(self):
        self.lists = {}
        self.strings = {}
        self.zsets = {}

    # strings
    def set(self, key, value):
        self.strings[key] = value

    def get(self, key):
        return self.strings.get(key)

    def setex(self, key, ttl, value):
        self.strings[key] = value

    def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self.strings:
                del self.strings[key]
                removed += 1
            if key in self.zsets:
                del self.zsets[key]
                removed += 1
            if key in self.lists:
                del self.lists[key]
                removed += 1
        return removed

    # lists
    def rpush(self, key, *values):
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    def lpop(self, key):
        bucket = self.lists.get(key)
        if not bucket:
            return None
        return bucket.pop(0)

    def llen(self, key):
        return len(self.lists.get(key, []))

    def lrange(self, key, start, stop):
        return self.lists.get(key, [])[start : stop + 1]

    # sorted sets
    def zadd(self, key, mapping):
        zset = self.zsets.setdefault(key, {})
        for member, score in mapping.items():
            zset[member] = float(score)
        return len(mapping)

    def zrem(self, key, *members):
        zset = self.zsets.get(key, {})
        removed = 0
        for member in members:
            if member in zset:
                del zset[member]
                removed += 1
        return removed

    def zcard(self, key):
        return len(self.zsets.get(key, {}))

    def zrangebyscore(self, key, min_score, max_score):
        if min_score == "-inf":
            min_score = float("-inf")
        if max_score == "+inf":
            max_score = float("inf")
        zset = self.zsets.get(key, {})
        return [
            member
            for member, score in sorted(zset.items(), key=lambda item: item[1])
            if min_score <= score <= max_score
        ]


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.value = start

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds
        return self.value


class PriorityValidationTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.clock = FakeClock()
        self.queue = DTMQueue(
            self.redis, processing_ttl=10, time_func=self.clock
        )

    def test_out_of_range_priorities_are_rejected(self):
        for bad_priority in (0, 11, -1, 100):
            with self.subTest(priority=bad_priority):
                with self.assertRaises(InvalidPriorityError):
                    self.queue.enqueue(f"t{bad_priority}", {}, bad_priority)

    def test_non_integer_priority_is_rejected(self):
        for bad_priority in (1.0, "5", None, True):
            with self.subTest(priority=bad_priority):
                with self.assertRaises(InvalidPriorityError):
                    self.queue.enqueue("t", {}, bad_priority)

    def test_invalid_priority_leaves_no_orphan_keys(self):
        with self.assertRaises(InvalidPriorityError):
            self.queue.enqueue("t", {}, 0)
        with self.assertRaises(InvalidPriorityError):
            self.queue.enqueue("t", {}, 11)
        self.assertNotIn("dtm:queue:0", self.redis.lists)
        self.assertNotIn("dtm:queue:11", self.redis.lists)
        self.assertEqual(self.queue.get_queue_status()["pending"], 0)

    def test_boundary_priorities_are_consumable_and_counted(self):
        self.queue.enqueue("low", {"x": 1}, priority=1)
        self.queue.enqueue("high", {"x": 2}, priority=10)
        status = self.queue.get_queue_status()
        self.assertEqual(status["pending"], 2)
        self.assertEqual(status["pending_by_priority"][1], 1)
        self.assertEqual(status["pending_by_priority"][10], 1)
        # dequeue scans high -> low
        first = self.queue.dequeue()
        second = self.queue.dequeue()
        self.assertEqual(first["task_id"], "high")
        self.assertEqual(second["task_id"], "low")
        self.assertIsNone(self.queue.dequeue())

    def test_status_covers_exactly_the_scanned_range(self):
        for priority in range(DTMQueue.MIN_PRIORITY, DTMQueue.MAX_PRIORITY + 1):
            self.queue.enqueue(f"t{priority}", {}, priority=priority)
        by_priority = self.queue.get_queue_status()["pending_by_priority"]
        self.assertEqual(set(by_priority), set(range(1, 11)))
        self.assertEqual(sum(by_priority.values()), 10)


class RetryBackoffTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.clock = FakeClock()
        self.queue = DTMQueue(
            self.redis,
            processing_ttl=10,
            base_retry_delay=60,
            time_func=self.clock,
        )

    def dequeue_one(self, task_id="t"):
        task = self.queue.dequeue()
        self.assertIsNotNone(task)
        self.assertEqual(task["task_id"], task_id)
        return task

    def test_failed_task_is_not_dequeueable_before_next_retry_at(self):
        self.queue.enqueue("t", {}, priority=5, max_retry_attempts=3)
        self.dequeue_one()
        failed = self.queue.mark_failed("t", error="link down")

        self.assertEqual(failed["status"], DTMQueue.STATUS_RETRY_WAIT)
        self.assertEqual(failed["retry_count"], 1)
        self.assertEqual(failed["next_retry_at"], self.clock() + 60)
        # parked in the delay zset, not pushed back to a ready queue
        self.assertEqual(self.redis.llen("dtm:queue:5"), 0)
        self.assertEqual(self.queue.get_queue_status()["pending"], 0)
        self.assertEqual(self.queue.get_queue_status()["retry_wait"], 1)

        # 59 seconds later: still waiting
        self.clock.advance(59)
        self.assertIsNone(self.queue.dequeue())
        self.assertEqual(
            self.queue.get_task("t")["status"], DTMQueue.STATUS_RETRY_WAIT
        )
        self.assertEqual(self.queue.get_queue_status()["pending"], 0)

        # at t+60 the task becomes consumable again
        self.clock.advance(1)
        retried = self.queue.dequeue()
        self.assertIsNotNone(retried)
        self.assertEqual(retried["task_id"], "t")
        self.assertEqual(retried["status"], DTMQueue.STATUS_PROCESSING)
        self.assertEqual(retried["attempts"], 2)
        self.assertEqual(self.queue.get_queue_status()["retry_wait"], 0)

    def test_exponential_backoff_doubles(self):
        self.queue.enqueue("t", {}, max_retry_attempts=5)
        start = self.clock()
        self.dequeue_one()
        first = self.queue.mark_failed("t", "err")
        self.assertEqual(first["next_retry_at"] - start, 60)

        self.clock.advance(60)
        self.dequeue_one()
        second = self.queue.mark_failed("t", "err")
        self.assertEqual(second["next_retry_at"] - self.clock(), 120)

        self.clock.advance(120)
        self.dequeue_one()
        third = self.queue.mark_failed("t", "err")
        self.assertEqual(third["next_retry_at"] - self.clock(), 240)

    def test_retry_budget_exhausted_task_is_not_requeued(self):
        self.queue.enqueue("t", {}, max_retry_attempts=2)
        # attempts 1 and 2 fail but retries remain
        for expected_count in (1, 2):
            self.dequeue_one()
            outcome = self.queue.mark_failed("t", "err")
            self.assertEqual(outcome["retry_count"], expected_count)
            self.assertEqual(outcome["status"], DTMQueue.STATUS_RETRY_WAIT)
            self.clock.advance(60 * (2 ** (expected_count - 1)))
        # attempt 3: retry_count (3) > max_retry_attempts (2) -> dead letter
        self.dequeue_one()
        final = self.queue.mark_failed("t", "err")
        self.assertEqual(final["retry_count"], 3)
        self.assertEqual(final["status"], DTMQueue.STATUS_FAILED)
        self.assertIsNone(final["next_retry_at"])
        self.clock.advance(10_000)
        self.assertIsNone(self.queue.dequeue())

    def test_retry_failed_task_rearms_with_fields_intact(self):
        self.queue.enqueue("t", {"k": "v"}, priority=7, max_retry_attempts=0)
        self.dequeue_one()
        failed = self.queue.mark_failed("t", "boom")
        self.assertEqual(failed["status"], DTMQueue.STATUS_FAILED)

        rearmed = self.queue.retry_failed_task("t")
        self.assertEqual(rearmed["status"], DTMQueue.STATUS_PENDING)
        self.assertEqual(rearmed["retry_count"], 0)
        self.assertIsNone(rearmed["next_retry_at"])
        self.assertEqual(rearmed["priority"], 7)
        self.assertEqual(rearmed["payload"], {"k": "v"})
        self.assertIsNotNone(self.queue.dequeue())

    def test_pending_listing_excludes_retry_wait_tasks(self):
        self.queue.enqueue("t", {})
        self.dequeue_one()
        self.queue.mark_failed("t", "err")
        self.assertEqual(self.queue.get_pending_tasks(10), [])


class ProcessingReclaimTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.clock = FakeClock()
        self.queue = DTMQueue(
            self.redis, processing_ttl=60, time_func=self.clock
        )

    def test_expired_processing_task_is_requeued(self):
        self.queue.enqueue("crashed", {}, priority=3)
        taken = self.queue.dequeue()
        self.assertEqual(taken["attempts"], 1)
        self.assertEqual(self.queue.get_queue_status()["processing"], 1)

        # worker dies; nothing comes back before the deadline
        self.clock.advance(59)
        self.assertEqual(self.queue.reclaim_expired_processing(), [])
        self.assertEqual(self.queue.get_queue_status()["processing"], 1)
        self.assertIsNone(self.queue.dequeue())

        # deadline passes: reclaim puts it back on the ready queue
        self.clock.advance(1)
        reclaimed = self.queue.reclaim_expired_processing()
        self.assertEqual(reclaimed, ["crashed"])
        task = self.queue.get_task("crashed")
        self.assertEqual(task["status"], DTMQueue.STATUS_PENDING)
        status = self.queue.get_queue_status()
        self.assertEqual(status["processing"], 0)
        self.assertEqual(status["pending"], 1)

        retaken = self.queue.dequeue()
        self.assertIsNotNone(retaken)
        self.assertEqual(retaken["task_id"], "crashed")
        self.assertEqual(retaken["attempts"], 2)

    def test_completed_and_in_flight_tasks_are_not_reclaimed(self):
        self.queue.enqueue("done", {}, priority=2)
        self.queue.enqueue("alive", {}, priority=4)
        self.queue.dequeue()  # takes "alive" (higher priority)
        self.queue.mark_completed("alive")
        self.clock.advance(10)

        # enqueue + take "done", then let only "done" expire
        self.queue.dequeue()
        self.clock.advance(61)
        reclaimed = self.queue.reclaim_expired_processing()
        self.assertEqual(reclaimed, ["done"])
        done = self.queue.get_task("done")
        self.assertEqual(done["status"], DTMQueue.STATUS_PENDING)
        alive = self.queue.get_task("alive")
        self.assertEqual(alive["status"], DTMQueue.STATUS_COMPLETED)

    def test_reclaim_only_runs_once_per_expiry(self):
        self.queue.enqueue("t", {})
        self.queue.dequeue()
        self.clock.advance(61)
        self.assertEqual(self.queue.reclaim_expired_processing(), ["t"])
        self.assertEqual(self.queue.reclaim_expired_processing(), [])

    def test_multiple_expired_tasks_keep_attempt_counts(self):
        self.queue.enqueue("a", {}, priority=1)
        self.queue.enqueue("b", {}, priority=2)
        self.queue.dequeue()  # b
        self.queue.dequeue()  # a
        self.clock.advance(61)
        reclaimed = self.queue.reclaim_expired_processing()
        self.assertEqual(sorted(reclaimed), ["a", "b"])
        first = self.queue.dequeue()
        second = self.queue.dequeue()
        self.assertEqual(first["task_id"], "b")
        self.assertEqual(first["attempts"], 2)
        self.assertEqual(second["attempts"], 2)


class PendingLimitTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.clock = FakeClock()
        self.queue = DTMQueue(
            self.redis, processing_ttl=10, time_func=self.clock
        )

    def test_limit_caps_total_across_priorities(self):
        self.queue.enqueue("p1-a", {}, priority=1)
        self.queue.enqueue("p5-a", {}, priority=5)
        self.queue.enqueue("p10-a", {}, priority=10)
        self.assertEqual(len(self.queue.get_pending_tasks(1)), 1)
        self.assertEqual(len(self.queue.get_pending_tasks(2)), 2)
        self.assertEqual(len(self.queue.get_pending_tasks(10)), 3)

    def test_results_are_highest_priority_first(self):
        self.queue.enqueue("p1-a", {}, priority=1)
        self.queue.enqueue("p1-b", {}, priority=1)
        self.queue.enqueue("p10-a", {}, priority=10)
        ids = [task["task_id"] for task in self.queue.get_pending_tasks(2)]
        self.assertEqual(ids, ["p10-a", "p1-a"])

    def test_non_positive_limit_returns_empty(self):
        self.queue.enqueue("t", {})
        self.assertEqual(self.queue.get_pending_tasks(0), [])


if __name__ == "__main__":
    unittest.main()
