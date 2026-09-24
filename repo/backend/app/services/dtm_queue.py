"""Deep-sea delay-tolerant messaging (DTM) queue backed by Redis.

Priorities are integers in [MIN_PRIORITY, MAX_PRIORITY] (10 is highest).
Every component (enqueue / dequeue / status / pending listing) uses the
same priority range, so a task can never land in a queue bucket that
nobody scans.

Retry backoff is real: a failed task is parked in a delayed-retry sorted
set keyed by ``next_retry_at`` and only becomes consumable again once
that timestamp is reached.

Processing tasks are tracked in an index sorted by their processing
deadline; ``reclaim_expired_processing`` moves crashed workers' tasks
back to the pending queue so nothing is lost.
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional


class InvalidPriorityError(ValueError):
    """Raised when a priority falls outside the supported range."""


class DTMQueue:
    MIN_PRIORITY = 1
    MAX_PRIORITY = 10
    DEFAULT_PRIORITY = 5
    PROCESSING_TTL_SECONDS = 3600
    BASE_RETRY_DELAY_SECONDS = 60

    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_RETRY_WAIT = "retry_wait"
    STATUS_FAILED = "failed"
    STATUS_COMPLETED = "completed"

    def __init__(
        self,
        redis_client: Any,
        key_prefix: str = "dtm",
        processing_ttl: int = PROCESSING_TTL_SECONDS,
        base_retry_delay: int = BASE_RETRY_DELAY_SECONDS,
        time_func: Callable[[], float] = time.time,
    ) -> None:
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.processing_ttl = processing_ttl
        self.base_retry_delay = base_retry_delay
        self._time_func = time_func

    # ------------------------------------------------------------------
    # key helpers
    # ------------------------------------------------------------------
    def _queue_key(self, priority: int) -> str:
        return f"{self.key_prefix}:queue:{priority}"

    def _task_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:task:{task_id}"

    def _processing_key(self, task_id: str) -> str:
        return f"{self.key_prefix}:processing:{task_id}"

    @property
    def _retry_key(self) -> str:
        return f"{self.key_prefix}:retry"

    @property
    def _processing_index_key(self) -> str:
        return f"{self.key_prefix}:processing_index"

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _validate_priority(self, priority: int) -> int:
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise InvalidPriorityError(
                f"priority must be an int in "
                f"[{self.MIN_PRIORITY}, {self.MAX_PRIORITY}], got {priority!r}"
            )
        if not self.MIN_PRIORITY <= priority <= self.MAX_PRIORITY:
            raise InvalidPriorityError(
                f"priority {priority} out of range "
                f"[{self.MIN_PRIORITY}, {self.MAX_PRIORITY}]"
            )
        return priority

    # ------------------------------------------------------------------
    # task record persistence
    # ------------------------------------------------------------------
    def _save_task(self, task: Dict[str, Any]) -> None:
        self.redis.set(self._task_key(task["task_id"]), json.dumps(task))

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        raw = self.redis.get(self._task_key(task_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    # ------------------------------------------------------------------
    # enqueue / dequeue
    # ------------------------------------------------------------------
    def enqueue(
        self,
        task_id: str,
        payload: Dict[str, Any],
        priority: int = DEFAULT_PRIORITY,
        max_retry_attempts: int = 3,
    ) -> Dict[str, Any]:
        priority = self._validate_priority(priority)
        now = self._time_func()
        task = {
            "task_id": task_id,
            "payload": payload,
            "priority": priority,
            "status": self.STATUS_PENDING,
            "retry_count": 0,
            "max_retry_attempts": max_retry_attempts,
            "attempts": 0,
            "next_retry_at": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "last_error": None,
        }
        self._save_task(task)
        self.redis.rpush(self._queue_key(priority), task_id)
        return task

    def _promote_due_retries(self, now: float) -> None:
        due = self.redis.zrangebyscore(self._retry_key, "-inf", now)
        for task_id in due:
            if isinstance(task_id, bytes):
                task_id = task_id.decode("utf-8")
            if self.redis.zrem(self._retry_key, task_id) == 0:
                continue
            task = self.get_task(task_id)
            if task is None:
                continue
            task["status"] = self.STATUS_PENDING
            task["next_retry_at"] = None
            task["updated_at"] = now
            self._save_task(task)
            self.redis.rpush(self._queue_key(task["priority"]), task_id)

    def dequeue(self) -> Optional[Dict[str, Any]]:
        now = self._time_func()
        self._promote_due_retries(now)
        for priority in range(self.MAX_PRIORITY, self.MIN_PRIORITY - 1, -1):
            task_id = self.redis.lpop(self._queue_key(priority))
            if task_id is None:
                continue
            if isinstance(task_id, bytes):
                task_id = task_id.decode("utf-8")
            task = self.get_task(task_id)
            if task is None:
                continue
            task["status"] = self.STATUS_PROCESSING
            task["attempts"] += 1
            task["updated_at"] = now
            self._save_task(task)
            self.redis.setex(
                self._processing_key(task_id), self.processing_ttl, task_id
            )
            self.redis.zadd(
                self._processing_index_key, {task_id: now + self.processing_ttl}
            )
            return task
        return None

    # ------------------------------------------------------------------
    # completion / failure
    # ------------------------------------------------------------------
    def _clear_processing(self, task_id: str) -> None:
        self.redis.delete(self._processing_key(task_id))
        self.redis.zrem(self._processing_index_key, task_id)

    def mark_completed(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.get_task(task_id)
        if task is None:
            return None
        now = self._time_func()
        self._clear_processing(task_id)
        task["status"] = self.STATUS_COMPLETED
        task["completed_at"] = now
        task["updated_at"] = now
        self._save_task(task)
        return task

    def mark_failed(
        self, task_id: str, error: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        task = self.get_task(task_id)
        if task is None:
            return None
        now = self._time_func()
        self._clear_processing(task_id)
        task["retry_count"] += 1
        task["last_error"] = error
        task["updated_at"] = now
        if task["retry_count"] > task["max_retry_attempts"]:
            task["status"] = self.STATUS_FAILED
            task["next_retry_at"] = None
        else:
            delay = self.base_retry_delay * (2 ** (task["retry_count"] - 1))
            task["status"] = self.STATUS_RETRY_WAIT
            task["next_retry_at"] = now + delay
            self.redis.zadd(self._retry_key, {task_id: task["next_retry_at"]})
        self._save_task(task)
        return task

    def retry_failed_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.get_task(task_id)
        if task is None:
            return None
        if task["status"] != self.STATUS_FAILED:
            return task
        now = self._time_func()
        task["status"] = self.STATUS_PENDING
        task["retry_count"] = 0
        task["next_retry_at"] = None
        task["updated_at"] = now
        self._save_task(task)
        self.redis.rpush(self._queue_key(task["priority"]), task_id)
        return task

    # ------------------------------------------------------------------
    # reclaim
    # ------------------------------------------------------------------
    def reclaim_expired_processing(
        self, now: Optional[float] = None
    ) -> List[str]:
        """Requeue tasks whose processing deadline has passed.

        Returns the ids of reclaimed tasks. Each reclaimed task keeps its
        ``attempts`` counter, so callers can tell which try it is on.
        """
        if now is None:
            now = self._time_func()
        expired = self.redis.zrangebyscore(self._processing_index_key, "-inf", now)
        reclaimed: List[str] = []
        for task_id in expired:
            if isinstance(task_id, bytes):
                task_id = task_id.decode("utf-8")
            if self.redis.zrem(self._processing_index_key, task_id) == 0:
                continue
            self.redis.delete(self._processing_key(task_id))
            task = self.get_task(task_id)
            if task is None or task["status"] != self.STATUS_PROCESSING:
                continue
            task["status"] = self.STATUS_PENDING
            task["updated_at"] = now
            self._save_task(task)
            self.redis.rpush(self._queue_key(task["priority"]), task_id)
            reclaimed.append(task_id)
        return reclaimed

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def get_queue_status(self) -> Dict[str, Any]:
        pending_by_priority = {
            priority: self.redis.llen(self._queue_key(priority))
            for priority in range(self.MIN_PRIORITY, self.MAX_PRIORITY + 1)
        }
        return {
            "pending": sum(pending_by_priority.values()),
            "pending_by_priority": pending_by_priority,
            "processing": self.redis.zcard(self._processing_index_key),
            "retry_wait": self.redis.zcard(self._retry_key),
        }

    def get_pending_tasks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return at most ``limit`` pending tasks, highest priority first."""
        if limit <= 0:
            return []
        tasks: List[Dict[str, Any]] = []
        for priority in range(self.MAX_PRIORITY, self.MIN_PRIORITY - 1, -1):
            remaining = limit - len(tasks)
            if remaining <= 0:
                break
            task_ids = self.redis.lrange(self._queue_key(priority), 0, remaining - 1)
            for task_id in task_ids:
                if isinstance(task_id, bytes):
                    task_id = task_id.decode("utf-8")
                task = self.get_task(task_id)
                if task is not None:
                    tasks.append(task)
        return tasks
