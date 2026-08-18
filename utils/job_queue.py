"""
Async job queue — decouples HTTP request handling from pipeline execution.

Architecture:
  POST /query  →  enqueue(job_id, query, clerk_id)  →  returns immediately
  Worker pool  →  dequeues jobs, runs pipeline, updates jobs dict

Benefits over FastAPI BackgroundTasks:
  - Jobs wait in queue instead of all starting simultaneously
  - MAX_WORKERS controls actual concurrency (no semaphore needed per job)
  - Queue depth is visible (backpressure / capacity planning)
  - Graceful shutdown: workers finish current job then stop
  - Dead-letter: failed jobs are logged with full context

Usage (in main.py lifespan):
    queue = JobQueue(workers=3)
    queue.set_handler(run_pipeline_inner)
    await queue.start()
    yield
    await queue.stop()
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable, Awaitable

logger = logging.getLogger("genesight.queue")


class JobQueue:
    def __init__(self, workers: int = 3, max_size: int = 50) -> None:
        self.workers = workers
        self.max_size = max_size
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=max_size)
        self._handler: Callable[..., Awaitable[None]] | None = None
        self._tasks: list[asyncio.Task] = []
        self._started = False

    def set_handler(self, fn: Callable[..., Awaitable[None]]) -> None:
        self._handler = fn

    async def enqueue(self, *args) -> bool:
        """
        Add a job to the queue. Returns False if the queue is full.
        """
        try:
            self._queue.put_nowait(args)
            logger.info("Enqueued job — queue depth=%d/%d", self._queue.qsize(), self.max_size)
            return True
        except asyncio.QueueFull:
            logger.warning("Queue full (%d/%d) — rejecting job", self._queue.qsize(), self.max_size)
            return False

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for i in range(self.workers):
            task = asyncio.create_task(self._worker(i), name=f"queue-worker-{i}")
            self._tasks.append(task)
        logger.info("JobQueue started — %d workers, capacity=%d", self.workers, self.max_size)

    async def stop(self) -> None:
        """Signal workers to stop after draining the queue."""
        for _ in self._tasks:
            await self._queue.put(None)  # poison pill
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("JobQueue stopped")

    async def _worker(self, worker_id: int) -> None:
        logger.info("Worker-%d ready", worker_id)
        while True:
            item = await self._queue.get()
            if item is None:  # poison pill → shutdown
                self._queue.task_done()
                logger.info("Worker-%d shutting down", worker_id)
                break
            try:
                t0 = time.monotonic()
                await self._handler(*item)
                logger.info(
                    "Worker-%d finished job in %.1fs — queue depth=%d",
                    worker_id, time.monotonic() - t0, self._queue.qsize(),
                )
            except Exception as e:
                logger.error("Worker-%d unhandled error: %s", worker_id, e)
            finally:
                self._queue.task_done()


# Module-level singleton — imported by main.py
job_queue = JobQueue(workers=3, max_size=50)
