"""
Circuit breaker for external API calls.

States:
  CLOSED   — normal, all calls go through
  OPEN     — too many failures, calls are blocked immediately
  HALF_OPEN — one probe request allowed; success → CLOSED, failure → OPEN

Usage:
    breaker = CircuitBreaker("openai", failure_threshold=5, recovery_time=60)

    async def call_openai():
        if breaker.is_open():
            raise RuntimeError("OpenAI circuit is open — too many recent failures")
        try:
            result = await some_openai_call()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise
"""
from __future__ import annotations
import time
import logging

logger = logging.getLogger("genesight")


class CircuitBreaker:
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_time: float = 60.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_at: float | None = None

    @property
    def state(self) -> str:
        return self._state

    def is_open(self) -> bool:
        """Return True if the circuit is blocking calls right now."""
        if self._state == self.CLOSED:
            return False

        if self._state == self.OPEN:
            elapsed = time.monotonic() - (self._last_failure_at or 0)
            if elapsed >= self.recovery_time:
                self._state = self.HALF_OPEN
                logger.info("CircuitBreaker [%s] → HALF_OPEN (probe allowed)", self.name)
                return False  # allow one probe
            return True  # still blocking

        # HALF_OPEN: allow the probe through
        return False

    def record_success(self) -> None:
        if self._state != self.CLOSED:
            logger.info("CircuitBreaker [%s] → CLOSED (recovered)", self.name)
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_at = None

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_at = time.monotonic()

        if self._state == self.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                "CircuitBreaker [%s] → OPEN (failures=%d, recovery in %.0fs)",
                self.name, self._failure_count, self.recovery_time,
            )

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self._state,
            "failure_count": self._failure_count,
            "last_failure_at": self._last_failure_at,
        }


# Module-level singletons — shared across all pipeline runs
openai_breaker = CircuitBreaker("openai", failure_threshold=5, recovery_time=60)
ncbi_breaker   = CircuitBreaker("ncbi",   failure_threshold=8, recovery_time=30)
neo4j_breaker  = CircuitBreaker("neo4j",  failure_threshold=3, recovery_time=120)
