"""
Retry utilities with exponential backoff — applied to all external API calls.

tenacity is already a project dependency (used by GenomicsDBAgent).

Two presets:
  with_network_retry  — NCBI / PubMed / UniProt HTTP calls
                        3 attempts, 2 → 4 → 8 s backoff, retries on any Exception
  with_openai_retry   — OpenAI API calls
                        4 attempts, 1 → 2 → 4 → 8 s backoff,
                        only retries transient errors (RateLimitError, APIConnectionError)

Usage (decorator):
    from utils.retry import with_network_retry, with_openai_retry

    @with_network_retry
    async def _fetch_ncbi(self, ...): ...

Usage (wrap a call inline):
    from utils.retry import network_retry_call
    result = await network_retry_call(some_coroutine_fn, arg1, arg2)
"""
from __future__ import annotations
import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Awaitable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger("genesight.retry")

# ── Transient error types we want to retry on ─────────────────────────────────

_NETWORK_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
    Exception,         # broad catch for httpx/biopython network errors
)

try:
    from openai import RateLimitError, APIConnectionError, APITimeoutError
    _OPENAI_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError)
except ImportError:
    _OPENAI_EXCEPTIONS = (Exception,)

# ── Decorator presets ─────────────────────────────────────────────────────────

def with_network_retry(fn: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
    """
    Retry decorator for NCBI / PubMed / external HTTP calls.
    3 attempts total, 2 → 4 → 8 s exponential wait.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(_NETWORK_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)
    return wrapper


def with_openai_retry(fn: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
    """
    Retry decorator for OpenAI API calls.
    4 attempts total, 1 → 2 → 4 → 8 s wait.
    Only retries transient OpenAI errors (RateLimit, Connection, Timeout).
    """
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_OPENAI_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)
    return wrapper


# ── Inline helper for one-off use without a decorator ─────────────────────────

async def network_retry_call(fn: Callable[..., Awaitable], *args: Any, **kwargs: Any) -> Any:
    """
    Call an async function with network retry semantics.
    Useful when you can't decorate the function (e.g. third-party code).

    Example:
        result = await network_retry_call(client.get, url, params=params)
    """
    decorated = with_network_retry(fn)
    return await decorated(*args, **kwargs)
