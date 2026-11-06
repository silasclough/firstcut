"""Retry with exponential backoff (for network calls and flaky subprocesses)."""

from __future__ import annotations

import functools
import random
import time
from typing import Callable, TypeVar

from .log import get_logger

T = TypeVar("T")
log = get_logger("retry")


class PermanentError(Exception):
    """Raise to stop retrying immediately (bad input, auth failure, policy block)."""


def backoff_delays(attempts: int, base: float = 2.0, cap: float = 60.0, jitter: float = 0.25) -> list[float]:
    """Delays slept between attempts: base, 2*base, 4*base ... capped, +-jitter fraction."""
    out = []
    for i in range(max(0, attempts - 1)):
        d = min(cap, base * (2 ** i))
        out.append(max(0.0, d * (1 + random.uniform(-jitter, jitter))))
    return out


def call_with_retry(fn: Callable[..., T], *args, attempts: int = 4, base: float = 2.0, cap: float = 60.0,
                    retry_on: tuple[type[BaseException], ...] = (Exception,), what: str = "",
                    sleep: Callable[[float], None] = time.sleep, **kwargs) -> T:
    delays = backoff_delays(attempts, base, cap)
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except PermanentError:
            raise
        except retry_on as e:  # type: ignore[misc]
            last = e
            if i >= attempts - 1:
                break
            delay = delays[i]
            log.warning(f"{what or getattr(fn, '__name__', 'call')} failed ({type(e).__name__}: {e}); "
                        f"retry {i + 1}/{attempts - 1} in {delay:.1f}s")
            sleep(delay)
    assert last is not None
    raise last


def retrying(attempts: int = 4, base: float = 2.0, cap: float = 60.0,
             retry_on: tuple[type[BaseException], ...] = (Exception,)):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return call_with_retry(fn, *args, attempts=attempts, base=base, cap=cap,
                                   retry_on=retry_on, what=fn.__qualname__, **kwargs)
        return wrapper
    return deco
