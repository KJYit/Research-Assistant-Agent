"""Share an API turn's time budget with graph stages and provider calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from math import inf
from time import monotonic


class ResearchDeadlineExceeded(TimeoutError):
    pass


_deadline = ContextVar("research_deadline", default=inf)


@contextmanager
def research_budget(seconds: float | None):
    if seconds is None:
        yield
        return
    token = _deadline.set(min(_deadline.get(), monotonic() + seconds))
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_seconds() -> float:
    return max(0.0, _deadline.get() - monotonic())


def provider_timeout(limit: float, *, minimum: float = 1, reserve: float = 0) -> float:
    available = min(limit, remaining_seconds() - reserve)
    if available < minimum:
        raise ResearchDeadlineExceeded("Insufficient time for a provider call")
    return available


@contextmanager
def within_budget(lock):
    remaining = remaining_seconds()
    if not lock.acquire(timeout=-1 if remaining == inf else remaining):
        raise ResearchDeadlineExceeded("Timed out waiting for a research lock")
    try:
        if remaining_seconds() == 0:
            raise ResearchDeadlineExceeded("Research deadline exceeded")
        yield
    finally:
        lock.release()
