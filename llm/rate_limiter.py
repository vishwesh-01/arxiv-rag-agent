import time
import random
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RateLimiter:
    """Simple client-side throttle plus exponential backoff helper."""

    def __init__(self, min_interval: float = 4.0):
        self.min_interval = max(0.0, min_interval)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def run(self, fn: Callable[[], T], max_retries: int = 5) -> T:
        delay = 5.0
        for attempt in range(max_retries + 1):
            self.wait()
            try:
                return fn()
            except Exception as exc:
                text = str(exc).lower()
                rate_limited = any(
                    x in text
                    for x in ("429", "rate limit", "resource exhausted", "quota")
                )
                if not rate_limited or attempt >= max_retries:
                    raise
                sleep_for = min(60.0, delay) + random.uniform(0, 1.5)
                print(
                    f"Gemini quota/rate limit encountered; "
                    f"retrying in {sleep_for:.1f}s "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_for)
                delay *= 2
        raise RuntimeError("Unreachable")
