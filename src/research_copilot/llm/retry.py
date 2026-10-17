import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0


def call_with_retry[T](
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    attempt = 1
    while True:
        try:
            return fn()
        except retry_on as exc:
            if attempt >= policy.max_attempts:
                raise
            ceiling = min(policy.max_delay, policy.base_delay * 2 ** (attempt - 1))
            delay = rng() * ceiling
            logger.warning(
                "transient error attempt=%d/%d retry_in=%.2fs error=%s",
                attempt,
                policy.max_attempts,
                delay,
                exc,
            )
            sleep(delay)
            attempt += 1
