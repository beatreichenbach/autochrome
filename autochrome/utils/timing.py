import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        label = f'{func.__qualname__}:'
        ms = (time.perf_counter() - start_time) * 1000
        logger.info(f'{label: <40}{ms:9.3f}ms')
        return result

    return wrapper


class Timer:
    def __init__(self, label: str | None = None) -> None:
        self.label = label or ''

    def __enter__(self) -> None:
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ms = (time.perf_counter() - self.start_time) * 1000
        logger.info(f'{self.label: <40}{ms:9.3f}ms')
