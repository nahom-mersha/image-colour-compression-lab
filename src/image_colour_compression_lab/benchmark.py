import time
from collections.abc import Callable
from typing import Any


def measure_runtime(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """Run a function and return its result and elapsed time."""
    start_time = time.perf_counter()

    result = function(*args, **kwargs)

    runtime_seconds = time.perf_counter() - start_time

    return result, runtime_seconds
