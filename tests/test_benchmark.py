import time

from image_colour_compression_lab.benchmark import measure_runtime


def test_measure_runtime_returns_result_and_nonnegative_time():
    def example_function(value):
        time.sleep(0.01)
        return value * 2

    result, runtime_seconds = measure_runtime(
        example_function,
        3,
    )

    assert result == 6
    assert runtime_seconds >= 0.0
