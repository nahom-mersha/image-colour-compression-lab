import pytest

from image_colour_compression_lab.statistics import calculate_mean


def test_calculate_mean() -> None:
    assert calculate_mean([2.0, 4.0, 6.0]) == 4.0


def test_calculate_mean_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        calculate_mean([])
