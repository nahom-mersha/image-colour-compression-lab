import numpy as np
import pytest

from image_colour_compression_lab.metrics import (
    assignment_error,
    count_unique_colours,
    mean_squared_error,
)


def test_count_unique_colours():
    pixels = np.array(
        [
            [255, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )

    result = count_unique_colours(pixels)

    assert result == 3


def test_count_unique_colours_rejects_invalid_shape():
    pixels = np.array(
        [
            [255, 0],
            [0, 255],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Expected pixels with shape",
    ):
        count_unique_colours(pixels)


def test_mean_squared_error():
    original_pixels = np.array(
        [[100, 120, 200]],
        dtype=np.uint8,
    )
    reconstructed_pixels = np.array(
        [[110, 100, 190]],
        dtype=np.uint8,
    )

    result = mean_squared_error(
        original_pixels,
        reconstructed_pixels,
    )

    assert result == 200.0


def test_mean_squared_error_is_zero_for_identical_pixels():
    pixels = np.array(
        [[100, 120, 200]],
        dtype=np.uint8,
    )

    assert mean_squared_error(pixels, pixels) == 0.0


def test_mean_squared_error_rejects_different_shapes():
    original_pixels = np.array([[100, 120, 200]])
    reconstructed_pixels = np.array([[100, 120, 200], [1, 2, 3]])

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        mean_squared_error(original_pixels, reconstructed_pixels)


def test_assignment_error():
    original_pixels = np.array(
        [
            [100, 120, 200],
            [20, 240, 30],
        ],
        dtype=np.uint8,
    )
    assigned_pixels = np.array(
        [
            [110, 100, 190],
            [30, 230, 40],
        ],
        dtype=np.uint8,
    )

    result = assignment_error(
        original_pixels,
        assigned_pixels,
    )

    assert result == 900.0


def test_assignment_error_is_zero_for_identical_pixels():
    pixels = np.array(
        [
            [100, 120, 200],
            [20, 240, 30],
        ],
        dtype=np.uint8,
    )

    assert assignment_error(pixels, pixels) == 0.0
