import numpy as np
import pytest

from image_colour_compression_lab.metrics import count_unique_colours


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
