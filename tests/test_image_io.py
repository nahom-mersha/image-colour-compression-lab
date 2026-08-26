import numpy as np
import pytest
from PIL import Image

from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
    pixels_to_image,
)


def test_load_image_returns_rgb_array(tmp_path):
    image_path = tmp_path / "test.png"

    original_image = Image.new("RGBA", (4, 3), (255, 0, 0, 128))
    original_image.save(image_path)

    image = load_image(image_path)

    assert image.shape == (3, 4, 3)
    assert image.dtype == np.uint8


def test_image_can_be_reshaped_and_reconstructed():
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    pixels = image_to_pixels(image)
    reconstructed = pixels_to_image(pixels, height=2, width=2)

    assert pixels.shape == (4, 3)
    assert np.array_equal(image, reconstructed)


def test_unsupported_format_raises_error(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(file_path)


def test_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        load_image(missing_path)
