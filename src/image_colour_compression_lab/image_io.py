from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}
DEFAULT_MAX_WIDTH = 1024
DEFAULT_MAX_HEIGHT = 1024


def resize_if_needed(
    image: Image.Image,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
) -> Image.Image:
    """Resize an image if it exceeds the configured dimensions."""
    if image.width <= max_width and image.height <= max_height:
        return image

    resized_image = image.copy()
    resized_image.thumbnail((max_width, max_height))

    return resized_image


def load_image(
    path: str | Path,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
) -> np.ndarray:
    """Load a PNG or JPEG image and return it as an RGB NumPy array."""
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            "Unsupported image format. Only PNG and JPEG files are supported."
        )

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:
        raise ValueError(f"Could not read image: {image_path}") from error

    try:
        with Image.open(image_path) as image:
            resized_image = resize_if_needed(
                image,
                max_width=max_width,
                max_height=max_height,
            )
            rgb_image = resized_image.convert("RGB")
            array = np.asarray(rgb_image)
    except Exception as error:
        raise ValueError(f"Could not convert image to RGB: {image_path}") from error

    if array.size == 0:
        raise ValueError("Image is empty.")

    return array


def image_to_pixels(image: np.ndarray) -> np.ndarray:
    """Convert an image from (height, width, 3) to (pixels, 3)."""
    height, width, channels = image.shape

    if channels != 3:
        raise ValueError("Expected an RGB image with three channels.")

    return image.reshape(height * width, channels)


def pixels_to_image(
    pixels: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Reconstruct an RGB image from a pixel matrix."""
    expected_shape = (height * width, 3)

    if pixels.shape != expected_shape:
        raise ValueError(
            f"Expected pixels with shape {expected_shape}, but received {pixels.shape}."
        )

    return pixels.reshape(height, width, 3)


def inspect_image(image: np.ndarray) -> dict[str, object]:
    """Return useful metadata about an RGB image array."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")

    height, width, channels = image.shape
    number_of_pixels = height * width
    number_of_unique_colours = np.unique(
        image.reshape(number_of_pixels, channels),
        axis=0,
    ).shape[0]

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(image.dtype),
        "min_value": int(image.min()),
        "max_value": int(image.max()),
        "number_of_pixels": number_of_pixels,
        "number_of_unique_colours": number_of_unique_colours,
        "memory_bytes": int(image.nbytes),
    }
