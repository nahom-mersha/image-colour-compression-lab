import numpy as np


def random_palette_compression(
    pixels: np.ndarray,
    k: int,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress pixels using a random fixed palette.

    The palette is selected once from the input pixels. Each pixel is then
    replaced by its nearest palette colour.
    """
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("Expected pixels with shape (number_of_pixels, 3).")

    if pixels.shape[0] == 0:
        raise ValueError("Cannot compress an empty pixel array.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if k > pixels.shape[0]:
        raise ValueError("k cannot be larger than the number of pixels.")

    random_generator = np.random.default_rng(random_seed)

    selected_indices = random_generator.choice(
        pixels.shape[0],
        size=k,
        replace=False,
    )

    palette = pixels[selected_indices].copy()

    distances = np.sum(
        (pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2,
        axis=2,
    )

    nearest_palette_indices = np.argmin(distances, axis=1)
    compressed_pixels = palette[nearest_palette_indices]

    return compressed_pixels, palette
