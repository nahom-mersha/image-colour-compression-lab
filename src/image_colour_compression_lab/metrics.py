import numpy as np


def count_unique_colours(pixels: np.ndarray) -> int:
    """Count the number of different RGB colours."""
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("Expected pixels with shape (number_of_pixels, 3).")

    return int(np.unique(pixels, axis=0).shape[0])
