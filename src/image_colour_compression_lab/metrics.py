import numpy as np


def count_unique_colours(pixels: np.ndarray) -> int:
    """Count the number of different RGB colours."""
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("Expected pixels with shape (number_of_pixels, 3).")

    return int(np.unique(pixels, axis=0).shape[0])


def mean_squared_error(
    original_pixels: np.ndarray,
    reconstructed_pixels: np.ndarray,
) -> float:
    """Calculate the average squared RGB reconstruction error."""
    if original_pixels.shape != reconstructed_pixels.shape:
        raise ValueError("Pixel arrays must have the same shape.")

    if original_pixels.ndim != 2 or original_pixels.shape[1] != 3:
        raise ValueError("Expected pixels with shape (number_of_pixels, 3).")

    differences = original_pixels.astype(np.float64) - reconstructed_pixels.astype(
        np.float64
    )

    return float(np.mean(differences**2))
