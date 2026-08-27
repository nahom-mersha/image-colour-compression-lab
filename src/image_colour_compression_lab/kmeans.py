import numpy as np


def initialize_centroids(
    pixels: np.ndarray,
    k: int,
    random_seed: int = 42,
) -> np.ndarray:
    """Select k different pixels as initial centroids."""
    if pixels.ndim != 2:
        raise ValueError("Expected a two-dimensional pixel array.")

    if pixels.shape[0] == 0:
        raise ValueError("Cannot initialize centroids from empty pixels.")

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

    return pixels[selected_indices].astype(np.float64, copy=True)
