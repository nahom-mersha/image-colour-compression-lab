import numpy as np
import pytest

from image_colour_compression_lab.kmeans import initialize_centroids


def test_initialize_centroids_returns_k_pixels():
    pixels = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]
    )

    centroids = initialize_centroids(
        pixels,
        k=2,
        random_seed=42,
    )

    assert centroids.shape == (2, 2)
    assert centroids.dtype == np.float64


def test_initialize_centroids_is_reproducible():
    pixels = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]
    )

    first_result = initialize_centroids(pixels, k=2, random_seed=42)
    second_result = initialize_centroids(pixels, k=2, random_seed=42)

    assert np.array_equal(first_result, second_result)


def test_initialize_centroids_rejects_too_large_k():
    pixels = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="larger"):
        initialize_centroids(pixels, k=3)
