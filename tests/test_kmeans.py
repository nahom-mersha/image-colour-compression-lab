import numpy as np
import pytest

from image_colour_compression_lab.kmeans import (
    assign_clusters,
    initialize_centroids,
)


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


def test_assign_clusters_returns_nearest_centroid():
    pixels = np.array(
        [
            [1, 1],
            [9, 9],
            [2, 1],
            [8, 10],
        ]
    )

    centroids = np.array(
        [
            [0, 0],
            [10, 10],
        ]
    )

    assignments, distances = assign_clusters(
        pixels,
        centroids,
    )

    assert np.array_equal(
        assignments,
        np.array([0, 1, 0, 1]),
    )

    assert np.allclose(
        distances,
        np.array(
            [
                np.sqrt(2),
                np.sqrt(2),
                np.sqrt(5),
                2.0,
            ]
        ),
    )
