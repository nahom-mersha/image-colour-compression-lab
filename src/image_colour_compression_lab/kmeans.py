import numpy as np

from image_colour_compression_lab.distance import (
    pairwise_euclidean_distances,
)


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


def assign_clusters(
    pixels: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign every pixel to its nearest centroid."""
    if pixels.ndim != 2:
        raise ValueError("Pixels must be a two-dimensional array.")

    if centroids.ndim != 2:
        raise ValueError("Centroids must be a two-dimensional array.")

    if pixels.shape[1] != centroids.shape[1]:
        raise ValueError("Pixels and centroids must have the same number of features.")

    distances = pairwise_euclidean_distances(
        pixels,
        centroids,
    )

    nearest_centroid_indices = np.argmin(
        distances,
        axis=1,
    )

    nearest_distances = np.take_along_axis(
        distances,
        nearest_centroid_indices[:, np.newaxis],
        axis=1,
    ).ravel()

    return nearest_centroid_indices, nearest_distances


def update_centroids(
    pixels: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Calculate the mean feature values for each cluster."""
    if pixels.ndim != 2:
        raise ValueError("Pixels must be a two-dimensional array.")

    if assignments.ndim != 1:
        raise ValueError("Assignments must be one-dimensional.")

    if centroids.ndim != 2:
        raise ValueError("Centroids must be a two-dimensional array.")

    if pixels.shape[0] != assignments.shape[0]:
        raise ValueError("Each pixel must have one assignment.")

    if pixels.shape[1] != centroids.shape[1]:
        raise ValueError("Pixels and centroids must have the same number of features.")

    updated_centroids = centroids.astype(
        np.float64,
        copy=True,
    )

    for centroid_index in range(centroids.shape[0]):
        cluster_pixels = pixels[assignments == centroid_index]

        if cluster_pixels.shape[0] > 0:
            updated_centroids[centroid_index] = np.mean(
                cluster_pixels,
                axis=0,
            )

    return updated_centroids
