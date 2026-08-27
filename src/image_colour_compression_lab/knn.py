import numpy as np

from image_colour_compression_lab.distance import (
    pairwise_euclidean_distances,
    pairwise_manhattan_distances,
)


def knn_search(
    queries: np.ndarray,
    references: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Find the k nearest references for each query."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if k > references.shape[0]:
        raise ValueError("k cannot be larger than the number of references.")

    if metric == "euclidean":
        distances = pairwise_euclidean_distances(
            queries,
            references,
        )
    elif metric == "manhattan":
        distances = pairwise_manhattan_distances(
            queries,
            references,
        )
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'.")

    nearest_indices = np.argsort(
        distances,
        axis=1,
    )[:, :k]

    nearest_distances = np.take_along_axis(
        distances,
        nearest_indices,
        axis=1,
    )

    return nearest_indices, nearest_distances
