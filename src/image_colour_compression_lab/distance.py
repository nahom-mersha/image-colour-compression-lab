import numpy as np


def euclidean_distance(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    """Calculate Euclidean distance between two vectors."""
    return float(np.sqrt(np.sum((first_vector - second_vector) ** 2)))


def manhattan_distance(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    """Calculate Manhattan distance between two vectors."""
    return float(np.sum(np.abs(first_vector - second_vector)))


def pairwise_euclidean_distances(
    queries: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """Calculate Euclidean distances between all query-reference pairs."""
    differences = queries[:, np.newaxis, :] - references[np.newaxis, :, :]

    return np.sqrt(np.sum(differences**2, axis=2))


def pairwise_manhattan_distances(
    queries: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """Calculate Manhattan distances between all query-reference pairs."""
    differences = queries[:, np.newaxis, :] - references[np.newaxis, :, :]

    return np.sum(
        np.abs(differences),
        axis=2,
    )
