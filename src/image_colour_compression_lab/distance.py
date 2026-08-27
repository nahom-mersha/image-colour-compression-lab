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
