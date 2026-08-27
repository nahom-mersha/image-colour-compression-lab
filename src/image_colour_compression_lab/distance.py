import numpy as np


def _validate_vector_pair(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first_vector = np.asarray(first_vector, dtype=float)
    second_vector = np.asarray(second_vector, dtype=float)

    if first_vector.ndim != 1 or second_vector.ndim != 1:
        raise ValueError("Both vectors must be one-dimensional.")

    if first_vector.shape != second_vector.shape:
        raise ValueError("Vectors must have the same length.")

    return first_vector, second_vector


def _validate_pairwise_inputs(
    queries: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(queries, dtype=float)
    references = np.asarray(references, dtype=float)

    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError("Queries and references must be two-dimensional.")

    if queries.shape[1] != references.shape[1]:
        raise ValueError(
            "Queries and references must have the same number of features."
        )

    return queries, references


def euclidean_distance(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    """Calculate Euclidean distance between two vectors."""
    first_vector, second_vector = _validate_vector_pair(
        first_vector,
        second_vector,
    )

    return float(np.sqrt(np.sum((first_vector - second_vector) ** 2)))


def manhattan_distance(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    """Calculate Manhattan distance between two vectors."""
    first_vector, second_vector = _validate_vector_pair(
        first_vector,
        second_vector,
    )

    return float(np.sum(np.abs(first_vector - second_vector)))


def pairwise_euclidean_distances(
    queries: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """Calculate Euclidean distances between all query-reference pairs."""
    queries, references = _validate_pairwise_inputs(
        queries,
        references,
    )

    differences = queries[:, np.newaxis, :] - references[np.newaxis, :, :]

    return np.sqrt(np.sum(differences**2, axis=2))


def pairwise_manhattan_distances(
    queries: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """Calculate Manhattan distances between all query-reference pairs."""
    queries, references = _validate_pairwise_inputs(
        queries,
        references,
    )

    differences = queries[:, np.newaxis, :] - references[np.newaxis, :, :]

    return np.sum(
        np.abs(differences),
        axis=2,
    )
