import numpy as np
import pytest

from image_colour_compression_lab.knn import knn_search


def test_knn_search_returns_nearest_euclidean_references():
    queries = np.array([[0, 0]])
    references = np.array(
        [
            [0, 0],
            [0, 4],
            [3, 4],
        ]
    )

    indices, distances = knn_search(
        queries,
        references,
        k=2,
    )

    assert np.array_equal(
        indices,
        np.array([[0, 1]]),
    )
    assert np.allclose(
        distances,
        np.array([[0.0, 4.0]]),
    )


def test_knn_search_supports_manhattan_distance():
    queries = np.array([[3, 4]])
    references = np.array(
        [
            [0, 0],
            [0, 4],
            [3, 4],
        ]
    )

    indices, distances = knn_search(
        queries,
        references,
        k=2,
        metric="manhattan",
    )

    assert np.array_equal(
        indices,
        np.array([[2, 1]]),
    )
    assert np.array_equal(
        distances,
        np.array([[0.0, 3.0]]),
    )


def test_knn_search_rejects_invalid_k():
    queries = np.array([[0, 0]])
    references = np.array([[0, 0], [1, 1]])

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        knn_search(queries, references, k=0)


def test_knn_search_rejects_too_large_k():
    queries = np.array([[0, 0]])
    references = np.array([[0, 0], [1, 1]])

    with pytest.raises(
        ValueError,
        match="larger than the number of references",
    ):
        knn_search(queries, references, k=3)


def test_knn_search_rejects_unknown_metric():
    queries = np.array([[0, 0]])
    references = np.array([[0, 0]])

    with pytest.raises(
        ValueError,
        match="euclidean.*manhattan",
    ):
        knn_search(
            queries,
            references,
            k=1,
            metric="cosine",
        )
