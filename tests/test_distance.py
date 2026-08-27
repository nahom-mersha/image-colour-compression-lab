import numpy as np

from image_colour_compression_lab.distance import (
    euclidean_distance,
    manhattan_distance,
    pairwise_euclidean_distances,
    pairwise_manhattan_distances,
)


def test_euclidean_distance_returns_known_result():
    first_vector = np.array([1, 2, 3])
    second_vector = np.array([4, 6, 3])

    result = euclidean_distance(
        first_vector,
        second_vector,
    )

    assert result == 5.0


def test_euclidean_distance_is_zero_for_identical_vectors():
    vector = np.array([10, 20, 30])

    assert euclidean_distance(vector, vector) == 0.0


def test_euclidean_distance_is_symmetric():
    first_vector = np.array([1, 5, 9])
    second_vector = np.array([4, 7, 2])

    assert euclidean_distance(
        first_vector,
        second_vector,
    ) == euclidean_distance(
        second_vector,
        first_vector,
    )


def test_manhattan_distance_returns_known_result():
    first_vector = np.array([1, 2, 3])
    second_vector = np.array([4, 6, 3])

    result = manhattan_distance(
        first_vector,
        second_vector,
    )

    assert result == 7.0


def test_manhattan_distance_from_vector_to_itself_is_zero():
    vector = np.array([10, 20, 30])

    assert manhattan_distance(vector, vector) == 0.0


def test_manhattan_distance_is_symmetric():
    first_vector = np.array([1, 5, 9])
    second_vector = np.array([4, 7, 2])

    assert manhattan_distance(
        first_vector,
        second_vector,
    ) == manhattan_distance(
        second_vector,
        first_vector,
    )


def test_pairwise_euclidean_distances():
    queries = np.array(
        [
            [0, 0],
            [3, 4],
        ]
    )
    references = np.array(
        [
            [0, 0],
            [0, 4],
        ]
    )

    result = pairwise_euclidean_distances(
        queries,
        references,
    )

    expected = np.array(
        [
            [0.0, 4.0],
            [5.0, 3.0],
        ]
    )

    assert np.allclose(result, expected)


def test_pairwise_manhattan_distances():
    queries = np.array(
        [
            [0, 0],
            [3, 4],
        ]
    )
    references = np.array(
        [
            [0, 0],
            [0, 4],
        ]
    )

    result = pairwise_manhattan_distances(
        queries,
        references,
    )

    expected = np.array(
        [
            [0.0, 4.0],
            [7.0, 3.0],
        ]
    )

    assert np.array_equal(result, expected)
