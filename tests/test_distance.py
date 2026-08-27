import numpy as np

from image_colour_compression_lab.distance import euclidean_distance


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
