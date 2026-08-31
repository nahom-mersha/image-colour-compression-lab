import numpy as np
import pytest

from image_colour_compression_lab.pca import (
    fit_pca,
    transform_pca,
)


def test_fit_pca_returns_expected_shapes():
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )

    mean, components, variance, variance_ratio = fit_pca(data)

    assert mean.shape == (3,)
    assert components.shape == (3, 3)
    assert variance.shape == (3,)
    assert variance_ratio.shape == (3,)


def test_fit_pca_mean_centres_data():
    data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    mean, _, _, _ = fit_pca(data)

    assert np.allclose(
        mean,
        np.array([3.0, 4.0]),
    )


def test_fit_pca_explained_ratios_sum_to_one():
    data = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )

    _, _, _, variance_ratio = fit_pca(data)

    assert np.sum(variance_ratio) == pytest.approx(1.0)


def test_transform_pca_returns_projected_data():
    data = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )

    mean, components, _, _ = fit_pca(data)

    transformed_data = transform_pca(
        data,
        mean,
        components,
    )

    assert transformed_data.shape == (3, 2)
