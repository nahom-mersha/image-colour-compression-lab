import numpy as np
import pytest
from sklearn.decomposition import PCA

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


def test_fit_pca_components_are_orthonormal():
    data = np.array(
        [
            [10.0, 2.0, 1.0],
            [8.0, 3.0, 2.0],
            [6.0, 4.0, 1.0],
            [4.0, 5.0, 3.0],
            [2.0, 6.0, 5.0],
            [1.0, 8.0, 7.0],
        ]
    )

    _, components, _, _ = fit_pca(data)

    product = components @ components.T

    assert np.allclose(
        product,
        np.eye(components.shape[0]),
    )


def test_fit_pca_matches_sklearn():
    data = np.array(
        [
            [10.0, 2.0, 1.0],
            [8.0, 3.0, 2.0],
            [6.0, 4.0, 1.0],
            [4.0, 5.0, 3.0],
            [2.0, 6.0, 5.0],
            [1.0, 8.0, 7.0],
        ]
    )

    (
        _,
        components,
        explained_variance,
        explained_variance_ratio,
    ) = fit_pca(data)

    sklearn_pca = PCA()
    sklearn_pca.fit(data)

    assert np.allclose(
        explained_variance,
        sklearn_pca.explained_variance_,
    )

    assert np.allclose(
        explained_variance_ratio,
        sklearn_pca.explained_variance_ratio_,
    )

    assert np.allclose(
        np.abs(components),
        np.abs(sklearn_pca.components_),
    )
