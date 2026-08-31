import numpy as np


def fit_pca(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA using covariance-matrix eigendecomposition."""
    if data.ndim != 2:
        raise ValueError("Expected data with shape (samples, features).")

    if data.shape[0] < 2:
        raise ValueError("PCA requires at least two samples.")

    mean = np.mean(data, axis=0)
    centered_data = data - mean

    covariance_matrix = (centered_data.T @ centered_data) / (data.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix,
    )

    descending_order = np.argsort(eigenvalues)[::-1]

    explained_variance = eigenvalues[descending_order]
    components = eigenvectors[:, descending_order].T

    total_variance = np.sum(explained_variance)

    if total_variance == 0:
        explained_variance_ratio = np.zeros_like(
            explained_variance,
        )
    else:
        explained_variance_ratio = explained_variance / total_variance

    return (
        mean,
        components,
        explained_variance,
        explained_variance_ratio,
    )


def transform_pca(
    data: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project data into principal-component coordinates."""
    if data.ndim != 2:
        raise ValueError("Expected data with shape (samples, features).")

    if mean.ndim != 1:
        raise ValueError("Expected mean with shape (features,).")

    if components.ndim != 2:
        raise ValueError("Expected components with shape (components, features).")

    if data.shape[1] != mean.shape[0]:
        raise ValueError("Data and mean must have the same number of features.")

    if components.shape[1] != data.shape[1]:
        raise ValueError("Data and components must have matching features.")

    centered_data = data - mean

    return centered_data @ components.T
