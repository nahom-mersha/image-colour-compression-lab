from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import TypeVar

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
)
from image_colour_compression_lab.kmeans import fit_kmeans
from image_colour_compression_lab.knn import knn_search

T = TypeVar("T")


def median_runtime(
    function: Callable[[], T],
    repeats: int = 3,
) -> tuple[T, float]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    runtimes = []

    start = perf_counter()
    result = function()
    runtimes.append(perf_counter() - start)

    for _ in range(repeats - 1):
        start = perf_counter()
        result = function()
        runtimes.append(perf_counter() - start)

    return result, float(np.median(runtimes))


def main() -> None:
    rng = np.random.default_rng(42)

    image = load_image(Path("data/samples/sample_image.jpg"))
    pixels = image_to_pixels(image).astype(float)

    indices = rng.choice(
        len(pixels),
        size=5000,
        replace=False,
    )
    sampled_pixels = pixels[indices]

    results = []

    # --------------------------------------------------
    # KNN comparison
    # --------------------------------------------------

    references = sampled_pixels[:2000]
    queries = sampled_pixels[2000:2050]
    neighbour_count = 5

    (
        scratch_knn_result,
        scratch_knn_runtime,
    ) = median_runtime(
        lambda: knn_search(
            queries,
            references,
            k=neighbour_count,
            metric="euclidean",
        )
    )

    scratch_indices, scratch_distances = scratch_knn_result

    sklearn_knn = NearestNeighbors(
        n_neighbors=neighbour_count,
        algorithm="brute",
        metric="euclidean",
    )
    sklearn_knn.fit(references)

    (
        sklearn_knn_result,
        sklearn_knn_runtime,
    ) = median_runtime(lambda: sklearn_knn.kneighbors(queries))

    sklearn_distances, sklearn_indices = sklearn_knn_result

    knn_indices_match = np.array_equal(
        scratch_indices,
        sklearn_indices,
    )

    knn_distances_match = np.allclose(
        scratch_distances,
        sklearn_distances,
    )

    results.append(
        {
            "algorithm": "KNN",
            "implementation": "from_scratch",
            "runtime_seconds": scratch_knn_runtime,
            "inertia": np.nan,
        }
    )

    results.append(
        {
            "algorithm": "KNN",
            "implementation": "scikit_learn",
            "runtime_seconds": sklearn_knn_runtime,
            "inertia": np.nan,
        }
    )

    # --------------------------------------------------
    # K-means comparison
    # --------------------------------------------------

    k = 8

    (
        scratch_kmeans_result,
        scratch_kmeans_runtime,
    ) = median_runtime(
        lambda: fit_kmeans(
            sampled_pixels,
            k=k,
            max_iterations=20,
            random_seed=42,
            initialization="kmeans++",
        )
    )

    _, _, inertia_history = scratch_kmeans_result

    scratch_inertia = inertia_history[-1]

    def run_sklearn_kmeans():
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=1,
            max_iter=20,
            tol=1e-4,
            random_state=42,
            algorithm="lloyd",
        )
        model.fit(sampled_pixels)
        return model

    (
        sklearn_kmeans,
        sklearn_kmeans_runtime,
    ) = median_runtime(run_sklearn_kmeans)

    results.append(
        {
            "algorithm": "K-means",
            "implementation": "from_scratch",
            "runtime_seconds": scratch_kmeans_runtime,
            "inertia": scratch_inertia,
        }
    )

    results.append(
        {
            "algorithm": "K-means",
            "implementation": "scikit_learn",
            "runtime_seconds": sklearn_kmeans_runtime,
            "inertia": sklearn_kmeans.inertia_,
        }
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    dataframe = pd.DataFrame(results)

    output_path = Path("reports/benchmarks/sklearn_comparison.csv")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print("\nKNN correctness:")
    print(f"Indices match: {knn_indices_match}")
    print(f"Distances match: {knn_distances_match}")

    print("\nRuntime / quality comparison:")
    print(
        dataframe.to_string(
            index=False,
        )
    )

    print(f"\nSaved table to: {output_path}")


if __name__ == "__main__":
    main()
