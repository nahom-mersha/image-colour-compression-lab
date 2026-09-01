from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from image_colour_compression_lab.benchmark import measure_runtime
from image_colour_compression_lab.distance import (
    pairwise_euclidean_distances,
)
from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
)


def main() -> None:
    rng = np.random.default_rng(42)

    image = load_image(Path("data/samples/sample_image.jpg"))

    pixels = image_to_pixels(image).astype(float)

    # Sample 250 pixels so pairwise distances stay cheap.
    indices = rng.choice(
        len(pixels),
        size=250,
        replace=False,
    )

    rgb = pixels[indices]

    # Standardize R, G, and B separately.
    rgb = (rgb - rgb.mean(axis=0)) / rgb.std(axis=0)

    dimensions_to_test = [
        3,
        5,
        10,
        20,
        50,
        100,
    ]

    neighbour_count = 5

    # RGB-only nearest neighbours = baseline.
    baseline_distances = pairwise_euclidean_distances(
        rgb,
        rgb,
    )

    np.fill_diagonal(
        baseline_distances,
        np.inf,
    )

    baseline_neighbours = np.argsort(
        baseline_distances,
        axis=1,
    )[:, :neighbour_count]

    # Generate all noise once.
    max_noise_dimensions = max(dimensions_to_test) - 3

    noise = rng.normal(
        size=(
            len(rgb),
            max_noise_dimensions,
        )
    )

    ratios = []

    print("dimensions | distance_ratio | knn_stability | runtime_ms | temp_memory_mb")

    for dimensions in dimensions_to_test:
        noise_dimensions = dimensions - 3

        if noise_dimensions == 0:
            features = rgb
        else:
            features = np.column_stack(
                (
                    rgb,
                    noise[:, :noise_dimensions],
                )
            )

        distances, runtime_seconds = measure_runtime(
            pairwise_euclidean_distances,
            features,
            features,
        )

        runtime_ms = runtime_seconds * 1000

        # Ignore each point's distance to itself
        # when finding the nearest neighbour.
        distances_without_self = distances.copy()

        np.fill_diagonal(
            distances_without_self,
            np.inf,
        )

        nearest = np.min(
            distances_without_self,
            axis=1,
        )

        farthest = np.max(
            distances,
            axis=1,
        )

        distance_ratio = np.mean(nearest) / np.mean(farthest)

        current_neighbours = np.argsort(
            distances_without_self,
            axis=1,
        )[:, :neighbour_count]

        overlaps = [
            len(
                np.intersect1d(
                    old,
                    new,
                )
            )
            / neighbour_count
            for old, new in zip(
                baseline_neighbours,
                current_neighbours,
            )
        ]

        knn_stability = np.mean(overlaps)

        # Temporary broadcast array has shape:
        # (250, 250, dimensions)
        temp_memory_mb = len(rgb) * len(rgb) * dimensions * 8 / 1024**2

        ratios.append(distance_ratio)

        print(
            f"{dimensions:10d} | "
            f"{distance_ratio:14.3f} | "
            f"{knn_stability:13.3f} | "
            f"{runtime_ms:10.2f} | "
            f"{temp_memory_mb:14.2f}"
        )

    output_path = Path("reports/experiments/curse_of_dimensionality.png")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.plot(
        dimensions_to_test,
        ratios,
        marker="o",
    )

    plt.xlabel("Number of dimensions")
    plt.ylabel("Nearest / farthest distance ratio")
    plt.title("Curse of Dimensionality: Distance Concentration")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()
