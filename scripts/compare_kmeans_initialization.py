import csv
from pathlib import Path

import matplotlib.pyplot as plt

from image_colour_compression_lab.benchmark import (
    measure_runtime,
)
from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
)
from image_colour_compression_lab.kmeans import (
    fit_kmeans,
    reconstruct_pixels,
)
from image_colour_compression_lab.metrics import (
    count_unique_colours,
    mean_squared_error,
)


def main() -> None:
    input_path = Path("data/samples/sample_image.jpg")
    csv_path = Path("reports/experiments/kmeans_initialization_comparison.csv")
    figure_path = Path("reports/experiments/kmeans_initialization_comparison.png")

    image = load_image(input_path)
    pixels = image_to_pixels(image)

    methods = ["random", "kmeans++"]
    seeds = [1, 2, 3, 4, 5]
    results = []

    for method in methods:
        for seed in seeds:
            fit_result, runtime_seconds = measure_runtime(
                fit_kmeans,
                pixels,
                k=8,
                max_iterations=20,
                random_seed=seed,
                initialization=method,
            )

            centroids, assignments, inertia_history = fit_result

            compressed_pixels = reconstruct_pixels(
                centroids,
                assignments,
            )

            mse = mean_squared_error(
                pixels,
                compressed_pixels,
            )

            unique_colours = count_unique_colours(
                compressed_pixels,
            )

            results.append(
                {
                    "initialization": method,
                    "seed": seed,
                    "final_inertia": inertia_history[-1],
                    "mse": mse,
                    "unique_colours": unique_colours,
                    "iterations": len(inertia_history),
                    "runtime_seconds": runtime_seconds,
                }
            )
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "initialization",
                "seed",
                "final_inertia",
                "mse",
                "unique_colours",
                "iterations",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    random_inertia = [
        result["final_inertia"]
        for result in results
        if result["initialization"] == "random"
    ]

    kmeans_plus_plus_inertia = [
        result["final_inertia"]
        for result in results
        if result["initialization"] == "kmeans++"
    ]

    figure, axis = plt.subplots(
        figsize=(8, 6),
    )

    axis.boxplot(
        [random_inertia, kmeans_plus_plus_inertia],
        tick_labels=["Random", "K-means++"],
    )
    axis.set_title("K-means Initialization Comparison")
    axis.set_ylabel("Final inertia")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    for result in results:
        print(
            f"{result['initialization']:10} "
            f"seed={result['seed']} "
            f"inertia={result['final_inertia']:.2f} "
            f"mse={result['mse']:.2f} "
            f"colours={result['unique_colours']} "
            f"iterations={result['iterations']} "
            f"runtime={result['runtime_seconds']:.3f}s"
        )

    print(f"\nSaved results to: {csv_path}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
