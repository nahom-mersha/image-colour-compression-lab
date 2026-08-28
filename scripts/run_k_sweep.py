import csv
from pathlib import Path

import matplotlib.pyplot as plt

from image_colour_compression_lab.benchmark import measure_runtime
from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
    pixels_to_image,
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
    output_directory = Path("reports/experiments/k_sweep")
    csv_path = output_directory / "k_sweep.csv"

    k_values = [4, 8, 16, 32]
    random_seed = 42
    max_iterations = 20

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_image = load_image(input_path)
    pixels = image_to_pixels(original_image)

    results = []

    for k in k_values:
        fit_result, runtime_seconds = measure_runtime(
            fit_kmeans,
            pixels,
            k=k,
            max_iterations=max_iterations,
            random_seed=random_seed,
            initialization="kmeans++",
        )

        centroids, assignments, inertia_history = fit_result

        compressed_pixels = reconstruct_pixels(
            centroids,
            assignments,
        )

        compressed_image = pixels_to_image(
            compressed_pixels,
            height=original_image.shape[0],
            width=original_image.shape[1],
        )

        compressed_image = compressed_image.clip(0, 255).round().astype("uint8")

        mse = mean_squared_error(
            pixels,
            compressed_pixels,
        )

        unique_colours = count_unique_colours(
            compressed_pixels,
        )

        results.append(
            {
                "k": k,
                "final_inertia": inertia_history[-1],
                "mse": mse,
                "unique_colours": unique_colours,
                "iterations": len(inertia_history),
                "runtime_seconds": runtime_seconds,
            }
        )

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(12, 6),
        )

        axes[0].imshow(original_image)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        axes[1].imshow(compressed_image)
        axes[1].set_title(f"K-means compressed image, k={k}")
        axes[1].axis("off")

        figure.tight_layout()
        figure.savefig(
            output_directory / f"k{k}.png",
            dpi=150,
        )
        plt.close(figure)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "k",
                "final_inertia",
                "mse",
                "unique_colours",
                "iterations",
                "runtime_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    for result in results:
        print(
            f"k={result['k']} "
            f"inertia={result['final_inertia']:.2f} "
            f"mse={result['mse']:.2f} "
            f"colours={result['unique_colours']} "
            f"iterations={result['iterations']} "
            f"runtime={result['runtime_seconds']:.3f}s"
        )

    print(f"\nSaved results to: {csv_path}")
    print(f"Saved images to: {output_directory}")


if __name__ == "__main__":
    main()
