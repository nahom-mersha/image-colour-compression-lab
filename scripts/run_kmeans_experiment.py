from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
    pixels_to_image,
)
from image_colour_compression_lab.kmeans import (
    fit_kmeans,
    reconstruct_pixels,
)


def main() -> None:
    input_path = Path("data/samples/sample_image.jpg")
    output_path = Path("reports/experiments/kmeans_k8.png")

    original_image = load_image(input_path)
    height, width, _ = original_image.shape

    pixels = image_to_pixels(original_image)

    centroids, assignments, inertia_history = fit_kmeans(
        pixels,
        k=8,
        max_iterations=20,
        random_seed=42,
    )

    compressed_pixels = reconstruct_pixels(
        centroids,
        assignments,
    )

    compressed_image = pixels_to_image(
        compressed_pixels,
        height=height,
        width=width,
    )

    compressed_image = np.clip(
        np.rint(compressed_image),
        0,
        255,
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
    axes[1].set_title("K-means compressed image")
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"Final inertia: {inertia_history[-1]:.2f}")
    print(f"Iterations: {len(inertia_history)}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
