from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
)
from image_colour_compression_lab.kmeans import fit_kmeans
from image_colour_compression_lab.pca import (
    fit_pca,
    transform_pca,
)


def main() -> None:
    input_path = Path("data/samples/sample_image.jpg")
    output_path = Path("reports/experiments/pca_rgb_analysis.png")

    image = load_image(input_path)
    pixels = image_to_pixels(image).astype(np.float64)

    random_generator = np.random.default_rng(42)

    sample_size = min(5000, pixels.shape[0])

    sampled_indices = random_generator.choice(
        pixels.shape[0],
        size=sample_size,
        replace=False,
    )

    sampled_pixels = pixels[sampled_indices]

    _, assignments, _ = fit_kmeans(
        sampled_pixels,
        k=8,
        max_iterations=20,
        random_seed=42,
        initialization="kmeans++",
    )

    (
        mean,
        components,
        explained_variance,
        explained_variance_ratio,
    ) = fit_pca(sampled_pixels)

    transformed_pixels = transform_pca(
        sampled_pixels,
        mean,
        components,
    )

    sklearn_pca = PCA()
    sklearn_pca.fit(sampled_pixels)

    print("From-scratch explained variance:")
    print(explained_variance)

    print("\nFrom-scratch explained variance ratio:")
    print(explained_variance_ratio)

    print("\nScikit-learn explained variance ratio:")
    print(sklearn_pca.explained_variance_ratio_)

    print(
        "\nRatios match sklearn:",
        np.allclose(
            explained_variance_ratio,
            sklearn_pca.explained_variance_ratio_,
        ),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    component_numbers = np.arange(
        1,
        len(explained_variance_ratio) + 1,
    )

    axes[0].bar(
        component_numbers,
        explained_variance_ratio,
    )
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA Explained Variance")
    axes[0].set_xticks(component_numbers)

    scatter = axes[1].scatter(
        transformed_pixels[:, 0],
        transformed_pixels[:, 1],
        c=assignments,
        s=8,
        alpha=0.6,
    )

    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Image Pixels in PCA Space")

    figure.colorbar(
        scatter,
        ax=axes[1],
        label="K-means cluster",
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    print(f"\nSaved PCA figure to: {output_path}")


if __name__ == "__main__":
    main()
