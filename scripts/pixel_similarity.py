from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
)
from image_colour_compression_lab.knn import knn_search


def main() -> None:
    image_path = Path("data/samples/sample_image.jpg")
    output_path = Path("reports/experiments/pixel_similarity_euclidean.png")

    image = load_image(image_path)
    height, width, _ = image.shape
    pixels = image_to_pixels(image)

    query_row = height // 2
    query_column = width // 2
    query_index = query_row * width + query_column

    query_pixel = pixels[query_index : query_index + 1]

    neighbour_indices, neighbour_distances = knn_search(
        query_pixel,
        pixels,
        k=8,
        metric="euclidean",
    )

    selected_indices = neighbour_indices[0]
    selected_distances = neighbour_distances[0]
    selected_pixels = pixels[selected_indices]

    rows, columns = np.unravel_index(
        selected_indices,
        (height, width),
    )

    print(f"Query location: ({query_row}, {query_column})")
    print(f"Query RGB value: {query_pixel[0]}")
    print()

    for index, pixel, distance, row, column in zip(
        selected_indices,
        selected_pixels,
        selected_distances,
        rows,
        columns,
    ):
        print(
            f"Pixel index: {index}, "
            f"location: ({row}, {column}), "
            f"RGB: {pixel}, "
            f"distance: {distance:.2f}"
        )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(10, 5),
    )

    for axis, pixel, distance in zip(
        axes.ravel(),
        selected_pixels,
        selected_distances,
    ):
        swatch = np.zeros((50, 50, 3), dtype=np.uint8)
        swatch[:, :] = pixel

        axis.imshow(swatch)
        axis.set_title(f"{distance:.1f}")
        axis.axis("off")

    figure.suptitle(f"Nearest colours to pixel RGB {query_pixel[0]}")
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print()
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
