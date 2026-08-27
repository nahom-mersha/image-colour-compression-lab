from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_side_by_side_comparison(
    original_image: np.ndarray,
    compressed_image: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save the original and compressed images side by side."""
    if original_image.shape != compressed_image.shape:
        raise ValueError("Images must have the same shape.")

    if original_image.ndim != 3 or original_image.shape[2] != 3:
        raise ValueError("Expected RGB images with shape (height, width, 3).")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(compressed_image)
    axes[1].set_title("Compressed image")
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
