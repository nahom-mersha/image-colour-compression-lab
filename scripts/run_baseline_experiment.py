from pathlib import Path

from image_colour_compression_lab.baseline import (
    random_palette_compression,
)
from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
    pixels_to_image,
)
from image_colour_compression_lab.visualization import (
    save_side_by_side_comparison,
)


def main() -> None:
    image_path = Path("data/samples/sample_image.jpg")
    output_path = Path("reports/experiments/baseline_k4.png")

    image = load_image(image_path)
    height, width, _ = image.shape

    pixels = image_to_pixels(image)

    compressed_pixels, palette = random_palette_compression(
        pixels,
        k=4,
        random_seed=42,
    )

    compressed_image = pixels_to_image(
        compressed_pixels,
        height=height,
        width=width,
    )

    save_side_by_side_comparison(
        original_image=image,
        compressed_image=compressed_image,
        output_path=output_path,
    )

    print(f"Saved comparison to: {output_path}")
    print(f"Palette shape: {palette.shape}")


if __name__ == "__main__":
    main()
