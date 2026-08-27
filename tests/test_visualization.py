import numpy as np

from image_colour_compression_lab.visualization import (
    save_side_by_side_comparison,
)


def test_save_side_by_side_comparison(tmp_path):
    original_image = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )
    compressed_image = np.full(
        (2, 2, 3),
        255,
        dtype=np.uint8,
    )

    output_path = tmp_path / "comparison.png"

    save_side_by_side_comparison(
        original_image,
        compressed_image,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
