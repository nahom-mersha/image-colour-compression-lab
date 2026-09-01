from pathlib import Path
from time import perf_counter

import numpy as np
import streamlit as st
from PIL import Image

from image_colour_compression_lab.image_io import (
    image_to_pixels,
    load_image,
    pixels_to_image,
    resize_if_needed,
)
from image_colour_compression_lab.kmeans import (
    assign_clusters,
    fit_kmeans,
    reconstruct_pixels,
)

MAX_IMAGE_DIMENSION = 600
FIT_SAMPLE_SIZE = 5000


def load_uploaded_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")

    resized_image = resize_if_needed(
        image,
        max_width=MAX_IMAGE_DIMENSION,
        max_height=MAX_IMAGE_DIMENSION,
    )

    return np.asarray(resized_image)


def main() -> None:
    st.set_page_config(
        page_title="Image Colour Compression Lab",
        layout="wide",
    )

    st.title("Image Colour Compression Laboratory")

    (
        compression_tab,
        similarity_tab,
        pca_tab,
        experiments_tab,
    ) = st.tabs(
        [
            "Compress Image",
            "Explore Similar Pixels",
            "PCA & Clusters",
            "Experiments",
        ]
    )

    with compression_tab:
        st.header("K-means Image Compression")

        image_source = st.radio(
            "Image source",
            ["Sample image", "Upload image"],
            horizontal=True,
        )

        if image_source == "Sample image":
            image = load_image(
                Path("data/samples/sample_image.jpg"),
                max_width=MAX_IMAGE_DIMENSION,
                max_height=MAX_IMAGE_DIMENSION,
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a PNG or JPEG image",
                type=["png", "jpg", "jpeg"],
            )

            if uploaded_file is None:
                st.info("Upload an image to continue.")
                return

            image = load_uploaded_image(uploaded_file)

        k = st.select_slider(
            "Number of colours (k)",
            options=[4, 8, 16, 32],
            value=8,
        )

        initialization = st.selectbox(
            "Centroid initialization",
            ["kmeans++", "random"],
        )

        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            value=42,
            step=1,
        )

        if st.button(
            "Compress image",
            type="primary",
        ):
            pixels = image_to_pixels(image).astype(np.float64)

            rng = np.random.default_rng(int(random_seed))

            sample_size = min(
                FIT_SAMPLE_SIZE,
                len(pixels),
            )

            sampled_indices = rng.choice(
                len(pixels),
                size=sample_size,
                replace=False,
            )

            sampled_pixels = pixels[sampled_indices]

            start_time = perf_counter()

            (
                centroids,
                _,
                inertia_history,
            ) = fit_kmeans(
                sampled_pixels,
                k=k,
                max_iterations=20,
                random_seed=int(random_seed),
                initialization=initialization,
            )

            full_assignments, _ = assign_clusters(
                pixels,
                centroids,
            )

            compressed_pixels = reconstruct_pixels(
                centroids,
                full_assignments,
            )

            runtime_seconds = perf_counter() - start_time

            compressed_pixels = np.clip(
                np.rint(compressed_pixels),
                0,
                255,
            ).astype(np.uint8)

            height, width, _ = image.shape

            compressed_image = pixels_to_image(
                compressed_pixels,
                height=height,
                width=width,
            )

            mse = float(np.mean((pixels - compressed_pixels.astype(np.float64)) ** 2))

            original_colours = len(
                np.unique(
                    image.reshape(-1, 3),
                    axis=0,
                )
            )

            final_colours = len(
                np.unique(
                    compressed_pixels,
                    axis=0,
                )
            )

            left, right = st.columns(2)

            with left:
                st.image(
                    image,
                    caption="Original",
                    use_container_width=True,
                )

            with right:
                st.image(
                    compressed_image,
                    caption="Compressed",
                    use_container_width=True,
                )

            metric_columns = st.columns(5)

            metric_columns[0].metric(
                "Original colours",
                f"{original_colours:,}",
            )

            metric_columns[1].metric(
                "Compressed colours",
                final_colours,
            )

            metric_columns[2].metric(
                "RGB MSE",
                f"{mse:.2f}",
            )

            metric_columns[3].metric(
                "Iterations",
                len(inertia_history),
            )

            metric_columns[4].metric(
                "Runtime",
                f"{runtime_seconds:.2f}s",
            )

            st.subheader("Learned Palette")

            palette = np.clip(
                np.rint(centroids),
                0,
                255,
            ).astype(np.uint8)

            palette_image = np.repeat(
                palette[np.newaxis, :, :],
                repeats=50,
                axis=0,
            )

            palette_image = np.repeat(
                palette_image,
                repeats=40,
                axis=1,
            )

            st.image(
                palette_image,
                use_container_width=False,
            )

    with similarity_tab:
        st.info("KNN pixel similarity explorer coming next.")

    with pca_tab:
        st.info("PCA visualization coming next.")

    with experiments_tab:
        st.info("Experiment comparisons coming next.")


if __name__ == "__main__":
    main()
