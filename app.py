import logging
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from image_colour_compression_lab.knn import knn_search
from image_colour_compression_lab.pca import (
    fit_pca,
    transform_pca,
)

MAX_IMAGE_DIMENSION = 600
FIT_SAMPLE_SIZE = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_uploaded_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")

    resized_image = resize_if_needed(
        image,
        max_width=MAX_IMAGE_DIMENSION,
        max_height=MAX_IMAGE_DIMENSION,
    )
    logger.info(
        "Loaded uploaded image: %dx%d",
        resized_image.size[0],
        resized_image.size[1],
    )
    return np.asarray(resized_image)


def hex_to_rgb(hex_colour: str) -> np.ndarray:
    """Convert a hex colour such as #FF0000 to RGB values."""
    return np.array(
        [
            int(hex_colour[1:3], 16),
            int(hex_colour[3:5], 16),
            int(hex_colour[5:7], 16),
        ],
        dtype=np.float64,
    )


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
            logger.info(
                "Starting K-means: sample=%d, k=%d, initialization=%s, seed=%d",
                sample_size,
                k,
                initialization,
                int(random_seed),
            )
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
            logger.info(
                "K-means finished: iterations=%d, runtime=%.2fs",
                len(inertia_history),
                runtime_seconds,
            )
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
        st.header("KNN Pixel Similarity")

        st.write(
            "Choose a colour and find the most similar "
            "pixels that actually occur in the selected image."
        )

        similarity_source = st.radio(
            "Image source",
            ["Sample image", "Upload image"],
            horizontal=True,
            key="knn_image_source",
        )

        similarity_image = None

        if similarity_source == "Sample image":
            similarity_image = load_image(
                Path("data/samples/sample_image.jpg"),
                max_width=MAX_IMAGE_DIMENSION,
                max_height=MAX_IMAGE_DIMENSION,
            )

        else:
            similarity_upload = st.file_uploader(
                "Upload a PNG or JPEG image",
                type=["png", "jpg", "jpeg"],
                key="knn_image_upload",
            )

            if similarity_upload is None:
                st.info("Upload an image to explore similar pixels.")
            else:
                similarity_image = load_uploaded_image(similarity_upload)

        if similarity_image is not None:
            st.image(
                similarity_image,
                caption="Similarity search image",
                width=400,
            )

            similarity_pixels = image_to_pixels(similarity_image).astype(np.float64)

            random_generator = np.random.default_rng(42)

            reference_count = min(
                5000,
                len(similarity_pixels),
            )

            reference_indices = random_generator.choice(
                len(similarity_pixels),
                size=reference_count,
                replace=False,
            )

            reference_pixels = similarity_pixels[reference_indices]

            query_hex = st.color_picker(
                "Choose query colour",
                "#FF0000",
            )

            metric = st.selectbox(
                "Distance metric",
                ["euclidean", "manhattan"],
                key="knn_metric",
            )

            neighbour_count = st.slider(
                "Number of neighbours",
                min_value=1,
                max_value=20,
                value=5,
            )

            if st.button(
                "Find similar pixels",
                type="primary",
            ):
                query_colour = hex_to_rgb(query_hex)

                query = query_colour.reshape(
                    1,
                    3,
                )

                logger.info(
                    "Running KNN: references=%d, neighbours=%d, metric=%s",
                    len(reference_pixels),
                    neighbour_count,
                    metric,
                )
                (
                    neighbour_indices,
                    neighbour_distances,
                ) = knn_search(
                    query,
                    reference_pixels,
                    k=neighbour_count,
                    metric=metric,
                )

                neighbours = reference_pixels[neighbour_indices[0]].astype(np.uint8)

                st.subheader("Query Colour")

                query_swatch = np.full(
                    (80, 160, 3),
                    query_colour,
                    dtype=np.uint8,
                )

                st.image(
                    query_swatch,
                    width=160,
                )

                query_rgb = tuple(int(value) for value in query_colour)

                st.caption(f"RGB {query_rgb}")

                st.subheader(f"{neighbour_count} Nearest Pixels")

                columns_per_row = 5

                for row_start in range(
                    0,
                    neighbour_count,
                    columns_per_row,
                ):
                    columns = st.columns(columns_per_row)

                    row_end = min(
                        row_start + columns_per_row,
                        neighbour_count,
                    )

                    for neighbour_index in range(
                        row_start,
                        row_end,
                    ):
                        column = columns[neighbour_index - row_start]

                        colour = neighbours[neighbour_index]

                        rgb_tuple = tuple(int(value) for value in colour)

                        swatch = np.full(
                            (100, 100, 3),
                            colour,
                            dtype=np.uint8,
                        )

                        with column:
                            st.image(
                                swatch,
                                use_container_width=True,
                            )

                            st.caption(
                                f"RGB {rgb_tuple}\n\n"
                                f"Distance: "
                                f"{neighbour_distances[0, neighbour_index]:.2f}"
                            )

    with pca_tab:
        st.header("PCA and K-means Clusters")

        st.write(
            "Project RGB pixels into two principal components "
            "and visualize the K-means clusters."
        )

        pca_source = st.radio(
            "Image source",
            ["Sample image", "Upload image"],
            horizontal=True,
            key="pca_image_source",
        )

        pca_image = None

        if pca_source == "Sample image":
            pca_image = load_image(
                Path("data/samples/sample_image.jpg"),
                max_width=MAX_IMAGE_DIMENSION,
                max_height=MAX_IMAGE_DIMENSION,
            )

        else:
            pca_upload = st.file_uploader(
                "Upload a PNG or JPEG image",
                type=["png", "jpg", "jpeg"],
                key="pca_image_upload",
            )

            if pca_upload is None:
                st.info("Upload an image to visualize PCA.")
            else:
                pca_image = load_uploaded_image(pca_upload)

        if pca_image is not None:
            st.image(
                pca_image,
                caption="PCA analysis image",
                width=400,
            )

            cluster_count = st.select_slider(
                "Number of clusters shown in the PCA visualization",
                options=[4, 8, 16],
                value=8,
            )
            st.caption(
                "Changing k affects the K-means cluster labels shown in the "
                "scatter plot. It does not change PCA explained variance."
            )
            if st.button(
                "Run PCA analysis",
                type="primary",
            ):
                pixels = image_to_pixels(pca_image).astype(np.float64)

                random_generator = np.random.default_rng(42)

                sample_size = min(
                    5000,
                    len(pixels),
                )

                sampled_indices = random_generator.choice(
                    len(pixels),
                    size=sample_size,
                    replace=False,
                )

                sampled_pixels = pixels[sampled_indices]
                logger.info(
                    "Running PCA analysis: sample=%d, clusters=%d",
                    sample_size,
                    cluster_count,
                )
                (
                    _,
                    assignments,
                    _,
                ) = fit_kmeans(
                    sampled_pixels,
                    k=cluster_count,
                    max_iterations=20,
                    random_seed=42,
                    initialization="kmeans++",
                )

                (
                    mean,
                    components,
                    _,
                    explained_variance_ratio,
                ) = fit_pca(sampled_pixels)

                transformed_pixels = transform_pca(
                    sampled_pixels,
                    mean,
                    components,
                )

                st.subheader("Explained Variance")

                variance_columns = st.columns(3)

                for index in range(3):
                    variance_columns[index].metric(
                        f"PC{index + 1}",
                        (f"{explained_variance_ratio[index] * 100:.2f}%"),
                    )

                figure_variance, axis_variance = plt.subplots(figsize=(7, 4))

                component_numbers = np.arange(
                    1,
                    len(explained_variance_ratio) + 1,
                )

                axis_variance.bar(
                    component_numbers,
                    explained_variance_ratio,
                )

                axis_variance.set_xlabel("Principal component")

                axis_variance.set_ylabel("Explained variance ratio")

                axis_variance.set_title("PCA Explained Variance")

                axis_variance.set_xticks(component_numbers)

                figure_variance.tight_layout()

                st.pyplot(figure_variance)

                plt.close(figure_variance)

                st.subheader("Clusters in PCA Space")

                figure_scatter, axis_scatter = plt.subplots(figsize=(8, 5))

                scatter = axis_scatter.scatter(
                    transformed_pixels[:, 0],
                    transformed_pixels[:, 1],
                    c=assignments,
                    s=8,
                    alpha=0.6,
                )

                axis_scatter.set_xlabel("PC1")

                axis_scatter.set_ylabel("PC2")

                axis_scatter.set_title("K-means Clusters in PCA Space")

                figure_scatter.colorbar(
                    scatter,
                    ax=axis_scatter,
                    label="K-means cluster",
                )

                figure_scatter.tight_layout()

                st.pyplot(figure_scatter)

                plt.close(figure_scatter)

                st.caption(
                    "K-means creates the cluster labels. "
                    "PCA only projects the RGB pixels into "
                    "two dimensions for visualization."
                )

    with experiments_tab:
        st.header("Experiment Comparisons")

        st.write(
            "Explore experiment results generated locally and pushed "
            "to the project's repository, covering K-means behaviour, "
            "runtime, and dimensionality."
        )

        experiment = st.selectbox(
            "Choose experiment",
            [
                "Number of clusters (k)",
                "Initialization",
                "Feature scaling — concept",
                "Runtime comparison",
                "Curse of dimensionality",
            ],
            key="experiment_choice",
        )

        if experiment == "Number of clusters (k)":
            st.subheader("Effect of k")

            k_results = pd.read_csv("reports/experiments/k_sweep/k_sweep.csv")

            st.dataframe(
                k_results,
                use_container_width=True,
                hide_index=True,
            )

            st.line_chart(
                k_results,
                x="k",
                y="mse",
            )

            st.write(
                "As k increases, more colours are available "
                "to represent the image, so reconstruction "
                "error decreases. Runtime generally increases."
            )

        elif experiment == "Initialization":
            st.subheader("Random vs K-means++ Initialization")

            initialization_results = pd.read_csv(
                "reports/experiments/kmeans_initialization_comparison.csv"
            )

            st.dataframe(
                initialization_results,
                use_container_width=True,
                hide_index=True,
            )

            st.image(
                "reports/experiments/kmeans_initialization_comparison.png",
                caption="K-means initialization comparison",
            )

            st.write(
                "K-means++ generally provides more reliable "
                "starting centroids than random initialization, "
                "reducing sensitivity to poor initial choices."
            )

        elif experiment == "Feature scaling — concept":
            st.subheader("Feature Scaling")

            st.info(
                "No standalone scaling experiment was saved for this project. "
                "This section summarizes the scaling behaviour investigated "
                "during development."
            )

            scaling_results = pd.DataFrame(
                {
                    "Features": [
                        "RGB only",
                        "RGB + normalized position",
                        "RGB + balanced position",
                    ],
                    "RGB scale": [
                        "0–255",
                        "0–255",
                        "0–255",
                    ],
                    "Position scale": [
                        "Not used",
                        "0–1",
                        "0–255",
                    ],
                    "Effect": [
                        "Similarity depends only on colour",
                        "RGB strongly dominates distance",
                        "Colour and position can both matter",
                    ],
                }
            )

            st.dataframe(
                scaling_results,
                use_container_width=True,
                hide_index=True,
            )

            st.write(
                "Feature scaling changes how much each feature influences distance. "
                "RGB values naturally range from 0 to 255, so RGB-only features are "
                "already on comparable scales. The issue appears when we add features "
                "such as pixel position x and y. If x and y are normalized to 0–1 "
                "while RGB remains 0–255, colour differences dominate and position "
                "has very little influence. If the features are brought onto similar "
                "scales, colour and position can both affect what the algorithm "
                "considers similar. For pure colour compression, this may reduce "
                "performance because K-means starts grouping pixels partly by location "
                "instead of only by colour."
            )
        elif experiment == "Runtime comparison":
            st.subheader("From Scratch vs Scikit-learn")

            benchmark_results = pd.read_csv("reports/benchmarks/sklearn_comparison.csv")

            st.dataframe(
                benchmark_results,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                benchmark_results,
                x="implementation",
                y="runtime_seconds",
                color="algorithm",
            )

            st.write(
                "Runtime differences depend on the dataset "
                "size and implementation overhead. A small "
                "benchmark should not be interpreted as a "
                "general claim that one implementation is "
                "always faster."
            )

        elif experiment == "Curse of dimensionality":
            st.subheader("Curse of Dimensionality")

            st.image(
                "reports/experiments/curse_of_dimensionality.png",
                caption=("Distance concentration as dimensionality increases"),
            )

            st.write(
                "As irrelevant dimensions are added, nearest "
                "and farthest distances become less distinct. "
                "KNN neighbour relationships also become less "
                "stable, while runtime and memory usage increase."
            )


if __name__ == "__main__":
    main()
