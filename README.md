# Image Colour Compression Laboratory

An interactive laboratory for exploring image colour compression, pixel similarity, clustering, and dimensionality reduction.

The project implements core algorithms from scratch with NumPy and compares their behaviour, output, and runtime with scikit-learn implementations. It focuses on understanding how distance metrics, centroid initialization, the number of clusters, feature representation, and dimensionality affect image compression and similarity-based results.

## Live Demo

**Streamlit app:** https://image-colour-compression-lab.streamlit.app/


## Learning Notes

Detailed notes from this project are available in my AI Notes repository:

[Project 4 — Similarity and Clustering Laboratory](https://github.com/nahom-mersha/ai-notes/tree/main/Project%204%20-%20Similarity%20and%20Clustering%20Laboratory)

## Main Laboratory Areas

- **Image colour compression:** reduce an image to a selected number of representative colours using K-means.
- **Pixel similarity:** find the nearest colours in an image using K-nearest neighbours with Euclidean or Manhattan distance.
- **PCA and cluster visualization:** project RGB pixels into principal-component space and inspect K-means clusters.
- **Experiment comparisons:** review locally generated experiment results for values of `k`, initialization strategies, runtime, and the curse of dimensionality.

## Implemented Features

- Image loading, validation, RGB conversion, and safe resizing
- From-scratch Euclidean and Manhattan distance
- From-scratch K-nearest-neighbour search
- From-scratch K-means clustering
- Random and K-means++ centroid initialization
- Image reconstruction from learned K-means centroids
- RGB reconstruction error, inertia, silhouette score, runtime, and colour-count measurements
- K-means experiments across multiple values of `k`
- Random versus K-means++ initialization experiments
- From-scratch PCA using covariance-matrix eigendecomposition
- PCA explained-variance analysis and two-dimensional cluster visualization
- Curse-of-dimensionality experiment
- Comparisons with scikit-learn for correctness, behaviour, and runtime
- Interactive Streamlit application
- Automated tests, reproducible configuration, CI, Docker support, and technical documentation

## Streamlit Laboratory

The application contains four main tabs.

### Compress Image

Upload an image or use the included sample image, choose the number of colours, select the centroid initialization strategy, and run K-means compression.

The interface displays:

- the original and compressed images;
- the learned colour palette;
- original and compressed colour counts;
- RGB mean squared error;
- number of K-means iterations;
- runtime.

### Explore Similar Pixels

Choose a query colour and search for similar pixels in a sample or uploaded image.

The user can select:

- Euclidean or Manhattan distance;
- the number of nearest neighbours.

The application displays the nearest RGB colours and their distances from the query colour.

### PCA & Clusters

Run PCA on sampled RGB pixels and visualize:

- explained variance for each principal component;
- a two-dimensional `PC1` versus `PC2` projection;
- K-means cluster assignments in PCA space.

K-means creates the cluster labels. PCA is used only to project the RGB data into a lower-dimensional space for visualization.

### Experiments

Review experiment results generated locally and saved in the repository.

The tab includes:

- the effect of the number of clusters `k`;
- random versus K-means++ initialization;
- a conceptual explanation of feature scaling;
- from-scratch versus scikit-learn runtime comparisons;
- the curse of dimensionality.

The feature-scaling section is an explanatory summary rather than a saved standalone experiment. RGB channels already share the same `0–255` scale. Scaling becomes important when additional features such as pixel coordinates are introduced because differences in numerical scale change how strongly each feature contributes to distance.

## Project Structure

```text
image-colour-compression-lab/
├── app.py                       # Streamlit application
├── configs/
│   └── default.yaml             # Reproducible project configuration
├── data/
│   ├── samples/                 # Licence-safe sample images
│   └── generated/               # Locally generated outputs
├── docs/                        # Experiment and complexity documentation
├── reports/
│   ├── benchmarks/              # Runtime and implementation comparisons
│   └── experiments/             # Selected experiment results and figures
├── scripts/                     # Reproducible experiment scripts
├── src/
│   └── image_colour_compression_lab/
│       ├── benchmark.py
│       ├── distance.py
│       ├── image_io.py
│       ├── kmeans.py
│       ├── knn.py
│       ├── metrics.py
│       ├── pca.py
│       └── ...
├── tests/                       # Automated tests
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

The included sample image is:

```text
data/samples/sample_image.jpg
```

Small sample images, source code, documentation, and selected portfolio experiment results are tracked in Git. Other generated outputs can remain excluded from version control.

## Installation

Clone the repository and install the project with its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Run the Application

Start the Streamlit laboratory locally with:

```bash
streamlit run app.py
```

The deployed version is available at:

https://image-colour-compression-lab.streamlit.app/

## Run Tests

Run the automated test suite:

```bash
python -m pytest
```

Run formatting and linting checks:

```bash
ruff format --check .
ruff check .
```

## Configuration

Default settings are stored in:

```text
configs/default.yaml
```

The configuration includes:

- random seed;
- maximum image dimensions;
- pixel sample size;
- default palette size;
- maximum K-means iterations;
- convergence tolerance;
- initialization method;
- logging level.

## Experiments

Reproducible experiment scripts are stored in `scripts/`.

Key experiments include:

- K-means compression;
- K-means initialization comparison;
- `k` sweep;
- PCA analysis;
- curse of dimensionality;
- from-scratch versus scikit-learn comparison.

Experiment outputs and selected figures are stored under `reports/`, while written interpretations and complexity notes are stored under `docs/`.

## Complexity

The project also documents the computational cost of the main algorithms.

At a high level:

- pairwise distance computation grows with the number of queries, reference points, and dimensions;
- brute-force KNN adds neighbour-ranking cost;
- K-means grows with the number of pixels, clusters, dimensions, and iterations;
- PCA using a covariance matrix is inexpensive for RGB because the feature dimension is only three.

See `docs/complexity.md` for the detailed discussion.

## Docker

Build the image:

```bash
docker build -t image-colour-compression-lab .
```

Run the container:

```bash
docker run --rm image-colour-compression-lab
```

## Learning Goals

This project develops practical understanding of:

- unsupervised learning;
- image colour quantization;
- Euclidean and Manhattan distance;
- nearest-neighbour search;
- K-means and K-means++ initialization;
- cluster evaluation;
- PCA and explained variance;
- the curse of dimensionality;
- feature scaling in distance-based algorithms;
- runtime and complexity analysis;
- comparison between from-scratch and library implementations;
- reproducible experimentation and testing.

## AI-Assisted Learning

This is an AI-assisted learning project. I direct the work, study the underlying concepts, review the implementation, run the experiments, interpret the results, and document what I learn.

The goal is not only to produce a working application, but to understand and be able to explain the algorithms and design decisions behind it.

## Scope Boundaries

The required version does not include neural networks, learned image embeddings, real-time video, user accounts, a database, or a cloud API.

RGB is the primary colour representation. More advanced colour spaces, perceptual image-quality metrics, MiniBatchKMeans, and additional deployment improvements may be explored later.
