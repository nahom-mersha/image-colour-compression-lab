# Image Colour Compression Laboratory

An interactive laboratory for exploring image colour compression, pixel similarity, clustering, and dimensionality reduction.

The project implements important algorithms from scratch with NumPy and compares them with professional implementations from scikit-learn. It focuses on how distance measures, initialization strategies, feature representations, and the number of clusters affect image quality and computational performance.

## Project Status

Project 4 of my AI Engineering roadmap. Project setup is complete; algorithm implementation and experiments are in progress.

## Main Laboratory Areas

- Image colour compression using K-means
- Pixel similarity using K-nearest neighbours
- Dimensionality reduction and visualization using PCA
- Experiments and comparisons with scikit-learn

## Planned Features

- Image loading, validation, and preprocessing
- From-scratch K-means and K-means++ initialization
- Euclidean and Manhattan distance
- From-scratch K-nearest-neighbour search
- PCA implemented from scratch
- RGB and optional spatial-feature experiments
- Compression and cluster-quality metrics
- Original-versus-compressed image comparison
- Interactive Streamlit interface
- Reproducible configuration, tests, logging, CI, and Docker support

## Project Structure

```text
image-colour-compression-lab/
├── app/                         # Streamlit application
├── configs/                     # YAML configuration
├── data/
│   ├── samples/                 # Licence-safe sample images
│   └── generated/               # Generated images and figures
├── reports/
│   ├── benchmarks/              # Runtime and implementation comparisons
│   └── experiments/             # Experiment results and reports
├── src/
│   └── image_colour_compression_lab/
│       └──                    # Reusable project source code
├── tests/                       # Automated tests
├── Dockerfile
├── pyproject.toml
└── README.md
```

The data/samples/ directory contains small images used for development and experiments. Sample images must be personally owned or licence-safe.

The included sample image is:

data/samples/sample_image.jpg

Generated outputs are excluded from version control. Sample images and source code are tracked.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Run Tests

On Windows, run:

```bash
python -m pytest
```

Run formatting and linting checks:

```bash
ruff format --check .
ruff check .
```

## Configuration

Default settings are stored in `configs/default.yaml`. They include the random seed, image limits, pixel sample size, default palette size, K-means limits, initialization method, and logging level.

## Run the Application

The Streamlit interface will be added during the application stage.

```bash
streamlit run app/streamlit_app.py
```

## Docker

```bash
docker build -t image-colour-compression-lab .
docker run --rm image-colour-compression-lab
```

## Learning Goals

This project develops practical understanding of unsupervised learning, image colour quantization, distance metrics, nearest-neighbour search, clustering, dimensionality reduction, reconstruction error, algorithm comparison, and reproducible experimentation.

## AI-Assisted Learning

This is an AI-assisted learning project. I direct the work, study the underlying concepts, review the implementation, run the experiments, and document what I learn.

## Learning Notes

Project-specific learning notes will be published in my AI notes repository after the implementation and experiments are complete.

## Scope Boundaries

The required version does not include neural networks, learned image embeddings, real-time video, user accounts, a database, or a cloud API. RGB is the main colour representation. More advanced colour spaces and perceptual metrics may be considered after the required project is complete.
