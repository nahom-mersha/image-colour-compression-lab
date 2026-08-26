# Image Colour Compression Laboratory

An interactive laboratory for exploring image colour compression, similarity search, and clustering.

The project compares algorithms implemented from scratch with professional library implementations. It focuses on understanding how distance measures, initialization strategies, feature representations, and the number of clusters affect both image quality and computational performance.

## Project Status

Project 4 of my AI Engineering roadmap. Implementation is currently in progress.

## Planned Features

* Image loading, validation, and preprocessing
* Colour compression with K-means clustering
* From-scratch K-means implementation
* Comparison with scikit-learn K-means
* Random and K-means++ initialization
* RGB colour-space experiments
* Optional spatial features using pixel coordinates
* Image similarity and nearest-colour experiments
* PCA implemented from scratch
* Cluster-quality and compression metrics
* Interactive Streamlit interface
* Visual comparison of original and compressed images
* Reproducible configuration, tests, logging, CI, and Docker support

## Quick Start

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Build the Docker image:

```bash
docker build -t image-colour-compression-lab .
```

Run the Docker container:

```bash
docker run --rm image-colour-compression-lab
```

## AI-Assisted Learning

This is an AI-assisted learning project. I direct the work, study the underlying concepts, review the implementation, run the experiments, and document what I learn.

## Learning Notes

Project-specific learning notes will be published in my AI notes repository after the implementation and experiments are complete.

## Purpose

This repository is Project 4 of my AI Engineering roadmap. Its purpose is to develop a practical understanding of similarity, distance measures, clustering, dimensionality reduction, and algorithm evaluation through a visual, experiment-driven application.
