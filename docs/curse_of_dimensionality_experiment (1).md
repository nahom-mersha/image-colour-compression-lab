# Curse of Dimensionality Experiment

## Objective

This experiment demonstrates how adding irrelevant dimensions affects distance-based similarity.

The experiment starts with standardized RGB pixel features and progressively adds random noise dimensions. For each dimensionality, it measures:

- nearest-to-farthest distance ratio;
- KNN neighbour stability;
- pairwise-distance runtime;
- estimated temporary memory use.

## Setup

- Sample size: 250 image pixels
- Baseline features: standardized RGB
- Dimensions tested: 3, 5, 10, 20, 50, 100
- Neighbours per point: 5
- Random seed: 42

Noise dimensions were generated from a standard normal distribution and added to the RGB features.

## Results

| Dimensions | Distance Ratio | KNN Stability | Runtime (ms) | Temp Memory (MB) |
|---:|---:|---:|---:|---:|
| 3 | 0.007 | 1.000 | 1.72 | 1.43 |
| 5 | 0.074 | 0.123 | 2.47 | 2.38 |
| 10 | 0.251 | 0.060 | 5.71 | 4.77 |
| 20 | 0.421 | 0.050 | 7.44 | 9.54 |
| 50 | 0.604 | 0.046 | 16.65 | 23.84 |
| 100 | 0.705 | 0.028 | 33.87 | 47.68 |

## Interpretation

The nearest-to-farthest distance ratio increased from **0.007** in 3 dimensions to **0.705** in 100 dimensions.

This shows **distance concentration**: as irrelevant dimensions are added, the nearest points become less clearly distinguishable from the farthest points.

KNN stability dropped sharply from **1.000** in the original RGB space to **0.028** at 100 dimensions. This means that only about 2.8% of the original five nearest neighbours remained among the nearest neighbours on average.

The experiment therefore shows that irrelevant dimensions can strongly change neighbour relationships and make distance-based similarity less reliable.

Runtime also increased from **1.72 ms** to **33.87 ms**, while estimated temporary memory increased from **1.43 MB** to **47.68 MB**. Higher-dimensional distance calculations therefore also become more computationally expensive.

## Conclusion

The experiment successfully demonstrates the curse of dimensionality for this project:

- distance contrast weakens as dimensionality increases;
- nearest-neighbour relationships become unstable;
- runtime increases;
- memory requirements increase.

This explains why irrelevant features can hurt KNN and other distance-based methods, and why techniques such as feature selection or dimensionality reduction can be useful.

## Visualization

The generated figure is stored at:

`reports/experiments/curse_of_dimensionality.png`

It shows the nearest/farthest distance ratio increasing toward 1 as dimensionality grows.
