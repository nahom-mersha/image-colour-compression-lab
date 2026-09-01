# Algorithm Complexity

This project implements distance-based algorithms from scratch using NumPy and
compares them with scikit-learn.

## Pairwise Euclidean Distance

For `q` query points, `r` reference points, and `d` features:

- Time: O(q × r × d)
- Temporary memory: O(q × r × d)

The implementation uses NumPy broadcasting to create differences between all
query-reference pairs.

## Brute-Force KNN

For each query, KNN calculates its distance to every reference point and sorts
the distances.

- Distance calculation: O(q × r × d)
- Sorting: O(q × r log r)

This works well for small datasets but becomes expensive as the number of
points or dimensions increases.

## K-Means

For:

- `n` data points;
- `k` clusters;
- `d` features;
- `i` iterations;

the main cost is approximately:

- Time: O(i × n × k × d)

Most computation occurs during the assignment step, where every point is
compared with every centroid.

## PCA

The from-scratch PCA implementation uses a covariance matrix and
eigendecomposition.

For `n` samples and `d` features:

- Centering: O(n × d)
- Covariance calculation: O(n × d²)
- Eigendecomposition: O(d³)

For RGB data, `d = 3`, so PCA is inexpensive. The cost becomes more important
for datasets with many features.

## Why Scikit-Learn Can Be Faster

Scikit-learn uses mature optimized implementations and can take advantage of
efficient compiled numerical routines, optimized memory handling, and
specialized algorithms.

The from-scratch implementations remain useful because they expose the
mathematics and algorithm steps directly, while scikit-learn represents the
professional implementation used in real applications.

## Benchmark Results

| Algorithm | Implementation | Runtime (s) | Inertia |
| --- | --- | ---: | ---: |
| KNN | From scratch | 0.0064 | - |
| KNN | Scikit-learn | 0.0271 | - |
| K-means | From scratch | 0.0352 | 709506.80 |
| K-means | Scikit-learn | 0.0313 | 709867.48 |

For KNN, the neighbour distances matched scikit-learn, while the exact
indices did not. This can occur when multiple RGB neighbours have equal distances, allowing equally valid neighbours to be returned in different orders.

For K-means, both implementations produced very similar inertia. Exact
equality is not expected because K-means++ initialization and optimization
details can lead to different local solutions.

The runtime measurements are specific to this small benchmark and should
not be interpreted as evidence that the from-scratch implementation is
generally faster than scikit-learn.