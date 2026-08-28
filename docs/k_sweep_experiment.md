# K-means Compression Sweep Experiment

## Objective

This experiment studies how the number of K-means clusters affects image
compression quality and computational cost.

The same sample image, random seed, initialization method, and iteration limit
were used for every run so that the comparison is reproducible and fair.

## Experimental setup

- Input image: data/samples/sample_image.jpg
- Algorithm: K-means implemented from scratch with NumPy
- Initialization: K-means++
- Values of k: 4, 8, 16, and 32
- Random seed: 42
- Maximum iterations: 20

## Metrics

- **Final inertia:** total squared distance between each pixel and its assigned
  centroid. Lower is better.
- **RGB reconstruction MSE:** average squared difference between original and
  reconstructed RGB channel values. Lower is better.
- **Unique colours:** number of colours in the reconstructed image.
- **Iterations:** number of K-means iterations completed.
- **Runtime:** time required to fit the K-means model.

## Results

| k | Final inertia | RGB MSE | Unique colours | Iterations | Runtime (seconds) |
|---:|---:|---:|---:|---:|---:|
| 4 | 508,620,906.17 | 185.61 | 4 | 14 | 4.496 |
| 8 | 128,457,123.16 | 46.88 | 8 | 20 | 17.131 |
| 16 | 41,173,313.96 | 15.03 | 16 | 20 | 26.951 |
| 32 | 18,813,063.98 | 6.87 | 32 | 20 | 37.947 |

## Generated outputs

- [Results CSV](../reports/experiments/k_sweep/k_sweep.csv)
- [Compressed image with k=4](../reports/experiments/k_sweep/k4.png)
- [Compressed image with k=8](../reports/experiments/k_sweep/k8.png)
- [Compressed image with k=16](../reports/experiments/k_sweep/k16.png)
- [Compressed image with k=32](../reports/experiments/k_sweep/k32.png)

## Findings

Increasing k consistently improved numerical reconstruction quality:

- Inertia decreased from approximately 508.6 million at k=4 to 18.8 million
  at k=32.
- MSE decreased from 185.61 to 6.87.
- The reconstructed image used exactly k colours in every run.

The improvement came with increased computational cost. Runtime increased from
approximately 4.5 seconds at k=4 to 37.9 seconds at k=32. The runs with k=8,
k=16, and k=32 all reached the maximum of 20 iterations, while the k=4 run
converged after 14 iterations.

## Interpretation

A smaller value of k provides stronger colour compression and faster
execution, but it loses more colour detail. A larger value of k preserves
more detail and produces a lower reconstruction error, but requires more
computation and provides less colour reduction.

Therefore, selecting k is a quality-versus-compression trade-off rather than
simply choosing the largest possible value.

## Reproducibility

The experiment can be rerun with:

~~~bash
python scripts/run_k_sweep.py
~~~
