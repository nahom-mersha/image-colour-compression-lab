# K-means Compression Sweep Experiment

## Objective

This experiment studies how the number of K-means clusters affects image compression quality and computational cost.

The same sample image, random seed, initialization method, silhouette sample, and iteration limit were used for every run so that the comparison is reproducible and fair.

## Experimental setup

- Input image: `data/samples/sample_image.jpg`
- Algorithm: K-means implemented from scratch with NumPy
- Initialization: K-means++
- Values of `k`: 4, 8, 16, and 32
- Random seed: 42
- Maximum iterations: 20
- Silhouette sample size: 5,000 pixels

## Metrics

- **Final inertia:** total squared distance between each pixel and its assigned centroid. Lower is better.
- **RGB reconstruction MSE:** average squared difference between original and reconstructed RGB channel values. Lower is better.
- **Unique colours:** number of colours in the reconstructed image.
- **Silhouette score:** measures how compact and well-separated the clusters are. Higher is better. It was calculated using a reproducible sample of 5,000 pixels.
- **Iterations:** number of K-means iterations completed.
- **Runtime:** time required to fit the K-means model.

## Results

| k | Final inertia | RGB MSE | Unique colours | Silhouette score | Iterations | Runtime (seconds) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 508,620,906.17 | 185.61 | 4 | 0.5919 | 14 | 6.472 |
| 8 | 128,457,123.16 | 46.88 | 8 | 0.5811 | 20 | 17.789 |
| 16 | 41,173,313.96 | 15.03 | 16 | 0.4817 | 20 | 27.933 |
| 32 | 18,813,063.98 | 6.87 | 32 | 0.4109 | 20 | 52.237 |

## Generated outputs

- [Results CSV](../reports/experiments/k_sweep/k_sweep.csv)
- [Compressed image with k=4](../reports/experiments/k_sweep/k4.png)
- [Compressed image with k=8](../reports/experiments/k_sweep/k8.png)
- [Compressed image with k=16](../reports/experiments/k_sweep/k16.png)
- [Compressed image with k=32](../reports/experiments/k_sweep/k32.png)

## Findings

Increasing `k` consistently improved numerical reconstruction quality:

- Inertia decreased from approximately 508.6 million at `k=4` to 18.8 million at `k=32`.
- MSE decreased from 185.61 to 6.87.
- The reconstructed image used exactly `k` colours in every run.
- The silhouette score was highest for `k=4` at 0.5919 and decreased to 0.4109 at `k=32`.

The improvement in reconstruction quality came with increased computational cost. Runtime increased from approximately 6.5 seconds at `k=4` to 52.2 seconds at `k=32`.

The runs with `k=8`, `k=16`, and `k=32` all reached the maximum of 20 iterations, while the `k=4` run converged after 14 iterations.

## Interpretation

A smaller value of `k` provides stronger colour compression and faster execution, but it loses more colour detail.

A larger value of `k` preserves more detail and produces a lower reconstruction error, but requires more computation and provides less colour reduction.

The silhouette score shows a different pattern. The clusters were more compact and separated at smaller values of `k`. As `k` increased, the algorithm created more fine-grained colour groups that were numerically closer to one another in RGB space, causing the silhouette score to decrease.

This demonstrates that the metrics measure different properties:

- MSE and inertia measure reconstruction error.
- Unique-colour count measures palette reduction.
- Silhouette score measures cluster compactness and separation.
- Runtime measures computational cost.

Therefore, selecting `k` is a quality-versus-compression trade-off rather than simply choosing the largest possible value. Numerical metrics should also be considered alongside visual inspection because a lower RGB error does not always correspond perfectly to better human-perceived image quality.

## Reproducibility

The experiment can be rerun with:

```bash
python scripts/run_k_sweep.py