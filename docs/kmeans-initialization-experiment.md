# K-means Initialization Experiment

## Objective

Compare random initialization with K-means++ initialization for image colour compression.

## Experimental setup

- Image: `data/samples/sample_image.jpg`
- Number of clusters: `k = 8`
- Initialization methods: random and K-means++
- Seeds: 1, 2, 3, 4, and 5
- Maximum iterations: 20
- Distance: Euclidean RGB distance

The same image, number of clusters, seeds, and K-means procedure were used for both methods. Only the centroid-initialization strategy differed.

## Recorded results

| Initialization | Seed | Final inertia | MSE | Unique colours | Iterations | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Random | 1 | 175,364,152.72 | 63.9963 | 8 | 20 | 13.092 |
| Random | 2 | 164,817,327.16 | 60.1474 | 8 | 20 | 16.405 |
| Random | 3 | 133,702,975.08 | 48.7927 | 8 | 20 | 12.635 |
| Random | 4 | 177,242,244.07 | 64.6817 | 8 | 20 | 15.037 |
| Random | 5 | 180,948,858.10 | 66.0343 | 8 | 20 | 14.904 |
| K-means++ | 1 | 130,890,182.66 | 47.7662 | 8 | 20 | 13.106 |
| K-means++ | 2 | 145,278,075.65 | 53.0169 | 8 | 20 | 14.624 |
| K-means++ | 3 | 145,828,325.84 | 53.2177 | 8 | 20 | 12.618 |
| K-means++ | 4 | 137,471,469.97 | 50.1680 | 8 | 20 | 12.384 |
| K-means++ | 5 | 145,332,216.19 | 53.0366 | 8 | 20 | 13.823 |

## Summary by initialization method

| Metric | Random | K-means++ |
|---|---:|---:|
| Mean final inertia | 166.41 million | 140.96 million |
| Mean MSE | 60.73 | 51.44 |
| Final inertia range | 133.70–180.95 million | 130.89–145.83 million |
| MSE range | 48.79–66.03 | 47.77–53.22 |
| Mean runtime | 14.41 s | 13.31 s |
| Unique colours | 8 in every run | 8 in every run |
| Iterations | 20 in every run | 20 in every run |

## Interpretation

Lower inertia and lower MSE indicate that the reconstructed pixels are, on average, closer to their original RGB values.

K-means++ achieved lower final inertia and lower MSE in four of the five seed comparisons. Random initialization performed better only for seed 3. However, random initialization had substantially greater variation between seeds, while K-means++ stayed within a narrower and generally lower error range.

The average final inertia for K-means++ was approximately 15.3% lower than for random initialization. The average MSE was also approximately 15.3% lower.

Both methods produced exactly eight unique colours in every run. This confirms that both correctly applied the requested palette size, but it does not distinguish their reconstruction quality.

The average runtime was slightly lower for K-means++ in this experiment. Runtime differences should be interpreted cautiously because they can be affected by system load and implementation details.

## Important limitation

Every run used all 20 iterations. Therefore, the experiment does not show that either method converged earlier. The runs may have stopped because they reached `max_iterations=20`.

Also, inertia and MSE measure closely related squared RGB reconstruction errors. They provide consistent evidence here, but they should not be treated as completely independent metrics.

## Conclusion

For this image and `k = 8`, K-means++ produced better and more stable results than random initialization. It should therefore be the default initialization method for the project, while random initialization should remain available for comparison and for studying stochastic variability.
