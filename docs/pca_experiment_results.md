# PCA Experiment Results

## Objective

Verify the from-scratch PCA implementation and apply it to sampled RGB pixels from the image-compression laboratory.

## Method

- Sampled 5,000 RGB pixels from the sample image.
- Applied the from-scratch PCA implementation.
- Compared explained-variance ratios with `sklearn.decomposition.PCA`.
- Projected pixels onto PC1 and PC2.
- Coloured the PCA scatter plot using K-means cluster assignments with `k = 8`.

## Results

| Component | Explained Variance | Explained Variance Ratio |
| --- | ---: | ---: |
| PC1 | 8718.77044 | 99.8171% |
| PC2 | 14.93162 | 0.1709% |
| PC3 | 1.04584 | 0.0120% |

PC1 and PC2 together preserve approximately **99.988%** of the total variance.

The from-scratch explained-variance ratios matched scikit-learn:

```text
From scratch:
[9.98170815e-01 1.70945086e-03 1.19733766e-04]

Scikit-learn:
[9.98170815e-01 1.70945086e-03 1.19733766e-04]

Ratios match sklearn: True
```

## Interpretation

Almost all variation in the sampled RGB data lies along the first principal component.

The PC1/PC2 scatter plot provides a two-dimensional view of the image pixels. K-means supplies the cluster labels; PCA only projects the RGB features into a lower-dimensional space for visualization.

## Conclusion

The experiment confirms that the from-scratch PCA implementation:

- produces valid principal components;
- calculates explained variance and explained-variance ratios correctly;
- agrees numerically with scikit-learn;
- successfully projects real image RGB data for cluster visualization.