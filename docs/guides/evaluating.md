# Evaluating Clusterings

The `evaluate` package scores a clustering against known classes (external
metrics) or from the data alone (internal metrics). Reference values are
cross-checked against scikit-learn; see `bench/reference/`.

## External metrics — with ground truth

| Function | Range | Meaning |
|----------|-------|---------|
| `AdjustedRandIndex(x, y)` | ≈0 … 1 | Agreement between two labelings corrected for chance; 1 = identical up to relabelling, ≈0 = independent. The standard way to compare two partitions. |
| `NormalizedMutualInformation(x, y)` | 0 … 1 | I(x;y)/√(H(x)·H(y)) (scikit-learn's geometric average). Less sensitive than ARI to many small clusters. |
| `Purity(clusters, classes)` | 0 … 1 | Fraction of rows whose cluster's majority class is their own class. |
| `Entropy(clusters, classes)` | 0 … 1 | Size-weighted class entropy per cluster, normalized by log(#classes); 0 is pure. CLUTO reports purity and entropy. |

Labels are `[]int32` of equal length; negative values are ordinary labels.
`cluto.ClassLabels` turns an `.rclass` file's strings into ids.

## Internal metrics — without ground truth

All take the matrix, the labels and k, use Euclidean distance (on
L2-normalized rows this ranks neighbours like cosine), and need 2 ≤ k < n.

| Function | Better | Meaning |
|----------|--------|---------|
| `Silhouette(m, labels, k, opts)` | higher, ≤ 1 | Mean over rows of (b − a)/max(a, b): a = mean distance to own cluster, b = smallest mean distance to another. Singleton clusters score 0. Exact up to 10,000 rows, then a seeded sample (`SampleAbove`, `Seed`). |
| `DaviesBouldin(m, labels, k)` | lower, ≥ 0 | Mean over clusters of the worst (s_i + s_j)/d(c_i, c_j); 0 is perfect; +Inf if two centroids coincide. |
| `CalinskiHarabasz(m, labels, k)` | higher | Between-cluster over within-cluster dispersion, [B/(k−1)]/[W/(n−k)]. |

Internal metrics compare clusterings of the *same* data (different k,
different algorithms). They are not comparable across datasets.

## Comparing with a reference clustering

```go
rs, skips, err := golden.RunAll(ctx, corpora, golden.Options{Trials: 10, RowModel: matrix.RowL2})
golden.WriteTable(os.Stdout, rs)
```

The `golden` package re-clusters every `<name>` with a `<name>.clustering.<k>`
file in a directory and reports ARI, NMI, and the I2 of both partitions — so
you can tell "different but equally good" from "worse". `clutro golden
<dir>` does the same from the command line. See
[Benchmarks and Validation](benchmarks.md) for what it says about CLUTO.
