# CLUTRO

**Modern clustering toolkit for Go.** Cluster sparse or dense rows with
repeated bisection or spherical k-means, optimize a real criterion function,
explain every cluster by its features, and get the same answer every time.

CLUTRO brings the ideas that made [CLUTO](https://conservancy.umn.edu/items/4fbef165-f964-41ed-a239-86a8f931ffbe)
the standard tool for document clustering — criterion-driven partitioning,
repeated bisection, cluster interpretation — to a pure-Go library with no
dependencies, `context.Context` cancellation, and a tested determinism
contract.

!!! note "Independent implementation"
    CLUTRO is inspired by CLUTO and implements the published algorithms from
    scratch. It is not affiliated with, derived from, or endorsed by the
    Karypis lab or the University of Minnesota.

## A first look

```go
m, _ := matrix.NewCSRFromTriplets(nRows, nCols, triplets)
m.NormalizeRowsL2()

res, err := clutro.Fit(ctx, m,
    clutro.WithAlgorithm(clutro.RepeatedBisection),
    clutro.WithClusters(12),
    clutro.WithRefinement(true),  // CLUTO "rbr"
    clutro.WithSeed(42),
    clutro.WithColumnLabels(vocab),
)
for k := range res.K {
    fmt.Println(res.Sizes()[k], res.Describe(k, 5)) // top terms per cluster
}
```

```bash
clutro vcluster -clmethod=rbr -rowmodel=l2 docs.mat 50   # writes docs.mat.clustering.50
clutro evaluate docs.mat.clustering.50 docs.rclass        # ARI, NMI, purity, entropy
```

## Why CLUTRO

- **Sparse-first.** CSR matrices with float32 values; 100k-row TF-IDF
  corpora fit in memory and cluster in seconds.
- **The right objective.** Spherical k-means and repeated bisection maximize
  I2 (Σ‖D_k‖), the criterion that works for cosine geometry, with
  refinement that moves single rows while it improves.
- **Explained clusters.** Descriptive and discriminating features,
  representative rows, and internal/external similarity per cluster.
- **Deterministic.** Same input, same seed, same version ⇒ same labels,
  whatever `GOMAXPROCS` is. Optimized paths are tested against slow reference
  implementations.
- **CLUTO-compatible where it matters.** The same file formats, flag names
  and defaults, validated against CLUTO's own outputs on 103 real corpora.
- **Zero dependencies**, library and CLI alike.

## Where to go next

- [Installation](guides/installation.md) and [Quick Start](guides/quickstart.md)
- [Concepts](guides/concepts.md) — matrices, criteria, the bisection tree
- [Algorithms](guides/algorithms.md) — rb, rbr, direct, k-means, and when to use which
- [Explaining Clusters](guides/explaining.md) and [Evaluating Clusterings](guides/evaluating.md)
- [Numerics and Determinism](guides/numerics.md)
- [CLI Reference](guides/cli-reference.md), [CLUTO Compatibility](guides/cluto-compatibility.md), [Benchmarks and Validation](guides/benchmarks.md)
