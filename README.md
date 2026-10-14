# CLUTRO

Modern Clustering Toolkit — a pure-Go library and CLI for clustering sparse, dense, and (later) graph-structured data, built around the optimization-criterion approach that made [CLUTO](https://conservancy.umn.edu/items/4fbef165-f964-41ed-a239-86a8f931ffbe) valuable: repeated bisection, spherical k-means, pluggable criterion functions, and cluster interpretation.

CLUTRO is inspired by CLUTO. It is an independent clean-room implementation of the published algorithms and is not affiliated with, derived from, or endorsed by the Karypis lab or the University of Minnesota.

## Status

Pre-release. The design is specified in [`docs/specs/`](docs/specs/) (PRD, TRD, PLAN, ROADMAP); implementation follows the roadmap phases. Nothing below should be read as a performance or compatibility claim until the corresponding roadmap item is checked off.

## Why

Go has several dense k-means ports and no clustering engine that handles the shape of real text and embedding workloads: tens of thousands to millions of **sparse** rows, cosine similarity, a criterion function to optimize rather than a distance to minimize, and an answer to "why are these together?" CLUTRO fills that gap and stays an engine — embedding generation, vector stores, LLM calls, and visualization remain the caller's business.

## Intended API (v0.1 target)

```go
import (
    "github.com/grokify/clutro"
    "github.com/grokify/clutro/criterion"
    "github.com/grokify/clutro/matrix"
    "github.com/grokify/clutro/metric"
)

m := matrix.NewCSR(rows, cols, nnz) // or matrix.NewDense(rows, cols)
// ... fill ...

res, err := clutro.Fit(ctx, m,
    clutro.WithAlgorithm(clutro.RepeatedBisection),
    clutro.WithClusters(12),
    clutro.WithSimilarity(metric.Cosine),
    clutro.WithCriterion(criterion.I2),
    clutro.WithRefinement(true), // CLUTO "rbr"
    clutro.WithSeed(42),
)
if err != nil {
    return err
}
for k, members := range res.Clusters() {
    fmt.Println(k, len(members), res.Describe(k, 5)) // descriptive features
}
```

Same input, same seed, same version ⇒ same labels, regardless of `GOMAXPROCS`.

## CLI

`clutro vcluster <matrix> <k>` accepts CLUTO's sparse matrix format (`.mat` plus optional `.clabel`/`.rlabel`) and writes `<matrix>.clustering.<k>`, so existing CLUTO workflows can be migrated by swapping the binary.

## License

Apache-2.0 — see [LICENSE](LICENSE).
