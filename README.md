# CLUTRO

Modern Clustering Toolkit — a pure-Go library and CLI for clustering sparse, dense, and (later) graph-structured data, built around the optimization-criterion approach that made [CLUTO](https://conservancy.umn.edu/items/4fbef165-f964-41ed-a239-86a8f931ffbe) valuable: repeated bisection, spherical k-means, pluggable criterion functions, and cluster interpretation.

CLUTRO is inspired by CLUTO. It is an independent clean-room implementation of the published algorithms and is not affiliated with, derived from, or endorsed by the Karypis lab or the University of Minnesota.

## Status

Pre-release, under active development toward v0.1.0. Implemented so far: sparse/dense matrices, cosine and Euclidean similarity, the I2 criterion, k-means, repeated bisection with refinement, direct k-way clustering, the `Fit` facade, and external evaluation metrics. Still to come for v0.1.0: CLUTO file formats, the CLI, cluster explanation, internal metrics. The design is specified in [`docs/specs/`](docs/specs/) (PRD, TRD, PLAN, ROADMAP). Nothing below should be read as a performance or compatibility claim until the corresponding roadmap item is checked off.

## Why

Go has several dense k-means ports and no clustering engine that handles the shape of real text and embedding workloads: tens of thousands to millions of **sparse** rows, cosine similarity, a criterion function to optimize rather than a distance to minimize, and an answer to "why are these together?" CLUTRO fills that gap and stays an engine — embedding generation, vector stores, LLM calls, and visualization remain the caller's business.

## Usage

```go
import (
    "github.com/grokify/clutro"
    "github.com/grokify/clutro/matrix"
)

m, err := matrix.NewCSRFromTriplets(nRows, nCols, triplets) // or matrix.NewDense(nRows, nCols, values)
if err != nil {
    return err
}
m.NormalizeRowsL2()

res, err := clutro.Fit(ctx, m,
    clutro.WithAlgorithm(clutro.RepeatedBisection), // rb; Direct and KMeans also available
    clutro.WithClusters(12),
    clutro.WithRefinement(true), // CLUTO "rbr"
    clutro.WithSeed(42),
)
if err != nil {
    return err
}
for k, rows := range res.Clusters() {
    fmt.Println(k, len(rows))
}
tree := res.Tree()            // bisection hierarchy (RepeatedBisection only)
_ = tree.CutMaxSize(12)       // re-partition to ≈12 rows per leaf without re-clustering
```

Defaults match CLUTO's: cosine similarity, the I2 criterion, 10 seeded trials, `best` cluster selection. Same input, same seed, same version ⇒ same labels, regardless of `GOMAXPROCS` — the optimized paths are tested against slow reference implementations and a CI job runs the suite under several `GOMAXPROCS` values. Cluster explanation (`Describe`) and CLUTO file formats are the next roadmap items.

## CLI

`clutro vcluster <matrix> <k>` accepts CLUTO's sparse matrix format (`.mat` plus optional `.clabel`/`.rlabel`) and writes `<matrix>.clustering.<k>`, so existing CLUTO workflows can be migrated by swapping the binary.

## License

Apache-2.0 — see [LICENSE](LICENSE).
