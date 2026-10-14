# CLUTRO

[![Go CI][go-ci-svg]][go-ci-url]
[![Go Lint][go-lint-svg]][go-lint-url]
[![Determinism][det-svg]][det-url]
[![Docs][docs-godoc-svg]][docs-godoc-url]
[![License][license-svg]][license-url]

 [go-ci-svg]: https://github.com/grokify/clutro/actions/workflows/go-ci.yaml/badge.svg?branch=main
 [go-ci-url]: https://github.com/grokify/clutro/actions/workflows/go-ci.yaml
 [go-lint-svg]: https://github.com/grokify/clutro/actions/workflows/go-lint.yaml/badge.svg?branch=main
 [go-lint-url]: https://github.com/grokify/clutro/actions/workflows/go-lint.yaml
 [det-svg]: https://github.com/grokify/clutro/actions/workflows/determinism.yaml/badge.svg?branch=main
 [det-url]: https://github.com/grokify/clutro/actions/workflows/determinism.yaml
 [docs-godoc-svg]: https://pkg.go.dev/badge/github.com/grokify/clutro
 [docs-godoc-url]: https://pkg.go.dev/github.com/grokify/clutro
 [license-svg]: https://img.shields.io/badge/license-Apache--2.0-blue.svg
 [license-url]: https://github.com/grokify/clutro/blob/main/LICENSE

Modern Clustering Toolkit — a pure-Go library and CLI for clustering sparse and dense data, built around the optimization-criterion approach that made [CLUTO](https://conservancy.umn.edu/items/4fbef165-f964-41ed-a239-86a8f931ffbe) valuable: repeated bisection, spherical k-means, pluggable criterion functions, cluster explanation, and deterministic results. No dependencies, library or CLI.

CLUTRO is inspired by CLUTO. It is an independent clean-room implementation of the published algorithms and is not affiliated with, derived from, or endorsed by the Karypis lab or the University of Minnesota.

Documentation: [grokify.github.io/clutro](https://grokify.github.io/clutro/) — user guide, CLI reference, CLUTO compatibility, benchmarks, specs, release notes.

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
if err := matrix.Preprocess(m, matrix.RowL2, matrix.ColIDF); err != nil { // CLUTO-style TF-IDF rows
    return err
}

res, err := clutro.Fit(ctx, m,
    clutro.WithAlgorithm(clutro.RepeatedBisection), // rb; Direct and KMeans also available
    clutro.WithClusters(12),
    clutro.WithRefinement(true),                    // CLUTO "rbr"
    clutro.WithSeed(42),
    clutro.WithColumnLabels(vocab),                 // so explanations come back as words
)
if err != nil {
    return err
}
for k, rows := range res.Clusters() {
    fmt.Println(k, len(rows), res.Describe(k, 5)) // top descriptive features
}
tree := res.Tree()       // bisection hierarchy
_ = tree.CutMaxSize(12)  // re-partition to ≈12 rows per cluster without re-clustering
```

Defaults match CLUTO's: cosine similarity, the I2 criterion, 10 seeded trials, `best` cluster selection. Same input, same seed, same version ⇒ same labels, regardless of `GOMAXPROCS`; the optimized paths are tested against slow reference implementations, and a CI job runs the suite under several `GOMAXPROCS` values.

See [`examples/`](examples/) for runnable programs: sparse text with explanations, dense embeddings with internal metrics, choosing k from the tree, a CLUTO file workflow, and density clustering via [kinn-gg/hdbscan](https://github.com/kinn-gg/hdbscan).

## CLI

```bash
go install github.com/grokify/clutro/cmd/clutro@latest

clutro vcluster -clmethod=rbr -rowmodel=l2 -colmodel=idf docs.mat 50   # → docs.mat.clustering.50
clutro evaluate docs.mat.clustering.50 docs.rclass                      # ARI, NMI, purity, entropy
clutro golden ./corpora                                                 # compare with reference clusterings
```

`vcluster` accepts CLUTO's sparse matrix format and flag vocabulary, so existing CLUTO scripts port by swapping the binary.

## How close to CLUTO?

On 103 real text corpora clustered by CLUTO with its defaults, CLUTRO's partitions score **1.0007×** (`rb`) to **1.033×** (`rbr`) of CLUTO's on the criterion CLUTO optimized, in 6–15 s for all 103. Label agreement is partial (mean ARI 0.30–0.38) because k ≈ n/12 leaves many optima of equal quality — CLUTRO agrees with itself across seeds at ARI 0.48. Eight numeric-only corpora are published in [`testdata/cluto`](testdata/cluto/) with an asserted band. Details in [bench/README.md](bench/README.md).

## Status

v0.1.0 — matrices, cosine/Euclidean, I2, k-means, repeated bisection with refinement, direct k-way, explanation, external and internal evaluation metrics, CLUTO file formats, CLI, golden harness. v0.2 adds the remaining CLUTO criteria (I1, E1, G1, G1′, H1, H2), per-level refinement and agglomerative clustering. The design is specified in [`docs/specs/`](docs/specs/).

## License

Apache-2.0 — see [LICENSE](LICENSE).
