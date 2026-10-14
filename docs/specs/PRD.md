# CLUTRO — Product Requirements Document (PRD)

**Status:** Draft
**Date:** 2026-10-14
**Repo:** github.com/grokify/clutro

## 1. Summary

CLUTRO is a pure-Go clustering toolkit for sparse, dense, and (later) graph-structured data. It brings the ideas that made CLUTO (Karypis, 2002) the standard tool for document clustering — criterion-driven partitional clustering, repeated bisection, spherical k-means, and cluster interpretation — to a modern Go API with deterministic results, `context.Context` cancellation, and no cgo.

CLUTRO is a clustering **engine**, not an ML framework. It takes a matrix and returns an assignment plus an explanation. Embedding generation, vector databases, LLM calls, and visualization are the caller's responsibility.

Naming: the project is **CLUTRO** (all caps, like CLUTO, LLVM); the Go module, package, and executable are `clutro`. No acronym expansion. CLUTRO is an independent clean-room implementation of published algorithms and is not affiliated with, derived from, or endorsed by the Karypis lab or the University of Minnesota; the README and every release note say so.

## 2. Problem

- **Go has no sparse-aware clustering engine.** Existing packages (`mpraski/clusters`, `muesli/kmeans`, `pa-m/sklearn`) take dense `[][]float64`, are unmaintained, and cannot hold a 100k × 500k TF-IDF matrix. `james-bowman/nlp` has sparse TF-IDF/LSA but no clustering and has been dormant since 2021. `go-nlp/dmmclust` is a short-text topic model, not a general clusterer.
- **CLUTO is unavailable.** It was distributed as research-use binaries for 2000s platforms, is copyrighted and not open-source, and its clustering behaviour (repeated bisection optimizing I2 over cosine similarity) has no faithful reimplementation in any language that a Go program can call without cgo or a subprocess.
- **Modern clustering libraries optimize the wrong thing for text.** Lloyd's k-means minimizes Euclidean SSE; for L2-normalized sparse vectors the right objective is a similarity-based criterion (I2, H2, …), and the right algorithm is bisection with refinement. Few libraries expose criterion choice at all.
- **Nobody explains clusters.** "Cluster 7 has 41 documents" is not an answer; "cluster 7 is about `ldap`, `directory`, `authentication`" is. CLUTO's descriptive/discriminating features have no mainstream equivalent.

## 3. Target users

| User | Need |
|------|------|
| **Go developers clustering text** (TF-IDF, BM25 vectors) | Sparse input, cosine, sensible defaults, deterministic output, an explanation per cluster |
| **Developers clustering embeddings** (OpenAI, local models) | Dense input, spherical k-means / bisection, quality metrics, no cgo |
| **Former CLUTO users** | The same algorithms and vocabulary (`rb`, `rbr`, `i2`, `cos`, `.mat`/`.clabel`), a drop-in CLI, documented differences |
| **Library authors** | A small dependency-free core they can embed (search engines, dedup, log grouping, glossary/related-items features) |
| **Researchers** | Correct criterion functions with citations, reproducibility, benchmarks against reference implementations |

The first consumer is a private text application that groups related items across ~100 topic corpora of up to ~8k sparse documents each; it previously used CLUTO and supplies the golden corpus described in §9.1.

## 4. Goals (V1 = v0.1 → v0.2)

- **G1 — Sparse-first.** CSR matrices are the primary input; dense is supported through the same interface. A 100k-row, 10M-nnz corpus clusters in memory on a laptop.
- **G2 — Criterion-driven partitional clustering.** Repeated bisection (`rb`), repeated bisection with refinement (`rbr`), direct k-way (`direct`), and k-means/spherical k-means, optimizing a selectable criterion. v0.1 ships I2; v0.2 adds I1, E1, G1, G1′, H1, H2 with incremental delta updates for refinement.
- **G3 — Interpretation.** Descriptive and discriminating features per cluster (CLUTO `-showfeatures`), representative rows (closest to centroid), per-cluster internal/external similarity statistics (`-showsummaries`-style).
- **G4 — Determinism contract.** Same input, same options, same seed, same CLUTRO version ⇒ identical labels, independent of `GOMAXPROCS`. Tested in CI.
- **G5 — CLUTO file compatibility.** Read/write CLUTO sparse and dense matrix files, `.clabel`/`.rlabel`, and `.clustering.<k>`; the `clutro vcluster` CLI uses CLUTO's flag vocabulary so existing scripts port by changing the binary name.
- **G6 — Evaluation.** External metrics (adjusted Rand index, NMI, purity, entropy) and internal metrics (silhouette, Davies–Bouldin, Calinski–Harabasz) so users can compare runs and so CI can compare CLUTRO with reference implementations.
- **G7 — Engine, not framework.** Zero third-party dependencies in the library packages (standard library only; `cobra` in `cmd/` only). No cgo. `context.Context` on every long operation.
- **G8 — Twenty-line example.** A working sparse-clustering example with explanation fits in 20 lines of Go.

## 5. Non-goals

- **Not in V1:** graph clustering (`scluster` equivalent, multilevel partitioning), agglomerative/divisive hierarchical with arbitrary linkage, streaming/incremental clustering. These are roadmap candidates for v0.3+, each judged on its own merits.
- **Density methods (DBSCAN/HDBSCAN/OPTICS) are not planned at all.** A dependency-free, deterministic Go HDBSCAN already exists (`github.com/kinn-gg/hdbscan`, a port of scikit-learn-contrib's hdbscan with parity fixtures). CLUTRO documents how to hand it a dense matrix or a CLUTRO-built similarity graph rather than reimplementing it.
- **Never:** embedding generation, tokenization/TF-IDF (the caller or a sibling text package produces the matrix), vector-database integration, LLM-generated cluster descriptions, GUI/visualization. CLUTRO may document how to combine with such tools; it does not depend on them.
- **No binary or API compatibility with CLUTO.** Algorithmic fidelity (same objective, same algorithm, comparable results) is the goal, not reproducing a C interface or matching CLUTO's random number stream.

## 6. Core concepts

- **Matrix** — rows are objects, columns are features. `matrix.CSR` (sparse) and `matrix.Dense` implement one read interface used by every algorithm. Rows are typically L2-normalized so cosine similarity is a dot product.
- **Similarity vs distance** — algorithms are written against a `metric.Similarity`; distances are wrapped. Cosine and Euclidean in V1.
- **Criterion function** — a global objective over a partition, e.g. I2 = Σ_k ‖D_k‖ (sum of composite-vector norms, maximize). The algorithm's job is to find a partition that optimizes it; `criterion.Delta` gives the change from moving one row, which is what refinement needs.
- **Repeated bisection** — start with one cluster; k−1 times, choose a cluster (largest, or the one whose split improves the criterion most), split it 2-way with several seeded trials, keep the best. Produces a binary tree as a by-product.
- **Refinement** — after bisection or direct clustering, iteratively move rows between clusters while the criterion improves (CLUTO `rbr`, and the inner loop of `direct`).
- **Explanation** — for each cluster, the features that contribute most to within-cluster similarity (descriptive) and most to separation from other clusters (discriminating); plus rows nearest the centroid.
- **Result** — labels, sizes, criterion value, per-cluster statistics, the bisection tree when applicable, and accessors for explanation.

## 7. Key user flows

```go
// Sparse text clustering, explained
m := matrix.NewCSRFromTriplets(nRows, nCols, triplets)
m.NormalizeRowsL2()
res, err := clutro.Fit(ctx, m, clutro.WithAlgorithm(clutro.RepeatedBisection),
    clutro.WithClusters(50), clutro.WithRefinement(true), clutro.WithSeed(1))
for k := range res.K() {
    fmt.Println(res.Size(k), res.Describe(k, 5), res.Representatives(k, 3))
}

// Dense embeddings
d := matrix.NewDense(nRows, 1536, values)
res, err := clutro.Fit(ctx, d, clutro.WithAlgorithm(clutro.SphericalKMeans), clutro.WithClusters(20))

// Pick k after the fact from the bisection tree
tree := res.Tree()            // nil unless rb/rbr
labels := tree.CutMaxSize(12) // ≈12 rows per leaf
```

```sh
# CLUTO migration
clutro vcluster -clmethod=rbr -crfun=i2 -sim=cos -showfeatures docs.mat 50
# → docs.mat.clustering.50, features on stdout
clutro evaluate --truth labels.txt docs.mat.clustering.50   # ARI, NMI, purity, entropy
```

## 8. Success metrics

| Dimension | Target |
|-----------|--------|
| Correctness | I2 and refinement validated against (a) hand-computable fixtures, (b) a 103-corpus CLUTO golden set (ARI reported per corpus; band set in Phase 3), (c) scikit-learn for k-means/spherical k-means on dense data |
| Reproducibility | CI job runs the suite at `GOMAXPROCS=1` and `=8` and diffs labels: must be identical |
| Performance | 100k × 10M-nnz CSR, `rb`, k=100: < 60 s on a laptop in v0.1; benchmark numbers published per release, never claimed in advance |
| Scale | Benchmarks at 10k, 100k, 1M rows; memory ≤ 3× the CSR size |
| Usability | README example ≤ 20 lines and compiles in CI (`examples/`) |
| Portability | `CGO_ENABLED=0` builds on linux/darwin/windows, amd64/arm64 |
| Adoption | The first consumer in production on CLUTRO; at least one external Go project importing it within a year of v1.0 |

## 9. Open questions

1. ~~Golden corpus publication~~ — **decided:** a numeric-only subset is published under `testdata/cluto/` under the repository license (see TRD §9.3): CLUTO sparse matrices and `.clustering.<k>` files with row and column labels stripped and corpora renamed `corpus-NN`. No vocabulary, no text, no identifiers; the full set stays with its owner and runs there.
2. **Float width.** `float32` storage with `float64` accumulation is the proposal (halves memory, adequate for cosine on normalized rows). Confirm after the Phase 1 benchmark.
3. **Generics.** A `Number` type parameter would let callers choose width but complicates every signature. Proposal: no generics in v0.1; revisit if a user needs `float64` storage.
4. **License.** Apache-2.0 as proposed; confirm before the first public push (MIT is the alternative).
