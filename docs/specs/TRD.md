# CLUTRO — Technical Requirements Document (TRD)

**Status:** Draft
**Date:** 2026-10-14
**Repo:** github.com/grokify/clutro

## 1. Architecture overview

A small, layered, dependency-free library. Every algorithm is written against two interfaces — a row-readable matrix and a criterion — so adding a criterion, a metric, or a matrix type never touches the algorithms.

```text
                 clutro (facade: Fit, Options, Result)
                               │
   ┌───────────┬───────────────┼────────────────┬──────────────┐
   │ partition │  hierarchical │   explain      │   evaluate   │
   │ kmeans    │  (v0.2)       │ describe/      │ ARI NMI      │
   │ skmeans   │  agglo        │ discriminate/  │ purity       │
   │ bisect    │               │ representatives│ silhouette … │
   │ direct    │               │                │              │
   └─────┬─────┴───────┬───────┴───────┬────────┴──────┬───────┘
         │             │               │               │
   ┌─────▼─────────────▼───────────────▼───────────────▼───────┐
   │ criterion   I2 (v0.1) · I1 E1 G1 G1′ H1 H2 (v0.2)          │
   │ metric      Cosine · Euclidean · (Jaccard, Pearson later)  │
   │ matrix      CSR · Dense · Rows interface · composite vecs  │
   │ internal/rng  PCG, per-trial streams, deterministic reduce │
   └────────────────────────────────────────────────────────────┘
         ▲                                   ▲
   cluto/ (file formats: .mat .clabel .rlabel .clustering)
   cmd/clutro (CLI: vcluster, evaluate, describe)   examples/   bench/
```

Design rules:

- **Library packages import only the standard library.** `cobra` is permitted in `cmd/clutro` only. No cgo anywhere.
- **Algorithms never allocate per row in the inner loop.** Work buffers are sized once per call; sparse dot products are merge-joins over sorted column indices.
- **Determinism is a tested contract**, not an accident: every random choice derives from the caller's seed through named sub-streams, and every parallel reduction is order-independent or re-sorted before use.
- **`context.Context` on every exported operation that iterates**; cancellation is checked once per outer iteration and returns `ctx.Err()` wrapped with progress information.
- **Errors are returned, never logged.** Invariant violations (NaN in a normalized row, k > rows) are returned as typed errors from validation before any work starts; internal impossibilities panic.

## 2. Module and dependencies

- **Module:** `github.com/grokify/clutro`, Go 1.26+
- **Library dependencies:** none
- **CLI:** `github.com/spf13/cobra`
- **Dev/CI:** `golangci-lint`, `gocoverbadge`; Python + scikit-learn only inside `bench/reference/` for cross-validation scripts, never imported by Go code
- Verify versions at adoption time (`go list -m -versions`); do not assume versions in this document are current.

## 3. Package layout

```text
clutro/
├── clutro.go              # Fit, Algorithm enum, Result accessors
├── options.go             # functional options (clusters, similarity, criterion, seed, trials, refinement, selection, parallelism)
├── result.go              # Result, Tree, per-cluster stats
├── errors.go              # typed errors
├── matrix/                # Rows interface, CSR, Dense, builders, L2 normalize, composite vector, dot products
├── metric/                # Similarity interface, Cosine, Euclidean (as similarity 1/(1+d) and as distance)
├── criterion/             # Criterion interface, I2; v0.2: I1 E1 G1 G1′ H1 H2
├── partition/             # KMeans (k-means++), SphericalKMeans, Bisect (rb), Refine (rbr), Direct
├── hierarchical/          # v0.2: agglomerative over Rows with criterion-driven merging, dendrogram
├── explain/               # Describe, Discriminate, Representatives, Summaries
├── evaluate/              # ARI, NMI, purity, entropy, silhouette, Davies–Bouldin, Calinski–Harabasz
├── cluto/                 # CLUTO file formats: ReadMatrix (sparse/dense), ReadLabels, Read/WriteClustering
├── internal/rng/          # PCG64 wrapper, SubStream(seed, name) derivation, deterministic shuffles
├── internal/par/          # bounded worker pool with ordered reduction
├── cmd/clutro/            # Cobra CLI
├── examples/              # compiled-in-CI examples (sparse text, dense embeddings, tree cut, CLUTO migration)
├── bench/                 # Go benchmarks + reference scripts + generated datasets (gitignored outputs)
├── testdata/              # small synthetic fixtures; CLUTO format samples
└── docs/specs/
```

## 4. Core types

```go
// matrix
type Rows interface {
    NumRows() int
    NumCols() int
    Row(i int) SparseVec          // for Dense: a view with Idx = 0..n-1 (no allocation beyond the view)
    RowNorm(i int) float32        // cached L2 norm
}
type SparseVec struct { Idx []uint32; Val []float32 }   // Idx sorted ascending
type CSR struct { rowPtr []uint32; idx []uint32; val []float32; norms []float32; nCols int }
type Dense struct { data []float32; nCols int; norms []float32 }

// composite vector of a cluster: D_k = Σ rows ∈ S_k, kept dense (len = NumCols) for speed; sparse fallback when NumCols is huge
type Composite struct { Sum []float64; N int; Norm float64 }

// metric
type Similarity interface {
    Sim(a, b SparseVec, normA, normB float32) float64
    RowToComposite(r SparseVec, normR float32, c *Composite) float64
    Name() string
}

// criterion
type Criterion interface {
    Name() string
    Value(comps []Composite) float64                 // global objective
    Maximize() bool
    Delta(from, to *Composite, r SparseVec, normR float32) float64 // change if r moves from→to; must be O(nnz(r))
}

// partition
type Assignment struct { Labels []int32; K int }
type Tree struct { Root *Node } ; type Node struct { Rows []int32; Left, Right *Node; Criterion float64 }

// facade
type Result struct { /* Assignment, Composites, Tree (optional), Options, stats */ }
func Fit(ctx context.Context, m matrix.Rows, opts ...Option) (*Result, error)
```

The `Rows` interface is deliberately narrow so a caller with a custom store (e.g. memory-mapped vectors) can implement it.

## 5. Algorithms (v0.1)

### 5.1 k-means and spherical k-means

- Init: k-means++ with `trials` candidate sets; each trial has its own RNG sub-stream `rng.SubStream(seed, "kmeans++", trial)`.
- Lloyd iterations until no label changes or `maxIter` (default 20, matching CLUTO's `-niter`). Spherical variant normalizes centroids and uses cosine.
- Tie-breaking: lowest cluster index wins; rows are visited in index order; parallel assignment writes to a per-row slot so order never affects results.

### 5.2 Repeated bisection (`rb`)

CLUTO's default and the method of record for text:

1. Start with all rows in one cluster.
2. Repeat k−1 times: choose the cluster to split by **selection policy** — `largest` (most rows) or `best` (the split that most improves the criterion; computed lazily and cached per cluster, invalidated on split). Default `best`, as CLUTO.
3. Bisect it with 2-way spherical k-means from `trials` seeded starts (default 10, CLUTO `-ntrials`), each run to convergence; keep the bisection with the best criterion value. Clusters with < 2 rows are not split.
4. Record both children in the `Tree`.

The result is both a flat k-way assignment and a binary tree whose leaves are the k clusters; `Tree.CutMaxSize(n)` and `Tree.CutK(k')` re-partition without re-clustering.

### 5.3 Refinement (`rbr`, and `direct`)

Given an assignment, repeat until no improvement or `maxIter`: visit rows in a seeded random order; for each row compute `criterion.Delta` to every other cluster (O(k · nnz(r)) with composite vectors kept dense); move if the best delta > 0 and apply the move immediately (incremental composite update). `rbr` = `rb` then refine the final k-way partition (CLUTO also refines at each bisection level when `-rbr`; v0.1 refines at the end, the per-level variant is an option). `direct` = k-means++ initial k-way split then refine.

### 5.4 Criterion I2

For cosine on L2-normalized rows, cluster k's composite `D_k = Σ_{i∈S_k} d_i`, and

- `I2 = Σ_k ‖D_k‖` (maximize).
- `Delta(from, to, r) = (‖D_to + r‖ − ‖D_to‖) + (‖D_from − r‖ − ‖D_from‖)`, computed from the stored norms and two sparse dot products: `‖D ± r‖² = ‖D‖² ± 2 D·r + ‖r‖²`.

v0.2 criteria (same interface, same delta discipline; definitions per Zhao & Karypis, *Machine Learning* 55, 2004):

| Criterion | Objective | Sense |
|-----------|-----------|-------|
| I1 | Σ_k (1/n_k) ‖D_k‖² | max |
| I2 | Σ_k ‖D_k‖ | max |
| E1 | Σ_k n_k · (D_kᵀD)/‖D_k‖ | min |
| G1 | Σ_k (D_kᵀ D)/‖D_k‖² (cut / internal) | min |
| G1′ | Σ_k n_k² · (D_kᵀ D)/‖D_k‖² | min |
| H1 | I1 / E1 | max |
| H2 | I2 / E1 | max |

where `D` is the composite of all rows.

### 5.5 Parallelism and determinism

- The distance/similarity sweep of each iteration is split across a bounded pool (`WithParallelism(n)`, default `runtime.NumCPU()`); each worker writes results for its row range into pre-allocated slices, so the merge is positional and order-free.
- Any reduction that sums floats across workers sums per-worker partials in worker-index order (fixed), never in completion order.
- Trials run sequentially by default (their RNG streams are independent, so running them in parallel is a pure optimization and must produce the same winner; a test asserts this).
- RNG: PCG64 (`math/rand/v2` `PCG`), sub-streams derived by hashing `(seed, name, index)`.

### 5.6 Explanation (`explain/`)

- **Descriptive features** of cluster k: columns j ranked by their contribution to the within-cluster similarity, `D_k[j]² / ‖D_k‖²`; report the top-n and the cumulative share (CLUTO prints enough features to reach a percentage).
- **Discriminating features**: columns ranked by `D_k[j]·(D[j] − D_k[j])` contribution to the between-cluster similarity, i.e. what most separates k from the rest.
- **Representatives**: rows with highest `RowToComposite` similarity to their cluster's composite.
- **Summaries**: per cluster, size, mean internal similarity (ISim), mean external similarity (ESim), and their standard deviations.

Feature names come from an optional `[]string` column-label slice (CLUTO `.clabel`) supplied via `WithColumnLabels`; otherwise indices are reported.

## 6. CLUTO file formats (`cluto/`)

- **Sparse matrix:** header `nrows ncols nnz`, then one row per line of `col val` pairs with **1-based** columns. Dense: `nrows ncols` then values.
- **Labels:** `.clabel` (one column name per line), `.rlabel` (row names), `.rclass` (true class per row, for evaluation).
- **Clustering:** `<matrix>.clustering.<k>`, one 0-based cluster id per row, `-1` for unassigned.
- Readers are streaming and validate shape; writers are deterministic.

## 7. CLI (`cmd/clutro`)

```text
clutro vcluster [flags] <matrix> <k>
  -clmethod rb|rbr|direct|kmeans|skmeans   (default rb)
  -crfun    i2|i1|e1|g1|g1p|h1|h2         (default i2; non-i2 from v0.2)
  -sim      cos|euclid                     (default cos)
  -cstype   best|largest                   (default best)
  -ntrials N  -niter N  -seed N  -parallel N
  -rowmodel none|l2      -colmodel none|idf  (preprocessing, CLUTO vocabulary)
  -showfeatures [-nfeatures N]  -showsummaries  -showtree
  -clustfile PATH  -fulltree PATH  -json
clutro evaluate [-truth FILE | -rclass] <matrix> <clustering>
clutro describe [-nfeatures N] <matrix> <clustering>
```

Flag names deliberately mirror CLUTO's so that scripts port by substitution; `-json` emits the full `Result` for programmatic use.

## 8. Evaluation (`evaluate/`)

External (need truth labels): adjusted Rand index, normalized mutual information, purity, entropy (CLUTO's two). Internal: silhouette (sampled above 10k rows, sample seeded), Davies–Bouldin, Calinski–Harabasz. All pure functions over `[]int32` labels and `Rows`.

## 9. Validation strategy

0. **Reference oracle in-tree.** `partition/reference` holds slow, obviously-correct implementations of k-means, bisection, and refinement (no parallelism, no incremental deltas, dense composites). The optimized paths must reproduce the reference's labels exactly on every fixture; tie-breaking rules are defined by the reference and documented. (Pattern borrowed from `kinn-gg/hdbscan`'s `Reference` vs `Exact` split.)
1. **Hand-computable fixtures** (≤ 12 rows): I2 values, deltas, and the exact bisection result are asserted numerically.
2. **Planted-structure corpora** (`testdata/gen`): synthetic sparse matrices with known clusters; ARI ≥ 0.95 asserted for `rb`/`rbr`/`direct`.
3. **CLUTO golden set:** CLUTO's own inputs and outputs for real corpora — the sparse matrix `vcluster` read and the `.clustering.<k>` it wrote. The published subset in `testdata/cluto/` is **numeric only**: no `.clabel`/`.rlabel`, corpora renamed `corpus-NN`, chosen to span sizes (three of ~100–300 rows, three of ~500–1,500, two of ~3,000–6,500 rows with up to ~100k nonzeros) with their original `k`, under 2 MB total, listed in `SHA256SUMS.json`. Because CLUTO's RNG stream cannot be reproduced, agreement is measured with ARI (and per-row label agreement after Hungarian matching), reported per corpus as a table in CI; an accepted lower band is set in Phase 3 after inspection and then asserted. The harness also runs with `-colmodel none` and `idf` and reports both, since CLUTO's effective default on these files is established empirically rather than assumed. The full ~100-corpus set stays with its owner, who runs the same harness privately.
4. **scikit-learn cross-check** as pinned parity fixtures: `bench/reference/` scripts generate `testdata/parity/*.json` (named per case: `kmeans_fixed_init`, `skmeans_duplicates_ties`, `silhouette_sampled`, …) from a pinned scikit-learn version recorded in `NOTICE`, with a `SHA256SUMS.json`; Go tests consume the JSON and never call Python. k-means and spherical k-means with identical init centers must converge to identical labels; silhouette/DB/CH/ARI/NMI values must match to 1e-6.
5. **Determinism job:** the whole suite at `GOMAXPROCS=1` and `GOMAXPROCS=8`, labels diffed.

## 10. Performance targets and benchmarks

| Workload | Target (v0.1, laptop, 8 cores) |
|----------|-------------------------------|
| 10k rows × 1M nnz, `rb` k=50 | < 2 s |
| 100k rows × 10M nnz, `rb` k=100 | < 60 s |
| 100k rows × 10M nnz, `rbr` k=100 | < 120 s |
| 1M rows × 100M nnz, `rb` k=500 | completes in < 30 min; memory ≤ 3× CSR size |
| Dense 100k × 1536, `skmeans` k=100 | < 60 s |

Published numbers come from `bench/` runs recorded in release notes; the table above is the target, not a claim.

## 11. Testing and quality gates

- `go test ./...` with race detector; coverage badge; `golangci-lint` clean; `CGO_ENABLED=0` cross-build matrix in CI.
- `examples/` compile and run in CI; the README example is `examples/sparse/main.go` verbatim.
- API stability: pre-v1 may break between minors with a CHANGELOG entry; v1.0 freezes `clutro`, `matrix`, `metric`, `criterion`, `evaluate`, `cluto` public APIs.

## 12. Publication hygiene

The repo is public from its first push. Commit messages, fixtures, and docs contain no private project data, credentials, or internal paths. The CLUTO golden corpus is referenced by name only until the owner publishes a subset; CI fetches nothing private.
