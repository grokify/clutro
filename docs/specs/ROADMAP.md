# CLUTRO — Roadmap (ROADMAP)

**Initiative:** `INIT-CLUTRO-001`
**Repository:** `github.com/grokify/clutro`

**Status:** Phases 1–5 planned (Phases 1–4 = v0.1.0, Phase 5 = v0.2.0)
**RMI slug:** `CLUTRO`

Phase status is always derived from member RMI statuses, never set directly. Review and execution happen by phase. Commits implementing an RMI carry the git trailer `Refs: RMI-CLUTRO-NNN`.

## Phase 1 — Matrix, Metric & RNG Foundation
**Theme:** Dependency-free sparse/dense primitives, similarity metrics, and the determinism machinery every algorithm will inherit — all benchmarked before any algorithm exists.

- [x] `RMI-CLUTRO-001` Module scaffold: go.mod, Apache-2.0 LICENSE, README with non-affiliation statement, CI (race tests, lint, CGO_ENABLED=0 cross-build), CHANGELOG.json — the determinism job lands with the first algorithm in Phase 2
- [x] `RMI-CLUTRO-002` matrix/: SparseVec, Rows interface, CSR and Dense, L2 normalization with cached norms, merge-join dot products, Composite with incremental add/remove; benchmarks
- [x] `RMI-CLUTRO-003` metric/: Similarity interface, Cosine (normalized fast path), Euclidean as distance and similarity
- [x] `RMI-CLUTRO-004` internal/rng (PCG64 sub-streams, seeded shuffle) and internal/par (bounded pool, positional writes, worker-order reduction)
- [x] `RMI-CLUTRO-005` internal/synth planted-structure generator (sparse and dense) and the float32-storage precision test

## Phase 2 — Criterion I2, k-means Family & Repeated Bisection
**Theme:** The CLUTO core — I2 with incremental deltas, k-means++/spherical k-means, repeated bisection with trials and selection policies, refinement — behind one `Fit` facade returning labels and a bisection tree.

- [ ] `RMI-CLUTRO-006` criterion/: Criterion interface and I2 (Value, Delta) with hand-computed numeric tests
- [ ] `RMI-CLUTRO-007` partition/kmeans: k-means++ init with seeded trials, Lloyd, spherical variant, parallel assignment
- [ ] `RMI-CLUTRO-008` partition/bisect: 2-way bisection with trials, `best`/`largest` selection, Tree with CutMaxSize and CutK
- [ ] `RMI-CLUTRO-009` partition/refine: incremental criterion-driven refinement (`rbr`) and Direct (k-way init + refine)
- [ ] `RMI-CLUTRO-010` clutro facade: Fit, Options with CLUTO-matching defaults, Result, typed errors; `partition/reference` oracle implementations; determinism CI job asserting identical labels across GOMAXPROCS, trial scheduling, and against the reference
- [ ] `RMI-CLUTRO-026` Density interop: `matrix.Dense` → `kinn-gg/hdbscan` `Dense64` adapter and a kNN cosine similarity graph → `SparsePrecomputed` distance graph, as a compiled example (no library dependency on hdbscan)

## Phase 3 — CLUTO Formats, CLI & Golden-Set Validation
**Theme:** Drop-in migration for CLUTO users and a measured answer to "how close to CLUTO is it?"

- [ ] `RMI-CLUTRO-011` cluto/: readers and writers for sparse/dense matrices, .clabel/.rlabel/.rclass, .clustering.<k>; round-trip and malformed-input tests
- [ ] `RMI-CLUTRO-012` evaluate/ external metrics: adjusted Rand index, NMI, purity, entropy
- [ ] `RMI-CLUTRO-013` cmd/clutro: `vcluster` with CLUTO flag vocabulary and `-json`, `evaluate`; progress and exit codes
- [ ] `RMI-CLUTRO-014` Golden-set harness: per-corpus ARI and matched-label table against CLUTO outputs under `none`/`idf` column models, manual inspection of outliers, accepted band asserted; numeric-only renamed fixture subset in testdata/cluto with SHA256SUMS
- [ ] `RMI-CLUTRO-015` Preprocessing: `-rowmodel l2` and `-colmodel idf` so raw count matrices cluster as CLUTO does

## Phase 4 — Explanation, Internal Metrics & v0.1.0
**Theme:** Clusters that explain themselves and runs that can be scored without truth labels; the first tagged release with a production consumer.

- [ ] `RMI-CLUTRO-016` explain/: descriptive and discriminating features with cumulative share, representatives, ISim/ESim summaries; column labels option; `-showfeatures`, `-showsummaries`, `describe`
- [ ] `RMI-CLUTRO-017` evaluate/ internal metrics: silhouette (seeded sampling), Davies–Bouldin, Calinski–Harabasz; scikit-learn cross-check script
- [ ] `RMI-CLUTRO-018` examples/: sparse text with explanation, dense embeddings, tree cut, CLUTO migration — compiled in CI; README example sourced from examples/sparse
- [ ] `RMI-CLUTRO-019` Docs: `docs/{algorithms,numerics,compatibility,production,benchmarking}.md` — algorithm reference with citations, numerics and determinism contract, "differences from CLUTO", benchmark page from bench/ runs with recorded environment
- [ ] `RMI-CLUTRO-020` Release v0.1.0: CHANGELOG, coverage badge, CI-green tag; first consumer switched to the tag

## Phase 5 — v0.2: Full Criterion Family & Hierarchical
**Theme:** Complete CLUTO's optimization vocabulary and extend it to agglomerative clustering, keeping the same delta discipline and determinism.

- [ ] `RMI-CLUTRO-021` criterion/: I1, E1, G1, G1′ with incremental deltas; numeric tests per criterion
- [ ] `RMI-CLUTRO-022` criterion/: H1 and H2 ratio criteria with numerator/denominator delta tracking; `-crfun` complete
- [ ] `RMI-CLUTRO-023` `rbr` per-level refinement option and golden-set comparison against end-only refinement
- [ ] `RMI-CLUTRO-024` hierarchical/: criterion-driven agglomerative clustering, dendrogram, CutK, `-fulltree` over rb leaves
- [ ] `RMI-CLUTRO-025` Jaccard and Pearson similarities; v0.2.0 release with per-criterion benchmarks

## Post-v0.2 backlog (unphased)

Not yet broken into RMIs; promote into a phase once a consumer needs it.

- Similarity-graph input and multilevel graph clustering (`scluster` equivalent)
- Feature-subspace cluster selection (CLUTO `-cstype=…` variants beyond best/largest)
- Incremental/streaming assignment of new rows to an existing partition (cheap) vs. online re-clustering (expensive; probably out of scope)
- Stability analysis (bootstrap ARI across seeds/subsamples)
- Optional float64 storage via a type parameter
- Nearest-neighbor acceleration for very large k (inverted-index candidate generation for sparse rows)
