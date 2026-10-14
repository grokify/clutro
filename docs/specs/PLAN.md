# CLUTRO — Implementation Plan (PLAN)

**Status:** Draft
**Date:** 2026-10-14
**Repo:** github.com/grokify/clutro

This plan sequences the work defined in the [PRD](PRD.md) and [TRD](TRD.md). Roadmap items and phase membership are tracked in [ROADMAP.md](ROADMAP.md); this document explains ordering, dependencies, and verification per phase.

## Guiding constraints

- **Correct before fast, fast before broad.** Phase 1 proves the sparse primitives and determinism machinery with benchmarks; algorithms are added only on top of measured primitives; new algorithm families wait for v0.2+.
- **Every phase leaves `main` building with `CGO_ENABLED=0`, tested with `-race`, lint-clean, and with the README example compiling.**
- **Library packages stay dependency-free.** A `go.mod` diff that adds a library dependency is a design review, not a routine change.
- **Determinism is enforced from Phase 1**, so later algorithms inherit it instead of retrofitting it.
- **Public from the first commit:** generic technical commit messages, synthetic fixtures only, no private data or paths.
- **Verify dependency versions** (`cobra`) at adoption time.

## Phase 1 — Matrix, metric, RNG foundation

**Goal:** `matrix.CSR` / `matrix.Dense` behind `Rows`, cosine and Euclidean similarity, composite vectors with incremental update, deterministic RNG sub-streams and a bounded pool with ordered reduction — all benchmarked.

Order of work:

1. Module scaffold: `go.mod`, `LICENSE` (Apache-2.0), README, CI workflows (test with `-race`, lint, `CGO_ENABLED=0` cross-build matrix, determinism job stub), `CHANGELOG.json`.
2. `matrix/`: `SparseVec`, `Rows`, `CSR` (from triplets, from row slices), `Dense`, `NormalizeRowsL2`, cached norms, sparse·sparse and sparse·dense dot products (merge-join), `Composite` with `Add`/`Remove` and norm maintenance. Benchmarks for dot products at 100/1k/10k nnz.
3. `metric/`: `Similarity` interface; `Cosine` (dot on normalized rows, with a slow path for unnormalized), `Euclidean` as distance and as similarity.
4. `internal/rng/`: PCG64 sub-streams `SubStream(seed, name, idx)`, seeded shuffle; `internal/par/`: bounded pool, positional writes, worker-order reduction.
5. `testdata/gen`: planted-structure generator (sparse and dense) used by every later phase.

**Verify:** benchmarks recorded in `bench/README.md`; float32-storage/float64-accumulation decision confirmed by a precision test on 1M-nnz composites (PRD §9.2); `go test -race` clean.

## Phase 2 — Criterion I2, k-means family, repeated bisection

**Goal:** `clutro.Fit` with `KMeans`, `SphericalKMeans`, `RepeatedBisection` (+`WithRefinement`), `Direct`, optimizing I2, returning a `Result` with a `Tree`.

Order of work:

1. `criterion/`: interface, `I2` with `Value` and `Delta` (TRD §5.4), numeric tests against hand-computed fixtures.
2. `partition/kmeans.go`: k-means++ init with trials, Lloyd, spherical variant; parallel assignment via `internal/par`.
3. `partition/bisect.go`: 2-way bisection with trials; selection policies `best` and `largest`; `Tree` construction; `CutMaxSize`, `CutK`.
4. `partition/refine.go`: incremental refinement using `Criterion.Delta`; `Direct` = k-way init + refine.
5. `clutro.go`, `options.go`, `result.go`, `errors.go`: facade, validation, option defaults mirroring CLUTO (`rb`, `i2`, `cos`, `best`, 10 trials, 20 iterations).
6. Density interop example (`examples/hdbscan`): adapt `matrix.Dense` to `kinn-gg/hdbscan`'s `Dense64`, and build a kNN cosine similarity graph from CSR rows and convert it to its `SparsePrecomputed` distance graph. Lives under `examples/` with its own `go.mod` so the library keeps zero dependencies.

**Depends on:** Phase 1. **Verify:** planted-structure ARI ≥ 0.95 for all algorithms; determinism job passes (`GOMAXPROCS` 1 vs 8, sequential vs parallel trials); TRD §10 10k-row target met; README example compiles from `examples/sparse`.

## Phase 3 — CLUTO formats, CLI, golden-set validation

**Goal:** `clutro vcluster` is a drop-in for CLUTO's `vcluster` on the supported flags, and agreement with CLUTO's own results is measured.

Order of work:

1. `cluto/`: readers/writers for sparse/dense matrices (1-based columns), `.clabel`, `.rlabel`, `.rclass`, `.clustering.<k>`; round-trip tests; malformed-input tests.
2. `evaluate/` external metrics: ARI, NMI, purity, entropy.
3. `cmd/clutro`: `vcluster` with the TRD §7 flags, `-json`; `evaluate`; exit codes and stderr progress.
4. Golden-set harness (`bench/golden`): run every CLUTO corpus in a directory (the public `testdata/cluto/` subset in CI; the owner's full set privately), compute ARI and matched label agreement vs CLUTO's `.clustering.<k>` under `-colmodel none` and `idf`, print a table; inspect the worst corpora by hand; set and assert the lower band. The published subset is numeric-only and renamed (TRD §9.3).
5. Row/column preprocessing flags (`-rowmodel l2`, `-colmodel idf`) so raw count matrices can be clustered like CLUTO does.

**Depends on:** Phase 2. **Verify:** a CLUTO script with the binary name swapped produces a `.clustering.<k>` file with ARI in the accepted band; `clutro evaluate` matches scikit-learn's ARI/NMI on fixtures to 1e-9.

## Phase 4 — Explanation, internal metrics, v0.1 release

**Goal:** Every cluster can be explained and every run can be scored without truth labels; v0.1.0 tagged.

Order of work:

1. `explain/`: descriptive and discriminating features with cumulative share, representatives, summaries (ISim/ESim); column labels via option; `-showfeatures`, `-showsummaries`, `describe` command.
2. `evaluate/` internal metrics: silhouette (seeded sampling above 10k rows), Davies–Bouldin, Calinski–Harabasz; scikit-learn cross-check script in `bench/reference/`.
3. `examples/`: sparse text with explanation, dense embeddings, tree cut, CLUTO migration — all compiled in CI.
4. Documentation: algorithm reference with citations and the "differences from CLUTO" page; benchmarks page from `bench/` runs on the 10k/100k workloads.
5. Release: CHANGELOG, coverage badge, `v0.1.0` after CI passes; the first consumer switches from a local `replace` to the tagged version.

**Depends on:** Phases 2–3. **Verify:** PRD §8 v0.1 rows; a cluster of a real text corpus reads sensibly from its top-5 descriptive features in manual review.

## Phase 5 — v0.2: full criterion family and refinement fidelity

**Goal:** CLUTO's optimization vocabulary is complete: I1, E1, G1, G1′, H1, H2 with incremental deltas; per-level refinement in `rbr`; agglomerative clustering driven by the same criteria.

Order of work:

1. `criterion/`: I1, E1, G1, G1′ with `Delta`; H1, H2 as ratios with delta via numerator/denominator tracking; numeric tests per criterion; `-crfun` wired.
2. `rbr` per-level refinement option (refine after every bisection, as CLUTO does); comparison against end-only refinement on the golden set.
3. `hierarchical/`: agglomerative merging that maximizes the chosen criterion (CLUTO `agglo`), dendrogram output, `CutK`; `-fulltree` building the hierarchy on top of `rb` leaves.
4. Jaccard and Pearson similarities; `-sim` extended.
5. Benchmarks and golden-set re-run per criterion; `v0.2.0`.

**Depends on:** Phase 4. **Decision point:** whether graph-input clustering (`scluster`) is next (v0.3) depends on demand from consumers; density methods are only scheduled if a concrete user needs them.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Float32 precision degrades I2 deltas on huge composites | Accumulate in float64; precision test in Phase 1 before committing to storage width |
| Parallel code silently breaks determinism | Dedicated CI job diffs labels across `GOMAXPROCS`; no completion-order reductions by construction |
| Golden-set agreement with CLUTO is low for legitimate reasons (different RNG, tie-breaks) | Measure with ARI, not exact match; inspect worst cases; document differences rather than chase bit-parity |
| Scope creep toward a framework | PRD non-goals; any new algorithm family requires a consumer and a roadmap phase |
| Name confusion with CLUTO | README and release notes carry the non-affiliation statement; no CLUTO code or binaries ever enter the repo |
