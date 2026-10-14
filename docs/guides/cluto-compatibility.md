# CLUTO Compatibility

CLUTRO aims for **algorithmic** compatibility with CLUTO — the same
objectives, the same algorithms, comparable results — and for file and
command-line compatibility where that lets existing work carry over. It does
not aim for bit-identical output: CLUTO's random number stream cannot be
reproduced, and two correct implementations of a stochastic optimizer find
different optima of equal quality.

## File formats (`cluto` package)

| File | Format | CLUTRO |
|------|--------|--------|
| sparse matrix | `nrows ncols nnz`, then `col val` pairs per row, 1-based columns, any order within a row, blank line = empty row | `cluto.ReadSparse`, `cluto.WriteSparse` |
| dense matrix | `nrows ncols`, then one row of values per line | `cluto.ReadMatrix`, `cluto.WriteDense` |
| `.clabel`, `.rlabel` | one column / row label per line | `cluto.ReadLabels`, `cluto.WriteLabels` |
| `.rclass` | one class per row | `cluto.ReadLabels` + `cluto.ClassLabels` |
| `.clustering.<k>` | one 0-based id per row, `-1` unassigned | `cluto.ReadClustering`, `cluto.WriteClustering` |

## `vcluster` flags

| CLUTO | CLUTRO | Notes |
|-------|--------|-------|
| `-clmethod=rb\|rbr\|direct` | same | `kmeans` added |
| `-crfun=i2` | same | I1, E1, G1, G1′, H1, H2 are roadmap v0.2 |
| `-sim=cos` | same | `euclid` added |
| `-cstype=best\|largest` | same | feature-subspace selection not implemented |
| `-rowmodel=none` | same; `l2` | CLUTO normalizes rows internally for cosine; pass `-rowmodel l2` on raw counts |
| `-colmodel=none\|idf` | same | |
| `-ntrials=10`, `-niter=20` | same defaults | |
| `-seed` | same | different stream |
| `-clustfile` | same | |
| `-showfeatures`, `-showsummaries` | library (`Describe`, `Summaries`); CLI planned | |
| `-fulltree`, `agglo` | roadmap v0.2 | |
| `scluster` (graph input) | roadmap v0.3 | |

## How close are the results?

On 103 real text corpora clustered by CLUTO with its defaults (`rb`, `i2`,
`cos`, `k = n/12 + 1`), re-clustering with CLUTRO gives:

| | Mean ARI vs CLUTO | I2(CLUTRO)/I2(CLUTO) |
|---|---|---|
| `rb` | 0.30 | 1.0007 |
| `rbr` | 0.38 | 1.033 |
| CLUTRO `rb` seed 2 vs seed 1 | 0.48 | 0.9999 |

The partitions differ in detail — with k ≈ n/12 there are hundreds of small
clusters and many optima of equal quality, which is why even two CLUTRO
seeds agree only at ARI 0.48 — but under the objective CLUTO optimized,
CLUTRO's partitions are as good or better. Eight of these corpora are
published as numeric fixtures in `testdata/cluto` and the band is asserted
by `go test`. See [Benchmarks and Validation](benchmarks.md).

## Known differences

- **Seeding.** CLUTRO uses greedy k-means++; CLUTO samples initial centroids
  differently. Both run multiple trials and keep the best criterion.
- **Refinement scope.** CLUTRO's `rbr` refines the final k-way partition;
  CLUTO also refines after each bisection. Per-level refinement is a v0.2
  option.
- **Determinism.** CLUTRO's results are independent of thread count by
  design; see [Numerics and Determinism](numerics.md).
- **Not reimplemented:** gCLUTO, `scluster`, the agglomerative methods,
  feature-subspace cluster selection. Density methods are out of scope;
  `examples/hdbscan` shows how to hand CLUTRO data to a density clusterer.

## Legal

CLUTO is copyrighted by the Regents of the University of Minnesota and was
distributed for research use. CLUTRO is an independent implementation from
the published papers and manual; no CLUTO source or binaries are used, and
CLUTRO is not affiliated with or endorsed by the Karypis lab or the
University of Minnesota.
