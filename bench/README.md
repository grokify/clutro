# Benchmarks and validation runs

Numbers recorded from a specific build on specific hardware; re-run before
quoting. `goos: darwin, goarch: arm64, cpu: Apple M1 Max` throughout.

## Phase 1 — primitives

`go test -run '^$' -bench . -benchmem ./matrix/`

| Benchmark | ns/op | allocs/op |
|-----------|-------|-----------|
| `matrix.Dot` of two 40-nnz sparse rows (merge join) | 96 | 0 |
| `matrix.Composite.Add` of a 100-nnz row into a 2000-column composite | 204 | 0 |

## Phase 3 — agreement with CLUTO on 103 text corpora

A private set of 103 CLUTO runs (the inputs `doc2mat` produced and the
`.clustering.<k>` files `vcluster` wrote, with `k = n/12 + 1`) was
re-clustered with `clutro golden`. Corpora range from 4 to 6,416 rows
(up to 96,630 nonzeros, k up to 535). Eight of them are published, renamed
and numeric-only, in `testdata/cluto`.

| Run | Wall time, all 103 | Mean ARI vs CLUTO | Mean NMI | I2(CLUTRO) / I2(CLUTO) |
|-----|--------------------|-------------------|----------|------------------------|
| `rb`, `-rowmodel l2 -colmodel none` | 6.4 s | 0.303 | 0.564 | **1.0007** |
| `rb`, `-rowmodel l2 -colmodel idf` | 6.2 s | 0.351 | 0.591 | 0.959 |
| `rbr`, `-rowmodel l2 -colmodel none` | 15.4 s | 0.382 | 0.629 | **1.0333** |
| `rb` seed 2 vs `rb` seed 1 (CLUTRO against itself) | — | 0.484 | 0.684 | 0.9999 |

Reading the table:

- Under the objective CLUTO optimized (`colmodel none`), CLUTRO's partitions
  score the same as CLUTO's (`rb`) or 3% better (`rbr`, which CLUTO's
  default run did not do).
- Label agreement is partial by nature: with k ≈ n/12 there are hundreds of
  small clusters and many optima of equal quality, so even two CLUTRO seeds
  agree only at ARI ≈ 0.48. CLUTO-vs-CLUTRO agreement (0.30–0.38) is lower
  than that because the seeding and tie-breaking differ, not because either
  partition is worse.
- The I2 ratio under `idf` is below 1 because the reference partitions were
  not optimized for IDF-weighted rows; it says the references were produced
  without column weighting, i.e. CLUTO's `-colmodel none` default.

The fixture test (`golden.TestPublishedFixtures`) asserts the band derived
from this run: mean ARI ≥ 0.25, min ARI ≥ 0.05, I2 ratio ≥ 0.99, and that
refinement never lowers the ratio.
