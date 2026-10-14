# Benchmarks and Validation

Numbers below come from `bench/README.md` and are measurements of a
specific build on specific hardware (`darwin/arm64, Apple M1 Max`), not
guarantees. Re-run before quoting.

## Primitives

| Benchmark | ns/op | allocs |
|-----------|-------|--------|
| `matrix.Dot`, two 40-nnz rows | 96 | 0 |
| `matrix.Composite.Add`, 100-nnz row, 2000 columns | 204 | 0 |

## Agreement with CLUTO on 103 text corpora

A private set of 103 CLUTO runs (the `doc2mat` inputs and the
`.clustering.<k>` files `vcluster` wrote, `k = n/12 + 1`, corpora from 4 to
6,416 rows and up to 96,630 nonzeros) was re-clustered with `clutro golden`:

| Run | Wall time, all 103 | Mean ARI | Mean NMI | I2(CLUTRO)/I2(CLUTO) |
|-----|--------------------|----------|----------|----------------------|
| `rb`, `-rowmodel l2` | 6.4 s | 0.303 | 0.564 | 1.0007 |
| `rb`, `-rowmodel l2 -colmodel idf` | 6.2 s | 0.351 | 0.591 | 0.959 |
| `rbr`, `-rowmodel l2` | 15.4 s | 0.382 | 0.629 | 1.0333 |
| CLUTRO `rb` seed 2 vs seed 1 | — | 0.484 | 0.684 | 0.9999 |

- Under CLUTO's objective (`colmodel none`), CLUTRO matches (`rb`) or
  beats (`rbr`, +3.3%) the reference partitions.
- Label agreement is partial by nature of the problem; CLUTRO disagrees
  with itself across seeds at a similar order (0.48).
- The sub-1 ratio under `idf` shows the references were produced without
  column weighting — CLUTO's default.

## Published fixtures

Eight of those corpora, renamed and numeric-only, live in `testdata/cluto`
with SHA-256 sums. `golden.TestPublishedFixtures` asserts mean ARI ≥ 0.25,
min ARI ≥ 0.05, I2 ratio ≥ 0.99, and that refinement never lowers the
ratio. Observed on them: mean ARI 0.260, min 0.152, I2 ratio 1.0071.

## Reference cross-checks

`bench/reference/internal_metrics.py` reproduces with scikit-learn the
values the `evaluate` tests assert (silhouette, Davies–Bouldin,
Calinski–Harabasz, adjusted Rand index, NMI).

## Running them

```bash
go test -run '^$' -bench . -benchmem ./matrix/
clutro golden -rowmodel=l2 ./testdata/cluto
clutro golden -rowmodel=l2 -refine ./testdata/cluto
```
