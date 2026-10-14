# CLI Reference

```bash
go install github.com/grokify/clutro/cmd/clutro@latest
```

Flags are parsed with Go's standard library, so CLUTO-style `-flag=value`
and `-flag value` work, as does `--flag value`.

## `vcluster [flags] <matrix> <k>`

Reads a CLUTO sparse or dense matrix file, clusters its rows into k
clusters and writes `<matrix>.clustering.<k>` (one 0-based cluster id per
row). Prints the shape, the criterion value and the cluster sizes.

| Flag | Default | Values |
|------|---------|--------|
| `-clmethod` | `rb` | `rb`, `rbr`, `direct`, `kmeans` |
| `-crfun` | `i2` | `i2` |
| `-sim` | `cos` | `cos`, `euclid` |
| `-cstype` | `best` | `best`, `largest` — which cluster to bisect next |
| `-rowmodel` | `none` | `none`, `l2` — row scaling before clustering |
| `-colmodel` | `none` | `none`, `idf` — column weighting (sparse only) |
| `-ntrials` | 10 | seeded starts per k-means or bisection |
| `-niter` | 20 | max k-means iterations and refinement passes |
| `-seed` | 0 | seed for every random choice |
| `-parallel` | 0 | worker count; 0 = all CPUs; results do not depend on it |
| `-clustfile` | `<matrix>.clustering.<k>` | output path |
| `-json` | off | print the result as JSON (shape, method, criterion, sizes, labels) instead of writing a file |
| `-showtree` | off | print the bisection tree (`rb`/`rbr`) |

## `evaluate [flags] <clustering> <truth>`

Prints ARI, NMI, purity and entropy of a `.clustering` file against `truth`:
another clustering file, or an `.rclass` file of one class name per line
(`-rclass`, implied by the extension).

## `golden [flags] <dir>`

For every `<name>.clustering.<k>` in `dir` whose matrix `<name>` exists,
re-clusters the matrix with repeated bisection into k clusters and prints a
table of ARI, NMI and the I2 of the reference and CLUTRO partitions, with
means. Corpora whose reference does not fit the matrix are skipped and
listed on stderr.

| Flag | Default |
|------|---------|
| `-seed` | 0 |
| `-ntrials` | 10 |
| `-parallel` | 0 |
| `-refine` | off |
| `-rowmodel` | `l2` |
| `-colmodel` | `none` |

## Exit codes

0 on success; 1 on bad arguments, unreadable or malformed files, or an
error during clustering, with the reason on stderr.
