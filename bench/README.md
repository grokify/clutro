# Benchmarks

Numbers recorded from `go test -run '^$' -bench . -benchmem ./...` on the
machine noted. They are measurements of a specific build on specific
hardware, not performance claims; re-run before quoting.

## Phase 1 — primitives

`goos: darwin, goarch: arm64, cpu: Apple M1 Max`

| Benchmark | ns/op | allocs/op |
|-----------|-------|-----------|
| `matrix.Dot` of two 40-nnz sparse rows (merge join) | 96 | 0 |
| `matrix.Composite.Add` of a 100-nnz row into a 2000-column composite | 204 | 0 |

At ~100 ns per 40-nnz dot product, a 2-way bisection step over 10k rows
costs on the order of 2 ms per iteration before parallelism.
