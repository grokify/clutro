# Numerics and Determinism

## The determinism contract

**Same input, same options, same seed, same CLUTRO version ⇒ identical
labels, independent of `GOMAXPROCS` or `WithParallelism`.**

This is tested, not hoped for:

- every random choice derives from the caller's seed through named
  sub-streams (`rng.SubStream(seed, "kmeans", trial)`), so trials, splits
  and refinement passes never share or race for random numbers;
- parallel sweeps write to positions they own and partial sums are combined
  in worker-index order, never completion order;
- ties break to the lowest cluster index with rows visited in index order;
- a CI job runs the whole suite at `GOMAXPROCS` 1, 2 and 8, and tests
  compare labels across worker counts and against slow reference
  implementations (`partition/reference`) label for label.

Results **may** change between CLUTRO versions: an algorithmic improvement
(such as greedy k-means++ seeding) changes which optimum a seed finds. The
changelog records such changes.

## Float32 storage, float64 accumulation

Matrix values and row norms are `float32`; composites, dot products,
criterion values and deltas are `float64`. Storage is half the size of
float64 and cosine on L2-normalized rows is accurate to about 1e-7 relative,
far below the differences that separate candidate assignments in practice.

Two consequences to know about:

- a composite's norm is maintained incrementally from float32 row norms;
  `Composite.Recompute` resets accumulated rounding, and refinement calls it
  after every pass. A test bounds the drift after 20,000 adds and 10,000
  removes at under 1e-6 relative;
- feature shares from `Describe` sum to 1 within about 1e-7, and
  `Discriminate` drops shares below 1e-9 as rounding noise.

If you need float64 storage, implement `matrix.Rows` over your own data;
the algorithms never assume `CSR` or `Dense`.

## Cancellation

Every algorithm takes a `context.Context` and checks it once per outer
iteration (k-means iteration, bisection step, refinement pass, harness
corpus). Cancellation returns `ctx.Err()`; partial results are discarded.

## Non-finite input

Constructors reject NaN and ±Inf values and out-of-range indices with
`matrix.ErrShape`. Zero rows are allowed: they have similarity 0 to
everything and stay wherever seeding put them.
