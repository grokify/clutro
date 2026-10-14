# Concepts

## Rows, not points

Every algorithm reads data through one small interface:

```go
type Rows interface {
    NumRows() int
    NumCols() int
    Row(i int) SparseVec   // sorted column indices + float32 values
    RowNorm(i int) float32 // cached L2 norm
}
```

`matrix.CSR` is the sparse implementation (compressed sparse row, float32
values, uint32 columns); `matrix.Dense` is row-major dense storage whose
rows are exposed as sparse views sharing one index slice. A caller with a
memory-mapped or columnar store can implement `Rows` directly.

Values are stored as **float32 and accumulated in float64**. Cosine
similarities on normalized rows stay accurate to well beyond single
precision in practice; see [Numerics](numerics.md).

## Similarity

Algorithms are written against `metric.Similarity`. `Cosine` is the default
and is a plain dot product on L2-normalized rows. `Euclidean` is exposed as a
similarity 1/(1+d) so k-means can use it, with the true distance available
too. Normalize rows (`NormalizeRowsL2`, or `-rowmodel l2`) before using
cosine; CLUTO does this implicitly.

## Composite vectors

A cluster's **composite** is the sum of its rows, kept dense in float64
with its norm maintained incrementally. Adding or removing a row costs
O(nnz(row)), not O(columns), which is what makes refinement cheap. The
composite's direction is the cluster centroid's direction, so cosine to the
composite is cosine to the centroid.

## Criterion functions

A **criterion** scores a whole partition from its composites. CLUTRO
optimizes **I2 = Σ_k ‖D_k‖**, the sum of composite norms — the objective
behind spherical k-means and CLUTO's default. Every criterion also prices a
single-row move (`Delta`) so refinement can evaluate a candidate move in
O(nnz(row)). The remaining CLUTO criteria (I1, E1, G1, G1′, H1, H2) are
Phase 5 of the roadmap.

## Repeated bisection and the tree

Rather than placing k centers at once, **repeated bisection** starts with one
cluster and splits one cluster at a time: pick a cluster (the one whose split
improves the criterion most, or the largest), run a 2-way spherical k-means
from several seeded starts, keep the best split, repeat until k clusters
exist. Each split is recorded in a **tree**, so the result is both a flat
partition and a hierarchy you can re-cut (`CutMaxSize`, `CutK`) without
re-clustering. Bisection is slower than direct k-means per cluster but far
more robust on text, where direct k-means finds poor local optima.

## Refinement

After bisection (or a direct k-way split), **refinement** visits rows in a
seeded random order and moves each to the cluster that improves the
criterion most, applying moves immediately, until a pass makes no move.
This is CLUTO's `rbr` and the inner loop of `direct`.

## Result

`Fit` returns a `Result`: labels, sizes, clusters, criterion value, the
composites, the tree (for bisection), and — through the `explain` package —
descriptions, discriminators, representatives and summaries that use the
column labels you passed in.
