# Explaining Clusters

"Cluster 7 has 41 documents" is not an answer. The `explain` package — and
the matching `Result` methods — say what a cluster is about, what sets it
apart, which rows represent it, and how tight it is. Pass
`WithColumnLabels(vocab)` to `Fit` and the features come back as words.

## Descriptive features

```go
for _, f := range res.Describe(k, 5) {
    fmt.Printf("%s %.0f%%\n", f.Label, 100*f.Weight)
}
```

The columns that contribute most to the cluster's **internal similarity**:
each column's share is `D_k[j]² / ‖D_k‖²`, where `D_k` is the composite.
Shares sum to 1 over all columns; `Describe` returns the top n. These are
the features the cluster's rows have in common, including ones shared with
every other cluster.

## Discriminating features

```go
res.Discriminate(k, 5)
```

The columns that most **separate** the cluster from the rest of the rows:
the contribution of each column to the gap between the cluster's internal
similarity and its similarity to everything else, normalized to shares.
Features that make the cluster *more* like the rest (such as a stopword
present everywhere) are dropped, so the list says what is distinctive.

A cluster described by `permiss access list token acl` and discriminated by
`acl token permiss` is about access control; a cluster described by
`the of and` has not found anything, and its rows probably need better
preprocessing (IDF weighting, stopword removal).

## Representatives

```go
rows, sims := res.Representatives(k, 3)
```

The rows most similar to the cluster composite, best first — useful for
labelling a cluster by its most central document.

## Summaries

```go
for _, s := range res.Summaries() {
    fmt.Println(s.Cluster, s.Size, s.ISim, s.ISimDev, s.ESim, s.ESimDev)
}
```

For each cluster: size; **ISim**, the mean similarity of its rows to its own
composite (tightness); **ESim**, the mean similarity of its rows to the
composite of all other rows (separation); and their standard deviations.
A good cluster has ISim well above ESim. These are CLUTO's
`-showsummaries` figures.

## From the CLI

`clutro vcluster -showtree` prints the bisection tree with each cluster's
criterion contribution; per-cluster features from the CLI are planned
alongside the `describe` command.
