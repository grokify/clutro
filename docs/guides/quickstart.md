# Quick Start

## 1. Build a matrix

Rows are the things you want to cluster; columns are features. For text,
rows are documents and columns are terms with TF-IDF weights; for
embeddings, rows are vectors.

```go
// Sparse, from (row, col, value) entries in any order.
m, err := matrix.NewCSRFromTriplets(nDocs, nTerms, triplets)
if err != nil {
    return err
}
// CLUTO-style preprocessing: IDF column weights, then unit-length rows.
if err := matrix.Preprocess(m, matrix.RowL2, matrix.ColIDF); err != nil {
    return err
}

// Dense, from row-major values.
d, err := matrix.NewDense(nRows, dims, values)
d.NormalizeRowsL2()
```

Both types implement `matrix.Rows`, which is all the algorithms need — so
can your own store.

## 2. Fit

```go
res, err := clutro.Fit(ctx, m,
    clutro.WithAlgorithm(clutro.RepeatedBisection), // the default
    clutro.WithClusters(20),
    clutro.WithRefinement(true),                    // rbr: refine after bisecting
    clutro.WithSeed(1),
    clutro.WithColumnLabels(vocab),                 // optional, for explanations
)
```

Defaults match CLUTO's: cosine similarity, the I2 criterion, 10 seeded
trials per split, the `best` selection policy. `res.Labels` holds a cluster
index per row; `res.Clusters()` the rows per cluster; `res.Criterion` the
objective value.

## 3. Explain

```go
for k := range res.K {
    fmt.Println(res.Sizes()[k], "rows")
    for _, f := range res.Describe(k, 5) {     // what the cluster is about
        fmt.Printf("  %s %.0f%%\n", f.Label, 100*f.Weight)
    }
    for _, f := range res.Discriminate(k, 3) { // what sets it apart
        fmt.Printf("  vs rest: %s\n", f.Label)
    }
    rows, sims := res.Representatives(k, 3)    // most central rows
    _ = rows; _ = sims
}
for _, s := range res.Summaries() {            // tightness and separation
    fmt.Println(s.Cluster, s.Size, s.ISim, s.ESim)
}
```

## 4. Choose k afterwards

Repeated bisection records every split, so you can re-cut without
re-clustering:

```go
tree := res.Tree()
byTwelve := tree.CutMaxSize(12) // clusters of about 12 rows or fewer
five := tree.CutK(5)
```

## 5. Evaluate

With ground truth:

```go
ari, _ := evaluate.AdjustedRandIndex(res.Labels, truth)
nmi, _ := evaluate.NormalizedMutualInformation(res.Labels, truth)
```

Without:

```go
sil, _ := evaluate.Silhouette(m, res.Labels, res.K, evaluate.SilhouetteOptions{})
db, _  := evaluate.DaviesBouldin(m, res.Labels, res.K)
ch, _  := evaluate.CalinskiHarabasz(m, res.Labels, res.K)
```

## 6. From the command line

```bash
clutro vcluster -clmethod=rbr -rowmodel=l2 -colmodel=idf docs.mat 20
clutro evaluate docs.mat.clustering.20 docs.rclass
clutro golden ./corpora     # compare against reference clusterings
```

The `vcluster` flags are CLUTO's; see [CLUTO Compatibility](cluto-compatibility.md).
