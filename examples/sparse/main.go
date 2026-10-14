// Command sparse clusters a small bag-of-words corpus with repeated
// bisection and explains each cluster by its top terms. It is the README
// example; it compiles in CI.
package main

import (
	"context"
	"fmt"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/matrix"
)

func main() {
	// Rows are documents, columns are terms.
	vocab := []string{"go", "compile", "goroutine", "flour", "oven", "yeast", "the"}
	docs := [][]int{
		{0, 1, 6}, {0, 2, 6}, {1, 2, 6}, // programming
		{3, 4, 6}, {3, 5, 6}, {4, 5, 6}, // baking
	}
	var ts []matrix.Triplet
	for r, cols := range docs {
		for _, c := range cols {
			ts = append(ts, matrix.Triplet{Row: r, Col: c, Val: 1})
		}
	}
	m, err := matrix.NewCSRFromTriplets(len(docs), len(vocab), ts)
	if err != nil {
		panic(err)
	}
	// Down-weight "the", which appears everywhere, then normalize rows.
	if err := matrix.Preprocess(m, matrix.RowL2, matrix.ColIDF); err != nil {
		panic(err)
	}

	res, err := clutro.Fit(context.Background(), m,
		clutro.WithAlgorithm(clutro.RepeatedBisection),
		clutro.WithClusters(2),
		clutro.WithRefinement(true),
		clutro.WithSeed(1),
		clutro.WithColumnLabels(vocab),
	)
	if err != nil {
		panic(err)
	}
	for k, rows := range res.Clusters() {
		fmt.Printf("cluster %d: docs %v\n", k, rows)
		for _, f := range res.Describe(k, 3) {
			fmt.Printf("  %-10s %.0f%%\n", f.Label, 100*f.Weight)
		}
	}
	for _, s := range res.Summaries() {
		fmt.Printf("cluster %d: size %d, internal sim %.2f, external sim %.2f\n", s.Cluster, s.Size, s.ISim, s.ESim)
	}
}
