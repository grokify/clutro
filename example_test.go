package clutro_test

import (
	"context"
	"fmt"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/matrix"
)

// ExampleFit clusters six short "documents" over a vocabulary of five terms
// into two groups with repeated bisection.
func ExampleFit() {
	// Rows are documents, columns are terms: {go, code, test, cook, bake}.
	m, err := matrix.NewCSRFromTriplets(6, 5, []matrix.Triplet{
		{Row: 0, Col: 0, Val: 1}, {Row: 0, Col: 1, Val: 1},
		{Row: 1, Col: 1, Val: 1}, {Row: 1, Col: 2, Val: 1},
		{Row: 2, Col: 0, Val: 1}, {Row: 2, Col: 2, Val: 1},
		{Row: 3, Col: 3, Val: 1}, {Row: 3, Col: 4, Val: 1},
		{Row: 4, Col: 3, Val: 1},
		{Row: 5, Col: 4, Val: 1}, {Row: 5, Col: 3, Val: 1},
	})
	if err != nil {
		panic(err)
	}
	m.NormalizeRowsL2()

	res, err := clutro.Fit(context.Background(), m,
		clutro.WithAlgorithm(clutro.RepeatedBisection),
		clutro.WithClusters(2),
		clutro.WithSeed(1),
	)
	if err != nil {
		panic(err)
	}
	for k, rows := range res.Clusters() {
		fmt.Println(k, rows)
	}
	// Output:
	// 0 [0 1 2]
	// 1 [3 4 5]
}
