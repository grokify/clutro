// Command treecut shows how repeated bisection's hierarchy lets you choose
// the number of clusters after the fact: one fit, several cuts.
package main

import (
	"context"
	"fmt"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/matrix"
)

func main() {
	// 60 rows in 5 planted groups of 12 over 50 columns (10 per group).
	var ts []matrix.Triplet
	for i := 0; i < 60; i++ {
		g := i / 12
		for j := 0; j < 4; j++ {
			ts = append(ts, matrix.Triplet{Row: i, Col: g*10 + (i+j)%10, Val: 1})
		}
	}
	m, err := matrix.NewCSRFromTriplets(60, 50, ts)
	if err != nil {
		panic(err)
	}
	m.NormalizeRowsL2()

	// Bisect further than needed; the tree remembers every split.
	res, err := clutro.Fit(context.Background(), m,
		clutro.WithAlgorithm(clutro.RepeatedBisection),
		clutro.WithClusters(10),
		clutro.WithSeed(1),
	)
	if err != nil {
		panic(err)
	}
	tree := res.Tree()
	fmt.Printf("fitted k=%d, sizes %v\n", res.K, res.Sizes())
	for _, maxSize := range []int{30, 15, 12} {
		cut := tree.CutMaxSize(maxSize)
		fmt.Printf("cut to <= %2d rows per cluster: k=%d sizes %v\n", maxSize, cut.K, cut.Sizes())
	}
	five := tree.CutK(5)
	fmt.Printf("cut to k=5: sizes %v\n", five.Sizes())
}
