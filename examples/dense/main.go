// Command dense clusters dense vectors — the shape of embeddings from any
// provider — with spherical k-means, and scores the result with internal
// metrics that need no ground truth.
package main

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/evaluate"
	"github.com/grokify/clutro/matrix"
)

func main() {
	const rows, dims, k = 300, 64, 4
	// Stand-in for embeddings: k Gaussian blobs.
	r := rand.New(rand.NewPCG(7, 7)) //nolint:gosec // synthetic data
	centers := make([][]float32, k)
	for c := range centers {
		centers[c] = make([]float32, dims)
		for d := range centers[c] {
			centers[c][d] = float32(r.NormFloat64())
		}
	}
	data := make([]float32, 0, rows*dims)
	for i := 0; i < rows; i++ {
		c := centers[i%k]
		for d := 0; d < dims; d++ {
			data = append(data, c[d]+float32(r.NormFloat64())*0.3)
		}
	}
	m, err := matrix.NewDense(rows, dims, data)
	if err != nil {
		panic(err)
	}
	m.NormalizeRowsL2() // cosine geometry, as embedding providers recommend

	res, err := clutro.Fit(context.Background(), m,
		clutro.WithAlgorithm(clutro.KMeans), // spherical k-means under cosine
		clutro.WithClusters(k),
		clutro.WithSeed(1),
	)
	if err != nil {
		panic(err)
	}
	sil, err := evaluate.Silhouette(m, res.Labels, res.K, evaluate.SilhouetteOptions{})
	if err != nil {
		panic(err)
	}
	db, _ := evaluate.DaviesBouldin(m, res.Labels, res.K)
	ch, _ := evaluate.CalinskiHarabasz(m, res.Labels, res.K)
	fmt.Printf("sizes %v\nsilhouette %.3f  davies-bouldin %.3f  calinski-harabasz %.1f\n", res.Sizes(), sil, db, ch)
}
