// Command hdbscan-interop shows how CLUTRO data structures feed the
// density-based clustering in github.com/kinn-gg/hdbscan, which CLUTRO
// deliberately does not reimplement:
//
//   - a matrix.Dense becomes an hdbscan.Dense64 (float32 → float64 copy);
//   - a sparse CSR becomes a symmetric k-nearest-neighbour cosine distance
//     graph, an hdbscan.SparsePrecomputed.
//
// This example is its own module so the CLUTRO library keeps zero
// dependencies.
package main

import (
	"context"
	"fmt"
	"sort"

	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
	"github.com/kinn-gg/hdbscan"
)

func main() {
	ctx := context.Background()

	// Dense: three blobs in 2-D.
	var data []float32
	for c := 0; c < 3; c++ {
		for i := 0; i < 30; i++ {
			data = append(data, float32(c*10)+float32(i%5)*0.1, float32(c*10)+float32(i%7)*0.1)
		}
	}
	dense, err := matrix.NewDense(90, 2, data)
	if err != nil {
		panic(err)
	}
	labels, err := hdbscan.FitPredict(ctx, ToDense64(dense), hdbscan.Config{MinClusterSize: 5})
	if err != nil {
		panic(err)
	}
	fmt.Println("dense blobs → hdbscan clusters:", countLabels(labels))

	// Sparse: build a 5-NN cosine distance graph and cluster it.
	csr, err := matrix.NewCSRFromTriplets(90, 6, blockTriplets())
	if err != nil {
		panic(err)
	}
	csr.NormalizeRowsL2()
	graph := KNNGraph(csr, 5)
	res, err := hdbscan.ReferenceSparse(ctx, graph, hdbscan.Config{MinClusterSize: 5})
	if err != nil {
		panic(err)
	}
	fmt.Println("sparse kNN graph → hdbscan clusters:", countLabels(res.Labels))
}

// ToDense64 copies a CLUTRO dense matrix into hdbscan's row-major float64
// form.
func ToDense64(m *matrix.Dense) hdbscan.Dense64 {
	out := hdbscan.Dense64{Rows: m.NumRows(), Cols: m.NumCols(), Data: make([]float64, 0, m.NumRows()*m.NumCols())}
	for i := 0; i < m.NumRows(); i++ {
		for _, v := range m.Row(i).Val {
			out.Data = append(out.Data, float64(v))
		}
	}
	return out
}

// KNNGraph builds a symmetric k-nearest-neighbour graph of cosine distances
// (1 − cos) over the rows of a CSR, in the CSR form hdbscan expects: sorted
// unique column indices per row and every edge present in both directions.
// HDBSCAN needs a connected graph, so after the kNN edges are added the
// components are joined by their shortest inter-component edges (a minimum
// spanning connector), as kNN graphs of well-separated groups are otherwise
// disconnected.
func KNNGraph(m *matrix.CSR, k int) hdbscan.SparsePrecomputed {
	n := m.NumRows()
	cos := metric.Cosine{}
	dist := func(i, j int) float64 {
		d := 1 - cos.Sim(m.Row(i), m.Row(j), m.RowNorm(i), m.RowNorm(j))
		if d < 0 {
			return 0
		}
		return d
	}
	adj := make([]map[int]float64, n)
	for i := range adj {
		adj[i] = map[int]float64{}
	}
	type cand struct {
		j int
		d float64
	}
	for i := 0; i < n; i++ {
		cands := make([]cand, 0, n-1)
		for j := 0; j < n; j++ {
			if j != i {
				cands = append(cands, cand{j, dist(i, j)})
			}
		}
		sort.Slice(cands, func(a, b int) bool {
			if cands[a].d != cands[b].d {
				return cands[a].d < cands[b].d
			}
			return cands[a].j < cands[b].j
		})
		for _, c := range cands[:min(k, len(cands))] {
			adj[i][c.j] = c.d
			adj[c.j][i] = c.d // symmetrize
		}
	}

	// Connect components: union-find over kNN edges, then repeatedly add the
	// shortest edge between two different components.
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	for i := range adj {
		for j := range adj[i] {
			parent[find(i)] = find(j)
		}
	}
	for {
		bi, bj, bd := -1, -1, 0.0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if find(i) == find(j) {
					continue
				}
				if d := dist(i, j); bi < 0 || d < bd {
					bi, bj, bd = i, j, d
				}
			}
		}
		if bi < 0 {
			break // one component
		}
		adj[bi][bj], adj[bj][bi] = bd, bd
		parent[find(bi)] = find(bj)
	}

	g := hdbscan.SparsePrecomputed{Rows: n, Cols: n, IndPtr: make([]int, n+1)}
	for i := 0; i < n; i++ {
		cols := make([]int, 0, len(adj[i]))
		for j := range adj[i] {
			cols = append(cols, j)
		}
		sort.Ints(cols)
		for _, j := range cols {
			g.Indices = append(g.Indices, j)
			g.Data = append(g.Data, adj[i][j])
		}
		g.IndPtr[i+1] = len(g.Indices)
	}
	return g
}

// blockTriplets makes three groups of 30 rows, each group living on its own
// pair of columns with varied weights so rows within a group are similar
// but not identical.
func blockTriplets() []matrix.Triplet {
	var ts []matrix.Triplet
	for i := 0; i < 90; i++ {
		block := i / 30
		ts = append(ts,
			matrix.Triplet{Row: i, Col: 2 * block, Val: float32(1 + i%7)},
			matrix.Triplet{Row: i, Col: 2*block + 1, Val: float32(1 + i%5)},
		)
	}
	return ts
}

func countLabels(labels []int) map[int]int {
	out := map[int]int{}
	for _, l := range labels {
		out[l]++
	}
	return out
}
