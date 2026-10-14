// Package synth generates synthetic corpora with planted cluster structure
// for tests and benchmarks.
package synth

import (
	"github.com/grokify/clutro/internal/rng"
	"github.com/grokify/clutro/matrix"
)

// Params describes a planted-structure sparse corpus: k clusters, each with
// its own block of "topic" columns; every row draws most of its nonzeros
// from its cluster's block and a few from a shared background block.
type Params struct {
	Rows, Cols, K int
	// NNZ is the number of nonzeros per row.
	NNZ int
	// Noise is the fraction of a row's nonzeros drawn from the background
	// columns instead of the cluster's block (0..1).
	Noise float64
	Seed  uint64
}

// Sparse returns an L2-normalized CSR and the planted label of every row
// (rows are assigned to clusters round-robin, then shuffled).
func Sparse(p Params) (*matrix.CSR, []int32) {
	r := rng.SubStream(p.Seed, "synth", 0)
	if p.K < 1 {
		p.K = 1
	}
	block := p.Cols / (p.K + 1) // last block is background
	if block < 1 {
		block = 1
	}
	labels := make([]int32, p.Rows)
	for i := range labels {
		labels[i] = int32(i % p.K)
	}
	r.Shuffle(len(labels), func(i, j int) { labels[i], labels[j] = labels[j], labels[i] })

	rows := make([]matrix.SparseVec, p.Rows)
	for i := range rows {
		cols := map[uint32]float32{}
		for len(cols) < p.NNZ {
			var c int
			if r.Float64() < p.Noise {
				c = p.K*block + r.IntN(max(p.Cols-p.K*block, 1))
			} else {
				c = int(labels[i])*block + r.IntN(block)
			}
			if c >= p.Cols {
				c = p.Cols - 1
			}
			cols[uint32(c)] = float32(1 + r.IntN(3))
		}
		rows[i] = fromMap(cols)
	}
	m, err := matrix.NewCSRFromRows(p.Cols, rows)
	if err != nil {
		panic(err) // generator bug
	}
	m.NormalizeRowsL2()
	return m, labels
}

// Dense returns a dense matrix of k Gaussian blobs in dims dimensions and
// the planted labels.
func Dense(rows, dims, k int, spread float64, seed uint64) (*matrix.Dense, []int32) {
	r := rng.SubStream(seed, "synth-dense", 0)
	centers := make([][]float64, k)
	for c := range centers {
		centers[c] = make([]float64, dims)
		for d := range centers[c] {
			centers[c][d] = r.NormFloat64() * 10
		}
	}
	labels := make([]int32, rows)
	data := make([]float32, 0, rows*dims)
	for i := 0; i < rows; i++ {
		c := i % k
		labels[i] = int32(c)
		for d := 0; d < dims; d++ {
			data = append(data, float32(centers[c][d]+r.NormFloat64()*spread))
		}
	}
	m, err := matrix.NewDense(rows, dims, data)
	if err != nil {
		panic(err)
	}
	return m, labels
}

func fromMap(cols map[uint32]float32) matrix.SparseVec {
	idx := make([]uint32, 0, len(cols))
	for c := range cols {
		idx = append(idx, c)
	}
	// insertion sort: rows are short
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0 && idx[j] < idx[j-1]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
	val := make([]float32, len(idx))
	for k, c := range idx {
		val[k] = cols[c]
	}
	return matrix.SparseVec{Idx: idx, Val: val}
}

// Shuffled returns a seeded permutation of the rows of a labelled corpus,
// for tests that assert input-order independence.
func Shuffled(m matrix.Rows, labels []int32, seed uint64) (*matrix.CSR, []int32) {
	r := rng.SubStream(seed, "shuffle", 0)
	perm := rng.Perm(r, m.NumRows())
	rows := make([]matrix.SparseVec, m.NumRows())
	out := make([]int32, m.NumRows())
	for i, p := range perm {
		rows[i] = m.Row(p)
		out[i] = labels[p]
	}
	csr, err := matrix.NewCSRFromRows(m.NumCols(), rows)
	if err != nil {
		panic(err)
	}
	return csr, out
}
