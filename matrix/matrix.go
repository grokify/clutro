// Package matrix holds the row-oriented data structures every CLUTRO
// algorithm reads: sparse CSR and dense matrices behind one Rows interface,
// sparse vectors, dot products, and composite (cluster sum) vectors with
// incremental updates.
//
// Values are stored as float32 and accumulated in float64, which halves the
// memory of large sparse matrices while keeping cosine similarities on
// L2-normalized rows accurate to well beyond single precision in practice.
package matrix

import (
	"errors"
	"fmt"
	"math"
)

// SparseVec is a row with sorted, unique column indices. Idx and Val have the
// same length. A SparseVec returned by a matrix is a view; do not modify it.
type SparseVec struct {
	Idx []uint32
	Val []float32
}

// NNZ is the number of stored entries.
func (v SparseVec) NNZ() int { return len(v.Idx) }

// Norm is the L2 norm of the vector.
func (v SparseVec) Norm() float64 {
	var s float64
	for _, x := range v.Val {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

// Rows is what algorithms read: a fixed-shape matrix exposing one row at a
// time with its cached L2 norm. CSR and Dense implement it; so can a
// caller's own store.
type Rows interface {
	NumRows() int
	NumCols() int
	Row(i int) SparseVec
	RowNorm(i int) float32
}

// Dot is the inner product of two sparse vectors by merge join over their
// sorted indices.
func Dot(a, b SparseVec) float64 {
	var s float64
	i, j := 0, 0
	for i < len(a.Idx) && j < len(b.Idx) {
		switch {
		case a.Idx[i] == b.Idx[j]:
			s += float64(a.Val[i]) * float64(b.Val[j])
			i++
			j++
		case a.Idx[i] < b.Idx[j]:
			i++
		default:
			j++
		}
	}
	return s
}

// DotDense is the inner product of a sparse vector with a dense float64
// vector indexed by column.
func DotDense(a SparseVec, d []float64) float64 {
	var s float64
	for k, idx := range a.Idx {
		s += float64(a.Val[k]) * d[idx]
	}
	return s
}

// ErrShape reports a matrix construction problem.
var ErrShape = errors.New("matrix: invalid shape")

func checkRow(nCols int, r SparseVec) error {
	if len(r.Idx) != len(r.Val) {
		return fmt.Errorf("%w: idx/val length mismatch %d/%d", ErrShape, len(r.Idx), len(r.Val))
	}
	for k, idx := range r.Idx {
		if int(idx) >= nCols {
			return fmt.Errorf("%w: column %d out of range (cols=%d)", ErrShape, idx, nCols)
		}
		if k > 0 && idx <= r.Idx[k-1] {
			return fmt.Errorf("%w: column indices must be sorted and unique (%d after %d)", ErrShape, idx, r.Idx[k-1])
		}
		if v := r.Val[k]; math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%w: non-finite value at column %d", ErrShape, idx)
		}
	}
	return nil
}
