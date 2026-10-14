// Package criterion defines the global objective functions a partition is
// optimized for, with incremental deltas for moving one row, in the
// terminology of CLUTO and Zhao & Karypis (Machine Learning 55, 2004).
package criterion

import (
	"github.com/grokify/clutro/matrix"
)

// Criterion scores a whole partition from its cluster composite vectors and
// prices a single-row move, so refinement can run in O(nnz(row)) per
// candidate instead of recomputing the objective.
type Criterion interface {
	// Name is the CLUTO-style identifier ("i2").
	Name() string
	// Maximize reports whether larger values are better.
	Maximize() bool
	// Value is the objective of the partition described by comps. Empty
	// composites contribute nothing.
	Value(comps []*matrix.Composite) float64
	// Delta is the change in Value if row r (with norm normR) leaves `from`
	// and joins `to`. Either may be nil for a row entering or leaving the
	// partition. A positive delta is an improvement when Maximize is true.
	Delta(from, to *matrix.Composite, r matrix.SparseVec, normR float32) float64
}

// I2 is Σ_k ‖D_k‖: the sum over clusters of the composite vector's L2 norm.
// For L2-normalized rows under cosine similarity it equals Σ_k √(Σ_{i,j∈S_k}
// cos(d_i, d_j)), CLUTO's default and the criterion behind spherical
// k-means. Maximized.
type I2 struct{}

// Name implements Criterion.
func (I2) Name() string { return "i2" }

// Maximize implements Criterion.
func (I2) Maximize() bool { return true }

// Value implements Criterion.
func (I2) Value(comps []*matrix.Composite) float64 {
	var s float64
	for _, c := range comps {
		if c != nil && c.N > 0 {
			s += c.Norm()
		}
	}
	return s
}

// Delta implements Criterion: (‖D_to + r‖ − ‖D_to‖) + (‖D_from − r‖ − ‖D_from‖).
func (I2) Delta(from, to *matrix.Composite, r matrix.SparseVec, normR float32) float64 {
	var d float64
	if to != nil {
		d += to.NormIfAdded(r, normR) - to.Norm()
	}
	if from != nil {
		d += from.NormIfRemoved(r, normR) - from.Norm()
	}
	return d
}

// Parse returns the criterion for a CLUTO-style name.
func Parse(name string) (Criterion, bool) {
	switch name {
	case "i2", "I2":
		return I2{}, true
	}
	return nil, false
}

// Better reports whether value a is an improvement over b under c.
func Better(c Criterion, a, b float64) bool {
	if c.Maximize() {
		return a > b
	}
	return a < b
}
