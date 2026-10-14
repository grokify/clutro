// Package partition implements criterion-driven partitional clustering:
// k-means and spherical k-means, repeated bisection, refinement, and direct
// k-way clustering, all deterministic for a given seed and independent of
// the number of workers.
package partition

import (
	"errors"
	"fmt"

	"github.com/grokify/clutro/criterion"
	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
)

// Selection chooses which cluster repeated bisection splits next.
type Selection int

const (
	// SelectBest splits the cluster whose bisection improves the criterion
	// most (CLUTO -cstype=best). Bisections are computed lazily and cached.
	SelectBest Selection = iota
	// SelectLargest splits the cluster with the most rows.
	SelectLargest
)

// Options configures every algorithm in the package. Zero values take the
// CLUTO-matching defaults listed on each field.
type Options struct {
	// Similarity defaults to metric.Cosine.
	Similarity metric.Similarity
	// Criterion defaults to criterion.I2.
	Criterion criterion.Criterion
	// Trials is the number of seeded starts per k-means or bisection; the
	// best by criterion wins. Default 10.
	Trials int
	// MaxIter bounds k-means iterations and refinement passes. Default 20.
	MaxIter int
	// Seed drives every random choice. Default 0.
	Seed uint64
	// Workers bounds parallelism; 0 means runtime.NumCPU(). Results do not
	// depend on it.
	Workers int
	// Selection is the bisection policy. Default SelectBest.
	Selection Selection
	// Refine runs criterion-driven refinement after bisection (CLUTO rbr).
	Refine bool
}

func (o Options) withDefaults() Options {
	if o.Similarity == nil {
		o.Similarity = metric.Cosine{}
	}
	if o.Criterion == nil {
		o.Criterion = criterion.I2{}
	}
	if o.Trials <= 0 {
		o.Trials = 10
	}
	if o.MaxIter <= 0 {
		o.MaxIter = 20
	}
	return o
}

// Assignment is a k-way partition of the rows.
type Assignment struct {
	Labels []int32 // cluster index per row, 0..K-1
	K      int
	// Criterion is the objective value of the partition.
	Criterion float64
}

// Sizes returns the number of rows per cluster.
func (a Assignment) Sizes() []int {
	s := make([]int, a.K)
	for _, l := range a.Labels {
		s[l]++
	}
	return s
}

// ErrInput reports invalid input to an algorithm.
var ErrInput = errors.New("partition: invalid input")

func check(m matrix.Rows, k int) error {
	if m == nil || m.NumRows() == 0 {
		return fmt.Errorf("%w: empty matrix", ErrInput)
	}
	if k < 1 || k > m.NumRows() {
		return fmt.Errorf("%w: k=%d for %d rows", ErrInput, k, m.NumRows())
	}
	return nil
}

// composites builds one composite per cluster from labels.
func composites(m matrix.Rows, labels []int32, k int) []*matrix.Composite {
	cs := make([]*matrix.Composite, k)
	for i := range cs {
		cs[i] = matrix.NewComposite(m.NumCols())
	}
	for i, l := range labels {
		cs[l].Add(m.Row(i), m.RowNorm(i))
	}
	return cs
}
