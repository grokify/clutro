// Package reference holds slow, obviously-correct implementations of the
// partition algorithms, used as oracles in tests: no composites, no
// incremental deltas, no parallelism. Dense centroids are recomputed from
// scratch at every step.
package reference

import (
	"context"
	"math"

	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
)

// Lloyd runs plain Lloyd iterations from the given initial labels under
// the similarity, recomputing every centroid from scratch each iteration,
// and returns the labels and the I2 value Σ_k ‖D_k‖.
func Lloyd(ctx context.Context, m matrix.Rows, labels []int32, k int, sim metric.Similarity, maxIter int) ([]int32, float64, error) {
	labels = append([]int32(nil), labels...)
	n, cols := m.NumRows(), m.NumCols()
	for iter := 0; iter < maxIter; iter++ {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		sums := centroids(m, labels, k, cols)
		changed := false
		for i := 0; i < n; i++ {
			best, bestS := int32(-1), math.Inf(-1)
			for c := 0; c < k; c++ {
				if sums[c].N == 0 {
					continue
				}
				s := sim.RowToComposite(m.Row(i), m.RowNorm(i), sums[c])
				if s > bestS {
					best, bestS = int32(c), s
				}
			}
			if best != labels[i] {
				labels[i] = best
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return labels, I2(m, labels, k), nil
}

// I2 computes Σ_k ‖D_k‖ from scratch.
func I2(m matrix.Rows, labels []int32, k int) float64 {
	var s float64
	for _, c := range centroids(m, labels, k, m.NumCols()) {
		s += c.Norm()
	}
	return s
}

// Refine applies single-row moves greedily in the given visit order using
// from-scratch I2 recomputation, until a full pass makes no move.
func Refine(m matrix.Rows, labels []int32, k int, order []int, maxPasses int) ([]int32, float64) {
	labels = append([]int32(nil), labels...)
	sizes := make([]int, k)
	for _, l := range labels {
		sizes[l]++
	}
	cur := I2(m, labels, k)
	for pass := 0; pass < maxPasses; pass++ {
		moved := false
		for _, i := range order {
			from := labels[i]
			if sizes[from] <= 1 {
				continue
			}
			bestTo, bestVal := from, cur
			for to := int32(0); int(to) < k; to++ {
				if to == from {
					continue
				}
				labels[i] = to
				v := I2(m, labels, k)
				if v > bestVal+1e-12 {
					bestTo, bestVal = to, v
				}
			}
			labels[i] = bestTo
			if bestTo != from {
				sizes[from]--
				sizes[bestTo]++
				cur = bestVal
				moved = true
			}
		}
		if !moved {
			break
		}
	}
	return labels, cur
}

func centroids(m matrix.Rows, labels []int32, k, cols int) []*matrix.Composite {
	cs := make([]*matrix.Composite, k)
	for c := range cs {
		cs[c] = matrix.NewComposite(cols)
	}
	for i, l := range labels {
		cs[l].Add(m.Row(i), m.RowNorm(i))
	}
	for _, c := range cs {
		c.Recompute()
	}
	return cs
}
