package partition

import (
	"context"

	"github.com/grokify/clutro/internal/rng"
	"github.com/grokify/clutro/matrix"
)

// Refine improves an assignment by moving single rows between clusters
// while the criterion improves: each pass visits the rows in a seeded random
// order, computes the delta of moving the row to every other cluster, and
// applies the best positive move immediately. It stops when a pass makes no
// move or after MaxIter passes. Clusters are never emptied.
func Refine(ctx context.Context, m matrix.Rows, a Assignment, o Options) (Assignment, error) {
	if err := check(m, a.K); err != nil {
		return Assignment{}, err
	}
	if len(a.Labels) != m.NumRows() {
		return Assignment{}, ErrInput
	}
	o = o.withDefaults()
	labels := append([]int32(nil), a.Labels...)
	comps := composites(m, labels, a.K)
	const eps = 1e-12
	for pass := 0; pass < o.MaxIter; pass++ {
		if err := ctx.Err(); err != nil {
			return Assignment{}, err
		}
		order := rng.Perm(rng.SubStream(o.Seed, "refine", pass), len(labels))
		moved := 0
		for _, i := range order {
			from := labels[i]
			if comps[from].N <= 1 {
				continue
			}
			row, norm := m.Row(i), m.RowNorm(i)
			bestTo, bestDelta := from, 0.0
			for to := range comps {
				if int32(to) == from {
					continue
				}
				d := o.Criterion.Delta(comps[from], comps[to], row, norm)
				if !o.Criterion.Maximize() {
					d = -d
				}
				if d > bestDelta+eps {
					bestTo, bestDelta = int32(to), d
				}
			}
			if bestTo != from {
				comps[from].Remove(row, norm)
				comps[bestTo].Add(row, norm)
				labels[i] = bestTo
				moved++
			}
		}
		if moved == 0 {
			break
		}
		for _, c := range comps {
			c.Recompute()
		}
	}
	return Assignment{Labels: labels, K: a.K, Criterion: o.Criterion.Value(comps)}, nil
}

// Direct computes a k-way partition directly (CLUTO direct): k-means++ and
// Lloyd over Trials starts, then refinement.
func Direct(ctx context.Context, m matrix.Rows, k int, o Options) (Assignment, error) {
	a, err := KMeans(ctx, m, k, o)
	if err != nil {
		return Assignment{}, err
	}
	if k == 1 {
		return a, nil
	}
	return Refine(ctx, m, a, o)
}
