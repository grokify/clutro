package partition

import (
	"context"
	"sort"

	"github.com/grokify/clutro/matrix"
)

// Bisect performs repeated bisection (CLUTO rb): starting from one cluster,
// it repeatedly selects a cluster (SelectBest or SelectLargest), splits it
// with a 2-way k-means over Trials seeded starts keeping the split with the
// best criterion, and stops at k clusters. With Options.Refine it then runs
// Refine on the k-way result (rbr). The bisection tree is returned alongside
// the flat assignment.
func Bisect(ctx context.Context, m matrix.Rows, k int, o Options) (Assignment, *Tree, error) {
	if err := check(m, k); err != nil {
		return Assignment{}, nil, err
	}
	o = o.withDefaults()
	n := m.NumRows()
	all := make([]int32, n)
	for i := range all {
		all[i] = int32(i)
	}
	root := &Node{Rows: all, Label: -1, Criterion: clusterCriterion(m, all, o)}
	frontier := []*Node{root}
	splits := map[*Node]*split{} // cached best bisection per frontier node
	splitIdx := 0

	for len(frontier) < k {
		if err := ctx.Err(); err != nil {
			return Assignment{}, nil, err
		}
		// Choose the node to split.
		var chosen *Node
		switch o.Selection {
		case SelectLargest:
			for _, f := range frontier {
				if len(f.Rows) >= 2 && (chosen == nil || len(f.Rows) > len(chosen.Rows)) {
					chosen = f
				}
			}
		default: // SelectBest: largest criterion gain
			bestGain := 0.0
			for _, f := range frontier {
				if len(f.Rows) < 2 {
					continue
				}
				s, err := cachedSplit(ctx, m, f, o, splits, &splitIdx)
				if err != nil {
					return Assignment{}, nil, err
				}
				gain := s.gain(o)
				if chosen == nil || gain > bestGain {
					chosen, bestGain = f, gain
				}
			}
		}
		if chosen == nil {
			break // nothing splittable (all singletons)
		}
		s, err := cachedSplit(ctx, m, chosen, o, splits, &splitIdx)
		if err != nil {
			return Assignment{}, nil, err
		}
		chosen.Left, chosen.Right = s.left, s.right
		delete(splits, chosen)
		// Replace chosen in the frontier with its children, keeping order.
		for i, f := range frontier {
			if f == chosen {
				frontier = append(frontier[:i], append([]*Node{s.left, s.right}, frontier[i+1:]...)...)
				break
			}
		}
	}

	labels := make([]int32, n)
	for l, f := range frontier {
		f.Label = int32(l)
		for _, r := range f.Rows {
			labels[r] = int32(l)
		}
	}
	a := Assignment{Labels: labels, K: len(frontier)}
	a.Criterion = o.Criterion.Value(composites(m, labels, a.K))
	if o.Refine && a.K > 1 {
		var err error
		a, err = Refine(ctx, m, a, o)
		if err != nil {
			return Assignment{}, nil, err
		}
		// Leaves keep their row sets from the bisection; the refined
		// assignment is the flat result.
	}
	return a, &Tree{Root: root}, nil
}

type split struct {
	left, right *Node
	parent      float64
}

// gain is the criterion improvement of the split over the unsplit cluster.
func (s *split) gain(o Options) float64 {
	g := (s.left.Criterion + s.right.Criterion) - s.parent
	if !o.Criterion.Maximize() {
		g = -g
	}
	return g
}

func cachedSplit(ctx context.Context, m matrix.Rows, f *Node, o Options, cache map[*Node]*split, idx *int) (*split, error) {
	if s, ok := cache[f]; ok {
		return s, nil
	}
	// Each bisection gets its own stream index so trials never repeat
	// another split's random choices.
	stream := "bisect"
	o2 := o
	o2.Seed = o.Seed ^ uint64(*idx)*0x9E3779B97F4A7C15
	*idx++
	a, err := kmeansSubset(ctx, m, f.Rows, 2, o2, stream)
	if err != nil {
		return nil, err
	}
	var lrows, rrows []int32
	for i, r := range f.Rows {
		if a.Labels[i] == 0 {
			lrows = append(lrows, r)
		} else {
			rrows = append(rrows, r)
		}
	}
	sort.Slice(lrows, func(i, j int) bool { return lrows[i] < lrows[j] })
	sort.Slice(rrows, func(i, j int) bool { return rrows[i] < rrows[j] })
	s := &split{
		left:   &Node{Rows: lrows, Label: -1, Criterion: clusterCriterion(m, lrows, o)},
		right:  &Node{Rows: rrows, Label: -1, Criterion: clusterCriterion(m, rrows, o)},
		parent: f.Criterion,
	}
	cache[f] = s
	return s, nil
}

// clusterCriterion is the objective contribution of one cluster on its own.
func clusterCriterion(m matrix.Rows, rows []int32, o Options) float64 {
	c := matrix.NewComposite(m.NumCols())
	for _, r := range rows {
		c.Add(m.Row(int(r)), m.RowNorm(int(r)))
	}
	return o.Criterion.Value([]*matrix.Composite{c})
}
