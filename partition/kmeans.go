package partition

import (
	"context"
	"math"

	"github.com/grokify/clutro/criterion"
	"github.com/grokify/clutro/internal/par"
	"github.com/grokify/clutro/internal/rng"
	"github.com/grokify/clutro/matrix"
)

// KMeans partitions the rows of m into k clusters: k-means++ seeding, then
// Lloyd iterations assigning each row to the composite it is most similar to
// (spherical k-means under cosine, classic k-means under Euclidean), run
// Trials times from independent seeds, keeping the partition with the best
// criterion value.
func KMeans(ctx context.Context, m matrix.Rows, k int, o Options) (Assignment, error) {
	if err := check(m, k); err != nil {
		return Assignment{}, err
	}
	rows := make([]int32, m.NumRows())
	for i := range rows {
		rows[i] = int32(i)
	}
	return kmeansSubset(ctx, m, rows, k, o.withDefaults(), "kmeans")
}

// kmeansSubset clusters the given rows (indices into m) into k clusters.
// Labels are returned per position in rows. The stream name keeps the random
// choices of nested uses (bisection) independent of top-level ones.
func kmeansSubset(ctx context.Context, m matrix.Rows, rows []int32, k int, o Options, stream string) (Assignment, error) {
	best := Assignment{K: k}
	bestSet := false
	for trial := 0; trial < o.Trials; trial++ {
		if err := ctx.Err(); err != nil {
			return Assignment{}, err
		}
		r := rng.SubStream(o.Seed, stream, trial)
		labels := seedPlusPlus(m, rows, k, o, r)
		a, err := lloyd(ctx, m, rows, labels, k, o)
		if err != nil {
			return Assignment{}, err
		}
		if !bestSet || criterion.Better(o.Criterion, a.Criterion, best.Criterion) {
			best, bestSet = a, true
		}
	}
	return best, nil
}

// seedPlusPlus picks k initial centers by greedy k-means++: each new center
// is chosen among 2+ln(k) candidates drawn by D² sampling, keeping the
// candidate that lowers the total potential most (as scikit-learn does).
// Every row is then assigned to its most similar center. Distance is
// 1 − similarity.
func seedPlusPlus(m matrix.Rows, rows []int32, k int, o Options, r interface {
	Float64() float64
	IntN(int) int
}) []int32 {
	n := len(rows)
	centers := make([]int32, 0, k)
	d2 := make([]float64, n) // current squared distance to the nearest center
	dist2To := func(c int32, i int) float64 {
		d := 1 - o.Similarity.Sim(m.Row(int(rows[i])), m.Row(int(c)), m.RowNorm(int(rows[i])), m.RowNorm(int(c)))
		if d < 0 {
			d = 0
		}
		return d * d
	}
	first := rows[r.IntN(n)]
	centers = append(centers, first)
	var total float64
	for i := range d2 {
		d2[i] = dist2To(first, i)
		total += d2[i]
	}
	nCand := 2 + int(math.Log(float64(k)))
	cand := make([]float64, n)
	for len(centers) < k {
		if total <= 0 {
			// Every row coincides with a center: take the first row that is
			// not already one.
			centers = append(centers, rows[firstNonCenter(rows, centers)])
			continue
		}
		bestPick, bestPot := -1, math.Inf(1)
		var bestD2 []float64
		for t := 0; t < nCand; t++ {
			x := r.Float64() * total
			acc, pick := 0.0, n-1
			for i := range rows {
				acc += d2[i]
				if acc >= x {
					pick = i
					break
				}
			}
			var pot float64
			for i := range rows {
				cand[i] = math.Min(d2[i], dist2To(rows[pick], i))
				pot += cand[i]
			}
			if pot < bestPot {
				bestPick, bestPot = pick, pot
				bestD2 = append(bestD2[:0], cand...)
			}
		}
		centers = append(centers, rows[bestPick])
		copy(d2, bestD2)
		total = bestPot
	}
	labels := make([]int32, n)
	for i, ri := range rows {
		bestL, bestS := int32(0), math.Inf(-1)
		for l, c := range centers {
			s := o.Similarity.Sim(m.Row(int(ri)), m.Row(int(c)), m.RowNorm(int(ri)), m.RowNorm(int(c)))
			if s > bestS {
				bestL, bestS = int32(l), s
			}
		}
		labels[i] = bestL
	}
	return labels
}

func firstNonCenter(rows, centers []int32) int {
	isCenter := map[int32]bool{}
	for _, c := range centers {
		isCenter[c] = true
	}
	for i, r := range rows {
		if !isCenter[r] {
			return i
		}
	}
	return 0
}

// lloyd iterates assignment and composite update until labels stop
// changing or MaxIter. Assignment is parallel and positional; ties break to
// the lowest cluster index; empty clusters are reseeded with the row least
// similar to its own composite, deterministically.
func lloyd(ctx context.Context, m matrix.Rows, rows, labels []int32, k int, o Options) (Assignment, error) {
	n := len(rows)
	comps := make([]*matrix.Composite, k)
	for i := range comps {
		comps[i] = matrix.NewComposite(m.NumCols())
	}
	rebuild := func() {
		for _, c := range comps {
			c.Reset()
		}
		for i, ri := range rows {
			comps[labels[i]].Add(m.Row(int(ri)), m.RowNorm(int(ri)))
		}
	}
	rebuild()
	next := make([]int32, n)
	for iter := 0; iter < o.MaxIter; iter++ {
		err := par.Run(ctx, n, o.Workers, func(c par.Chunk) {
			for i := c.Start; i < c.End; i++ {
				ri := rows[i]
				row, norm := m.Row(int(ri)), m.RowNorm(int(ri))
				bestL, bestS := int32(0), math.Inf(-1)
				for l, comp := range comps {
					if comp.N == 0 {
						continue
					}
					s := o.Similarity.RowToComposite(row, norm, comp)
					if s > bestS {
						bestL, bestS = int32(l), s
					}
				}
				next[i] = bestL
			}
		})
		if err != nil {
			return Assignment{}, err
		}
		changed := false
		for i := range labels {
			if labels[i] != next[i] {
				labels[i] = next[i]
				changed = true
			}
		}
		rebuild()
		if reseedEmpty(m, rows, labels, comps, o) {
			rebuild()
			changed = true
		}
		if !changed {
			break
		}
	}
	return Assignment{Labels: labels, K: k, Criterion: o.Criterion.Value(comps)}, nil
}

// reseedEmpty moves, into each empty cluster, the row least similar to its
// current composite among clusters with more than one row. Deterministic:
// candidates are scanned in row order and the first minimum wins.
func reseedEmpty(m matrix.Rows, rows, labels []int32, comps []*matrix.Composite, o Options) bool {
	moved := false
	for l, comp := range comps {
		if comp.N != 0 {
			continue
		}
		worst, worstS := -1, math.Inf(1)
		for i, ri := range rows {
			if comps[labels[i]].N <= 1 {
				continue
			}
			s := o.Similarity.RowToComposite(m.Row(int(ri)), m.RowNorm(int(ri)), comps[labels[i]])
			if s < worstS {
				worst, worstS = i, s
			}
		}
		if worst < 0 {
			continue
		}
		ri := rows[worst]
		comps[labels[worst]].Remove(m.Row(int(ri)), m.RowNorm(int(ri)))
		labels[worst] = int32(l)
		comp.Add(m.Row(int(ri)), m.RowNorm(int(ri)))
		moved = true
	}
	return moved
}
