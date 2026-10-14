// Package explain interprets clusters the way CLUTO's -showfeatures and
// -showsummaries do: which features describe a cluster, which distinguish
// it from the rest, which rows represent it best, and how tight and how
// separated it is.
//
// All functions take the fitted matrix, the labels and k; the clutro.Result
// methods wrap them with the result's own data and column labels.
package explain

import (
	"math"
	"sort"

	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
)

// Feature is one column's contribution to a cluster's description.
type Feature struct {
	// Col is the column index; Label is its name when labels were given.
	Col   int
	Label string
	// Weight is the column's share of the quantity being explained (within-
	// cluster similarity for descriptive features, separation for
	// discriminating features), in [0, 1]; the shares of all columns sum to 1.
	Weight float64
}

// Summary is a cluster's size and internal/external similarity statistics.
type Summary struct {
	Cluster int
	Size    int
	// ISim is the mean similarity of the cluster's rows to their own
	// composite; ISimDev its standard deviation.
	ISim, ISimDev float64
	// ESim is the mean similarity of the cluster's rows to the composite of
	// all other rows; ESimDev its standard deviation.
	ESim, ESimDev float64
}

// composites returns per-cluster composites and the composite of everything.
func composites(m matrix.Rows, labels []int32, k int) ([]*matrix.Composite, *matrix.Composite) {
	cs := make([]*matrix.Composite, k)
	for i := range cs {
		cs[i] = matrix.NewComposite(m.NumCols())
	}
	all := matrix.NewComposite(m.NumCols())
	for i, l := range labels {
		cs[l].Add(m.Row(i), m.RowNorm(i))
		all.Add(m.Row(i), m.RowNorm(i))
	}
	return cs, all
}

// Describe returns the top-n descriptive features of cluster c: the columns
// whose squared composite weight contributes most to the cluster's internal
// similarity, D_c[j]² / ‖D_c‖². The returned Weight is that share.
func Describe(m matrix.Rows, labels []int32, k, c, n int, colLabels []string) []Feature {
	cs, _ := composites(m, labels, k)
	comp := cs[c]
	norm2 := comp.Norm2()
	if norm2 == 0 {
		return nil
	}
	feats := make([]Feature, 0, m.NumCols())
	for j, v := range comp.Sum {
		if v == 0 {
			continue
		}
		feats = append(feats, Feature{Col: j, Label: label(colLabels, j), Weight: v * v / norm2})
	}
	return topN(feats, n)
}

// Discriminate returns the top-n discriminating features of cluster c: the
// columns that contribute most to the difference between the cluster's
// internal similarity and its similarity to the rest of the data, i.e. the
// features that set it apart. Contributions are D_c[j]·(D_c[j]/‖D_c‖ −
// (D[j]−D_c[j])/‖D−D_c‖), normalized to shares; columns that make the
// cluster more like the rest are dropped.
func Discriminate(m matrix.Rows, labels []int32, k, c, n int, colLabels []string) []Feature {
	cs, all := composites(m, labels, k)
	comp := cs[c]
	cn := comp.Norm()
	if cn == 0 {
		return nil
	}
	rest := make([]float64, m.NumCols())
	var restNorm2 float64
	for j := range rest {
		rest[j] = all.Sum[j] - comp.Sum[j]
		restNorm2 += rest[j] * rest[j]
	}
	restNorm := math.Sqrt(restNorm2)
	feats := make([]Feature, 0, m.NumCols())
	var total float64
	for j, v := range comp.Sum {
		if v == 0 {
			continue
		}
		own := v / cn
		other := 0.0
		if restNorm > 0 {
			other = rest[j] / restNorm
		}
		contrib := v * (own - other)
		if contrib <= 0 {
			continue
		}
		feats = append(feats, Feature{Col: j, Label: label(colLabels, j), Weight: contrib})
		total += contrib
	}
	if total == 0 {
		return nil
	}
	for i := range feats {
		feats[i].Weight /= total
	}
	return topN(feats, n)
}

// Representatives returns the n rows of cluster c most similar to its
// composite, best first, with their similarity.
func Representatives(m matrix.Rows, labels []int32, k, c, n int, sim metric.Similarity) ([]int, []float64) {
	cs, _ := composites(m, labels, k)
	type scored struct {
		row int
		s   float64
	}
	var rows []scored
	for i, l := range labels {
		if int(l) == c {
			rows = append(rows, scored{i, sim.RowToComposite(m.Row(i), m.RowNorm(i), cs[c])})
		}
	}
	sort.SliceStable(rows, func(a, b int) bool {
		if rows[a].s != rows[b].s {
			return rows[a].s > rows[b].s
		}
		return rows[a].row < rows[b].row
	})
	if n < len(rows) {
		rows = rows[:n]
	}
	idx := make([]int, len(rows))
	sims := make([]float64, len(rows))
	for i, r := range rows {
		idx[i], sims[i] = r.row, r.s
	}
	return idx, sims
}

// Summaries computes the per-cluster size and ISim/ESim statistics.
func Summaries(m matrix.Rows, labels []int32, k int, sim metric.Similarity) []Summary {
	cs, all := composites(m, labels, k)
	out := make([]Summary, k)
	for c := range out {
		out[c].Cluster = c
	}
	isim := make([][]float64, k)
	esim := make([][]float64, k)
	rest := matrix.NewComposite(m.NumCols())
	for c := 0; c < k; c++ {
		// composite of everything outside c
		rest.Reset()
		for j := range rest.Sum {
			rest.Sum[j] = all.Sum[j] - cs[c].Sum[j]
		}
		rest.N = all.N - cs[c].N
		rest.Recompute()
		for i, l := range labels {
			if int(l) != c {
				continue
			}
			isim[c] = append(isim[c], sim.RowToComposite(m.Row(i), m.RowNorm(i), cs[c]))
			if rest.N > 0 {
				esim[c] = append(esim[c], sim.RowToComposite(m.Row(i), m.RowNorm(i), rest))
			}
		}
		out[c].Size = len(isim[c])
		out[c].ISim, out[c].ISimDev = meanDev(isim[c])
		out[c].ESim, out[c].ESimDev = meanDev(esim[c])
	}
	return out
}

func meanDev(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	mean := s / float64(len(xs))
	var v float64
	for _, x := range xs {
		v += (x - mean) * (x - mean)
	}
	return mean, math.Sqrt(v / float64(len(xs)))
}

func label(labels []string, j int) string {
	if j < len(labels) {
		return labels[j]
	}
	return ""
}

// topN sorts by weight descending (ties by column) and truncates.
func topN(feats []Feature, n int) []Feature {
	sort.SliceStable(feats, func(a, b int) bool {
		if feats[a].Weight != feats[b].Weight {
			return feats[a].Weight > feats[b].Weight
		}
		return feats[a].Col < feats[b].Col
	})
	if n >= 0 && n < len(feats) {
		feats = feats[:n]
	}
	return feats
}
