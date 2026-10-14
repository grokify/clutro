package evaluate

import (
	"errors"
	"fmt"
	"math"

	"github.com/grokify/clutro/internal/rng"
	"github.com/grokify/clutro/matrix"
)

// ErrLabels reports labels that do not fit the matrix or k.
var ErrLabels = errors.New("evaluate: invalid labels")

// Distance between two rows, with their norms.
type Distance func(a, b matrix.SparseVec, normA, normB float32) float64

// Euclidean is the default distance for the internal metrics. On
// L2-normalized rows it is monotone in cosine similarity, so it ranks
// neighbours the same way.
func Euclidean(a, b matrix.SparseVec, normA, normB float32) float64 {
	d2 := float64(normA)*float64(normA) + float64(normB)*float64(normB) - 2*matrix.Dot(a, b)
	if d2 <= 0 {
		return 0
	}
	return math.Sqrt(d2)
}

func checkLabels(m matrix.Rows, labels []int32, k int) error {
	if m == nil || len(labels) != m.NumRows() {
		return fmt.Errorf("%w: %d labels for %d rows", ErrLabels, len(labels), rows(m))
	}
	if k < 1 {
		return fmt.Errorf("%w: k=%d", ErrLabels, k)
	}
	for i, l := range labels {
		if l < 0 || int(l) >= k {
			return fmt.Errorf("%w: label %d at row %d outside [0,%d)", ErrLabels, l, i, k)
		}
	}
	return nil
}

func rows(m matrix.Rows) int {
	if m == nil {
		return 0
	}
	return m.NumRows()
}

// SilhouetteOptions configures Silhouette.
type SilhouetteOptions struct {
	// Distance defaults to Euclidean.
	Distance Distance
	// SampleAbove caps the exact O(n²) computation: above this many rows a
	// seeded sample of this size is scored instead. Default 10000; 0 keeps
	// the default, a negative value disables sampling.
	SampleAbove int
	Seed        uint64
}

// Silhouette is the mean silhouette coefficient over rows (or a seeded
// sample of them): (b − a) / max(a, b), where a is a row's mean distance to
// its own cluster and b the smallest mean distance to another cluster. Rows
// in singleton clusters score 0, as in scikit-learn. Needs 2 ≤ k < n.
func Silhouette(m matrix.Rows, labels []int32, k int, o SilhouetteOptions) (float64, error) {
	if err := checkLabels(m, labels, k); err != nil {
		return 0, err
	}
	n := m.NumRows()
	if k < 2 || k >= n {
		return 0, fmt.Errorf("%w: silhouette needs 2 <= k < rows (k=%d, rows=%d)", ErrLabels, k, n)
	}
	dist := o.Distance
	if dist == nil {
		dist = Euclidean
	}
	cap := o.SampleAbove
	if cap == 0 {
		cap = 10000
	}
	sizes := make([]int, k)
	for _, l := range labels {
		sizes[l]++
	}
	scored := make([]int, 0, n)
	if cap > 0 && n > cap {
		perm := rng.Perm(rng.SubStream(o.Seed, "silhouette", 0), n)
		scored = append(scored, perm[:cap]...)
	} else {
		for i := 0; i < n; i++ {
			scored = append(scored, i)
		}
	}
	sum := make([]float64, k)
	var total float64
	for _, i := range scored {
		for c := range sum {
			sum[c] = 0
		}
		ri, ni := m.Row(i), m.RowNorm(i)
		for j := 0; j < n; j++ {
			if j != i {
				sum[labels[j]] += dist(ri, m.Row(j), ni, m.RowNorm(j))
			}
		}
		own := labels[i]
		if sizes[own] <= 1 {
			continue // s = 0
		}
		a := sum[own] / float64(sizes[own]-1)
		b := math.Inf(1)
		for c := 0; c < k; c++ {
			if int32(c) == own || sizes[c] == 0 { //nolint:gosec // k is small
				continue
			}
			if v := sum[c] / float64(sizes[c]); v < b {
				b = v
			}
		}
		if mx := math.Max(a, b); mx > 0 {
			total += (b - a) / mx
		}
	}
	return total / float64(len(scored)), nil
}

// centroidsAndScatter returns each cluster's centroid (mean vector), the
// mean distance of its rows to that centroid, and the overall mean vector.
func centroidsAndScatter(m matrix.Rows, labels []int32, k int) ([][]float64, []float64, []int, []float64) {
	cols := m.NumCols()
	cent := make([][]float64, k)
	for c := range cent {
		cent[c] = make([]float64, cols)
	}
	sizes := make([]int, k)
	all := make([]float64, cols)
	for i, l := range labels {
		r := m.Row(i)
		for t, idx := range r.Idx {
			cent[l][idx] += float64(r.Val[t])
			all[idx] += float64(r.Val[t])
		}
		sizes[l]++
	}
	for c := range cent {
		if sizes[c] > 0 {
			for j := range cent[c] {
				cent[c][j] /= float64(sizes[c])
			}
		}
	}
	for j := range all {
		all[j] /= float64(len(labels))
	}
	scatter := make([]float64, k)
	for i, l := range labels {
		scatter[l] += distToDense(m.Row(i), cent[l])
	}
	for c := range scatter {
		if sizes[c] > 0 {
			scatter[c] /= float64(sizes[c])
		}
	}
	return cent, scatter, sizes, all
}

// distToDense is the Euclidean distance from a sparse row to a dense vector.
func distToDense(r matrix.SparseVec, d []float64) float64 {
	var s float64
	t := 0
	for j, v := range d {
		if t < len(r.Idx) && int(r.Idx[t]) == j {
			s += (float64(r.Val[t]) - v) * (float64(r.Val[t]) - v)
			t++
		} else {
			s += v * v
		}
	}
	return math.Sqrt(s)
}

func denseDist(a, b []float64) float64 {
	var s float64
	for j := range a {
		s += (a[j] - b[j]) * (a[j] - b[j])
	}
	return math.Sqrt(s)
}

// DaviesBouldin is the mean over clusters of the worst ratio (s_i + s_j) /
// d(c_i, c_j), where s is a cluster's mean distance to its centroid and d
// the distance between centroids. Lower is better; 0 is perfect. Needs
// 2 ≤ k < n.
func DaviesBouldin(m matrix.Rows, labels []int32, k int) (float64, error) {
	if err := checkLabels(m, labels, k); err != nil {
		return 0, err
	}
	if k < 2 || k >= m.NumRows() {
		return 0, fmt.Errorf("%w: Davies-Bouldin needs 2 <= k < rows", ErrLabels)
	}
	cent, scatter, sizes, _ := centroidsAndScatter(m, labels, k)
	var total float64
	active := 0
	for i := 0; i < k; i++ {
		if sizes[i] == 0 {
			continue
		}
		worst := 0.0
		for j := 0; j < k; j++ {
			if j == i || sizes[j] == 0 {
				continue
			}
			d := denseDist(cent[i], cent[j])
			if d == 0 {
				return math.Inf(1), nil
			}
			if r := (scatter[i] + scatter[j]) / d; r > worst {
				worst = r
			}
		}
		total += worst
		active++
	}
	if active == 0 {
		return 0, nil
	}
	return total / float64(active), nil
}

// CalinskiHarabasz is the ratio of between-cluster to within-cluster
// dispersion, [B/(k−1)] / [W/(n−k)], with B = Σ n_c ‖c − mean‖² and
// W = Σ ‖x − c_x‖². Higher is better. Needs 2 ≤ k < n.
func CalinskiHarabasz(m matrix.Rows, labels []int32, k int) (float64, error) {
	if err := checkLabels(m, labels, k); err != nil {
		return 0, err
	}
	n := m.NumRows()
	if k < 2 || k >= n {
		return 0, fmt.Errorf("%w: Calinski-Harabasz needs 2 <= k < rows", ErrLabels)
	}
	cent, _, sizes, all := centroidsAndScatter(m, labels, k)
	var between, within float64
	for c := 0; c < k; c++ {
		if sizes[c] == 0 {
			continue
		}
		d := denseDist(cent[c], all)
		between += float64(sizes[c]) * d * d
	}
	for i, l := range labels {
		d := distToDense(m.Row(i), cent[l])
		within += d * d
	}
	if within == 0 {
		return math.Inf(1), nil
	}
	return (between / float64(k-1)) / (within / float64(n-k)), nil
}
