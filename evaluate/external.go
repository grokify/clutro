// Package evaluate scores clusterings: external metrics against known
// labels (adjusted Rand index, normalized mutual information, purity,
// entropy) and internal metrics from the data alone.
package evaluate

import (
	"errors"
	"math"
)

// ErrLength reports label slices of different lengths.
var ErrLength = errors.New("evaluate: label slices differ in length")

// contingency builds the counts n[a][b] of rows with label a in x and b in y.
func contingency(x, y []int32) (map[int32]map[int32]int, map[int32]int, map[int32]int) {
	n := map[int32]map[int32]int{}
	rx, ry := map[int32]int{}, map[int32]int{}
	for i := range x {
		if n[x[i]] == nil {
			n[x[i]] = map[int32]int{}
		}
		n[x[i]][y[i]]++
		rx[x[i]]++
		ry[y[i]]++
	}
	return n, rx, ry
}

func choose2(n int) float64 { return float64(n) * float64(n-1) / 2 }

// AdjustedRandIndex measures agreement between two labelings, corrected for
// chance: 1 for identical partitions (up to relabelling), about 0 for
// independent ones. Negative labels are treated as ordinary labels.
func AdjustedRandIndex(x, y []int32) (float64, error) {
	if len(x) != len(y) {
		return 0, ErrLength
	}
	if len(x) == 0 {
		return 1, nil
	}
	n, rx, ry := contingency(x, y)
	var sumCells, sumX, sumY float64
	for _, row := range n {
		for _, c := range row {
			sumCells += choose2(c)
		}
	}
	for _, c := range rx {
		sumX += choose2(c)
	}
	for _, c := range ry {
		sumY += choose2(c)
	}
	total := choose2(len(x))
	if total == 0 {
		return 1, nil
	}
	expected := sumX * sumY / total
	maxIndex := (sumX + sumY) / 2
	if maxIndex == expected {
		return 1, nil // both partitions trivial
	}
	return (sumCells - expected) / (maxIndex - expected), nil
}

// NormalizedMutualInformation is I(x;y) / sqrt(H(x) H(y)), in [0, 1].
func NormalizedMutualInformation(x, y []int32) (float64, error) {
	if len(x) != len(y) {
		return 0, ErrLength
	}
	if len(x) == 0 {
		return 1, nil
	}
	n, rx, ry := contingency(x, y)
	N := float64(len(x))
	var mi, hx, hy float64
	for a, row := range n {
		for b, c := range row {
			p := float64(c) / N
			mi += p * math.Log(p/((float64(rx[a])/N)*(float64(ry[b])/N)))
		}
	}
	for _, c := range rx {
		p := float64(c) / N
		hx -= p * math.Log(p)
	}
	for _, c := range ry {
		p := float64(c) / N
		hy -= p * math.Log(p)
	}
	if hx == 0 || hy == 0 {
		if hx == 0 && hy == 0 {
			return 1, nil
		}
		return 0, nil
	}
	return mi / math.Sqrt(hx*hy), nil
}

// Purity is the fraction of rows whose cluster's majority class is their
// own class: Σ_k max_c n_kc / N, where x are cluster labels and truth the
// class labels.
func Purity(x, truth []int32) (float64, error) {
	if len(x) != len(truth) {
		return 0, ErrLength
	}
	if len(x) == 0 {
		return 1, nil
	}
	n, _, _ := contingency(x, truth)
	var s int
	for _, row := range n {
		best := 0
		for _, c := range row {
			if c > best {
				best = c
			}
		}
		s += best
	}
	return float64(s) / float64(len(x)), nil
}

// Entropy is the size-weighted average class entropy of the clusters,
// normalized by log(number of classes) as CLUTO reports it; 0 is pure.
func Entropy(x, truth []int32) (float64, error) {
	if len(x) != len(truth) {
		return 0, ErrLength
	}
	if len(x) == 0 {
		return 0, nil
	}
	n, rx, rt := contingency(x, truth)
	if len(rt) <= 1 {
		return 0, nil
	}
	norm := math.Log(float64(len(rt)))
	var e float64
	for k, row := range n {
		var ek float64
		nk := float64(rx[k])
		for _, c := range row {
			p := float64(c) / nk
			ek -= p * math.Log(p)
		}
		e += nk / float64(len(x)) * ek / norm
	}
	return e, nil
}
