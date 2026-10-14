package explain

import (
	"math"
	"testing"

	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
)

// Two clusters over four terms: cluster 0 is about {go, code}, cluster 1
// about {cook, bake}; "the" appears everywhere.
func fixture(t *testing.T) (*matrix.CSR, []int32, []string) {
	t.Helper()
	tr := func(r, c int, v float32) matrix.Triplet { return matrix.Triplet{Row: r, Col: c, Val: v} }
	m, err := matrix.NewCSRFromTriplets(6, 5, []matrix.Triplet{
		tr(0, 0, 2), tr(0, 1, 1), tr(0, 4, 1),
		tr(1, 0, 1), tr(1, 1, 2), tr(1, 4, 1),
		tr(2, 0, 1), tr(2, 1, 1), tr(2, 4, 1),
		tr(3, 2, 2), tr(3, 3, 1), tr(3, 4, 1),
		tr(4, 2, 1), tr(4, 3, 2), tr(4, 4, 1),
		tr(5, 2, 1), tr(5, 3, 1), tr(5, 4, 1),
	})
	if err != nil {
		t.Fatal(err)
	}
	m.NormalizeRowsL2()
	return m, []int32{0, 0, 0, 1, 1, 1}, []string{"go", "code", "cook", "bake", "the"}
}

func TestDescribeAndDiscriminate(t *testing.T) {
	m, labels, cols := fixture(t)
	d := Describe(m, labels, 2, 0, 3, cols)
	if len(d) != 3 || d[0].Label != "go" && d[0].Label != "code" {
		t.Fatalf("describe: %+v", d)
	}
	var total float64
	for _, f := range Describe(m, labels, 2, 0, -1, cols) {
		total += f.Weight
	}
	if math.Abs(total-1) > 1e-9 {
		t.Fatalf("descriptive weights sum to %v", total)
	}
	// "the" describes cluster 0 (it is present) but does not discriminate it.
	disc := Discriminate(m, labels, 2, 0, 5, cols)
	for _, f := range disc {
		if f.Label == "the" {
			t.Fatalf("shared feature must not discriminate: %+v", disc)
		}
	}
	if len(disc) != 2 || (disc[0].Label != "go" && disc[0].Label != "code") {
		t.Fatalf("discriminate: %+v", disc)
	}
	if Describe(m, []int32{0, 0, 0, 0, 0, 0}, 2, 1, 3, cols) != nil {
		t.Fatal("empty cluster has no features")
	}
	if f := Describe(m, labels, 2, 0, 1, nil); f[0].Label != "" {
		t.Fatal("no labels → empty label")
	}
}

func TestRepresentativesAndSummaries(t *testing.T) {
	m, labels, _ := fixture(t)
	rows, sims := Representatives(m, labels, 2, 1, 2, metric.Cosine{})
	if len(rows) != 2 || len(sims) != 2 || sims[0] < sims[1] {
		t.Fatalf("representatives: %v %v", rows, sims)
	}
	for _, r := range rows {
		if labels[r] != 1 {
			t.Fatal("representatives come from the cluster")
		}
	}
	all, _ := Representatives(m, labels, 2, 1, 10, metric.Cosine{})
	if len(all) != 3 {
		t.Fatal("n larger than the cluster returns every row")
	}
	s := Summaries(m, labels, 2, metric.Cosine{})
	if len(s) != 2 || s[0].Size != 3 || s[1].Size != 3 {
		t.Fatalf("summaries: %+v", s)
	}
	for _, c := range s {
		if c.ISim <= c.ESim {
			t.Fatalf("cluster %d: ISim %v should exceed ESim %v", c.Cluster, c.ISim, c.ESim)
		}
		if c.ISimDev < 0 || c.ESimDev < 0 {
			t.Fatal("deviations")
		}
	}
	if mean, dev := meanDev(nil); mean != 0 || dev != 0 {
		t.Fatal("empty meanDev")
	}
}
