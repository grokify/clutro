package metric

import (
	"math"
	"testing"

	"github.com/grokify/clutro/matrix"
)

func vec(idx []uint32, val []float32) (matrix.SparseVec, float32) {
	v := matrix.SparseVec{Idx: idx, Val: val}
	return v, float32(v.Norm())
}

func TestCosine(t *testing.T) {
	a, na := vec([]uint32{0, 1}, []float32{3, 4})
	b, nb := vec([]uint32{0, 1}, []float32{4, 3})
	if got := (Cosine{}).Sim(a, b, na, nb); math.Abs(got-24.0/25.0) > 1e-9 {
		t.Fatalf("cosine %v", got)
	}
	z, nz := vec(nil, nil)
	if (Cosine{}).Sim(a, z, na, nz) != 0 {
		t.Fatal("zero row")
	}
	// Normalized fast path equals the general path.
	an, _ := vec([]uint32{0, 1}, []float32{0.6, 0.8})
	bn, _ := vec([]uint32{0, 1}, []float32{0.8, 0.6})
	if got := (Cosine{}).Sim(an, bn, 1, 1); math.Abs(got-24.0/25.0) > 1e-6 {
		t.Fatalf("normalized cosine %v", got)
	}
	c := matrix.NewComposite(2)
	c.Add(a, na)
	c.Add(b, nb)
	// composite = [7 7]; cos(a, composite) = (21+28)/(5·√98)
	if got := (Cosine{}).RowToComposite(a, na, c); math.Abs(got-49/(5*math.Sqrt(98))) > 1e-9 {
		t.Fatalf("row to composite %v", got)
	}
	if (Cosine{}).RowToComposite(a, na, matrix.NewComposite(2)) != 0 {
		t.Fatal("empty composite")
	}
}

func TestEuclidean(t *testing.T) {
	a, na := vec([]uint32{0, 1}, []float32{0, 0})
	b, nb := vec([]uint32{0, 1}, []float32{3, 4})
	e := Euclidean{}
	if e.Distance(a, b, na, nb) != 5 || e.Sim(a, b, na, nb) != 1.0/6 {
		t.Fatal("distance")
	}
	if e.Distance(b, b, nb, nb) != 0 {
		t.Fatal("self distance")
	}
	c := matrix.NewComposite(2)
	c.Add(b, nb)
	c.Add(a, na)
	// centroid = [1.5 2]; distance from b = sqrt(1.5²+2²) = 2.5
	if got := e.RowToComposite(b, nb, c); math.Abs(got-1/(1+2.5)) > 1e-9 {
		t.Fatalf("row to centroid %v", got)
	}
	if e.RowToComposite(b, nb, matrix.NewComposite(2)) != 0 {
		t.Fatal("empty composite")
	}
}

func TestParse(t *testing.T) {
	if s, ok := Parse("cos"); !ok || s.Name() != "cos" {
		t.Fatal("cos")
	}
	if s, ok := Parse("euclidean"); !ok || s.Name() != "euclid" {
		t.Fatal("euclid")
	}
	if _, ok := Parse("jaccard"); ok {
		t.Fatal("unknown")
	}
}
