package matrix

import (
	"math"
	"testing"
)

func TestPreprocess(t *testing.T) {
	m, err := NewCSRFromTriplets(4, 3, []Triplet{
		{0, 0, 1}, {0, 1, 1},
		{1, 0, 1}, {1, 2, 1},
		{2, 0, 1},
		{3, 0, 1}, {3, 1, 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	w := IDFWeights(m)
	// column 0 in every row → 0; column 1 in 2 of 4 → log 2; column 2 in 1 of 4 → log 4
	if w[0] != 0 || math.Abs(float64(w[1])-math.Log(2)) > 1e-6 || math.Abs(float64(w[2])-math.Log(4)) > 1e-6 {
		t.Fatalf("idf %v", w)
	}
	if err := Preprocess(m, RowL2, ColIDF); err != nil {
		t.Fatal(err)
	}
	if r2 := m.Row(2); r2.Val[0] != 0 || m.RowNorm(2) != 0 {
		t.Fatal("a row with only an everywhere-column becomes zero and stays zero")
	}
	if math.Abs(float64(m.RowNorm(0))-1) > 1e-6 || m.Row(0).Val[0] != 0 || math.Abs(float64(m.Row(0).Val[1])-1) > 1e-6 {
		t.Fatalf("row 0 after idf+l2: %+v norm %v", m.Row(0), m.RowNorm(0))
	}
	for _, s := range []string{"none", "l2"} {
		if _, err := ParseRowModel(s); err != nil {
			t.Fatal(s)
		}
	}
	for _, s := range []string{"none", "idf"} {
		if _, err := ParseColModel(s); err != nil {
			t.Fatal(s)
		}
	}
	if _, err := ParseRowModel("maxtf"); err == nil {
		t.Fatal("unknown row model")
	}
	if _, err := ParseColModel("bm25"); err == nil {
		t.Fatal("unknown col model")
	}
}
