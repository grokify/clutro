package synth

import (
	"testing"

	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
)

func TestSparsePlantedStructure(t *testing.T) {
	p := Params{Rows: 300, Cols: 400, K: 3, NNZ: 20, Noise: 0.1, Seed: 1}
	m, labels := Sparse(p)
	if m.NumRows() != 300 || m.NumCols() != 400 || len(labels) != 300 {
		t.Fatal("shape")
	}
	for i := 0; i < m.NumRows(); i++ {
		if n := m.RowNorm(i); n < 0.999 || n > 1.001 {
			t.Fatalf("row %d not normalized: %v", i, n)
		}
	}
	// Rows in the same planted cluster must be more similar on average than
	// rows in different clusters.
	var same, diff float64
	var ns, nd int
	cos := metric.Cosine{}
	for i := 0; i < 100; i++ {
		for j := i + 1; j < 100; j++ {
			s := cos.Sim(m.Row(i), m.Row(j), 1, 1)
			if labels[i] == labels[j] {
				same += s
				ns++
			} else {
				diff += s
				nd++
			}
		}
	}
	if same/float64(ns) <= 3*diff/float64(nd) {
		t.Fatalf("planted structure too weak: same %v diff %v", same/float64(ns), diff/float64(nd))
	}
	m2, labels2 := Sparse(p)
	if m2.NNZ() != m.NNZ() || labels2[7] != labels[7] {
		t.Fatal("generator must be deterministic")
	}
	sh, shl := Shuffled(m, labels, 9)
	if sh.NumRows() != m.NumRows() || len(shl) != len(labels) {
		t.Fatal("shuffled shape")
	}
}

func TestDenseBlobs(t *testing.T) {
	m, labels := Dense(90, 8, 3, 0.5, 2)
	if m.NumRows() != 90 || m.NumCols() != 8 || labels[4] != 1 {
		t.Fatal("dense shape/labels")
	}
	var _ matrix.Rows = m
}
