package criterion

import (
	"math"
	"testing"

	"github.com/grokify/clutro/matrix"
)

func row(idx []uint32, val []float32) (matrix.SparseVec, float32) {
	r := matrix.SparseVec{Idx: idx, Val: val}
	return r, float32(r.Norm())
}

func TestI2ValueAndDelta(t *testing.T) {
	a, na := row([]uint32{0}, []float32{1})
	b, nb := row([]uint32{1}, []float32{1})
	c, nc := row([]uint32{0, 1}, []float32{0.6, 0.8})

	c1 := matrix.NewComposite(2)
	c1.Add(a, na)
	c1.Add(c, nc) // sum [1.6 0.8], norm √3.2
	c2 := matrix.NewComposite(2)
	c2.Add(b, nb) // norm 1
	crit := I2{}
	if crit.Name() != "i2" || !crit.Maximize() {
		t.Fatal("identity")
	}
	want := math.Sqrt(3.2) + 1
	if got := crit.Value([]*matrix.Composite{c1, c2, matrix.NewComposite(2), nil}); math.Abs(got-want) > 1e-6 {
		t.Fatalf("value %v want %v", got, want)
	}

	// Moving c from c1 to c2: c1 → [1 0] norm 1; c2 → [0.6 1.8] norm √3.6.
	delta := crit.Delta(c1, c2, c, nc)
	wantDelta := (1 + math.Sqrt(3.6)) - want
	if math.Abs(delta-wantDelta) > 1e-6 {
		t.Fatalf("delta %v want %v", delta, wantDelta)
	}
	// Applying the move must change Value by exactly delta.
	before := crit.Value([]*matrix.Composite{c1, c2})
	c1.Remove(c, nc)
	c2.Add(c, nc)
	after := crit.Value([]*matrix.Composite{c1, c2})
	if math.Abs((after-before)-delta) > 1e-6 {
		t.Fatalf("delta %v but value changed by %v", delta, after-before)
	}
	// Entering and leaving the partition.
	if d := crit.Delta(nil, c2, a, na); math.Abs(d-(c2.NormIfAdded(a, na)-c2.Norm())) > 1e-9 {
		t.Fatal("delta with nil from")
	}
	if d := crit.Delta(c2, nil, c, nc); math.Abs(d-(c2.NormIfRemoved(c, nc)-c2.Norm())) > 1e-9 {
		t.Fatal("delta with nil to")
	}
	if !Better(crit, 2, 1) || Better(crit, 1, 2) {
		t.Fatal("Better")
	}
	if _, ok := Parse("i2"); !ok {
		t.Fatal("parse")
	}
	if _, ok := Parse("g1"); ok {
		t.Fatal("unknown criterion")
	}
}
