package rng

import (
	"reflect"
	"testing"
)

func TestSubStreamsAreReproducibleAndIndependent(t *testing.T) {
	a := SubStream(42, "trial", 0)
	b := SubStream(42, "trial", 0)
	for i := 0; i < 3; i++ {
		if a.Uint64() != b.Uint64() {
			t.Fatal("same inputs must give the same stream")
		}
	}
	c := SubStream(42, "trial", 1)
	d := SubStream(43, "trial", 0)
	e := SubStream(42, "init", 0)
	first := SubStream(42, "trial", 0).Uint64()
	for _, r := range []uint64{c.Uint64(), d.Uint64(), e.Uint64()} {
		if r == first {
			t.Fatal("different index, seed or name must give a different stream")
		}
	}
	p1 := Perm(SubStream(1, "perm", 0), 10)
	p2 := Perm(SubStream(1, "perm", 0), 10)
	if !reflect.DeepEqual(p1, p2) || len(p1) != 10 {
		t.Fatal("perm")
	}
	seen := map[int]bool{}
	for _, v := range p1 {
		seen[v] = true
	}
	if len(seen) != 10 {
		t.Fatal("perm is not a permutation")
	}
}
