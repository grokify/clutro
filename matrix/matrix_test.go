package matrix

import (
	"math"
	"testing"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestCSRFromTripletsAndRows(t *testing.T) {
	m, err := NewCSRFromTriplets(3, 4, []Triplet{
		{0, 3, 1}, {0, 1, 2}, {1, 0, 1}, {1, 0, 2}, // duplicate summed
		{2, 2, 0}, // zero dropped
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRows() != 3 || m.NumCols() != 4 || m.NNZ() != 3 {
		t.Fatalf("shape %dx%d nnz %d", m.NumRows(), m.NumCols(), m.NNZ())
	}
	r0 := m.Row(0)
	if r0.Idx[0] != 1 || r0.Idx[1] != 3 || r0.Val[0] != 2 || r0.Val[1] != 1 {
		t.Fatalf("row 0 sorted: %+v", r0)
	}
	if r1 := m.Row(1); r1.Val[0] != 3 {
		t.Fatalf("duplicates summed: %+v", r1)
	}
	if m.Row(2).NNZ() != 0 || m.RowNorm(2) != 0 {
		t.Fatal("zero row")
	}
	if !approx(float64(m.RowNorm(0)), math.Sqrt(5), 1e-6) {
		t.Fatalf("norm %v", m.RowNorm(0))
	}
	m.NormalizeRowsL2()
	if !approx(float64(m.RowNorm(0)), 1, 1e-6) || !approx(float64(m.Row(1).Val[0]), 1, 1e-6) {
		t.Fatal("normalize")
	}
	df := m.ColumnDocFreq()
	if df[0] != 1 || df[1] != 1 || df[2] != 0 || df[3] != 1 {
		t.Fatalf("df %v", df)
	}
	if err := m.ScaleColumns([]float32{1, 2, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if !approx(float64(m.Row(0).Val[0]), 2*2/math.Sqrt(5), 1e-6) {
		t.Fatal("scale columns")
	}
	if err := m.ScaleColumns([]float32{1}); err == nil {
		t.Fatal("wrong weight length")
	}
}

func TestCSRErrors(t *testing.T) {
	if _, err := NewCSRFromTriplets(1, 1, []Triplet{{0, 5, 1}}); err == nil {
		t.Fatal("out of range")
	}
	if _, err := NewCSRFromRows(3, []SparseVec{{Idx: []uint32{2, 1}, Val: []float32{1, 1}}}); err == nil {
		t.Fatal("unsorted")
	}
	if _, err := NewCSRFromRows(3, []SparseVec{{Idx: []uint32{1}, Val: []float32{1, 2}}}); err == nil {
		t.Fatal("length mismatch")
	}
	if _, err := NewCSRFromRows(3, []SparseVec{{Idx: []uint32{1}, Val: []float32{float32(math.NaN())}}}); err == nil {
		t.Fatal("nan")
	}
}

func TestDense(t *testing.T) {
	d, err := NewDense(2, 3, []float32{3, 0, 4, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if d.NumRows() != 2 || d.NumCols() != 3 || d.RowNorm(0) != 5 || d.RowNorm(1) != 0 {
		t.Fatal("dense shape/norms")
	}
	r := d.Row(0)
	if len(r.Idx) != 3 || r.Idx[2] != 2 || r.Val[2] != 4 {
		t.Fatalf("dense row view: %+v", r)
	}
	if !approx(Dot(d.Row(0), d.Row(0)), 25, 1e-9) {
		t.Fatal("dot of dense views")
	}
	d.NormalizeRowsL2()
	if !approx(float64(d.RowNorm(0)), 1, 1e-6) {
		t.Fatal("dense normalize")
	}
	if _, err := NewDense(2, 2, []float32{1}); err == nil {
		t.Fatal("size mismatch")
	}
}

func TestDotAndComposite(t *testing.T) {
	a := SparseVec{Idx: []uint32{0, 2, 5}, Val: []float32{1, 2, 3}}
	b := SparseVec{Idx: []uint32{2, 3, 5}, Val: []float32{1, 1, 1}}
	if Dot(a, b) != 5 {
		t.Fatal("merge-join dot")
	}
	c := NewComposite(6)
	c.Add(a, float32(a.Norm()))
	c.Add(b, float32(b.Norm()))
	// Row norms are passed as float32, so the maintained norm is accurate
	// to single precision, not double.
	const tol = 1e-6
	want := math.Sqrt(1 + 9 + 1 + 16) // sum = [1 0 3 1 0 4]
	if !approx(c.Norm(), want, tol) || c.N != 2 {
		t.Fatalf("composite norm %v want %v", c.Norm(), want)
	}
	if !approx(c.NormIfRemoved(b, float32(b.Norm())), a.Norm(), tol) {
		t.Fatal("NormIfRemoved")
	}
	if !approx(c.NormIfAdded(a, float32(a.Norm())), math.Sqrt(4+25+1+49), tol) {
		t.Fatal("NormIfAdded")
	}
	c.Remove(b, float32(b.Norm()))
	if !approx(c.Norm(), a.Norm(), tol) || c.N != 1 {
		t.Fatal("remove")
	}
	c.Remove(a, float32(a.Norm()))
	if c.N != 0 || c.Norm() != 0 || c.Sum[2] != 0 {
		t.Fatal("empty composite is exactly zero")
	}
	c.Add(a, float32(a.Norm()))
	c.Recompute()
	if !approx(c.Norm(), a.Norm(), 1e-9) {
		t.Fatal("recompute")
	}
}

// TestCompositePrecision checks that float32 storage with float64
// accumulation keeps the incrementally maintained norm within 1e-6 relative
// error of a from-scratch float64 computation after many adds and removes.
func TestCompositePrecision(t *testing.T) {
	const nRows, nCols, nnz = 20000, 5000, 40
	rows := make([]SparseVec, nRows)
	x := uint64(12345)
	next := func() uint64 { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x }
	for i := range rows {
		seen := map[uint32]bool{}
		for len(seen) < nnz {
			seen[uint32(next()%nCols)] = true
		}
		idx := make([]uint32, 0, nnz)
		for k := range seen {
			idx = append(idx, k)
		}
		sortU32(idx)
		val := make([]float32, nnz)
		for k := range val {
			val[k] = float32(next()%1000)/1000 + 0.001
		}
		rows[i] = SparseVec{Idx: idx, Val: val}
	}
	m, err := NewCSRFromRows(nCols, rows)
	if err != nil {
		t.Fatal(err)
	}
	m.NormalizeRowsL2()
	c := NewComposite(nCols)
	for i := 0; i < nRows; i++ {
		c.Add(m.Row(i), m.RowNorm(i))
	}
	for i := 0; i < nRows; i += 2 {
		c.Remove(m.Row(i), m.RowNorm(i))
	}
	incremental := c.Norm()
	c.Recompute()
	exact := c.Norm()
	if rel := math.Abs(incremental-exact) / exact; rel > 1e-6 {
		t.Fatalf("incremental norm drifted: rel err %g (%v vs %v)", rel, incremental, exact)
	}
}

func sortU32(a []uint32) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j] < a[j-1]; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}

func BenchmarkDot40(b *testing.B) {
	a := SparseVec{Idx: make([]uint32, 40), Val: make([]float32, 40)}
	c := SparseVec{Idx: make([]uint32, 40), Val: make([]float32, 40)}
	for i := range a.Idx {
		a.Idx[i], c.Idx[i] = uint32(i*7), uint32(i*5)
		a.Val[i], c.Val[i] = 1, 1
	}
	for b.Loop() {
		Dot(a, c)
	}
}

func BenchmarkCompositeAdd(b *testing.B) {
	r := SparseVec{Idx: make([]uint32, 100), Val: make([]float32, 100)}
	for i := range r.Idx {
		r.Idx[i], r.Val[i] = uint32(i*13), 0.1
	}
	c := NewComposite(2000)
	n := float32(r.Norm())
	for b.Loop() {
		c.Add(r, n)
	}
}
