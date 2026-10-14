package cluto

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"github.com/grokify/clutro/matrix"
)

// Four rows: the third is empty (a blank line), the fourth lists its
// columns out of order.
const sparse = `4 4 6
1 1 3 2
2 1.5

4 1 1 2 2 0.25
`

func TestReadSparse(t *testing.T) {
	m, err := ReadSparse(strings.NewReader(sparse))
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRows() != 4 || m.NumCols() != 4 || m.NNZ() != 6 {
		t.Fatalf("shape %dx%d nnz %d", m.NumRows(), m.NumCols(), m.NNZ())
	}
	r0 := m.Row(0)
	if !reflect.DeepEqual(r0.Idx, []uint32{0, 2}) || r0.Val[1] != 2 {
		t.Fatalf("row 0: %+v", r0)
	}
	if m.Row(2).NNZ() != 0 {
		t.Fatal("blank line is an empty row")
	}
	// Columns are stored 0-based and sorted; the file is 1-based and may be
	// in any order within a row.
	if r3 := m.Row(3); !reflect.DeepEqual(r3.Idx, []uint32{0, 1, 3}) || r3.Val[2] != 1 || r3.Val[1] != 0.25 {
		t.Fatalf("row 3: %+v", r3)
	}
}

func TestReadSparseRejectsBadInput(t *testing.T) {
	bad := []string{
		"1 4 2\n3 1 3 1\n", // repeated column
		"1 4 2\n1 1\n",     // nnz mismatch
		"2 4 1\n1 1\n",     // missing row
		"1 4 1\n5 1\n",     // column out of range
		"1 4 1\n1\n",       // odd fields
		"1 4 1\nx 1\n",     // bad column
		"1 4 1\n1 y\n",     // bad value
		"a b c\n",          // bad header
		"1 2 3 4\n",        // too many header fields
		"",                 // empty
		"2 2\n1 2\n1\n",    // dense row too short
		"1 2\nq 1\n",       // dense bad value
		"-1 2 3\n",         // negative header
	}
	for _, in := range bad {
		if _, err := ReadMatrix(strings.NewReader(in)); err == nil {
			t.Errorf("accepted malformed input %q", in)
		}
	}
}

func TestSparseRoundTrip(t *testing.T) {
	m, err := ReadSparse(strings.NewReader(sparse))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := WriteSparse(&buf, m); err != nil {
		t.Fatal(err)
	}
	// Rows are written in column order; an empty row is an empty line.
	if got, want := buf.String(), "4 4 6\n1 1 3 2\n2 1.5\n\n1 2 2 0.25 4 1\n"; got != want {
		t.Fatalf("write:\n%s\nwant:\n%s", got, want)
	}
	back, err := ReadSparse(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < m.NumRows(); i++ {
		if !reflect.DeepEqual(back.Row(i), m.Row(i)) {
			t.Fatalf("row %d changed in round trip", i)
		}
	}
}

func TestDenseRoundTrip(t *testing.T) {
	in := "2 3\n1 2 3\n0.5 0 -1\n"
	m, err := ReadMatrix(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	d, ok := m.(*matrix.Dense)
	if !ok || d.NumRows() != 2 || d.NumCols() != 3 || d.Row(1).Val[2] != -1 {
		t.Fatalf("dense: %T %+v", m, m)
	}
	var buf bytes.Buffer
	if err := WriteDense(&buf, d); err != nil {
		t.Fatal(err)
	}
	if buf.String() != in {
		t.Fatalf("dense write:\n%s", buf.String())
	}
	if _, err := ReadSparse(strings.NewReader(in)); err == nil {
		t.Fatal("ReadSparse must reject a dense header")
	}
}

func TestLabelsAndClustering(t *testing.T) {
	labels, err := ReadLabels(strings.NewReader("alpha\n beta \ngamma\n"))
	if err != nil || !reflect.DeepEqual(labels, []string{"alpha", "beta", "gamma"}) {
		t.Fatalf("labels %v %v", labels, err)
	}
	var buf bytes.Buffer
	if err := WriteLabels(&buf, labels); err != nil || buf.String() != "alpha\nbeta\ngamma\n" {
		t.Fatal("write labels")
	}
	if err := WriteLabels(&buf, []string{"a\nb"}); err == nil {
		t.Fatal("newline in label")
	}
	cl, k, err := ReadClustering(strings.NewReader("0\n2\n\n-1\n1\n"))
	if err != nil || !reflect.DeepEqual(cl, []int32{0, 2, -1, 1}) || k != 3 {
		t.Fatalf("clustering %v %d %v", cl, k, err)
	}
	if _, _, err := ReadClustering(strings.NewReader("0\n-2\n")); err == nil {
		t.Fatal("bad id")
	}
	buf.Reset()
	if err := WriteClustering(&buf, cl); err != nil || buf.String() != "0\n2\n-1\n1\n" {
		t.Fatal("write clustering")
	}
	if got := ClassLabels([]string{"x", "y", "x", "z"}); !reflect.DeepEqual(got, []int32{0, 1, 0, 2}) {
		t.Fatalf("class labels %v", got)
	}
}
