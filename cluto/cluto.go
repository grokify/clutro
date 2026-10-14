// Package cluto reads and writes CLUTO's file formats so existing CLUTO
// workflows and reference outputs can be used directly:
//
//   - sparse matrix: a header line "nrows ncols nnz" followed by one line per
//     row of "col val" pairs with 1-based column indices;
//   - dense matrix: a header line "nrows ncols" followed by one line per row
//     of ncols values;
//   - label files (.clabel, .rlabel, .rclass): one label per line;
//   - clustering files (.clustering.<k>): one 0-based cluster id per row,
//     -1 for an unassigned row.
//
// Readers stream line by line and validate shape; writers are
// deterministic.
package cluto

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"

	"github.com/grokify/clutro/matrix"
)

// ErrFormat reports malformed input.
var ErrFormat = errors.New("cluto: malformed file")

type entry struct {
	col uint32
	val float32
}

// ReadMatrix parses a CLUTO matrix file, sparse or dense, by its header.
func ReadMatrix(r io.Reader) (matrix.Rows, error) {
	sc := scanner(r)
	header, ok := nextLine(sc)
	if !ok {
		return nil, fmt.Errorf("%w: missing header", ErrFormat)
	}
	fields := strings.Fields(header)
	switch len(fields) {
	case 3:
		return readSparse(sc, fields)
	case 2:
		return readDense(sc, fields)
	}
	return nil, fmt.Errorf("%w: header must be \"nrows ncols [nnz]\", got %q", ErrFormat, header)
}

// ReadSparse parses a sparse matrix file into a CSR.
func ReadSparse(r io.Reader) (*matrix.CSR, error) {
	m, err := ReadMatrix(r)
	if err != nil {
		return nil, err
	}
	csr, ok := m.(*matrix.CSR)
	if !ok {
		return nil, fmt.Errorf("%w: expected a sparse matrix header", ErrFormat)
	}
	return csr, nil
}

func readSparse(sc *bufio.Scanner, fields []string) (*matrix.CSR, error) {
	nRows, nCols, nnz, err := ints3(fields)
	if err != nil {
		return nil, err
	}
	rows := make([]matrix.SparseVec, nRows)
	total := 0
	for i := 0; i < nRows; i++ {
		// One line per row; a blank line is an empty row.
		if !sc.Scan() {
			return nil, fmt.Errorf("%w: expected %d rows, found %d", ErrFormat, nRows, i)
		}
		f := strings.Fields(sc.Text())
		if len(f)%2 != 0 {
			return nil, fmt.Errorf("%w: row %d has an odd number of fields", ErrFormat, i+1)
		}
		entries := make([]entry, 0, len(f)/2)
		for j := 0; j < len(f); j += 2 {
			col, err := strconv.Atoi(f[j])
			if err != nil || col < 1 || col > nCols {
				return nil, fmt.Errorf("%w: row %d: bad column %q", ErrFormat, i+1, f[j])
			}
			v, err := strconv.ParseFloat(f[j+1], 32)
			if err != nil {
				return nil, fmt.Errorf("%w: row %d: bad value %q", ErrFormat, i+1, f[j+1])
			}
			entries = append(entries, entry{uint32(col - 1), float32(v)}) //nolint:gosec // 1 <= col <= nCols, and nCols is range-checked by NewCSRFromRows
		}
		// CLUTO does not require column order within a row; CSR does.
		sort.Slice(entries, func(a, b int) bool { return entries[a].col < entries[b].col })
		row := matrix.SparseVec{Idx: make([]uint32, 0, len(entries)), Val: make([]float32, 0, len(entries))}
		for j, e := range entries {
			if j > 0 && e.col == entries[j-1].col {
				return nil, fmt.Errorf("%w: row %d: column %d repeated", ErrFormat, i+1, e.col+1)
			}
			row.Idx = append(row.Idx, e.col)
			row.Val = append(row.Val, e.val)
		}
		total += len(row.Idx)
		rows[i] = row
	}
	if total != nnz {
		return nil, fmt.Errorf("%w: header declares %d nonzeros, found %d", ErrFormat, nnz, total)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return matrix.NewCSRFromRows(nCols, rows)
}

func readDense(sc *bufio.Scanner, fields []string) (*matrix.Dense, error) {
	nRows, err := strconv.Atoi(fields[0])
	if err != nil || nRows < 0 {
		return nil, fmt.Errorf("%w: bad row count %q", ErrFormat, fields[0])
	}
	nCols, err := strconv.Atoi(fields[1])
	if err != nil || nCols < 0 {
		return nil, fmt.Errorf("%w: bad column count %q", ErrFormat, fields[1])
	}
	data := make([]float32, 0, nRows*nCols)
	for i := 0; i < nRows; i++ {
		if !sc.Scan() {
			return nil, fmt.Errorf("%w: expected %d rows, found %d", ErrFormat, nRows, i)
		}
		f := strings.Fields(sc.Text())
		if len(f) != nCols {
			return nil, fmt.Errorf("%w: row %d has %d values, want %d", ErrFormat, i+1, len(f), nCols)
		}
		for _, s := range f {
			v, err := strconv.ParseFloat(s, 32)
			if err != nil {
				return nil, fmt.Errorf("%w: row %d: bad value %q", ErrFormat, i+1, s)
			}
			data = append(data, float32(v))
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return matrix.NewDense(nRows, nCols, data)
}

// WriteSparse writes a CSR in CLUTO sparse format with 1-based columns.
func WriteSparse(w io.Writer, m *matrix.CSR) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%d %d %d\n", m.NumRows(), m.NumCols(), m.NNZ())
	for i := 0; i < m.NumRows(); i++ {
		r := m.Row(i)
		for k, idx := range r.Idx {
			if k > 0 {
				fmt.Fprint(bw, " ")
			}
			fmt.Fprintf(bw, "%d %s", idx+1, strconv.FormatFloat(float64(r.Val[k]), 'g', -1, 32))
		}
		fmt.Fprint(bw, "\n")
	}
	return bw.Flush()
}

// WriteDense writes a Dense matrix in CLUTO dense format.
func WriteDense(w io.Writer, m *matrix.Dense) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%d %d\n", m.NumRows(), m.NumCols())
	for i := 0; i < m.NumRows(); i++ {
		r := m.Row(i)
		for k, v := range r.Val {
			if k > 0 {
				fmt.Fprint(bw, " ")
			}
			fmt.Fprint(bw, strconv.FormatFloat(float64(v), 'g', -1, 32))
		}
		fmt.Fprint(bw, "\n")
	}
	return bw.Flush()
}

// ReadLabels reads one label per line (.clabel, .rlabel, .rclass).
func ReadLabels(r io.Reader) ([]string, error) {
	sc := scanner(r)
	var out []string
	for sc.Scan() {
		out = append(out, strings.TrimSpace(sc.Text()))
	}
	return out, sc.Err()
}

// WriteLabels writes one label per line.
func WriteLabels(w io.Writer, labels []string) error {
	bw := bufio.NewWriter(w)
	for _, l := range labels {
		if strings.ContainsAny(l, "\r\n") {
			return fmt.Errorf("%w: label %q contains a newline", ErrFormat, l)
		}
		fmt.Fprintln(bw, l)
	}
	return bw.Flush()
}

// ReadClustering reads a .clustering file: one integer per line, -1 for an
// unassigned row. It returns the labels and the number of clusters (max
// label + 1).
func ReadClustering(r io.Reader) ([]int32, int, error) {
	sc := scanner(r)
	var labels []int32
	k := 0
	for sc.Scan() {
		s := strings.TrimSpace(sc.Text())
		if s == "" {
			continue
		}
		v, err := strconv.ParseInt(s, 10, 32)
		if err != nil || v < -1 {
			return nil, 0, fmt.Errorf("%w: bad cluster id %q on line %d", ErrFormat, s, len(labels)+1)
		}
		labels = append(labels, int32(v))
		if int(v)+1 > k {
			k = int(v) + 1
		}
	}
	return labels, k, sc.Err()
}

// WriteClustering writes one cluster id per line.
func WriteClustering(w io.Writer, labels []int32) error {
	bw := bufio.NewWriter(w)
	for _, l := range labels {
		fmt.Fprintln(bw, l)
	}
	return bw.Flush()
}

// ClassLabels maps string class labels (an .rclass file) to dense int32
// ids in order of first appearance, for use with evaluate.
func ClassLabels(classes []string) []int32 {
	ids := map[string]int32{}
	out := make([]int32, len(classes))
	for i, c := range classes {
		id, ok := ids[c]
		if !ok {
			id = int32(len(ids)) //nolint:gosec // distinct classes never approach 2^31
			ids[c] = id
		}
		out[i] = id
	}
	return out
}

func scanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 64*1024*1024)
	return sc
}

// nextLine returns the next non-blank line.
func nextLine(sc *bufio.Scanner) (string, bool) {
	for sc.Scan() {
		if t := strings.TrimSpace(sc.Text()); t != "" {
			return t, true
		}
	}
	return "", false
}

func ints3(f []string) (int, int, int, error) {
	var out [3]int
	for i, s := range f {
		v, err := strconv.Atoi(s)
		if err != nil || v < 0 {
			return 0, 0, 0, fmt.Errorf("%w: bad header field %q", ErrFormat, s)
		}
		out[i] = v
	}
	return out[0], out[1], out[2], nil
}
