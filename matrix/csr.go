package matrix

import (
	"fmt"
	"math"
	"sort"
)

// CSR is a compressed sparse row matrix.
type CSR struct {
	rowPtr []uint32
	idx    []uint32
	val    []float32
	norms  []float32
	nCols  int
}

// Triplet is one entry for NewCSRFromTriplets.
type Triplet struct {
	Row, Col int
	Val      float32
}

// NewCSRFromRows builds a CSR from per-row sparse vectors, which must have
// sorted unique indices below nCols. The rows are copied.
func NewCSRFromRows(nCols int, rows []SparseVec) (*CSR, error) {
	if nCols < 0 || nCols > math.MaxUint32 {
		return nil, fmt.Errorf("%w: column count %d outside [0, 2^32)", ErrShape, nCols)
	}
	nnz := 0
	for i, r := range rows {
		if err := checkRow(nCols, r); err != nil {
			return nil, fmt.Errorf("row %d: %w", i, err)
		}
		nnz += len(r.Idx)
	}
	if nnz > math.MaxUint32 {
		return nil, fmt.Errorf("%w: %d nonzeros exceed the 32-bit row pointer range", ErrShape, nnz)
	}
	m := &CSR{
		rowPtr: make([]uint32, len(rows)+1),
		idx:    make([]uint32, 0, nnz),
		val:    make([]float32, 0, nnz),
		norms:  make([]float32, len(rows)),
		nCols:  nCols,
	}
	for i, r := range rows {
		m.idx = append(m.idx, r.Idx...)
		m.val = append(m.val, r.Val...)
		m.rowPtr[i+1] = uint32(len(m.idx)) //nolint:gosec // nnz <= MaxUint32 checked above
		m.norms[i] = float32(r.Norm())
	}
	return m, nil
}

// NewCSRFromTriplets builds a CSR from (row, col, val) entries in any order.
// Duplicate (row, col) pairs are summed; zero values are dropped.
func NewCSRFromTriplets(nRows, nCols int, ts []Triplet) (*CSR, error) {
	if nRows < 0 || nCols < 0 {
		return nil, fmt.Errorf("%w: negative size", ErrShape)
	}
	sorted := make([]Triplet, len(ts))
	copy(sorted, ts)
	for _, t := range sorted {
		if t.Row < 0 || t.Row >= nRows || t.Col < 0 || t.Col >= nCols {
			return nil, fmt.Errorf("%w: entry (%d,%d) outside %dx%d", ErrShape, t.Row, t.Col, nRows, nCols)
		}
	}
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Row != sorted[j].Row {
			return sorted[i].Row < sorted[j].Row
		}
		return sorted[i].Col < sorted[j].Col
	})
	rows := make([]SparseVec, nRows)
	for i := 0; i < len(sorted); {
		j := i
		var sum float64
		for j < len(sorted) && sorted[j].Row == sorted[i].Row && sorted[j].Col == sorted[i].Col {
			sum += float64(sorted[j].Val)
			j++
		}
		if sum != 0 {
			r := &rows[sorted[i].Row]
			r.Idx = append(r.Idx, uint32(sorted[i].Col)) //nolint:gosec // 0 <= Col < nCols, checked above
			r.Val = append(r.Val, float32(sum))
		}
		i = j
	}
	return NewCSRFromRows(nCols, rows)
}

// NumRows implements Rows.
func (m *CSR) NumRows() int { return len(m.rowPtr) - 1 }

// NumCols implements Rows.
func (m *CSR) NumCols() int { return m.nCols }

// NNZ is the number of stored entries.
func (m *CSR) NNZ() int { return len(m.idx) }

// Row implements Rows; the returned vector is a view into the matrix.
func (m *CSR) Row(i int) SparseVec {
	return SparseVec{Idx: m.idx[m.rowPtr[i]:m.rowPtr[i+1]], Val: m.val[m.rowPtr[i]:m.rowPtr[i+1]]}
}

// RowNorm implements Rows.
func (m *CSR) RowNorm(i int) float32 { return m.norms[i] }

// NormalizeRowsL2 scales every row to unit length in place. Zero rows are
// left as they are.
func (m *CSR) NormalizeRowsL2() {
	for i := range m.norms {
		n := m.norms[i]
		if n == 0 || n == 1 {
			continue
		}
		inv := float32(1 / float64(n))
		for k := m.rowPtr[i]; k < m.rowPtr[i+1]; k++ {
			m.val[k] *= inv
		}
		m.norms[i] = float32(m.Row(i).Norm())
	}
}

// ScaleColumns multiplies every entry by a per-column weight (for example an
// IDF vector) and recomputes row norms.
func (m *CSR) ScaleColumns(w []float32) error {
	if len(w) != m.nCols {
		return fmt.Errorf("%w: %d weights for %d columns", ErrShape, len(w), m.nCols)
	}
	for k, idx := range m.idx {
		m.val[k] *= w[idx]
	}
	for i := range m.norms {
		m.norms[i] = float32(m.Row(i).Norm())
	}
	return nil
}

// ColumnDocFreq counts, per column, the rows with a nonzero entry.
func (m *CSR) ColumnDocFreq() []int {
	df := make([]int, m.nCols)
	for k, idx := range m.idx {
		if m.val[k] != 0 {
			df[idx]++
		}
	}
	return df
}

// Dense is a row-major dense matrix. Row views share one index slice.
type Dense struct {
	data  []float32
	norms []float32
	index []uint32
	nCols int
}

// NewDense wraps row-major data of nRows*nCols values (copied).
func NewDense(nRows, nCols int, data []float32) (*Dense, error) {
	if nRows < 0 || nCols < 0 || len(data) != nRows*nCols {
		return nil, fmt.Errorf("%w: %d values for %dx%d", ErrShape, len(data), nRows, nCols)
	}
	d := &Dense{data: append([]float32(nil), data...), norms: make([]float32, nRows), index: make([]uint32, nCols), nCols: nCols}
	for j := range d.index {
		d.index[j] = uint32(j)
	}
	for i := 0; i < nRows; i++ {
		d.norms[i] = float32(d.Row(i).Norm())
	}
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("%w: non-finite value", ErrShape)
		}
	}
	return d, nil
}

// NumRows implements Rows.
func (d *Dense) NumRows() int {
	if d.nCols == 0 {
		return 0
	}
	return len(d.data) / d.nCols
}

// NumCols implements Rows.
func (d *Dense) NumCols() int { return d.nCols }

// Row implements Rows; the view's Idx is the shared 0..nCols-1 slice.
func (d *Dense) Row(i int) SparseVec {
	return SparseVec{Idx: d.index, Val: d.data[i*d.nCols : (i+1)*d.nCols]}
}

// RowNorm implements Rows.
func (d *Dense) RowNorm(i int) float32 { return d.norms[i] }

// NormalizeRowsL2 scales every row to unit length in place.
func (d *Dense) NormalizeRowsL2() {
	for i := range d.norms {
		n := d.norms[i]
		if n == 0 || n == 1 {
			continue
		}
		inv := float32(1 / float64(n))
		row := d.data[i*d.nCols : (i+1)*d.nCols]
		for k := range row {
			row[k] *= inv
		}
		d.norms[i] = float32(d.Row(i).Norm())
	}
}
