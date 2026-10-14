package matrix

import (
	"fmt"
	"math"
)

// RowModel is a per-row scaling applied before clustering (CLUTO -rowmodel).
type RowModel int

const (
	// RowNone leaves rows as they are.
	RowNone RowModel = iota
	// RowL2 scales every row to unit length.
	RowL2
)

// ColModel is a per-column weighting applied before clustering (CLUTO
// -colmodel).
type ColModel int

const (
	// ColNone leaves columns as they are.
	ColNone ColModel = iota
	// ColIDF multiplies each column by log(N / df), the inverse document
	// frequency; columns present in every row are zeroed.
	ColIDF
)

// ParseRowModel maps "none" or "l2".
func ParseRowModel(s string) (RowModel, error) {
	switch s {
	case "none", "":
		return RowNone, nil
	case "l2", "norm":
		return RowL2, nil
	}
	return RowNone, fmt.Errorf("matrix: unknown row model %q", s)
}

// ParseColModel maps "none" or "idf".
func ParseColModel(s string) (ColModel, error) {
	switch s {
	case "none", "":
		return ColNone, nil
	case "idf":
		return ColIDF, nil
	}
	return ColNone, fmt.Errorf("matrix: unknown column model %q", s)
}

// IDFWeights returns log(N/df) per column for a CSR with N rows, 0 for
// columns that appear in no row.
func IDFWeights(m *CSR) []float32 {
	n := float64(m.NumRows())
	w := make([]float32, m.NumCols())
	for j, df := range m.ColumnDocFreq() {
		if df > 0 {
			w[j] = float32(math.Log(n / float64(df)))
		}
	}
	return w
}

// Preprocess applies the column model then the row model in place, in the
// order CLUTO does (column scaling first, then row normalization).
func Preprocess(m *CSR, row RowModel, col ColModel) error {
	if col == ColIDF {
		if err := m.ScaleColumns(IDFWeights(m)); err != nil {
			return err
		}
	}
	if row == RowL2 {
		m.NormalizeRowsL2()
	}
	return nil
}
