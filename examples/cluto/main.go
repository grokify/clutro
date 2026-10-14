// Command cluto shows the migration path from CLUTO: read a matrix in
// CLUTO's sparse format, cluster it with CLUTO's defaults, write a
// .clustering file, and score it against a reference clustering — the same
// steps as `clutro vcluster` and `clutro evaluate`, from Go.
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/cluto"
	"github.com/grokify/clutro/evaluate"
)

// A tiny CLUTO sparse matrix: "rows cols nnz", then "col val" pairs with
// 1-based columns. Rows 1–3 share columns 1–2, rows 4–6 share columns 3–4.
const mat = `6 4 12
1 1 2 1
1 2 2 1
1 1 2 2
3 1 4 1
3 2 4 1
3 1 4 2
`

func main() {
	dir, err := os.MkdirTemp("", "clutro-cluto-example")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "docs.mat")
	if err := os.WriteFile(path, []byte(mat), 0o600); err != nil {
		panic(err)
	}

	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	m, err := cluto.ReadSparse(f)
	f.Close()
	if err != nil {
		panic(err)
	}
	m.NormalizeRowsL2() // CLUTO normalizes rows; -rowmodel l2

	res, err := clutro.Fit(context.Background(), m, clutro.WithClusters(2), clutro.WithSeed(1))
	if err != nil {
		panic(err)
	}
	out, err := os.Create(path + ".clustering.2")
	if err != nil {
		panic(err)
	}
	if err := cluto.WriteClustering(out, res.Labels); err != nil {
		panic(err)
	}
	out.Close()

	// Compare with a reference clustering, as evaluate does with .rclass files.
	ref, _, err := cluto.ReadClustering(strings.NewReader("0\n0\n0\n1\n1\n1\n"))
	if err != nil {
		panic(err)
	}
	ari, err := evaluate.AdjustedRandIndex(res.Labels, ref)
	if err != nil {
		panic(err)
	}
	fmt.Printf("labels %v\nARI vs reference %.2f\n", res.Labels, ari)
}
