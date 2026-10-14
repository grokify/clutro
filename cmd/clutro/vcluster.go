package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/cluto"
	"github.com/grokify/clutro/criterion"
	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
	"github.com/grokify/clutro/partition"
)

// jsonResult is the -json output shape.
type jsonResult struct {
	Matrix    string  `json:"matrix"`
	Rows      int     `json:"rows"`
	Cols      int     `json:"cols"`
	K         int     `json:"k"`
	Method    string  `json:"method"`
	Criterion string  `json:"criterion"`
	Value     float64 `json:"value"`
	Sizes     []int   `json:"sizes"`
	Labels    []int32 `json:"labels"`
}

func runVcluster(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("clutro vcluster", flag.ContinueOnError)
	fs.SetOutput(stderr)
	clmethod := fs.String("clmethod", "rb", "clustering method: rb, rbr, direct, kmeans")
	crfun := fs.String("crfun", "i2", "criterion function: i2")
	sim := fs.String("sim", "cos", "similarity: cos, euclid")
	cstype := fs.String("cstype", "best", "cluster to bisect next: best, largest")
	rowmodel := fs.String("rowmodel", "none", "row scaling before clustering: none, l2")
	colmodel := fs.String("colmodel", "none", "column weighting before clustering: none, idf")
	ntrials := fs.Int("ntrials", 10, "seeded starts per k-means or bisection")
	niter := fs.Int("niter", 20, "maximum k-means iterations and refinement passes")
	seed := fs.Uint64("seed", 0, "seed for every random choice")
	parallel := fs.Int("parallel", 0, "worker count (0 = all CPUs); results do not depend on it")
	clustfile := fs.String("clustfile", "", "output clustering file (default <matrix>.clustering.<k>)")
	jsonOut := fs.Bool("json", false, "print the result as JSON to stdout instead of writing a clustering file")
	showTree := fs.Bool("showtree", false, "print the bisection tree (rb/rbr)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: clutro vcluster [flags] <matrix> <k>")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		fs.Usage()
		return errors.New("vcluster needs a matrix file and k")
	}
	path := fs.Arg(0)
	k, err := strconv.Atoi(fs.Arg(1))
	if err != nil || k < 1 {
		return fmt.Errorf("k must be a positive integer, got %q", fs.Arg(1))
	}

	alg, refine, ok := clutro.ParseAlgorithm(*clmethod)
	if !ok {
		return fmt.Errorf("unknown clmethod %q", *clmethod)
	}
	crit, ok := criterion.Parse(*crfun)
	if !ok {
		return fmt.Errorf("unknown crfun %q", *crfun)
	}
	similarity, ok := metric.Parse(*sim)
	if !ok {
		return fmt.Errorf("unknown sim %q", *sim)
	}
	var sel partition.Selection
	switch *cstype {
	case "best":
		sel = partition.SelectBest
	case "largest":
		sel = partition.SelectLargest
	default:
		return fmt.Errorf("unknown cstype %q", *cstype)
	}
	rowModel, err := matrix.ParseRowModel(*rowmodel)
	if err != nil {
		return err
	}
	colModel, err := matrix.ParseColModel(*colmodel)
	if err != nil {
		return err
	}

	m, err := readMatrix(path)
	if err != nil {
		return err
	}
	switch mm := m.(type) {
	case *matrix.CSR:
		if err := matrix.Preprocess(mm, rowModel, colModel); err != nil {
			return err
		}
	case *matrix.Dense:
		if colModel != matrix.ColNone {
			return errors.New("colmodel applies to sparse matrices only")
		}
		if rowModel == matrix.RowL2 {
			mm.NormalizeRowsL2()
		}
	}

	res, err := clutro.Fit(context.Background(), m,
		clutro.WithAlgorithm(alg), clutro.WithClusters(k), clutro.WithRefinement(refine),
		clutro.WithCriterion(crit), clutro.WithSimilarity(similarity), clutro.WithSelection(sel),
		clutro.WithTrials(*ntrials), clutro.WithMaxIter(*niter), clutro.WithSeed(*seed),
		clutro.WithParallelism(*parallel))
	if err != nil {
		return err
	}

	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(jsonResult{Matrix: path, Rows: m.NumRows(), Cols: m.NumCols(), K: res.K, Method: *clmethod,
			Criterion: crit.Name(), Value: res.Criterion, Sizes: res.Sizes(), Labels: res.Labels})
	}
	out := *clustfile
	if out == "" {
		out = fmt.Sprintf("%s.clustering.%d", path, k)
	}
	if err := writeClustering(out, res.Labels); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "%s: %d rows, %d cols, %d clusters, %s=%.4f, method=%s sim=%s\n",
		path, m.NumRows(), m.NumCols(), res.K, crit.Name(), res.Criterion, *clmethod, similarity.Name())
	fmt.Fprintf(stdout, "sizes: %v\n", res.Sizes())
	fmt.Fprintf(stdout, "wrote %s\n", out)
	if *showTree && res.Tree() != nil {
		printTree(stdout, res.Tree().Root, 0)
	}
	return nil
}

func printTree(w io.Writer, n *partition.Node, depth int) {
	if n == nil {
		return
	}
	label := ""
	if n.Label >= 0 {
		label = fmt.Sprintf(" → cluster %d", n.Label)
	}
	fmt.Fprintf(w, "%*s%d rows, %.3f%s\n", depth*2, "", len(n.Rows), n.Criterion, label)
	printTree(w, n.Left, depth+1)
	printTree(w, n.Right, depth+1)
}

func readMatrix(path string) (matrix.Rows, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return cluto.ReadMatrix(f)
}

func writeClustering(path string, labels []int32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := cluto.WriteClustering(f, labels); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
