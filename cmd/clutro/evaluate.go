package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/grokify/clutro/cluto"
	"github.com/grokify/clutro/evaluate"
	"github.com/grokify/clutro/golden"
	"github.com/grokify/clutro/matrix"
)

func runEvaluate(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("clutro evaluate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rclass := fs.Bool("rclass", false, "treat <truth> as an .rclass file of one class name per line (implied by the .rclass extension)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: clutro evaluate [flags] <clustering> <truth>")
		fmt.Fprintln(stderr, "prints adjusted Rand index, NMI, purity and entropy of <clustering> against <truth>,")
		fmt.Fprintln(stderr, "another clustering file or an .rclass file")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		fs.Usage()
		return errors.New("evaluate needs a clustering file and a truth file")
	}
	got, err := readClustering(fs.Arg(0))
	if err != nil {
		return err
	}
	var truth []int32
	if *rclass || strings.HasSuffix(fs.Arg(1), ".rclass") {
		classes, err := readLabels(fs.Arg(1))
		if err != nil {
			return err
		}
		truth = cluto.ClassLabels(classes)
	} else if truth, err = readClustering(fs.Arg(1)); err != nil {
		return err
	}
	ari, err := evaluate.AdjustedRandIndex(got, truth)
	if err != nil {
		return err
	}
	nmi, _ := evaluate.NormalizedMutualInformation(got, truth)
	pur, _ := evaluate.Purity(got, truth)
	ent, _ := evaluate.Entropy(got, truth)
	fmt.Fprintf(stdout, "rows %d\nARI %.4f\nNMI %.4f\npurity %.4f\nentropy %.4f\n", len(got), ari, nmi, pur, ent)
	return nil
}

func runGolden(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("clutro golden", flag.ContinueOnError)
	fs.SetOutput(stderr)
	seed := fs.Uint64("seed", 0, "seed")
	trials := fs.Int("ntrials", 10, "seeded starts per bisection")
	workers := fs.Int("parallel", 0, "worker count (0 = all CPUs)")
	refine := fs.Bool("refine", false, "refine after bisection (rbr)")
	rowmodel := fs.String("rowmodel", "l2", "row scaling: none, l2")
	colmodel := fs.String("colmodel", "none", "column weighting: none, idf")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: clutro golden [flags] <dir>")
		fmt.Fprintln(stderr, "re-clusters every <name> with a <name>.clustering.<k> reference in <dir> and compares them")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return errors.New("golden needs a directory")
	}
	rm, err := matrix.ParseRowModel(*rowmodel)
	if err != nil {
		return err
	}
	cm, err := matrix.ParseColModel(*colmodel)
	if err != nil {
		return err
	}
	corpora, err := golden.Find(fs.Arg(0))
	if err != nil {
		return err
	}
	if len(corpora) == 0 {
		return fmt.Errorf("no <name>.clustering.<k> files with a matching matrix in %s", fs.Arg(0))
	}
	rs, skips, err := golden.RunAll(context.Background(), corpora, golden.Options{Seed: *seed, Trials: *trials, Refine: *refine, RowModel: rm, ColModel: cm, Workers: *workers})
	if err != nil {
		return err
	}
	golden.WriteTable(stdout, rs)
	for _, s := range skips {
		fmt.Fprintf(stderr, "skipped %s: %s\n", filepath.Base(s.RefPath), s.Reason)
	}
	if len(rs) == 0 {
		return errors.New("no corpus could be run")
	}
	return nil
}

func readClustering(path string) ([]int32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	labels, _, err := cluto.ReadClustering(f)
	return labels, err
}

func readLabels(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return cluto.ReadLabels(f)
}
