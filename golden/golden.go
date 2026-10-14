// Package golden measures CLUTRO against reference clusterings: a directory
// of CLUTO corpora, each a matrix file with a reference ".clustering.<k>"
// file, is re-clustered with the same k and compared by adjusted Rand index
// and normalized mutual information. It is the validation harness for the
// published fixture subset in testdata/cluto and for private corpora.
package golden

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/cluto"
	"github.com/grokify/clutro/evaluate"
	"github.com/grokify/clutro/matrix"
)

// Corpus is one matrix with its reference clustering.
type Corpus struct {
	Name       string
	MatrixPath string
	RefPath    string
	K          int
}

var clusteringRe = regexp.MustCompile(`\.clustering\.(\d+)$`)

// Find lists the corpora in dir: every file "<name>.clustering.<k>" whose
// matrix "<name>" exists, sorted by name.
func Find(dir string) ([]Corpus, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var out []Corpus
	for _, e := range entries {
		m := clusteringRe.FindStringSubmatch(e.Name())
		if m == nil || e.IsDir() {
			continue
		}
		k, err := strconv.Atoi(m[1])
		if err != nil || k < 1 {
			continue
		}
		base := strings.TrimSuffix(e.Name(), m[0])
		matrixPath := filepath.Join(dir, base)
		if _, err := os.Stat(matrixPath); err != nil {
			continue
		}
		out = append(out, Corpus{Name: base, MatrixPath: matrixPath, RefPath: filepath.Join(dir, e.Name()), K: k})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Options configures a harness run.
type Options struct {
	Seed     uint64
	Trials   int
	Refine   bool
	RowModel matrix.RowModel
	ColModel matrix.ColModel
	Workers  int
}

// Result is the comparison for one corpus.
type Result struct {
	Corpus
	Rows, Cols, NNZ int
	ARI, NMI        float64
	// RefCriterion and Criterion are the I2 values of the reference and the
	// CLUTRO partition on the preprocessed matrix, for a sense of which is
	// the better optimum rather than just how similar they are.
	RefCriterion, Criterion float64
}

// Run clusters one corpus and compares it with the reference.
func Run(ctx context.Context, c Corpus, o Options) (Result, error) {
	f, err := os.Open(c.MatrixPath)
	if err != nil {
		return Result{}, err
	}
	m, err := cluto.ReadSparse(f)
	f.Close()
	if err != nil {
		return Result{}, fmt.Errorf("%s: %w", c.Name, err)
	}
	rf, err := os.Open(c.RefPath)
	if err != nil {
		return Result{}, err
	}
	ref, refK, err := cluto.ReadClustering(rf)
	rf.Close()
	if err != nil {
		return Result{}, fmt.Errorf("%s: %w", c.Name, err)
	}
	if len(ref) != m.NumRows() {
		return Result{}, fmt.Errorf("%s: reference has %d labels for %d rows", c.Name, len(ref), m.NumRows())
	}
	if err := matrix.Preprocess(m, o.RowModel, o.ColModel); err != nil {
		return Result{}, err
	}
	res, err := clutro.Fit(ctx, m,
		clutro.WithAlgorithm(clutro.RepeatedBisection),
		clutro.WithClusters(c.K),
		clutro.WithRefinement(o.Refine),
		clutro.WithSeed(o.Seed),
		clutro.WithTrials(o.Trials),
		clutro.WithParallelism(o.Workers),
	)
	if err != nil {
		return Result{}, fmt.Errorf("%s: %w", c.Name, err)
	}
	ari, err := evaluate.AdjustedRandIndex(res.Labels, ref)
	if err != nil {
		return Result{}, err
	}
	nmi, err := evaluate.NormalizedMutualInformation(res.Labels, ref)
	if err != nil {
		return Result{}, err
	}
	// Reference criterion: unassigned rows (-1) are placed in their own
	// cluster so the composite sums stay well defined.
	refLabels := append([]int32(nil), ref...)
	for i, l := range refLabels {
		if l < 0 {
			refLabels[i] = int32(refK) //nolint:gosec // k is small
		}
	}
	refComps := make([]*matrix.Composite, refK+1)
	for i := range refComps {
		refComps[i] = matrix.NewComposite(m.NumCols())
	}
	for i, l := range refLabels {
		refComps[l].Add(m.Row(i), m.RowNorm(i))
	}
	var refCrit float64
	for _, c := range refComps {
		refCrit += c.Norm()
	}
	return Result{Corpus: c, Rows: m.NumRows(), Cols: m.NumCols(), NNZ: m.NNZ(), ARI: ari, NMI: nmi, RefCriterion: refCrit, Criterion: res.Criterion}, nil
}

// Skip records a corpus the harness could not run and why (for example a
// reference clustering whose length does not match its matrix).
type Skip struct {
	Corpus
	Reason string
}

// RunAll runs every corpus in order. Corpora whose files are unusable are
// skipped and reported; a cancelled context stops the run with its error.
func RunAll(ctx context.Context, corpora []Corpus, o Options) ([]Result, []Skip, error) {
	out := make([]Result, 0, len(corpora))
	var skips []Skip
	for _, c := range corpora {
		if err := ctx.Err(); err != nil {
			return out, skips, err
		}
		r, err := Run(ctx, c, o)
		if err != nil {
			if ctx.Err() != nil {
				return out, skips, err
			}
			skips = append(skips, Skip{Corpus: c, Reason: err.Error()})
			continue
		}
		out = append(out, r)
	}
	return out, skips, nil
}

// Summary aggregates results.
type Summary struct {
	Corpora        int
	MeanARI        float64
	MinARI         float64
	MeanNMI        float64
	CriterionRatio float64 // mean of CLUTRO I2 / reference I2
}

// Summarize computes the aggregate figures.
func Summarize(rs []Result) Summary {
	if len(rs) == 0 {
		return Summary{}
	}
	s := Summary{Corpora: len(rs), MinARI: 1}
	for _, r := range rs {
		s.MeanARI += r.ARI
		s.MeanNMI += r.NMI
		if r.ARI < s.MinARI {
			s.MinARI = r.ARI
		}
		if r.RefCriterion > 0 {
			s.CriterionRatio += r.Criterion / r.RefCriterion
		}
	}
	n := float64(len(rs))
	s.MeanARI /= n
	s.MeanNMI /= n
	s.CriterionRatio /= n
	return s
}

// WriteTable prints results and the summary as an aligned table.
func WriteTable(w io.Writer, rs []Result) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "corpus\trows\tcols\tnnz\tk\tARI\tNMI\tI2 ref\tI2 clutro")
	for _, r := range rs {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.2f\t%.2f\n", r.Name, r.Rows, r.Cols, r.NNZ, r.K, r.ARI, r.NMI, r.RefCriterion, r.Criterion)
	}
	s := Summarize(rs)
	fmt.Fprintf(tw, "mean of %d\t\t\t\t\t%.3f\t%.3f\t\tI2 ratio %.4f\n", s.Corpora, s.MeanARI, s.MeanNMI, s.CriterionRatio)
	fmt.Fprintf(tw, "min\t\t\t\t\t%.3f\t\t\t\n", s.MinARI)
	tw.Flush()
}
