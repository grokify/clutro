// Package clutro is a pure-Go clustering toolkit for sparse and dense data,
// built around the optimization-criterion approach of CLUTO: repeated
// bisection, spherical k-means, pluggable criterion functions, deterministic
// results.
//
// Fit is the entry point:
//
//	res, err := clutro.Fit(ctx, m,
//	    clutro.WithAlgorithm(clutro.RepeatedBisection),
//	    clutro.WithClusters(12),
//	    clutro.WithRefinement(true),
//	    clutro.WithSeed(42))
//
// CLUTRO is inspired by CLUTO and is an independent implementation of the
// published algorithms; it is not affiliated with or endorsed by the Karypis
// lab or the University of Minnesota.
package clutro

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/grokify/clutro/criterion"
	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
	"github.com/grokify/clutro/partition"
)

// Algorithm selects the clustering method.
type Algorithm int

const (
	// RepeatedBisection splits one cluster at a time until k clusters exist
	// (CLUTO "rb"; "rbr" with WithRefinement). The default.
	RepeatedBisection Algorithm = iota
	// Direct computes a k-way partition at once and refines it (CLUTO
	// "direct").
	Direct
	// KMeans is k-means++ seeding and Lloyd iterations with no refinement;
	// under cosine similarity this is spherical k-means.
	KMeans
)

// String returns the CLUTO-style method name.
func (a Algorithm) String() string {
	switch a {
	case RepeatedBisection:
		return "rb"
	case Direct:
		return "direct"
	case KMeans:
		return "kmeans"
	}
	return fmt.Sprintf("Algorithm(%d)", int(a))
}

// ParseAlgorithm maps "rb", "rbr", "direct", "kmeans" or "skmeans" to an
// Algorithm and whether refinement is implied.
func ParseAlgorithm(name string) (Algorithm, bool, bool) {
	switch name {
	case "rb":
		return RepeatedBisection, false, true
	case "rbr":
		return RepeatedBisection, true, true
	case "direct":
		return Direct, true, true
	case "kmeans", "skmeans":
		return KMeans, false, true
	}
	return 0, false, false
}

type config struct {
	algorithm    Algorithm
	k            int
	opts         partition.Options
	refineSet    bool
	columnLabels []string
	rowLabels    []string
}

// Option configures Fit.
type Option func(*config)

// WithAlgorithm selects the method; default RepeatedBisection.
func WithAlgorithm(a Algorithm) Option { return func(c *config) { c.algorithm = a } }

// WithClusters sets k. Required unless the data has one row.
func WithClusters(k int) Option { return func(c *config) { c.k = k } }

// WithSimilarity sets the similarity; default metric.Cosine.
func WithSimilarity(s metric.Similarity) Option { return func(c *config) { c.opts.Similarity = s } }

// WithCriterion sets the objective; default criterion.I2.
func WithCriterion(cr criterion.Criterion) Option { return func(c *config) { c.opts.Criterion = cr } }

// WithRefinement enables criterion-driven refinement after bisection
// ("rbr"). Direct always refines; KMeans never does.
func WithRefinement(on bool) Option {
	return func(c *config) { c.opts.Refine = on; c.refineSet = true }
}

// WithSeed sets the seed every random choice derives from; default 0.
func WithSeed(seed uint64) Option { return func(c *config) { c.opts.Seed = seed } }

// WithTrials sets the seeded starts per k-means or bisection; default 10.
func WithTrials(n int) Option { return func(c *config) { c.opts.Trials = n } }

// WithMaxIter bounds k-means iterations and refinement passes; default 20.
func WithMaxIter(n int) Option { return func(c *config) { c.opts.MaxIter = n } }

// WithSelection sets which cluster repeated bisection splits next; default
// partition.SelectBest.
func WithSelection(s partition.Selection) Option { return func(c *config) { c.opts.Selection = s } }

// WithParallelism bounds the worker count; 0 means all CPUs. Results do not
// depend on it.
func WithParallelism(workers int) Option { return func(c *config) { c.opts.Workers = workers } }

// WithColumnLabels names the columns (features), used by explanations.
func WithColumnLabels(labels []string) Option { return func(c *config) { c.columnLabels = labels } }

// WithRowLabels names the rows, used by explanations and output.
func WithRowLabels(labels []string) Option { return func(c *config) { c.rowLabels = labels } }

// ErrOptions reports an invalid option combination.
var ErrOptions = errors.New("clutro: invalid options")

// Result is a fitted clustering.
type Result struct {
	// Labels is the cluster index of every row.
	Labels []int32
	// K is the number of clusters.
	K int
	// Criterion is the objective value of the partition.
	Criterion float64
	// Algorithm is the method used.
	Algorithm Algorithm

	tree         *partition.Tree
	m            matrix.Rows
	comps        []*matrix.Composite
	similarity   metric.Similarity
	columnLabels []string
	rowLabels    []string
}

// Fit clusters the rows of m.
func Fit(ctx context.Context, m matrix.Rows, opts ...Option) (*Result, error) {
	c := config{}
	for _, o := range opts {
		o(&c)
	}
	if m == nil || m.NumRows() == 0 {
		return nil, fmt.Errorf("%w: empty matrix", ErrOptions)
	}
	if c.k == 0 {
		return nil, fmt.Errorf("%w: WithClusters is required", ErrOptions)
	}
	if c.k < 1 || c.k > m.NumRows() {
		return nil, fmt.Errorf("%w: k=%d for %d rows", ErrOptions, c.k, m.NumRows())
	}
	if c.columnLabels != nil && len(c.columnLabels) != m.NumCols() {
		return nil, fmt.Errorf("%w: %d column labels for %d columns", ErrOptions, len(c.columnLabels), m.NumCols())
	}
	if c.rowLabels != nil && len(c.rowLabels) != m.NumRows() {
		return nil, fmt.Errorf("%w: %d row labels for %d rows", ErrOptions, len(c.rowLabels), m.NumRows())
	}

	var (
		a    partition.Assignment
		tree *partition.Tree
		err  error
	)
	switch c.algorithm {
	case RepeatedBisection:
		a, tree, err = partition.Bisect(ctx, m, c.k, c.opts)
	case Direct:
		a, err = partition.Direct(ctx, m, c.k, c.opts)
	case KMeans:
		a, err = partition.KMeans(ctx, m, c.k, c.opts)
	default:
		return nil, fmt.Errorf("%w: unknown algorithm %v", ErrOptions, c.algorithm)
	}
	if err != nil {
		return nil, err
	}
	o := c.opts
	if o.Similarity == nil {
		o.Similarity = metric.Cosine{}
	}
	res := &Result{
		Labels:       a.Labels,
		K:            a.K,
		Criterion:    a.Criterion,
		Algorithm:    c.algorithm,
		tree:         tree,
		m:            m,
		similarity:   o.Similarity,
		columnLabels: c.columnLabels,
		rowLabels:    c.rowLabels,
	}
	res.comps = make([]*matrix.Composite, a.K)
	for i := range res.comps {
		res.comps[i] = matrix.NewComposite(m.NumCols())
	}
	for i, l := range a.Labels {
		res.comps[l].Add(m.Row(i), m.RowNorm(i))
	}
	return res, nil
}

// Sizes returns the number of rows per cluster.
func (r *Result) Sizes() []int {
	s := make([]int, r.K)
	for _, l := range r.Labels {
		s[l]++
	}
	return s
}

// Clusters returns the row indices of every cluster, each ascending.
func (r *Result) Clusters() [][]int32 {
	out := make([][]int32, r.K)
	for i, l := range r.Labels {
		out[l] = append(out[l], int32(i))
	}
	for _, c := range out {
		sort.Slice(c, func(i, j int) bool { return c[i] < c[j] })
	}
	return out
}

// Tree returns the bisection tree, or nil for algorithms that build none.
func (r *Result) Tree() *partition.Tree { return r.tree }

// Composite returns cluster k's composite vector (the sum of its rows).
func (r *Result) Composite(k int) *matrix.Composite { return r.comps[k] }

// ColumnLabels returns the column names given to Fit, or nil.
func (r *Result) ColumnLabels() []string { return r.columnLabels }

// RowLabels returns the row names given to Fit, or nil.
func (r *Result) RowLabels() []string { return r.rowLabels }

// Matrix returns the data the result was fitted on.
func (r *Result) Matrix() matrix.Rows { return r.m }

// Similarity returns the similarity the result was fitted with.
func (r *Result) Similarity() metric.Similarity { return r.similarity }
