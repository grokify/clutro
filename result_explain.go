package clutro

import "github.com/grokify/clutro/explain"

// Describe returns cluster k's top-n descriptive features — the columns that
// contribute most to its internal similarity — with labels when column
// labels were given. n < 0 returns all.
func (r *Result) Describe(k, n int) []explain.Feature {
	return explain.Describe(r.m, r.Labels, r.K, k, n, r.columnLabels)
}

// Discriminate returns cluster k's top-n discriminating features — the
// columns that most set it apart from the rest of the rows.
func (r *Result) Discriminate(k, n int) []explain.Feature {
	return explain.Discriminate(r.m, r.Labels, r.K, k, n, r.columnLabels)
}

// Representatives returns the n rows of cluster k most similar to its
// composite, best first, with their similarities.
func (r *Result) Representatives(k, n int) ([]int, []float64) {
	return explain.Representatives(r.m, r.Labels, r.K, k, n, r.similarity)
}

// Summaries returns size and internal/external similarity statistics for
// every cluster.
func (r *Result) Summaries() []explain.Summary {
	return explain.Summaries(r.m, r.Labels, r.K, r.similarity)
}
