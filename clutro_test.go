package clutro_test

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/grokify/clutro"
	"github.com/grokify/clutro/evaluate"
	"github.com/grokify/clutro/internal/synth"
	"github.com/grokify/clutro/matrix"
	"github.com/grokify/clutro/metric"
	"github.com/grokify/clutro/partition"
)

func TestFitAlgorithms(t *testing.T) {
	m, truth := synth.Sparse(synth.Params{Rows: 400, Cols: 600, K: 4, NNZ: 20, Noise: 0.1, Seed: 11})
	for _, alg := range []clutro.Algorithm{clutro.RepeatedBisection, clutro.Direct, clutro.KMeans} {
		res, err := clutro.Fit(context.Background(), m, clutro.WithAlgorithm(alg), clutro.WithClusters(4), clutro.WithSeed(1), clutro.WithRefinement(true))
		if err != nil {
			t.Fatalf("%v: %v", alg, err)
		}
		ari, _ := evaluate.AdjustedRandIndex(res.Labels, truth)
		if ari < 0.95 {
			t.Fatalf("%v: ARI %v", alg, ari)
		}
		if res.K != 4 || res.Algorithm != alg || len(res.Sizes()) != 4 {
			t.Fatalf("%v: result shape", alg)
		}
		cl := res.Clusters()
		total := 0
		for k, rows := range cl {
			total += len(rows)
			for i := 1; i < len(rows); i++ {
				if rows[i] <= rows[i-1] {
					t.Fatal("cluster rows ascending")
				}
			}
			if res.Composite(k).N != len(rows) {
				t.Fatal("composite size")
			}
		}
		if total != m.NumRows() {
			t.Fatal("clusters partition the rows")
		}
		if (res.Tree() != nil) != (alg == clutro.RepeatedBisection) {
			t.Fatalf("%v: tree presence", alg)
		}
		if res.Matrix() != m || res.Similarity().Name() != "cos" {
			t.Fatal("accessors")
		}
	}
}

func TestFitOptionsAndErrors(t *testing.T) {
	m, _ := synth.Sparse(synth.Params{Rows: 20, Cols: 30, K: 2, NNZ: 5, Seed: 1})
	if _, err := clutro.Fit(context.Background(), m); err == nil {
		t.Fatal("k required")
	}
	if _, err := clutro.Fit(context.Background(), m, clutro.WithClusters(21)); err == nil {
		t.Fatal("k > rows")
	}
	if _, err := clutro.Fit(context.Background(), nil, clutro.WithClusters(1)); err == nil {
		t.Fatal("nil matrix")
	}
	if _, err := clutro.Fit(context.Background(), m, clutro.WithClusters(2), clutro.WithColumnLabels([]string{"x"})); err == nil {
		t.Fatal("column label count")
	}
	if _, err := clutro.Fit(context.Background(), m, clutro.WithClusters(2), clutro.WithRowLabels([]string{"x"})); err == nil {
		t.Fatal("row label count")
	}
	if _, err := clutro.Fit(context.Background(), m, clutro.WithClusters(2), clutro.WithAlgorithm(clutro.Algorithm(99))); err == nil {
		t.Fatal("unknown algorithm")
	}
	cols := make([]string, m.NumCols())
	for i := range cols {
		cols[i] = fmt.Sprintf("f%d", i)
	}
	res, err := clutro.Fit(context.Background(), m, clutro.WithClusters(2), clutro.WithColumnLabels(cols),
		clutro.WithSimilarity(metric.Cosine{}), clutro.WithTrials(3), clutro.WithMaxIter(5),
		clutro.WithSelection(partition.SelectLargest), clutro.WithParallelism(2))
	if err != nil {
		t.Fatal(err)
	}
	if res.ColumnLabels()[1] != "f1" || res.RowLabels() != nil {
		t.Fatal("labels")
	}
	for _, name := range []string{"rb", "rbr", "direct", "kmeans", "skmeans"} {
		if _, _, ok := clutro.ParseAlgorithm(name); !ok {
			t.Fatalf("parse %s", name)
		}
	}
	if _, refine, _ := clutro.ParseAlgorithm("rbr"); !refine {
		t.Fatal("rbr implies refinement")
	}
	if _, _, ok := clutro.ParseAlgorithm("agglo"); ok {
		t.Fatal("unknown name")
	}
	if clutro.RepeatedBisection.String() != "rb" || clutro.Algorithm(5).String() == "" {
		t.Fatal("String")
	}
	var _ matrix.Rows = m
}

func TestFitIsDeterministicAcrossWorkers(t *testing.T) {
	m, _ := synth.Sparse(synth.Params{Rows: 500, Cols: 700, K: 5, NNZ: 20, Noise: 0.2, Seed: 4})
	var ref []int32
	for _, w := range []int{1, 2, 4, 8} {
		res, err := clutro.Fit(context.Background(), m, clutro.WithClusters(5), clutro.WithSeed(9), clutro.WithRefinement(true), clutro.WithParallelism(w))
		if err != nil {
			t.Fatal(err)
		}
		if ref == nil {
			ref = res.Labels
			continue
		}
		if !reflect.DeepEqual(res.Labels, ref) {
			t.Fatalf("labels differ with %d workers", w)
		}
	}
}
