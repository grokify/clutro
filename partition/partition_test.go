package partition

import (
	"context"
	"math"
	"reflect"
	"testing"

	"github.com/grokify/clutro/evaluate"
	"github.com/grokify/clutro/internal/synth"
	"github.com/grokify/clutro/metric"
)

var planted = synth.Params{Rows: 600, Cols: 800, K: 6, NNZ: 25, Noise: 0.15, Seed: 3}

func ari(t *testing.T, got, want []int32) float64 {
	t.Helper()
	v, err := evaluate.AdjustedRandIndex(got, want)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestKMeansRecoversPlantedClusters(t *testing.T) {
	m, truth := synth.Sparse(planted)
	a, err := KMeans(context.Background(), m, 6, Options{Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if a.K != 6 || len(a.Labels) != m.NumRows() {
		t.Fatal("shape")
	}
	if v := ari(t, a.Labels, truth); v < 0.95 {
		t.Fatalf("k-means ARI %v", v)
	}
	for _, s := range a.Sizes() {
		if s == 0 {
			t.Fatal("empty cluster")
		}
	}
}

func TestBisectRecoversPlantedClusters(t *testing.T) {
	m, truth := synth.Sparse(planted)
	for _, sel := range []Selection{SelectBest, SelectLargest} {
		a, tree, err := Bisect(context.Background(), m, 6, Options{Seed: 1, Selection: sel})
		if err != nil {
			t.Fatal(err)
		}
		if v := ari(t, a.Labels, truth); v < 0.95 {
			t.Fatalf("bisect (%v) ARI %v", sel, v)
		}
		leaves := tree.Leaves()
		if len(leaves) != 6 || leaves[0].Label != 0 || leaves[5].Label != 5 {
			t.Fatalf("leaves %d", len(leaves))
		}
		if len(tree.Root.Rows) != m.NumRows() || tree.Root.IsLeaf() {
			t.Fatal("root")
		}
		// Leaf row sets must equal the flat assignment (no refinement).
		for _, leaf := range leaves {
			for _, r := range leaf.Rows {
				if a.Labels[r] != leaf.Label {
					t.Fatal("leaf rows disagree with labels")
				}
			}
		}
	}
	// Refinement must not decrease the criterion.
	plain, _, _ := Bisect(context.Background(), m, 6, Options{Seed: 1})
	refined, _, err := Bisect(context.Background(), m, 6, Options{Seed: 1, Refine: true})
	if err != nil {
		t.Fatal(err)
	}
	if refined.Criterion < plain.Criterion-1e-9 {
		t.Fatalf("rbr criterion %v < rb %v", refined.Criterion, plain.Criterion)
	}
}

func TestDirectAndRefine(t *testing.T) {
	m, truth := synth.Sparse(planted)
	a, err := Direct(context.Background(), m, 6, Options{Seed: 2})
	if err != nil {
		t.Fatal(err)
	}
	if v := ari(t, a.Labels, truth); v < 0.95 {
		t.Fatalf("direct ARI %v", v)
	}
	// Refining a deliberately bad assignment must improve it.
	bad := Assignment{Labels: make([]int32, m.NumRows()), K: 6}
	for i := range bad.Labels {
		bad.Labels[i] = int32(i % 6)
	}
	bad.Criterion = (Options{}).withDefaults().Criterion.Value(composites(m, bad.Labels, 6))
	r, err := Refine(context.Background(), m, bad, Options{Seed: 2, MaxIter: 50})
	if err != nil {
		t.Fatal(err)
	}
	if r.Criterion <= bad.Criterion {
		t.Fatalf("refine did not improve: %v -> %v", bad.Criterion, r.Criterion)
	}
	if v := ari(t, r.Labels, truth); v < 0.8 {
		t.Fatalf("refined-from-bad ARI %v", v)
	}
}

func TestDeterminism(t *testing.T) {
	m, _ := synth.Sparse(planted)
	ref, refTree, err := Bisect(context.Background(), m, 6, Options{Seed: 7, Refine: true, Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []int{2, 3, 8} {
		a, tree, err := Bisect(context.Background(), m, 6, Options{Seed: 7, Refine: true, Workers: w})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a.Labels, ref.Labels) || a.Criterion != ref.Criterion {
			t.Fatalf("labels differ with %d workers", w)
		}
		if !reflect.DeepEqual(tree.Root.Left.Rows, refTree.Root.Left.Rows) {
			t.Fatalf("tree differs with %d workers", w)
		}
	}
	k1, _ := KMeans(context.Background(), m, 6, Options{Seed: 7, Workers: 1})
	k8, _ := KMeans(context.Background(), m, 6, Options{Seed: 7, Workers: 8})
	if !reflect.DeepEqual(k1.Labels, k8.Labels) {
		t.Fatal("k-means labels differ with workers")
	}
	other, _ := KMeans(context.Background(), m, 6, Options{Seed: 8})
	if reflect.DeepEqual(other.Labels, k1.Labels) && other.Criterion == k1.Criterion {
		t.Log("different seeds gave identical labels (possible on easy data)")
	}
}

func TestTreeCuts(t *testing.T) {
	m, truth := synth.Sparse(planted)
	_, tree, err := Bisect(context.Background(), m, 12, Options{Seed: 1, Selection: SelectLargest})
	if err != nil {
		t.Fatal(err)
	}
	cut := tree.CutMaxSize(110) // 600 rows / 6 planted clusters = 100 each
	if cut.K < 6 || cut.K > 12 {
		t.Fatalf("CutMaxSize K=%d", cut.K)
	}
	if v := ari(t, cut.Labels, truth); v < 0.8 {
		t.Fatalf("CutMaxSize ARI %v", v)
	}
	k6 := tree.CutK(6)
	if k6.K != 6 {
		t.Fatalf("CutK K=%d", k6.K)
	}
	if v := ari(t, k6.Labels, truth); v < 0.9 {
		t.Fatalf("CutK ARI %v", v)
	}
	if tree.CutK(0).K != 0 {
		t.Fatal("CutK 0")
	}
}

func TestEuclideanDense(t *testing.T) {
	m, truth := synth.Dense(300, 6, 3, 1.0, 5)
	a, err := KMeans(context.Background(), m, 3, Options{Seed: 1, Similarity: metric.Euclidean{}})
	if err != nil {
		t.Fatal(err)
	}
	if v := ari(t, a.Labels, truth); v < 0.95 {
		t.Fatalf("euclidean k-means ARI %v", v)
	}
}

func TestErrorsAndEdges(t *testing.T) {
	m, _ := synth.Sparse(synth.Params{Rows: 5, Cols: 20, K: 1, NNZ: 3, Seed: 1})
	if _, err := KMeans(context.Background(), m, 0, Options{}); err == nil {
		t.Fatal("k=0")
	}
	if _, err := KMeans(context.Background(), m, 6, Options{}); err == nil {
		t.Fatal("k>rows")
	}
	if _, _, err := Bisect(context.Background(), nil, 1, Options{}); err == nil {
		t.Fatal("nil matrix")
	}
	a, tree, err := Bisect(context.Background(), m, 1, Options{})
	if err != nil || a.K != 1 || !tree.Root.IsLeaf() {
		t.Fatalf("k=1: %v %+v", err, a)
	}
	a, _, err = Bisect(context.Background(), m, 5, Options{Seed: 1})
	if err != nil || a.K != 5 {
		t.Fatalf("k=rows: %v K=%d", err, a.K)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := KMeans(ctx, m, 2, Options{}); err == nil {
		t.Fatal("cancelled")
	}
	if _, err := Refine(context.Background(), m, Assignment{Labels: []int32{0}, K: 1}, Options{}); err == nil {
		t.Fatal("label length mismatch")
	}
	if math.IsNaN(a.Criterion) {
		t.Fatal("criterion")
	}
}
