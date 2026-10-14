package evaluate

import (
	"math"
	"testing"

	"github.com/grokify/clutro/internal/synth"
	"github.com/grokify/clutro/matrix"
)

// Four points in two tight pairs far apart. Hand-computed (and matching
// scikit-learn's silhouette_score, davies_bouldin_score and
// calinski_harabasz_score on the same input):
//
//	silhouette = 0.92929..., DB = 0.070710..., CH = 400
//
// The silhouette is the mean of two distinct per-point values: the outer
// points (0,0) and (10,11) see b = (√200+√221)/2, the inner points (0,1)
// and (10,10) see b = (√181+√200)/2.
func pairs(t *testing.T) (*matrix.Dense, []int32) {
	t.Helper()
	m, err := matrix.NewDense(4, 2, []float32{0, 0, 0, 1, 10, 10, 10, 11})
	if err != nil {
		t.Fatal(err)
	}
	return m, []int32{0, 0, 1, 1}
}

func TestInternalMetricsHandComputed(t *testing.T) {
	m, labels := pairs(t)
	s, err := Silhouette(m, labels, 2, SilhouetteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	a := 1.0
	bOuter := (math.Sqrt(200) + math.Sqrt(221)) / 2
	bInner := (math.Sqrt(181) + math.Sqrt(200)) / 2
	want := ((bOuter-a)/bOuter + (bInner-a)/bInner) / 2
	if math.Abs(s-want) > 1e-6 { // float32 row norms
		t.Fatalf("silhouette %v want %v", s, want)
	}
	db, err := DaviesBouldin(m, labels, 2)
	if err != nil {
		t.Fatal(err)
	}
	if want := 1.0 / math.Sqrt(200); math.Abs(db-want) > 1e-9 {
		t.Fatalf("davies-bouldin %v want %v", db, want)
	}
	ch, err := CalinskiHarabasz(m, labels, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(ch-400) > 1e-9 {
		t.Fatalf("calinski-harabasz %v want 400", ch)
	}
}

func TestInternalMetricsEdges(t *testing.T) {
	m, labels := pairs(t)
	if _, err := Silhouette(m, labels, 1, SilhouetteOptions{}); err == nil {
		t.Fatal("k=1")
	}
	if _, err := Silhouette(m, []int32{0, 0, 1}, 2, SilhouetteOptions{}); err == nil {
		t.Fatal("label count")
	}
	if _, err := DaviesBouldin(m, []int32{0, 0, 5, 1}, 2); err == nil {
		t.Fatal("label range")
	}
	if _, err := CalinskiHarabasz(nil, nil, 2); err == nil {
		t.Fatal("nil matrix")
	}
	// Singleton clusters score 0 in the silhouette.
	s, err := Silhouette(m, []int32{0, 1, 2, 2}, 3, SilhouetteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if s <= 0 || s >= 1 {
		t.Fatalf("silhouette with singletons %v", s)
	}
	// Identical centroids (both 0.5) make Davies-Bouldin infinite.
	same, _ := matrix.NewDense(4, 1, []float32{0, 1, 0, 1})
	if db, _ := DaviesBouldin(same, []int32{0, 0, 1, 1}, 2); !math.IsInf(db, 1) {
		t.Fatalf("db with identical centroids %v", db)
	}
	// Perfectly tight clusters score 0.
	if db, _ := DaviesBouldin(same, []int32{0, 1, 0, 1}, 2); db != 0 {
		t.Fatalf("db with zero scatter %v", db)
	}
	if ch, _ := CalinskiHarabasz(m, []int32{0, 0, 1, 1}, 2); ch <= 0 {
		t.Fatal("ch positive")
	}
}

func TestSilhouetteSamplingIsDeterministicAndClose(t *testing.T) {
	m, labels := synth.Sparse(synth.Params{Rows: 600, Cols: 400, K: 3, NNZ: 15, Noise: 0.1, Seed: 2})
	exact, err := Silhouette(m, labels, 3, SilhouetteOptions{SampleAbove: -1})
	if err != nil {
		t.Fatal(err)
	}
	s1, _ := Silhouette(m, labels, 3, SilhouetteOptions{SampleAbove: 200, Seed: 1})
	s2, _ := Silhouette(m, labels, 3, SilhouetteOptions{SampleAbove: 200, Seed: 1})
	if s1 != s2 {
		t.Fatal("sampled silhouette must be deterministic for a seed")
	}
	if math.Abs(s1-exact) > 0.1 {
		t.Fatalf("sampled %v vs exact %v", s1, exact)
	}
	// Sparse high-dimensional rows have small silhouettes in absolute terms
	// (distances concentrate); the planted labels must still beat a
	// scrambled labeling.
	scrambled := make([]int32, len(labels))
	for i, l := range labels {
		scrambled[i] = (l + int32(i%3)) % 3 //nolint:gosec // tiny values
	}
	bad, _ := Silhouette(m, scrambled, 3, SilhouetteOptions{SampleAbove: -1})
	if exact <= 0 || exact <= bad {
		t.Fatalf("planted silhouette %v should be positive and above scrambled %v", exact, bad)
	}
	// Cosine-normalized rows: Euclidean distance orders like cosine.
	d := Euclidean(m.Row(0), m.Row(0), m.RowNorm(0), m.RowNorm(0))
	if d != 0 {
		t.Fatal("self distance")
	}
}
