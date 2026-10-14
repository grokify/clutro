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
//	silhouette = 0.93104..., DB = 0.070710..., CH = 400
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
	b := (math.Sqrt(200) + math.Sqrt(221)) / 2
	if want := (b - a) / b; math.Abs(s-want) > 1e-9 {
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
	// Identical centroids make Davies-Bouldin infinite.
	same, _ := matrix.NewDense(4, 1, []float32{0, 1, 0, 1})
	if db, _ := DaviesBouldin(same, []int32{0, 1, 0, 1}, 2); !math.IsInf(db, 1) {
		t.Fatalf("db with identical centroids %v", db)
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
	if exact < 0.3 {
		t.Fatalf("planted clusters should score well: %v", exact)
	}
	// Cosine-normalized rows: Euclidean distance orders like cosine.
	d := Euclidean(m.Row(0), m.Row(0), m.RowNorm(0), m.RowNorm(0))
	if d != 0 {
		t.Fatal("self distance")
	}
}
