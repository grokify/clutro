package evaluate

import (
	"math"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestExternalMetrics(t *testing.T) {
	truth := []int32{0, 0, 0, 1, 1, 1}
	same := []int32{5, 5, 5, 9, 9, 9} // identical up to relabelling
	ari, _ := AdjustedRandIndex(same, truth)
	nmi, _ := NormalizedMutualInformation(same, truth)
	pur, _ := Purity(same, truth)
	ent, _ := Entropy(same, truth)
	if !near(ari, 1) || !near(nmi, 1) || !near(pur, 1) || !near(ent, 0) {
		t.Fatalf("identical partitions: ari %v nmi %v purity %v entropy %v", ari, nmi, pur, ent)
	}

	// Reference values from scikit-learn for this pair:
	//   adjusted_rand_score([0,0,1,1,2,2],[0,0,1,2,2,2]) = 0.4444444444444444
	x := []int32{0, 0, 1, 1, 2, 2}
	y := []int32{0, 0, 1, 2, 2, 2}
	ari, _ = AdjustedRandIndex(x, y)
	if !near(ari, 4.0/9.0) {
		t.Fatalf("ari %v want 0.4444", ari)
	}

	one := []int32{0, 0, 0, 0, 0, 0}
	pur, _ = Purity(one, truth)
	ent, _ = Entropy(one, truth)
	if !near(pur, 0.5) || !near(ent, 1) {
		t.Fatalf("single cluster: purity %v entropy %v", pur, ent)
	}
	if ari, _ := AdjustedRandIndex(one, one); !near(ari, 1) {
		t.Fatal("trivial partitions agree")
	}
	if nmi, _ := NormalizedMutualInformation(one, truth); !near(nmi, 0) {
		t.Fatalf("nmi of a constant labeling %v", nmi)
	}
	if _, err := AdjustedRandIndex([]int32{0}, []int32{0, 1}); err == nil {
		t.Fatal("length mismatch")
	}
	if ari, _ := AdjustedRandIndex(nil, nil); ari != 1 {
		t.Fatal("empty")
	}
}
