package partition

import (
	"context"
	"math"
	"testing"

	"github.com/grokify/clutro/evaluate"
	"github.com/grokify/clutro/internal/rng"
	"github.com/grokify/clutro/internal/synth"
	"github.com/grokify/clutro/partition/reference"
)

// The optimized paths must reproduce the reference oracle: same seeding,
// same Lloyd fixed point, same refinement result.
func TestMatchesReference(t *testing.T) {
	m, _ := synth.Sparse(synth.Params{Rows: 300, Cols: 400, K: 4, NNZ: 15, Noise: 0.2, Seed: 21})
	rows := make([]int32, m.NumRows())
	for i := range rows {
		rows[i] = int32(i)
	}
	o := (Options{Seed: 5, Trials: 1, MaxIter: 30}).withDefaults()
	for trial := 0; trial < 5; trial++ {
		init := seedPlusPlus(m, rows, 4, o, rng.SubStream(o.Seed, "ref", trial))
		fast, err := lloyd(context.Background(), m, rows, append([]int32(nil), init...), 4, o)
		if err != nil {
			t.Fatal(err)
		}
		slow, slowCrit, err := reference.Lloyd(context.Background(), m, init, 4, o.Similarity, o.MaxIter)
		if err != nil {
			t.Fatal(err)
		}
		ari, _ := evaluate.AdjustedRandIndex(fast.Labels, slow)
		if ari != 1 || math.Abs(fast.Criterion-slowCrit) > 1e-6 {
			t.Fatalf("trial %d: lloyd differs from reference (ARI %v, crit %v vs %v)", trial, ari, fast.Criterion, slowCrit)
		}

		order := rng.Perm(rng.SubStream(o.Seed, "refine", 0), m.NumRows())
		refSlow, refSlowCrit := reference.Refine(m, fast.Labels, 4, order, 1)
		refFast, err := Refine(context.Background(), m, Assignment{Labels: fast.Labels, K: 4}, Options{Seed: o.Seed, MaxIter: 1})
		if err != nil {
			t.Fatal(err)
		}
		ari, _ = evaluate.AdjustedRandIndex(refFast.Labels, refSlow)
		if ari != 1 || math.Abs(refFast.Criterion-refSlowCrit) > 1e-6 {
			t.Fatalf("trial %d: refine differs from reference (ARI %v, crit %v vs %v)", trial, ari, refFast.Criterion, refSlowCrit)
		}
	}
}
