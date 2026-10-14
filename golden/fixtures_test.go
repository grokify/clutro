package golden

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/grokify/clutro/matrix"
)

// TestPublishedFixtures runs the harness over testdata/cluto: real CLUTO
// inputs and outputs (numeric only, renamed) from a text corpus. The band
// below was set from a 103-corpus run: agreement with CLUTO's own partitions
// is partial because k ≈ n/12 leaves many equally good optima (CLUTRO agrees
// with itself across seeds at ARI ≈ 0.48), while on the criterion CLUTO
// optimized, CLUTRO's partitions score as well or better.
func TestPublishedFixtures(t *testing.T) {
	dir := filepath.Join("..", "testdata", "cluto")
	corpora, err := Find(dir)
	if err != nil {
		t.Skipf("no fixtures: %v", err)
	}
	if len(corpora) == 0 {
		t.Skip("no fixtures")
	}
	verifyChecksums(t, dir)

	rs, skips, err := RunAll(context.Background(), corpora, Options{Seed: 1, Trials: 10, RowModel: matrix.RowL2})
	if err != nil || len(skips) != 0 {
		t.Fatalf("%v %v", err, skips)
	}
	s := Summarize(rs)
	t.Logf("%d corpora: mean ARI %.3f, min ARI %.3f, mean NMI %.3f, I2 ratio %.4f", s.Corpora, s.MeanARI, s.MinARI, s.MeanNMI, s.CriterionRatio)
	if s.MeanARI < 0.25 {
		t.Errorf("mean ARI against CLUTO fell to %.3f (band: ≥ 0.25)", s.MeanARI)
	}
	if s.MinARI < 0.05 {
		t.Errorf("min ARI against CLUTO fell to %.3f (band: ≥ 0.05)", s.MinARI)
	}
	if s.CriterionRatio < 0.99 {
		t.Errorf("CLUTRO's I2 is %.4f of CLUTO's (band: ≥ 0.99)", s.CriterionRatio)
	}
	refined, _, err := RunAll(context.Background(), corpora, Options{Seed: 1, Trials: 10, Refine: true, RowModel: matrix.RowL2})
	if err != nil {
		t.Fatal(err)
	}
	if rr := Summarize(refined).CriterionRatio; rr < s.CriterionRatio {
		t.Errorf("refinement lowered the criterion ratio: %.4f < %.4f", rr, s.CriterionRatio)
	}
}

func verifyChecksums(t *testing.T, dir string) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "SHA256SUMS.json"))
	if err != nil {
		t.Fatalf("fixtures need SHA256SUMS.json: %v", err)
	}
	var s struct {
		Files map[string]string `json:"files"`
	}
	if err := json.Unmarshal(data, &s); err != nil {
		t.Fatal(err)
	}
	for name, want := range s.Files {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		if _, err := io.Copy(h, f); err != nil {
			t.Fatal(err)
		}
		f.Close()
		if got := hex.EncodeToString(h.Sum(nil)); got != want {
			t.Errorf("%s: checksum %s, want %s", name, got, want)
		}
	}
}
