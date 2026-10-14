package golden

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/grokify/clutro/cluto"
	"github.com/grokify/clutro/internal/synth"
	"github.com/grokify/clutro/matrix"
)

// writeCorpus writes a planted corpus and its planted labels as a reference
// clustering, so the harness has a corpus whose "reference" is the truth.
func writeCorpus(t *testing.T, dir, name string, p synth.Params) {
	t.Helper()
	m, labels := synth.Sparse(p)
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		t.Fatal(err)
	}
	if err := cluto.WriteSparse(f, m); err != nil {
		t.Fatal(err)
	}
	f.Close()
	cf, err := os.Create(filepath.Join(dir, name+".clustering."+strconv.Itoa(p.K)))
	if err != nil {
		t.Fatal(err)
	}
	if err := cluto.WriteClustering(cf, labels); err != nil {
		t.Fatal(err)
	}
	cf.Close()
}

func TestHarness(t *testing.T) {
	dir := t.TempDir()
	writeCorpus(t, dir, "corpus-01.mat", synth.Params{Rows: 120, Cols: 200, K: 3, NNZ: 12, Noise: 0.1, Seed: 1})
	writeCorpus(t, dir, "corpus-02.mat", synth.Params{Rows: 200, Cols: 300, K: 5, NNZ: 15, Noise: 0.1, Seed: 2})
	if err := os.WriteFile(filepath.Join(dir, "orphan.mat.clustering.4"), []byte("0\n"), 0o600); err != nil {
		t.Fatal(err) // no matrix: must be skipped
	}
	corpora, err := Find(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(corpora) != 2 || corpora[0].Name != "corpus-01.mat" || corpora[0].K != 3 || corpora[1].K != 5 {
		t.Fatalf("find: %+v", corpora)
	}
	rs, skips, err := RunAll(context.Background(), corpora, Options{Seed: 1, Trials: 5, Refine: true, RowModel: matrix.RowL2})
	if err != nil || len(skips) != 0 {
		t.Fatal(err, skips)
	}
	for _, r := range rs {
		if r.ARI < 0.95 || r.NMI < 0.9 {
			t.Fatalf("%s: ARI %v NMI %v", r.Name, r.ARI, r.NMI)
		}
		if r.Rows == 0 || r.RefCriterion <= 0 || r.Criterion <= 0 {
			t.Fatalf("%s: %+v", r.Name, r)
		}
	}
	s := Summarize(rs)
	if s.Corpora != 2 || s.MinARI < 0.95 || s.CriterionRatio < 0.99 {
		t.Fatalf("summary %+v", s)
	}
	var buf bytes.Buffer
	WriteTable(&buf, rs)
	if !strings.Contains(buf.String(), "corpus-01.mat") || !strings.Contains(buf.String(), "mean of 2") {
		t.Fatalf("table:\n%s", buf.String())
	}
	if Summarize(nil).Corpora != 0 {
		t.Fatal("empty summary")
	}
	if _, err := Find(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("missing dir")
	}
	// A reference of the wrong length is an error, not a silent mismatch.
	if err := os.WriteFile(filepath.Join(dir, "corpus-01.mat.clustering.3"), []byte("0\n1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Run(context.Background(), corpora[0], Options{Trials: 1}); err == nil {
		t.Fatal("length mismatch")
	}
	rs, skips, err = RunAll(context.Background(), corpora, Options{Trials: 1})
	if err != nil || len(rs) != 1 || len(skips) != 1 || skips[0].Name != "corpus-01.mat" || !strings.Contains(skips[0].Reason, "labels") {
		t.Fatalf("RunAll should skip the bad corpus: %v %v %v", err, rs, skips)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := RunAll(ctx, corpora, Options{Trials: 1}); err == nil {
		t.Fatal("cancelled context")
	}
}
