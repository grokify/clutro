package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grokify/clutro/cluto"
	"github.com/grokify/clutro/internal/synth"
)

func runCLI(t *testing.T, args ...string) (string, error) {
	t.Helper()
	var out bytes.Buffer
	err := run(args, &out, &out)
	return out.String(), err
}

func writePlanted(t *testing.T, dir string) (string, string) {
	t.Helper()
	m, labels := synth.Sparse(synth.Params{Rows: 150, Cols: 200, K: 3, NNZ: 12, Noise: 0.1, Seed: 7})
	mat := filepath.Join(dir, "docs.mat")
	f, err := os.Create(mat)
	if err != nil {
		t.Fatal(err)
	}
	if err := cluto.WriteSparse(f, m); err != nil {
		t.Fatal(err)
	}
	f.Close()
	truth := filepath.Join(dir, "docs.mat.clustering.3")
	tf, err := os.Create(truth)
	if err != nil {
		t.Fatal(err)
	}
	if err := cluto.WriteClustering(tf, labels); err != nil {
		t.Fatal(err)
	}
	tf.Close()
	return mat, truth
}

func TestVclusterEvaluateGolden(t *testing.T) {
	dir := t.TempDir()
	mat, truth := writePlanted(t, dir)

	// CLUTO-style single-dash flags.
	out, err := runCLI(t, "vcluster", "-clmethod=rbr", "-rowmodel=l2", "-clustfile", filepath.Join(dir, "out.clustering.3"), "-showtree", mat, "3")
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if !strings.Contains(out, "3 clusters") || !strings.Contains(out, "wrote ") || !strings.Contains(out, "→ cluster") {
		t.Fatalf("vcluster output:\n%s", out)
	}

	out, err = runCLI(t, "evaluate", filepath.Join(dir, "out.clustering.3"), truth)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "ARI 1.0000") && !strings.Contains(out, "ARI 0.9") {
		t.Fatalf("evaluate output:\n%s", out)
	}

	rclass := filepath.Join(dir, "docs.rclass")
	if err := os.WriteFile(rclass, []byte(strings.Repeat("a\n", 50)+strings.Repeat("b\n", 50)+strings.Repeat("c\n", 50)), 0o600); err != nil {
		t.Fatal(err)
	}
	if out, err := runCLI(t, "evaluate", truth, rclass); err != nil || !strings.Contains(out, "purity") {
		t.Fatalf("rclass: %v %s", err, out)
	}

	// GNU-style double-dash flags work too.
	out, err = runCLI(t, "vcluster", "--json", "--clmethod", "direct", "--colmodel", "idf", "--rowmodel", "l2", mat, "3")
	if err != nil || !strings.Contains(out, `"method": "direct"`) || !strings.Contains(out, `"labels"`) {
		t.Fatalf("json: %v\n%s", err, out)
	}

	out, err = runCLI(t, "golden", "-ntrials=3", dir)
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if !strings.Contains(out, "docs.mat") || !strings.Contains(out, "mean of 1") {
		t.Fatalf("golden:\n%s", out)
	}

	if out, err := runCLI(t, "help"); err != nil || !strings.Contains(out, "vcluster") {
		t.Fatal("help")
	}
	if err := os.Mkdir(filepath.Join(dir, "empty"), 0o750); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]string{
		{},
		{"nope"},
		{"vcluster"},
		{"vcluster", mat, "zero"},
		{"vcluster", "-clmethod=agglo", mat, "2"},
		{"vcluster", "-crfun=h2", mat, "2"},
		{"vcluster", "-sim=jaccard", mat, "2"},
		{"vcluster", "-cstype=random", mat, "2"},
		{"vcluster", "-rowmodel=sqrt", mat, "2"},
		{"vcluster", "-colmodel=bm25", mat, "2"},
		{"vcluster", "-bogus", mat, "2"},
		{"vcluster", filepath.Join(dir, "missing.mat"), "2"},
		{"evaluate", truth},
		{"evaluate", truth, filepath.Join(dir, "missing")},
		{"golden"},
		{"golden", filepath.Join(dir, "missing")},
		{"golden", filepath.Join(dir, "empty")},
	} {
		if _, err := runCLI(t, bad...); err == nil {
			t.Errorf("expected an error for %v", bad)
		}
	}
}
