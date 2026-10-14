package par

import (
	"context"
	"testing"
)

func TestChunks(t *testing.T) {
	cs := Chunks(10, 4)
	if len(cs) != 4 || cs[0] != (Chunk{0, 0, 3}) || cs[1] != (Chunk{1, 3, 6}) || cs[2] != (Chunk{2, 6, 8}) || cs[3] != (Chunk{3, 8, 10}) {
		t.Fatalf("chunks: %+v", cs)
	}
	if cs := Chunks(2, 8); len(cs) != 2 {
		t.Fatalf("more workers than rows: %+v", cs)
	}
	if Chunks(0, 4) != nil {
		t.Fatal("empty")
	}
	if cs := Chunks(5, 0); len(cs) != 1 || cs[0].End != 5 {
		t.Fatalf("zero workers: %+v", cs)
	}
	if Workers(0) < 1 || Workers(1) != 1 {
		t.Fatal("workers")
	}
}

func TestRunAndSumAreOrderIndependent(t *testing.T) {
	const n = 100000
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = float64(i%7) * 0.1
	}
	var ref float64
	for _, v := range vals {
		ref += v
	}
	seq, err := Sum(context.Background(), n, 1, func(c Chunk) float64 {
		var s float64
		for i := c.Start; i < c.End; i++ {
			s += vals[i]
		}
		return s
	})
	if err != nil {
		t.Fatal(err)
	}
	for trial := 0; trial < 20; trial++ {
		got, err := Sum(context.Background(), n, 8, func(c Chunk) float64 {
			var s float64
			for i := c.Start; i < c.End; i++ {
				s += vals[i]
			}
			return s
		})
		if err != nil {
			t.Fatal(err)
		}
		if got != seq && trial > 0 {
			// Different chunking than sequential may differ by rounding, but
			// repeated parallel runs must be bit-identical.
			t.Fatalf("parallel sums differ between runs")
		}
		seq = got
	}
	out := make([]int, n)
	if err := Run(context.Background(), n, 8, func(c Chunk) {
		for i := c.Start; i < c.End; i++ {
			out[i] = c.Worker
		}
	}); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < n; i++ {
		if out[i] < out[i-1] {
			t.Fatal("chunks must be contiguous and ordered")
		}
	}
}

func TestRunCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Run(ctx, 10, 2, func(Chunk) {}); err == nil {
		t.Fatal("cancelled context must be reported")
	}
}
