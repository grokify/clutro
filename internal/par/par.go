// Package par runs row-range work across a bounded set of workers in a way
// that keeps results deterministic: every worker gets a fixed contiguous
// chunk, writes only to positions it owns, and partial results are combined
// in worker order rather than completion order.
package par

import (
	"context"
	"runtime"
	"sync"
)

// Workers clamps a requested parallelism to [1, NumCPU] (0 means NumCPU).
func Workers(requested int) int {
	n := runtime.NumCPU()
	if requested > 0 && requested < n {
		return requested
	}
	if requested > n {
		return requested
	}
	return n
}

// Chunk is a contiguous half-open row range assigned to one worker.
type Chunk struct{ Worker, Start, End int }

// Chunks splits [0, n) into at most workers contiguous ranges of nearly equal
// size, in order. Fewer chunks are returned when n is small.
func Chunks(n, workers int) []Chunk {
	if n <= 0 {
		return nil
	}
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	out := make([]Chunk, 0, workers)
	size, rem := n/workers, n%workers
	start := 0
	for w := 0; w < workers; w++ {
		end := start + size
		if w < rem {
			end++
		}
		out = append(out, Chunk{Worker: w, Start: start, End: end})
		start = end
	}
	return out
}

// Run calls fn once per chunk concurrently and waits. fn must write only to
// positions in its range. It returns ctx.Err() if the context was cancelled
// before or while running; fn should check ctx itself for long chunks.
func Run(ctx context.Context, n, workers int, fn func(c Chunk)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	chunks := Chunks(n, workers)
	if len(chunks) <= 1 {
		for _, c := range chunks {
			fn(c)
		}
		return ctx.Err()
	}
	var wg sync.WaitGroup
	wg.Add(len(chunks))
	for _, c := range chunks {
		go func(c Chunk) {
			defer wg.Done()
			fn(c)
		}(c)
	}
	wg.Wait()
	return ctx.Err()
}

// Sum runs fn per chunk, each returning a partial float64, and adds the
// partials in worker order so the result does not depend on scheduling.
func Sum(ctx context.Context, n, workers int, fn func(c Chunk) float64) (float64, error) {
	chunks := Chunks(n, workers)
	partial := make([]float64, len(chunks))
	err := Run(ctx, n, workers, func(c Chunk) { partial[c.Worker] = fn(c) })
	var s float64
	for _, p := range partial {
		s += p
	}
	return s, err
}
