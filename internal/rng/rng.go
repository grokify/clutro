// Package rng derives independent, reproducible random streams from one
// caller-supplied seed, so that every random choice an algorithm makes is a
// pure function of (seed, purpose, index) and never of scheduling.
package rng

import (
	"hash/fnv"
	"math/rand/v2"
)

// SubStream returns a PCG generator for a named purpose and index under the
// given seed. The same (seed, name, idx) always yields the same stream;
// different names or indices yield independent streams.
func SubStream(seed uint64, name string, idx int) *rand.Rand {
	h := fnv.New64a()
	_, _ = h.Write([]byte(name)) // fnv never returns an error
	var b [8]byte
	for i := 0; i < 8; i++ {
		b[i] = byte(uint64(idx) >> (8 * i))
	}
	_, _ = h.Write(b[:])
	return rand.New(rand.NewPCG(seed, h.Sum64())) //nolint:gosec // deterministic streams are the point
}

// Perm returns a seeded permutation of 0..n-1.
func Perm(r *rand.Rand, n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	r.Shuffle(n, func(i, j int) { p[i], p[j] = p[j], p[i] })
	return p
}
