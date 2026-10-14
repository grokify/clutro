// Package metric defines the similarity functions clustering algorithms
// optimize over, and the distances they are derived from.
package metric

import (
	"math"

	"github.com/grokify/clutro/matrix"
)

// Similarity scores how alike two rows are; larger is more similar. Row
// norms are passed in so implementations never recompute them.
type Similarity interface {
	// Sim is the similarity of two rows.
	Sim(a, b matrix.SparseVec, normA, normB float32) float64
	// RowToComposite is the similarity of a row to a cluster composite,
	// treating the composite as a vector (for cosine, its direction is the
	// cluster centroid's direction).
	RowToComposite(r matrix.SparseVec, normR float32, c *matrix.Composite) float64
	// Name identifies the metric in output ("cos", "euclid").
	Name() string
}

// Cosine is cosine similarity. On L2-normalized rows it is the dot product;
// otherwise the dot product is divided by the norms. Zero rows have
// similarity 0 to everything.
type Cosine struct{}

// Sim implements Similarity.
func (Cosine) Sim(a, b matrix.SparseVec, normA, normB float32) float64 {
	if normA == 0 || normB == 0 {
		return 0
	}
	d := matrix.Dot(a, b)
	if normA == 1 && normB == 1 {
		return d
	}
	return d / (float64(normA) * float64(normB))
}

// RowToComposite implements Similarity.
func (Cosine) RowToComposite(r matrix.SparseVec, normR float32, c *matrix.Composite) float64 {
	cn := c.Norm()
	if normR == 0 || cn == 0 {
		return 0
	}
	return c.Dot(r) / (float64(normR) * cn)
}

// Name implements Similarity.
func (Cosine) Name() string { return "cos" }

// Euclidean is Euclidean distance expressed as a similarity, 1/(1+d), so
// that algorithms written against Similarity can use it. Distance is also
// available directly.
type Euclidean struct{}

// Distance is the Euclidean distance between two rows.
func (Euclidean) Distance(a, b matrix.SparseVec, normA, normB float32) float64 {
	d2 := float64(normA)*float64(normA) + float64(normB)*float64(normB) - 2*matrix.Dot(a, b)
	if d2 <= 0 {
		return 0
	}
	return math.Sqrt(d2)
}

// Sim implements Similarity as 1/(1+distance).
func (e Euclidean) Sim(a, b matrix.SparseVec, normA, normB float32) float64 {
	return 1 / (1 + e.Distance(a, b, normA, normB))
}

// RowToComposite implements Similarity against the composite's mean (the
// centroid), as k-means does.
func (Euclidean) RowToComposite(r matrix.SparseVec, normR float32, c *matrix.Composite) float64 {
	if c.N == 0 {
		return 0
	}
	n := float64(c.N)
	// ‖r − D/n‖² = ‖r‖² − 2 r·D/n + ‖D‖²/n²
	d2 := float64(normR)*float64(normR) - 2*c.Dot(r)/n + c.Norm2()/(n*n)
	if d2 <= 0 {
		return 1
	}
	return 1 / (1 + math.Sqrt(d2))
}

// Name implements Similarity.
func (Euclidean) Name() string { return "euclid" }

// Parse returns the similarity for a CLUTO-style name: "cos" or "euclid".
func Parse(name string) (Similarity, bool) {
	switch name {
	case "cos", "cosine":
		return Cosine{}, true
	case "euclid", "euclidean":
		return Euclidean{}, true
	}
	return nil, false
}
