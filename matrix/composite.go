package matrix

import "math"

// Composite is the sum of a set of rows — a cluster's composite vector —
// kept dense in float64 with its squared norm maintained incrementally, so
// adding or removing a row costs O(nnz(row)) rather than O(cols).
type Composite struct {
	Sum   []float64
	N     int
	norm2 float64
}

// NewComposite returns an empty composite over nCols columns.
func NewComposite(nCols int) *Composite {
	return &Composite{Sum: make([]float64, nCols)}
}

// Reset empties the composite.
func (c *Composite) Reset() {
	for i := range c.Sum {
		c.Sum[i] = 0
	}
	c.N = 0
	c.norm2 = 0
}

// Dot is the inner product of the composite with a row.
func (c *Composite) Dot(r SparseVec) float64 { return DotDense(r, c.Sum) }

// Norm is the L2 norm of the composite.
func (c *Composite) Norm() float64 {
	if c.norm2 <= 0 {
		return 0
	}
	return math.Sqrt(c.norm2)
}

// Norm2 is the squared L2 norm.
func (c *Composite) Norm2() float64 { return c.norm2 }

// Add adds a row; normR is the row's L2 norm.
func (c *Composite) Add(r SparseVec, normR float32) {
	c.norm2 += 2*c.Dot(r) + float64(normR)*float64(normR)
	for k, idx := range r.Idx {
		c.Sum[idx] += float64(r.Val[k])
	}
	c.N++
}

// Remove subtracts a row previously added; normR is the row's L2 norm.
func (c *Composite) Remove(r SparseVec, normR float32) {
	for k, idx := range r.Idx {
		c.Sum[idx] -= float64(r.Val[k])
	}
	c.norm2 += -2*c.Dot(r) - float64(normR)*float64(normR)
	c.N--
	if c.N == 0 || c.norm2 < 0 {
		// Guard against drift when a cluster empties.
		c.norm2 = 0
		if c.N == 0 {
			for i := range c.Sum {
				c.Sum[i] = 0
			}
		}
	}
}

// NormIfAdded returns ‖D + r‖ without modifying the composite.
func (c *Composite) NormIfAdded(r SparseVec, normR float32) float64 {
	n2 := c.norm2 + 2*c.Dot(r) + float64(normR)*float64(normR)
	if n2 <= 0 {
		return 0
	}
	return math.Sqrt(n2)
}

// NormIfRemoved returns ‖D − r‖ without modifying the composite, for a row
// that is currently part of it.
func (c *Composite) NormIfRemoved(r SparseVec, normR float32) float64 {
	n2 := c.norm2 - 2*c.Dot(r) + float64(normR)*float64(normR)
	if n2 <= 0 {
		return 0
	}
	return math.Sqrt(n2)
}

// Recompute recalculates the squared norm from Sum, discarding accumulated
// rounding; call it occasionally in long refinement loops.
func (c *Composite) Recompute() {
	var s float64
	for _, v := range c.Sum {
		s += v * v
	}
	c.norm2 = s
}
