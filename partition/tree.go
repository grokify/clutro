package partition

import "sort"

// Tree is the binary hierarchy produced by repeated bisection: each internal
// node is a cluster that was split into its two children; leaves are the
// final clusters in label order.
type Tree struct {
	Root *Node
}

// Node is one cluster in the bisection tree.
type Node struct {
	// Rows are the row indices in this cluster, ascending.
	Rows []int32
	// Label is the leaf's cluster index in the final assignment, or -1 for
	// an internal node.
	Label int32
	// Criterion is the cluster's own contribution to the objective (for I2,
	// the norm of its composite) at the time it was formed.
	Criterion   float64
	Left, Right *Node
}

// IsLeaf reports whether the node was not split further.
func (n *Node) IsLeaf() bool { return n.Left == nil && n.Right == nil }

// Leaves returns the leaves in label order.
func (t *Tree) Leaves() []*Node {
	var out []*Node
	var walk func(n *Node)
	walk = func(n *Node) {
		if n == nil {
			return
		}
		if n.IsLeaf() {
			out = append(out, n)
			return
		}
		walk(n.Left)
		walk(n.Right)
	}
	walk(t.Root)
	sort.Slice(out, func(i, j int) bool { return out[i].Label < out[j].Label })
	return out
}

// CutMaxSize re-partitions the rows by stopping the descent at the first
// node with at most maxSize rows, so most clusters hold about maxSize rows
// or fewer without re-running the clustering. The result is relabelled
// 0..K-1 in tree order.
func (t *Tree) CutMaxSize(maxSize int) Assignment {
	return t.cut(func(n *Node) bool { return len(n.Rows) <= maxSize })
}

// CutK re-partitions into the first k clusters formed by the bisection
// sequence (the clusters that existed when the k-th cluster was created),
// approximated by taking the k shallowest-split nodes in tree order.
func (t *Tree) CutK(k int) Assignment {
	if t.Root == nil || k < 1 {
		return Assignment{}
	}
	// Repeatedly expand the largest frontier node until k nodes exist,
	// mirroring SelectLargest; exact for trees built with SelectLargest.
	frontier := []*Node{t.Root}
	for len(frontier) < k {
		bi, bn := -1, -1
		for i, n := range frontier {
			if !n.IsLeaf() && len(n.Rows) > bn {
				bi, bn = i, len(n.Rows)
			}
		}
		if bi < 0 {
			break
		}
		n := frontier[bi]
		frontier = append(frontier[:bi], append([]*Node{n.Left, n.Right}, frontier[bi+1:]...)...)
	}
	return labelFrontier(t, frontier)
}

func (t *Tree) cut(stop func(*Node) bool) Assignment {
	if t.Root == nil {
		return Assignment{}
	}
	var frontier []*Node
	var walk func(n *Node)
	walk = func(n *Node) {
		if n.IsLeaf() || stop(n) {
			frontier = append(frontier, n)
			return
		}
		walk(n.Left)
		walk(n.Right)
	}
	walk(t.Root)
	return labelFrontier(t, frontier)
}

func labelFrontier(t *Tree, frontier []*Node) Assignment {
	labels := make([]int32, len(t.Root.Rows))
	for l, n := range frontier {
		for _, r := range n.Rows {
			labels[r] = int32(l)
		}
	}
	return Assignment{Labels: labels, K: len(frontier)}
}
