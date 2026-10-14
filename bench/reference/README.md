# Reference cross-checks

Scripts here reproduce, with scikit-learn, the reference values that the
Go tests assert. They are run by hand when a metric is added or changed; the
Go test suite never calls Python.

| Script | Checks | Go test |
|--------|--------|---------|
| `internal_metrics.py` | silhouette, Davies–Bouldin, Calinski–Harabasz on a four-point fixture; adjusted Rand index and NMI on a six-label fixture | `evaluate/internal_test.go`, `evaluate/external_test.go` |

The four-point fixture is also solvable by hand, which is how the asserted
values were first derived:

- two pairs `(0,0),(0,1)` and `(10,10),(10,11)`: within-pair distance 1,
  cross distances √200 and √221;
- silhouette: a = 1 for every point; the outer points see b = (√200 + √221)/2 and the inner points b = (√181 + √200)/2, so the mean is 0.92929;
- Davies–Bouldin = (0.5 + 0.5)/√200 → 0.070711;
- Calinski–Harabasz = (200/1)/(1/2) → 400.

CLUTRO's NMI uses the geometric mean of the entropies, scikit-learn's
`average_method="geometric"`.
