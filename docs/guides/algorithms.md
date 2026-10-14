# Algorithms

All algorithms share `partition.Options` (exposed through `clutro.With…`):
similarity, criterion, trials, max iterations, seed, workers, selection
policy, refinement.

| Algorithm | `clutro` | CLUTO | What it does |
|-----------|----------|-------|--------------|
| Repeated bisection | `RepeatedBisection` | `rb` | Split one cluster at a time with a 2-way spherical k-means (several seeded trials), choosing the cluster by `SelectBest` (largest criterion gain) or `SelectLargest`, until k clusters exist. Returns a tree. **Default.** |
| Repeated bisection + refinement | `RepeatedBisection` + `WithRefinement(true)` | `rbr` | As above, then criterion-driven single-row moves over the k-way result. Typically 1–4% better criterion than `rb`. |
| Direct | `Direct` | `direct` | k-means++ seeding and Lloyd iterations for all k at once, then refinement. |
| k-means | `KMeans` | — | k-means++ and Lloyd only. Spherical k-means under cosine; classic under Euclidean. |

## Seeding

k-means and every bisection start from **greedy k-means++**: each new center
is chosen among 2 + ln(k) candidates drawn by D² sampling, keeping the one
that lowers the total potential most (as scikit-learn does). On text with
cosine geometry, plain k-means++ has weak contrast (distances cluster around
0.7–0.95) and found the planted structure of a 6-cluster test corpus only 17%
of the time per trial; greedy seeding raised that to 47%. With the default
10 trials the chance of missing it is below 0.2%.

## Trials

Each k-means or bisection runs `Trials` times (default 10) from independent
seeded streams and keeps the partition with the best criterion value.
Trials are independent, so running them in parallel is a pure optimization;
the result must not change.

## Lloyd iterations

Rows are assigned to the composite they are most similar to; ties break to
the lowest cluster index; rows are visited in index order; the assignment
sweep is parallel but writes positionally. Empty clusters are reseeded with
the row least similar to its own composite — deterministically.

## Choosing

- **Text, TF-IDF, cosine:** `RepeatedBisection` with refinement. Choose
  k ≈ rows/12 for "related items" boxes, or bisect further and `CutMaxSize`.
- **Dense embeddings:** `RepeatedBisection` or `KMeans` under cosine after
  normalizing rows; `Direct` when k is small and you want the fastest
  k-way answer.
- **Euclidean data:** `KMeans` or `Direct` with `metric.Euclidean{}`.
- **Unknown k:** bisect to a generous k and cut the tree.

## Complexity

Bisection is O(trials · iterations · n_cluster · nnz) per split; the whole
tree is roughly O(k · trials · iterations · n · nnz / k) = O(trials ·
iterations · n · nnz) — linear in the data for a fixed k. Refinement is
O(passes · n · k · nnz). On a laptop, 103 real corpora (up to 6,416 rows,
96k nonzeros, k = 535) cluster in 6 s with `rb` and 15 s with `rbr`.
