#!/usr/bin/env python3
"""Cross-check CLUTRO's evaluate metrics against scikit-learn.

Prints the reference values for the hand-computed fixture used in
evaluate/internal_test.go and evaluate/external_test.go. Requires
scikit-learn; it is not part of the Go build or tests.

    pip install scikit-learn numpy
    python3 bench/reference/internal_metrics.py
"""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
labels = np.array([0, 0, 1, 1])

print("silhouette         ", silhouette_score(X, labels))          # 0.92929...
print("davies_bouldin     ", davies_bouldin_score(X, labels))      # 0.0707...
print("calinski_harabasz  ", calinski_harabasz_score(X, labels))   # 400.0

x = [0, 0, 1, 1, 2, 2]
y = [0, 0, 1, 2, 2, 2]
print("adjusted_rand      ", adjusted_rand_score(x, y))            # 0.4444...
print("nmi (geometric)    ", normalized_mutual_info_score(x, y, average_method="geometric"))
