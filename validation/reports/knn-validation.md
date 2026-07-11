# K-Nearest Neighbors (Python) — Validation Report
- **Target**: `knn_analysis.py` · `/api/analysis/knn` · sklearn
- **Method**: iris; reproduced StandardScaler → stratified split → `KNeighborsClassifier(n_neighbors=5)` (find_optimal_k off).
## Summary — 1/1 pass
`metrics.accuracy` matches a direct `sklearn.neighbors.KNeighborsClassifier` fit. Package-based. Repro: `validation/validate_knn.py`.
