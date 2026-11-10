# K-Nearest Neighbors (Python) — Validation Report
- **Target**: `knn_analysis.py` · `/api/analysis/knn` · sklearn
- **Method**: iris; reproduced StandardScaler → stratified split → `KNeighborsClassifier(n_neighbors=5)` (find_optimal_k off).
## Summary — 16/16 pass
All 16 checks match: test and train accuracy, macro precision / recall / F1, every confusion-matrix cell and the train/test split sizes, recomputed with scikit-learn on a direct `sklearn.neighbors.KNeighborsClassifier` fit. Package-based. Repro: `validation/validate_knn.py`.
