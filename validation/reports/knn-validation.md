# K-Nearest Neighbors (Python) — Validation Report
- **Target**: `knn_analysis.py` · `/api/analysis/knn` · sklearn
- **Method**: iris; reproduced StandardScaler → stratified split → `KNeighborsClassifier(n_neighbors=5)` (find_optimal_k off), plus a second run with find_optimal_k on for the K-search block.
## Summary — 61/61 pass
Generic classifier metrics (`metrics.accuracy`, train accuracy, macro precision/recall/F1, confusion matrix) match a direct `sklearn.neighbors.KNeighborsClassifier` fit. Beyond those, the following method-specific STEP6 blocks are now independently recomputed and cross-checked:
- **cross-validation** (`cv_results`) — `cv_mean`, `cv_std` and each fold score from `StratifiedKFold(n_splits=5, shuffle, random_state=42)` on the full scaled X.
- **per-class metrics** (`per_class_metrics`) — precision / recall / f1 / support per class from `classification_report`.
- **feature importance** (`feature_importance`) — permutation importance mean and std (`permutation_importance`, n_repeats=10, random_state=42).
- **optimal-K search** (`k_search_result`) — `optimal_k`, `optimal_score`, and every `k_scores` mean/std from the CV grid over `k_range`.

Package-based. Repro: `validation/validate_knn.py`.
