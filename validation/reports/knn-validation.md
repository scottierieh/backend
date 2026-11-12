# K-Nearest Neighbors (Python) — Validation Report
- **Target**: `knn_analysis.py` · `/api/analysis/knn` · sklearn
- **Method**: iris; reproduced StandardScaler → stratified split → `KNeighborsClassifier(n_neighbors=5)` (find_optimal_k off).
## Summary — 30/30 pass
The original 16 checks match: test and train accuracy, macro precision / recall / F1, every confusion-matrix cell and the train/test split sizes, recomputed with scikit-learn on a direct `sklearn.neighbors.KNeighborsClassifier` fit. Package-based. Repro: `validation/validate_knn.py`.

## Exploration-tab payloads (30 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
