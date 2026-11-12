# SVM (Python) — Validation Report
- **Target**: `svm_analysis.py` · `/api/analysis/svm` · Backend: `scottierieh/backend` (sklearn)
- **Method**: Ran the CLI script on iris; reproduced its pipeline (StandardScaler → stratified split → LabelEncoder → `sklearn.svm.SVC`).
## Summary — 30/30 pass
| Field | Basis | Result |
|---|---|---|
| `metrics.accuracy` | `accuracy_score` of `SVC(kernel='rbf',C=1,gamma='scale')` on the test split | ✅ |
| `metrics.train_accuracy` | train-split accuracy | ✅ |
| macro precision / recall / F1 | `precision_score` / `recall_score` / `f1_score` (macro) on the test split | ✅ |
| `confusion_matrix` (every cell) | `confusion_matrix` on the test split | ✅ |
| `n_train` / `n_test` | split sizes | ✅ |
Package-based (`sklearn.svm.SVC`); metrics match a direct sklearn fit exactly. Repro: `validation/validate_svm.py`.

## Exploration-tab payloads (30 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
