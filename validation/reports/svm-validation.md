# SVM (Python) — Validation Report
- **Target**: `svm_analysis.py` · `/api/analysis/svm` · Backend: `scottierieh/backend` (sklearn)
- **Method**: Ran the CLI script on iris; reproduced its pipeline (StandardScaler → stratified split → LabelEncoder → `sklearn.svm.SVC`).
## Summary — 16/16 pass
| Field | Basis | Result |
|---|---|---|
| `metrics.accuracy` | `accuracy_score` of `SVC(kernel='rbf',C=1,gamma='scale')` on the test split | ✅ |
| `metrics.train_accuracy` | train-split accuracy | ✅ |
| macro precision / recall / F1 | `precision_score` / `recall_score` / `f1_score` (macro) on the test split | ✅ |
| `confusion_matrix` (every cell) | `confusion_matrix` on the test split | ✅ |
| `n_train` / `n_test` | split sizes | ✅ |
Package-based (`sklearn.svm.SVC`); metrics match a direct sklearn fit exactly. Repro: `validation/validate_svm.py`.
