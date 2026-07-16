# SVM (Python) — Validation Report
- **Target**: `svm_analysis.py` · `/api/analysis/svm` · Backend: `scottierieh/backend` (sklearn)
- **Method**: Ran the CLI script on iris; reproduced its pipeline (StandardScaler → stratified split → LabelEncoder → `sklearn.svm.SVC`).
## Summary — 39/39 pass
| Field | Basis | Result |
|---|---|---|
| `metrics.accuracy` | `accuracy_score` of `SVC(kernel='rbf',C=1,gamma='scale')` on the test split | ✅ |
| `metrics.train_accuracy` | train-split accuracy | ✅ |
Package-based (`sklearn.svm.SVC`); metrics match a direct sklearn fit exactly. Repro: `validation/validate_svm.py`.

### Method-specific blocks now validated
- **`metrics.n_support_vectors`** — total support vectors = `sum(model.n_support_)`.
- **`support_per_class`** — per-class label + support-vector count from `model.n_support_` (in `LabelEncoder` class order).
- **`cv_results`** — 5-fold `StratifiedKFold(shuffle,random_state=42)` accuracy on the full scaled X: `cv_mean`, `cv_std`, and each fold score.
- **`feature_importance`** — permutation importance (`n_repeats=10, random_state=42`): per-feature `importance` and `std`, plus descending-rank ordering.
