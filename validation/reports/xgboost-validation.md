# xgboost (Python) — Validation Report
- **Target**: `xgboost_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `xgboost.XGBClassifier` fit.
## Summary — pass (45 checks)
`metrics.accuracy` matches a direct `xgboost.XGBClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_xgboost.py`.

### Method-specific blocks now independently validated
- **feature_importance** — `importance`, `importance_pct`, `normalized_importance` and descending `rank` recomputed from `model.feature_importances_`.
- **metrics.auc** — binary ROC-AUC recomputed from positive-class probabilities (`roc_auc_score`).
- **per_class_metrics** — per-class precision/recall/f1/support recomputed one-vs-rest from the model's own predictions.
- **cv_results** — `cv_folds`, `cv_mean`, `cv_std`, and per-fold `cv_scores` recomputed via `StratifiedKFold` accuracy over the full dataset (backend contract).
- **tree_rules** — `n_leaves` recomputed from the booster's tree-0 text dump.
