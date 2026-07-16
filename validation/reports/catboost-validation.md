# catboost (Python) — Validation Report
- **Target**: `catboost_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `catboost.CatBoostClassifier` fit.
## Summary — pass (41 checks, 0 fail)
`metrics.accuracy` matches a direct `catboost.CatBoostClassifier` fit on the same stratified split, plus generic classifier metrics (train/test accuracy, macro precision/recall/F1, full confusion matrix, split sizes). Package-based. Repro: `validation/validate_catboost.py`.

Method-specific boosting blocks now independently recomputed and validated:
- **best_iteration** — reproduced via `Pool` + `eval_set` + `use_best_model` with the handler's default `early_stopping_rounds=20`.
- **feature_importance** — native `PredictionValuesChange` values, `importance_pct`, `normalized_importance`, and descending rank order.
- **per_class_metrics** — per-class precision/recall/F1/support aligned to the `LabelEncoder` class order.
- **shap_importance** — native exact CatBoost SHAP values (bias column dropped, mean of absolute values per feature).
- **cv_results** — 5-fold `StratifiedKFold` (shuffle, seed 42) accuracy `cv_mean` / `cv_std`.
