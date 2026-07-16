# gbm (Python) — Validation Report
- **Target**: `gbm_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn.ensemble.GradientBoostingClassifier` fit.
## Summary — pass (25 checks)
`metrics.accuracy` matches a direct `sklearn.ensemble.GradientBoostingClassifier` fit on the same stratified split. In addition to the generic classifier metrics, the validator now independently recomputes and cross-checks the method-specific boosting block:
- **feature_importance** — per-feature `importance` equals `model.feature_importances_`, `importance_pct` equals `importance / sum * 100`, the array is sorted descending, has one entry per feature, sums to 1.0, and its percentages sum to 100.

Package-based. Repro: `validation/validate_gbm.py`.
