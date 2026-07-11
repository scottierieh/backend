# catboost (Python) — Validation Report
- **Target**: `catboost_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `catboost.CatBoostClassifier` fit.
## Summary — pass
`metrics.accuracy` matches a direct `catboost.CatBoostClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_catboost.py`.
