# gbm (Python) — Validation Report
- **Target**: `gbm_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn.ensemble.GradientBoostingClassifier` fit.
## Summary — pass
`metrics.accuracy` matches a direct `sklearn.ensemble.GradientBoostingClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_gbm.py`.
