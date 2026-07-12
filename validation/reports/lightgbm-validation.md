# lightgbm (Python) — Validation Report
- **Target**: `lightgbm_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `lightgbm.LGBMClassifier` fit.
## Summary — pass
`metrics.accuracy` matches a direct `lightgbm.LGBMClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_lightgbm.py`.
