# ensemble-stacking (Python) — Validation Report
- **Target**: `ensemble_stacking_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn Voting/StackingClassifier` fit.
## Summary — pass
`metrics.accuracy` matches a direct `sklearn Voting/StackingClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_ensemble_stacking.py`.
