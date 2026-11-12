# cross-validation (Python) — Validation Report
- **Target**: `cross_validation_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script; reproduced its computation with **sklearn cross_val_score (KFold)** and compared the key result.
## Summary — pass
Result matches a direct reproduction with sklearn cross_val_score (KFold). Repro: `validation/validate_cross_validation.py`.

## Exploration-tab payloads (54 checks in total)
- `fold_scores` / `primary_metric`: primary metric first, its per-fold list is `cv_results.scores` (mean equals the reported CV mean and the reference `cross_val_score`); the other per-fold metrics equal those recomputed from the reference out-of-fold predictions; regression RMSE run checked too.
