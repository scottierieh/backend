# random-forest (Python) — Validation Report
- **Target**: `random_forest_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn.ensemble.RandomForestClassifier` fit.
## Summary — pass (45/45 checks)
`metrics.accuracy` matches a direct `sklearn.ensemble.RandomForestClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_random_forest.py`.

### Method-specific blocks independently validated
- **oob_score** — out-of-bag generalization estimate vs `model.oob_score_`.
- **feature_importance** — Gini `importance`, `importance_pct` (imp/sum·100), `normalized_importance` (imp/max), and descending `rank` order vs `model.feature_importances_`.
- **cv_results** — `cv_mean`, `cv_std`, and each fold in `cv_scores` vs `cross_val_score` with `StratifiedKFold(shuffle, random_state=42)`.
- **perm_importance** — per-feature `importance_mean`/`importance_std` vs `sklearn.inspection.permutation_importance` (n_repeats=10, random_state=42) on the test set.
