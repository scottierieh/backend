# ensemble-stacking (Python) — Validation Report
- **Target**: `ensemble_stacking_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn Voting/StackingClassifier` fit.
## Summary — pass (52 checks, 0 fail)
`metrics.accuracy` matches a direct `sklearn Voting/StackingClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_ensemble_stacking.py`.

Beyond the generic classifier metrics, the validator now independently recomputes the method-specific blocks the STEP6 ensemble page renders:
- **individual_scores** — each base learner's standalone test accuracy plus the ensemble's own score.
- **base_estimators** — the ordered list of base-learner names.
- **cv_results** — per-fold `cv_scores`, `cv_mean`, `cv_std`, `cv_folds` from a `StratifiedKFold(shuffle, rs=42)` accuracy run on label-encoded targets.
- **per_class_metrics** — precision / recall / f1 / support per class (label order = `le.classes_`).
- **perm_importance** — permutation-importance `importance_mean` and `importance_std` per feature (`n_repeats=10`, `rs=42`).
- **feature_importance** — normalized `importance_pct` (positive-share of clipped permutation means).
