# lightgbm (Python) — Validation Report
- **Target**: `lightgbm_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `lightgbm.LGBMClassifier` fit.
## Summary — pass (43 checks)
`metrics.accuracy` matches a direct `lightgbm.LGBMClassifier` fit on the same stratified split, plus independent recomputation of the method-specific boosting blocks the STEP6 page shows. Package-based. Repro: `validation/validate_lightgbm.py`.

Method-specific blocks now validated:
- **best_iteration** — vs `LGBMClassifier.best_iteration_` (eval_set + early_stopping(20)).
- **feature_importance** — split-count `feature_importances_`, `importance_pct` (share of total), `normalized_importance` (share of max), and the descending rank/order.
- **per_class_metrics** — per-class precision/recall/f1/support vs scikit-learn on encoded labels.
- **metrics.auc** — binary ROC-AUC on the positive-class probability.
- **cv_results** — `cv_mean`/`cv_std`/`cv_folds` from `StratifiedKFold(5, shuffle, random_state=42)` accuracy.

Not validated: `train_history`/learning curve is only rendered into `learning_plot` and not emitted as a numeric block in the response, so it has no reproducible value to cross-check.
