# adaboost (Python) — Validation Report
- **Target**: `adaboost_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and independently recomputed both generic classifier metrics and the method-specific boosting blocks against a direct `sklearn.ensemble.AdaBoostClassifier` fit.
## Summary — pass (257 checks, 0 fail)
`metrics.accuracy` matches a direct `sklearn.ensemble.AdaBoostClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_adaboost.py`.

Method-specific blocks now independently validated:
- **estimator_weights** — per-learner SAMME alpha (`model.estimator_weights_`), value-by-value
- **estimator_errors** — per-learner weighted error (`model.estimator_errors_`), value-by-value
- **staged_train_scores / staged_test_scores** — learning curve via `model.staged_score`, every boosting round
- **feature_importance** — impurity importances with `importance_pct`, `normalized_importance`, descending-sort order and rank
- **per_class_metrics** — precision/recall/f1/support per class
- **auc** — multiclass macro one-vs-rest ROC-AUC
- **cv_results** — StratifiedKFold(shuffle, rs=42) accuracy: per-fold scores, mean and std
