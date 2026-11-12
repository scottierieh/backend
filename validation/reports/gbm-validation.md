# gbm (Python) — Validation Report
- **Target**: `gbm_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline and compared test accuracy to a direct `sklearn.ensemble.GradientBoostingClassifier` fit.
## Summary — pass
`metrics.accuracy` matches a direct `sklearn.ensemble.GradientBoostingClassifier` fit on the same stratified split. Package-based. Repro: `validation/validate_gbm.py`.

## Exploration-tab payloads (53 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
- `learning_curve_data`: every round's train / validation metric, the last round and `best_iteration` (1-based) equal the library's own per-iteration record on the reproduced model.
