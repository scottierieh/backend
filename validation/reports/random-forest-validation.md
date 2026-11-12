# random-forest (Python) — Validation Report
- **Target**: `random_forest_analysis.py` · Backend: `scottierieh/backend` (Python)
- **Method**: Ran the CLI script on iris; reproduced its pipeline (stratified split, `random_state=42`, and the handler's default balanced class weighting, `class_weight='balanced'`) with a direct `sklearn.ensemble.RandomForestClassifier` fit and compared every classification metric.
## Summary — pass
All 18 checks match: the reported class weighting (`class_weight='balanced'`), test and train accuracy, macro precision / recall / F1, every confusion-matrix cell and the split sizes, against a direct `sklearn.ensemble.RandomForestClassifier` fit on the same stratified split. The reference must carry the same `class_weight='balanced'` as the handler: on iris the balanced weights are all 1, but scikit-learn ≥ 1.8 draws each tree's bootstrap sample from the sample weights, so an unweighted reference grows a different forest there. Package-based. Repro: `validation/validate_random_forest.py`.

## Exploration-tab payloads (33 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
