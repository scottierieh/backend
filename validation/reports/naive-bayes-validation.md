# Naive Bayes (Python) — Validation Report
- **Target**: `naive_bayes_analysis.py` · `/api/analysis/naive-bayes` · sklearn
- **Method**: iris; reproduced stratified split → `GaussianNB(var_smoothing=1e-9)` (Gaussian NB is unscaled).
## Summary — 30/30 pass
The original 16 checks match: test and train accuracy, macro precision / recall / F1, every confusion-matrix cell and the train/test split sizes, recomputed with scikit-learn on a direct `sklearn.naive_bayes.GaussianNB` fit. Package-based. Repro: `validation/validate_naive_bayes.py`.

## Exploration-tab payloads (30 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
