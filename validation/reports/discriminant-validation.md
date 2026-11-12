# Discriminant Analysis (LDA/QDA) (Python) — Validation Report
- **Target**: `discriminant_analysis.py` · `/api/analysis/discriminant` · sklearn
- **Method**: iris; reproduced stratified split → StandardScaler(fit on train) → `LinearDiscriminantAnalysis(solver='svd')`.
## Summary — 44/44 pass
The original 16 checks match: test and train accuracy, macro precision / recall / F1, every confusion-matrix cell and the train/test split sizes, recomputed with scikit-learn on a direct `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` fit. Package-based. Repro: `validation/validate_discriminant.py`.

## Exploration-tab payloads (44 checks in total)
- `threshold_curve` (binary breast-cancer run, 10 features): equals `sklearn.metrics.roc_curve(drop_intermediate=False)` on the test rows and positive-class probabilities of an independently reproduced model (same split, scaling and class weighting) — every threshold/FPR/TPR, `n_pos`/`n_neg`, labels, `tpr·n_pos` and `fpr·n_neg` integer-valued, and the curve's AUC equals the reported `metrics.auc`.
