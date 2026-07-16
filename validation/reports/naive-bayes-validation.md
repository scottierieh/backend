# Naive Bayes (Python) — Validation Report
- **Target**: `naive_bayes_analysis.py` · `/api/analysis/naive-bayes` · sklearn
- **Method**: iris; reproduced stratified split → `GaussianNB(var_smoothing=1e-9)` (Gaussian NB is unscaled).
## Summary — 47/47 pass
Generic classifier metrics (accuracy, train accuracy, macro precision/recall/F1, per-cell confusion matrix, split sizes) match a direct `sklearn.naive_bayes.GaussianNB` fit. In addition, the method-specific blocks a STEP6 page shows are now independently recomputed and cross-checked:
- **class_priors** — per-class `GaussianNB.class_prior_` (training class frequencies)
- **feature_importance** — Gaussian inverse-variance heuristic `1/(mean_var+1e-10)` normalized, plus `importance_pct` and descending-sort ordering
- **cv_results** — `StratifiedKFold(5, shuffle, random_state=42)` `cross_val_score` per-fold scores, `cv_mean`, `cv_std`
- **per_class_metrics** — per-class precision/recall/F1/support

Package-based. Repro: `validation/validate_naive_bayes.py`.
