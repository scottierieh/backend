# Naive Bayes (Python) — Validation Report
- **Target**: `naive_bayes_analysis.py` · `/api/analysis/naive-bayes` · sklearn
- **Method**: iris; reproduced stratified split → `GaussianNB(var_smoothing=1e-9)` (Gaussian NB is unscaled).
## Summary — 16/16 pass
All 16 checks match: test and train accuracy, macro precision / recall / F1, every confusion-matrix cell and the train/test split sizes, recomputed with scikit-learn on a direct `sklearn.naive_bayes.GaussianNB` fit. Package-based. Repro: `validation/validate_naive_bayes.py`.
