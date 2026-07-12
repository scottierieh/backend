# Naive Bayes (Python) — Validation Report
- **Target**: `naive_bayes_analysis.py` · `/api/analysis/naive-bayes` · sklearn
- **Method**: iris; reproduced stratified split → `GaussianNB(var_smoothing=1e-9)` (Gaussian NB is unscaled).
## Summary — 1/1 pass
`metrics.accuracy` matches a direct `sklearn.naive_bayes.GaussianNB` fit. Package-based. Repro: `validation/validate_naive_bayes.py`.
