# Discriminant Analysis (LDA/QDA) (Python) — Validation Report
- **Target**: `discriminant_analysis.py` · `/api/analysis/discriminant` · sklearn
- **Method**: iris; reproduced stratified split → StandardScaler(fit on train) → `LinearDiscriminantAnalysis(solver='svd')`.
## Summary — 1/1 pass
`metrics.accuracy` matches a direct `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` fit. Package-based. Repro: `validation/validate_discriminant.py`.
