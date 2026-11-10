# Discriminant Analysis (LDA/QDA) (Python) — Validation Report
- **Target**: `discriminant_analysis.py` · `/api/analysis/discriminant` · sklearn
- **Method**: iris; reproduced stratified split → StandardScaler(fit on train) → `LinearDiscriminantAnalysis(solver='svd')`.
## Summary — 16/16 pass
All 16 checks match: test and train accuracy, macro precision / recall / F1, every confusion-matrix cell and the train/test split sizes, recomputed with scikit-learn on a direct `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` fit. Package-based. Repro: `validation/validate_discriminant.py`.
