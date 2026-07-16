# Discriminant Analysis (LDA/QDA) (Python) — Validation Report
- **Target**: `discriminant_analysis.py` · `/api/analysis/discriminant` · sklearn
- **Method**: iris; reproduced stratified split → StandardScaler(fit on train) → `LinearDiscriminantAnalysis(solver='svd')`.
## Summary — 64/64 pass
Generic classifier metrics match a direct `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` fit, and the SPSS-level method-specific blocks the STEP6 page shows are each independently recomputed and cross-checked. Package-based. Repro: `validation/validate_discriminant.py`.

Method-specific blocks now validated:
- `lda_info`: class priors, explained variance ratio, per-class means (`means_`).
- `lda_statistics.eigenvalues`: eigenvalues of `pinv(S_W)·S_B`, canonical correlations, variance-explained %.
- `lda_statistics.wilks_lambda`: Λ, χ² statistic, degrees of freedom.
- `lda_statistics.structure_matrix`: feature-to-discriminant-score correlations.
- `lda_statistics.group_centroids`: mean LD scores per class.
- `lda_statistics.anova_f_statistics`: one-way ANOVA F and p per feature.
