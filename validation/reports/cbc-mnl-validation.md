# cbc-mnl (Python) — Validation Report
- **Target**: `conjoint_analysis.py` (`POST /api/analysis/conjoint`, the survey CBC tab's aggregate MNL) · Backend: `scottierieh/backend` (Python)
- **Method**: Posted simulated choice data in the exact row format the app sends (respondent_id, task_id, alternative 1..n | "none", choice, one column per attribute; 250 respondents × 10 tasks × 3 alternatives, with and without a "none" alternative) to the real route; reproduced it with **statsmodels ConditionalLogit** on an independently built dummy design.
## Summary — pass (162/162)
- Coefficients, SE, z, p and 95% CI per level (alphabetical-first reference level), the none constant, log-likelihood, null log-likelihood, AIC, AICc, BIC (n = choice sets), McFadden and adjusted McFadden R², hit rate, importance, data summary, share-of-preference simulation, WTP point estimates (continuous-price refit with the same none constant), segmentation bookkeeping.
- Recovery: planted utilities within 0.2, the none constant within 0.2, WTP within 35%.
- Tolerances are about 1.5% of a standard error: the backend's BFGS stops at scipy's default gradient tolerance, which leaves coefficients ~2×10⁻⁴ (≈0.3% of an SE) from the exact optimum — no practical effect.
## Findings
1. **WTP confidence interval omits cov(β_level, β_price).** The delta-method SE is `|1/βp|·√(se_k² + (β_k/βp)²·se_p²)`; the full delta method adds `−2(β_k/βp)·cov(β_k, βp)`. On the test data the reported 95% half-widths are 1.3–4.9% wider than the full delta method (conservative, but not the stated method).
2. **Segmentation** fits a separate conditional logit per respondent (often ~10 choices for 7+ parameters, so many fits separate and are clipped to ±10) and clusters those with K-means. The bookkeeping is correct; the method is fragile — the latent class estimator (R `cbc-choice-model`) is the sound way to find segments.
Repro: `validation/validate_cbc_mnl.py`.
