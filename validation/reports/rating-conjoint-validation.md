# rating-conjoint (Python) — Validation Report
- **Target**: `rc_analysis.py` (`POST /api/analysis/rating-conjoint`, the survey Rating Conjoint tab) · Backend: `scottierieh/backend` (Python)
- **Method**: Posted simulated data (12 estimation + 2 holdout profiles, 80 respondents, 1–10 ratings from known part-worths with respondent heterogeneity, a few blanks, a segment variable) to the real route; reproduced every number with **statsmodels OLS** on an independently built effects-coded design (patsy `Sum` coding).
## Summary — pass (167/167)
- Part-worths of every level, including the reference level (−Σβ), with SE, t, p and 95% CI (statsmodels `t_test` contrasts); intercept and its SE/p.
- R², adjusted R², RMSE (√(SSR/n)), log-likelihood, AIC, AICc, BIC, overall F and its p — the hand-written t/F p-value functions agree with scipy-based statsmodels to 1e-6.
- Importance (range share), WTP (least-squares price slope of the price utilities), holdout MSE/MAE/R², per-respondent part-worths, per-segment importance, and the respondent-level bootstrap (same seed, refitted with statsmodels).
- Recovery: fitted part-worths within 0.25 of the planted ones.
## Notes
- Not checked by data: a profile whose level is missing is coded as the attribute's last (reference) level, and a level not in the attribute's list as its first level (`build_effects_matrix`). Survey-generated profiles always carry valid levels; hand-edited data would be miscoded silently.
Repro: `validation/validate_rating_conjoint.py`.
