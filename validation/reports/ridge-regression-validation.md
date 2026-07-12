# Ridge Regression (Python) — Validation Report

- **Target**: `ridge_regression_analysis.py` (endpoint `POST /api/analysis/ridge-regression`)
- **Backend**: `scottierieh/backend` (Python / FastAPI `statistica-api`)
- **Method**: Ran the CLI script (JSON stdin→stdout) and compared to a direct
  `sklearn.linear_model.Ridge` fit on the same data / split / scaler.

## Summary
| Area | Result |
|------|--------|
| Intercept | ✅ |
| Coefficients (x1,x2,x3) | ✅ (3/3) |
| Test R² / RMSE | ✅ |
| Train R² | ✅ |

**7/7 pass.** The script already uses `sklearn.Ridge(alpha, random_state=42)` with
`StandardScaler` + `train_test_split(random_state=42)`, so it is package-based and
deterministic; outputs match an independent sklearn reproduction exactly.

## Details
| Field | Basis | Result |
|-------|-------|--------|
| `intercept` | `Ridge.intercept_` | ✅ |
| `coefficients[f]` | `Ridge.coef_` | ✅ |
| `metrics.test.r2_score` | `r2_score(y_test, pred)` | ✅ |
| `metrics.test.rmse` | `sqrt(mean_squared_error)` | ✅ |
| `metrics.train.r2_score` | `r2_score(y_train, pred)` | ✅ |

## Conclusion
The Python Ridge endpoint is **reliable and package-based** (scikit-learn); coefficients,
intercept, and train/test metrics match a direct sklearn fit across 7 checks.

Repro: `validation/validate_ridge_regression.py`.
