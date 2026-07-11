# Python Backend Validation

Reproducible validation for the Python analysis scripts in this repo
(`scottierieh/backend`, the deployed FastAPI `statistica-api`). Each script is a
CLI contract: one JSON object on stdin → one JSON object on stdout. The
harness runs the real script and compares its output to a direct
scikit-learn / statsmodels reference on the same data.

> This is the **Python** validation track, kept separate from the R backend's
> validation (`statistica-frontend/r-backend/validation`). Python is validated
> with Python packages; R with R packages.

## Run
```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn
cd validation
python3 validate_ridge_regression.py     # -> "RIDGE REGRESSION (Python): 7 PASS, 0 FAIL"
python3 validate_lasso_regression.py     # -> "LASSO REGRESSION (Python): 7 PASS, 0 FAIL"
python3 validate_elastic_net_regression.py  # -> "ELASTIC NET REGRESSION (Python): 7 PASS, 0 FAIL"
```

## Files
| Script | Backend script | Reference | Result |
|--------|----------------|-----------|--------|
| `_pyharness.py` | — | — | shared `run_script`/`chk`/`report` |
| `validate_ridge_regression.py` | `ridge_regression_analysis.py` | `sklearn.linear_model.Ridge` | 7/7 |
| `validate_lasso_regression.py` | `lasso_regression_analysis.py` | `sklearn.linear_model.Lasso` | 7/7 |
| `validate_elastic_net_regression.py` | `elastic_net_regression_analysis.py` | `sklearn.linear_model.ElasticNet` | 19/19 |
| `validate_svm.py` | `svm_analysis.py` | `sklearn.svm.SVC` | 2/2 |
| `validate_knn.py` | `knn_analysis.py` | `sklearn.neighbors.KNeighborsClassifier` | 1/1 |
| `validate_naive_bayes.py` | `naive_bayes_analysis.py` | `sklearn.naive_bayes.GaussianNB` | 1/1 |
| `validate_discriminant.py` | `discriminant_analysis.py` | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | 1/1 |

Reports: `validation/reports/*.md`.
| `validate_decision_tree.py` | `decision_tree_analysis.py` | pass |
| `validate_random_forest.py` | `random_forest_analysis.py` | pass |
| `validate_gbm.py` | `gbm_analysis.py` | pass |
| `validate_adaboost.py` | `adaboost_analysis.py` | pass |
| `validate_xgboost.py` | `xgboost_analysis.py` | pass |
| `validate_lightgbm.py` | `lightgbm_analysis.py` | pass |
| `validate_catboost.py` | `catboost_analysis.py` | pass |
| `validate_ensemble_stacking.py` | `ensemble_stacking_analysis.py` | pass |
