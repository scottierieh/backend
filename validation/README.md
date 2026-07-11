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
