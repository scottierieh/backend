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
python3 validate_ridge_regression.py     # -> "RIDGE REGRESSION (Python): 13 PASS, 0 FAIL"
python3 validate_lasso_regression.py     # -> "LASSO REGRESSION (Python): 13 PASS, 0 FAIL"
python3 validate_elastic_net_regression.py  # -> "ELASTIC NET REGRESSION (Python): 15 PASS, 0 FAIL"
```

## Files
| Script | Backend script | Reference | Result |
|--------|----------------|-----------|--------|
| `_pyharness.py` | — | — | shared `run_script`/`chk`/`report` |
| `validate_ridge_regression.py` | `ridge_regression_analysis.py` | `sklearn.linear_model.Ridge` | 13/13 |
| `validate_lasso_regression.py` | `lasso_regression_analysis.py` | `sklearn.linear_model.Lasso` | 13/13 |
| `validate_elastic_net_regression.py` | `elastic_net_regression_analysis.py` | `sklearn.linear_model.ElasticNet` | 15/15 |
| `validate_svm.py` | `svm_analysis.py` | `sklearn.svm.SVC` | 30/30 |
| `validate_knn.py` | `knn_analysis.py` | `sklearn.neighbors.KNeighborsClassifier` | 30/30 |
| `validate_naive_bayes.py` | `naive_bayes_analysis.py` | `sklearn.naive_bayes.GaussianNB` | 30/30 |
| `validate_discriminant.py` | `discriminant_analysis.py` | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | 44/44 |

Reports: `validation/reports/*.md`.
| `validate_decision_tree.py` | `decision_tree_analysis.py` | pass |
| `validate_random_forest.py` | `random_forest_analysis.py` | pass |
| `validate_gbm.py` | `gbm_analysis.py` | pass |
| `validate_adaboost.py` | `adaboost_analysis.py` | pass |
| `validate_xgboost.py` | `xgboost_analysis.py` | pass |
| `validate_lightgbm.py` | `lightgbm_analysis.py` | pass |
| `validate_catboost.py` | `catboost_analysis.py` | pass |
| `validate_ensemble_stacking.py` | `ensemble_stacking_analysis.py` | pass |
| `validate_mlp.py` | `mlp_analysis.py` | pass |
| `validate_cross_validation.py` | `cross_validation_analysis.py` | pass |
| `validate_hyperparameter_tuning.py` | `hyperparameter_tuning_analysis.py` | pass |
| `validate_relative_importance.py` | `relative_importance_analysis.py` | pass |
| `validate_kmeans.py` | `kmeans_analysis.py` | ARI=1 |
| `validate_dbscan.py` | `dbscan_analysis.py` | ARI=1 |
| `validate_gmm.py` | `gmm_analysis.py` | ARI=1 |
| `validate_hca.py` | `hca_analysis.py` | ARI=1 |
| `validate_hdbscan.py` | `hdbscan_analysis.py` | ARI=1 |
| `validate_kmedoids.py` | `kmedoids_analysis.py` | pass (pinned env, numpy<2) |
| `validate_lstm_forecasting.py` | `lstm_forecasting_analysis.py` | pass (pinned env, TensorFlow) |

## Refreshing the public verification page
The page `/crosschecking` (statistica-frontend) shows these runs. Its Python
record is `public/docs-pdf/<id>.pdf`, built from `_results/*.json`,
`_results/*.refs.json` and `reports/*.md`, so re-running a script here is not
enough on its own. From the frontend checkout, one command re-runs everything
and rebuilds the page data:

```bash
cd statistica-frontend
BACKEND_DIR=../backend PYTHON=/path/to/venv/bin/python npm run verification:refresh
```

- `PYTHON` must be an interpreter pinned to `requirements.txt` (Python 3.11, as
  in the Dockerfile; `python3.11 -m venv v && v/bin/pip install -r requirements.txt`).
  Without it only the PDFs are rebuilt from the records already in `_results`.
- It runs every R validation (`r-backend/validation/validate_*.R`) and every
  `validate_*.py` here, regenerates `_results/*.refs.json`
  (`scripts/extract-refs.mjs`), the Python PDFs and `public/verification`, and
  prints the page totals. `SKIP_R=1` skips the R part, `ONLY='kmeans|umap'`
  runs a subset; see the header of `scripts/refresh-verification.mjs`.
- A script whose package is missing (0-check "SKIPPED" record, or "No module
  named") keeps its previous record and is listed as skipped.
- `reports/*.md` are hand-written prose: when a script's checks change, update
  the counts stated there (the measured numbers are appended automatically).
