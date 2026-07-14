
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import subprocess
import sys
import os
import json

# This backend serves ONLY the "통계분석" (Statistica) analyses that have no
# R-backend equivalent yet -- everything else (conjoint/survey analyses,
# the now-R-backed descriptive stats route, and the unused effectiveness
# route) was intentionally dropped from here; see r-backend/ for the
# migrated analyses and the conjoint-family Python files under src/backend/
# if the survey module ever needs its own service again.

app = FastAPI()

origins = [
    "http://localhost:9002",
    "http://127.0.0.1:9002",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Statistica Backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Generic wiring for the analysis scripts. Every script below follows the
# same self-contained CLI contract (unchanged, not rewritten): read one JSON
# object from stdin, print one JSON object to stdout on success, or print
# {"error": "..."} to stderr and exit(1) on failure. This runner just shells
# out to each script as a subprocess and forwards the request body / response
# body untouched.
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_file: str, payload: dict) -> dict:
    script_path = os.path.join(_BACKEND_DIR, script_file)
    proc = subprocess.run(
        [sys.executable, script_path],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        try:
            parsed = json.loads(proc.stdout or proc.stderr)
            if isinstance(parsed, dict) and "error" in parsed:
                detail = parsed["error"]
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=400, detail=detail)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Non-JSON output from {script_file}: {proc.stdout[:500]!r}",
        )


def register_script_route(path: str, script_file: str):
    # run_script() calls the blocking subprocess.run() — inside an `async def` route that
    # would block this worker's entire event loop until the subprocess exits, serializing
    # every other concurrent request. run_in_threadpool offloads it to a worker thread so
    # other requests keep being served while this one's subprocess runs.
    async def _route(request: Request):
        payload = await request.json()
        return await run_in_threadpool(run_script, script_file, payload)
    app.add_api_route(path, _route, methods=["POST"], name=script_file)


# (endpoint path, script filename, confirmed?) -- entries marked confirmed=False
# were NOT found verbatim in the current frontend source; the path is a
# best-guess kebab-case derivation of the filename. Check the calling page's
# fetch()/endpoint constant before relying on those and correct the path here
# if it doesn't match.
SCRIPT_ROUTES = [
    ("/api/analysis/adaboost",                    "adaboost_analysis.py",                    True),
    ("/api/analysis/ahp",                         "ahp_analysis.py",                         False),
    ("/api/analysis/catboost",                    "catboost_analysis.py",                    True),
    ("/api/analysis/classifier-comparison",       "classifier_comparison_analysis.py",       False),
    ("/api/analysis/cross-validation",            "cross_validation_analysis.py",            True),
    ("/api/analysis/dbscan",                      "dbscan_analysis.py",                      True),
    ("/api/analysis/dea-efficiency",              "dea_analysis.py",                         True),
    ("/api/analysis/decision-tree",               "decision_tree_analysis.py",               True),
    ("/api/analysis/delphi",                      "delphi_analysis.py",                      False),
    ("/api/analysis/lda",                         "discriminant_analysis.py",                True),
    ("/api/analysis/elasticnet-regression",       "elastic_net_regression_analysis.py",      True),
    ("/api/analysis/ensemble-voting-stacking",    "ensemble_stacking_analysis.py",           True),
    ("/api/analysis/fruit-clustering",            "fruit_clustering_analysis.py",            False),
    ("/api/analysis/gradient-boosting",           "gbm_analysis.py",                         True),
    ("/api/analysis/gradient-descent-simulation", "gradient_descent_simulation.py",          False),
    ("/api/analysis/gmm",                         "gmm_analysis.py",                         True),
    ("/api/analysis/hca",                         "hca_analysis.py",                         True),
    ("/api/analysis/hdbscan",                     "hdbscan_analysis.py",                     True),
    ("/api/analysis/homogeneity-test",            "homogeneity_test.py",                     False),
    ("/api/analysis/hyperparameter-tuning",       "hyperparameter_tuning_analysis.py",       True),
    ("/api/analysis/tune",                        "tune_analysis.py",                        True),
    ("/api/analysis/ipa",                         "ipa_analysis.py",                         False),
    ("/api/analysis/kmeans",                      "kmeans_analysis.py",                      True),
    ("/api/analysis/kmedoids",                    "kmedoids_analysis.py",                    True),
    ("/api/analysis/knn",                         "knn_analysis.py",                         True),
    ("/api/analysis/lasso-regression",            "lasso_regression_analysis.py",            True),
    ("/api/analysis/lightgbm",                    "lightgbm_analysis.py",                    True),
    ("/api/analysis/linear-programming",          "linear_programming_analysis.py",          False),
    ("/api/analysis/lstm-forecast",               "lstm_forecasting_analysis.py",            True),
    ("/api/analysis/marketing-dashboard",         "marketing_dashboard_analysis.py",         False),
    ("/api/analysis/mlp",                         "mlp_analysis.py",                         True),
    ("/api/analysis/naive-bayes",                 "naive_bayes_analysis.py",                 True),
    ("/api/analysis/neural-network",              "neural_network_analysis.py",              False),
    ("/api/analysis/nonlinear-regression",        "nonlinear_regression_analysis.py",        False),
    ("/api/analysis/nonparametric",               "nonparametric_analysis.py",               False),
    ("/api/analysis/nps",                         "nps_analysis.py",                         False),
    ("/api/analysis/pareto",                      "pareto_analysis.py",                      False),
    ("/api/analysis/partial-correlation",         "partial_correlation_analysis.py",         False),
    ("/api/analysis/randomforest",                "random_forest_analysis.py",               True),
    ("/api/analysis/relative-importance",         "relative_importance_analysis.py",         False),
    ("/api/analysis/ridge-regression",            "ridge_regression_analysis.py",             True),
    ("/api/analysis/rfm-segmentation",            "rfm_analysis.py",                         True),
    ("/api/analysis/seasonal-analysis",           "seasonal_decomposition_analysis.py",      True),
    ("/api/analysis/sentiment-analysis",          "sentiment_analyzer.py",                   True),
    ("/api/analysis/som",                         "som_analysis.py",                         True),
    ("/api/analysis/spatial-autoregressive-model","spatial_autoregressive_model_analysis.py",False),
    ("/api/analysis/spatial-error-model",         "spatial_error_model_analysis.py",         False),
    ("/api/analysis/svm",                         "svm_analysis.py",                         True),
    ("/api/analysis/tscss",                       "tscss_analysis.py",                       False),
    ("/api/analysis/tsne",                        "tsne_analysis.py",                        True),
    ("/api/analysis/umap",                        "umap_analysis.py",                        True),
    ("/api/analysis/van-westendorp",              "van_westendorp_analysis.py",              False),
    ("/api/analysis/variability",                 "variability_analysis.py",                 False),
    ("/api/analysis/wordcloud",                   "wordcloud_analysis.py",                   False),
    ("/api/analysis/xgboost",                     "xgboost_analysis.py",                     True),
]

for _path, _script, _confirmed in SCRIPT_ROUTES:
    register_script_route(_path, _script)


# ---------------------------------------------------------------------------
# Predict / What-if — load a model persisted by model_store.save_model_bundle
# (currently only gbm_analysis.py calls it; other scripts still train-and-discard,
# so predict-schema/predict 404 for models other than GBM until they're wired the
# same way) and either report its input contract or run it on new rows.
# ---------------------------------------------------------------------------
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from model_store import load_model_bundle, align_features


class PredictSchemaRequest(BaseModel):
    model_id: str


class PredictRequest(BaseModel):
    model_id: str
    rows: List[Dict[str, Any]]


def _load_bundle_or_404(model_id: str) -> dict:
    try:
        return load_model_bundle(model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.post("/api/analysis/predict-schema")
def predict_schema(req: PredictSchemaRequest):
    bundle = _load_bundle_or_404(req.model_id)
    return {
        "feature_cols": bundle["feature_cols"],
        "num_cols": bundle["num_cols"],
        "cat_cols": bundle["cat_cols"],
        "target_col": bundle["target_col"],
        "task_type": bundle["task_type"],
        "classes": bundle["classes"],
        "analysis_type": bundle["analysis_type"],
    }


@app.post("/api/analysis/predict")
def predict(req: PredictRequest):
    import pandas as pd
    import numpy as np

    bundle = _load_bundle_or_404(req.model_id)
    if not req.rows:
        raise HTTPException(status_code=400, detail="No rows provided")

    try:
        df = pd.DataFrame(req.rows)
        missing = [c for c in bundle["feature_cols"] if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")
        X = align_features(df, bundle["feature_cols"], bundle["dummy_columns"])
        model = bundle["model"]
        preds = model.predict(X)
        predictions = [p.item() if hasattr(p, "item") else p for p in preds]
        response: Dict[str, Any] = {"predictions": predictions, "target_col": bundle["target_col"]}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            response["probabilities"] = [float(np.max(row)) for row in proba]
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
