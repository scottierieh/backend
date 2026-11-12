
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import subprocess
import sys
import os
import json
import time
from collections import defaultdict

# The conjoint/survey analysis family (RC, CBC, ACA, CBC-HB, ACBC) was
# reinstated here as proper FastAPI routers (see rc_analysis.py,
# conjoint_analysis.py, aca_analysis.py, conjoint_hb.py, acbc_analysis.py) —
# the survey module calls this same service, not a separate one. CBC-HB and
# ACBC share their individual-level estimation core (hb_mnl.py — a pure
# numpy/scipy Gibbs/Metropolis sampler, not PyMC, so no new compiler-needing
# dependency was added).

app = FastAPI()

import logging
import signal

from auth import AuthMiddleware

_skari_log = logging.getLogger("skari")
if not _skari_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _skari_log.addHandler(_h)
    _skari_log.setLevel(logging.INFO)
    _skari_log.propagate = False

# Origins are unchanged ("*"): authentication is by bearer token / service-key
# header, never by cookie, so allow_credentials stays off — with origin "*"
# that combination would make Starlette reflect any Origin back verbatim.
origins = [
    "http://localhost:9002",
    "http://127.0.0.1:9002",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]

# Request body cap. Training uploads (Model Lab /api/models/{id}/train) are the
# largest legitimate bodies; Cloud Run itself refuses HTTP/1 bodies over 32 MiB.
_MAX_BODY_BYTES = int(float(os.environ.get("SKARI_MAX_BODY_MB", "25")) * 1024 * 1024)

# Per-caller rate limit: fixed 60s window, 60 requests/window per key (see
# _rl_key). Covers the Model Lab "Auto Compare" bursts while capping a scripted
# flood. Service-key callers (the Next.js server, which fans in many users)
# share ONE bucket with a larger cap. Caveat: per-instance, not shared across
# Cloud Run instances — bound the total with --max-instances.
_RL_WINDOW_SECS = 60
_RL_MAX_REQUESTS = 60
_RL_SERVICE_MAX_REQUESTS = int(os.environ.get("SKARI_RL_SERVICE_MAX", "600"))
_rl_store = defaultdict(list)

def _rl_client_ip(request: Request) -> str:
    # Cloud Run's front end APPENDS the real client address to
    # X-Forwarded-For, so only the LAST entry is trustworthy; earlier entries
    # are whatever the caller sent and would let one client rotate buckets.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        last = fwd.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"

def _rl_key(request: Request):
    state = request.scope.get("state") or {}
    kind = state.get("auth_kind")
    if kind == "user" and state.get("uid"):
        return "uid:" + state["uid"], _RL_MAX_REQUESTS
    if kind == "service":
        return "service", _RL_SERVICE_MAX_REQUESTS
    return _rl_client_ip(request), _RL_MAX_REQUESTS

# Middleware order (Starlette: the LAST one added is the OUTERMOST):
#   CORS -> body-size limit -> auth -> rate limit -> routes
# CORS outermost so 401/413/429 responses still carry CORS headers and
# preflights are answered before auth; auth before the limiter so the limiter
# can key on the authenticated uid.
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/health", "/") or request.method == "OPTIONS":
        return await call_next(request)
    key, cap = _rl_key(request)
    now = time.time()
    hits = [t for t in _rl_store[key] if t > now - _RL_WINDOW_SECS]
    if len(hits) >= cap:
        return JSONResponse(status_code=429, content={"error": "Too many requests, slow down."})
    hits.append(now)
    _rl_store[key] = hits
    return await call_next(request)

app.add_middleware(AuthMiddleware)


class BodySizeLimitMiddleware:
    """413 for bodies over _MAX_BODY_BYTES — by Content-Length up front, and by
    counting streamed chunks for chunked uploads that declare no length."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for k, v in scope.get("headers") or []:
            if k == b"content-length":
                try:
                    too_big = int(v) > self.max_bytes
                except ValueError:
                    too_big = True
                if too_big:
                    return await self._reject(send)
        received = 0
        started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send)

    async def _reject(self, send):
        body = json.dumps({"detail": "Request body too large."}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(HTTPException):
    # An HTTPException so FastAPI's body parsing re-raises it as-is (it turns
    # any other exception into a generic 400) and ExceptionMiddleware answers 413.
    def __init__(self):
        super().__init__(status_code=413, detail="Request body too large.")


app.add_middleware(BodySizeLimitMiddleware, max_bytes=_MAX_BODY_BYTES)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # No allow_credentials: auth is a bearer token / service-key header, not a
    # cookie, so credentialed CORS is never needed — and "*" + credentials
    # would make Starlette reflect the caller's Origin verbatim.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "X-Skari-Service-Key"],
)

@app.get("/")
def read_root():
    return {"message": "Statistica Backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


# Conjoint / survey-analysis family — real APIRouter modules (not the
# subprocess-script pattern below), each already prefixed with its own path
# in-module (e.g. rc_analysis.py's router defines POST /rating-conjoint), so
# they're mounted here under the same /api/analysis prefix the script routes use.
from rc_analysis import router as rc_router
from conjoint_analysis import router as cbc_router
from aca_analysis import router as aca_router
from conjoint_hb import router as cbc_hb_router
from acbc_analysis import router as acbc_router

app.include_router(rc_router, prefix="/api/analysis")
app.include_router(cbc_router, prefix="/api/analysis")
app.include_router(aca_router, prefix="/api/analysis")
app.include_router(cbc_hb_router, prefix="/api/analysis")
app.include_router(acbc_router, prefix="/api/analysis")

# Model Lab's model registry — train/predict/explain/delete by model_id. Also
# a real APIRouter (stateful: persists a fitted pipeline and reloads it by
# id), mounted at /api directly since the frontend calls /api/models/{id}/...,
# not /api/analysis/models/... — see models_api.py's module docstring.
from models_api import router as models_router

app.include_router(models_router, prefix="/api")


# ---------------------------------------------------------------------------
# Generic wiring for the analysis scripts. Every script below follows the
# same self-contained CLI contract (unchanged, not rewritten): read one JSON
# object from stdin, print one JSON object to stdout on success, or print
# {"error": "..."} to stderr and exit(1) on failure. This runner just shells
# out to each script as a subprocess and forwards the request body / response
# body untouched.
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Wall-clock cap per analysis subprocess. 900s covers the longest legitimate
# run (AutoGluon "Thorough" = 600s budget + refit/evaluation); Cloud Run's
# --timeout must be at least this. Override with SKARI_SCRIPT_TIMEOUT_SECS.
_SCRIPT_TIMEOUT_SECS = float(os.environ.get("SKARI_SCRIPT_TIMEOUT_SECS", "900"))

def run_script(script_file: str, payload: dict) -> dict:
    script_path = os.path.join(_BACKEND_DIR, script_file)
    # Own process group, so a timeout also kills the workers a script forks
    # (joblib/loky, n_jobs=-1) rather than orphaning them.
    child = subprocess.Popen(
        [sys.executable, script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = child.communicate(json.dumps(payload), timeout=_SCRIPT_TIMEOUT_SECS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            child.kill()
        child.communicate()
        raise HTTPException(
            status_code=504,
            detail=f"Analysis exceeded the {int(_SCRIPT_TIMEOUT_SECS)}s time limit.",
        )
    proc = subprocess.CompletedProcess(child.args, child.returncode, out, err)
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
    ("/api/analysis/ab-test-analysis",            "ab_test_analysis.py",                     True),
    ("/api/analysis/uplift-modeling",              "uplift_modeling_analysis.py",             True),
    ("/api/analysis/market-basket",                "market_basket_analysis.py",               True),
    ("/api/analysis/attribution-analysis",        "attribution_analysis.py",                 True),
    ("/api/analysis/churn-prediction-v2",         "churn_prediction_analysis.py",           True),
    ("/api/analysis/data-profile",                "data_profile_analysis.py",                True),
    ("/api/analysis/adaboost",                    "adaboost_analysis.py",                    True),
    ("/api/analysis/ahp",                         "ahp_analysis.py",                         False),
    ("/api/analysis/catboost",                    "catboost_analysis.py",                    True),
    ("/api/analysis/capm",                       "capm_analysis.py",                        True),
    ("/api/analysis/portfolio-optimization",     "portfolio_optimization_analysis.py",      True),
    ("/api/analysis/black-scholes",              "black_scholes_analysis.py",               True),
    ("/api/analysis/dcf-valuation",              "dcf_valuation_analysis.py",               True),
    ("/api/analysis/wacc-analysis",              "wacc_analysis.py",                        True),
    ("/api/analysis/greeks-analysis",            "greeks_analysis.py",                      True),
    ("/api/analysis/portfolio-risk",             "portfolio_risk_analysis.py",              True),
    ("/api/analysis/value-at-risk",              "value_at_risk_analysis.py",               True),
    ("/api/analysis/fin-risk-analysis",          "fin_risk_analysis_backend.py",            True),
    ("/api/analysis/fama-french-factor-model",   "fama_french_analysis.py",                 True),
    ("/api/analysis/option-pricing",             "option_pricing_analysis.py",              True),
    ("/api/analysis/fin-monte-carlo",            "monte_carlo_analysis.py",                 True),
    ("/api/analysis/multifactor-models",         "multifactor_analysis.py",                 True),
    ("/api/analysis/event-study",                "event_study_analysis.py",                 True),
    ("/api/analysis/portfolio-performance",      "portfolio_performance_analysis.py",       True),
    ("/api/analysis/volatility-modeling",        "volatility_modeling_analysis.py",         True),
    ("/api/analysis/volatility-analysis",        "volatility_analysis_backend.py",          True),
    ("/api/analysis/comparable-company-analysis","comparable_company_analysis.py",          True),
    ("/api/analysis/enterprise-value-analysis",  "enterprise_value_analysis.py",            True),
    ("/api/analysis/return-time-series",         "return_time_series_analysis.py",          True),
    ("/api/analysis/returns-volatility",         "returns_volatility_analysis.py",          True),
    ("/api/analysis/fin-autocorrelation-analysis","fin_autocorrelation_analysis.py",        True),
    ("/api/analysis/fin-stationarity-analysis",  "fin_stationarity_analysis.py",            True),
    ("/api/analysis/financial-forecasting",      "financial_forecasting_analysis.py",       True),
    ("/api/analysis/abnormal-return-analysis",   "abnormal_return_analysis.py",             True),
    ("/api/analysis/market-reaction-analysis",   "market_reaction_analysis.py",             True),
    ("/api/analysis/fin-asset-allocation",       "fin_asset_allocation_analysis.py",        True),
    ("/api/analysis/classifier-comparison",       "classifier_comparison_analysis.py",       False),
    ("/api/analysis/clv",                          "clv_analysis.py",                         True),
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
    ("/api/analysis/autogluon",                   "autogluon_analysis.py",                   True),
    ("/api/analysis/feature-engineering",         "feature_engineering_analysis.py",         True),
    ("/api/analysis/drift-stats",                 "drift_stats_analysis.py",                 True),
    ("/api/analysis/feature-scoring",             "feature_scoring_analysis.py",             True),
    ("/api/analysis/ivd-accuracy",                "ivd_accuracy_analysis.py",                True),
    ("/api/analysis/ivd-plan",                    "ivd_plan_analysis.py",                    True),
    ("/api/analysis/ipa",                         "ipa_analysis.py",                         False),
    ("/api/analysis/kmeans",                      "kmeans_analysis.py",                      True),
    ("/api/analysis/customer-segmentation",       "customer_segmentation_analysis.py",       True),
    ("/api/analysis/kmedoids",                    "kmedoids_analysis.py",                    True),
    ("/api/analysis/knn",                         "knn_analysis.py",                         True),
    ("/api/analysis/lasso-regression",            "lasso_regression_analysis.py",            True),
    ("/api/analysis/lightgbm",                    "lightgbm_analysis.py",                    True),
    ("/api/analysis/linear-programming",          "linear_programming_analysis.py",          False),
    ("/api/analysis/transportation-problem",     "transportation_problem_analysis.py",      True),
    ("/api/analysis/assignment-problem",                    "assignment_problem_analysis.py",            True),
    ("/api/analysis/shortest-path",                         "shortest_path_analysis.py",                 True),
    ("/api/analysis/maximum-flow",                          "maximum_flow_analysis.py",                  True),
    ("/api/analysis/minimum-cost-flow",                     "minimum_cost_flow_analysis.py",             True),
    ("/api/analysis/inventory-optimization",                "inventory_optimization_analysis.py",        True),
    ("/api/analysis/network-flow-optimization", "network_flow_optimization_analysis.py", True),
    ("/api/analysis/production-planning", "production_planning_analysis.py", True),
    ("/api/analysis/resource-allocation", "resource_allocation_analysis.py", True),
    ("/api/analysis/workforce-scheduling", "workforce_scheduling_analysis.py", True),
    ("/api/analysis/scheduling-optimization", "scheduling_optimization_analysis.py", True),
    ("/api/analysis/facility-location", "facility_location_analysis.py", True),
    ("/api/analysis/nonlinear-programming", "nonlinear_programming_analysis.py", True),
    ("/api/analysis/quadratic-programming", "quadratic_programming_analysis.py", True),
    ("/api/analysis/convex-optimization", "convex_optimization_analysis.py", True),
    ("/api/analysis/constrained-optimization", "constrained_optimization_analysis.py", True),
    ("/api/analysis/multi-objective-optimization", "multi_objective_optimization_analysis.py", True),
    ("/api/analysis/lstm-forecast",               "lstm_forecasting_analysis.py",            True),
    ("/api/analysis/marketing-dashboard",         "marketing_dashboard_analysis.py",         False),
    ("/api/analysis/mmm",                         "mmm_analysis.py",                         True),
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
    ("/api/analysis/price-sensitivity",           "price_sensitivity_analysis.py",           False),
    ("/api/analysis/variability",                 "variability_analysis.py",                 False),
    ("/api/analysis/wordcloud",                   "wordcloud_analysis.py",                   False),
    ("/api/analysis/xgboost",                     "xgboost_analysis.py",                     True),
    ("/api/analysis/income-statement-analysis",   "income_statement_analysis.py",            True),
    ("/api/analysis/balance-sheet-analysis",      "balance_sheet_analysis.py",                True),
    ("/api/analysis/cash-flow-analysis",          "cash_flow_analysis.py",                   True),
    ("/api/analysis/cohort",                      "cohort_analysis.py",                      True),
    ("/api/analysis/funnel",                      "funnel_analysis.py",                      True),
]

for _path, _script, _confirmed in SCRIPT_ROUTES:
    register_script_route(_path, _script)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
