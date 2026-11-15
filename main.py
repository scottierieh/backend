
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import subprocess
import sys
import os
import json
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict

# The conjoint/survey analysis family (RC, CBC, ACA, CBC-HB, ACBC) was
# reinstated here as proper FastAPI routers (see rc_analysis.py,
# conjoint_analysis.py, aca_analysis.py, conjoint_hb.py, acbc_analysis.py) —
# the survey module calls this same service, not a separate one. CBC-HB and
# ACBC share their individual-level estimation core (hb_mnl.py — a pure
# numpy/scipy Gibbs/Metropolis sampler, not PyMC, so no new compiler-needing
# dependency was added).

# /docs, /redoc and /openapi.json describe all ~130 routes to any signed-in
# account; off unless SKARI_API_DOCS=1 (local development).
_API_DOCS = os.environ.get("SKARI_API_DOCS", "").strip() == "1"
app = FastAPI(
    docs_url="/docs" if _API_DOCS else None,
    redoc_url="/redoc" if _API_DOCS else None,
    openapi_url="/openapi.json" if _API_DOCS else None,
)

import logging
import signal

from auth import AuthMiddleware
import auth as _auth
import usage_ticket

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
# Two more ceilings, the same as r-backend/plumber.R's (2026-11 audit L5): many
# accounts from one address share one per-minute bucket by IP -- counted
# BEFORE authentication, so a flood of rejected (401) requests, each an RS256
# verification, is limited too -- and one account has a daily total. Generous
# for real use (a class behind one NAT, a day of Auto Compare); per instance,
# like the rest. The service key is exempt from the IP bucket (it is one
# trusted caller fanning in many users, with its own bucket above).
_RL_IP_MAX_REQUESTS = int(os.environ.get("SKARI_RL_IP_PER_MIN", "300"))
_RL_DAY_MAX_REQUESTS = int(os.environ.get("SKARI_RL_UID_PER_DAY", "5000"))
_rl_ip_store = defaultdict(list)
_rl_day_store = {}
_rl_calls = 0
# Requests one caller may have running at once (2026-11 audit M3). The rate
# limit counts starts, not work in flight: 60 a minute of 900-second jobs is
# every worker thread on the instance. Model Lab's Auto Compare runs 3 at a
# time; the service key fans in many users.
_INFLIGHT_MAX = int(os.environ.get("SKARI_MAX_INFLIGHT_PER_UID", "6"))
_INFLIGHT_SERVICE_MAX = int(os.environ.get("SKARI_MAX_INFLIGHT_SERVICE", "16"))
_inflight = defaultdict(int)
_inflight_lock = threading.Lock()
_RL_EXEMPT = ("/health", "/")

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


def _rl_prune(now):
    """Drop idle buckets now and then so the stores do not grow without bound."""
    global _rl_calls
    _rl_calls += 1
    if _rl_calls % 1000:
        return
    for store in (_rl_store, _rl_ip_store):
        for k in [k for k, v in store.items() if not v or v[-1] <= now - _RL_WINDOW_SECS]:
            del store[k]
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    for k in [k for k, v in _rl_day_store.items() if v[0] != today]:
        del _rl_day_store[k]


def _too_many(message="Too many requests, slow down."):
    return JSONResponse(status_code=429, content={"error": message, "detail": message})

# Middleware order (Starlette: the LAST one added is the OUTERMOST):
#   CORS -> body-size limit -> per-IP limit -> auth -> rate limit (uid/minute,
#   uid/day, in-flight) -> usage pass -> routes
# CORS outermost so 401/413/429 responses still carry CORS headers and
# preflights are answered before auth; auth before the limiter so the limiter
# can key on the authenticated uid.
# Usage pass for metered runs (AutoML): the app signs one when it counts a run
# against the monthly limit (POST /api/usage/consume); this checks it so the
# count cannot be skipped by calling here directly. Defined before the rate
# limiter so it sits inside it (after auth, which sets the uid). Mode
# SKARI_USAGE_TICKETS = off (default) | monitor | on — usage_ticket.py.
#
# Every middleware reads the path the ROUTER uses, request.scope["path"].
# request.url.path is rebuilt from the Host header, so `Host: x/health?` made
# a metered route look like /health to these checks (2026-11 audit L1).
@app.middleware("http")
async def usage_ticket_middleware(request: Request, call_next):
    state = request.scope.get("state") or {}
    ok, reason = usage_ticket.check_request(
        request.scope.get("path", ""), request.method, state.get("auth_kind"), state.get("uid"),
        request.headers, usage_ticket.signing_secret(), usage_ticket.mode())
    if not ok:
        return JSONResponse(status_code=403, content={
            "detail": usage_ticket.MESSAGE, "error": usage_ticket.MESSAGE,
            "code": f"usage_ticket_{reason}"})
    return await call_next(request)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.scope.get("path") in _RL_EXEMPT or request.method == "OPTIONS":
        return await call_next(request)
    key, cap = _rl_key(request)
    now = time.time()
    _rl_prune(now)
    hits = [t for t in _rl_store[key] if t > now - _RL_WINDOW_SECS]
    if len(hits) >= cap:
        return _too_many()
    state = request.scope.get("state") or {}
    uid = state.get("uid") if state.get("auth_kind") == "user" else None
    if uid:
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        day, n = _rl_day_store.get(uid, (today, 0))
        if day != today:
            n = 0
        if n >= _RL_DAY_MAX_REQUESTS:
            return _too_many("Daily analysis limit reached. Try again tomorrow.")
        _rl_day_store[uid] = (today, n + 1)
    hits.append(now)
    _rl_store[key] = hits

    inflight_cap = _INFLIGHT_SERVICE_MAX if key == "service" else _INFLIGHT_MAX
    with _inflight_lock:
        if _inflight[key] >= inflight_cap:
            busy = True
        else:
            busy = False
            _inflight[key] += 1
    if busy:
        return _too_many("Too many analyses running at once for this account. "
                         "Wait for one to finish and try again.")
    try:
        return await call_next(request)
    finally:
        # Released when the response starts. A handler running in the thread
        # pool after its client went away is not stopped by this; the cap
        # bounds how many one account can START, which is what it is for.
        with _inflight_lock:
            _inflight[key] -= 1
            if _inflight[key] <= 0:
                del _inflight[key]


app.add_middleware(AuthMiddleware)


class IpRateLimitMiddleware:
    """Per-IP ceiling (SKARI_RL_IP_PER_MIN), outside AuthMiddleware so that
    rejected requests count too. Pure ASGI, like AuthMiddleware."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") == "OPTIONS"
                or scope.get("path") in _RL_EXEMPT):
            return await self.app(scope, receive, send)
        request = Request(scope)
        if _auth.check_service_key(request.headers.get(_auth.SERVICE_KEY_HEADER, "")):
            return await self.app(scope, receive, send)
        ip = _rl_client_ip(request)
        now = time.time()
        hits = [t for t in _rl_ip_store[ip] if t > now - _RL_WINDOW_SECS]
        if len(hits) >= _RL_IP_MAX_REQUESTS:
            return await _too_many()(scope, receive, send)
        hits.append(now)
        _rl_ip_store[ip] = hits
        return await self.app(scope, receive, send)


app.add_middleware(IpRateLimitMiddleware)


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
    allow_headers=["Authorization", "Content-Type", "X-Skari-Service-Key", "X-Skari-Usage-Ticket"],
)

@app.get("/")
def read_root():
    return {"message": "Statistica Backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


def _error_ref() -> str:
    return uuid.uuid4().hex[:12]


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    # Tracebacks go to the log under a reference; the caller gets the
    # reference, never the stack (2026-11 audit L2).
    ref = _error_ref()
    _skari_log.error("[%s] unhandled error on %s %s", ref, request.method,
                     request.scope.get("path"), exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": f"Internal error (ref {ref})."})


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

# What an analysis subprocess inherits from this server's environment: runtime
# and library settings only. Never the service key, the model-signing key or
# anything else secret -- the scripts parse user data with a large dependency
# set and need none of it (2026-11 audit L4). Prefix match for the families.
_CHILD_ENV_NAMES = frozenset({
    "PATH", "HOME", "LANG", "LANGUAGE", "TZ", "TMPDIR", "TEMP", "TMP",
    "VIRTUAL_ENV", "LD_LIBRARY_PATH", "MPLCONFIGDIR", "MPLBACKEND",
})
_CHILD_ENV_PREFIXES = ("LC_", "PYTHON", "OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_",
                       "NUMBA_", "KMP_", "TF_", "JOBLIB_", "LOKY_")


def _child_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if k in _CHILD_ENV_NAMES or k.startswith(_CHILD_ENV_PREFIXES)}


def _script_error_message(text: str):
    """The {"error": "..."} a script printed on failure, if any (the last
    JSON line wins: a library may have printed warnings first)."""
    for chunk in [text] + list(reversed(text.strip().splitlines())):
        try:
            parsed = json.loads(chunk)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            msg = parsed["error"]
            if "Traceback (most recent call last)" not in msg:
                return msg[:2000]
    return None


def run_script(script_file: str, payload: dict) -> dict:
    script_path = os.path.join(_BACKEND_DIR, script_file)
    # A fresh working directory per run: a script that writes relative files
    # (fruit_clustering_analysis.py) writes them there, not into /app shared
    # by every request (and read-only for the non-root image user). Imports
    # are unaffected -- sys.path[0] is the script's own directory.
    workdir = tempfile.mkdtemp(prefix="skari_run_")
    try:
        # Own process group, so a timeout also kills the workers a script forks
        # (joblib/loky, n_jobs=-1) rather than orphaning them.
        child = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=_child_env(),
            cwd=workdir,
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
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    proc = subprocess.CompletedProcess(child.args, child.returncode, out, err)
    if proc.returncode != 0:
        # The script's own {"error": ...} is written for the user and goes
        # back as it always has. Anything else -- an uncaught traceback, a
        # native crash -- stays in the log under a reference (audit L2).
        detail = _script_error_message(proc.stderr or "") or _script_error_message(proc.stdout or "")
        if detail is None:
            ref = _error_ref()
            _skari_log.error("[%s] %s exited %s; stderr tail: %s", ref, script_file,
                             proc.returncode, (proc.stderr or proc.stdout or "")[-4000:])
            detail = f"The analysis failed unexpectedly (ref {ref})."
        raise HTTPException(status_code=400, detail=detail)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        ref = _error_ref()
        _skari_log.error("[%s] non-JSON output from %s: %r", ref, script_file, proc.stdout[:2000])
        raise HTTPException(
            status_code=500,
            detail=f"The analysis returned an unreadable result (ref {ref}).",
        )


def register_script_route(path: str, script_file: str):
    # run_script() calls the blocking subprocess.run() — inside an `async def` route that
    # would block this worker's entire event loop until the subprocess exits, serializing
    # every other concurrent request. run_in_threadpool offloads it to a worker thread so
    # other requests keep being served while this one's subprocess runs.
    async def _route(request: Request):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="The request body must be JSON.")
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
