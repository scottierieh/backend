from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class MultiFactorRequest(BaseModel):
    data: List[Dict[str, Any]]

    # Column mappings
    dateCol:   str
    returnCol: str
    mktCol:    str
    smbCol:    Optional[str] = None
    hmlCol:    Optional[str] = None
    momCol:    Optional[str] = None
    rmwCol:    Optional[str] = None
    cmaCol:    Optional[str] = None
    rfCol:     Optional[str] = None

    # Model: "capm" | "ff3" | "carhart" | "ff5"
    modelId: str = "ff3"

    # Rolling window (days)
    rollingWindow: int = 60


# ══════════════════════════════════════════════════════════════
# Factor model definitions
# ══════════════════════════════════════════════════════════════

FACTOR_MODELS: Dict[str, List[str]] = {
    "capm":    ["market"],
    "ff3":     ["market", "smb", "hml"],
    "carhart": ["market", "smb", "hml", "mom"],
    "ff5":     ["market", "smb", "hml", "rmw", "cma"],
}

FACTOR_LABELS: Dict[str, str] = {
    "market": "Mkt-Rf",
    "smb":    "SMB",
    "hml":    "HML",
    "mom":    "MOM",
    "rmw":    "RMW",
    "cma":    "CMA",
}

CHART_COLORS = ['#6C3AED', '#F59E0B', '#10B981', '#EF4444', '#3B82F6', '#EC4899']


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def normal_cdf(z: float) -> float:
    """Abramowitz & Stegun approximation."""
    if z < 0:
        return 1.0 - normal_cdf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z) * poly


# ══════════════════════════════════════════════════════════════
# OLS Regression
# ══════════════════════════════════════════════════════════════

def ols_regression(y: np.ndarray, X_factors: List[np.ndarray]) -> Dict[str, Any]:
    """
    OLS multiple regression: y = alpha + b1*x1 + ... + epsilon
    Normal equations: beta = (X'X)^-1 X'y

    Returns alpha, betas, rSquared, adjR2, residuals,
            tStats, pValues, fStat, se, trackingError, informationRatio
    """
    n = len(y)
    k = len(X_factors)
    p = k + 1

    zero_result = dict(
        alpha=0.0, betas=[0.0] * k, rSquared=0.0, adjR2=0.0,
        residuals=y.tolist(), tStats=[0.0] * p, pValues=[1.0] * p,
        fStat=0.0, se=[0.0] * p, trackingError=0.0, informationRatio=0.0,
    )
    if n < p + 2:
        return zero_result

    ones = np.ones((n, 1))
    X    = np.hstack([ones] + [f.reshape(-1, 1) for f in X_factors])  # (n, p)

    XtX = X.T @ X
    Xty = X.T @ y

    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(p))

    coeffs    = XtX_inv @ Xty
    alpha     = float(coeffs[0])
    betas     = coeffs[1:].tolist()
    fitted    = X @ coeffs
    residuals = y - fitted

    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2     = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)

    sigma2  = ss_res / (n - p)
    se      = [float(np.sqrt(max(0.0, sigma2 * XtX_inv[i, i]))) for i in range(p)]
    t_stats = [float(coeffs[i] / se[i]) if se[i] > 0 else 0.0 for i in range(p)]
    p_vals  = [float(2.0 * (1.0 - normal_cdf(abs(t)))) for t in t_stats]

    ms_reg = (ss_tot - ss_res) / k if k > 0 else 0.0
    mse    = ss_res / (n - p)
    f_stat = float(ms_reg / mse) if mse > 0 else 0.0

    tracking_error    = float(np.std(residuals, ddof=1) * np.sqrt(252))
    ann_alpha         = float(np.power(1.0 + alpha, 252) - 1.0)
    information_ratio = float(ann_alpha / tracking_error) if tracking_error > 0 else 0.0

    return dict(
        alpha=alpha,
        betas=betas,
        rSquared=r2,
        adjR2=adj_r2,
        residuals=residuals.tolist(),
        tStats=t_stats,
        pValues=p_vals,
        fStat=f_stat,
        se=se,
        trackingError=tracking_error,
        informationRatio=information_ratio,
    )


# ══════════════════════════════════════════════════════════════
# Rolling Betas
# ══════════════════════════════════════════════════════════════

def compute_rolling_betas(
    dates: List[str],
    y: np.ndarray,
    X_factors: List[np.ndarray],
    factor_names: List[str],
    window: int,
) -> List[Dict[str, Any]]:
    n = len(y)
    if n < window:
        return []

    result = []
    for i in range(window - 1, n):
        y_sl  = y[i - window + 1 : i + 1]
        xs_sl = [f[i - window + 1 : i + 1] for f in X_factors]
        res   = ols_regression(y_sl, xs_sl)

        ann_alpha = (np.power(1.0 + res["alpha"], 252) - 1.0) * 100.0
        row: Dict[str, Any] = {"date": dates[i], "alpha": safe_float(ann_alpha)}
        for fname, beta in zip(factor_names, res["betas"]):
            row[fname] = safe_float(beta)
        result.append(row)

    return result


# ══════════════════════════════════════════════════════════════
# Return Statistics
# ══════════════════════════════════════════════════════════════

def compute_return_stats(
    y: np.ndarray,
    rf_arr: np.ndarray,
    cum_asset: np.ndarray,
) -> Dict[str, Any]:
    ann_ret = float(np.power(1.0 + np.mean(y), 252) - 1.0)
    ann_vol = float(np.std(y, ddof=1) * np.sqrt(252))

    excess     = y - rf_arr
    ann_exc    = float(np.mean(excess) * 252)
    ann_vol_e  = float(np.std(excess, ddof=1) * np.sqrt(252))
    sharpe     = float(ann_exc / ann_vol_e) if ann_vol_e > 0 else 0.0

    peak   = np.maximum.accumulate(cum_asset)
    dd     = (cum_asset - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    return dict(
        annReturn=safe_float(ann_ret * 100),
        annVol=safe_float(ann_vol * 100),
        sharpe=safe_float(sharpe),
        maxDrawdown=safe_float(max_dd * 100),
    )


# ══════════════════════════════════════════════════════════════
# Factor Contributions
# ══════════════════════════════════════════════════════════════

def compute_factor_contributions(
    ols: Dict[str, Any],
    factor_names: List[str],
    X_factors: List[np.ndarray],
) -> List[Dict[str, Any]]:
    alpha     = ols["alpha"]
    ann_alpha = (np.power(1.0 + alpha, 252) - 1.0) * 100.0

    rows = [dict(name="Alpha", contribution=safe_float(ann_alpha), beta=None, color=CHART_COLORS[0])]
    for i, (fname, beta, fdata) in enumerate(zip(factor_names, ols["betas"], X_factors)):
        contrib = beta * float(np.mean(fdata)) * 252 * 100.0
        rows.append(dict(
            name=FACTOR_LABELS.get(fname, fname.upper()),
            contribution=safe_float(contrib),
            beta=safe_float(beta),
            color=CHART_COLORS[i % len(CHART_COLORS)],
        ))
    return rows


# ══════════════════════════════════════════════════════════════
# Factor Correlation Matrix
# ══════════════════════════════════════════════════════════════

def compute_factor_corr_matrix(
    factor_names: List[str],
    X_factors: List[np.ndarray],
) -> List[Dict[str, Any]]:
    result = []
    for i, a in enumerate(factor_names):
        for j, b in enumerate(factor_names):
            r = 1.0 if i == j else safe_float(float(np.corrcoef(X_factors[i], X_factors[j])[0, 1]))
            result.append({"row": a, "col": b, "r": r})
    return result


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/multi-factor-pricing")
async def multi_factor_pricing(request: MultiFactorRequest):
    try:
        factors = FACTOR_MODELS.get(request.modelId)
        if factors is None:
            raise HTTPException(status_code=400, detail=f"Unknown modelId: {request.modelId}")

        # Column map: factor name → actual CSV column name
        col_map: Dict[str, Optional[str]] = {
            "market": request.mktCol,
            "smb":    request.smbCol,
            "hml":    request.hmlCol,
            "mom":    request.momCol,
            "rmw":    request.rmwCol,
            "cma":    request.cmaCol,
        }

        missing = [f for f in factors if not col_map.get(f)]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing column mappings for factors: {missing}",
            )

        # ── 1. Build & sort DataFrame ───────────────────────────
        df = pd.DataFrame(request.data)
        df["_date_str"] = df[request.dateCol].astype(str)
        df = df.sort_values("_date_str").reset_index(drop=True)

        needed = [request.returnCol] + [col_map[f] for f in factors]
        if request.rfCol:
            needed.append(request.rfCol)
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=needed).reset_index(drop=True)

        n = len(df)
        if n < 30:
            raise HTTPException(status_code=400, detail=f"Need at least 30 observations, got {n}")

        dates    = df["_date_str"].tolist()
        y        = df[request.returnCol].values.astype(np.float64)
        rf_arr   = df[request.rfCol].values.astype(np.float64) if request.rfCol else np.zeros(n)
        X_factors = [df[col_map[f]].values.astype(np.float64) for f in factors]

        # ── 2. Full-period OLS ──────────────────────────────────
        ols = ols_regression(y, X_factors)

        # ── 3. Rolling betas ────────────────────────────────────
        rolling = compute_rolling_betas(
            dates=dates,
            y=y,
            X_factors=X_factors,
            factor_names=factors,
            window=request.rollingWindow,
        )

        # ── 4. Cumulative residual return ───────────────────────
        residuals       = np.array(ols["residuals"])
        cum_resid       = np.cumprod(1.0 + residuals)
        cum_resid_chart = [
            {"date": dates[i], "value": safe_float((cum_resid[i] - 1.0) * 100.0)}
            for i in range(n)
        ]

        # ── 5. Cumulative asset return ──────────────────────────
        cum_asset        = np.cumprod(1.0 + y)
        cum_returns_chart = [
            {"date": dates[i], "value": safe_float(cum_asset[i] * 100.0)}
            for i in range(n)
        ]

        # ── 6. Return statistics ────────────────────────────────
        ret_stats = compute_return_stats(y, rf_arr, cum_asset)

        # ── 7. Factor contribution decomposition ────────────────
        contribs = compute_factor_contributions(ols, factors, X_factors)

        # ── 8. Factor correlation matrix ─────────────────────────
        corr_matrix = compute_factor_corr_matrix(factors, X_factors) if len(factors) > 1 else []

        # ── 9. Assemble response ─────────────────────────────────
        result = {
            "modelId":   request.modelId,
            "factors":   factors,
            "n":         n,
            "dateRange": {"start": dates[0], "end": dates[-1]},
            "regression": {
                "alpha":            safe_float(ols["alpha"]),
                "alphaAnnualized":  safe_float((np.power(1.0 + ols["alpha"], 252) - 1.0) * 100.0),
                "betas":            {f: safe_float(b) for f, b in zip(factors, ols["betas"])},
                "rSquared":         safe_float(ols["rSquared"]),
                "adjR2":            safe_float(ols["adjR2"]),
                "fStat":            safe_float(ols["fStat"]),
                "trackingError":    safe_float(ols["trackingError"] * 100.0),
                "informationRatio": safe_float(ols["informationRatio"]),
                "tStats": {
                    "alpha": safe_float(ols["tStats"][0]),
                    **{f: safe_float(ols["tStats"][i + 1]) for i, f in enumerate(factors)},
                },
                "pValues": {
                    "alpha": safe_float(ols["pValues"][0]),
                    **{f: safe_float(ols["pValues"][i + 1]) for i, f in enumerate(factors)},
                },
                "se": {
                    "alpha": safe_float(ols["se"][0] * np.sqrt(252) * 100.0),
                    **{f: safe_float(ols["se"][i + 1]) for i, f in enumerate(factors)},
                },
            },
            "returnStats":    ret_stats,
            "factorContribs": contribs,
            "corrMatrix":     corr_matrix,
            "charts": {
                "rollingBetas": rolling,
                "cumResidual":  cum_resid_chart,
                "cumReturns":   cum_returns_chart,
            },
        }

        return _to_native({"results": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
