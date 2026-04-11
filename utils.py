"""
utils.py — numpy-vectorised shared helpers.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

Returns = List[float]
Prices  = List[float]


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _arr(xs) -> np.ndarray:
    return np.asarray(xs, dtype=np.float64)


def mean(xs: Returns) -> float:
    a = _arr(xs)
    return float(np.mean(a)) if len(a) else 0.0


def std(xs: Returns, ddof: int = 1) -> float:
    a = _arr(xs)
    return float(np.std(a, ddof=ddof)) if len(a) > ddof else 0.0


def calc_returns(prices: Prices) -> Returns:
    p = _arr(prices)
    if len(p) < 2:
        return []
    r = np.diff(p) / p[:-1]
    return np.where(np.isfinite(r), r, 0.0).tolist()


def calc_stats(returns: Returns) -> Dict[str, float]:
    if not returns:
        return {"mean": 0, "std": 0, "sharpe": 0, "maxDrawdown": 0, "totalReturn": 0}
    r    = _arr(returns)
    m    = float(np.mean(r))
    s    = float(np.std(r, ddof=1)) or 1e-8
    cum  = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd   = (peak - cum) / np.where(peak > 0, peak, 1.0)
    return {
        "mean":        safe_float(m),
        "std":         safe_float(s),
        "sharpe":      safe_float(m / s),
        "maxDrawdown": safe_float(float(np.max(dd))),
        "totalReturn": safe_float(float(cum[-1] - 1)),
    }


def drawdown_series(returns: Returns) -> List[float]:
    r = _arr(returns)
    if len(r) == 0:
        return []
    cum  = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / np.where(peak > 0, peak, 1.0)
    return dd.tolist()


def cov_matrix(return_series: List[Returns]) -> List[List[float]]:
    mat = np.column_stack([_arr(r) for r in return_series])
    L   = min(len(r) for r in return_series)
    cov = np.cov(mat[-L:].T, ddof=1)
    return np.atleast_2d(cov).tolist()


def port_vol(weights: List[float], cov: List[List[float]]) -> float:
    w   = _arr(weights)
    C   = np.asarray(cov)
    var = float(w @ C @ w)
    return math.sqrt(max(0.0, var))


def port_ret(weights: List[float], means: List[float]) -> float:
    return float(np.dot(_arr(weights), _arr(means)))


def correlation(a: Returns, b: Returns) -> float:
    L = min(len(a), len(b))
    if L < 2:
        return 0.0
    return safe_float(float(np.corrcoef(_arr(a[-L:]), _arr(b[-L:]))[0, 1]))


def annualise_ret(period_ret: float, freq: int) -> float:
    return (1 + period_ret) ** freq - 1


def annualise_vol(period_vol: float, freq: int) -> float:
    return period_vol * math.sqrt(freq)


def norm_cdf(x: float) -> float:
    return float(scipy_stats.norm.cdf(x))


def norm_pdf(x: float) -> float:
    return float(scipy_stats.norm.pdf(x))


def project_simplex(v: List[float]) -> List[float]:
    u  = np.sort(_arr(v))[::-1]
    cs = np.cumsum(u)
    rho_arr = np.nonzero(u * np.arange(1, len(u)+1) > (cs - 1))[0]
    rho = int(rho_arr[-1]) if len(rho_arr) else 0
    theta = (cs[rho] - 1.0) / (rho + 1.0)
    return np.maximum(_arr(v) - theta, 0.0).tolist()


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types → JSON-safe Python types."""
    # numpy scalar types first
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    # Python native
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj
