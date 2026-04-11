from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import math
import traceback

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class WeibullRequest(BaseModel):
    data: List[Dict[str, Any]]
    colTime: Optional[str] = None
    colCensored: Optional[str] = None          # 0=failed, 1=censored
    confidenceLevel: float = 0.95
    generate: bool = False


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
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
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Weibull MLE (Newton-Raphson)
# ══════════════════════════════════════════════════════════════

def weibull_mle(fail_times: np.ndarray, censor_times: np.ndarray = None, max_iter=200, tol=1e-10):
    """
    Maximum Likelihood Estimation for 2-parameter Weibull.
    fail_times: array of failure times
    censor_times: array of right-censored times (optional)
    Returns: (beta, eta)
    """
    n_fail = len(fail_times)
    if n_fail < 2:
        raise ValueError("Need at least 2 failure times for MLE.")

    all_times = np.concatenate([fail_times, censor_times]) if censor_times is not None and len(censor_times) > 0 else fail_times.copy()
    n_total = len(all_times)
    is_failed = np.concatenate([np.ones(n_fail), np.zeros(len(censor_times) if censor_times is not None else 0)])

    # Initial estimate via median rank regression
    sorted_fail = np.sort(fail_times)
    median_ranks = np.array([(i - 0.3) / (n_fail + 0.4) for i in range(1, n_fail + 1)])
    y = np.log(-np.log(1 - median_ranks))
    x = np.log(sorted_fail)
    # Simple OLS
    x_mean, y_mean = np.mean(x), np.mean(y)
    beta_init = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    beta_init = max(0.1, min(beta_init, 20.0))

    beta = beta_init

    for iteration in range(max_iter):
        t_beta = all_times ** beta
        ln_t = np.log(np.maximum(all_times, 1e-30))

        sum_t_beta = np.sum(t_beta)
        sum_t_beta_ln = np.sum(t_beta * ln_t)
        sum_t_beta_ln2 = np.sum(t_beta * ln_t ** 2)
        sum_ln_fail = np.sum(np.log(np.maximum(fail_times, 1e-30)))

        # Profile log-likelihood derivative w.r.t. beta
        if sum_t_beta < 1e-30:
            break

        g = n_fail / beta + sum_ln_fail - n_fail * sum_t_beta_ln / sum_t_beta
        # Second derivative
        g_prime = -n_fail / (beta ** 2) - n_fail * (
            sum_t_beta_ln2 * sum_t_beta - sum_t_beta_ln ** 2
        ) / (sum_t_beta ** 2)

        if abs(g_prime) < 1e-30:
            break

        delta = g / g_prime
        beta_new = beta - delta

        if beta_new <= 0:
            beta_new = beta / 2

        beta = max(0.01, min(beta_new, 50.0))

        if abs(delta) < tol:
            break

    # Eta from MLE
    t_beta = all_times ** beta
    eta = (np.sum(t_beta) / n_fail) ** (1.0 / beta)

    return float(beta), float(eta)


# ══════════════════════════════════════════════════════════════
# Weibull Functions
# ══════════════════════════════════════════════════════════════

def weibull_reliability(t, beta, eta):
    return np.exp(-(t / eta) ** beta)

def weibull_cdf(t, beta, eta):
    return 1.0 - np.exp(-(t / eta) ** beta)

def weibull_pdf(t, beta, eta):
    return (beta / eta) * (t / eta) ** (beta - 1) * np.exp(-(t / eta) ** beta)

def weibull_hazard(t, beta, eta):
    return (beta / eta) * (t / eta) ** (beta - 1)

def weibull_b_life(p, beta, eta):
    """Time at which fraction p has failed. B10 = weibull_b_life(0.10, beta, eta)"""
    return eta * (-np.log(1 - p)) ** (1.0 / beta)

def weibull_mean_life(beta, eta):
    """MTTF = eta * Gamma(1 + 1/beta)"""
    return eta * math.gamma(1 + 1.0 / beta)


# ══════════════════════════════════════════════════════════════
# Goodness-of-Fit: Anderson-Darling for Weibull
# ══════════════════════════════════════════════════════════════

def anderson_darling_weibull(fail_times, beta, eta):
    """Compute Anderson-Darling statistic for Weibull fit (uncensored data only)."""
    n = len(fail_times)
    if n < 3:
        return None

    sorted_t = np.sort(fail_times)
    F_i = weibull_cdf(sorted_t, beta, eta)
    F_i = np.clip(F_i, 1e-15, 1 - 1e-15)

    S = 0.0
    for i in range(n):
        S += (2 * (i + 1) - 1) * (np.log(F_i[i]) + np.log(1 - F_i[n - 1 - i]))

    AD = -n - S / n

    # Modified AD for Weibull (Lawless, 2003)
    AD_star = AD * (1 + 0.2 / np.sqrt(n))

    # Approximate p-value (Lawless)
    if AD_star <= 0:
        p_value = 1.0
    elif AD_star < 0.474:
        p_value = 1 - np.exp(-12.2204 + 67.459 * AD_star - 110.3 * AD_star ** 2)
        p_value = max(0, min(1, p_value))
    elif AD_star < 6.0:
        p_value = np.exp(0.9209 - 3.353 * AD_star + 0.8 * AD_star ** 2 - 0.082 * AD_star ** 3)
        p_value = max(0, min(1, p_value))
    else:
        p_value = 0.0

    return {
        "AD": safe_float(AD),
        "AD_star": safe_float(AD_star),
        "p_value": safe_float(p_value),
        "fit_adequate": p_value > 0.05,
    }


# ══════════════════════════════════════════════════════════════
# Fisher Information → Confidence Intervals
# ══════════════════════════════════════════════════════════════

def weibull_ci(fail_times, censor_times, beta, eta, conf=0.95):
    """Approximate CIs via observed Fisher information (delta method)."""
    import scipy.stats as st

    n_fail = len(fail_times)
    all_times = np.concatenate([fail_times, censor_times]) if censor_times is not None and len(censor_times) > 0 else fail_times.copy()

    z = st.norm.ppf(1 - (1 - conf) / 2)

    # Log-parameterization for better CI coverage
    ln_eta = np.log(eta)

    # Observed Fisher information (numerical Hessian)
    eps_b = beta * 1e-5
    eps_e = 0.01

    def neg_loglik(b, e):
        if b <= 0 or e <= 0:
            return 1e30
        ll = 0
        for t in fail_times:
            ll += np.log(b / e) + (b - 1) * np.log(t / e) - (t / e) ** b
        if censor_times is not None:
            for t in censor_times:
                ll += -(t / e) ** b
        return -ll

    nll0 = neg_loglik(beta, eta)

    # Hessian via central differences
    nll_bp = neg_loglik(beta + eps_b, eta)
    nll_bm = neg_loglik(beta - eps_b, eta)
    d2_bb = (nll_bp - 2 * nll0 + nll_bm) / (eps_b ** 2)

    nll_ep = neg_loglik(beta, eta + eps_e)
    nll_em = neg_loglik(beta, eta - eps_e)
    d2_ee = (nll_ep - 2 * nll0 + nll_em) / (eps_e ** 2)

    nll_bpep = neg_loglik(beta + eps_b, eta + eps_e)
    nll_bmem = neg_loglik(beta - eps_b, eta - eps_e)
    nll_bpem = neg_loglik(beta + eps_b, eta - eps_e)
    nll_bmep = neg_loglik(beta - eps_b, eta + eps_e)
    d2_be = (nll_bpep - nll_bpem - nll_bmep + nll_bmem) / (4 * eps_b * eps_e)

    det = d2_bb * d2_ee - d2_be ** 2
    if det > 0:
        var_beta = d2_ee / det
        var_eta = d2_bb / det
    else:
        var_beta = 1.0 / max(d2_bb, 1e-30)
        var_eta = 1.0 / max(d2_ee, 1e-30)

    se_beta = np.sqrt(max(var_beta, 0))
    se_eta = np.sqrt(max(var_eta, 0))

    # Log-normal intervals for positivity
    beta_lower = beta * np.exp(-z * se_beta / beta) if beta > 0 else 0
    beta_upper = beta * np.exp(z * se_beta / beta) if beta > 0 else 0
    eta_lower = eta * np.exp(-z * se_eta / eta) if eta > 0 else 0
    eta_upper = eta * np.exp(z * se_eta / eta) if eta > 0 else 0

    return {
        "beta": {"estimate": safe_float(beta), "se": safe_float(se_beta),
                 "lower": safe_float(beta_lower), "upper": safe_float(beta_upper)},
        "eta": {"estimate": safe_float(eta), "se": safe_float(se_eta),
                "lower": safe_float(eta_lower), "upper": safe_float(eta_upper)},
    }


# ══════════════════════════════════════════════════════════════
# Column Auto-Detection
# ══════════════════════════════════════════════════════════════

def _detect_column(headers, keywords):
    lower_headers = [h.lower() for h in headers]
    for kw in keywords:
        for i, lh in enumerate(lower_headers):
            if lh == kw:
                return headers[i]
    for kw in keywords:
        if len(kw) <= 3:
            continue
        for i, lh in enumerate(lower_headers):
            if kw in lh:
                return headers[i]
    return None


# ══════════════════════════════════════════════════════════════
# Example Data
# ══════════════════════════════════════════════════════════════

def generate_example_data():
    rng = np.random.default_rng(42)
    rows = []
    beta_true, eta_true = 2.2, 500.0
    n_units = 80
    obs_window = 400.0

    for i in range(n_units):
        u = max(0.001, min(0.999, rng.random()))
        t_fail = eta_true * (-np.log(u)) ** (1.0 / beta_true)
        if t_fail > obs_window:
            rows.append({"unit_id": f"U{i+1:04d}", "hours": round(obs_window, 1), "censored": 1,
                         "failure_mode": "Running"})
        else:
            mode = rng.choice(["Bearing", "Seal", "Motor", "Electrical"])
            rows.append({"unit_id": f"U{i+1:04d}", "hours": round(t_fail, 1), "censored": 0,
                         "failure_mode": mode})
    return rows


# ══════════════════════════════════════════════════════════════
# Main Computation
# ══════════════════════════════════════════════════════════════

def compute_weibull(data, time_col, censor_col, confidence_level=0.95):
    # Parse
    fail_times = []
    censor_times = []

    for row in data:
        t_val = row.get(time_col)
        try:
            t = float(t_val)
        except (TypeError, ValueError):
            continue
        if t <= 0:
            continue

        is_censored = False
        if censor_col and censor_col in row:
            c_val = row[censor_col]
            try:
                c = float(c_val)
                is_censored = c == 1
            except (TypeError, ValueError):
                c_str = str(c_val).strip().lower()
                is_censored = c_str in ('1', 'true', 'yes', 'censored', 'suspended')

        if is_censored:
            censor_times.append(t)
        else:
            fail_times.append(t)

    fail_arr = np.array(fail_times, dtype=float)
    censor_arr = np.array(censor_times, dtype=float) if censor_times else np.array([], dtype=float)

    n_total = len(fail_arr) + len(censor_arr)
    n_failed = len(fail_arr)
    n_censored = len(censor_arr)

    if n_failed < 2:
        raise HTTPException(status_code=400, detail=f"Need at least 2 failures. Got {n_failed}.")

    # MLE
    beta, eta = weibull_mle(fail_arr, censor_arr if len(censor_arr) > 0 else None)
    mttf = weibull_mean_life(beta, eta)

    # Failure mode
    if beta < 0.95:
        failure_mode = "infant_mortality"
        failure_desc = f"β = {beta:.3f} < 1: Decreasing failure rate. Early-life failures dominate — investigate manufacturing defects, assembly errors, or material quality issues."
    elif beta <= 1.05:
        failure_mode = "random"
        failure_desc = f"β = {beta:.3f} ≈ 1: Approximately constant failure rate. Failures are random — typical of electronic components or externally-caused failures."
    else:
        failure_mode = "wearout"
        failure_desc = f"β = {beta:.3f} > 1: Increasing failure rate. Wear-out mechanism — consider preventive maintenance at or before B10 life."

    # B-Lives
    b_lives = {
        "B1": safe_float(weibull_b_life(0.01, beta, eta)),
        "B5": safe_float(weibull_b_life(0.05, beta, eta)),
        "B10": safe_float(weibull_b_life(0.10, beta, eta)),
        "B20": safe_float(weibull_b_life(0.20, beta, eta)),
        "B50": safe_float(weibull_b_life(0.50, beta, eta)),
    }

    # Curves
    all_times = np.concatenate([fail_arr, censor_arr])
    t_max = np.max(all_times) * 1.5
    t_points = np.linspace(0.01, t_max, 200)

    reliability_curve = []
    pdf_curve = []
    hazard_curve = []
    for t in t_points:
        r = weibull_reliability(t, beta, eta)
        f = weibull_pdf(t, beta, eta)
        h = weibull_hazard(t, beta, eta)
        reliability_curve.append({"time": safe_float(t), "reliability": safe_float(r * 100), "failure_pct": safe_float((1 - r) * 100), "hazard": safe_float(h)})
        pdf_curve.append({"time": safe_float(t), "pdf": safe_float(f)})
        hazard_curve.append({"time": safe_float(t), "hazard": safe_float(h)})

    # Probability Plot (Median Rank)
    sorted_fail = np.sort(fail_arr)
    n_f = len(sorted_fail)
    prob_plot = []
    for i in range(n_f):
        median_rank = (i + 1 - 0.3) / (n_f + 0.4)  # Bernard's approximation
        ln_t = np.log(sorted_fail[i])
        lnln_rank = np.log(-np.log(1 - median_rank))
        fitted_lnln = beta * (ln_t - np.log(eta))
        prob_plot.append({
            "time": safe_float(sorted_fail[i]),
            "lnTime": safe_float(ln_t),
            "medianRank": safe_float(median_rank),
            "lnlnRank": safe_float(lnln_rank),
            "fitted_lnlnRank": safe_float(fitted_lnln),
        })

    # Goodness-of-fit
    gof = anderson_darling_weibull(fail_arr, beta, eta)

    # Confidence intervals
    ci = None
    try:
        ci = weibull_ci(fail_arr, censor_arr if len(censor_arr) > 0 else None, beta, eta, confidence_level)
    except Exception:
        pass

    # Percentile table (reliability at key times)
    key_times = [t_max * f for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    percentile_table = []
    for t in key_times:
        r = weibull_reliability(t, beta, eta)
        percentile_table.append({
            "time": safe_float(t),
            "reliability": safe_float(r * 100),
            "failure_pct": safe_float((1 - r) * 100),
            "hazard": safe_float(weibull_hazard(t, beta, eta)),
        })

    return {
        "summary": {
            "nTotal": n_total,
            "nFailed": n_failed,
            "nCensored": n_censored,
            "censoredPct": safe_float(n_censored / n_total * 100) if n_total > 0 else 0,
            "beta": safe_float(beta),
            "eta": safe_float(eta),
            "mttf": safe_float(mttf),
            "failureMode": failure_mode,
            "failureDesc": failure_desc,
            "minTime": safe_float(np.min(all_times)),
            "maxTime": safe_float(np.max(all_times)),
            "meanTime": safe_float(np.mean(all_times)),
        },
        "bLives": b_lives,
        "confidenceIntervals": ci,
        "goodnessOfFit": gof,
        "reliabilityCurve": reliability_curve,
        "pdfCurve": pdf_curve,
        "hazardCurve": hazard_curve,
        "probabilityPlot": prob_plot,
        "percentileTable": percentile_table,
    }


# ══════════════════════════════════════════════════════════════
# Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/weibull-analysis")
async def weibull_analysis(request: WeibullRequest):
    try:
        if request.generate or not request.data:
            data = generate_example_data()
            time_col = "hours"
            censor_col = "censored"
        else:
            data = request.data
            if not data:
                raise HTTPException(status_code=400, detail="No data provided.")

            headers = list(data[0].keys())
            time_col = request.colTime or _detect_column(headers, [
                "time", "hours", "cycles", "months", "mileage", "age", "life", "ttf", "duration",
                "time_hours", "time_months", "operating_hours",
            ])
            censor_col = request.colCensored or _detect_column(headers, [
                "censored", "censor", "status", "event", "suspended",
            ])

            if not time_col:
                raise HTTPException(status_code=400, detail="Cannot find time column.")

        result = compute_weibull(data, time_col, censor_col, request.confidenceLevel)
        return _to_native({"results": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
