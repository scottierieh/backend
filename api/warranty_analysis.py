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

class WarrantyRequest(BaseModel):
    data: List[Dict[str, Any]]
    colTime: Optional[str] = None
    colCensored: Optional[str] = None          # 0=failed, 1=censored (still running)
    warrantyPeriod: Optional[float] = None     # warranty duration in same time units
    unitCost: Optional[float] = None           # cost per warranty claim
    totalUnits: Optional[int] = None           # total units in field
    generate: bool = False
    confidenceLevel: float = 0.95


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
# Weibull MLE Fitting
# ══════════════════════════════════════════════════════════════

def weibull_mle(times: np.ndarray, censored: np.ndarray, max_iter: int = 200, tol: float = 1e-8):
    """
    Maximum Likelihood Estimation for 2-parameter Weibull distribution.
    times: array of failure/censoring times (must be > 0)
    censored: array of 0/1 (0=failure, 1=right-censored)

    Returns (beta, eta) — shape and scale parameters.
    Uses Newton-Raphson on the profile log-likelihood.
    """
    t = times.astype(float)
    c = censored.astype(float)  # 1 = censored, 0 = failed
    failed = (c == 0)
    n_failed = failed.sum()

    if n_failed < 2:
        raise ValueError("Need at least 2 failure observations for Weibull MLE.")

    # Remove zeros
    t = np.maximum(t, 1e-10)

    ln_t = np.log(t)

    # Initial estimate of beta using median rank regression
    # Simple: start with beta=1.0
    beta = 1.0

    for iteration in range(max_iter):
        t_beta = t ** beta
        ln_t_t_beta = ln_t * t_beta

        # Sum components
        sum_t_beta = np.sum(t_beta)
        sum_ln_t_t_beta = np.sum(ln_t_t_beta)
        sum_ln_t_failed = np.sum(ln_t[failed])
        sum_ln_t_t_beta_sq = np.sum(ln_t * ln_t_t_beta)

        if sum_t_beta == 0:
            break

        # Profile log-likelihood derivative w.r.t. beta
        # d/dbeta: n_failed/beta + sum(ln_t_i * delta_i) - (n_failed * sum(t^beta * ln_t)) / sum(t^beta)
        f_beta = (n_failed / beta) + sum_ln_t_failed - (n_failed * sum_ln_t_t_beta / sum_t_beta)

        # Second derivative
        f_prime = -(n_failed / (beta ** 2)) - n_failed * (
            (sum_ln_t_t_beta_sq * sum_t_beta - sum_ln_t_t_beta ** 2) / (sum_t_beta ** 2)
        )

        if abs(f_prime) < 1e-30:
            break

        # Newton step
        delta = f_beta / f_prime
        beta_new = beta - delta

        # Constrain beta to reasonable range
        beta_new = max(0.01, min(beta_new, 20.0))

        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Estimate eta from beta
    sum_t_beta = np.sum(t ** beta)
    eta = (sum_t_beta / n_failed) ** (1.0 / beta)

    return float(beta), float(eta)


def weibull_reliability(t, beta, eta):
    """R(t) = exp(-(t/eta)^beta)"""
    return np.exp(-((t / eta) ** beta))


def weibull_hazard(t, beta, eta):
    """h(t) = (beta/eta) * (t/eta)^(beta-1)"""
    return (beta / eta) * ((t / eta) ** (beta - 1))


def weibull_pdf(t, beta, eta):
    """f(t) = (beta/eta) * (t/eta)^(beta-1) * exp(-(t/eta)^beta)"""
    return weibull_hazard(t, beta, eta) * weibull_reliability(t, beta, eta)


def weibull_cdf(t, beta, eta):
    """F(t) = 1 - R(t)"""
    return 1.0 - weibull_reliability(t, beta, eta)


def weibull_b_life(p, beta, eta):
    """B-life: time at which p% have failed. B10 → p=0.10"""
    return eta * ((-np.log(1 - p)) ** (1.0 / beta))


def weibull_mean_life(beta, eta):
    """MTTF = eta * Gamma(1 + 1/beta)"""
    return eta * math.gamma(1.0 + 1.0 / beta)


# ══════════════════════════════════════════════════════════════
# Example Data Generation
# ══════════════════════════════════════════════════════════════

def generate_example_data() -> List[Dict[str, Any]]:
    """
    Generate warranty claim data:
    - Weibull distributed failure times (beta=1.8, eta=36 months)
    - Some right-censored (still running, no failure)
    """
    rng = np.random.default_rng(42)

    beta_true = 1.8
    eta_true = 36.0
    n_total = 150

    rows = []
    for i in range(n_total):
        # Weibull random variate: t = eta * (-ln(U))^(1/beta)
        u = rng.uniform(0.001, 0.999)
        t_fail = eta_true * ((-np.log(u)) ** (1.0 / beta_true))

        # Censoring: observation window of 24 months
        obs_window = 24.0
        if t_fail > obs_window:
            # Censored — still running at obs_window
            rows.append({
                "unit_id": f"U{i+1:04d}",
                "time_months": round(obs_window, 1),
                "censored": 1,
                "product": f"Model-{chr(65 + i % 3)}",
            })
        else:
            # Failed
            rows.append({
                "unit_id": f"U{i+1:04d}",
                "time_months": round(t_fail, 1),
                "censored": 0,
                "product": f"Model-{chr(65 + i % 3)}",
            })

    return rows


# ══════════════════════════════════════════════════════════════
# Column auto-detection
# ══════════════════════════════════════════════════════════════

def _detect_column(headers: List[str], keywords: List[str]) -> Optional[str]:
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
# Main Computation
# ══════════════════════════════════════════════════════════════

def compute_warranty(
    times: np.ndarray,
    censored: np.ndarray,
    warranty_period: Optional[float],
    unit_cost: Optional[float],
    total_units: Optional[int],
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Full warranty analysis with Weibull fitting."""

    n = len(times)
    n_failed = int((censored == 0).sum())
    n_censored = int((censored == 1).sum())

    # ── Weibull MLE ──
    beta, eta = weibull_mle(times, censored)
    mttf = weibull_mean_life(beta, eta)

    # ── Failure mode interpretation ──
    if beta < 0.95:
        failure_mode = "infant_mortality"
        failure_desc = "Decreasing failure rate — early-life failures dominate (infant mortality). Consider burn-in testing or incoming quality improvements."
    elif beta <= 1.05:
        failure_mode = "random"
        failure_desc = "Approximately constant failure rate — random failures. Typical of electronic components or externally-caused failures."
    else:
        failure_mode = "wearout"
        failure_desc = "Increasing failure rate — wear-out failures dominate. Consider preventive maintenance or design life extension."

    # ── B-life calculations ──
    b_lives = {}
    for pct in [1, 5, 10, 20, 50]:
        b_lives[f"B{pct}"] = safe_float(weibull_b_life(pct / 100.0, beta, eta))

    # ── Reliability at key times ──
    max_time = float(np.max(times)) * 1.5
    if warranty_period:
        max_time = max(max_time, warranty_period * 2)

    time_points = np.linspace(0.01, max_time, 200)
    reliability_curve = []
    for t in time_points:
        reliability_curve.append({
            "time": safe_float(t),
            "reliability": safe_float(weibull_reliability(t, beta, eta) * 100),
            "failure_pct": safe_float(weibull_cdf(t, beta, eta) * 100),
            "hazard": safe_float(weibull_hazard(t, beta, eta)),
        })

    # ── PDF curve ──
    pdf_curve = []
    for t in time_points:
        pdf_curve.append({
            "time": safe_float(t),
            "pdf": safe_float(weibull_pdf(t, beta, eta)),
        })

    # ── Hazard curve ──
    hazard_curve = []
    for t in time_points:
        hazard_curve.append({
            "time": safe_float(t),
            "hazard": safe_float(weibull_hazard(t, beta, eta)),
        })

    # ── Warranty period analysis ──
    warranty_info = None
    if warranty_period and warranty_period > 0:
        fail_prob = weibull_cdf(warranty_period, beta, eta)
        reliability_at_wp = weibull_reliability(warranty_period, beta, eta)

        expected_failures = None
        expected_cost = None
        if total_units and total_units > 0:
            expected_failures = int(round(fail_prob * total_units))
            if unit_cost and unit_cost > 0:
                expected_cost = safe_float(expected_failures * unit_cost)

        warranty_info = {
            "period": warranty_period,
            "failureProbability": safe_float(fail_prob * 100),
            "reliability": safe_float(reliability_at_wp * 100),
            "expectedFailures": expected_failures,
            "expectedCost": expected_cost,
            "unitCost": unit_cost,
            "totalUnits": total_units,
        }

    # ── Extended warranty scenarios ──
    warranty_scenarios = []
    if warranty_period and total_units:
        for multiplier in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            wp = warranty_period * multiplier
            fp = weibull_cdf(wp, beta, eta)
            ef = int(round(fp * total_units))
            ec = safe_float(ef * (unit_cost or 0)) if unit_cost else None
            warranty_scenarios.append({
                "period": safe_float(wp),
                "failurePct": safe_float(fp * 100),
                "expectedFailures": ef,
                "expectedCost": ec,
            })

    # ── Weibull probability plot data (for visual fit check) ──
    # Median rank method
    failed_times = np.sort(times[censored == 0])
    n_f = len(failed_times)
    prob_plot = []
    for i, t in enumerate(failed_times):
        # Bernard's approximation for median rank
        median_rank = (i + 1 - 0.3) / (n_f + 0.4)
        # Weibull linearization: ln(t) vs ln(-ln(1-F))
        if 0 < median_rank < 1:
            prob_plot.append({
                "time": safe_float(t),
                "lnTime": safe_float(np.log(t)),
                "medianRank": safe_float(median_rank * 100),
                "lnlnRank": safe_float(np.log(-np.log(1 - median_rank))),
            })

    # ── Weibull fitted line for probability plot ──
    if len(prob_plot) > 1:
        ln_eta = np.log(eta)
        for pt in prob_plot:
            # Fitted: ln(-ln(1-F)) = beta * (ln(t) - ln(eta))
            pt["fitted_lnlnRank"] = safe_float(beta * (pt["lnTime"] - ln_eta))

    # ── Summary stats ──
    summary = {
        "nTotal": n,
        "nFailed": n_failed,
        "nCensored": n_censored,
        "censoredPct": safe_float(n_censored / n * 100),
        "beta": safe_float(beta),
        "eta": safe_float(eta),
        "mttf": safe_float(mttf),
        "failureMode": failure_mode,
        "failureDesc": failure_desc,
        "minTime": safe_float(float(np.min(times))),
        "maxTime": safe_float(float(np.max(times))),
        "meanTime": safe_float(float(np.mean(times))),
    }

    return {
        "summary": summary,
        "bLives": b_lives,
        "reliabilityCurve": reliability_curve,
        "pdfCurve": pdf_curve,
        "hazardCurve": hazard_curve,
        "warrantyInfo": warranty_info,
        "warrantyScenarios": warranty_scenarios,
        "probabilityPlot": prob_plot,
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/warranty-analysis")
async def warranty_analysis(request: WarrantyRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            data = generate_example_data()
            col_time = "time_months"
            col_censored = "censored"
        else:
            data = request.data
            if not data or len(data) == 0:
                raise HTTPException(status_code=400, detail="No data provided.")

            headers = list(data[0].keys())

            col_time = request.colTime or _detect_column(
                headers, ["time", "time_months", "months", "hours", "cycles",
                           "mileage", "age", "life", "ttf", "time_to_failure", "duration"]
            )
            col_censored = request.colCensored or _detect_column(
                headers, ["censored", "censor", "status", "event", "failure",
                           "suspended", "right_censored"]
            )

            if not col_time:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot find time column. Provide colTime or use standard naming (time, months, hours, cycles, ttf)."
                )

        # ── 2. Parse data ──
        times_list = []
        censored_list = []

        for row in data:
            try:
                t = float(row.get(col_time, ""))
            except (ValueError, TypeError):
                continue
            if math.isnan(t) or t <= 0:
                continue

            # Censoring: default to 0 (failed) if no censored column
            c = 0
            if col_censored and col_censored in row:
                try:
                    c_val = str(row[col_censored]).strip().lower()
                    if c_val in ("1", "true", "yes", "censored", "suspended"):
                        c = 1
                    else:
                        c = 0
                except Exception:
                    c = 0

            times_list.append(t)
            censored_list.append(c)

        if len(times_list) < 3:
            raise HTTPException(status_code=400, detail=f"Need at least 3 observations. Got {len(times_list)}.")

        times = np.array(times_list)
        censored = np.array(censored_list)

        n_failed = int((censored == 0).sum())
        if n_failed < 2:
            raise HTTPException(status_code=400, detail=f"Need at least 2 failure observations. Got {n_failed} (rest are censored).")

        # ── 3. Compute ──
        result = compute_warranty(
            times=times,
            censored=censored,
            warranty_period=request.warrantyPeriod,
            unit_cost=request.unitCost,
            total_units=request.totalUnits,
            confidence_level=request.confidenceLevel,
        )

        return _to_native({"results": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
