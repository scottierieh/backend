"""
valuation.py
POST /dcf              — DCF model (dcf-model-page)
POST /multiples        — multiples table (multiples-page)
POST /relative         — relative valuation via regression (relative-valuation-page)
"""

import math
import traceback
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils import safe_float, to_native, _arr, mean, std, Returns
import numpy as np
from scipy import stats as scipy_stats
import statsmodels.api as sm

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# 1. DCF Model
# ══════════════════════════════════════════════════════════════════════════════

class DCFRequest(BaseModel):
    ticker:        str
    currentFCF:    float          # free cash flow (latest year)
    growthRate1:   float = 10.0   # % growth years 1-5
    growthRate2:   float = 5.0    # % growth years 6-10
    terminalGrowth: float = 2.5   # % terminal growth
    discountRate:  float = 10.0   # WACC %
    sharesOut:     float = 1.0    # shares outstanding (same unit as FCF)
    currentPrice:  Optional[float] = None
    # WACC inputs (optional, overrides discountRate if provided)
    beta:          Optional[float] = None
    riskFreeRate:  Optional[float] = None   # %
    erp:           Optional[float] = None   # equity risk premium %
    costOfDebt:    Optional[float] = None   # %
    taxRate:       Optional[float] = None   # %
    debtRatio:     Optional[float] = None   # D/(D+E)


def _capm_ke(rf: float, beta: float, erp: float) -> float:
    return rf + beta * erp


def _wacc(ke: float, kd: float, tax: float, e_ratio: float) -> float:
    """All inputs in decimal form.
    ke = cost of equity (e.g. 0.10 for 10%)
    kd = cost of debt   (e.g. 0.06 for 6%)
    tax = tax rate      (e.g. 0.25 for 25%)
    e_ratio = equity / (debt + equity)  (e.g. 0.70)
    """
    d_ratio = 1.0 - e_ratio
    return ke * e_ratio + kd * (1.0 - tax) * d_ratio


@router.post("/dcf")
def run_dcf(req: DCFRequest):
    try:
        # Determine discount rate
        wacc = req.discountRate / 100
        if all(v is not None for v in [req.beta, req.riskFreeRate, req.erp, req.costOfDebt, req.taxRate, req.debtRatio]):
            ke      = _capm_ke(req.riskFreeRate / 100, req.beta, req.erp / 100)
            kd      = req.costOfDebt / 100
            tax_dec = req.taxRate / 100
            e_ratio = 1.0 - req.debtRatio / 100
            wacc    = _wacc(ke, kd, tax_dec, e_ratio)   # all decimals

        r1 = req.growthRate1 / 100
        r2 = req.growthRate2 / 100
        tg = req.terminalGrowth / 100

        fcf = req.currentFCF
        cashflows = []
        pv_total  = 0.0
        fcf_t     = fcf

        # Stage 1: years 1-5
        for yr in range(1, 6):
            fcf_t *= (1 + r1)
            pv = fcf_t / (1 + wacc) ** yr
            pv_total += pv
            cashflows.append({"year": yr, "fcf": safe_float(fcf_t), "pv": safe_float(pv)})

        # Stage 2: years 6-10
        for yr in range(6, 11):
            fcf_t *= (1 + r2)
            pv = fcf_t / (1 + wacc) ** yr
            pv_total += pv
            cashflows.append({"year": yr, "fcf": safe_float(fcf_t), "pv": safe_float(pv)})

        # Terminal value
        if wacc <= tg:
            raise HTTPException(status_code=400, detail="Discount rate must exceed terminal growth rate.")
        tv   = fcf_t * (1 + tg) / (wacc - tg)
        pv_tv = tv / (1 + wacc) ** 10
        total_pv = pv_total + pv_tv

        intrinsic_per_share = total_pv / req.sharesOut if req.sharesOut > 0 else 0.0
        margin_of_safety    = None
        upside              = None
        if req.currentPrice and req.currentPrice > 0:
            margin_of_safety = (intrinsic_per_share - req.currentPrice) / req.currentPrice
            upside           = intrinsic_per_share / req.currentPrice - 1

        # ── Sensitivity table (WACC rows × terminal growth cols) ─────────────
        # Precompute the FCF path (year 1-10) once — it depends only on growth
        # rates (r1, r2), not on WACC or terminal growth. Reusing it avoids
        # the misleading comment "cashflows don't change" while actually
        # recomputing them inside the loop.
        fcf_path: List[float] = []
        f_tmp = req.currentFCF
        for yr in range(1, 11):
            f_tmp *= (1 + r1) if yr <= 5 else (1 + r2)
            fcf_path.append(f_tmp)
        # fcf_path[-1] == fcf_t (year-10 FCF, used for terminal value)

        wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
        tg_range   = [max(0.005, tg - 0.01), tg, min(wacc - 0.005, tg + 0.01)]
        sensitivity: List[dict] = []
        for w in wacc_range:
            row_s: dict = {"wacc": safe_float(w * 100)}
            for g in tg_range:
                if w <= g:
                    row_s[f"tg_{g*100:.1f}"] = None
                    continue
                # PV of cashflows: discount precomputed path at new WACC
                pv_s    = sum(f / (1 + w) ** yr for yr, f in enumerate(fcf_path, 1))
                # Terminal value: uses year-10 FCF and new tg/wacc
                tv_s    = fcf_path[-1] * (1 + g) / (w - g)
                pv_tv_s = tv_s / (1 + w) ** 10
                ips_s   = (pv_s + pv_tv_s) / req.sharesOut if req.sharesOut > 0 else 0.0
                row_s[f"tg_{g*100:.1f}"] = safe_float(ips_s)
            sensitivity.append(row_s)

        # Bull / Base / Bear scenarios
        # Each scenario gets its own FCF path (different growth rates),
        # so we can't reuse fcf_path here — but the structure is now explicit.
        def _scenario_pv(r1_s, r2_s, w_s):
            """PV of 10-year cashflows + terminal value under scenario parameters."""
            path: List[float] = []
            f = req.currentFCF
            for yr in range(1, 11):
                f *= (1 + r1_s) if yr <= 5 else (1 + r2_s)
                path.append(f)
            pv   = sum(f / (1 + w_s) ** yr for yr, f in enumerate(path, 1))
            tv   = path[-1] * (1 + tg) / (w_s - tg)
            pv_tv = tv / (1 + w_s) ** 10
            return pv + pv_tv

        scenarios = []
        for label, g1_mult, g2_mult, w_mult in [
            ("Bull",  1.3, 1.2, 0.9),
            ("Base",  1.0, 1.0, 1.0),
            ("Bear",  0.6, 0.7, 1.1),
        ]:
            w_s = wacc * w_mult
            if w_s <= tg:
                scenarios.append({"label": label, "intrinsic": None, "upside": None})
                continue
            total_s = _scenario_pv(r1 * g1_mult, r2 * g2_mult, w_s)
            ips_s   = total_s / req.sharesOut if req.sharesOut > 0 else 0.0
            up_s    = (ips_s / req.currentPrice - 1) if req.currentPrice and req.currentPrice > 0 else None
            scenarios.append({
                "label":     label,
                "intrinsic": safe_float(ips_s),
                "upside":    safe_float(up_s) if up_s is not None else None,
            })

        return to_native({
            "results": {
                "ticker":            req.ticker,
                "cashflows":         cashflows,
                "terminalValue":     safe_float(tv),
                "pvTerminalValue":   safe_float(pv_tv),
                "pvCashflows":       safe_float(pv_total),
                "totalPV":           safe_float(total_pv),
                "intrinsicPerShare": safe_float(intrinsic_per_share),
                "marginOfSafety":    safe_float(margin_of_safety) if margin_of_safety is not None else None,
                "upside":            safe_float(upside) if upside is not None else None,
                "wacc":              safe_float(wacc),
                "sensitivity":       sensitivity,
                "scenarios":         scenarios,
                "tgRange":           [safe_float(g * 100) for g in tg_range],
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Multiples
# ══════════════════════════════════════════════════════════════════════════════

class AssetMultiples(BaseModel):
    ticker:        str
    pe:            Optional[float] = None
    pb:            Optional[float] = None
    evEbitda:      Optional[float] = None
    divYield:      Optional[float] = None
    fcfYield:      Optional[float] = None
    roe:           Optional[float] = None
    grossMargin:   Optional[float] = None
    revenueGrowth: Optional[float] = None
    debtEquity:    Optional[float] = None
    marketCap:     Optional[float] = None


class MultiplesRequest(BaseModel):
    assets: List[AssetMultiples]


MULTIPLE_KEYS = ["pe","pb","evEbitda","divYield","fcfYield","roe","grossMargin","revenueGrowth","debtEquity"]
LOWER_BETTER  = {"pe","pb","evEbitda","debtEquity"}


@router.post("/multiples")
def run_multiples(req: MultiplesRequest):
    try:
        assets = req.assets
        stats: Dict[str, Any] = {}

        for key in MULTIPLE_KEYS:
            vals = [getattr(a, key) for a in assets if getattr(a, key) is not None]
            if not vals:
                stats[key] = {"median": None, "mean": None, "min": None, "max": None}
            else:
                vals_sorted = sorted(vals)
                n = len(vals_sorted)
                med = (vals_sorted[n//2 - 1] + vals_sorted[n//2]) / 2 if n % 2 == 0 else vals_sorted[n//2]
                stats[key] = {
                    "median": safe_float(med),
                    "mean":   safe_float(mean(vals)),
                    "min":    safe_float(min(vals)),
                    "max":    safe_float(max(vals)),
                    "count":  n,
                }

        # Per-asset discount/premium vs median
        asset_rows = []
        for a in assets:
            row: Dict[str, Any] = {"ticker": a.ticker}
            for key in MULTIPLE_KEYS:
                val = getattr(a, key)
                med = stats[key]["median"] if stats[key]["median"] is not None else None
                if val is not None and med is not None and med != 0:
                    prem = (val - med) / abs(med)
                    cheap = prem < 0 if key in LOWER_BETTER else prem > 0
                    row[key] = {"value": safe_float(val), "premium": safe_float(prem), "cheap": cheap}
                else:
                    row[key] = {"value": val, "premium": None, "cheap": None}
            asset_rows.append(row)

        return to_native({"results": {"assets": asset_rows, "sectorStats": stats}})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Relative Valuation (regression-based)
# ══════════════════════════════════════════════════════════════════════════════

class RelativeValuationRequest(BaseModel):
    assets:    List[AssetMultiples]
    xMetric:   str = "revenueGrowth"   # predictor
    yMetric:   str = "pe"              # outcome (e.g. P/E)
    winsorise: bool = True


def _winsorise(vals: List[float], p: float = 5.0) -> List[float]:
    n   = len(vals)
    lo  = sorted(vals)[max(0, int(n * p / 100) - 1)]
    hi  = sorted(vals)[min(n - 1, int(n * (1 - p / 100)))]
    return [max(lo, min(hi, v)) for v in vals]


def _ols_regression(x: List[float], y: List[float]):
    n   = len(x)
    mx, my = mean(x), mean(y)
    ss_xx  = sum((xi - mx) ** 2 for xi in x)
    ss_xy  = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b1     = ss_xy / ss_xx if ss_xx > 0 else 0.0
    b0     = my - b1 * mx
    y_hat  = [b0 + b1 * xi for xi in x]
    residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]
    ss_res    = sum(r ** 2 for r in residuals)
    ss_tot    = sum((yi - my) ** 2 for yi in y)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b0, b1, r2, y_hat, residuals


@router.post("/relative")
def run_relative_valuation(req: RelativeValuationRequest):
    try:
        pairs = [
            (a, getattr(a, req.xMetric), getattr(a, req.yMetric))
            for a in req.assets
            if getattr(a, req.xMetric) is not None and getattr(a, req.yMetric) is not None
        ]
        if len(pairs) < 3:
            raise HTTPException(status_code=400, detail="Need ≥ 3 assets with both metrics populated.")

        assets_f, x_vals, y_vals = zip(*pairs)
        x_list = list(x_vals)
        y_list = list(y_vals)

        if req.winsorise:
            x_list = _winsorise(x_list)
            y_list = _winsorise(y_list)

        b0, b1, r2, y_hat, residuals = _ols_regression(x_list, y_list)

        # Z-score residuals
        res_std = std(residuals) or 1.0
        z_resids = [r / res_std for r in residuals]

        result_assets = []
        for i, a in enumerate(assets_f):
            result_assets.append({
                "ticker":    a.ticker,
                "x":         safe_float(x_list[i]),
                "y":         safe_float(y_list[i]),
                "yHat":      safe_float(y_hat[i]),
                "residual":  safe_float(residuals[i]),
                "zResidual": safe_float(z_resids[i]),
                "cheap":     z_resids[i] < -1.0,
                "expensive":  z_resids[i] > 1.0,
            })

        # Regression line (for chart)
        x_min, x_max = min(x_list), max(x_list)
        line = [
            {"x": safe_float(x_min), "y": safe_float(b0 + b1 * x_min)},
            {"x": safe_float(x_max), "y": safe_float(b0 + b1 * x_max)},
        ]

        # Residual histogram
        res_arr = np.array(residuals, dtype=float)
        counts, edges = np.histogram(res_arr, bins=min(10, len(res_arr)))
        res_hist = [
            {"range": f"{edges[k]:.2f}", "count": int(counts[k])}
            for k in range(len(counts))
        ]

        return to_native({
            "results": {
                "assets":    result_assets,
                "line":      line,
                "intercept": safe_float(b0),
                "slope":     safe_float(b1),
                "r2":        safe_float(r2),
                "xMetric":   req.xMetric,
                "yMetric":   req.yMetric,
                "residualHist": res_hist,
                "cheapAssets":   [a["ticker"] for a in result_assets if a["cheap"]],
                "expensiveAssets":[a["ticker"] for a in result_assets if a["expensive"]],
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
