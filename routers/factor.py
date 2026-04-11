"""
factor.py
POST /beta-alpha             — statsmodels HAC-robust OLS: alpha, beta, t-stats, p-values
POST /factor-exposure        — style factor exposures (numpy vectorised)
POST /rolling-beta           — rolling OLS beta
POST /performance-attribution — Brinson-Hood-Beebower
"""
import math, traceback
from typing import List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas import AssetIn, BenchmarkIn
from utils import safe_float, to_native, _arr

router = APIRouter()


def _align(a, b):
    L = min(len(a), len(b))
    return np.asarray(a[-L:], dtype=float), np.asarray(b[-L:], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Beta / Alpha — HAC-robust OLS with full inference
# ══════════════════════════════════════════════════════════════════════════════

class BetaAlphaRequest(BaseModel):
    assets:      List[AssetIn]
    benchmark:   BenchmarkIn
    freqPerYear: int = 12


@router.post("/beta-alpha")
def run_beta_alpha(req: BetaAlphaRequest):
    try:
        freq    = req.freqPerYear
        bm_full = _arr(req.benchmark.returns)
        results = []

        for a in req.assets:
            ar, br = _align(a.returns, bm_full)
            n      = len(ar)

            # HAC-robust OLS (Newey-West SE)
            X     = sm.add_constant(br)
            model = sm.OLS(ar, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": min(3, n // 10)}
            )
            alpha_per = float(model.params[0])
            beta      = float(model.params[1])
            alpha_ann = float((1 + alpha_per) ** freq - 1)

            ann_vol_a  = float(ar.std(ddof=1)) * math.sqrt(freq)
            te         = float((ar - br).std(ddof=1)) * math.sqrt(freq)
            info_ratio = (float(ar.mean()) - float(br.mean())) * freq / te if te > 0 else 0.0

            # Risk decomposition: total = systematic + idiosyncratic
            sys_var   = beta ** 2 * float(br.var(ddof=1)) * freq
            total_var = float(ar.var(ddof=1)) * freq
            idio_var  = max(0.0, total_var - sys_var)

            # Residuals for distribution chart
            residuals = (ar - model.fittedvalues).tolist()
            # Residual histogram (20 bins)
            res_arr = np.array(residuals)
            counts, edges = np.histogram(res_arr, bins=20)
            res_hist = [
                {"range": f"{edges[k]:.3%}", "count": int(counts[k])}
                for k in range(len(counts))
            ]

            # Beta confidence interval
            beta_se = float(model.bse[1])
            beta_ci = [beta - 1.96 * beta_se, beta + 1.96 * beta_se]

            # Regression scatter (subsample 60 points max)
            step = max(1, n // 60)
            scatter = [
                {"bmkRet":  safe_float(float(br[i])),
                 "portRet": safe_float(float(ar[i])),
                 "fitted":  safe_float(float(model.fittedvalues[i]))}
                for i in range(0, n, step)
            ]

            # Rolling alpha (window = max(6, freq//2))
            rw = max(6, freq // 2)
            rolling_alpha = [None] * n
            for i in range(rw, n + 1):
                wi_a = ar[i-rw:i]
                wi_b = br[i-rw:i]
                Xw   = sm.add_constant(wi_b)
                try:
                    m_w = sm.OLS(wi_a, Xw).fit()
                    a_ann = float((1 + float(m_w.params[0])) ** freq - 1)
                    rolling_alpha[i-1] = safe_float(a_ann)
                except Exception:
                    pass

            results.append({
                "ticker":          a.ticker,
                "beta":            safe_float(beta),
                "betaCI":          [safe_float(v) for v in beta_ci],
                "alpha":           safe_float(alpha_per),
                "annAlpha":        safe_float(alpha_ann),
                "alphaTStat":      safe_float(float(model.tvalues[0])),
                "alphaPValue":     safe_float(float(model.pvalues[0])),
                "alphaSig":        bool(abs(float(model.tvalues[0])) > 2.0),
                "betaTStat":       safe_float(float(model.tvalues[1])),
                "betaPValue":      safe_float(float(model.pvalues[1])),
                "rSquared":        safe_float(float(model.rsquared)),
                "adjRSquared":     safe_float(float(model.rsquared_adj)),
                "correlation":     safe_float(float(np.corrcoef(ar, br)[0, 1])),
                "trackingError":   safe_float(te),
                "informationRatio":safe_float(info_ratio),
                "annVol":          safe_float(ann_vol_a),
                "treynor":         safe_float(float(ar.mean() * freq) / abs(beta)) if beta != 0 else 0.0,
                "riskDecomp": {
                    "totalRisk":        safe_float(math.sqrt(total_var)),
                    "systematicRisk":   safe_float(math.sqrt(sys_var)),
                    "idiosyncraticRisk":safe_float(math.sqrt(idio_var)),
                    "systematicPct":    safe_float(sys_var  / total_var * 100) if total_var > 0 else None,
                    "idiosyncraticPct": safe_float(idio_var / total_var * 100) if total_var > 0 else None,
                },
                "residuals":     [safe_float(v) for v in residuals[:200]],
                "residualHist":  res_hist,
                "rollingAlpha":  rolling_alpha,
                "scatter":       scatter,
            })

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Factor Exposure — style factors (numpy vectorised)
# ══════════════════════════════════════════════════════════════════════════════

class FactorExposureRequest(BaseModel):
    assets:     List[AssetIn]
    normMethod: str = "rank"
    rollWindow: int = 12


@router.post("/factor-exposure")
def run_factor_exposure(req: FactorExposureRequest):
    try:
        assets = req.assets
        n      = len(assets)

        def _raw(a: AssetIn):
            r    = _arr(a.returns)
            p    = _arr(a.prices) if a.prices else None
            def _ts_mom(lb):
                if p is not None and len(p) > lb:
                    return float((p[-1] - p[-lb-1]) / p[-lb-1])
                if len(r) >= lb:
                    return float(np.prod(1 + r[-lb:]) - 1)
                return 0.0
            vol12 = float(r[-12:].std(ddof=1)) if len(r) >= 12 else float(r.std(ddof=1))
            return {
                "momentum":  _ts_mom(12),
                "lowVol":   -vol12,
                "shortTerm": _ts_mom(1),
                "medTerm":   _ts_mom(6),
                "longTerm":  _ts_mom(24) if len(r) >= 24 else _ts_mom(12),
            }

        raw     = [_raw(a) for a in assets]
        factors = list(raw[0].keys())

        def _norm(vals):
            v = np.array(vals, dtype=float)
            if req.normMethod == "zscore":
                s = v.std(ddof=1)
                return ((v - v.mean()) / s).tolist() if s > 0 else np.zeros(n).tolist()
            ranks = np.argsort(np.argsort(v)).astype(float)
            return (2 * ranks / max(n - 1, 1) - 1).tolist()

        norm_by_f = {f: _norm([r[f] for r in raw]) for f in factors}

        result_assets = [
            {
                "ticker": a.ticker,
                "scores": {f: safe_float(norm_by_f[f][i]) for f in factors},
                # Dominant factor label
                "dominantFactor": max(factors, key=lambda f: abs(norm_by_f[f][i])),
            }
            for i, a in enumerate(assets)
        ]

        # Factor summary (mean, std, top/bottom exposure)
        factor_summary = []
        for f in factors:
            vals = np.array(norm_by_f[f], dtype=float)
            sorted_vals = np.sort(vals)[::-1]
            factor_summary.append({
                "factor": f,
                "mean":   safe_float(float(vals.mean())),
                "std":    safe_float(float(vals.std(ddof=1))) if n > 1 else None,
                "max":    safe_float(float(vals.max())),
                "min":    safe_float(float(vals.min())),
                "topExposureTicker":    assets[int(np.argmax(vals))].ticker if n > 0 else None,
                "bottomExposureTicker": assets[int(np.argmin(vals))].ticker if n > 0 else None,
            })

        # Factor-factor correlation matrix
        factor_matrix: List[List[float]] = []
        for f1 in factors:
            row = []
            for f2 in factors:
                v1 = np.array(norm_by_f[f1], dtype=float)
                v2 = np.array(norm_by_f[f2], dtype=float)
                if v1.std() > 0 and v2.std() > 0:
                    corr = float(np.corrcoef(v1, v2)[0, 1])
                else:
                    corr = 1.0 if f1 == f2 else 0.0
                row.append(safe_float(corr))
            factor_matrix.append(row)

        # High vs Low factor portfolio return (top half vs bottom half)
        factor_returns: List[dict] = []
        for f in factors:
            scores = norm_by_f[f]
            sorted_idx = np.argsort(scores)
            half = max(1, n // 2)
            high_idx = sorted_idx[-half:]
            low_idx  = sorted_idx[:half]

            def _avg_cum_ret(indices):
                rets = []
                for idx in indices:
                    r = _arr(assets[idx].returns)
                    rets.append(float(np.prod(1 + r) - 1))
                return float(np.mean(rets)) if rets else 0.0

            factor_returns.append({
                "factor":      f,
                "highPortRet": safe_float(_avg_cum_ret(high_idx)),
                "lowPortRet":  safe_float(_avg_cum_ret(low_idx)),
                "spread":      safe_float(_avg_cum_ret(high_idx) - _avg_cum_ret(low_idx)),
            })

        # Factor score distribution histogram per factor
        factor_hist: dict = {}
        for f in factors:
            vals = np.array(norm_by_f[f], dtype=float)
            counts, edges = np.histogram(vals, bins=min(10, n))
            factor_hist[f] = [
                {"range": f"{edges[k]:.2f}", "count": int(counts[k])}
                for k in range(len(counts))
            ]

        # Rolling momentum exposure per asset (w-period compounded return, annualised sign)
        w = req.rollWindow
        rolling_by_asset: dict = {}
        for a in assets:
            r0 = _arr(a.returns)
            series = [
                safe_float(float(np.prod(1 + r0[i-w:i]) - 1))
                for i in range(w, len(r0) + 1)
            ]
            rolling_by_asset[a.ticker] = series
        # Keep backward-compat "rolling" key as the first asset for chart use
        rolling = rolling_by_asset.get(assets[0].ticker, []) if assets else []

        return to_native({
            "assets":        result_assets,
            "factors":       factors,
            "factorSummary": factor_summary,
            "factorMatrix":  factor_matrix,
            "factorReturns": factor_returns,
            "factorHist":    factor_hist,
            "rolling":          rolling,
            "rollingByAsset":   rolling_by_asset,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rolling Beta — OLS per window
# ══════════════════════════════════════════════════════════════════════════════

class RollingBetaRequest(BaseModel):
    assets:    List[AssetIn]
    benchmark: BenchmarkIn
    window:    int = 12


@router.post("/rolling-beta")
def run_rolling_beta(req: RollingBetaRequest):
    try:
        bm = _arr(req.benchmark.returns)
        results = []
        for a in req.assets:
            ar, br = _align(a.returns, bm)
            n      = len(ar)
            w      = req.window
            rb     = []
            ra     = []     # rolling alpha (annualised)
            freq   = 12     # default; rolling alpha is directional only

            for i in range(w, n + 1):
                wi = ar[i-w:i]
                wb = br[i-w:i]
                wb_dm = wb - wb.mean()
                ss_xx = float(np.dot(wb_dm, wb_dm))
                ss_xy = float(np.dot(wb_dm, wi - wi.mean()))
                beta  = ss_xy / ss_xx if ss_xx > 0 else 0.0
                alpha = float(wi.mean() - beta * wb.mean())
                rb.append(safe_float(beta))
                ra.append(safe_float(alpha * freq))   # annualised per-period alpha

            # Regime labels based on rolling beta vs median
            valid_betas = [v for v in rb if v is not None]
            med_beta    = float(np.median(valid_betas)) if valid_betas else 1.0
            regimes     = [
                "high" if (v or 0) > med_beta * 1.2
                else "low" if (v or 0) < med_beta * 0.8
                else "normal"
                for v in rb
            ]

            # Beta distribution histogram (20 bins)
            if valid_betas:
                counts, edges = np.histogram(valid_betas, bins=min(20, len(valid_betas)))
                beta_hist = [
                    {"range": f"{edges[k]:.3f}", "count": int(counts[k])}
                    for k in range(len(counts))
                ]
            else:
                beta_hist = []

            # Summary
            arr_b = np.array(valid_betas, dtype=float) if valid_betas else np.array([0.0])
            summary = {
                "avg": safe_float(float(arr_b.mean())),
                "max": safe_float(float(arr_b.max())),
                "min": safe_float(float(arr_b.min())),
                "std": safe_float(float(arr_b.std(ddof=1))) if len(arr_b) > 1 else None,
            }

            results.append({
                "ticker":      a.ticker,
                "rollingBeta": rb,
                "rollingAlpha": ra,
                "regimes":     regimes,
                "betaHist":    beta_hist,
                "summary":     summary,
            })
        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Performance Attribution — Brinson-Hood-Beebower
# ══════════════════════════════════════════════════════════════════════════════

class Segment(BaseModel):
    name:            str
    portfolioWeight: float
    benchmarkWeight: float
    portfolioReturn: float
    benchmarkReturn: float

class PerfAttrRequest(BaseModel):
    segments:       List[Segment]
    portfolioTotal: float
    benchmarkTotal: float


@router.post("/performance-attribution")
def run_perf_attr(req: PerfAttrRequest):
    try:
        bm_total = req.benchmarkTotal
        rows     = []
        totals   = {"allocation": 0.0, "selection": 0.0, "interaction": 0.0}

        for s in req.segments:
            alloc = (s.portfolioWeight - s.benchmarkWeight) * (s.benchmarkReturn - bm_total)
            sel   = s.benchmarkWeight * (s.portfolioReturn - s.benchmarkReturn)
            inter = (s.portfolioWeight - s.benchmarkWeight) * (s.portfolioReturn - s.benchmarkReturn)
            for k, v in [("allocation", alloc), ("selection", sel), ("interaction", inter)]:
                totals[k] += v
            rows.append({
                "name":        s.name,
                "allocation":  safe_float(alloc),
                "selection":   safe_float(sel),
                "interaction": safe_float(inter),
                "active":      safe_float(alloc + sel + inter),
            })

        return to_native({
            "segments":         rows,
            "totalAllocation":  safe_float(totals["allocation"]),
            "totalSelection":   safe_float(totals["selection"]),
            "totalInteraction": safe_float(totals["interaction"]),
            "activeReturn":     safe_float(req.portfolioTotal - bm_total),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Multi-Benchmark Comparison
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkSpec(BaseModel):
    name:          str
    returns:       List[float]
    periods:       Optional[List[str]] = None
    benchmarkType: str = "index"       # index | factor | custom | equal_weight | cap_weight
    rfRate:        float = 0.0         # annual % — per-benchmark override


class MultiBenchmarkRequest(BaseModel):
    assets:      List[AssetIn]
    benchmarks:  List[BenchmarkSpec]   # 1–5 benchmarks
    freqPerYear: int   = 12
    rfRate:      float = 4.0           # global default annual %


def _norm_period(p: str) -> str:
    """Normalise any date string to YYYY-MM for comparison."""
    return p[:7]

def _align_np(port_r: np.ndarray, port_p: List[str],
               bmk_r: np.ndarray,  bmk_p: List[str]):
    """Inner-join on period labels (normalised to YYYY-MM). Falls back to length alignment."""
    if not port_p or not bmk_p:
        L = min(len(port_r), len(bmk_r))
        return port_r[-L:], bmk_r[-L:]
    # Normalise benchmark map keys to YYYY-MM
    bmk_map = {_norm_period(p): bmk_r[i] for i, p in enumerate(bmk_p)}
    port_aligned, bmk_aligned = [], []
    for i, p in enumerate(port_p):
        key = _norm_period(p)
        if key in bmk_map:
            port_aligned.append(port_r[i])
            bmk_aligned.append(bmk_map[key])
    if not port_aligned:
        L = min(len(port_r), len(bmk_r))
        return port_r[-L:], bmk_r[-L:]
    return np.array(port_aligned), np.array(bmk_aligned)


def _rebalance_returns(component_returns: np.ndarray,
                       method: str = "equal_weight") -> np.ndarray:
    """
    Simulate a rebalanced benchmark from component returns.
    component_returns: shape (T, N)
    method: equal_weight | cap_weight (cap_weight uses cumulative wealth as proxy)
    """
    T, N = component_returns.shape
    if method == "equal_weight":
        # Monthly rebalance to equal weight
        weights = np.ones(N) / N
        out = np.zeros(T)
        for t in range(T):
            out[t] = float(component_returns[t] @ weights)
            # Rebalance every period (equal-weight = always rebalanced)
        return out
    elif method == "cap_weight":
        # Buy-and-hold from equal start, let weights drift (approx cap-weight)
        cum = np.ones(N)
        out = np.zeros(T)
        for t in range(T):
            total = cum.sum()
            w = cum / total
            out[t] = float(component_returns[t] @ w)
            cum *= (1 + component_returns[t])
        return out
    return component_returns.mean(axis=1)


def _factor_alpha(port_r: np.ndarray, factor_returns: List[np.ndarray],
                  rf_per: float) -> dict:
    """
    Multi-factor OLS: Rp - rf = α + β1*F1 + β2*F2 + ... + ε
    Returns alpha, betas, t-stats, p-values, R²
    """
    n = len(port_r)
    if n < len(factor_returns) + 5:
        return {"alpha": 0, "betas": [], "r2": 0}
    y = port_r - rf_per
    X = np.column_stack([np.ones(n)] + list(factor_returns))
    try:
        import statsmodels.api as sm
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': min(3, n//10)})
        return {
            "alpha":   safe_float(float(model.params[0])),
            "betas":   [safe_float(float(b)) for b in model.params[1:]],
            "tStats":  [safe_float(float(t)) for t in model.tvalues],
            "pValues": [safe_float(float(p)) for p in model.pvalues],
            "r2":      safe_float(float(model.rsquared)),
            "adjR2":   safe_float(float(model.rsquared_adj)),
        }
    except Exception:
        return {"alpha": 0, "betas": [], "r2": 0}


@router.post("/multi-benchmark")
def run_multi_benchmark(req: MultiBenchmarkRequest):
    try:
        freq   = req.freqPerYear

        # Portfolio weighted returns
        min_len = min(len(a.returns) for a in req.assets)
        ws      = np.array([a.weight for a in req.assets]); ws /= ws.sum()
        ret_mat = np.column_stack([_arr(a.returns)[-min_len:] for a in req.assets])
        port_r  = ret_mat @ ws
        port_p  = req.assets[0].periods or [] if req.assets else []
        # periods is optional on AssetIn — guard
        port_p_list: List[str] = []
        if req.assets and req.assets[0].periods:
            raw_p = req.assets[0].periods
            # periods[0] is the base price date; returns[i] corresponds to periods[i+1]
            port_p_list = raw_p[1:] if len(raw_p) > len(port_r) else raw_p

        results = []
        factor_rets_for_fama: List[np.ndarray] = []

        for bmk in req.benchmarks:
            bmk_r = _arr(bmk.returns)
            bmk_p = bmk.periods or []
            rf_ann = bmk.rfRate if bmk.rfRate > 0 else req.rfRate
            rf_per = rf_ann / 100 / freq

            # ── Rebalancing simulation for equal/cap weight benchmarks ─────
            # (For single-series benchmarks, no rebalancing needed)
            # Future: accept component_returns and simulate here

            # ── Align ───────────────────────────────────────────────────────
            pa, ba = _align_np(port_r, port_p_list, bmk_r, bmk_p)
            n = len(pa)
            if n < 4:
                continue

            # ── CAPM / multi-factor alpha ──────────────────────────────────
            if bmk.benchmarkType == "factor":
                factor_rets_for_fama.append(ba)

            # Beta, alpha (CAPM) — excess return regression
            # Rp - rf = alpha + beta * (Rm - rf) + eps
            import statsmodels.api as sm
            pa_ex = pa - rf_per          # portfolio excess return
            ba_ex = ba - rf_per          # benchmark excess return
            X = sm.add_constant(ba_ex)
            ols = sm.OLS(pa_ex, X).fit(
                cov_type='HAC', cov_kwds={'maxlags': min(3, n // 10)}
            )
            beta_val  = safe_float(float(ols.params[1]))
            alpha_per = safe_float(float(ols.params[0]))   # per-period alpha (already rf-adjusted)
            ann_alpha = float((1 + alpha_per) ** freq - 1)
            ann_te    = float((pa - ba).std(ddof=1)) * float(np.sqrt(freq))
            ann_ir    = (ann_alpha / ann_te) if ann_te > 0 else 0.0

            # Cumulative curves
            port_cum = np.cumprod(1 + pa).tolist()
            bmk_cum  = np.cumprod(1 + ba).tolist()

            # Active return series
            active = (pa - ba).tolist()
            cum_active = np.cumsum(pa - ba).tolist()

            # Up/Down capture
            up_mask  = ba > 0
            dn_mask  = ba < 0
            up_cap   = (pa[up_mask].mean() / ba[up_mask].mean()) if up_mask.any() and ba[up_mask].mean() != 0 else 0.0
            dn_cap   = (pa[dn_mask].mean() / ba[dn_mask].mean()) if dn_mask.any() and ba[dn_mask].mean() != 0 else 0.0

            results.append({
                "name":          bmk.name,
                "benchmarkType": bmk.benchmarkType,
                "nAligned":      n,
                "overlapPct":    safe_float(n / len(port_r) * 100),
                "rfRate":        safe_float(rf_ann),
                # CAPM stats
                "beta":          safe_float(beta_val),
                "betaTStat":     safe_float(float(ols.tvalues[1])),
                "betaPValue":    safe_float(float(ols.pvalues[1])),
                "alphaPer":      safe_float(alpha_per),
                "annAlpha":      safe_float(ann_alpha),
                "alphaTStat":    safe_float(float(ols.tvalues[0])),
                "alphaPValue":   safe_float(float(ols.pvalues[0])),
                "trackingError": safe_float(ann_te),
                "infoRatio":     safe_float(ann_ir),
                "r2":            safe_float(float(ols.rsquared)),
                "correlation":   safe_float(float(np.corrcoef(pa, ba)[0, 1])),
                # Capture ratios
                "upCapture":     safe_float(float(up_cap) * 100),
                "downCapture":   safe_float(float(dn_cap) * 100),
                # Return series for charts
                "portCurve":     [safe_float(v) for v in port_cum],
                "bmkCurve":      [safe_float(v) for v in bmk_cum],
                "activeSeries":  [safe_float(v * 100) for v in active],
                "cumActive":     [safe_float(v * 100) for v in cum_active],
            })

        # Multi-factor alpha (if factor benchmarks provided)
        factor_result = None
        if factor_rets_for_fama:
            factor_result = _factor_alpha(port_r, factor_rets_for_fama, req.rfRate / 100 / freq)

        return to_native({
            "benchmarks":   results,
            "factorAlpha":  factor_result,
            "portLength":   len(port_r),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
