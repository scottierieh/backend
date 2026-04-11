"""
risk.py
POST /var-cvar        — Historical percentile VaR + Cornish-Fisher adjusted VaR + historical CVaR
                        (no separate parametric/normal VaR — "parametric" in older docs refers to CF)
POST /drawdown        — drawdown episodes
POST /sharpe-sortino  — QuantStats: sharpe, sortino, calmar, omega, tail-ratio
POST /stress-testing  — scenario shock proxy: equity shock + GARCH vol multiplier.
                        NOTE: corrShock is defined in scenario params but not applied to the
                        covariance matrix. This is a shock-augmented risk proxy, not a full
                        stressed-covariance model.
POST /risk-attribution — MCTR/CTR via full covariance
POST /liquidity       — liquidity-adjusted VaR (linear cost approximation, not full L-VaR)
"""
import math, traceback
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as scipy_stats
from scipy.stats import norm as norm_dist

import quantstats as qs
from arch import arch_model

from schemas import AssetIn
from utils import safe_float, to_native, _arr, drawdown_series, cov_matrix, port_vol

router = APIRouter()


def _port_returns(assets: List[AssetIn]) -> np.ndarray:
    # Use the shortest common window — 0-padding absent periods as 0% return
    # biases vol downward, understates VaR, and distorts the mean.
    min_len = min(len(a.returns) for a in assets)
    ws  = np.array([a.weight for a in assets])
    ws /= ws.sum()
    mat = np.column_stack([_arr(a.returns)[-min_len:] for a in assets])
    return mat @ ws


def _cornish_fisher_var(r: np.ndarray, alpha: float = 0.05) -> float:
    mu, sig = r.mean(), r.std(ddof=1)
    sk  = float(scipy_stats.skew(r))
    ku  = float(scipy_stats.kurtosis(r))   # excess
    z   = norm_dist.ppf(alpha)
    z_cf = z + (z**2-1)*sk/6 + (z**3-3*z)*ku/24 - (2*z**3-5*z)*sk**2/36
    return float(mu + z_cf * sig)


# ══════════════════════════════════════════════════════════════════════════════
# 1. VaR / CVaR  — three methods
# ══════════════════════════════════════════════════════════════════════════════

class VarCvarRequest(BaseModel):
    assets:     List[AssetIn]
    confidence: float = 0.95


@router.post("/var-cvar")
def run_var_cvar(req: VarCvarRequest):
    try:
        alpha  = 1 - req.confidence
        port_r = _port_returns(req.assets)

        def _metrics(r: np.ndarray):
            var_hist  = float(np.percentile(r, alpha * 100))
            var_norm  = _cornish_fisher_var(r, alpha)   # Cornish-Fisher
            cvar_hist = float(r[r <= var_hist].mean()) if (r <= var_hist).any() else var_hist
            return var_hist, var_norm, cvar_hist

        pv, pvn, pc = _metrics(port_r)

        # Histogram
        mn, mx = float(port_r.min()), float(port_r.max())
        bins   = np.linspace(mn, mx, 21)
        counts, edges = np.histogram(port_r, bins=bins)
        hist_data = [
            {
                "range":  f"{edges[i]:.2%}",
                "count":  int(counts[i]),
                "isTail": edges[i + 1] <= pv,
            }
            for i in range(len(counts))
        ]

        asset_results = []
        for a in req.assets:
            r = _arr(a.returns)
            av, avn, ac = _metrics(r)
            asset_results.append({
                "ticker": a.ticker,
                "var":    safe_float(av),
                "varCF":  safe_float(avn),   # Cornish-Fisher
                "cvar":   safe_float(ac),
                "std":    safe_float(float(r.std(ddof=1))),
                "mean":   safe_float(float(r.mean())),
            })

        return to_native({
            "portfolio": {
                "var":   safe_float(pv),
                "varCF": safe_float(pvn),
                "cvar":  safe_float(pc),
            },
            "assets":    asset_results,
            "histogram": hist_data,
            "confidence": req.confidence,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Drawdown
# ══════════════════════════════════════════════════════════════════════════════

class DrawdownRequest(BaseModel):
    assets:     List[AssetIn]
    rollWindow: int = 12


@router.post("/drawdown")
def run_drawdown(req: DrawdownRequest):
    try:
        port_r  = _port_returns(req.assets)
        port_dd = _arr(drawdown_series(port_r.tolist()))

        def _episodes(returns):
            dd = _arr(drawdown_series(returns.tolist() if isinstance(returns, np.ndarray) else returns))
            eps = []
            start = None
            for i, v in enumerate(dd):
                if v < -1e-6 and start is None:
                    start = i
                elif v >= -1e-6 and start is not None:
                    t_idx = int(np.argmin(dd[start:i+1])) + start
                    eps.append({
                        "start":    start,
                        "trough":   t_idx,
                        "end":      i,
                        "depth":    safe_float(float(dd[t_idx])),
                        "timeToTrough": t_idx - start,   # periods from start to worst point
                        "timeToRecovery": i - t_idx,     # periods from trough back to zero
                        "duration": t_idx - start,       # kept for backward compat
                        "recovery": i - t_idx,           # kept for backward compat
                    })
                    start = None
            if start is not None:
                t_idx = int(np.argmin(dd[start:])) + start
                eps.append({
                    "start":    start,
                    "trough":   t_idx,
                    "end":      None,
                    "depth":    safe_float(float(dd[t_idx])),
                    "timeToTrough": t_idx - start,
                    "timeToRecovery": None,   # episode still open
                    "duration": t_idx - start,
                    "recovery": None,
                })
            return eps

        w = req.rollWindow

        def _window_mdd(returns_window: np.ndarray) -> float:
            """True peak-to-trough max drawdown within a returns window.
            Re-starts the peak from 1.0 so prior-window peaks don't carry over."""
            peak = 1.0; nav = 1.0; mdd = 0.0
            for r in returns_window:
                nav  *= (1 + r)
                peak  = max(peak, nav)
                mdd   = min(mdd, (nav - peak) / peak)
            return mdd

        # Recompute peak-to-trough within each window — using port_dd.min()
        # is wrong because port_dd carries the pre-window peak into each window,
        # making rolling MDD look worse than what actually occurred in that window.
        rolling_mdd = [
            safe_float(_window_mdd(port_r[i-w:i])) if i >= w else None
            for i in range(1, len(port_r) + 1)
        ]

        asset_results = []
        for a in req.assets:
            dd  = _arr(drawdown_series(a.returns))
            eps = _episodes(a.returns)
            asset_results.append({
                "ticker":   a.ticker,
                "series":   [safe_float(v) for v in dd.tolist()],
                "episodes": eps,
                "maxDD":    safe_float(float(np.min(dd))) if len(dd) else 0.0,
            })

        return to_native({
            "portfolio": {
                "series":       [safe_float(v) for v in port_dd.tolist()],
                "rollingMaxDD": rolling_mdd,
                "episodes":     _episodes(port_r),
                "maxDD":        safe_float(float(np.min(port_dd))) if len(port_dd) else 0.0,
            },
            "assets": asset_results,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sharpe / Sortino / Calmar — QuantStats
# ══════════════════════════════════════════════════════════════════════════════

class SharpeSortinoRequest(BaseModel):
    assets:      List[AssetIn]
    rfRate:      float = 4.0
    freqPerYear: int   = 12
    sortTarget:  float = 0.0


@router.post("/sharpe-sortino")
def run_sharpe_sortino(req: SharpeSortinoRequest):
    try:
        freq   = req.freqPerYear
        rf_ann = req.rfRate / 100
        # Exact per-period compounding (vs simple division which understates at high rates)
        rf_per = (1 + rf_ann) ** (1 / freq) - 1
        port_r = _port_returns(req.assets)
        results = []

        def _metrics(r: np.ndarray, ticker: str):
            s  = pd.Series(
                r,
                index=pd.date_range("2000-01-01", periods=len(r), freq="ME")
            )
            ann_r  = float(qs.stats.cagr(s, periods=freq))
            ann_v  = float(r.std(ddof=1)) * math.sqrt(freq)
            sharpe = qs.stats.sharpe(s, rf=rf_per, periods=freq, annualize=True)
            sortino= qs.stats.sortino(s, rf=rf_per, periods=freq, annualize=True)
            calmar = qs.stats.calmar(s)
            omega  = qs.stats.omega(s, rf=rf_per)
            return {
                "ticker":    ticker,
                "annReturn": safe_float(ann_r),
                "annVol":    safe_float(ann_v),
                "sharpe":    safe_float(float(sharpe)),
                "sortino":   safe_float(float(sortino)),
                "calmar":    safe_float(float(calmar)),
                "omega":     safe_float(float(omega)),
            }

        for a in req.assets:
            results.append(_metrics(_arr(a.returns), a.ticker))

        results.append(_metrics(port_r, "Portfolio"))

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Stress Testing — GARCH conditional vol + scenario shocks
# ══════════════════════════════════════════════════════════════════════════════

# Scenario parameter sets calibrated to approximate historical episode severity.
# These are PARAMETRIC PROXIES — not replays of actual historical return series.
# equityShock: approximate peak-to-trough portfolio-level shock
# corrShock:   correlation contagion factor (0 = no change, 1 = perfect correlation)
# volMult:     volatility multiplier vs GARCH conditional vol baseline
SCENARIOS = {
    "gfc":       {"label": "GFC 2008-09 (proxy)",       "equityShock": -0.50, "corrShock": 0.30, "volMult": 3.0},
    "dotcom":    {"label": "Dot-com 2000-02 (proxy)",    "equityShock": -0.45, "corrShock": 0.20, "volMult": 2.5},
    "covid":     {"label": "COVID Mar 2020 (proxy)",     "equityShock": -0.35, "corrShock": 0.25, "volMult": 4.0},
    "rates2022": {"label": "Rate Hike 2022 (proxy)",     "equityShock": -0.20, "corrShock": 0.10, "volMult": 1.5},
    "em_crisis": {"label": "EM Crisis 1997-98 (proxy)",  "equityShock": -0.40, "corrShock": 0.35, "volMult": 2.0},
}


class StressRequest(BaseModel):
    assets:        List[AssetIn]
    scenario:      str   = "gfc"
    customShock:   float = -30.0
    customVolMult: float = 2.0
    freqPerYear:   int   = 12


@router.post("/stress-testing")
def run_stress_testing(req: StressRequest):
    try:
        freq   = getattr(req, "freqPerYear", None) or 12
        params = SCENARIOS.get(req.scenario) or {
            "label":       "Custom",
            "equityShock": req.customShock / 100,
            "corrShock":   0.0,
            "volMult":     req.customVolMult,
        }
        shock      = params["equityShock"]
        vol_mult   = params["volMult"]
        corr_shock = params.get("corrShock", 0.0)   # correlation increase under stress
        port_r     = _port_returns(req.assets)

        # ── GARCH(1,1) conditional vol for portfolio ───────────────────────
        try:
            am  = arch_model(port_r * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off")
            cond_vol_pct    = float(res.conditional_volatility[-1]) / 100
            garch_vol_ann   = cond_vol_pct * math.sqrt(freq)
            garch_params    = {
                "omega": safe_float(float(res.params["omega"])),
                "alpha": safe_float(float(res.params["alpha[1]"])),
                "beta":  safe_float(float(res.params["beta[1]"])),
                "persistence": safe_float(float(res.params["alpha[1]"] + res.params["beta[1]"])),
            }
        except Exception:
            _freq         = freq  # re-bind to ensure name is available in fallback
            cond_vol_pct  = float(port_r.std(ddof=1))
            garch_vol_ann = cond_vol_pct * math.sqrt(_freq)
            garch_params  = {}

        normal_vol   = float(port_r.std(ddof=1)) * math.sqrt(freq)
        stressed_vol = garch_vol_ann * vol_mult   # GARCH base × scenario multiplier

        # ── Stressed covariance: apply corrShock to off-diagonal elements ────
        # Build a stressed correlation matrix: blend sample corr toward all-ones
        # matrix by corrShock weight. This raises pairwise correlations as they
        # do during market stress (correlation contagion).
        #   Σ_stressed = D * [(1-δ)*C + δ*1·1^T] * D  where D = diag(stressed vols)
        # Then re-derive portfolio stressed vol from Σ_stressed.
        n_assets = len(req.assets)
        if n_assets >= 2:
            ret_mat_s = np.column_stack([_arr(a.returns)[-len(port_r):] for a in req.assets])
            sample_cov = np.cov(ret_mat_s.T, ddof=1)
            vols_per   = np.sqrt(np.diag(sample_cov))
            # Per-period stressed vols (vol_mult applied)
            vols_stressed = vols_per * vol_mult
            D_s = np.diag(vols_stressed)
            # Sample correlation matrix
            corr_mat = sample_cov / np.outer(vols_per, vols_per)
            np.fill_diagonal(corr_mat, 1.0)
            # Stressed correlation: blend toward all-ones (perfect correlation)
            ones_mat  = np.ones_like(corr_mat)
            corr_stressed = (1.0 - corr_shock) * corr_mat + corr_shock * ones_mat
            np.fill_diagonal(corr_stressed, 1.0)
            # Stressed covariance
            cov_stressed = D_s @ corr_stressed @ D_s
            ws = np.array([a.weight for a in req.assets]); ws /= ws.sum()
            stressed_vol_port = float(np.sqrt(max(ws @ cov_stressed @ ws, 0))) * math.sqrt(freq)
        else:
            cov_stressed     = None
            stressed_vol_port = stressed_vol  # single asset: no correlation effect

        # Stressed VaR: Cornish-Fisher on vol-scaled + shock-injected returns
        # (distribution shape from history, scale from stressed vol)
        stressed_r = np.concatenate([[shock], port_r * vol_mult])
        alpha = 0.05
        stress_var  = _cornish_fisher_var(stressed_r, alpha)
        normal_var  = _cornish_fisher_var(port_r, alpha)

        asset_impacts = []
        for i, a in enumerate(req.assets):
            r      = _arr(a.returns)
            n_vol  = float(r.std(ddof=1)) * math.sqrt(freq)
            s_vol  = n_vol * vol_mult
            # Weight-scaled direct equity shock impact
            impact = shock * a.weight
            # Per-asset stressed vol from stressed cov diagonal (if available)
            if cov_stressed is not None and i < cov_stressed.shape[0]:
                s_vol_cov = float(np.sqrt(max(cov_stressed[i, i], 0))) * math.sqrt(freq)
            else:
                s_vol_cov = s_vol
            asset_impacts.append({
                "ticker":            a.ticker,
                "normalVol":         safe_float(n_vol),
                "stressedVol":       safe_float(s_vol_cov),
                "impactReturn":      safe_float(impact),
            })

        return to_native({
            "scenario":          req.scenario,
            "scenarioLabel":     params["label"],
            "equityShock":       safe_float(shock),
            "corrShock":         safe_float(corr_shock),
            "volMultiplier":     safe_float(vol_mult),
            "normalVol":         safe_float(normal_vol),
            "stressedVol":       safe_float(stressed_vol),        # GARCH × volMult (single-asset proxy)
            "stressedVolCorr":   safe_float(stressed_vol_port),   # stressed cov including corrShock
            "garchVol":          safe_float(garch_vol_ann),
            "garchParams":       garch_params,
            "normalVaR":         safe_float(normal_var),
            "stressedVaR":       safe_float(stress_var),
            "assets":            asset_impacts,
            "methodNote":        (
                "Stress vol uses GARCH conditional vol × volMult. "
                "stressedVolCorr additionally blends the correlation matrix "
                f"toward 1 by corrShock={corr_shock:.2f} (correlation contagion). "
                "VaR is Cornish-Fisher on vol-scaled historical returns + equity shock injection."
            ),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Risk Attribution — full covariance MCTR/CTR
# ══════════════════════════════════════════════════════════════════════════════

class RiskAttrRequest(BaseModel):
    assets: List[AssetIn]


@router.post("/risk-attribution")
def run_risk_attribution(req: RiskAttrRequest):
    try:
        n       = len(req.assets)
        ws      = np.array([a.weight for a in req.assets])
        ws     /= ws.sum()
        # Trim to common length first to avoid column_stack shape mismatch
        L       = min(len(a.returns) for a in req.assets)
        ret_mat = np.column_stack([_arr(a.returns)[-L:] for a in req.assets])
        cov     = np.cov(ret_mat.T, ddof=1)

        pv      = float(np.sqrt(ws @ cov @ ws))
        mctr    = (cov @ ws) / (pv + 1e-12)
        ctr     = ws * mctr
        pct_rc  = ctr / (ctr.sum() + 1e-12)
        div_ratio = float(np.dot(ws, np.sqrt(np.diag(cov)))) / (pv + 1e-12)

        results = [
            {
                "ticker":          req.assets[i].ticker,
                "weight":          safe_float(float(ws[i])),
                "mctr":            safe_float(float(mctr[i])),
                "ctr":             safe_float(float(ctr[i])),
                "pctRisk":         safe_float(float(pct_rc[i])),
                "riskBudgeted":    safe_float(float(ws[i])),
            }
            for i in range(n)
        ]

        return to_native({
            "assets":              results,
            "portVol":             safe_float(pv),
            "diversificationRatio":safe_float(div_ratio),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Liquidity Risk — Almgren-Chriss + liq-adjusted VaR
# ══════════════════════════════════════════════════════════════════════════════

class LiquidityRequest(BaseModel):
    assets: List[AssetIn]


@router.post("/liquidity")
def run_liquidity_risk(req: LiquidityRequest):
    try:
        results = []
        for a in req.assets:
            r          = _arr(a.returns)
            adv        = a.adv or 0.0
            bid_ask    = a.bidAsk or 0.0
            pos_size   = a.weight * 1_000_000

            # Almgren-Chriss square-root impact
            if adv > 0:
                participation  = pos_size / adv
                days_to_liq    = participation / 0.3
                impact_bps     = 50.0 * math.sqrt(participation)
                liq_cost_bps   = impact_bps + bid_ask * 0.5
            else:
                participation = days_to_liq = None
                liq_cost_bps  = bid_ask * 0.5

            # Liquidity-adjusted VaR: linear cost subtraction from Cornish-Fisher VaR.
            # This is a practical approximation (not a full L-VaR model):
            #   L-VaR ≈ VaR - liquidation_cost
            # where liquidation cost is the Almgren-Chriss market impact estimate.
            # A full L-VaR model would also account for the uncertain liquidation
            # horizon and its interaction with the return distribution.
            normal_var = _cornish_fisher_var(r, 0.05)
            liq_adj    = normal_var - liq_cost_bps / 10000

            # Liquidity score
            if adv == 0:
                score = "Unknown"
            elif adv > 1e7:
                score = "High"
            elif adv > 1e6:
                score = "Medium"
            else:
                score = "Low"

            results.append({
                "ticker":             a.ticker,
                "adv":                safe_float(adv),
                "bidAskBps":          safe_float(bid_ask),
                "participation":      safe_float(participation) if participation else None,
                "daysToLiquidate":    safe_float(days_to_liq)   if days_to_liq  else None,
                "liquidationCostBps": safe_float(liq_cost_bps),
                "normalVaR":          safe_float(normal_var),
                "liquidityAdjVaR":    safe_float(liq_adj),
                "liquidityScore":     score,
            })

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
