"""
portfolio.py
POST /optimization       — EfficientFrontier: max_sharpe / min_vol / max_utility
POST /efficient-frontier — CLA exact frontier + Monte Carlo cloud
POST /risk-parity        — HRP (PyPortfolioOpt) + HERC (Riskfolio) + Inv-Vol
"""
import math, traceback
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.covariance import LedoitWolf

from pypfopt import EfficientFrontier, HRPOpt, CLA
from pypfopt.exceptions import OptimizationError
import riskfolio as rp

from schemas import AssetIn
from utils import safe_float, to_native, _arr

router = APIRouter()


# ── Helpers: correct scale for returns data ───────────────────────────────────

def _returns_df(assets: List[AssetIn]) -> pd.DataFrame:
    min_len = min(len(a.returns) for a in assets)
    return pd.DataFrame({a.ticker: _arr(a.returns)[-min_len:] for a in assets})


def _mu_cov(df: pd.DataFrame, freq: int, shrinkage: bool = True):
    """
    Returns annualised expected-return Series and annualised covariance DataFrame.
    Uses sklearn LedoitWolf for robust shrinkage covariance (direct returns input).
    """
    # Expected returns — per-period arithmetic mean, annualised via geometric compounding.
    # Note: df.mean() is the arithmetic (not geometric) per-period mean.
    # We then compound as (1 + mu_arithmetic)^freq - 1, which is a standard
    # approximation used by PyPortfolioOpt. For long horizons the geometric
    # mean would be lower by ~σ²/2 (variance drag), but arithmetic mean
    # is the correct input for single-period mean-variance optimisation.
    mu_per = df.mean().values
    mu_ann = (1 + mu_per) ** freq - 1
    mu_s   = pd.Series(mu_ann, index=df.columns)

    # Covariance — per-period sample/LW → annualised by ×freq
    if shrinkage and len(df) > len(df.columns) + 5:
        lw     = LedoitWolf().fit(df.values)
        cov_per = lw.covariance_
    else:
        cov_per = np.cov(df.values.T, ddof=1)

    cov_ann = np.atleast_2d(cov_per) * freq
    cov_df  = pd.DataFrame(cov_ann, index=df.columns, columns=df.columns)

    return mu_s, cov_df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Portfolio Optimisation
# ══════════════════════════════════════════════════════════════════════════════

class OptimisationRequest(BaseModel):
    assets:          List[AssetIn]
    objective:       str   = "sharpe"
    rfRate:          float = 4.0
    freqPerYear:     int   = 12
    maxWeight:       float = 100.0
    minWeight:       float = 0.0
    turnoverPenalty: float = 0.0
    currentWeights:  Optional[List[float]] = None
    targetRisk:      Optional[float] = None
    riskAversion:    float = 1.0


@router.post("/optimization")
def run_optimisation(req: OptimisationRequest):
    try:
        df      = _returns_df(req.assets)
        mu, S   = _mu_cov(df, req.freqPerYear)
        rf      = req.rfRate / 100
        max_w   = req.maxWeight / 100
        min_w   = req.minWeight / 100

        def _make_ef():
            return EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))

        # Turnover penalty setup
        # When turnoverPenalty > 0 and currentWeights provided, we add an L1
        # penalty term to penalise deviation from the prior weights.
        #
        # IMPORTANT: pypfopt does not support adding custom objectives to
        # max_sharpe (fractional programming). When a turnover penalty is
        # active, we therefore use max_quadratic_utility as the base objective
        # for "sharpe" requests — this solves μ - λσ² instead of (μ-rf)/σ.
        # The result is labelled "sharpe_with_turnover" in the response so the
        # caller can distinguish. Without a penalty, max_sharpe is used as
        # normal and the objectives are correctly named.
        #
        # Penalty scale: turnoverPenalty is in "return units". A value of 0.01
        # means 1% penalty per unit of one-way turnover.  We scale by 0.01 so
        # that a UI slider of 1–10 maps to 0.01–0.10 (matching annualised
        # return magnitude and avoiding over-penalising).
        import cvxpy as cp
        prev_w = np.array(req.currentWeights) if (
            req.currentWeights and len(req.currentWeights) == len(req.assets)
            and req.turnoverPenalty > 0
        ) else None
        lam_to = req.turnoverPenalty * 0.01   # scale to return units

        effective_objective = req.objective
        ef = _make_ef()
        try:
            if prev_w is not None:
                # Add L1 turnover penalty (works with any convex base objective)
                ef.add_objective(lambda w: lam_to * cp.norm1(w - prev_w))
                if req.objective == "minVol":
                    ef.min_volatility()
                elif req.objective == "efficientRisk" and req.targetRisk:
                    ef.efficient_risk(target_volatility=req.targetRisk / 100)
                else:
                    # max_sharpe is not compatible with add_objective in pypfopt.
                    # Fall back to mean-variance utility (equivalent at rf=0).
                    # Risk aversion derived from Sharpe tangency: λ ≈ (μ-rf)/σ²
                    # Use riskAversion if explicitly set, else derive from data.
                    if req.objective == "sharpe":
                        # Implied risk aversion from tangency portfolio:
                        # λ* = 1 / (μ_excess^T Σ^{-1} μ_excess)
                        # Using excess returns (μ - rf) so rf is properly accounted for.
                        # Clamped to [0.5, 20] to avoid degenerate solutions.
                        excess_mu   = mu.values - rf
                        S_inv       = np.linalg.pinv(S.values)
                        port_vol_sq = float(excess_mu @ S_inv @ excess_mu)
                        implied_ra  = max(0.5, min(20.0, 1.0 / (port_vol_sq + 1e-8)))
                        ef.max_quadratic_utility(risk_aversion=implied_ra)
                        effective_objective = "sharpe_approx_with_turnover"
                    else:
                        ef.max_quadratic_utility(risk_aversion=req.riskAversion)
            elif req.objective == "sharpe":
                ef.max_sharpe(risk_free_rate=rf)
            elif req.objective == "minVol":
                ef.min_volatility()
            elif req.objective == "maxUtility":
                ef.max_quadratic_utility(risk_aversion=req.riskAversion)
            elif req.objective == "efficientRisk" and req.targetRisk:
                ef.efficient_risk(target_volatility=req.targetRisk / 100)
            else:
                ef.max_sharpe(risk_free_rate=rf)
        except (OptimizationError, Exception):
            ef = _make_ef()
            ef.min_volatility()
            effective_objective = "minVol_fallback"

        w_clean          = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=rf)

        # Risk contributions
        w_arr    = np.array([w_clean.get(a.ticker, 0) for a in req.assets])
        S_arr    = S.values
        pv       = float(np.sqrt(max(w_arr @ S_arr @ w_arr, 0)))
        mctr     = (S_arr @ w_arr) / (pv + 1e-12)
        ctr      = w_arr * mctr
        pct_risk = ctr / (ctr.sum() + 1e-12)

        # Turnover vs current weights
        if req.currentWeights and len(req.currentWeights) == len(req.assets):
            turnover = float(np.sum(np.abs(w_arr - np.array(req.currentWeights))) / 2)
        else:
            turnover = None

        # Return contribution per asset (weight × expected return)
        ret_contribs = w_arr * mu.values
        ret_contribs_pct = ret_contribs / (abs(ret_contribs.sum()) + 1e-12)

        # Concentration metrics
        herfindahl = float(np.sum(w_arr ** 2))
        top3_idx   = np.argsort(w_arr)[::-1][:3]
        top3_weight = float(w_arr[top3_idx].sum())

        return to_native({
            "weights": [
                {
                    "ticker":             a.ticker,
                    "weight":             safe_float(w_clean.get(a.ticker, 0)),
                    "riskContribution":   safe_float(float(pct_risk[i])),
                    "returnContribution": safe_float(float(ret_contribs[i])),
                    "retContribPct":      safe_float(float(ret_contribs_pct[i])),
                }
                for i, a in enumerate(req.assets)
            ],
            "annReturn":    safe_float(ret),
            "annVol":       safe_float(vol),
            "sharpe":       safe_float(sharpe),
            "objective":    effective_objective,
            "turnover":     safe_float(turnover) if turnover is not None else None,
            "herfindahl":   safe_float(herfindahl),
            "top3Weight":   safe_float(top3_weight),
            "constraints": {
                "maxWeight":       safe_float(req.maxWeight),
                "minWeight":       safe_float(req.minWeight),
                "targetRisk":      safe_float(req.targetRisk) if req.targetRisk else None,
                "riskAversion":    safe_float(req.riskAversion),
                "turnoverPenalty": safe_float(req.turnoverPenalty),
            },
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Efficient Frontier
# ══════════════════════════════════════════════════════════════════════════════

class EfficientFrontierRequest(BaseModel):
    assets:      List[AssetIn]
    nPoints:     int   = 200
    rfRate:      float = 4.0
    freqPerYear: int   = 12
    maxWeight:   float = 100.0


@router.post("/efficient-frontier")
def run_efficient_frontier(req: EfficientFrontierRequest):
    try:
        df       = _returns_df(req.assets)
        mu, S    = _mu_cov(df, req.freqPerYear)
        rf       = req.rfRate / 100
        max_w    = req.maxWeight / 100
        n        = len(req.assets)
        S_arr    = S.values
        mu_arr   = mu.values

        # Monte Carlo cloud — feasibility-aware sampling.
        # For loose constraints (max_w >= 1/n * 2) pure rejection is fast.
        # For tight constraints we use an iterative projection: sample Dirichlet,
        # then iteratively clip-and-renormalise until all weights are feasible.
        # This converges in O(n) iterations and always stays in the feasible set.
        def _feasible_weight(rng, n, max_w):
            w = rng.dirichlet(np.ones(n))
            if max_w >= 1.0:
                return w
            # Dykstra-like iterative projection onto box [0, max_w] + simplex
            for _ in range(30):   # converges in <10 iterations in practice
                w = np.clip(w, 0.0, max_w)
                s = w.sum()
                if s < 1e-8:
                    return None
                w = w / s
                if np.all(w <= max_w + 1e-9):
                    return w
            return w if np.all(w <= max_w + 1e-9) else None

        rng = np.random.default_rng(42)
        mc  = []
        attempts = 0
        max_attempts = req.nPoints * 10   # projection rarely fails
        while len(mc) < req.nPoints and attempts < max_attempts:
            attempts += 1
            w = _feasible_weight(rng, n, max_w)
            if w is None:
                continue
            r_  = float(mu_arr @ w)
            v_  = float(np.sqrt(max(w @ S_arr @ w, 0)))
            mc.append({
                "ret":    safe_float(r_),
                "vol":    safe_float(v_),
                "sharpe": safe_float((r_ - rf) / v_) if v_ > 0 else 0.0,
                "weights":[safe_float(wi) for wi in w],
            })

        # Exact optimal portfolios
        def _opt(objective):
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
            try:
                if objective == "sharpe":
                    ef.max_sharpe(risk_free_rate=rf)
                else:
                    ef.min_volatility()
            except Exception:
                ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
                ef.min_volatility()
            w_d = ef.clean_weights()
            r_, v_, s_ = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
            return w_d, r_, v_, s_

        w_ms, r_ms, v_ms, s_ms = _opt("sharpe")
        w_mv, r_mv, v_mv, _    = _opt("minvol")

        def _w_list(w_dict):
            return [safe_float(w_dict.get(a.ticker, 0)) for a in req.assets]

        return to_native({
            "points":   mc,
            "maxSharpe":{
                "ret": safe_float(r_ms), "vol": safe_float(v_ms),
                "sharpe": safe_float(s_ms), "weights": _w_list(w_ms),
            },
            "minVol":{
                "ret": safe_float(r_mv), "vol": safe_float(v_mv),
                "sharpe": safe_float((r_mv - rf) / v_mv) if v_mv > 0 else 0.0,
                "weights": _w_list(w_mv),
            },
            "assets":[
                {
                    "ticker": a.ticker,
                    "ret":    safe_float(float(mu[a.ticker])),
                    "vol":    safe_float(float(np.sqrt(S.loc[a.ticker, a.ticker]))),
                    "sharpe": safe_float(float(
                        (mu[a.ticker] - rf) / np.sqrt(S.loc[a.ticker, a.ticker])
                    ) if S.loc[a.ticker, a.ticker] > 0 else 0.0),
                }
                for a in req.assets
            ],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Risk Parity — HRP + HERC + Inv-Vol
# ══════════════════════════════════════════════════════════════════════════════

class RiskParityRequest(BaseModel):
    assets:      List[AssetIn]
    freqPerYear: int   = 12
    rfRate:      float = 4.0


@router.post("/risk-parity")
def run_risk_parity(req: RiskParityRequest):
    try:
        df   = _returns_df(req.assets)
        freq = req.freqPerYear
        rf   = req.rfRate / 100
        # HRP — PyPortfolioOpt
        hrp       = HRPOpt(returns=df)
        w_hrp     = hrp.optimize()
        hrp_ret, hrp_vol, _ = hrp.portfolio_performance(verbose=False)

        # HERC — Riskfolio-Lib
        try:
            hc_port   = rp.HCPortfolio(returns=df)
            w_herc_df = hc_port.optimization(model="HERC", rm="CVaR", rf=0.0, linkage="ward")
            w_herc    = {t: float(w_herc_df.loc[t, "weights"]) for t in df.columns}
        except Exception:
            w_herc = dict(w_hrp)

        # Inv-Vol
        vols  = df.std(ddof=1).values
        inv   = 1.0 / (vols + 1e-12)
        inv_w = (inv / inv.sum()).tolist()

        # Risk contributions (HRP weights, annualised cov)
        # Reuse mu/S_ann already computed above instead of calling _mu_cov again
        mu_ann, S_ann = _mu_cov(df, freq)
        S_arr    = S_ann.values
        w_arr    = np.array([w_hrp.get(t, 0) for t in df.columns])
        pv       = float(np.sqrt(max(w_arr @ S_arr @ w_arr, 0)))
        mctr     = (S_arr @ w_arr) / (pv + 1e-12)
        ctr      = w_arr * mctr
        pct_rc   = (ctr / (ctr.sum() + 1e-12)).tolist()

        # Per-method performance: annualised return + vol
        def _port_perf(weights_list):
            w = np.array(weights_list, dtype=float)
            # Reuse mu_ann and S_arr computed above — no need to call _mu_cov again
            r_ = float(mu_ann.values @ w)
            v_ = float(np.sqrt(max(w @ S_arr @ w, 0)))
            s_ = (r_ - rf) / v_ if v_ > 0 else 0.0
            return safe_float(r_), safe_float(v_), safe_float(s_)

        # HERC performance
        herc_w = [w_herc.get(t, 0) for t in df.columns]
        herc_ret, herc_vol, herc_sharpe = _port_perf(herc_w)

        # HRP Sharpe
        hrp_sharpe = (hrp_ret - rf) / hrp_vol if hrp_vol > 0 else 0.0

        # Inv-Vol performance
        iv_ret, iv_vol, iv_sharpe = _port_perf(inv_w)

        # Equal-weight
        n_eq = len(req.assets)
        eq_w = [1.0 / n_eq] * n_eq
        eq_ret, eq_vol, eq_sharpe = _port_perf(eq_w)

        # HERC risk contributions
        herc_w_arr = np.array(herc_w, dtype=float)
        herc_pv  = float(np.sqrt(max(herc_w_arr @ S_arr @ herc_w_arr, 0)))
        herc_mctr = (S_arr @ herc_w_arr) / (herc_pv + 1e-12)
        herc_ctr  = herc_w_arr * herc_mctr
        herc_pct_rc = (herc_ctr / (herc_ctr.sum() + 1e-12)).tolist()

        # HRP linkage order (for hierarchical tree display)
        try:
            link_order = hrp.ordered_tickers if hasattr(hrp, "ordered_tickers") else list(df.columns)
        except Exception:
            link_order = list(df.columns)

        return to_native({
            "assets": [
                {
                    "ticker":               a.ticker,
                    "weight":               safe_float(w_hrp.get(a.ticker, 0)),
                    "invVolWeight":         safe_float(float(inv_w[i])),
                    "hercWeight":           safe_float(w_herc.get(a.ticker, 0)),
                    "equalWeight":          safe_float(1.0 / n_eq),
                    "riskContribution":     safe_float(float(pct_rc[i])),
                    "hercRiskContribution": safe_float(float(herc_pct_rc[i])),
                }
                for i, a in enumerate(req.assets)
            ],
            "riskParity": {
                "weights":      [safe_float(w_hrp.get(a.ticker, 0)) for a in req.assets],
                "riskContribs": [safe_float(v) for v in pct_rc],
                "annReturn":    safe_float(hrp_ret),
                "annVol":       safe_float(hrp_vol),
                "sharpe":       safe_float(hrp_sharpe),
            },
            "herc": {
                "weights":      [safe_float(v) for v in herc_w],
                "riskContribs": [safe_float(v) for v in herc_pct_rc],
                "annReturn":    herc_ret,
                "annVol":       herc_vol,
                "sharpe":       herc_sharpe,
            },
            "invVol": {
                "weights":      [safe_float(v) for v in inv_w],
                "riskContribs": [safe_float(v) for v in pct_rc],
                "annReturn":    iv_ret,
                "annVol":       iv_vol,
                "sharpe":       iv_sharpe,
            },
            "equalWeight": {
                "weights":   [safe_float(v) for v in eq_w],
                "annReturn": eq_ret,
                "annVol":    eq_vol,
                "sharpe":    eq_sharpe,
            },
            "linkOrder": link_order,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
