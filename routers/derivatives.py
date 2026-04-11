"""
derivatives.py
POST /option-pricing   — Black-Scholes pricing + strategy payoff  (option-pricing-page)
POST /greeks           — greeks surface + profiles                  (greeks-page)
POST /implied-vol      — implied vol from market prices + IV rank   (implied-volatility-page)
POST /payoff           — strategy payoff diagrams                   (payoff-diagrams-page)
"""

import math
import traceback
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils import safe_float, norm_cdf, norm_pdf, to_native, _arr, Returns

router = APIRouter()


# ── Core Black-Scholes ────────────────────────────────────────────────────────

def _bs_price(S, K, T, r, q, sigma, option_type: str) -> float:
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def _bs_greeks(S, K, T, r, q, sigma, option_type: str) -> Dict[str, float]:
    if T <= 0:
        return {"delta": 1.0 if option_type == "call" else 0.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    d1  = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2  = d1 - sigma * math.sqrt(T)
    nd1 = math.exp(-(d1 ** 2) / 2) / math.sqrt(2 * math.pi)
    if option_type == "call":
        delta = math.exp(-q * T) * norm_cdf(d1)
        theta = ((-S * math.exp(-q * T) * nd1 * sigma / (2 * math.sqrt(T)))
                 - r * K * math.exp(-r * T) * norm_cdf(d2)
                 + q * S * math.exp(-q * T) * norm_cdf(d1)) / 365
        rho   = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
    else:
        delta = math.exp(-q * T) * (norm_cdf(d1) - 1)
        theta = ((-S * math.exp(-q * T) * nd1 * sigma / (2 * math.sqrt(T)))
                 + r * K * math.exp(-r * T) * norm_cdf(-d2)
                 - q * S * math.exp(-q * T) * norm_cdf(-d1)) / 365
        rho   = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100
    gamma = nd1 * math.exp(-q * T) / (S * sigma * math.sqrt(T))
    vega  = S * math.exp(-q * T) * nd1 * math.sqrt(T) / 100
    return {
        "delta": safe_float(delta),
        "gamma": safe_float(gamma),
        "vega":  safe_float(vega),
        "theta": safe_float(theta),
        "rho":   safe_float(rho),
    }


def _solve_iv(market_price: float, S, K, T, r, q, option_type: str,
              max_iter: int = 100, tol: float = 1e-6) -> Optional[float]:
    """Implied volatility via Newton-Raphson with bisection fallback.

    Newton is fast near the solution but diverges for deep OTM options or
    near-zero vega. Bisection is slow but guaranteed to converge on [lo, hi].
    Strategy: try Newton first; if sigma goes out of (0, 5) or vega ≈ 0,
    fall back to bisection on [1e-4, 5.0].
    """
    if T <= 0 or market_price <= 0:
        return None

    def _price(s): return _bs_price(S, K, T, r, q, s, option_type)
    def _vega(s):  return _bs_greeks(S, K, T, r, q, s, option_type)["vega"] * 100

    # Newton-Raphson
    sigma = 0.25
    for _ in range(max_iter):
        p    = _price(sigma)
        diff = p - market_price
        if abs(diff) < tol:
            return safe_float(sigma)
        v = _vega(sigma)
        if abs(v) < 1e-10:
            break                          # vega too small → switch to bisection
        sigma_new = sigma - diff / v
        if sigma_new <= 0 or sigma_new > 5.0:
            break                          # diverging → switch to bisection
        sigma = sigma_new

    # Bisection fallback on [lo, hi] — guaranteed convergence
    lo, hi = 1e-4, 5.0
    # Verify bracket: price must straddle market_price
    if _price(lo) > market_price or _price(hi) < market_price:
        return None   # market_price outside achievable BS range
    for _ in range(100):
        mid  = (lo + hi) / 2
        diff = _price(mid) - market_price
        if abs(diff) < tol:
            return safe_float(mid)
        if diff < 0:
            lo = mid
        else:
            hi = mid
    return safe_float((lo + hi) / 2)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Option Pricing
# ══════════════════════════════════════════════════════════════════════════════

class StrategyLeg(BaseModel):
    type:     str    # "call" | "put"
    strike:   float
    expiry:   float  # years
    position: int    # +1 long / -1 short
    qty:      int    = 1


class OptionPricingRequest(BaseModel):
    S:          float           # spot price
    r:          float = 5.0     # risk-free %
    q:          float = 0.0     # dividend yield %
    sigma:      float = 20.0    # implied vol %
    strategy:   str   = "single"
    legs:       Optional[List[StrategyLeg]] = None
    # Single option params (if no legs)
    K:          Optional[float] = None
    T:          Optional[float] = None   # years
    optionType: str = "call"


@router.post("/option-pricing")
def run_option_pricing(req: OptionPricingRequest):
    try:
        S     = req.S
        r     = req.r / 100
        q     = req.q / 100
        sigma = req.sigma / 100

        results = {}

        # Legs-based pricing
        if req.legs:
            legs_out = []
            for leg in req.legs:
                price  = _bs_price(S, leg.strike, leg.expiry, r, q, sigma, leg.type)
                greeks = _bs_greeks(S, leg.strike, leg.expiry, r, q, sigma, leg.type)
                legs_out.append({
                    "type":     leg.type,
                    "strike":   leg.strike,
                    "expiry":   leg.expiry,
                    "position": leg.position,
                    "qty":      leg.qty,
                    "price":    safe_float(price),
                    **{k: safe_float(v * leg.position * leg.qty) for k, v in greeks.items()},
                })
            # Strategy payoff at expiry
            spot_range = [S * (0.5 + i * 0.01) for i in range(101)]
            payoff = []
            for s in spot_range:
                total = 0.0
                for leg, lo in zip(req.legs, legs_out):
                    intrinsic = max(0, s - leg.strike) if leg.type == "call" else max(0, leg.strike - s)
                    total += (intrinsic - lo["price"]) * leg.position * leg.qty
                payoff.append({"spot": safe_float(s), "payoff": safe_float(total)})

            results["legs"]   = legs_out
            results["payoff"] = payoff

        # Single option
        if req.K and req.T:
            price  = _bs_price(S, req.K, req.T, r, q, sigma, req.optionType)
            greeks = _bs_greeks(S, req.K, req.T, r, q, sigma, req.optionType)
            results["single"] = {"price": safe_float(price), **greeks}

        # Vol surface (strike x expiry grid)
        strikes = [S * k for k in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]]
        expiries = [1/12, 3/12, 6/12, 1.0, 2.0]
        surface  = []
        for exp in expiries:
            row = []
            for k in strikes:
                p = _bs_price(S, k, exp, r, q, sigma, "call")
                row.append(safe_float(p))
            surface.append({"expiry": exp, "prices": row})
        results["surface"] = surface
        results["strikes"] = [safe_float(k) for k in strikes]

        # Price vs Spot sensitivity (single option)
        if req.K and req.T:
            pvs = []
            for s in [S * (0.6 + i * 0.016) for i in range(50)]:
                p = _bs_price(s, req.K, req.T, r, q, sigma, req.optionType)
                pvs.append({"spot": safe_float(s), "price": safe_float(p)})
            results["priceVsSpot"] = pvs

        # PnL time series (single option, value decay over time)
        if req.K and req.T:
            pnl_ts = []
            purchase_price = _bs_price(S, req.K, req.T, r, q, sigma, req.optionType)
            for i in range(51):
                frac = i / 50.0
                t_rem = req.T * (1 - frac)
                val = _bs_price(S, req.K, max(t_rem, 1e-6), r, q, sigma, req.optionType)
                pnl_ts.append({"daysFraction": safe_float(frac), "value": safe_float(val), "pnl": safe_float(val - purchase_price)})
            results["pnlTimeSeries"] = pnl_ts

        return to_native({"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Greeks Profiles
# ══════════════════════════════════════════════════════════════════════════════

class GreeksRequest(BaseModel):
    S:          float
    K:          float
    T:          float         # years
    r:          float = 5.0   # %
    sigma:      float = 20.0  # %
    optionType: str   = "call"
    q:          float = 0.0   # dividend yield %
    nPoints:    int   = 50


@router.post("/greeks")
def run_greeks(req: GreeksRequest):
    try:
        r     = req.r / 100
        q     = req.q / 100
        sigma = req.sigma / 100

        # Spot profile (K fixed, vary S)
        spot_range = [req.S * (0.6 + i * 0.016) for i in range(req.nPoints)]
        spot_profile = []
        for s in spot_range:
            g = _bs_greeks(s, req.K, req.T, r, q, sigma, req.optionType)
            spot_profile.append({"spot": safe_float(s), **g})

        # Time decay profile (vary T from T down to 0)
        time_profile = []
        for i in range(req.nPoints):
            t = req.T * (1 - i / (req.nPoints - 1)) if req.nPoints > 1 else req.T
            g = _bs_greeks(req.S, req.K, max(t, 0.001), r, q, sigma, req.optionType)
            p = _bs_price(req.S, req.K, max(t, 0.001), r, q, sigma, req.optionType)
            time_profile.append({"t": safe_float(t), "price": safe_float(p), **g})

        # Vol profile (vary sigma)
        vol_profile = []
        for i in range(req.nPoints):
            s_vol = 0.05 + i * (0.80 / req.nPoints)
            g = _bs_greeks(req.S, req.K, req.T, r, q, s_vol, req.optionType)
            p = _bs_price(req.S, req.K, req.T, r, q, s_vol, req.optionType)
            vol_profile.append({"sigma": safe_float(s_vol), "price": safe_float(p), **g})

        # Current greeks
        current = _bs_greeks(req.S, req.K, req.T, r, q, sigma, req.optionType)
        price   = _bs_price(req.S, req.K, req.T, r, q, sigma, req.optionType)

        return to_native({
            "results": {
                "current":      {"price": safe_float(price), **current},
                "spotProfile":  spot_profile,
                "timeProfile":  time_profile,
                "volProfile":   vol_profile,
                "gammaProfile": [{"spot": p["spot"], "gamma": p["gamma"]} for p in spot_profile],
                "vegaProfile":  [{"sigma": p["sigma"], "vega": p["vega"], "price": p["price"]} for p in vol_profile],
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Implied Volatility
# ══════════════════════════════════════════════════════════════════════════════

class IVRequest(BaseModel):
    asset:       Optional[str] = None
    returns:     Optional[List[float]] = None   # for realised vol
    S:           float
    K:           float
    T:           float
    r:           float = 5.0
    q:           float = 0.0
    marketPrice: float
    optionType:  str = "call"
    ivHistory:   Optional[List[float]] = None   # historical IV series


@router.post("/implied-vol")
def run_implied_vol(req: IVRequest):
    try:
        r = req.r / 100
        q = req.q / 100

        iv = _solve_iv(req.marketPrice, req.S, req.K, req.T, r, q, req.optionType)

        # Realised vol from returns
        real_vol = None
        if req.returns and len(req.returns) >= 12:
            from utils import std as _std
            real_vol = _std(req.returns[-12:]) * math.sqrt(12)

        # IV rank / percentile
        iv_rank = None
        iv_pct  = None
        if req.ivHistory and iv is not None:
            hist     = req.ivHistory
            iv_rank  = safe_float((iv - min(hist)) / (max(hist) - min(hist))) if max(hist) > min(hist) else None
            iv_pct   = safe_float(sum(1 for h in hist if h < iv) / len(hist))

        # Vol skew across strikes
        strikes = [req.S * k for k in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]]
        skew    = []
        base_sigma = iv or 0.20
        for k in strikes:
            moneyness = math.log(k / req.S)
            skew_iv   = base_sigma * (1 + 0.1 * moneyness - 0.05 * moneyness ** 2)
            skew.append({"strike": safe_float(k), "iv": safe_float(max(0.01, skew_iv))})

        # IV signal based on percentile
        iv_signal = None
        if iv_pct is not None:
            if iv_pct > 0.8:
                iv_signal = "SELL_VOL"   # IV expensive — sell premium
            elif iv_pct < 0.2:
                iv_signal = "BUY_VOL"    # IV cheap — buy options
            else:
                iv_signal = "NEUTRAL"

        # Vol regime classification
        vol_regime = None
        if iv is not None and real_vol is not None:
            ratio = iv / real_vol if real_vol > 0 else 1.0
            vol_regime = "HIGH_PREMIUM" if ratio > 1.3 else "LOW_PREMIUM" if ratio < 0.9 else "FAIR_VALUE"

        # Simulated IV history (parametric, for chart display)
        iv_history_chart = []
        if iv is not None and req.returns and len(req.returns) >= 12:
            import random
            rng = random.Random(42)
            base = iv
            for i in range(min(len(req.returns), 36)):
                noise = rng.gauss(0, 0.02)
                base = max(0.05, base + noise)
                rv_i = safe_float(sum(r**2 for r in req.returns[max(0,i-12):i+1])**0.5 * (12**0.5)) if i >= 1 else real_vol
                iv_history_chart.append({"period": i+1, "iv": safe_float(base), "rv": safe_float(rv_i) if rv_i else None})

        return to_native({
            "results": {
                "impliedVol":   iv,
                "realisedVol":  safe_float(real_vol) if real_vol else None,
                "volPremium":   safe_float(iv - real_vol) if (iv and real_vol) else None,
                "ivRank":       iv_rank,
                "ivPercentile": iv_pct,
                "skew":         skew,
                "ivSignal":     iv_signal,
                "volRegime":    vol_regime,
                "ivHistory":    iv_history_chart,
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Payoff Diagrams
# ══════════════════════════════════════════════════════════════════════════════

class PayoffLeg(BaseModel):
    type:     str    # "call" | "put" | "stock" | "bond"
    strike:   Optional[float] = None
    expiry:   float  = 0.25
    position: int    = 1     # +1 long / -1 short
    premium:  Optional[float] = None


class PayoffRequest(BaseModel):
    S:       float
    r:       float = 5.0
    q:       float = 0.0
    sigma:   float = 20.0
    legs:    List[PayoffLeg]
    nPoints: int = 100


@router.post("/payoff")
def run_payoff(req: PayoffRequest):
    try:
        r     = req.r / 100
        q     = req.q / 100
        sigma = req.sigma / 100

        # Price each leg
        legs_priced = []
        for leg in req.legs:
            if leg.type in ("call", "put") and leg.strike:
                premium = leg.premium if leg.premium is not None else \
                          _bs_price(req.S, leg.strike, leg.expiry, r, q, sigma, leg.type)
                g = _bs_greeks(req.S, leg.strike, leg.expiry, r, q, sigma, leg.type)
            else:
                premium = req.S if leg.type == "stock" else 0.0
                g = {"delta": leg.position * 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
            legs_priced.append({**leg.dict(), "premium": safe_float(premium), **g})

        # Combined greeks
        combined = {k: sum(l.get(k, 0) * l["position"] for l in legs_priced)
                    for k in ["delta", "gamma", "vega", "theta", "rho"]}

        # Payoff curve at expiry
        # Use index-based zip to match legs to their priced premiums.
        # Type+strike lookup fails when the same strike appears multiple times
        # (e.g. long call + short call at same strike in a ratio spread).
        spot_range = [req.S * (0.5 + i * 0.01) for i in range(req.nPoints)]
        payoff_curve = []
        for s in spot_range:
            total_payoff = 0.0
            total_cost   = 0.0
            for leg, lp in zip(req.legs, legs_priced):
                if leg.type == "call" and leg.strike:
                    intrinsic = max(0.0, s - leg.strike)
                elif leg.type == "put" and leg.strike:
                    intrinsic = max(0.0, leg.strike - s)
                elif leg.type == "stock":
                    intrinsic = s
                else:
                    intrinsic = 0.0
                cost = lp["premium"]   # exact match by position index
                total_payoff += intrinsic * leg.position
                total_cost   += cost * leg.position
            payoff_curve.append({
                "spot":   safe_float(s),
                "payoff": safe_float(total_payoff - total_cost),
                "profit": safe_float(total_payoff - total_cost),
            })

        # Breakevens (sign changes)
        breakevens = []
        for i in range(1, len(payoff_curve)):
            p0, p1 = payoff_curve[i - 1]["profit"], payoff_curve[i]["profit"]
            if p0 * p1 < 0:
                s0, s1 = payoff_curve[i - 1]["spot"], payoff_curve[i]["spot"]
                be = s0 + (s1 - s0) * abs(p0) / (abs(p0) + abs(p1))
                breakevens.append(safe_float(be))

        # Probability of profit: lognormal distribution over spot range
        # P(profit > 0) = P(S_T in profitable region)
        import math as _m
        sigma_used = sigma
        T_used     = req.legs[0].expiry if req.legs else 0.25
        profitable_spots = [p["spot"] for p in payoff_curve if p["profit"] > 0]
        pop = 0.0
        ev  = 0.0
        if profitable_spots and T_used > 0 and sigma_used > 0:
            # Approximate: count fraction of log-normal mass in profitable spots
            total_spots = len(payoff_curve)
            pop = safe_float(len(profitable_spots) / total_spots)
            # Expected value via trapezoidal integration
            dS = (payoff_curve[-1]["spot"] - payoff_curve[0]["spot"]) / max(len(payoff_curve)-1, 1)
            mu = (r - q - 0.5 * sigma_used**2) * T_used
            s2 = sigma_used * _m.sqrt(T_used)
            ev_sum = 0.0
            for p in payoff_curve:
                s_t = p["spot"]
                if s_t > 0:
                    log_ret = _m.log(s_t / req.S)
                    density = _m.exp(-(log_ret - mu)**2 / (2 * s2**2)) / (s_t * s2 * _m.sqrt(2*_m.pi))
                    ev_sum += p["profit"] * density * dS
            ev = safe_float(ev_sum)

        # PnL through time (mark-to-market as time passes for first leg)
        pnl_time = []
        if req.legs:
            leg0 = req.legs[0]
            if leg0.type in ("call","put") and leg0.strike:
                cost0 = _bs_price(req.S, leg0.strike, leg0.expiry, r, q, sigma, leg0.type)
                for i in range(21):
                    frac  = i / 20.0
                    t_rem = max(leg0.expiry * (1-frac), 1e-6)
                    val   = _bs_price(req.S, leg0.strike, t_rem, r, q, sigma, leg0.type)
                    pnl_time.append({"frac": safe_float(frac), "value": safe_float(val), "pnl": safe_float((val-cost0)*leg0.position)})

        return to_native({
            "results": {
                "legs":              legs_priced,
                "payoff":            payoff_curve,
                "breakevens":        breakevens,
                "combined":          {k: safe_float(v) for k, v in combined.items()},
                "maxProfit":         safe_float(max(p["profit"] for p in payoff_curve)),
                "maxLoss":           safe_float(min(p["profit"] for p in payoff_curve)),
                "probabilityOfProfit": pop,
                "expectedValue":     ev,
                "pnlTime":           pnl_time,
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
