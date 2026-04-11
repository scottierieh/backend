"""
backtest.py
POST /run          — strategy backtest with QuantStats performance metrics
POST /walk-forward — walk-forward OOS validation
POST /transaction  — Almgren-Chriss transaction cost model
"""
import math, traceback
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import quantstats as qs

from schemas import AssetIn
from utils import safe_float, to_native, _arr

router = APIRouter()


# ── Strategy functions ────────────────────────────────────────────────────────

def _build_prices(returns: np.ndarray) -> np.ndarray:
    return np.concatenate([[100.0], 100.0 * np.cumprod(1 + returns)])


def _momentum_strategy(prices: np.ndarray, returns: np.ndarray, lookback: int, long_only: bool):
    """Returns (signal_shifted, gross_returns). Signal is the actual position,
    not inferred from returns, so trade counting is exact."""
    n   = len(returns)
    sig = np.zeros(n)
    for i in range(lookback, n):
        past_ret = (prices[i] - prices[i - lookback]) / prices[i - lookback]
        sig[i] = 1.0 if past_ret > 0 else (0.0 if long_only else -1.0)
    # Shift: signal[i] → applied to returns[i+1] (no look-ahead)
    sig_shifted = np.roll(sig, 1); sig_shifted[0] = 0.0
    return sig_shifted, sig_shifted * returns


def _mean_rev_strategy(prices: np.ndarray, returns: np.ndarray, lookback: int, long_only: bool):
    n   = len(returns)
    sig = np.zeros(n)
    for i in range(lookback, n):
        past_ret = (prices[i] - prices[i - lookback]) / prices[i - lookback]
        sig[i] = 1.0 if past_ret < 0 else (0.0 if long_only else -1.0)
    sig_shifted = np.roll(sig, 1); sig_shifted[0] = 0.0
    return sig_shifted, sig_shifted * returns


def _vol_target_strategy(returns: np.ndarray, ann_vol_target: float, lookback: int, freq: int):
    # Scalar at period i uses realized vol from returns[i-lookback:i],
    # applied to returns[i+1] (no look-ahead).
    # Vol-target signal is continuous (scalar, not ±1), capped at 2× leverage.
    n   = len(returns)
    sig = np.ones(n)    # continuous position scalar
    out = np.zeros(n)
    for i in range(lookback, n - 1):
        w  = returns[max(0, i - lookback):i]
        rv = float(w.std(ddof=1)) * math.sqrt(freq)
        sc = min((ann_vol_target / 100) / rv, 2.0) if rv > 0 else 1.0
        sig[i + 1]  = sc
        out[i + 1]  = returns[i + 1] * sc
    return sig, out


def _run_strategy(a: AssetIn, strategy: str, lookback: int, long_only: bool, vol_target: float, freq: int):
    """Returns (signal_series, gross_returns). signal_series is the raw position
    (±1 for directional, continuous scalar for vol-target) before cost deduction."""
    r = _arr(a.returns)
    p = _arr(a.prices) if a.prices else _build_prices(r)
    if strategy == "momentum":
        return _momentum_strategy(p, r, lookback, long_only)
    if strategy == "meanReversion":
        return _mean_rev_strategy(p, r, lookback, long_only)
    if strategy == "volTarget":
        return _vol_target_strategy(r, vol_target, lookback, freq)
    # Buy-and-hold: signal = 1 always
    return np.ones(len(r)), r


def _qs_metrics(r: np.ndarray, freq: int) -> dict:
    """Full QuantStats performance metrics."""
    if len(r) < 4:
        return {}
    dates = pd.date_range("2000-01-01", periods=len(r), freq="ME")
    s     = pd.Series(r, index=dates)
    try:
        return {
            "annReturn":    safe_float(float(qs.stats.cagr(s, periods=freq))),
            "annVol":       safe_float(float(r.std(ddof=1)) * math.sqrt(freq)),
            "sharpe":       safe_float(float(qs.stats.sharpe(s, periods=freq))),
            "sortino":      safe_float(float(qs.stats.sortino(s, periods=freq))),
            "calmar":       safe_float(float(qs.stats.calmar(s))),
            "omega":        safe_float(float(qs.stats.omega(s))),
            "maxDrawdown":  safe_float(float(qs.stats.max_drawdown(s))),
            "winRate":      safe_float(float(qs.stats.win_rate(s))),
            "tailRatio":    safe_float(float(qs.stats.tail_ratio(s))),
            "numTrades":    int(np.sum(np.abs(np.diff(np.sign(r))) > 0)),
            "totalCostsBps":0.0,
            "turnover":     safe_float(float(np.mean(np.abs(np.diff(r))))),
            "curve":        np.cumprod(1 + r).tolist(),
            "bhCurve":      np.cumprod(1 + _arr(a.returns)).tolist()
                            if hasattr(_qs_metrics, "_asset") else [],
        }
    except Exception:
        ann_v = float(r.std(ddof=1)) * math.sqrt(freq)
        ann_r = float(np.prod(1 + r) ** (freq / len(r)) - 1)
        return {
            "annReturn": safe_float(ann_r), "annVol": safe_float(ann_v),
            "sharpe":    safe_float(ann_r / ann_v) if ann_v > 0 else 0.0,
            "maxDrawdown": 0.0, "winRate": 0.5, "numTrades": 0,
            "totalCostsBps": 0.0, "turnover": 0.0,
            "curve": np.cumprod(1 + r).tolist(), "bhCurve": [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Backtest
# ══════════════════════════════════════════════════════════════════════════════

class BacktestRequest(BaseModel):
    assets:      List[AssetIn]
    strategy:    str   = "momentum"
    lookback:    int   = 6
    longOnly:    bool  = True
    volTarget:   float = 15.0
    freqPerYear: int   = 12
    rfRate:      float = 4.0    # annual % risk-free rate for Sharpe
    costBps:     float = 0.0    # one-way transaction cost in basis points


@router.post("/run")
def run_backtest(req: BacktestRequest):
    try:
        freq    = req.freqPerYear
        results = []

        rf_ann  = req.rfRate / 100
        rf_per  = (1 + rf_ann) ** (1 / freq) - 1   # exact per-period compounding
        cost_dec = req.costBps / 10000               # one-way cost as decimal

        for a in req.assets:
            bh_r    = _arr(a.returns)

            # ── Build raw signal + gross returns (no look-ahead) ─────────────
            sig_series, gross_r = _run_strategy(
                a, req.strategy, req.lookback, req.longOnly, req.volTarget, freq
            )

            # ── Transaction cost — Almgren-Chriss if ADV available, else flat ─
            # Trade occurs when the position signal changes.
            # Cost model priority:
            #   1. Almgren-Chriss (sqrt market impact) if asset has adv + bid_ask
            #   2. User-supplied costBps flat rate
            #   3. Zero
            sig_changes = np.diff(sig_series, prepend=0.0)
            trade_mask  = sig_changes != 0

            adv     = getattr(a, "adv",    None) or 0.0
            bid_ask = getattr(a, "bidAsk", None) or 0.0
            # Position size proxy: weight × $1M notional (same as liquidity module)
            pos_size = a.weight * 1_000_000

            if adv > 0:
                # Almgren-Chriss: impact = 50 bps × √(trade_size / ADV)
                participation = abs(pos_size) / adv
                impact_bps    = 50.0 * math.sqrt(participation)
                slippage_bps  = bid_ask * 0.5
                ac_cost_dec   = (impact_bps + slippage_bps) / 10000
                cost_series   = np.where(trade_mask, ac_cost_dec, 0.0)
            else:
                # Fall back to user-supplied flat costBps
                cost_series   = np.where(trade_mask, cost_dec, 0.0)

            strat_r = gross_r - cost_series

            # ── Signal-based numTrades and turnover ───────────────────────────
            num_trades      = int(np.sum(trade_mask))
            turnover        = safe_float(num_trades / max(len(strat_r) - 1, 1))
            total_costs_bps = safe_float(float(np.sum(cost_series) * 10000))

            dates   = pd.date_range("2000-01-01", periods=len(strat_r), freq="ME")
            strat_s = pd.Series(strat_r, index=dates)
            bh_s    = pd.Series(bh_r,    index=dates)

            def _m(s, r, rf_per=rf_per):
                r = np.array(r)
                try:
                    return {
                        "annReturn":   safe_float(float(qs.stats.cagr(s, periods=freq))),
                        "annVol":      safe_float(float(r.std(ddof=1)) * math.sqrt(freq)),
                        "sharpe":      safe_float(float(qs.stats.sharpe(s, rf=rf_per, periods=freq, annualize=True))),
                        "sortino":     safe_float(float(qs.stats.sortino(s, rf=rf_per, periods=freq, annualize=True))),
                        "calmar":      safe_float(float(qs.stats.calmar(s))),
                        "omega":       safe_float(float(qs.stats.omega(s, rf=rf_per))),
                        "maxDrawdown": safe_float(float(qs.stats.max_drawdown(s))),
                        "winRate":     safe_float(float(qs.stats.win_rate(s))),
                        "tailRatio":   safe_float(float(qs.stats.tail_ratio(s))),
                        "curve":       np.cumprod(1 + r).tolist(),
                    }
                except Exception:
                    rv = float(r.std(ddof=1)) * math.sqrt(freq)
                    ra = float(np.prod(1 + r) ** (freq / max(len(r), 1)) - 1)
                    return {
                        "annReturn": safe_float(ra), "annVol": safe_float(rv),
                        "sharpe":    safe_float((ra - rf_ann) / rv) if rv > 0 else None,
                        "sortino": None, "calmar": None, "omega": None,
                        "maxDrawdown": None, "winRate": None, "tailRatio": None,
                        "curve": np.cumprod(1 + r).tolist(),
                    }

            strat_m = _m(strat_s, strat_r)
            strat_m["numTrades"]      = num_trades
            strat_m["turnover"]       = turnover
            strat_m["totalCostsBps"]  = total_costs_bps
            strat_m["bhSharpe"]       = safe_float(float(qs.stats.sharpe(bh_s, rf=rf_per, periods=freq, annualize=True)))
            strat_m["bhAnnReturn"]    = safe_float(float(qs.stats.cagr(bh_s, periods=freq)))
            strat_m["bhCurve"]        = np.cumprod(1 + bh_r).tolist()

            # Drawdown series
            cum     = np.cumprod(1 + strat_r)
            running = np.maximum.accumulate(cum)
            dd_ser  = ((cum - running) / running).tolist()

            # Rolling Sharpe — fixed 12-period window
            rw = 12
            roll_sh = [None] * len(strat_r)
            for i in range(rw, len(strat_r)):
                chunk = strat_r[i-rw:i]
                m_, s_ = float(chunk.mean()), float(chunk.std(ddof=1))
                roll_sh[i] = safe_float((m_ - rf_per) / s_ * math.sqrt(freq)) if s_ > 0 else None

            # Monthly returns matrix — compounded, not first-value
            monthly_rets: list = []
            dates_full = pd.date_range("2000-01-01", periods=len(strat_r), freq="ME")
            for yr in sorted(set(d.year for d in dates_full)):
                row_yr: dict = {"year": yr}
                for mo in range(1, 13):
                    mask = np.array([d.year == yr and d.month == mo for d in dates_full])
                    vals = strat_r[mask]
                    # Compound all returns within the period (handles multi-period edge cases)
                    row_yr[str(mo)] = safe_float(float(np.prod(1 + vals) - 1)) if len(vals) > 0 else None
                monthly_rets.append(row_yr)

            # VaR / CVaR (95%)
            sorted_r  = np.sort(strat_r)
            var95_idx = max(0, int(len(sorted_r) * 0.05) - 1)
            var95  = safe_float(float(sorted_r[var95_idx]))
            cvar95 = safe_float(float(sorted_r[:var95_idx + 1].mean())) if var95_idx >= 0 else var95

            strat_m["drawdownSeries"] = [safe_float(v) for v in dd_ser]
            strat_m["rollingSharpe"]  = roll_sh
            strat_m["monthlyReturns"] = monthly_rets
            strat_m["var95"]          = var95
            strat_m["cvar95"]         = cvar95

            results.append({
                "ticker":   a.ticker,
                "strategy": strat_m,
                "buyHold": {
                    "annReturn":   strat_m["bhAnnReturn"],
                    "annVol":      safe_float(float(bh_r.std(ddof=1)) * math.sqrt(freq)),
                    "sharpe":      strat_m["bhSharpe"],
                    "maxDrawdown": safe_float(float(qs.stats.max_drawdown(bh_s))),
                    "curve":       strat_m["bhCurve"],
                },
            })

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Walk-Forward Validation
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardRequest(BaseModel):
    assets:      List[AssetIn]
    strategy:    str   = "momentum"
    lookback:    int   = 6
    longOnly:    bool  = True
    trainPct:    float = 70.0
    freqPerYear: int   = 12
    rfRate:      float = 4.0
    costBps:     float = 0.0


@router.post("/walk-forward")
def run_walk_forward(req: WalkForwardRequest):
    try:
        freq    = req.freqPerYear
        results = []

        for a in req.assets:
            _, all_strat = _run_strategy(a, req.strategy, req.lookback, req.longOnly, 15.0, freq)
            n         = len(all_strat)
            split     = int(n * req.trainPct / 100)
            is_r      = all_strat[:split]
            oos_r     = all_strat[split:]

            def _m(r):
                if len(r) < 4:
                    return {"annReturn": 0, "annVol": 0, "sharpe": 0, "maxDrawdown": 0, "curve": [1.0]}
                av = float(r.std(ddof=1)) * math.sqrt(freq)
                ar = float(np.prod(1 + r) ** (freq / len(r)) - 1) if len(r) > 0 else 0.0
                dates = pd.date_range("2000-01-01", periods=len(r), freq="ME")
                s     = pd.Series(r, index=dates)
                try:
                    sh = float(qs.stats.sharpe(s, periods=freq))
                    md = float(qs.stats.max_drawdown(s))
                except Exception:
                    sh = ar / av if av > 0 else 0.0
                    md = 0.0
                return {
                    "annReturn":   safe_float(ar),
                    "annVol":      safe_float(av),
                    "sharpe":      safe_float(sh),
                    "maxDrawdown": safe_float(abs(md)),
                    "curve":       np.cumprod(1 + r).tolist(),
                }

            is_m  = _m(is_r)
            oos_m = _m(oos_r)
            overfit = (is_m["sharpe"] - oos_m["sharpe"]) > 0.5 if is_r.any() and oos_r.any() else False

            results.append({
                "ticker":    a.ticker,
                "splitIdx":  split,
                "inSample":  is_m,
                "outSample": oos_m,
                "overfit":   bool(overfit),
            })

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Transaction Cost — Almgren-Chriss square-root model
# ══════════════════════════════════════════════════════════════════════════════

class TradeIn(BaseModel):
    ticker:     str
    tradeValue: float
    adv:        float
    spread:     float      # bid-ask spread bps
    commission: float      # commission bps


class TransactionCostRequest(BaseModel):
    trades:   List[TradeIn]
    urgency:  float = 1.0


@router.post("/transaction")
def run_transaction_cost(req: TransactionCostRequest):
    try:
        results = []
        for t in req.trades:
            participation = abs(t.tradeValue) / t.adv if t.adv > 0 else 0.0
            impact        = 50.0 * math.sqrt(participation) * req.urgency
            slippage      = t.spread * 0.5          # expected crossing cost
            total_bps     = impact + slippage + t.commission
            total_pct     = total_bps / 10000
            cost_usd      = t.tradeValue * total_pct

            results.append({
                "ticker":        t.ticker,
                "marketImpact":  safe_float(impact),
                "slippage":      safe_float(slippage),
                "commission":    safe_float(t.commission),
                "totalBps":      safe_float(total_bps),
                "totalPct":      safe_float(total_pct),
                "costUSD":       safe_float(cost_usd),
                "participation": safe_float(participation) if t.adv > 0 else None,
            })

        total_cost = sum(r["costUSD"] for r in results)

        # Participation vs Cost curve (for chart)
        part_curve = []
        if results:
            avg_adv = sum(t.adv for t in req.trades) / len(req.trades)
            avg_tv  = sum(t.tradeValue for t in req.trades) / len(req.trades)
            for pct in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                tv = avg_adv * pct
                imp = 50.0 * math.sqrt(pct) * req.urgency
                spr = (req.trades[0].spread if req.trades else 10) * 0.5
                com = (req.trades[0].commission if req.trades else 5)
                part_curve.append({
                    "participation": safe_float(pct * 100),
                    "totalBps":      safe_float(imp + spr + com),
                    "impactBps":     safe_float(imp),
                })

        return to_native({"results": results, "totalCostUSD": safe_float(total_cost), "participationCurve": part_curve})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
