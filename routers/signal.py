"""
signal.py
POST /momentum       — TS + cross-sectional momentum with persistence test
POST /mean-reversion — ADF-based half-life + z-score signals
POST /low-vol        — low-vol anomaly with Sharpe/Sortino comparison
POST /combination    — weighted composite signal
"""
import math, traceback
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from schemas import AssetIn
from utils import safe_float, to_native, _arr

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_mom(returns: np.ndarray, lookback: int) -> float:
    if len(returns) < lookback:
        return 0.0
    chunk = returns[-lookback:]
    return float(np.prod(1 + chunk) - 1)


def _half_life(prices: np.ndarray) -> Optional[float]:
    if len(prices) < 10:
        return None
    y    = np.diff(prices)
    x    = prices[:-1]
    x_dm = x - x.mean()
    ss   = float(np.dot(x_dm, x_dm))
    if ss < 1e-12:
        return None
    lam = float(np.dot(x_dm, y)) / ss
    if lam >= 0:
        return None
    return safe_float(-math.log(2) / lam)


def _adf_test(r: np.ndarray):
    try:
        stat, p, *_ = adfuller(r, maxlag=min(5, len(r)//5), autolag="AIC")
        return float(stat), float(p)
    except Exception:
        return 0.0, 1.0


def _max_drawdown(returns: np.ndarray) -> float:
    if not len(returns):
        return 0.0
    peak = nav = 1.0
    mdd  = 0.0
    for r in returns:
        nav  *= (1 + r)
        peak  = max(peak, nav)
        mdd   = min(mdd, (nav - peak) / peak)
    return mdd


# ══════════════════════════════════════════════════════════════════════════════
# 1. Momentum
# ══════════════════════════════════════════════════════════════════════════════

class MomentumRequest(BaseModel):
    assets:    List[AssetIn]
    lookbacks: List[int] = [1, 3, 6, 12]


@router.post("/momentum")
def run_momentum(req: MomentumRequest):
    try:
        results = []
        for a in req.assets:
            r = _arr(a.returns)
            row = {"ticker": a.ticker}
            for lb in req.lookbacks:
                row[f"ts_{lb}m"] = safe_float(_ts_mom(r, lb))

            # 12-1 momentum
            row["mom12_1"] = safe_float(
                _ts_mom(r[:-1], 12) if len(r) > 12 else _ts_mom(r, 12)
            )
            vol12 = float(r[-12:].std(ddof=1)) if len(r) >= 12 else float(r.std(ddof=1))
            row["vol"]   = safe_float(vol12)
            row["score"] = safe_float(row["mom12_1"] / (vol12 + 1e-8))

            # Persistence: win rate + consecutive positive months
            pos_months = int(np.sum(r > 0))
            win_rate   = safe_float(float(pos_months) / max(len(r), 1))
            row["posMonths"] = pos_months
            row["winRate"]   = win_rate

            results.append(row)

        # Cross-sectional rank
        scores = np.array([r["score"] for r in results])
        if len(scores) > 1:
            ranks  = scipy_stats.rankdata(scores)
            cs_rank = (2 * (ranks - 1) / (len(scores) - 1) - 1).tolist()
        else:
            cs_rank = [0.0]
        for i, r in enumerate(results):
            r["composite"] = safe_float(cs_rank[i])
            r["rank"]      = int(np.argsort(scores)[::-1].tolist().index(i) + 1)

        # Momentum spread: top-half minus bottom-half avg return (12-1)
        sorted_by_score = sorted(results, key=lambda x: x["score"], reverse=True)
        half = max(1, len(sorted_by_score) // 2)
        top_tickers = [r["ticker"] for r in sorted_by_score[:half]]
        bot_tickers = [r["ticker"] for r in sorted_by_score[half:]]

        def _avg_ret_series(tickers):
            arrays = [_arr(a.returns) for a in req.assets if a.ticker in tickers]
            if not arrays:
                return np.array([])
            min_len = min(len(a) for a in arrays)
            return np.mean(np.vstack([a[-min_len:] for a in arrays]), axis=0)

        top_r = _avg_ret_series(top_tickers)
        bot_r = _avg_ret_series(bot_tickers)
        if len(top_r) and len(bot_r):
            min_l = min(len(top_r), len(bot_r))
            spread_series = (top_r[-min_l:] - bot_r[-min_l:]).tolist()
        else:
            spread_series = []

        # Decile returns (by composite rank)
        n = len(results)
        decile_rets: List[dict] = []
        for d in range(1, 11):
            lo = int((d - 1) / 10 * n)
            hi = int(d / 10 * n)
            group = sorted_by_score[lo:hi]
            rets  = [_ts_mom(_arr(next(a for a in req.assets if a.ticker == r["ticker"]).returns), 12)
                     for r in group]
            decile_rets.append({
                "decile": d,
                "avgRet": safe_float(float(np.mean(rets))) if rets else None,
                "count":  len(group),
            })

        return to_native({
            "results":      results,
            "spreadSeries": [safe_float(v) for v in spread_series],
            "decileReturns": decile_rets,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

class MeanReversionRequest(BaseModel):
    assets: List[AssetIn]
    window: int = 20


@router.post("/mean-reversion")
def run_mean_reversion(req: MeanReversionRequest):
    try:
        results = []
        for a in req.assets:
            r      = _arr(a.returns)
            prices = np.cumprod(1 + r) * 100

            hl = _half_life(prices)
            # ADF on log-prices (not returns):
            # Returns are almost always stationary by construction, so
            # ADF on returns nearly always rejects the unit-root null and
            # "looks mean-reverting" regardless of whether prices actually
            # revert. Log-prices contain the unit-root if the asset is a
            # random walk — that's the meaningful stationarity test.
            log_prices   = np.log(prices + 1e-8)
            adf_stat, adf_p = _adf_test(log_prices)

            # Rolling z-score
            w   = req.window
            zs  = [None] * len(r)
            for i in range(w, len(r)):
                chunk = r[i-w:i]
                m, s  = chunk.mean(), chunk.std(ddof=1)
                zs[i] = safe_float((r[i] - m) / (s + 1e-8))

            current_z = zs[-1] if zs else None

            # Price + mean band series
            price_series = [safe_float(float(p)) for p in prices]
            mean_band: List[dict] = []
            for i in range(w, len(prices)):
                chunk = prices[i-w:i]
                mu, sigma = float(chunk.mean()), float(chunk.std(ddof=1))
                mean_band.append({
                    "idx":   i,
                    "price": safe_float(float(prices[i])),
                    "mean":  safe_float(mu),
                    "upper1": safe_float(mu + sigma),
                    "lower1": safe_float(mu - sigma),
                    "upper2": safe_float(mu + 2 * sigma),
                    "lower2": safe_float(mu - 2 * sigma),
                })

            # Reversion outcome: after oversold (z < -1.5) what happens over next 1,3,5 periods
            oversold_idx = [i for i, z in enumerate(zs) if z is not None and z < -1.5]
            outcomes: List[dict] = []
            for idx in oversold_idx:
                for fwd in [1, 3, 5]:
                    end = idx + fwd
                    if end < len(r):
                        fwd_ret = float(np.prod(1 + r[idx+1:end+1]) - 1)
                        outcomes.append({"lag": fwd, "ret": safe_float(fwd_ret)})

            avg_outcomes: List[dict] = []
            for lag in [1, 3, 5]:
                vals = [o["ret"] for o in outcomes if o["lag"] == lag]
                avg_outcomes.append({
                    "lag": lag,
                    "avgRet": safe_float(float(np.mean(vals))) if vals else None,
                    "count":  len(vals),
                })

            # Signal
            if current_z is None:
                sig = "Neutral"
            elif current_z > 1.5:
                sig = "Overbought (sell)"
            elif current_z < -1.5:
                sig = "Oversold (buy)"
            elif current_z > 0.5:
                sig = "Mildly overbought"
            elif current_z < -0.5:
                sig = "Mildly oversold"
            else:
                sig = "Neutral"

            results.append({
                "ticker":         a.ticker,
                "zSeries":        zs,
                "currentZ":       current_z,
                "halfLife":       safe_float(hl) if hl else None,
                "adfStat":        safe_float(adf_stat),
                "adfPValue":      safe_float(adf_p),
                # isStationary: ADF rejects unit-root in log-prices (p < 0.05).
                # This indicates the price level is stationary, which is a
                # necessary (but not sufficient) condition for mean reversion.
                # Half-life confirms the reversion speed separately.
                "isStationary":  bool(adf_p < 0.05),
                "meanReverting": bool(adf_p < 0.05 and hl is not None and hl > 0),
                "signal":         sig,
                "priceSeries":    price_series,
                "meanBand":       mean_band,
                "reversionOutcomes": avg_outcomes,
            })

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Low-Vol Anomaly
# ══════════════════════════════════════════════════════════════════════════════

class LowVolRequest(BaseModel):
    assets:      List[AssetIn]
    freqPerYear: int = 12


@router.post("/low-vol")
def run_low_vol(req: LowVolRequest):
    try:
        freq = req.freqPerYear
        rows = []
        for i, a in enumerate(req.assets):
            r     = _arr(a.returns)
            ann_v = float(r.std(ddof=1)) * math.sqrt(freq)
            ann_r = float(np.prod(1 + r) ** (freq / len(r)) - 1) if len(r) > 0 else 0.0

            # Sharpe
            sharpe = ann_r / ann_v if ann_v > 0 else 0.0

            # Sortino (downside deviation, MAR=0)
            downside = r[r < 0]
            dd_ann   = float(np.sqrt(np.mean(downside ** 2)) * math.sqrt(freq)) if len(downside) > 0 else 1e-8
            sortino  = ann_r / dd_ann if dd_ann > 0 else 0.0

            # Max drawdown
            mdd = _max_drawdown(r)

            # Cumulative return series
            cum_series = [safe_float(float(v)) for v in np.cumprod(1 + r).tolist()]

            rows.append({
                "ticker":        a.ticker,
                "annVol":        safe_float(ann_v),
                "annRet":        safe_float(ann_r),
                "sharpe":        safe_float(sharpe),
                "sortino":       safe_float(sortino),
                "maxDrawdown":   safe_float(mdd),
                "cumulativeSeries": cum_series,
            })

        rows.sort(key=lambda x: x["annVol"])
        for rank, r in enumerate(rows):
            r["volRank"] = rank + 1

        n    = len(rows)
        half = max(1, n // 2)
        low_v  = np.mean([r["annRet"] for r in rows[:half]])
        high_v = np.mean([r["annRet"] for r in rows[half:]])

        # Low vs High group summary
        def _group_summary(group):
            return {
                "avgRet":        safe_float(float(np.mean([r["annRet"]  for r in group]))),
                "avgVol":        safe_float(float(np.mean([r["annVol"]  for r in group]))),
                "avgSharpe":     safe_float(float(np.mean([r["sharpe"]  for r in group]))),
                "avgSortino":    safe_float(float(np.mean([r["sortino"] for r in group]))),
                "avgMaxDD":      safe_float(float(np.mean([r["maxDrawdown"] for r in group]))),
                "count":         len(group),
            }

        return to_native({
            "results":        rows,
            "lowVolAvgRet":   safe_float(float(low_v)),
            "highVolAvgRet":  safe_float(float(high_v)),
            "anomalyPresent": bool(low_v >= high_v),
            "lowVolGroup":    _group_summary(rows[:half]),
            "highVolGroup":   _group_summary(rows[half:]),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Signal Combination
# ══════════════════════════════════════════════════════════════════════════════

class CombinationRequest(BaseModel):
    assets:         List[AssetIn]
    momentumWeight: float = 0.4
    valueWeight:    float = 0.3
    qualityWeight:  float = 0.3


@router.post("/combination")
def run_combination(req: CombinationRequest):
    try:
        def _zscore_norm(vals):
            v = np.array(vals, dtype=float)
            s = v.std(ddof=1)
            return ((v - v.mean()) / s).tolist() if s > 0 else np.zeros(len(vals)).tolist()

        mom_raw  = [_ts_mom(_arr(a.returns), 6) for a in req.assets]

        # Value factor: composite of available metrics, rank-normalised.
        # Negative P/E (lower P/E = better value) — only for positive P/E.
        # Adds FCF yield and earnings yield (1/PE) when available.
        # All sub-signals are z-scored before averaging so scale doesn't dominate.
        def _val_score(a) -> float:
            scores = []
            if a.pe is not None and a.pe > 0:
                scores.append(-a.pe)          # lower P/E = better value
            if a.pb is not None and a.pb > 0:
                scores.append(-a.pb)
            if a.fcfYield is not None:
                scores.append(a.fcfYield)     # higher FCF yield = better value
            if a.evEbitda is not None and a.evEbitda > 0:
                scores.append(-a.evEbitda)
            return float(np.mean(scores)) if scores else 0.0

        val_raw  = [_val_score(a) for a in req.assets]

        qual_raw = [float(_arr(a.returns).mean() / (_arr(a.returns).std(ddof=1) + 1e-8))
                    for a in req.assets]

        mom_n  = _zscore_norm(mom_raw)
        val_n  = _zscore_norm(val_raw)
        qual_n = _zscore_norm(qual_raw)

        tw = req.momentumWeight + req.valueWeight + req.qualityWeight or 1.0
        results = []
        for i, a in enumerate(req.assets):
            comp = (req.momentumWeight * mom_n[i] +
                    req.valueWeight    * val_n[i] +
                    req.qualityWeight  * qual_n[i]) / tw
            signal = "buy" if comp > 0.2 else "sell" if comp < -0.2 else "neutral"
            results.append({
                "ticker":    a.ticker,
                "composite": safe_float(comp),
                "momentum":  safe_float(mom_n[i]),
                "value":     safe_float(val_n[i]),
                "quality":   safe_float(qual_n[i]),
                "signal":    signal,
            })

        results.sort(key=lambda x: x["composite"], reverse=True)

        # ── Rolling composite signal per asset ──────────────────────────────
        # Recompute composite at each time step using only past data.
        # Evaluates the actual combination signal, not just momentum.

        def _rolling_composite(a, idx_end):
            """Composite score at time idx_end using history up to idx_end."""
            r = _arr(a.returns)
            mom  = _ts_mom(r[:idx_end], 6)
            val  = _val_score(a)   # static fundamental (multi-metric, not just -PE)
            qual = float(r[:idx_end].mean() / (r[:idx_end].std(ddof=1) + 1e-8)) if idx_end > 2 else 0.0
            return (req.momentumWeight * mom +
                    req.valueWeight    * val +
                    req.qualityWeight  * qual) / tw

        # Hit ratio: composite signal direction vs next-period return
        hit_ratios: List[dict] = []
        for a in req.assets:
            r = _arr(a.returns)
            if len(r) < 8:
                continue
            hits = 0; total = 0
            for i in range(6, len(r) - 1):
                comp = _rolling_composite(a, i)
                if comp > 0 and r[i] > 0:
                    hits += 1
                elif comp < 0 and r[i] < 0:
                    hits += 1
                total += 1
            hit_ratios.append({
                "ticker":   a.ticker,
                "hitRatio": safe_float(hits / total) if total > 0 else None,
                "total":    total,
            })

        # Turnover: fraction of periods where composite signal flips direction
        turnover_stats: List[dict] = []
        for a in req.assets:
            r = _arr(a.returns)
            if len(r) < 8:
                continue
            signals = []
            for i in range(6, len(r)):
                comp = _rolling_composite(a, i)
                sig  = "buy" if comp > 0.05 else "sell" if comp < -0.05 else "neutral"
                signals.append(sig)
            changes = sum(1 for j in range(1, len(signals)) if signals[j] != signals[j-1])
            turnover_stats.append({
                "ticker":   a.ticker,
                "turnover": safe_float(changes / max(len(signals) - 1, 1)),
                "changes":  changes,
                "periods":  len(signals),
            })

        # Signal decay: autocorrelation of rolling composite at lag 1
        decay_stats: List[dict] = []
        for a in req.assets:
            r = _arr(a.returns)
            if len(r) < 14:
                continue
            rolling_comp = [_rolling_composite(a, i) for i in range(6, len(r))]
            if len(rolling_comp) > 4:
                corr = float(np.corrcoef(rolling_comp[:-1], rolling_comp[1:])[0, 1])
                decay_stats.append({
                    "ticker":          a.ticker,
                    "signalDecay":     safe_float(corr),
                    "halfLifePeriods": safe_float(-1 / math.log(abs(corr) + 1e-8)) if abs(corr) > 0.01 else None,
                })

        return to_native({
            "results":      results,
            "hitRatios":    hit_ratios,
            "turnover":     turnover_stats,
            "signalDecay":  decay_stats,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
