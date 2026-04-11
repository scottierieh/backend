"""
screening.py
POST /value-score    — value factor scoring           (value-score-page)
POST /quality-score  — quality factor scoring         (quality-score-page)
POST /momentum       — momentum ranking               (momentum-ranking-page)
POST /multi-factor   — combined multi-factor ranking  (multi-factor-page)
"""

import math
import traceback
from typing import List, Optional, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils import safe_float, to_native, _arr, Returns
from schemas import AssetIn

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# Pure-Python stat helpers
# (avoids silent NameError from missing std/mean imports)
# ══════════════════════════════════════════════════════════════════════════════

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _period_return(returns: Returns, n: int) -> float:
    """Compounded return over the last n periods."""
    window = returns[-n:] if len(returns) >= n else returns
    cum = 1.0
    for r in window:
        cum *= (1 + r)
    return cum - 1


def _annualised_vol(returns: Returns, freq: int) -> float:
    return _std(returns) * math.sqrt(freq) if len(returns) >= 2 else 0.0


def _max_drawdown(returns: Returns) -> float:
    """Maximum peak-to-trough drawdown. Returns a negative number."""
    if not returns:
        return 0.0
    peak = 1.0
    nav  = 1.0
    mdd  = 0.0
    for r in returns:
        nav  *= (1 + r)
        peak  = max(peak, nav)
        dd    = (nav - peak) / peak
        mdd   = min(mdd, dd)
    return mdd


def _assign_deciles(scores: List[Optional[float]]) -> List[Optional[int]]:
    """Assign decile 1 (best score) to 10 (worst score)."""
    indexed = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not indexed:
        return [None] * len(scores)
    indexed.sort(key=lambda x: -x[1])
    n = len(indexed)
    decile_map: Dict[int, int] = {}
    for rank, (i, _) in enumerate(indexed):
        decile_map[i] = min(int(rank / n * 10) + 1, 10)
    return [decile_map.get(i) for i in range(len(scores))]


# ══════════════════════════════════════════════════════════════════════════════
# Cross-sectional rank normalisation & z-score
# ══════════════════════════════════════════════════════════════════════════════

def _cs_rank_normalised(
    vals: List[Optional[float]],
    lower_better: bool = False,
) -> List[Optional[float]]:
    """Rank-normalise to [0, 1]. 1.0 = best, 0.0 = worst.
    lower_better=True  -> lowest value gets 1.0 (e.g. P/E, D/E)
    lower_better=False -> highest value gets 1.0 (e.g. ROE, FCF yield)
    None stays None (imputed as 0.5 at scoring time).
    """
    indexed = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not indexed:
        return [None] * len(vals)
    # Sort so the "best" value comes first (rank 0)
    indexed.sort(key=lambda x: x[1], reverse=not lower_better)
    n_valid = len(indexed)
    rank_map: Dict[int, float] = {}
    for rank, (i, _) in enumerate(indexed):
        # Invert: rank 0 (best) -> 1.0, rank N-1 (worst) -> 0.0
        rank_map[i] = 1.0 - rank / max(n_valid - 1, 1)
    return [rank_map.get(i) for i in range(len(vals))]


def _z_norm(vals: List[float]) -> List[float]:
    """Z-score normalisation."""
    m = _mean(vals)
    s = _std(vals) or 1.0
    return [safe_float((v - m) / s) for v in vals]


def _weighted_rank_score(
    rank_weight_pairs: List[Tuple[Optional[float], float]],
    total_w: float,
) -> float:
    """Weighted average of rank scores. Missing rank imputed as median 0.5."""
    return sum(
        (r if r is not None else 0.5) * w
        for r, w in rank_weight_pairs
    ) / total_w


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _has_fundamentals(a: AssetIn) -> bool:
    return any(
        getattr(a, k) is not None
        for k in [
            "pe", "pb", "evEbitda", "divYield", "fcfYield",
            "roe", "roa", "grossMargin", "operatingMargin", "netMargin",
            "revenueGrowth", "debtEquity", "interestCoverage",
        ]
    )


def _sector_median(assets: List[AssetIn], field: str) -> Dict[str, float]:
    """Return {sector: median_value} for a given field, ignoring None."""
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)
    for a in assets:
        sec = a.sector or "Unknown"
        val = getattr(a, field, None)
        if val is not None:
            buckets[sec].append(val)
    result = {}
    for sec, vals in buckets.items():
        vals.sort()
        n = len(vals)
        mid = n // 2
        result[sec] = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
    return result


def _earnings_stability(returns: Returns) -> float:
    """Stability proxy: inverse of return volatility, scaled to [0, 1]."""
    if not returns:
        return 0.5
    v = _std(returns)
    return safe_float(1.0 / (1.0 + v * 10))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Value Score
# ══════════════════════════════════════════════════════════════════════════════

class ValueScoreRequest(BaseModel):
    assets:                  List[AssetIn]
    peWeight:                float = 0.25
    pbWeight:                float = 0.20
    evEbitdaWeight:          float = 0.20
    divYieldWeight:          float = 0.15
    fcfYieldWeight:          float = 0.20
    excludeNegativeEarnings: bool  = False
    excludeLowLiquidity:     bool  = False
    liquidityThreshold:      float = 1_000_000  # ADV USD


@router.post("/value-score")
def run_value_score(req: ValueScoreRequest):
    try:
        assets = req.assets

        def _keep(a: AssetIn) -> bool:
            if req.excludeNegativeEarnings and a.pe is not None and a.pe <= 0:
                return False
            if req.excludeLowLiquidity and a.adv is not None and a.adv < req.liquidityThreshold:
                return False
            return True

        def _earnings_yield(a: AssetIn) -> Optional[float]:
            if a.pe and a.pe > 0:
                return 1.0 / a.pe
            return None

        # Cross-sectional ranks (full universe)
        pe_ranks  = _cs_rank_normalised([a.pe       for a in assets], lower_better=True)
        pb_ranks  = _cs_rank_normalised([a.pb       for a in assets], lower_better=True)
        ev_ranks  = _cs_rank_normalised([a.evEbitda for a in assets], lower_better=True)
        div_ranks = _cs_rank_normalised([a.divYield for a in assets], lower_better=False)
        fcf_ranks = _cs_rank_normalised([a.fcfYield for a in assets], lower_better=False)

        total_w = (
            req.peWeight + req.pbWeight + req.evEbitdaWeight
            + req.divYieldWeight + req.fcfYieldWeight
        )

        sec_median_pe = _sector_median(assets, "pe")
        sec_median_pb = _sector_median(assets, "pb")

        results = []
        scores_for_decile: List[Optional[float]] = []

        for i, a in enumerate(assets):
            has         = _has_fundamentals(a)
            filtered    = not _keep(a)

            if not has or filtered:
                results.append({
                    "ticker":          a.ticker,
                    "sector":          a.sector,
                    "marketCap":       safe_float(a.marketCap) if a.marketCap else None,
                    "valueScore":      None,
                    "hasFundamentals": has,
                    "filteredOut":     filtered,
                })
                scores_for_decile.append(None)
                continue

            score = _weighted_rank_score(
                [
                    (pe_ranks[i],  req.peWeight),
                    (pb_ranks[i],  req.pbWeight),
                    (ev_ranks[i],  req.evEbitdaWeight),
                    (div_ranks[i], req.divYieldWeight),
                    (fcf_ranks[i], req.fcfYieldWeight),
                ],
                total_w,
            )
            scores_for_decile.append(score)

            sec             = a.sector or "Unknown"
            sec_pe          = sec_median_pe.get(sec)
            sec_pb          = sec_median_pb.get(sec)
            rel_pe_discount = ((sec_pe - a.pe) / sec_pe * 100) if (sec_pe and a.pe) else None
            rel_pb_discount = ((sec_pb - a.pb) / sec_pb * 100) if (sec_pb and a.pb) else None

            results.append({
                "ticker":          a.ticker,
                "sector":          a.sector,
                "marketCap":       safe_float(a.marketCap) if a.marketCap else None,
                "hasFundamentals": True,
                "filteredOut":     False,
                "valueScore":      safe_float(score),
                "earningsYield":   safe_float(_earnings_yield(a)) if _earnings_yield(a) is not None else None,
                "sectorRelative": {
                    "sectorMedianPE":  safe_float(sec_pe)          if sec_pe  is not None else None,
                    "sectorMedianPB":  safe_float(sec_pb)          if sec_pb  is not None else None,
                    "relPEDiscount":   safe_float(rel_pe_discount)  if rel_pe_discount  is not None else None,
                    "relPBDiscount":   safe_float(rel_pb_discount)  if rel_pb_discount  is not None else None,
                },
                "components": {
                    "pe":       {"value": a.pe,       "rank": safe_float(pe_ranks[i])  if pe_ranks[i]  is not None else None},
                    "pb":       {"value": a.pb,       "rank": safe_float(pb_ranks[i])  if pb_ranks[i]  is not None else None},
                    "evEbitda": {"value": a.evEbitda, "rank": safe_float(ev_ranks[i])  if ev_ranks[i]  is not None else None},
                    "divYield": {"value": a.divYield, "rank": safe_float(div_ranks[i]) if div_ranks[i] is not None else None},
                    "fcfYield": {"value": a.fcfYield, "rank": safe_float(fcf_ranks[i]) if fcf_ranks[i] is not None else None},
                },
            })

        deciles = _assign_deciles(scores_for_decile)
        for i, r in enumerate(results):
            r["decile"] = deciles[i]

        results.sort(key=lambda r: (r["valueScore"] is None, -(r["valueScore"] or 0)))

        # Top 10% vs Bottom 10% group summary
        scored   = [r for r in results if r["valueScore"] is not None]
        n_scored = len(scored)
        cut      = max(1, n_scored // 10)
        top_g    = scored[:cut]
        bot_g    = scored[-cut:]

        def _gavg(group, field):
            vals = [g[field] for g in group if g.get(field) is not None]
            return safe_float(_mean(vals)) if vals else None

        def _gavg_comp(group, key):
            vals = [g["components"][key]["value"] for g in group
                    if g.get("components", {}).get(key, {}).get("value") is not None]
            return safe_float(_mean(vals)) if vals else None

        group_summary = {
            "top10pct": {
                "count":        len(top_g),
                "avgScore":     _gavg(top_g, "valueScore"),
                "avgPE":        _gavg_comp(top_g, "pe"),
                "avgPB":        _gavg_comp(top_g, "pb"),
                "avgFCFYield":  _gavg_comp(top_g, "fcfYield"),
                "avgMarketCap": _gavg(top_g, "marketCap"),
            },
            "bottom10pct": {
                "count":        len(bot_g),
                "avgScore":     _gavg(bot_g, "valueScore"),
                "avgPE":        _gavg_comp(bot_g, "pe"),
                "avgPB":        _gavg_comp(bot_g, "pb"),
                "avgFCFYield":  _gavg_comp(bot_g, "fcfYield"),
                "avgMarketCap": _gavg(bot_g, "marketCap"),
            },
        }

        return to_native({"results": results, "groupSummary": group_summary})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Quality Score
# ══════════════════════════════════════════════════════════════════════════════

class QualityScoreRequest(BaseModel):
    assets:          List[AssetIn]
    profitWeight:    float = 0.30   # ROE, ROA, margins
    stabilityWeight: float = 0.25   # earnings stability
    efficiencyWeight:float = 0.20   # revenue growth
    leverageWeight:  float = 0.25   # D/E, interest coverage


@router.post("/quality-score")
def run_quality_score(req: QualityScoreRequest):
    try:
        assets = req.assets

        # Profitability ranks
        roe_r  = _cs_rank_normalised([a.roe            for a in assets], lower_better=False)
        roa_r  = _cs_rank_normalised([a.roa            for a in assets], lower_better=False)
        gm_r   = _cs_rank_normalised([a.grossMargin    for a in assets], lower_better=False)
        om_r   = _cs_rank_normalised([a.operatingMargin for a in assets], lower_better=False)
        nm_r   = _cs_rank_normalised([a.netMargin      for a in assets], lower_better=False)
        # Efficiency ranks
        gr_r   = _cs_rank_normalised([a.revenueGrowth  for a in assets], lower_better=False)
        # Leverage ranks
        de_r   = _cs_rank_normalised([a.debtEquity     for a in assets], lower_better=True)
        ic_r   = _cs_rank_normalised([a.interestCoverage for a in assets], lower_better=False)
        # Stability
        stab_v = [_earnings_stability(a.returns) for a in assets]
        stab_r = _cs_rank_normalised(stab_v, lower_better=False)

        total_w = (
            req.profitWeight + req.stabilityWeight
            + req.efficiencyWeight + req.leverageWeight
        )

        results = []
        scores_for_decile: List[Optional[float]] = []

        for i, a in enumerate(assets):
            has = _has_fundamentals(a)

            profitability = _weighted_rank_score(
                [(roe_r[i], 1), (roa_r[i], 1), (gm_r[i], 1), (om_r[i], 1), (nm_r[i], 1)], 5.0
            )
            efficiency = _weighted_rank_score([(gr_r[i], 1)], 1.0)
            leverage   = _weighted_rank_score([(de_r[i], 1), (ic_r[i], 1)], 2.0)
            stability  = stab_r[i] if stab_r[i] is not None else 0.5

            quality_score = (
                profitability * req.profitWeight +
                stability     * req.stabilityWeight +
                efficiency    * req.efficiencyWeight +
                leverage      * req.leverageWeight
            ) / total_w

            scores_for_decile.append(quality_score)

            results.append({
                "ticker":          a.ticker,
                "sector":          a.sector,
                "marketCap":       safe_float(a.marketCap) if a.marketCap else None,
                "hasFundamentals": has,
                "qualityScore":    safe_float(quality_score),
                "subScores": {
                    "profitability": safe_float(profitability),
                    "stability":     safe_float(stability),
                    "efficiency":    safe_float(efficiency),
                    "leverage":      safe_float(leverage),
                },
                "components": {
                    "roe":             {"value": a.roe,             "rank": safe_float(roe_r[i])  if roe_r[i]  is not None else None},
                    "roa":             {"value": a.roa,             "rank": safe_float(roa_r[i])  if roa_r[i]  is not None else None},
                    "grossMargin":     {"value": a.grossMargin,     "rank": safe_float(gm_r[i])   if gm_r[i]   is not None else None},
                    "operatingMargin": {"value": a.operatingMargin, "rank": safe_float(om_r[i])   if om_r[i]   is not None else None},
                    "netMargin":       {"value": a.netMargin,       "rank": safe_float(nm_r[i])   if nm_r[i]   is not None else None},
                    "revenueGrowth":   {"value": a.revenueGrowth,   "rank": safe_float(gr_r[i])   if gr_r[i]   is not None else None},
                    "debtEquity":      {"value": a.debtEquity,      "rank": safe_float(de_r[i])   if de_r[i]   is not None else None},
                    "interestCoverage":{"value": a.interestCoverage,"rank": safe_float(ic_r[i])   if ic_r[i]   is not None else None},
                    "stability":       {"value": safe_float(stab_v[i]), "rank": safe_float(stab_r[i]) if stab_r[i] is not None else None},
                },
            })

        deciles = _assign_deciles(scores_for_decile)
        for i, r in enumerate(results):
            r["decile"] = deciles[i]

        results.sort(key=lambda r: -(r["qualityScore"] or 0))

        # High vs Low quality group summary
        cut   = max(1, len(results) // 10)
        top_g = results[:cut]
        bot_g = results[-cut:]

        def _gc(group, key):
            vals = [g["components"][key]["value"] for g in group
                    if g.get("components", {}).get(key, {}).get("value") is not None]
            return safe_float(_mean(vals)) if vals else None

        group_summary = {
            "high": {
                "count":        len(top_g),
                "avgROE":       _gc(top_g, "roe"),
                "avgMargin":    _gc(top_g, "operatingMargin"),
                "avgDebtRatio": _gc(top_g, "debtEquity"),
            },
            "low": {
                "count":        len(bot_g),
                "avgROE":       _gc(bot_g, "roe"),
                "avgMargin":    _gc(bot_g, "operatingMargin"),
                "avgDebtRatio": _gc(bot_g, "debtEquity"),
            },
        }

        return to_native({"results": results, "groupSummary": group_summary})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Momentum Ranking
# ══════════════════════════════════════════════════════════════════════════════

class MomentumRankingRequest(BaseModel):
    assets:      List[AssetIn]
    freqPerYear: int = 12


def _momentum_12_1(returns: Returns) -> float:
    """
    12-1 month momentum: compounded return over months [-13, -1],
    skipping the most recent month to avoid short-term reversal.
    Falls back to full history if fewer than 13 observations.
    """
    if len(returns) < 13:
        return _period_return(returns, len(returns))
    window = returns[-13:-1]
    cum = 1.0
    for r in window:
        cum *= (1 + r)
    return cum - 1


def _trend_strength(returns: Returns) -> float:
    """Fraction of positive periods in the 12-1 window. 0 = all down, 1 = all up."""
    window = returns[-13:-1] if len(returns) >= 13 else returns
    if not window:
        return 0.5
    return sum(1 for r in window if r > 0) / len(window)


def _risk_adj_momentum(mom: float, vol: float) -> float:
    return safe_float(mom / vol) if vol > 0 else 0.0


@router.post("/momentum")
def run_momentum_ranking(req: MomentumRankingRequest):
    try:
        assets = req.assets
        freq   = req.freqPerYear

        moms      = [_momentum_12_1(a.returns)       for a in assets]
        vols      = [_annualised_vol(a.returns, freq) for a in assets]
        scores    = [_risk_adj_momentum(m, v)         for m, v in zip(moms, vols)]
        drawdowns = [_max_drawdown(a.returns)         for a in assets]
        strengths = [_trend_strength(a.returns)       for a in assets]

        results = []
        for i, a in enumerate(assets):
            results.append({
                "ticker":        a.ticker,
                "sector":        a.sector,
                "marketCap":     safe_float(a.marketCap) if a.marketCap else None,
                "ret1m":         safe_float(_period_return(a.returns, 1)),
                "ret3m":         safe_float(_period_return(a.returns, 3)),
                "ret6m":         safe_float(_period_return(a.returns, 6)),
                "ret12m":        safe_float(_period_return(a.returns, 12)),
                "mom12_1":       safe_float(moms[i]),
                "vol":           safe_float(vols[i]),
                "maxDrawdown":   safe_float(drawdowns[i]),
                "score":         safe_float(scores[i]),
                "trendStrength": safe_float(strengths[i]),
            })

        results.sort(key=lambda r: -(r["score"] or 0))
        for rank, r in enumerate(results):
            r["rank"] = rank + 1

        deciles = _assign_deciles([r["score"] for r in results])
        for i, r in enumerate(results):
            r["decile"] = deciles[i]

        # Momentum regime summary: top vs bottom quartile
        cut   = max(1, len(results) // 4)
        top_q = results[:cut]
        bot_q = results[-cut:]

        def _gavg(group, field):
            vals = [g[field] for g in group if g.get(field) is not None]
            return safe_float(_mean(vals)) if vals else None

        regime_summary = {
            "topQuartile": {
                "count":       len(top_q),
                "avgReturn":   _gavg(top_q, "ret12m"),
                "avgVol":      _gavg(top_q, "vol"),
                "avgDrawdown": _gavg(top_q, "maxDrawdown"),
                "avgTrend":    _gavg(top_q, "trendStrength"),
                "winRate":     safe_float(
                    sum(1 for r in top_q if (r["ret12m"] or 0) > 0) / len(top_q)
                ) if top_q else None,
            },
            "bottomQuartile": {
                "count":       len(bot_q),
                "avgReturn":   _gavg(bot_q, "ret12m"),
                "avgVol":      _gavg(bot_q, "vol"),
                "avgDrawdown": _gavg(bot_q, "maxDrawdown"),
                "avgTrend":    _gavg(bot_q, "trendStrength"),
                "winRate":     safe_float(
                    sum(1 for r in bot_q if (r["ret12m"] or 0) > 0) / len(bot_q)
                ) if bot_q else None,
            },
        }

        return to_native({"results": results, "regimeSummary": regime_summary})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multi-Factor Ranking
# ══════════════════════════════════════════════════════════════════════════════

class MultifactorRequest(BaseModel):
    assets:         List[AssetIn]
    valueWeight:    float = 0.33
    qualityWeight:  float = 0.33
    momentumWeight: float = 0.34
    freqPerYear:    int   = 12


_STYLE_THRESHOLD = 0.3  # z-score above which a style label is assigned


def _value_z(assets: List[AssetIn]) -> List[float]:
    """
    Rank-normalised composite value score, then z-scored.
    Consistent with /value-score endpoint logic.
    """
    pe_r  = _cs_rank_normalised([a.pe       for a in assets], lower_better=True)
    pb_r  = _cs_rank_normalised([a.pb       for a in assets], lower_better=True)
    ev_r  = _cs_rank_normalised([a.evEbitda for a in assets], lower_better=True)
    div_r = _cs_rank_normalised([a.divYield for a in assets], lower_better=False)
    fcf_r = _cs_rank_normalised([a.fcfYield for a in assets], lower_better=False)
    raw = [
        _weighted_rank_score(
            [(pe_r[i], 0.25), (pb_r[i], 0.20), (ev_r[i], 0.20),
             (div_r[i], 0.15), (fcf_r[i], 0.20)], 1.0
        )
        for i in range(len(assets))
    ]
    return _z_norm(raw)


def _quality_z(assets: List[AssetIn]) -> List[float]:
    """
    Multi-metric quality composite, then z-scored.
    Consistent with /quality-score endpoint logic.
    """
    roe_r  = _cs_rank_normalised([a.roe             for a in assets], lower_better=False)
    roa_r  = _cs_rank_normalised([a.roa             for a in assets], lower_better=False)
    gm_r   = _cs_rank_normalised([a.grossMargin     for a in assets], lower_better=False)
    om_r   = _cs_rank_normalised([a.operatingMargin for a in assets], lower_better=False)
    de_r   = _cs_rank_normalised([a.debtEquity      for a in assets], lower_better=True)
    ic_r   = _cs_rank_normalised([a.interestCoverage for a in assets], lower_better=False)
    gr_r   = _cs_rank_normalised([a.revenueGrowth   for a in assets], lower_better=False)
    stab_v = [_earnings_stability(a.returns) for a in assets]
    stab_r = _cs_rank_normalised(stab_v, lower_better=False)

    raw = []
    for i in range(len(assets)):
        profit     = _weighted_rank_score([(roe_r[i], 1), (roa_r[i], 1), (gm_r[i], 1), (om_r[i], 1)], 4.0)
        leverage   = _weighted_rank_score([(de_r[i], 1), (ic_r[i], 1)], 2.0)
        efficiency = _weighted_rank_score([(gr_r[i], 1)], 1.0)
        stability  = stab_r[i] if stab_r[i] is not None else 0.5
        score      = profit * 0.30 + stability * 0.25 + efficiency * 0.20 + leverage * 0.25
        raw.append(score)
    return _z_norm(raw)


def _momentum_z(assets: List[AssetIn], freq: int) -> List[float]:
    moms   = [_momentum_12_1(a.returns)       for a in assets]
    vols   = [_annualised_vol(a.returns, freq) for a in assets]
    scores = [_risk_adj_momentum(m, v)         for m, v in zip(moms, vols)]
    return _z_norm(scores)


@router.post("/multi-factor")
def run_multi_factor(req: MultifactorRequest):
    try:
        assets = req.assets
        freq   = req.freqPerYear

        val_z  = _value_z(assets)
        qual_z = _quality_z(assets)
        mom_z  = _momentum_z(assets, freq)

        total_w = req.valueWeight + req.qualityWeight + req.momentumWeight

        results = []
        for i, a in enumerate(assets):
            composite = (
                val_z[i]  * req.valueWeight +
                qual_z[i] * req.qualityWeight +
                mom_z[i]  * req.momentumWeight
            ) / total_w

            styles: List[str] = []
            if val_z[i]  > _STYLE_THRESHOLD: styles.append("Value")
            if qual_z[i] > _STYLE_THRESHOLD: styles.append("Quality")
            if mom_z[i]  > _STYLE_THRESHOLD: styles.append("Momentum")

            conflict = val_z[i] > 0.5 and mom_z[i] < -0.5
            signal   = "buy" if composite > 0.5 else "sell" if composite < -0.5 else "hold"

            results.append({
                "ticker":    a.ticker,
                "sector":    a.sector,
                "marketCap": safe_float(a.marketCap) if a.marketCap else None,
                "composite": safe_float(composite),
                "value":     safe_float(val_z[i]),
                "quality":   safe_float(qual_z[i]),
                "momentum":  safe_float(mom_z[i]),
                "styles":    styles,
                "conflict":  conflict,
                "signal":    signal,
            })

        results.sort(key=lambda r: -(r["composite"] or 0))
        for rank, r in enumerate(results):
            r["rank"] = rank + 1

        deciles = _assign_deciles([r["composite"] for r in results])
        for i, r in enumerate(results):
            r["decile"] = deciles[i]

        # Factor exposure summary table
        def _factor_stats(z_vals: List[float], label: str) -> Dict[str, Any]:
            cut      = max(1, len(z_vals) // 5)
            sorted_z = sorted(z_vals, reverse=True)
            return {
                "factor":                  label,
                "avgScore":                safe_float(_mean(z_vals)),
                "stdDev":                  safe_float(_std(z_vals)),
                "topPortfolioExposure":    safe_float(_mean(sorted_z[:cut])),
                "bottomPortfolioExposure": safe_float(_mean(sorted_z[-cut:])),
            }

        factor_exposure = [
            _factor_stats(val_z,  "Value"),
            _factor_stats(qual_z, "Quality"),
            _factor_stats(mom_z,  "Momentum"),
        ]

        # Sector allocation of top 20 names
        top20 = results[:20]
        sec_counts: Dict[str, int] = {}
        for r in top20:
            sec = r["sector"] or "Unknown"
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
        sector_allocation = [
            {"sector": s, "count": c, "pct": safe_float(c / len(top20) * 100)}
            for s, c in sorted(sec_counts.items(), key=lambda x: -x[1])
        ]

        # Portfolio characteristics: top vs bottom decile
        top_d    = [r for r in results if r.get("decile") == 1]
        bottom_d = [r for r in results if r.get("decile") == 10]

        def _pc(group, field):
            vals = [g[field] for g in group if g.get(field) is not None]
            return safe_float(_mean(vals)) if vals else None

        portfolio_chars = {
            "topDecile": {
                "count":        len(top_d),
                "avgComposite": _pc(top_d, "composite"),
                "avgMarketCap": _pc(top_d, "marketCap"),
                "buySignals":   sum(1 for r in top_d if r["signal"] == "buy"),
            },
            "bottomDecile": {
                "count":        len(bottom_d),
                "avgComposite": _pc(bottom_d, "composite"),
                "avgMarketCap": _pc(bottom_d, "marketCap"),
                "sellSignals":  sum(1 for r in bottom_d if r["signal"] == "sell"),
            },
        }

        return to_native({
            "results":          results,
            "factorExposure":   factor_exposure,
            "sectorAllocation": sector_allocation,
            "portfolioChars":   portfolio_chars,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
