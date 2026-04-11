"""
statistics.py
POST /correlation     — Ledoit-Wolf shrinkage corr + rolling + cluster
POST /distribution    — Jarque-Bera, skewness, kurtosis, normality
POST /rolling-stats   — rolling sharpe/vol/return (vectorised)
POST /autocorrelation — statsmodels ACF/PACF/Ljung-Box/ADF
"""
import math, traceback
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pypfopt import risk_models

from schemas import AssetIn
from utils import safe_float, to_native, _arr

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Correlation / Covariance
# ══════════════════════════════════════════════════════════════════════════════

class CorrelationRequest(BaseModel):
    assets:      List[AssetIn]
    metric:      str           = "correlation"
    method:      str           = "pearson"
    rollWindow:  Optional[int] = None
    rollPair:    Optional[List[int]] = None
    lookback:    Optional[int] = None


def _detect_clusters(corr_mat: np.ndarray, threshold: float = 0.6) -> List[int]:
    n      = corr_mat.shape[0]
    labels = [-1] * n
    label  = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = label
        for j in range(i + 1, n):
            if labels[j] == -1 and corr_mat[i, j] > threshold:
                labels[j] = label
        label += 1
    return labels


def _hierarchical_linkage(corr_mat: np.ndarray, tickers: List[str]) -> List[dict]:
    from scipy.cluster.hierarchy import linkage
    dist = 1.0 - np.abs(corr_mat)
    np.fill_diagonal(dist, 0.0)
    n = len(tickers)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist[i, j])
    Z = linkage(condensed, method="average")
    steps = []
    labels = list(tickers)
    for step_i, row in enumerate(Z):
        a_idx, b_idx, dist_val = int(row[0]), int(row[1]), float(row[2])
        label_a = labels[a_idx] if a_idx < len(tickers) else f"Cluster{a_idx}"
        label_b = labels[b_idx] if b_idx < len(tickers) else f"Cluster{b_idx}"
        new_label = f"[{label_a}+{label_b}]"
        labels.append(new_label)
        steps.append({
            "step":     step_i + 1,
            "a":        label_a,
            "b":        label_b,
            "distance": safe_float(dist_val),
            "merged":   new_label,
        })
    return steps


@router.post("/correlation")
def run_correlation(req: CorrelationRequest):
    try:
        lb = req.lookback
        def _trim(returns):
            return returns[-lb:] if lb and lb < len(returns) else returns

        min_len = min(len(_trim(a.returns)) for a in req.assets)
        df = pd.DataFrame({
            a.ticker: _arr(_trim(a.returns))[-min_len:]
            for a in req.assets
        })
        n_assets = len(req.assets)
        tickers  = [a.ticker for a in req.assets]

        if req.metric == "covariance":
            S   = risk_models.CovarianceShrinkage(df).ledoit_wolf()
            mat = S.values
        elif req.method == "spearman":
            result = scipy_stats.spearmanr(df.values)
            if n_assets == 2:
                v = float(result.statistic)
                mat = np.array([[1.0, v], [v, 1.0]])
            else:
                mat = np.array(result.statistic)
            np.fill_diagonal(mat, 1.0)
        else:
            # Ledoit-Wolf shrinkage covariance → converted to correlation.
            # This is "shrinkage correlation", not plain sample Pearson.
            # It reduces estimation error in small samples at the cost of
            # shrinking extreme correlations toward zero.
            S   = risk_models.CovarianceShrinkage(df).ledoit_wolf()
            d   = np.sqrt(np.diag(S.values))
            mat = S.values / np.outer(d, d)
            np.fill_diagonal(mat, 1.0)

        clusters = _detect_clusters(mat) if req.metric != "covariance" else [0] * n_assets

        pair_corrs = [
            float(mat[i, j])
            for i in range(n_assets)
            for j in range(i + 1, n_assets)
        ]

        corr_hist = []
        if pair_corrs:
            counts, edges = np.histogram(pair_corrs, bins=min(20, max(5, len(pair_corrs))))
            corr_hist = [
                {
                    "range":    f"{edges[k]:.2f}",
                    "midpoint": safe_float(float((edges[k] + edges[k+1]) / 2)),
                    "count":    int(counts[k]),
                }
                for k in range(len(counts))
            ]

        dendro = []
        if n_assets >= 3:
            try:
                dendro = _hierarchical_linkage(mat, tickers)
            except Exception:
                dendro = []

        rolling = []
        if req.rollWindow and req.rollPair and len(req.rollPair) == 2:
            pi, pj = req.rollPair
            ri = _arr(_trim(req.assets[pi].returns))
            rj = _arr(_trim(req.assets[pj].returns))
            L  = min(len(ri), len(rj))
            ri, rj = ri[-L:], rj[-L:]
            w  = req.rollWindow
            for k in range(w, L):
                wi, wj = ri[k-w:k], rj[k-w:k]
                if req.method == "spearman":
                    val = float(scipy_stats.spearmanr(wi, wj).statistic)
                elif req.metric == "covariance":
                    val = float(np.cov(wi, wj, ddof=1)[0, 1])
                else:
                    val = float(np.corrcoef(wi, wj)[0, 1])
                rolling.append(safe_float(val))

        return to_native({
            "tickers":    tickers,
            "matrix":     mat.tolist(),
            "clusters":   clusters,
            "rolling":    rolling,
            "corrHist":   corr_hist,
            "dendrogram": dendro,
            "pairCorrs":  [safe_float(v) for v in pair_corrs],
            "avgCorr":    safe_float(float(np.mean(pair_corrs))) if pair_corrs else None,
            "corrMethod":  "spearman" if req.method == "spearman"
                            else "covariance" if req.metric == "covariance"
                            else "shrinkage_pearson",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Return Distribution — enriched with tail risk, regime, gain-loss analysis
# ══════════════════════════════════════════════════════════════════════════════

class DistributionRequest(BaseModel):
    assets:            List[AssetIn]
    nBins:             int   = 30
    logScale:          bool  = False
    extremeThresholds: List[float] = [-0.03, -0.05]   # extreme loss cutoffs


def _asset_distribution_stats(r: np.ndarray, extreme_thresholds: List[float]) -> dict:
    """Full distribution stats for a single return series."""
    n = len(r)
    if n < 4:
        return {}

    mu  = float(r.mean())
    sig = float(r.std(ddof=1))

    # Normality
    jb_stat, jb_p = scipy_stats.jarque_bera(r)
    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r))   # excess

    # Gains vs losses
    gains  = r[r > 0]
    losses = r[r < 0]
    avg_gain  = float(gains.mean())  if len(gains)  > 0 else 0.0
    avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.0
    gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else float("inf")

    # Downside deviation (MAR = 0)
    downside = r[r < 0]
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0

    # Left tail probabilities at -1%, -2%, -5%
    tail_probs = {
        "lt_1pct":  safe_float(float(np.mean(r < -0.01))),
        "lt_2pct":  safe_float(float(np.mean(r < -0.02))),
        "lt_5pct":  safe_float(float(np.mean(r < -0.05))),
    }

    # Extreme loss frequency
    extreme_freq = []
    for thr in extreme_thresholds:
        count = int(np.sum(r < thr))
        extreme_freq.append({
            "threshold": safe_float(thr),
            "count":     count,
            "pct":       safe_float(float(count / n)),
        })

    # Tail ratio: 95th pct / abs(5th pct)
    p95 = float(np.percentile(r, 95))
    p5  = float(np.percentile(r, 5))
    tail_ratio = safe_float(abs(p95 / p5)) if p5 != 0 else None

    # Positive vs negative period split
    pos_r = r[r >= 0]
    neg_r = r[r <  0]
    regime_split = {
        "positive": {
            "count": int(len(pos_r)),
            "pct":   safe_float(float(len(pos_r) / n)),
            "mean":  safe_float(float(pos_r.mean())) if len(pos_r) > 0 else None,
            "vol":   safe_float(float(pos_r.std(ddof=1))) if len(pos_r) > 1 else None,
        },
        "negative": {
            "count": int(len(neg_r)),
            "pct":   safe_float(float(len(neg_r) / n)),
            "mean":  safe_float(float(neg_r.mean()))  if len(neg_r) > 0 else None,
            "vol":   safe_float(float(neg_r.std(ddof=1))) if len(neg_r) > 1 else None,
        },
    }

    # High-vol vs low-vol regime split (median vol rolling-3 as threshold)
    roll_vol = pd.Series(r).rolling(3).std(ddof=1).dropna().values
    med_vol  = float(np.median(roll_vol)) if len(roll_vol) > 0 else sig
    # Align: label each period by whether its 3-period trailing vol is above median
    # Use pandas for convenience
    s = pd.Series(r)
    rv = s.rolling(3).std(ddof=1)
    high_mask = (rv >= med_vol).values
    low_mask  = (rv <  med_vol).values & ~np.isnan(rv.values)
    r_high = r[high_mask & ~np.isnan(rv.values)]
    r_low  = r[low_mask]
    vol_regime = {
        "highVol": {
            "mean": safe_float(float(r_high.mean())) if len(r_high) > 0 else None,
            "skew": safe_float(float(scipy_stats.skew(r_high))) if len(r_high) > 3 else None,
            "kurt": safe_float(float(scipy_stats.kurtosis(r_high))) if len(r_high) > 3 else None,
            "count": int(len(r_high)),
        },
        "lowVol": {
            "mean": safe_float(float(r_low.mean())) if len(r_low) > 0 else None,
            "skew": safe_float(float(scipy_stats.skew(r_low))) if len(r_low) > 3 else None,
            "kurt": safe_float(float(scipy_stats.kurtosis(r_low))) if len(r_low) > 3 else None,
            "count": int(len(r_low)),
        },
    }

    # Loss-only histogram (30 bins over negative returns)
    loss_hist = []
    if len(losses) >= 4:
        l_counts, l_edges = np.histogram(losses, bins=min(20, len(losses)))
        loss_hist = [
            {"range": f"{l_edges[i]:.3%}", "count": int(l_counts[i])}
            for i in range(len(l_counts))
        ]

    # Per-asset histogram (for overlay)
    h_counts, h_edges = np.histogram(r, bins=30)
    asset_hist = [
        {"range": f"{h_edges[i]:.3%}", "midpoint": safe_float(float((h_edges[i] + h_edges[i+1]) / 2)), "count": int(h_counts[i])}
        for i in range(len(h_counts))
    ]

    # Box-plot stats: min, q1, median, q3, max, outlier fence
    q1, med, q3 = float(np.percentile(r, 25)), float(np.percentile(r, 50)), float(np.percentile(r, 75))
    iqr   = q3 - q1
    fence_lo = q1 - 1.5 * iqr
    fence_hi = q3 + 1.5 * iqr
    outliers = r[(r < fence_lo) | (r > fence_hi)].tolist()
    boxplot = {
        "min":      safe_float(float(r.min())),
        "q1":       safe_float(q1),
        "median":   safe_float(med),
        "q3":       safe_float(q3),
        "max":      safe_float(float(r.max())),
        "iqr":      safe_float(iqr),
        "fenceLo":  safe_float(fence_lo),
        "fenceHi":  safe_float(fence_hi),
        "outliers": [safe_float(v) for v in outliers[:20]],   # cap at 20 outliers
    }

    return {
        "mean":           safe_float(mu),
        "std":            safe_float(sig),
        "skewness":       safe_float(skew),
        "kurtosis":       safe_float(kurt),
        "jb":             safe_float(float(jb_stat)),
        "jbP":            safe_float(float(jb_p)),
        "normal":         bool(jb_p >= 0.05),
        "avgGain":        safe_float(avg_gain),
        "avgLoss":        safe_float(avg_loss),
        "gainLossRatio":  safe_float(gain_loss_ratio) if not math.isinf(gain_loss_ratio) else None,
        "downsideDev":    safe_float(downside_dev),
        "tailProbs":      tail_probs,
        "extremeFreq":    extreme_freq,
        "tailRatio":      tail_ratio,
        "regimeSplit":    regime_split,
        "volRegime":      vol_regime,
        "lossHist":       loss_hist,
        "assetHist":      asset_hist,
        "boxplot":        boxplot,
    }


@router.post("/distribution")
def run_distribution(req: DistributionRequest):
    try:
        min_len = min(len(a.returns) for a in req.assets)
        ws      = np.array([a.weight for a in req.assets], dtype=float)
        ws     /= ws.sum()
        mat     = np.column_stack([_arr(a.returns)[-min_len:] for a in req.assets])
        port_r  = mat @ ws

        if req.logScale:
            port_r = np.log1p(port_r)

        # ── Portfolio histogram ───────────────────────────────────────────────
        counts, edges = np.histogram(port_r, bins=req.nBins)
        p5 = float(np.percentile(port_r, 5))
        hist = [
            {
                "range":  f"{edges[i]:.3%}",
                "midpoint": safe_float(float((edges[i] + edges[i+1]) / 2)),
                "count":  int(counts[i]),
                "isTail": edges[i+1] <= p5,
            }
            for i in range(len(counts))
        ]

        # ── Normal curve overlay (scaled to histogram counts) ─────────────────
        mu_p  = float(port_r.mean())
        sig_p = float(port_r.std(ddof=1))
        xs       = np.linspace(port_r.min(), port_r.max(), 80)
        pdf_vals = scipy_stats.norm.pdf(xs, mu_p, sig_p)
        bin_w    = (port_r.max() - port_r.min()) / req.nBins
        normal_curve = [
            {"x": safe_float(float(x)), "pdf": safe_float(float(p * len(port_r) * bin_w))}
            for x, p in zip(xs, pdf_vals)
        ]

        # ── Q-Q plot data: theoretical vs sample quantiles ────────────────────
        n_qq = min(len(port_r), 100)
        probs = np.linspace(0.01, 0.99, n_qq)
        theoretical = scipy_stats.norm.ppf(probs, loc=mu_p, scale=sig_p)
        sample_q    = np.percentile(port_r, probs * 100)
        qq_data = [
            {"theoretical": safe_float(float(t)), "sample": safe_float(float(s))}
            for t, s in zip(theoretical, sample_q)
        ]
        # Perfect-normal reference line
        qq_line = [
            {"theoretical": safe_float(float(theoretical[0])),  "ref": safe_float(float(theoretical[0]))},
            {"theoretical": safe_float(float(theoretical[-1])), "ref": safe_float(float(theoretical[-1]))},
        ]

        # ── Portfolio-level full stats ────────────────────────────────────────
        port_stats = _asset_distribution_stats(port_r, req.extremeThresholds)

        # ── Per-asset stats ───────────────────────────────────────────────────
        asset_stats = []
        for a in req.assets:
            r = _arr(a.returns)
            if req.logScale:
                r = np.log1p(r)
            stats = _asset_distribution_stats(r, req.extremeThresholds)
            stats["ticker"] = a.ticker
            asset_stats.append(stats)

        return to_native({
            "histogram":   hist,
            "normalCurve": normal_curve,
            "qqData":      qq_data,
            "qqLine":      qq_line,
            "portfolio":   port_stats,
            "assets":      asset_stats,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rolling Statistics
# ══════════════════════════════════════════════════════════════════════════════

class RollingStatsRequest(BaseModel):
    assets:      List[AssetIn]
    window:      int = 12
    freqPerYear: int = 12


def _rolling_max_drawdown(chunk: np.ndarray) -> float:
    """Max drawdown within a return window."""
    peak = 1.0
    nav  = 1.0
    mdd  = 0.0
    for r in chunk:
        nav  *= (1 + r)
        peak  = max(peak, nav)
        dd    = (nav - peak) / peak
        mdd   = min(mdd, dd)
    return mdd  # negative


@router.post("/rolling-stats")
def run_rolling_stats(req: RollingStatsRequest):
    try:
        freq = req.freqPerYear
        w    = req.window
        results = []

        for a in req.assets:
            r    = _arr(a.returns)
            n    = len(r)
            roll_mean   = [None] * n
            roll_vol    = [None] * n
            roll_sharpe = [None] * n
            roll_mdd    = [None] * n

            for i in range(w, n + 1):
                chunk = r[i - w:i]
                m     = float(chunk.mean())
                s     = float(chunk.std(ddof=1)) or 1e-8
                roll_mean[i-1]   = safe_float(m * freq)
                roll_vol[i-1]    = safe_float(s * math.sqrt(freq))
                roll_sharpe[i-1] = safe_float(m / s * math.sqrt(freq))
                roll_mdd[i-1]    = safe_float(_rolling_max_drawdown(chunk))

            # Summary stats over valid (non-None) rolling values
            valid_vol    = [v for v in roll_vol    if v is not None]
            valid_sharpe = [v for v in roll_sharpe if v is not None]
            valid_mdd    = [v for v in roll_mdd    if v is not None]

            def _summary(vals: list) -> dict:
                if not vals:
                    return {"avg": None, "max": None, "min": None, "std": None}
                arr = np.array(vals, dtype=float)
                return {
                    "avg": safe_float(float(arr.mean())),
                    "max": safe_float(float(arr.max())),
                    "min": safe_float(float(arr.min())),
                    "std": safe_float(float(arr.std(ddof=1))) if len(arr) > 1 else None,
                }

            # Regime detection: high-vol periods (rolling vol > median rolling vol)
            regimes: List[Optional[str]] = [None] * n
            if valid_vol:
                med_vol = float(np.median(valid_vol))
                for idx in range(n):
                    if roll_vol[idx] is not None:
                        regimes[idx] = "high" if (roll_vol[idx] or 0) > med_vol else "low"

            results.append({
                "ticker":        a.ticker,
                "rollingMean":   roll_mean,
                "rollingVol":    roll_vol,
                "rollingSharpe": roll_sharpe,
                "rollingMaxDD":  roll_mdd,
                "regimes":       regimes,
                "volSummary":    _summary(valid_vol),
                "sharpeSummary": _summary(valid_sharpe),
                "mddSummary":    _summary(valid_mdd),
            })

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Autocorrelation — statsmodels ACF/PACF/Ljung-Box/ADF
# ══════════════════════════════════════════════════════════════════════════════

class AutocorrRequest(BaseModel):
    assets:      List[AssetIn]
    maxLag:      int  = 20
    logRet:      bool = False
    diff:        bool = False
    rollWindow:  int  = 12


@router.post("/autocorrelation")
def run_autocorrelation(req: AutocorrRequest):
    try:
        results = []
        for a in req.assets:
            r = _arr(a.returns)
            if req.logRet:
                r = np.log1p(r)
            if req.diff:
                r = np.diff(r)
            if len(r) < req.maxLag + 5:
                continue

            lags = min(req.maxLag, len(r) // 4)
            n    = len(r)

            # ACF / PACF
            acf_vals, confint = acf(r, nlags=lags, fft=True, alpha=0.05)
            try:
                pacf_vals = pacf(r, nlags=lags, method="ywm")
            except Exception:
                pacf_vals = np.zeros(lags + 1)

            conf_band = 1.96 / math.sqrt(n)

            # ACF on |r| (volatility clustering) and r^2 (ARCH effect)
            abs_acf_vals, _ = acf(np.abs(r), nlags=lags, fft=True, alpha=0.05)
            sq_acf_vals,  _ = acf(r ** 2,    nlags=lags, fft=True, alpha=0.05)

            # Ljung-Box full table up to maxLag
            lb_all = acorr_ljungbox(r, lags=list(range(1, lags + 1)), return_df=True)
            lb_table = [
                {
                    "lag": int(lag),
                    "q":   safe_float(float(lb_all["lb_stat"].iloc[i])),
                    "p":   safe_float(float(lb_all["lb_pvalue"].iloc[i])),
                    "sig": bool(float(lb_all["lb_pvalue"].iloc[i]) < 0.05),
                }
                for i, lag in enumerate(range(1, lags + 1))
            ]
            lb_summary = {
                "q":           safe_float(float(lb_all["lb_stat"].iloc[-1])),
                "p":           safe_float(float(lb_all["lb_pvalue"].iloc[-1])),
                "significant": bool(float(lb_all["lb_pvalue"].iloc[-1]) < 0.05),
            }

            # ADF
            adf_stat, adf_p, *_ = adfuller(r, maxlag=min(5, lags), autolag="AIC")

            # Lag-return scatter (lags 1-5)
            lag_scatter = []
            for lag in range(1, min(6, n)):
                for t in range(lag, n):
                    lag_scatter.append({
                        "lag": lag,
                        "x":   safe_float(float(r[t - lag])),
                        "y":   safe_float(float(r[t])),
                    })

            # Rolling lag-1 ACF
            w = req.rollWindow
            rolling_acf1 = [None] * n
            for i in range(w, n + 1):
                chunk = r[i - w:i]
                if len(chunk) > 4:
                    try:
                        vals, _ = acf(chunk, nlags=1, fft=True, alpha=0.05)
                        rolling_acf1[i - 1] = safe_float(float(vals[1]))
                    except Exception:
                        pass

            results.append({
                "ticker":         a.ticker,
                "acf":            [safe_float(v) for v in acf_vals[1:lags+1].tolist()],
                "pacf":           [safe_float(v) for v in pacf_vals[1:lags+1].tolist()],
                "absAcf":         [safe_float(v) for v in abs_acf_vals[1:lags+1].tolist()],
                "sqAcf":          [safe_float(v) for v in sq_acf_vals[1:lags+1].tolist()],
                "confBound":      safe_float(conf_band),
                "ljungBox":       lb_summary,
                "ljungBoxTable":  lb_table,
                "adf": {
                    "stat":          safe_float(float(adf_stat)),
                    "pvalue":        safe_float(float(adf_p)),
                    "meanReverting": bool(adf_p < 0.05),
                },
                "lagScatter":     lag_scatter,
                "rollingAcf1":    rolling_acf1,
            })

        return to_native(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")
