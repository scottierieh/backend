#!/usr/bin/env python3
"""Volatility Analysis — rolling & EWMA volatility, regimes, vol-of-vol.
numpy / pandas / scipy.

Input (from volatility-analysis-page.tsx):
    data        : list[dict]
    value_col   : string   price or return column
    date_col    : string   (optional) column to sort by, "__none__" = row order
    is_returns  : bool
    log_returns : bool     (ignored if is_returns)
    periods_per_year : int (default 252)
    window      : int      rolling window length (default 20)

Output: { results: {...}, plot } — reproduces the fields the client used to
compute in-browser: annVol, rollingVol, ewma, volOfVol, avgRv, maxRv, minRv,
highVolPct, currentVol, currentEwma, skewness, chartData, plus n_obs/n_returns/
value_col/freq/window. Extra fields (regime table, histogram stats) are additive.
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EWMA_LAMBDA = 0.94


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _to_returns(values, is_returns, log_ret):
    values = np.asarray(values, dtype=float)
    if is_returns:
        return values[np.isfinite(values)]
    out = []
    for i in range(1, len(values)):
        p0, p1 = values[i - 1], values[i]
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0 or p1 <= 0:
            continue
        out.append(np.log(p1 / p0) if log_ret else (p1 / p0 - 1.0))
    return np.array(out, dtype=float)


def _rolling_vol(returns, window, ppy):
    out = []
    for i in range(window, len(returns) + 1):
        seg = returns[i - window:i]
        out.append(float(np.std(seg, ddof=1)) * np.sqrt(ppy))
    return np.array(out, dtype=float)


def _ewma_vol(returns, lam, ppy):
    if len(returns) == 0:
        return np.array([])
    out = np.zeros(len(returns))
    v2 = returns[0] ** 2
    out[0] = np.sqrt(v2) * np.sqrt(ppy)
    for i in range(1, len(returns)):
        v2 = lam * v2 + (1 - lam) * returns[i] ** 2
        out[i] = np.sqrt(v2) * np.sqrt(ppy)
    return out


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)

        value_col = p.get("value_col")
        if not value_col or value_col not in df.columns:
            raise ValueError("Select a valid price/return column.")
        date_col = p.get("date_col") or "__none__"
        is_returns = bool(p.get("is_returns", False))
        log_ret = bool(p.get("log_returns", False))
        ppy = int(p.get("periods_per_year") or 252)
        window = max(5, int(p.get("window") or 20))

        if date_col != "__none__" and date_col in df.columns:
            df = df.sort_values(by=date_col, key=lambda s: s.astype(str)).reset_index(drop=True)

        values = pd.to_numeric(df[value_col], errors="coerce")
        values = values[np.isfinite(values)].values

        returns = _to_returns(values, is_returns, log_ret)
        if len(returns) < 5:
            raise ValueError("Not enough valid observations to compute volatility.")

        win_n = min(window, len(returns))
        ann_vol = float(np.std(returns, ddof=1)) * np.sqrt(ppy) if len(returns) > 1 else 0.0
        rv = _rolling_vol(returns, win_n, ppy)
        ewma = _ewma_vol(returns, EWMA_LAMBDA, ppy)

        vol_of_vol = float(np.std(rv, ddof=1)) if len(rv) > 1 else 0.0
        avg_rv = float(np.mean(rv)) if len(rv) else 0.0
        max_rv = float(np.max(rv)) if len(rv) else 0.0
        min_rv = float(np.min(rv)) if len(rv) else 0.0
        high_vol_bars = int(np.sum(rv > avg_rv * 1.5)) if len(rv) else 0
        high_vol_pct = (high_vol_bars / len(rv)) if len(rv) else 0.0
        current_vol = float(rv[-1]) if len(rv) else ann_vol
        current_ewma = float(ewma[-1]) if len(ewma) else ann_vol
        skewness = float(sstats.skew(returns, bias=False)) if len(returns) > 2 else 0.0

        chart_data = []
        for i, v in enumerate(rv):
            ewma_idx = i + win_n - 1
            chart_data.append({
                "idx": i + win_n,
                "vol": _fin(v, 6),
                "ewma": _fin(ewma[ewma_idx], 6) if 0 <= ewma_idx < len(ewma) else None,
            })

        # additive extras: regime breakdown table
        regimes = []
        if len(rv):
            low_th, high_th = avg_rv * 0.75, avg_rv * 1.5
            calm = rv[rv <= low_th]
            normal = rv[(rv > low_th) & (rv <= high_th)]
            stressed = rv[rv > high_th]
            for name, seg in [("Calm", calm), ("Normal", normal), ("Stressed", stressed)]:
                regimes.append({
                    "regime": name, "n": int(len(seg)),
                    "pct": _fin(len(seg) / len(rv), 4) if len(rv) else None,
                    "avg_vol": _fin(float(np.mean(seg)), 5) if len(seg) else None,
                })

        # plot: rolling + EWMA overlay, clustering, return-vs-vol scatter, histogram
        plot = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=110)
            ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

            xs = [c["idx"] for c in chart_data]
            vol_vals = [c["vol"] for c in chart_data]
            ewma_vals = [c["ewma"] for c in chart_data]
            ax1.fill_between(xs, vol_vals, 0, color="#2563eb", alpha=0.15)
            ax1.plot(xs, vol_vals, color="#2563eb", lw=1.3, label="Rolling vol")
            ax1.plot(xs, ewma_vals, color="#f59e0b", lw=1.6, label="EWMA")
            ax1.set_title("Rolling & EWMA volatility"); ax1.set_xlabel("Period"); ax1.set_ylabel("Ann. vol")
            ax1.legend(fontsize=8, frameon=False)

            ax2.plot(range(len(returns)), returns * 100, color="#6366f1", lw=0.8)
            ax2.axhline(0, color="#9ca3af", lw=0.7)
            ax2.set_title("Volatility clustering (returns)"); ax2.set_xlabel("Period"); ax2.set_ylabel("Return (%)")

            if len(rv) == len(returns[win_n - 1:]):
                ax3.scatter(returns[win_n - 1:] * 100, rv * 100, s=10, color="#10b981", alpha=0.6)
            ax3.set_title("Return vs. rolling volatility"); ax3.set_xlabel("Return (%)"); ax3.set_ylabel("Rolling vol (%)")

            if len(rv):
                ax4.hist(rv * 100, bins=min(30, max(8, len(rv) // 4)), color="#93c5fd", edgecolor="white")
            ax4.set_title("Volatility distribution"); ax4.set_xlabel("Rolling vol (%)"); ax4.set_ylabel("Frequency")

            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        results = {
            "status": "ok",
            "n_obs": int(len(values)), "n_returns": int(len(returns)),
            "value_col": value_col, "freq": str(p.get("freq") or ppy), "window": int(win_n),
            "annVol": _fin(ann_vol, 6),
            "rollingVol": [_fin(v, 6) for v in rv],
            "ewma": [_fin(v, 6) for v in ewma],
            "volOfVol": _fin(vol_of_vol, 6),
            "avgRv": _fin(avg_rv, 6),
            "maxRv": _fin(max_rv, 6),
            "minRv": _fin(min_rv, 6),
            "highVolPct": _fin(high_vol_pct, 6),
            "currentVol": _fin(current_vol, 6),
            "currentEwma": _fin(current_ewma, 6),
            "skewness": _fin(skewness, 6),
            "chartData": chart_data,
            "regimes": regimes,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
