#!/usr/bin/env python3
"""Returns & Volatility — risk-return profile of a price/return series.
numpy / pandas / scipy.

Computes annualised return & volatility, Sharpe/Sortino/Calmar, drawdown,
VaR/CVaR, distribution moments, and a rolling-return/volatility view.

Input (from returns-volatility-page.tsx):
    data             : list[dict]
    value_col        : str
    date_col         : str | null
    is_returns       : bool
    return_type      : "simple"|"log"
    periods_per_year : int   (default 252)
    rf_annual        : float (default 0) annual risk-free rate, for Sharpe
Output: { results: { n_obs, value_col, is_returns, return_type,
          periods_per_year, rf_annual, roll_window, summary: {...},
          interpretation }, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        col = p.get("value_col")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        rf_annual = float(p.get("rf_annual") or 0)
        if not col or col not in df.columns:
            raise ValueError("Select the price or return column.")

        s = pd.to_numeric(df[col], errors="coerce")
        if is_returns:
            ret = s.dropna()
        elif rtype == "log":
            if (s <= 0).any():
                raise ValueError("Log returns require positive prices.")
            ret = np.log(s / s.shift(1)).dropna()
        else:
            ret = (s / s.shift(1) - 1.0).dropna()
        ret = ret.reset_index(drop=True)
        n = len(ret)
        if n < 3:
            raise ValueError("Need at least 3 return observations.")
        rv = ret.values

        mean = float(np.mean(rv))
        sd = float(np.std(rv, ddof=1)) if n > 1 else 0.0
        skew = float(sstats.skew(rv)) if n > 2 else None
        kurt = float(sstats.kurtosis(rv)) if n > 3 else None
        ann_ret = mean * ppy
        ann_vol = sd * np.sqrt(ppy)
        rf_period = rf_annual / ppy
        excess = rv - rf_period
        sharpe = (float(np.mean(excess)) / sd * np.sqrt(ppy)) if sd > 0 else None
        downside = rv[rv < rf_period]
        dd_sd = float(np.std(downside, ddof=1)) if len(downside) > 1 else None
        sortino = ((mean - rf_period) / dd_sd * np.sqrt(ppy)) if dd_sd and dd_sd > 0 else None

        wealth = np.cumprod(1 + rv)
        peak = np.maximum.accumulate(wealth)
        drawdown = wealth / peak - 1.0
        max_dd = float(drawdown.min())
        calmar = (ann_ret / abs(max_dd)) if max_dd < 0 else None
        cum = float(wealth[-1] - 1.0)

        var95 = float(np.percentile(rv, 5))
        tail = rv[rv <= var95]
        cvar95 = float(np.mean(tail)) if len(tail) else var95

        pos_pct = float(np.mean(rv > 0) * 100)
        best = float(np.max(rv)); worst = float(np.min(rv))

        roll = max(2, min(21, n // 2))
        roll_ret = pd.Series(rv).rolling(roll).mean() * ppy
        roll_vol = pd.Series(rv).rolling(roll).std(ddof=1) * np.sqrt(ppy)

        plot = None
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=115)
            ax1.plot((wealth - 1) * 100, color="#2563eb", lw=1.4)
            ax1.axhline(0, color="#111827", lw=0.7)
            ax1.set_title("Cumulative return (%)"); ax1.set_xlabel("Period"); ax1.grid(alpha=0.2)

            ax2.plot(roll_ret.values * 100, color="#16a34a", lw=1.2, label="Rolling ann. return")
            ax2.axhline(ann_ret * 100, color="#dc2626", ls="--", lw=1, label="Full-sample")
            ax2.set_title(f"Rolling return ({roll}p, ann. %)"); ax2.set_xlabel("Period")
            ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.2)

            ax3.hist(rv * 100, bins=min(50, max(10, n // 8)), color="#93c5fd", edgecolor="white", density=True)
            if n > 3:
                xs = np.linspace(rv.min(), rv.max(), 100)
                ax3.plot(xs * 100, sstats.norm.pdf(xs, mean, sd) / 100, color="#dc2626", lw=1.5, label="Normal")
                ax3.legend(fontsize=8, frameon=False)
            ax3.set_title("Return distribution"); ax3.set_xlabel("Return (%)")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Over {n} periods {col} returned {ann_ret:.1%} annualised with {ann_vol:.1%} volatility "
            f"(cumulative {cum:.1%}). Sharpe was {sharpe:.2f}" if sharpe is not None else
            f"Over {n} periods {col} returned {ann_ret:.1%} annualised with {ann_vol:.1%} volatility."
        )
        if sharpe is not None:
            interpretation += (
                f", and the deepest drawdown was {max_dd:.1%}. {pos_pct:.0f}% of periods were positive."
            )

        summary = {
            "ann_return": _fin(ann_ret, 5), "ann_vol": _fin(ann_vol, 5),
            "mean_period_return": _fin(mean, 6), "period_vol": _fin(sd, 6),
            "sharpe": _fin(sharpe, 4), "sortino": _fin(sortino, 4), "calmar": _fin(calmar, 4),
            "max_drawdown": _fin(max_dd, 5), "var_95": _fin(var95, 5), "cvar_95": _fin(cvar95, 5),
            "skew": _fin(skew, 4), "excess_kurtosis": _fin(kurt, 4), "pct_positive": _fin(pos_pct, 2),
            "best_period": _fin(best, 5), "worst_period": _fin(worst, 5), "cumulative_return": _fin(cum, 5),
        }
        results = {
            "status": "ok", "n_obs": n, "value_col": col, "is_returns": is_returns, "return_type": rtype,
            "periods_per_year": ppy, "rf_annual": _fin(rf_annual, 4), "roll_window": roll,
            "summary": summary, "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
