#!/usr/bin/env python3
"""Return Time Series — descriptive analysis of a financial return series.
numpy / pandas / scipy.

Summarises the distribution (moments, skew, kurtosis), annualised return and
volatility, cumulative performance, drawdown, rolling volatility, and normality.

Input (from return-time-series-page.tsx):
    data             : list[dict]
    asset_col        : str
    is_returns       : bool
    return_type      : "simple"|"log"
    periods_per_year : int   (default 252)
    roll_window      : int   (default 21) rolling volatility window
Output: { results: {...}, plot } (cumulative + rolling vol + histogram).
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
        col = p.get("asset_col")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        roll = int(p.get("roll_window") or 21)
        if not col or col not in df.columns:
            raise ValueError("Select the return column.")

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
        if n < 10:
            raise ValueError("Need at least 10 return observations.")
        rv = ret.values

        mean = float(np.mean(rv)); sd = float(np.std(rv, ddof=1))
        skew = float(sstats.skew(rv)); kurt = float(sstats.kurtosis(rv))  # excess
        ann_ret = mean * ppy
        ann_vol = sd * np.sqrt(ppy)
        sharpe0 = (ann_ret / ann_vol) if ann_vol > 0 else None
        cum = float(np.prod(1 + rv) - 1)
        pos = float(np.mean(rv > 0))
        best = float(np.max(rv)); worst = float(np.min(rv))
        # drawdown
        wealth = np.cumprod(1 + rv); peak = np.maximum.accumulate(wealth)
        max_dd = float((wealth / peak - 1).min())
        # normality (Jarque-Bera)
        jb, jb_p = sstats.jarque_bera(rv)
        var5 = float(np.percentile(rv, 5))

        roll = max(2, min(roll, n // 2))
        roll_vol = pd.Series(rv).rolling(roll).std(ddof=1) * np.sqrt(ppy)

        plot = None
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=115)
            ax1.plot((wealth - 1) * 100, color="#2563eb", lw=1.4)
            ax1.axhline(0, color="#111827", lw=0.7)
            ax1.set_title("Cumulative return (%)"); ax1.set_xlabel("Period"); ax1.grid(alpha=0.2)
            ax2.plot(roll_vol.values * 100, color="#f59e0b", lw=1.2)
            ax2.axhline(ann_vol * 100, color="#dc2626", ls="--", lw=1, label="Full-sample")
            ax2.set_title(f"Rolling volatility ({roll}p, ann. %)"); ax2.set_xlabel("Period")
            ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.2)
            ax3.hist(rv * 100, bins=min(50, max(10, n // 8)), color="#93c5fd", edgecolor="white", density=True)
            xs = np.linspace(rv.min(), rv.max(), 100)
            ax3.plot(xs * 100, sstats.norm.pdf(xs, mean, sd) / 100, color="#dc2626", lw=1.5, label="Normal")
            ax3.set_title(f"Return distribution (skew {skew:.2f}, kurt {kurt:.2f})")
            ax3.set_xlabel("Return (%)"); ax3.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        tail_txt = ("fat-tailed (leptokurtic) — extreme moves are more frequent than a normal distribution predicts"
                    if kurt > 1 else "close to normal-tailed" if abs(kurt) <= 1 else "thin-tailed")
        skew_txt = ("negatively skewed — the left tail is longer, so large losses outweigh large gains"
                    if skew < -0.2 else "positively skewed — the right tail is longer" if skew > 0.2 else "roughly symmetric")
        interpretation = (
            f"Over {n} periods the series returned {ann_ret:.1%} annualised with {ann_vol:.1%} volatility "
            f"(cumulative {cum:.1%}). The return distribution is {skew_txt} and {tail_txt}. "
            f"{pos:.0%} of periods were positive, the worst single period was {worst:.2%}, and the deepest drawdown "
            f"was {max_dd:.1%}. "
            + (f"The Jarque-Bera test rejects normality (p = {jb_p:.3f}), which is typical of financial returns and "
               "means risk measures assuming a bell curve will understate tail risk." if jb_p < 0.05 else
               f"The Jarque-Bera test does not reject normality (p = {jb_p:.3f}).")
        )

        results = {
            "status": "ok", "asset": col, "n_obs": n, "periods_per_year": ppy, "roll_window": roll,
            "mean": _fin(mean, 6), "std": _fin(sd, 6), "skew": _fin(skew, 4), "excess_kurtosis": _fin(kurt, 4),
            "annual_return": _fin(ann_ret, 5), "annual_volatility": _fin(ann_vol, 5), "sharpe_naive": _fin(sharpe0, 4),
            "cumulative_return": _fin(cum, 5), "pct_positive": _fin(pos, 4),
            "best_period": _fin(best, 5), "worst_period": _fin(worst, 5), "max_drawdown": _fin(max_dd, 5),
            "var_5pct": _fin(var5, 5),
            "jarque_bera": _fin(float(jb), 4), "jb_p_value": _fin(float(jb_p), 6), "is_normal": bool(jb_p >= 0.05),
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
