#!/usr/bin/env python3
"""Returns & Volatility — core quantitative finance stats for one price series.

CLI contract: read ONE JSON object from stdin, print ONE JSON object to stdout
on success, or {"error": ...} to stderr + exit(1).

From a price (or return) series it computes the standard risk/return report:
periodic & annualised return, volatility, Sharpe/Sortino, max drawdown,
historical VaR/CVaR, skew/kurtosis, %-positive periods, and an equity-curve +
rolling-volatility plot.

Input JSON (from src/components/pages/statistica/returns-volatility-page.tsx):
    data              : list[dict]     rows
    value_col         : str            price level column (or returns if is_returns)
    date_col          : str  (optional) for ordering / x-axis labels
    is_returns        : bool (optional) True = value_col already holds period returns
    return_type       : "simple"|"log" (optional; default "simple")
    periods_per_year  : int  (optional; default 252 = daily)
    rf_annual         : float(optional; default 0) annual risk-free rate
    roll_window       : int  (optional; default = periods_per_year // 4)

Output JSON: { "results": {...}, "plot": "data:image/png;base64,..." | None }
"""
import sys
import json
import io
import base64

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        value_col = p.get("value_col")
        if not value_col or value_col not in df.columns:
            raise ValueError("Select a valid price/return column (value_col).")
        date_col = p.get("date_col") or None
        is_returns = bool(p.get("is_returns", False))
        return_type = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        rf_annual = float(p.get("rf_annual") or 0.0)
        roll_window = int(p.get("roll_window") or max(2, ppy // 4))

        # order by date if given
        if date_col and date_col in df.columns:
            try:
                df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.sort_values("_d")
            except Exception:
                pass
        labels = df[date_col].astype(str).tolist() if (date_col and date_col in df.columns) else None

        series = pd.to_numeric(df[value_col], errors="coerce").dropna().reset_index(drop=True)
        if len(series) < 3:
            raise ValueError("Need at least 3 valid observations.")

        # ---- returns ----
        if is_returns:
            rets = series.astype(float).reset_index(drop=True)
        else:
            if (series <= 0).any() and return_type == "log":
                raise ValueError("Log returns require strictly positive prices.")
            if return_type == "log":
                rets = np.log(series / series.shift(1)).dropna().reset_index(drop=True)
            else:
                rets = (series / series.shift(1) - 1.0).dropna().reset_index(drop=True)
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        n = len(rets)
        if n < 2:
            raise ValueError("Not enough returns to analyse.")

        r = rets.to_numpy(dtype=float)
        # equity curve (growth of 1)
        equity = np.cumprod(1.0 + r)

        mean_p = float(np.mean(r))
        vol_p = float(np.std(r, ddof=1))
        ann_return = float((equity[-1]) ** (ppy / n) - 1.0)   # geometric annualised
        ann_vol = float(vol_p * np.sqrt(ppy))
        rf_p = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0
        sharpe = float((mean_p - rf_p) / vol_p * np.sqrt(ppy)) if vol_p > 0 else None
        downside = r[r < rf_p]
        dd_dev = float(np.sqrt(np.mean((downside - rf_p) ** 2))) if downside.size > 0 else 0.0
        sortino = float((mean_p - rf_p) / dd_dev * np.sqrt(ppy)) if dd_dev > 0 else None

        # max drawdown on equity curve
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        max_dd = float(drawdown.min())

        # historical VaR / CVaR at 95%
        var95 = float(np.percentile(r, 5))
        cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95

        from scipy import stats as sstats
        skew = float(sstats.skew(r)) if n > 2 else None
        kurt = float(sstats.kurtosis(r)) if n > 3 else None  # excess kurtosis
        pct_pos = float(np.mean(r > 0) * 100.0)
        best = float(np.max(r))
        worst = float(np.min(r))
        calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else None

        # rolling annualised volatility
        roll_vol = pd.Series(r).rolling(roll_window).std(ddof=1) * np.sqrt(ppy)

        # ---- plot: equity curve + rolling vol ----
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), dpi=120, sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1]})
            x = np.arange(len(equity))
            ax1.plot(x, equity, color="#2563eb", lw=1.8)
            ax1.fill_between(x, equity, 1.0, where=(equity >= 1.0), color="#2563eb", alpha=0.08)
            ax1.axhline(1.0, color="#94a3b8", lw=0.8, ls="--")
            ax1.set_ylabel("Growth of 1")
            ax1.set_title(f"Equity curve & rolling volatility ({value_col})")
            rv = roll_vol.to_numpy()
            ax2.plot(np.arange(len(rv)), rv, color="#dc2626", lw=1.5)
            ax2.set_ylabel(f"Roll. vol ({roll_window}p, ann.)")
            ax2.set_xlabel("Period")
            if labels and len(labels) >= len(equity):
                step = max(1, len(equity) // 8)
                ticks = list(range(0, len(equity), step))
                ax2.set_xticks(ticks)
                off = len(labels) - len(equity)
                ax2.set_xticklabels([labels[off + t] for t in ticks], rotation=45, ha="right", fontsize=7)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all")
            plot = None

        interpretation = (
            f"Over {n} periods, {ann_return*100:.2f}% annualised return against "
            f"{ann_vol*100:.2f}% annualised volatility"
            + (f" (Sharpe {sharpe:.2f})" if sharpe is not None else "")
            + f". The worst peak-to-trough drawdown was {max_dd*100:.1f}%, and on the worst "
            f"5% of periods the loss averaged {cvar95*100:.2f}% (CVaR 95%). "
            f"{pct_pos:.0f}% of periods were positive. "
            + ("Returns are left-skewed (bigger crashes than rallies)." if (skew is not None and skew < -0.2)
               else "Returns are right-skewed (bigger rallies than crashes)." if (skew is not None and skew > 0.2)
               else "Returns are roughly symmetric.")
            + (" Fat tails: extreme moves are more common than a normal distribution predicts."
               if (kurt is not None and kurt > 1) else "")
        )

        results = {
            "n_obs": n, "value_col": value_col,
            "is_returns": is_returns, "return_type": return_type,
            "periods_per_year": ppy, "rf_annual": _fin(rf_annual, 6), "roll_window": roll_window,
            "summary": {
                "ann_return": _fin(ann_return), "ann_vol": _fin(ann_vol),
                "mean_period_return": _fin(mean_p), "period_vol": _fin(vol_p),
                "sharpe": _fin(sharpe), "sortino": _fin(sortino), "calmar": _fin(calmar),
                "max_drawdown": _fin(max_dd),
                "var_95": _fin(var95), "cvar_95": _fin(cvar95),
                "skew": _fin(skew), "excess_kurtosis": _fin(kurt),
                "pct_positive": _fin(pct_pos, 2),
                "best_period": _fin(best), "worst_period": _fin(worst),
                "cumulative_return": _fin(equity[-1] - 1.0),
            },
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
