#!/usr/bin/env python3
"""Portfolio Performance — risk-adjusted performance metrics. empyrical-reloaded.

Computes Sharpe, Sortino, Calmar, Omega, annualised return/volatility, max
drawdown, and (against a benchmark) alpha, beta and the information ratio.

Input (from portfolio-performance-page.tsx):
    data             : list[dict]
    asset_col        : str    portfolio return/price column
    benchmark_col    : str    (optional) benchmark return/price column
    is_returns       : bool
    return_type      : "simple"|"log"
    periods_per_year : int    (default 252)
    rf_annual        : float  (default 0.0)
Output: { results: {...}, plot } (cumulative curve + metric bars).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import empyrical as ep

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _to_returns(s, is_returns, rtype):
    s = pd.to_numeric(s, errors="coerce")
    if is_returns:
        return s
    if rtype == "log":
        if (s <= 0).any():
            raise ValueError("Log returns require positive prices.")
        return np.log(s / s.shift(1))
    return s / s.shift(1) - 1.0


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        asset_col = p.get("asset_col")
        bench_col = p.get("benchmark_col") or None
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        rf_annual = float(p.get("rf_annual") or 0.0)
        if not asset_col or asset_col not in df.columns:
            raise ValueError("Select the portfolio return column.")
        rf_period = rf_annual / ppy

        R = _to_returns(df[asset_col], is_returns, rtype)
        has_bench = bool(bench_col and bench_col in df.columns)
        if has_bench:
            B = _to_returns(df[bench_col], is_returns, rtype)
            joined = pd.concat([R.rename("_r"), B.rename("_b")], axis=1).dropna().reset_index(drop=True)
            r = joined["_r"].values; b = joined["_b"].values
        else:
            r = R.dropna().reset_index(drop=True).values; b = None
        if len(r) < 10:
            raise ValueError("Need at least 10 aligned return observations.")

        ann_factor = ppy
        metrics = {
            "annual_return": _fin(ep.annual_return(r, annualization=ann_factor), 5),
            "annual_volatility": _fin(ep.annual_volatility(r, annualization=ann_factor), 5),
            "cumulative_return": _fin(ep.cum_returns_final(r), 5),
            "sharpe": _fin(ep.sharpe_ratio(r, risk_free=rf_period, annualization=ann_factor), 4),
            "sortino": _fin(ep.sortino_ratio(r, required_return=rf_period, annualization=ann_factor), 4),
            "calmar": _fin(ep.calmar_ratio(r, annualization=ann_factor), 4),
            "omega": _fin(ep.omega_ratio(r, risk_free=rf_period), 4),
            "max_drawdown": _fin(ep.max_drawdown(r), 5),
            "downside_risk": _fin(ep.downside_risk(r, required_return=rf_period, annualization=ann_factor), 5),
            "tail_ratio": _fin(ep.tail_ratio(r), 4),
            "value_at_risk": _fin(ep.value_at_risk(r, cutoff=0.05), 5),
            "stability": _fin(ep.stability_of_timeseries(r), 4),
        }

        benchmark = None
        if has_bench:
            alpha, beta = ep.alpha_beta(r, b, risk_free=rf_period, annualization=ann_factor)
            excess = r - b
            act_track = float(np.std(excess, ddof=1)) * np.sqrt(ann_factor)
            info_ratio = (float(np.mean(excess)) * ann_factor / act_track) if act_track > 0 else None
            benchmark = {
                "name": bench_col,
                "alpha": _fin(alpha, 5), "beta": _fin(beta, 4),
                "information_ratio": _fin(info_ratio, 4) if info_ratio is not None else None,
                "tracking_error": _fin(act_track, 5),
                "bench_annual_return": _fin(ep.annual_return(b, annualization=ann_factor), 5),
                "bench_cumulative": _fin(ep.cum_returns_final(b), 5),
                "up_capture": _fin(ep.up_capture(r, b), 4),
                "down_capture": _fin(ep.down_capture(r, b), 4),
            }

        # plot: cumulative curve + key ratio bars
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=118,
                                           gridspec_kw={"width_ratios": [1.5, 1]})
            cr = ep.cum_returns(r, starting_value=1.0)
            ax1.plot(np.asarray(cr) * 100 - 100, color="#2563eb", lw=2, label=asset_col)
            if has_bench:
                cb = ep.cum_returns(b, starting_value=1.0)
                ax1.plot(np.asarray(cb) * 100 - 100, color="#94a3b8", lw=1.5, ls="--", label=bench_col)
            ax1.axhline(0, color="#111827", lw=0.8)
            ax1.set_xlabel("Period"); ax1.set_ylabel("Cumulative return (%)")
            ax1.set_title("Cumulative performance"); ax1.legend(fontsize=8, frameon=False); ax1.grid(alpha=0.2)
            names = ["Sharpe", "Sortino", "Calmar"]
            vals = [metrics["sharpe"], metrics["sortino"], metrics["calmar"]]
            cols = ["#16a34a" if (v or 0) > 0 else "#dc2626" for v in vals]
            bars = ax2.bar(names, vals, color=cols)
            for bbar, v in zip(bars, vals):
                ax2.text(bbar.get_x() + bbar.get_width()/2, v, f"{v:.2f}", ha="center",
                         va="bottom" if (v or 0) >= 0 else "top", fontsize=9)
            ax2.axhline(0, color="#111827", lw=0.8)
            ax2.set_title("Risk-adjusted ratios"); ax2.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        sharpe = metrics["sharpe"] or 0
        quality = "strong" if sharpe > 1 else "reasonable" if sharpe > 0.5 else "weak"
        interpretation = (
            f"Over {len(r)} periods the portfolio returned {metrics['annual_return']:.1%} a year with "
            f"{metrics['annual_volatility']:.1%} volatility, for a Sharpe ratio of {sharpe:.2f} — {quality} "
            f"risk-adjusted performance. The Sortino ratio of {metrics['sortino']:.2f} focuses only on downside "
            f"risk, and the worst drawdown was {metrics['max_drawdown']:.1%}. "
            + (f"Against {bench_col}, the annualised alpha is {benchmark['alpha']:.2%} with a beta of "
               f"{benchmark['beta']:.2f}; the information ratio of {benchmark['information_ratio']:.2f} measures the "
               "consistency of that outperformance per unit of tracking error. " if benchmark else "")
            + "Sharpe compares return to total volatility; Sortino and Calmar penalise only harmful risk."
        )

        results = {
            "status": "ok", "asset": asset_col, "n_obs": int(len(r)), "periods_per_year": ppy,
            "rf_annual": _fin(rf_annual, 5), **metrics, "benchmark": benchmark,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
