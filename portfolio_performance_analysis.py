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

        # per-period time series (additive: does not change existing scalar metrics)
        cum_series = np.asarray(ep.cum_returns(r, starting_value=1.0)) - 1.0
        wealth = cum_series + 1.0
        running_peak = np.maximum.accumulate(wealth)
        dd_series = wealth / running_peak - 1.0
        period_returns = [{"period": int(i), "return": _fin(v, 6)} for i, v in enumerate(r)]
        cumulative_series = [_fin(v, 6) for v in cum_series]
        drawdown_series = [_fin(v, 6) for v in dd_series]

        roll = max(2, min(21 if ppy >= 200 else (4 if ppy <= 12 else 5), len(r) // 2))
        rf_ppy = rf_period
        roll_sharpe_vals = []
        for i in range(len(r)):
            if i + 1 < roll:
                roll_sharpe_vals.append(None)
                continue
            window = r[i + 1 - roll:i + 1]
            mu = np.mean(window) - rf_ppy
            sd = np.std(window, ddof=1)
            roll_sharpe_vals.append(float(mu / sd * np.sqrt(ann_factor)) if sd > 0 else None)
        rolling_sharpe = [_fin(v, 4) if v is not None else None for v in roll_sharpe_vals]

        benchmark_cumulative_series = None
        if has_bench:
            benchmark_cumulative_series = [_fin(v, 6) for v in (np.asarray(ep.cum_returns(b, starting_value=1.0)) - 1.0)]

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

        # plot: 2x3 grid — cumulative, drawdown, rolling Sharpe, ratio bars, portfolio vs benchmark
        plot = None
        try:
            ncols = 3 if has_bench else 2
            fig, axes = plt.subplots(2, ncols, figsize=(6.2 * ncols, 9), dpi=118)
            ax1 = axes[0, 0]
            cr = ep.cum_returns(r, starting_value=1.0)
            ax1.plot(np.asarray(cr) * 100 - 100, color="#2563eb", lw=2, label=asset_col)
            if has_bench:
                cb = ep.cum_returns(b, starting_value=1.0)
                ax1.plot(np.asarray(cb) * 100 - 100, color="#94a3b8", lw=1.5, ls="--", label=bench_col)
            ax1.axhline(0, color="#111827", lw=0.8)
            ax1.set_xlabel("Period"); ax1.set_ylabel("Cumulative return (%)")
            ax1.set_title("Cumulative performance"); ax1.legend(fontsize=8, frameon=False); ax1.grid(alpha=0.2)

            ax2 = axes[0, 1]
            names = ["Sharpe", "Sortino", "Calmar"]
            vals = [metrics["sharpe"], metrics["sortino"], metrics["calmar"]]
            cols = ["#16a34a" if (v or 0) > 0 else "#dc2626" for v in vals]
            bars = ax2.bar(names, vals, color=cols)
            for bbar, v in zip(bars, vals):
                ax2.text(bbar.get_x() + bbar.get_width()/2, v, f"{v:.2f}", ha="center",
                         va="bottom" if (v or 0) >= 0 else "top", fontsize=9)
            ax2.axhline(0, color="#111827", lw=0.8)
            ax2.set_title("Risk-adjusted ratios"); ax2.grid(axis="y", alpha=0.2)

            if has_bench:
                ax3 = axes[0, 2]
                ax3.plot(np.asarray(cr) * 100 - 100, color="#2563eb", lw=2, label=asset_col)
                ax3.plot(np.asarray(cb) * 100 - 100, color="#f59e0b", lw=2, label=bench_col)
                ax3.axhline(0, color="#111827", lw=0.8)
                ax3.set_xlabel("Period"); ax3.set_ylabel("Cumulative return (%)")
                ax3.set_title("Portfolio vs benchmark"); ax3.legend(fontsize=8, frameon=False); ax3.grid(alpha=0.2)

            ax4 = axes[1, 0]
            ax4.fill_between(np.arange(len(dd_series)), dd_series * 100, 0, color="#dc2626", alpha=0.35)
            ax4.plot(dd_series * 100, color="#dc2626", lw=1)
            ax4.set_xlabel("Period"); ax4.set_ylabel("Drawdown (%)")
            ax4.set_title("Drawdown"); ax4.grid(alpha=0.2)

            ax5 = axes[1, 1]
            rs_arr = np.array([v if v is not None else np.nan for v in roll_sharpe_vals])
            ax5.plot(rs_arr, color="#7c3aed", lw=1.5)
            ax5.axhline(0, color="#111827", lw=0.8)
            ax5.set_xlabel("Period"); ax5.set_ylabel("Rolling Sharpe")
            ax5.set_title(f"Rolling Sharpe ({roll}p)"); ax5.grid(alpha=0.2)

            if ncols == 3:
                fig.delaxes(axes[1, 2])
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
            "period_returns": period_returns,
            "cumulative_series": cumulative_series,
            "drawdown_series": drawdown_series,
            "rolling_sharpe": rolling_sharpe,
            "roll_window": roll,
            "benchmark_cumulative_series": benchmark_cumulative_series,
            "monthly_returns": None,  # no reliable date column in the input payload — skipped
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
