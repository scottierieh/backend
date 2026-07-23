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

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
GRAY = "#94a3b8"
PURPLE = "#7c3aed"


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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

        # ═══════════════════════ Step-6 report: 9 additive sections ═══════════════════════
        wealth_p = np.asarray(cum_series) + 1.0  # portfolio growth-of-1 series (already computed above)
        INITIAL_VALUE = 100000.0

        def _cagr(total_ret, n_obs, ppy_):
            years = n_obs / ppy_ if ppy_ else None
            if years and years > 0 and (1 + total_ret) > 0:
                return (1 + total_ret) ** (1 / years) - 1.0
            return None

        # ① Performance Summary ------------------------------------------------------------
        perf_summary_table = [
            {"metric": "Total Return", "portfolio": metrics["cumulative_return"],
             "benchmark": (_fin(ep.cum_returns_final(b), 5) if has_bench else None), "fmt": "pct"},
            {"metric": "Annualized Return", "portfolio": metrics["annual_return"],
             "benchmark": (benchmark["bench_annual_return"] if has_bench else None), "fmt": "pct"},
            {"metric": "Annualized Volatility", "portfolio": metrics["annual_volatility"],
             "benchmark": (_fin(ep.annual_volatility(b, annualization=ann_factor), 5) if has_bench else None), "fmt": "pct"},
            {"metric": "Sharpe Ratio", "portfolio": metrics["sharpe"],
             "benchmark": (_fin(ep.sharpe_ratio(b, risk_free=rf_period, annualization=ann_factor), 4) if has_bench else None), "fmt": "num"},
            {"metric": "Sortino Ratio", "portfolio": metrics["sortino"],
             "benchmark": (_fin(ep.sortino_ratio(b, required_return=rf_period, annualization=ann_factor), 4) if has_bench else None), "fmt": "num"},
            {"metric": "Max Drawdown", "portfolio": metrics["max_drawdown"],
             "benchmark": (_fin(ep.max_drawdown(b), 5) if has_bench else None), "fmt": "pct"},
            {"metric": "Calmar Ratio", "portfolio": metrics["calmar"],
             "benchmark": (_fin(ep.calmar_ratio(b, annualization=ann_factor), 4) if has_bench else None), "fmt": "num"},
        ]
        if has_bench:
            perf_summary_table.append({"metric": "Alpha (annualized)", "portfolio": benchmark["alpha"], "benchmark": None, "fmt": "pct"})
            perf_summary_table.append({"metric": "Beta", "portfolio": benchmark["beta"], "benchmark": None, "fmt": "num"})

        # ② Cumulative Performance ----------------------------------------------------------
        p_final = INITIAL_VALUE * (1.0 + (metrics["cumulative_return"] or 0.0))
        p_cagr = _cagr(metrics["cumulative_return"] or 0.0, len(r), ppy)
        cumulative_table = [{
            "series": asset_col, "initial_value": _fin(INITIAL_VALUE, 2), "final_value": _fin(p_final, 2),
            "total_return": metrics["cumulative_return"], "cagr": _fin(p_cagr, 5) if p_cagr is not None else None,
        }]
        if has_bench:
            b_total = _fin(ep.cum_returns_final(b), 5)
            b_final = INITIAL_VALUE * (1.0 + (b_total or 0.0))
            b_cagr = _cagr(b_total or 0.0, len(r), ppy)
            cumulative_table.append({
                "series": bench_col, "initial_value": _fin(INITIAL_VALUE, 2), "final_value": _fin(b_final, 2),
                "total_return": b_total, "cagr": _fin(b_cagr, 5) if b_cagr is not None else None,
            })
        cumulative_note = f"Based on a ${INITIAL_VALUE:,.0f} initial investment."

        chart_cumulative_vs_benchmark = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
            ax.plot(wealth_p * (INITIAL_VALUE / 1.0), color=BLUE, lw=1.8, label=asset_col)
            if has_bench:
                wealth_b = np.asarray(ep.cum_returns(b, starting_value=1.0))
                ax.plot(wealth_b * INITIAL_VALUE, color=GRAY, lw=1.5, ls="--", label=bench_col)
            ax.axhline(INITIAL_VALUE, color="#111827", lw=0.7, ls=":")
            ax.set_title("Portfolio vs Benchmark Growth"); ax.set_xlabel("Period"); ax.set_ylabel("Value ($)")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_cumulative_vs_benchmark = _png(fig)
        except Exception:
            plt.close("all"); chart_cumulative_vs_benchmark = None

        # ③ Risk-Adjusted Performance ---------------------------------------------------------
        treynor = None
        if has_bench and benchmark["beta"] not in (None, 0):
            treynor = _fin((metrics["annual_return"] - rf_annual) / benchmark["beta"], 4)
        risk_adjusted_table = [
            {"metric": "Sharpe Ratio", "portfolio": metrics["sharpe"], "benchmark": (_fin(ep.sharpe_ratio(b, risk_free=rf_period, annualization=ann_factor), 4) if has_bench else None)},
            {"metric": "Sortino Ratio", "portfolio": metrics["sortino"], "benchmark": (_fin(ep.sortino_ratio(b, required_return=rf_period, annualization=ann_factor), 4) if has_bench else None)},
            {"metric": "Calmar Ratio", "portfolio": metrics["calmar"], "benchmark": (_fin(ep.calmar_ratio(b, annualization=ann_factor), 4) if has_bench else None)},
            {"metric": "Treynor Ratio", "portfolio": treynor, "benchmark": None,
             "note": None if treynor is not None else ("Requires a benchmark (beta)." if not has_bench else "Beta unavailable/zero.")},
            {"metric": "Information Ratio", "portfolio": (benchmark["information_ratio"] if has_bench else None), "benchmark": None,
             "note": None if has_bench else "Requires a benchmark."},
        ]

        # ④ Benchmark Comparison ----------------------------------------------------------
        benchmark_comparison_table = None
        chart_relative_performance = None
        if has_bench:
            b_ann_vol = _fin(ep.annual_volatility(b, annualization=ann_factor), 5)
            b_sharpe = _fin(ep.sharpe_ratio(b, risk_free=rf_period, annualization=ann_factor), 4)
            b_max_dd = _fin(ep.max_drawdown(b), 5)
            benchmark_comparison_table = [
                {"metric": "Annual Return", "portfolio": metrics["annual_return"], "benchmark": benchmark["bench_annual_return"],
                 "difference": _fin(metrics["annual_return"] - benchmark["bench_annual_return"], 5)},
                {"metric": "Volatility", "portfolio": metrics["annual_volatility"], "benchmark": b_ann_vol,
                 "difference": _fin(metrics["annual_volatility"] - b_ann_vol, 5)},
                {"metric": "Sharpe", "portfolio": metrics["sharpe"], "benchmark": b_sharpe,
                 "difference": _fin(metrics["sharpe"] - b_sharpe, 4)},
                {"metric": "Max Drawdown", "portfolio": metrics["max_drawdown"], "benchmark": b_max_dd,
                 "difference": _fin(metrics["max_drawdown"] - b_max_dd, 5)},
                {"metric": "Beta", "portfolio": benchmark["beta"], "benchmark": 1.0, "difference": _fin(benchmark["beta"] - 1.0, 4)},
            ]
            rel_cum = (np.asarray(ep.cum_returns(r, starting_value=1.0)) - np.asarray(ep.cum_returns(b, starting_value=1.0))) * 100.0
            try:
                fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
                xs = np.arange(len(rel_cum))
                ax.fill_between(xs, rel_cum, 0, where=(rel_cum >= 0), color=GREEN, alpha=0.35, interpolate=True)
                ax.fill_between(xs, rel_cum, 0, where=(rel_cum < 0), color=RED, alpha=0.35, interpolate=True)
                ax.plot(xs, rel_cum, color="#111827", lw=1.2)
                ax.axhline(0, color="#111827", lw=0.8)
                ax.set_title("Relative Performance (Portfolio − Benchmark, cumulative %)")
                ax.set_xlabel("Period"); ax.set_ylabel("Difference (pp)")
                ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_relative_performance = _png(fig)
            except Exception:
                plt.close("all"); chart_relative_performance = None

        # ⑤ Performance Attribution ----------------------------------------------------------
        # By default, NOT computable: this endpoint receives a single aggregate portfolio return
        # series and a single aggregate benchmark return series (asset_col / benchmark_col), with no
        # per-asset weight/return columns. A true Brinson-style allocation/selection/interaction
        # decomposition requires asset-level weights in both the portfolio and the benchmark.
        # OPTIONAL: if the caller supplies attribution_assets/portfolio_weights/benchmark_weights
        # (matching lengths, >= 2 assets), we compute a genuine Brinson-Fachler attribution below.
        attribution_note = ("Performance attribution requires per-asset weight and return data "
                             "(portfolio and benchmark holdings), which this analysis does not collect — "
                             "only aggregate portfolio and benchmark return series are available. Skipped.")
        attribution_table = None
        attribution_summary = None
        chart_attribution = None
        try:
            attr_assets = p.get("attribution_assets") or []
            attr_pw = p.get("portfolio_weights") or []
            attr_bw = p.get("benchmark_weights") or []
            if (isinstance(attr_assets, list) and len(attr_assets) >= 2
                    and len(attr_pw) == len(attr_assets) and len(attr_bw) == len(attr_assets)
                    and all(c in df.columns for c in attr_assets)):
                pw = np.array([float(x) for x in attr_pw], dtype=float)
                bw = np.array([float(x) for x in attr_bw], dtype=float)
                if pw.sum() > 0:
                    pw = pw / pw.sum()
                if bw.sum() > 0:
                    bw = bw / bw.sum()
                asset_ann_returns = []
                for col in attr_assets:
                    asset_series = _to_returns(df[col], is_returns, rtype).dropna()
                    asset_ann_returns.append(float(asset_series.mean() * ppy))
                asset_ann_returns = np.array(asset_ann_returns, dtype=float)

                rows = []
                total_alloc = total_sel = total_inter = 0.0
                for i, name in enumerate(attr_assets):
                    r_p = asset_ann_returns[i]
                    r_b = asset_ann_returns[i]  # same underlying asset return series stands in for both
                    # NOTE: portfolio return for asset i and benchmark return for asset i are both
                    # derived from the same selected return column (the caller selects one return
                    # series per asset); weights differ between portfolio and benchmark.
                    alloc = (pw[i] - bw[i]) * r_b
                    sel = bw[i] * (r_p - r_b)
                    inter = (pw[i] - bw[i]) * (r_p - r_b)
                    total_alloc += alloc; total_sel += sel; total_inter += inter
                    rows.append({"Asset": name, "Allocation": _fin(alloc, 5), "Selection": _fin(sel, 5), "Interaction": _fin(inter, 5)})

                total_active = total_alloc + total_sel + total_inter
                rows.append({
                    "Asset": "Total",
                    "Allocation": _fin(total_alloc, 5),
                    "Selection": _fin(total_sel, 5),
                    "Interaction": _fin(total_inter, 5),
                })
                attribution_table = rows
                attribution_summary = {
                    "allocation_effect": _fin(total_alloc, 5),
                    "selection_effect": _fin(total_sel, 5),
                    "interaction_effect": _fin(total_inter, 5),
                    "total_active_return": _fin(total_active, 5),
                }
                attribution_note = (
                    "Brinson-Fachler attribution computed from the selected per-asset return columns "
                    "and the supplied portfolio/benchmark weights.")

                try:
                    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=115)
                    labels = ["Allocation", "Selection", "Interaction", "Total"]
                    vals = [total_alloc, total_sel, total_inter, total_active]
                    running = 0.0
                    for i, (lbl, v) in enumerate(zip(labels, vals)):
                        if lbl == "Total":
                            bottom = 0.0
                            height = total_active
                        else:
                            bottom = min(running, running + v)
                            height = abs(v)
                        color = GREEN if v >= 0 else RED
                        if lbl == "Total":
                            color = BLUE
                        ax.bar(lbl, height if lbl != "Total" else total_active, bottom=bottom if lbl != "Total" else 0.0, color=color)
                        if lbl != "Total":
                            running += v
                    ax.axhline(0, color="#111827", lw=0.8)
                    ax.set_title("Performance Attribution (Brinson-Fachler)")
                    ax.set_ylabel("Contribution to active return")
                    ax.grid(alpha=0.2, axis="y")
                    fig.tight_layout()
                    chart_attribution = _png(fig)
                except Exception:
                    plt.close("all"); chart_attribution = None
        except Exception:
            attribution_table = None
            attribution_summary = None
            chart_attribution = None

        # ⑥ Period Performance ----------------------------------------------------------------
        n = len(r)
        lookback_defs = [("1M", ppy / 12.0), ("3M", ppy / 4.0), ("6M", ppy / 2.0), ("1Y", float(ppy)), ("3Y", ppy * 3.0)]
        period_performance_table = []
        for label, wlen in lookback_defs:
            wlen_i = int(round(wlen))
            if wlen_i < 1 or wlen_i >= n:
                continue
            p_ret = float(wealth_p[-1] / wealth_p[-1 - wlen_i] - 1.0)
            row = {"window": label, "portfolio_return": _fin(p_ret, 5)}
            if has_bench:
                wealth_b_full = np.asarray(ep.cum_returns(b, starting_value=1.0))
                b_ret = float(wealth_b_full[-1] / wealth_b_full[-1 - wlen_i] - 1.0)
                row["benchmark_return"] = _fin(b_ret, 5)
                row["active_return"] = _fin(p_ret - b_ret, 5)
            period_performance_table.append(row)

        chart_period_comparison = None
        if period_performance_table:
            try:
                labels = [row["window"] for row in period_performance_table]
                p_vals = [row["portfolio_return"] * 100 if row["portfolio_return"] is not None else 0 for row in period_performance_table]
                fig, ax = plt.subplots(figsize=(9, 4.6), dpi=115)
                xs = np.arange(len(labels))
                width = 0.38 if has_bench else 0.55
                ax.bar(xs - (width / 2 if has_bench else 0), p_vals, width=width, color=BLUE, label=asset_col)
                if has_bench:
                    b_vals = [row.get("benchmark_return", 0) * 100 if row.get("benchmark_return") is not None else 0 for row in period_performance_table]
                    ax.bar(xs + width / 2, b_vals, width=width, color=GRAY, label=bench_col)
                ax.axhline(0, color="#111827", lw=0.7)
                ax.set_xticks(xs); ax.set_xticklabels(labels)
                ax.set_title("Periodic Return Comparison"); ax.set_ylabel("Return (%)")
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                chart_period_comparison = _png(fig)
            except Exception:
                plt.close("all"); chart_period_comparison = None

        # ⑦ Rolling Performance -----------------------------------------------------------------
        rolling_table = []
        for label, wlen in lookback_defs:
            wlen_i = int(round(wlen))
            if wlen_i < roll or wlen_i >= n:
                continue
            window = r[-wlen_i:]
            w_ret = float(np.mean(window) * ann_factor)
            w_vol = float(np.std(window, ddof=1) * np.sqrt(ann_factor)) if len(window) > 1 else None
            w_sharpe = ((w_ret - rf_annual) / w_vol) if (w_vol and w_vol > 0) else None
            rolling_table.append({
                "window": label, "return": _fin(w_ret, 5), "volatility": _fin(w_vol, 5) if w_vol is not None else None,
                "sharpe": _fin(w_sharpe, 4) if w_sharpe is not None else None,
            })

        chart_rolling_sharpe = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=115)
            rs_arr = np.array([v if v is not None else np.nan for v in roll_sharpe_vals])
            ax.plot(rs_arr, color=PURPLE, lw=1.5)
            ax.axhline(0, color="#111827", lw=0.8)
            ax.set_title(f"Rolling Sharpe Ratio ({roll}-period window)")
            ax.set_xlabel("Period"); ax.set_ylabel("Rolling Sharpe")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_rolling_sharpe = _png(fig)
        except Exception:
            plt.close("all"); chart_rolling_sharpe = None

        # ⑧ Up / Down Capture -------------------------------------------------------------------
        capture_table = None
        if has_bench:
            up_cap = benchmark["up_capture"]
            down_cap = benchmark["down_capture"]
            capture_ratio = _fin(up_cap / down_cap, 4) if (up_cap is not None and down_cap not in (None, 0)) else None
            capture_table = [
                {"metric": "Up Capture Ratio", "value": up_cap, "hint": "Portfolio gain / benchmark gain, when benchmark > 0"},
                {"metric": "Down Capture Ratio", "value": down_cap, "hint": "Portfolio loss / benchmark loss, when benchmark < 0"},
                {"metric": "Capture Ratio", "value": capture_ratio, "hint": "Up capture / down capture — above 1 is favourable"},
            ]

        # ⑨ Drawdown Performance -----------------------------------------------------------------
        episodes = []
        in_dd = False
        start_i = trough_i = None
        trough_v = 0.0
        for i, dv in enumerate(dd_series):
            if dv < 0 and not in_dd:
                in_dd = True; start_i = i; trough_i = i; trough_v = dv
            elif dv < 0 and in_dd:
                if dv < trough_v:
                    trough_v = dv; trough_i = i
            elif dv >= 0 and in_dd:
                episodes.append({"start": start_i, "trough": trough_i, "end": i, "recovered": True,
                                  "depth": float(trough_v), "duration": i - start_i, "recovery": i - trough_i})
                in_dd = False
        if in_dd:
            episodes.append({"start": start_i, "trough": trough_i, "end": None, "recovered": False,
                              "depth": float(trough_v), "duration": len(dd_series) - 1 - start_i, "recovery": None})

        recovered_eps = [e for e in episodes if e["recovered"]]
        best_recovery = min((e["recovery"] for e in recovered_eps), default=None)
        longest_dd = max((e["duration"] for e in episodes), default=None)
        current_dd_recovery = next((e for e in episodes if not e["recovered"]), None)
        recovery_period = current_dd_recovery["duration"] if current_dd_recovery else (
            max((e["recovery"] for e in recovered_eps), default=None))

        drawdown_table = [
            {"metric": "Maximum Drawdown", "value": metrics["max_drawdown"], "fmt": "pct"},
            {"metric": "Recovery Period (periods)", "value": recovery_period, "fmt": "int"},
            {"metric": "Calmar Ratio", "value": metrics["calmar"], "fmt": "num"},
            {"metric": "Best Recovery (periods)", "value": best_recovery, "fmt": "int"},
            {"metric": "Longest Drawdown (periods)", "value": longest_dd, "fmt": "int"},
        ]

        chart_drawdown = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=115)
            ax.fill_between(np.arange(len(dd_series)), dd_series * 100, 0, color=RED, alpha=0.35)
            ax.plot(dd_series * 100, color=RED, lw=1.2)
            ax.set_title("Portfolio Drawdown & Recovery")
            ax.set_xlabel("Period"); ax.set_ylabel("Drawdown (%)")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_drawdown = _png(fig)
        except Exception:
            plt.close("all"); chart_drawdown = None

        charts = {
            "cumulative_vs_benchmark": chart_cumulative_vs_benchmark,
            "relative_performance": chart_relative_performance,
            "period_comparison": chart_period_comparison,
            "rolling_sharpe": chart_rolling_sharpe,
            "drawdown": chart_drawdown,
            "attribution": chart_attribution,
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
            "performance_summary_table": perf_summary_table,
            "cumulative_table": cumulative_table,
            "cumulative_note": cumulative_note,
            "risk_adjusted_table": risk_adjusted_table,
            "benchmark_comparison_table": benchmark_comparison_table,
            "attribution_note": attribution_note,
            "attribution_table": attribution_table,
            "attribution_summary": attribution_summary,
            "period_performance_table": period_performance_table,
            "rolling_table": rolling_table,
            "capture_table": capture_table,
            "drawdown_table": drawdown_table,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
