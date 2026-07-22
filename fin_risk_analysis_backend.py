#!/usr/bin/env python3
"""Financial Risk Analysis — drawdown episodes, downside deviation, Sortino/Sharpe/Calmar,
Ulcer & Pain indices, rolling volatility. numpy / pandas.

Input (from fin-risk-analysis-page.tsx):
    data        : list[dict]
    value_col   : str            price or return column
    date_col    : str | None     optional ordering column
    is_returns  : bool
    log_returns : bool
    freq        : str        FREQ key used by the frontend ("252"/"52"/"12"/"4"), echoed back
    periods_per_year : int   (default 252)
    rf_annual   : float      (default 0) annual risk-free rate
    mar_annual  : float      (default 0) annual minimum acceptable return

Output: { results: {...}, plot } — results keys match the frontend's existing
in-browser computation exactly (camelCase) so this is a drop-in swap:
  n_returns, value_col, freq, ppy, smallSample, annReturn, annVol, downsideDev,
  sortino, sharpe, maxDD, calmar, ulcer, pain, pctNegative, avgNegReturn,
  longestLosingStreak, chartData[{idx,label,dd}], episodes[{depth,duration,
  recovery,recovered}], rollingChartData[{idx,vol}], rvWindow, rf, mar.
Plot: 2-panel matplotlib PNG — underwater (drawdown) curve + rolling volatility.
"""
import sys, json, io, math
import base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
LIGHT_BLUE = "#93c5fd"
LIGHT_RED = "#fca5a5"


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_returns(values, is_returns, log_ret):
    values = [v for v in values if np.isfinite(v)]
    if is_returns:
        return values
    out = []
    for i in range(1, len(values)):
        p0, p1 = values[i - 1], values[i]
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0 or p1 <= 0:
            continue
        out.append(math.log(p1 / p0) if log_ret else p1 / p0 - 1.0)
    return out


def to_cumulative(returns):
    cum = [1.0]
    for r in returns:
        cum.append(cum[-1] * (1 + r))
    return cum


def max_drawdown_series(cum):
    peak = cum[0]
    dd = []
    for v in cum:
        if v > peak:
            peak = v
        dd.append((v / peak - 1) if peak > 0 else 0.0)
    max_dd = min(dd) if dd else 0.0
    ulcer = math.sqrt(np.mean([d * d for d in dd])) if dd else 0.0
    pain = float(np.mean([abs(d) for d in dd])) if dd else 0.0
    return dd, max_dd, ulcer, pain


def find_drawdown_episodes(cum):
    episodes = []
    peak = cum[0]; peak_idx = 0
    underwater = False; trough_idx = 0; trough_val = cum[0]
    for i in range(1, len(cum)):
        if cum[i] >= peak:
            if underwater:
                episodes.append({
                    "depth": trough_val / peak - 1,
                    "startIdx": peak_idx, "troughIdx": trough_idx, "endIdx": i,
                    "duration": trough_idx - peak_idx, "recovery": i - trough_idx, "recovered": True,
                })
                underwater = False
            peak = cum[i]; peak_idx = i
        else:
            if not underwater:
                underwater = True; trough_idx = i; trough_val = cum[i]
            if cum[i] < trough_val:
                trough_val = cum[i]; trough_idx = i
    if underwater:
        episodes.append({
            "depth": trough_val / peak - 1,
            "startIdx": peak_idx, "troughIdx": trough_idx, "endIdx": None,
            "duration": trough_idx - peak_idx, "recovery": None, "recovered": False,
        })
    episodes.sort(key=lambda e: e["depth"])
    return episodes


def rolling_vol(returns, window, ppy):
    out = []
    n = len(returns)
    for i in range(window, n + 1):
        seg = returns[i - window:i]
        out.append(float(np.std(seg, ddof=1)) * math.sqrt(ppy) if len(seg) >= 2 else 0.0)
    return out


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        value_col = p.get("value_col")
        if not value_col:
            raise ValueError("Select the price/return column.")
        date_col = p.get("date_col")
        is_returns = bool(p.get("is_returns", False))
        log_ret = bool(p.get("log_returns", False))
        ppy = int(p.get("periods_per_year") or 252)
        rf = float(p.get("rf_annual") or 0)
        mar_annual = float(p.get("mar_annual") or 0)

        df = pd.DataFrame(rows)
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found.")
        if date_col and date_col in df.columns:
            df = df.sort_values(by=date_col, key=lambda s: s.astype(str)).reset_index(drop=True)

        values = pd.to_numeric(df[value_col], errors="coerce")
        values = values[np.isfinite(values)].tolist()
        returns = to_returns(values, is_returns, log_ret)
        if len(returns) < 5:
            raise ValueError("Not enough valid observations to compute risk metrics.")

        mar_period = mar_annual / ppy

        m = float(np.mean(returns))
        ann_return = m * ppy
        ann_vol = float(np.std(returns, ddof=1)) * math.sqrt(ppy) if len(returns) >= 2 else 0.0

        ds_sumsq = 0.0
        for rtn in returns:
            diff = rtn - mar_period
            if diff < 0:
                ds_sumsq += diff * diff
        downside_dev = math.sqrt(ds_sumsq / len(returns)) * math.sqrt(ppy)
        sortino = ((ann_return - rf) / downside_dev) if downside_dev > 0 else (99.0 if ann_return >= rf else -99.0)
        sharpe = ((ann_return - rf) / ann_vol) if ann_vol > 0 else 0.0

        if is_returns:
            cum = to_cumulative(returns)
        else:
            cum_candidate = [v for v in values if np.isfinite(v) and v > 0]
            cum = cum_candidate if len(cum_candidate) > 1 else to_cumulative(returns)

        dd, max_dd, ulcer, pain = max_drawdown_series(cum)
        calmar = (ann_return / abs(max_dd)) if max_dd != 0 else (99.0 if ann_return >= 0 else -99.0)

        neg_returns = [x for x in returns if x < 0]
        pct_negative = len(neg_returns) / len(returns)
        avg_neg_return = float(np.mean(neg_returns)) if neg_returns else float("nan")
        longest_losing_streak = 0
        cur_streak = 0
        for x in returns:
            if x < 0:
                cur_streak += 1
                longest_losing_streak = max(longest_losing_streak, cur_streak)
            else:
                cur_streak = 0

        date_rows = None
        if date_col and date_col in df.columns:
            date_rows = df[date_col].astype(str).tolist()
        chart_data = []
        for i, v in enumerate(dd):
            label = date_rows[min(i, len(date_rows) - 1)] if date_rows else str(i)
            chart_data.append({"idx": i, "label": label, "dd": _fin(v, 6)})

        all_episodes = find_drawdown_episodes(cum)  # sorted worst (most negative) first
        episodes = all_episodes[:5]
        episodes_out = [{
            "depth": _fin(e["depth"], 6), "duration": e["duration"],
            "recovery": e["recovery"], "recovered": e["recovered"],
        } for e in episodes]

        def _label(idx):
            if idx is None:
                return None
            if date_rows is not None:
                return date_rows[min(idx, len(date_rows) - 1)]
            return str(idx)

        rv_window = min(20, max(5, len(returns) // 4))
        rv = rolling_vol(returns, rv_window, ppy)
        rolling_chart_data = [{"idx": rv_window + i, "vol": _fin(v, 6)} for i, v in enumerate(rv)]

        small_sample = len(returns) < ppy

        # ---- plot: underwater (drawdown) curve + rolling volatility ----
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.4), dpi=115)
            xs = list(range(len(dd)))
            ax1.fill_between(xs, [d * 100 for d in dd], 0, color="#fca5a5")
            ax1.plot(xs, [d * 100 for d in dd], color="#dc2626", lw=1)
            ax1.axhline(0, color="#6b7280", lw=0.8)
            ax1.set_ylabel("Drawdown (%)")
            ax1.set_title(f"Underwater Curve (max {max_dd:.1%})")
            ax1.set_xlabel("Period")

            if rv:
                rx = [rv_window + i for i in range(len(rv))]
                ax2.plot(rx, [v * 100 for v in rv], color="#2563eb", lw=1.6)
                ax2.set_ylabel("Annualised volatility (%)")
                ax2.set_xlabel("Period")
                ax2.set_title(f"Rolling {rv_window}-Period Volatility")
            else:
                ax2.axis("off")
                ax2.text(0.5, 0.5, "Not enough data for rolling volatility", ha="center", va="center")

            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all")
            plot = None

        # ══════════════════════════════════════════════════════════════════
        # Additive step-6 full report: 7 sections, each with a table (and
        # most with their own PNG chart), rendered via SortableResultTable /
        # VisualizationTabs on the frontend. Does not alter any field above.
        # ══════════════════════════════════════════════════════════════════
        returns_arr = np.array(returns, dtype=float)
        worst_ep = all_episodes[0] if all_episodes else None

        # ---- ① Risk Summary ----
        worst_loss = float(np.min(returns_arr))
        risk_summary = {
            "maxDrawdown": _fin(max_dd, 6),
            "recoveryPeriod": (worst_ep["recovery"] if worst_ep else None),
            "downsideDeviation": _fin(downside_dev, 6),
            "lossFrequency": _fin(pct_negative, 6),
            "averageLoss": _fin(avg_neg_return, 6),
            "worstLoss": _fin(worst_loss, 6),
            "maxConsecutiveLosses": longest_losing_streak,
        }

        # ---- ② Drawdown Analysis (the worst single episode, in detail) ----
        drawdown_detail = None
        chart_drawdown_curve = None
        if worst_ep is not None:
            start_idx, trough_idx, end_idx = worst_ep["startIdx"], worst_ep["troughIdx"], worst_ep["endIdx"]
            duration_full = (end_idx - start_idx) if end_idx is not None else None  # peak-to-recovery
            drawdown_detail = {
                "maxDrawdown": _fin(worst_ep["depth"], 6),
                "startPeriod": _label(start_idx), "startIdx": start_idx,
                "troughPeriod": _label(trough_idx), "troughIdx": trough_idx,
                "recoveryPeriod": _label(end_idx) if end_idx is not None else None,
                "recoveryIdx": end_idx,
                "recovered": worst_ep["recovered"],
                "duration": duration_full,               # peak -> recovery
                "recoveryTime": worst_ep["recovery"],     # trough -> recovery
            }
            try:
                fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
                xs = list(range(len(dd)))
                ax.fill_between(xs, [d * 100 for d in dd], 0, color=LIGHT_RED, alpha=0.6)
                ax.plot(xs, [d * 100 for d in dd], color=RED, lw=1.2)
                ax.axhline(0, color="#6b7280", lw=0.8)
                ax.scatter([start_idx], [dd[start_idx] * 100], color=GREEN, zorder=5, s=45, label="Peak")
                ax.scatter([trough_idx], [dd[trough_idx] * 100], color=RED, zorder=5, s=45, label="Trough")
                if end_idx is not None:
                    ax.scatter([end_idx], [dd[end_idx] * 100], color=BLUE, zorder=5, s=45, label="Recovery")
                ax.set_title(f"Drawdown Curve (worst episode {worst_ep['depth']:.1%})")
                ax.set_xlabel("Period"); ax.set_ylabel("Drawdown (%)")
                ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_drawdown_curve = _png(fig)
            except Exception:
                plt.close("all"); chart_drawdown_curve = None

        # ---- ③ Historical Drawdowns (top 5, or top 10 if enough episodes) ----
        top_n = 10 if len(all_episodes) >= 10 else 5
        top_eps = all_episodes[:top_n]
        top_drawdowns = [{
            "rank": i + 1,
            "start": _label(e["startIdx"]),
            "trough": _label(e["troughIdx"]),
            "end": _label(e["endIdx"]) if e["endIdx"] is not None else "ongoing",
            "drawdown": _fin(e["depth"], 6),
            "duration": (e["endIdx"] - e["startIdx"]) if e["endIdx"] is not None else e["duration"],
        } for i, e in enumerate(top_eps)]

        chart_top_drawdowns = None
        if top_eps:
            try:
                fig, ax = plt.subplots(figsize=(9, 4.6), dpi=115)
                labels = [f"#{i+1}" for i in range(len(top_eps))]
                vals = [e["depth"] * 100 for e in top_eps]
                ax.bar(labels, vals, color=RED, width=0.6)
                ax.axhline(0, color="#111827", lw=0.7)
                ax.set_title(f"Top {len(top_eps)} Drawdowns")
                ax.set_ylabel("Drawdown (%)")
                ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                chart_top_drawdowns = _png(fig)
            except Exception:
                plt.close("all"); chart_top_drawdowns = None

        # ---- ④ Downside Risk ----
        neg_arr = returns_arr[returns_arr < 0]
        downside_table = {
            "downsideDeviation": _fin(downside_dev, 6),
            "averageLoss": _fin(avg_neg_return, 6),
            "worstLoss": _fin(worst_loss, 6),
            "lossFrequency": _fin(pct_negative, 6),
            "downsideCaptureSkipped": "No benchmark series provided for this analysis — downside capture ratio requires one and is skipped.",
        }
        chart_downside_dist = None
        if len(neg_arr) >= 2:
            try:
                fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
                ax.hist(neg_arr * 100, bins=min(30, max(6, len(neg_arr) // 3)), color=LIGHT_RED, edgecolor="white")
                ax.axvline(0, color="#111827", lw=0.7)
                ax.set_title("Downside Return Distribution (negative periods only)")
                ax.set_xlabel("Return (%)"); ax.set_ylabel("Count")
                ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_downside_dist = _png(fig)
            except Exception:
                plt.close("all"); chart_downside_dist = None

        # ---- ⑤ Loss Analysis ----
        median_loss = float(np.median(neg_arr)) if len(neg_arr) else None
        recoveries_all = [e["recovery"] for e in all_episodes if e["recovered"] and e["recovery"] is not None]
        avg_recovery_all = float(np.mean(recoveries_all)) if recoveries_all else None
        loss_table = {
            "lossPeriods": int(len(neg_arr)),
            "lossFrequency": _fin(pct_negative, 6),
            "averageLoss": _fin(avg_neg_return, 6),
            "medianLoss": _fin(median_loss, 6),
            "worstLoss": _fin(worst_loss, 6),
            "maxConsecutiveLosses": longest_losing_streak,
            "averageRecoveryTime": _fin(avg_recovery_all, 3) if avg_recovery_all is not None else None,
        }
        # Distinct from ④: bucketed bar chart of loss MAGNITUDE (not a continuous
        # histogram) so the two charts are not redundant.
        chart_loss_dist = None
        if len(neg_arr) >= 1:
            try:
                mags = np.abs(neg_arr) * 100
                bins_edges = [0, 1, 2, 5, 10, np.inf]
                bin_labels = ["<1%", "1-2%", "2-5%", "5-10%", ">10%"]
                counts = [int(np.sum((mags >= lo) & (mags < hi))) for lo, hi in zip(bins_edges[:-1], bins_edges[1:])]
                fig, ax = plt.subplots(figsize=(8, 4.6), dpi=115)
                ax.bar(bin_labels, counts, color=AMBER, width=0.6)
                ax.set_title("Loss Distribution (by magnitude)")
                ax.set_xlabel("Loss size"); ax.set_ylabel("Count")
                ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                chart_loss_dist = _png(fig)
            except Exception:
                plt.close("all"); chart_loss_dist = None

        # ---- ⑥ Recovery Analysis ----
        durations_all_recovered = recoveries_all
        median_recovery = float(np.median(durations_all_recovered)) if durations_all_recovered else None
        longest_recovery = max(durations_all_recovered) if durations_all_recovered else None
        n_episodes_total = len(all_episodes)
        n_recovered = sum(1 for e in all_episodes if e["recovered"])
        pct_recovered = (n_recovered / n_episodes_total) if n_episodes_total else None
        currently_in_dd = bool(dd[-1] < 0) if dd else False
        recovery_table = {
            "averageRecoveryTime": _fin(avg_recovery_all, 3) if avg_recovery_all is not None else None,
            "medianRecoveryTime": _fin(median_recovery, 3) if median_recovery is not None else None,
            "longestRecovery": longest_recovery,
            "fullyRecoveredPct": _fin(pct_recovered, 6) if pct_recovered is not None else None,
            "currentlyInDrawdown": currently_in_dd,
        }
        chart_recovery_timeline = None
        if all_episodes:
            try:
                eps_by_start = sorted(all_episodes, key=lambda e: e["startIdx"])
                fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.42 * len(eps_by_start) + 1)), dpi=115)
                ys = np.arange(len(eps_by_start))
                for i, e in enumerate(eps_by_start):
                    decline_len = e["troughIdx"] - e["startIdx"]
                    ax.barh(i, decline_len, left=e["startIdx"], color=RED, height=0.5, label="Decline" if i == 0 else None)
                    if e["recovered"] and e["endIdx"] is not None:
                        ax.barh(i, e["endIdx"] - e["troughIdx"], left=e["troughIdx"], color=GREEN, height=0.5, label="Recovery" if i == 0 else None)
                    else:
                        ax.barh(i, max(1, len(dd) - 1 - e["troughIdx"]), left=e["troughIdx"], color=AMBER, height=0.5, alpha=0.6, label="Ongoing" if i == 0 else None)
                ax.set_yticks(ys)
                ax.set_yticklabels([f"#{i+1}" for i in range(len(eps_by_start))], fontsize=8)
                ax.set_xlabel("Period")
                ax.set_title("Drawdown Recovery Timeline (decline vs. recovery, all episodes)")
                ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=0.2, axis="x")
                fig.tight_layout()
                chart_recovery_timeline = _png(fig)
            except Exception:
                plt.close("all"); chart_recovery_timeline = None

        # ---- ⑦ Tail Risk Profile (exploratory tail shape — NOT VaR/CVaR) ----
        p1 = float(np.percentile(returns_arr, 1))
        p5 = float(np.percentile(returns_arr, 5))
        p10 = float(np.percentile(returns_arr, 10))
        mean_r = float(np.mean(returns_arr))
        std_r = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0
        extreme_thresh = mean_r - 2 * std_r  # "extreme" := beyond 2 std devs below the mean
        extreme_freq = float(np.mean(returns_arr < extreme_thresh))
        tail_table = {
            "worst1Pct": _fin(p1, 6),
            "worst5Pct": _fin(p5, 6),
            "worst10Pct": _fin(p10, 6),
            "extremeLossFrequency": _fin(extreme_freq, 6),
            "_note": "Extreme Loss Frequency = share of periods below (mean - 2*std). Exploratory tail-shape metrics only; formal VaR/CVaR live on the Value-at-Risk page.",
        }
        chart_tail_dist = None
        try:
            tail_mask = returns_arr <= p10
            tail_vals = returns_arr[tail_mask] if np.any(tail_mask) else returns_arr[returns_arr <= np.percentile(returns_arr, 25)]
            if len(tail_vals) >= 2:
                fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
                ax.hist(tail_vals * 100, bins=min(25, max(5, len(tail_vals) // 2)), color=RED, edgecolor="white", alpha=0.85)
                ax.axvline(p1 * 100, color="#111827", ls="--", lw=1, label="1st pct")
                ax.axvline(p5 * 100, color="#374151", ls=":", lw=1, label="5th pct")
                ax.set_title("Left-tail Distribution (returns at/below 10th percentile)")
                ax.set_xlabel("Return (%)"); ax.set_ylabel("Count")
                ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_tail_dist = _png(fig)
        except Exception:
            plt.close("all"); chart_tail_dist = None

        charts = {
            "drawdown_curve": chart_drawdown_curve,
            "top_drawdowns": chart_top_drawdowns,
            "downside_dist": chart_downside_dist,
            "loss_dist": chart_loss_dist,
            "recovery_timeline": chart_recovery_timeline,
            "tail_dist": chart_tail_dist,
        }

        results = {
            "n_returns": len(returns), "value_col": value_col, "freq": str(p.get("freq") or "252"),
            "ppy": ppy, "smallSample": small_sample,
            "annReturn": _fin(ann_return, 6), "annVol": _fin(ann_vol, 6),
            "downsideDev": _fin(downside_dev, 6), "sortino": _fin(sortino, 6), "sharpe": _fin(sharpe, 6),
            "maxDD": _fin(max_dd, 6), "calmar": _fin(calmar, 6), "ulcer": _fin(ulcer, 6), "pain": _fin(pain, 6),
            "pctNegative": _fin(pct_negative, 6), "avgNegReturn": _fin(avg_neg_return, 6),
            "longestLosingStreak": longest_losing_streak,
            "chartData": chart_data, "episodes": episodes_out,
            "rollingChartData": rolling_chart_data, "rvWindow": rv_window,
            "rf": _fin(rf, 6), "mar": _fin(mar_annual, 6),
            "risk_summary": risk_summary,
            "drawdown_detail": drawdown_detail,
            "top_drawdowns": top_drawdowns,
            "downside_table": downside_table,
            "loss_table": loss_table,
            "recovery_table": recovery_table,
            "tail_table": tail_table,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
