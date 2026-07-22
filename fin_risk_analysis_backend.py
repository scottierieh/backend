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


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


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

        episodes = find_drawdown_episodes(cum)[:5]
        episodes_out = [{
            "depth": _fin(e["depth"], 6), "duration": e["duration"],
            "recovery": e["recovery"], "recovered": e["recovered"],
        } for e in episodes]

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
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
