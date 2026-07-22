#!/usr/bin/env python3
"""Returns & Volatility — risk-return profile of a price/return series.
numpy / pandas / scipy.

Computes annualised return & volatility, Sharpe/Sortino/Calmar, drawdown,
VaR/CVaR, distribution moments, and a rolling-return/volatility view.

Also builds a full step-6 report with 6 additive sections (return summary,
cumulative performance, period return, annual return, return distribution,
positive/negative return) each with its own chart (5 separate PNGs total),
rendered by the frontend via VisualizationTabs.

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
          interpretation, return_summary, lookback_table, period_return_table,
          annual_return_table, distribution_stats, pos_neg_stats,
          charts: {cumulative, period_return, annual_return, distribution, pos_neg} }, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
LIGHT_BLUE = "#93c5fd"


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


def _longest_streak(mask):
    """Longest run of consecutive True values in a boolean array."""
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        col = p.get("value_col")
        date_col = p.get("date_col") or None
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        rf_annual = float(p.get("rf_annual") or 0)
        if not col or col not in df.columns:
            raise ValueError("Select the price or return column.")

        s = pd.to_numeric(df[col], errors="coerce")

        if is_returns:
            ret_mask = s.notna()
            ret = s[ret_mask]
        elif rtype == "log":
            if (s <= 0).any():
                raise ValueError("Log returns require positive prices.")
            ret_full = np.log(s / s.shift(1))
            ret_mask = ret_full.notna()
            ret = ret_full[ret_mask]
        else:
            ret_full = (s / s.shift(1) - 1.0)
            ret_mask = ret_full.notna()
            ret = ret_full[ret_mask]

        # Optional date index, aligned positionally to the (masked) return series.
        dates = None
        if date_col and date_col in df.columns:
            try:
                dates_full = pd.to_datetime(df[date_col], errors="coerce").reset_index(drop=True)
                dates_aligned = dates_full[ret_mask.reset_index(drop=True).values].reset_index(drop=True)
                if not dates_aligned.isna().any() and len(dates_aligned) == len(ret):
                    dates = dates_aligned
            except Exception:
                dates = None

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

        # Original 3-panel plot (kept intact — other frontend parts rely on it).
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
            plot = _png(fig)
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

        # ─────────────────────────── Section 1: Return Summary ───────────────────────────
        pos_mask = rv > 0
        neg_mask = rv < 0
        n_pos = int(np.sum(pos_mask)); n_neg = int(np.sum(neg_mask))
        avg_gain = float(np.mean(rv[pos_mask])) if n_pos else None
        avg_loss = float(np.mean(rv[neg_mask])) if n_neg else None
        gain_loss_ratio = (avg_gain / abs(avg_loss)) if (avg_gain is not None and avg_loss not in (None, 0)) else None
        median_ret = float(np.median(rv))

        return_summary = {
            "total_return": _fin(cum, 5),
            "annualized_return": _fin(ann_ret, 5),
            "average_return": _fin(mean, 6),
            "median_return": _fin(median_ret, 6),
            "best_return": _fin(best, 5),
            "worst_return": _fin(worst, 5),
            "positive_periods": n_pos,
            "negative_periods": n_neg,
            "win_rate": _fin(pos_pct, 2),
            "gain_loss_ratio": _fin(gain_loss_ratio, 3),
        }

        # ─────────────────────────── Section 2: Cumulative Performance ───────────────────────────
        growth100 = wealth * 100.0
        lookback_defs = [("1M", ppy / 12.0), ("3M", ppy / 4.0), ("6M", ppy / 2.0), ("1Y", float(ppy)), ("3Y", ppy * 3.0)]
        lookback_table = []
        for label, wlen in lookback_defs:
            wlen_i = int(round(wlen))
            if wlen_i < 1 or wlen_i >= n:
                continue
            window_ret = float(wealth[-1] / wealth[-1 - wlen_i] - 1.0)
            lookback_table.append({
                "window": label,
                "return": _fin(window_ret, 5),
                "cumulative_return": _fin(window_ret, 5),
            })

        chart_cumulative = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
            xaxis = dates if dates is not None else np.arange(n)
            ax.plot(xaxis, growth100, color=BLUE, lw=1.6)
            ax.axhline(100, color="#111827", lw=0.7, ls="--")
            ax.set_title("Cumulative Performance — Growth of $100")
            ax.set_xlabel("Date" if dates is not None else "Period")
            ax.set_ylabel("Value ($)")
            ax.grid(alpha=0.2)
            if dates is not None:
                fig.autofmt_xdate()
            fig.tight_layout()
            chart_cumulative = _png(fig)
        except Exception:
            plt.close("all"); chart_cumulative = None

        # ─────────────────────────── Section 3: Period Return ───────────────────────────
        freq_defs = []
        if dates is not None:
            freq_defs = [("Daily", "D"), ("Weekly", "W"), ("Monthly", "ME"), ("Quarterly", "QE"), ("Yearly", "YE")]
        else:
            bucket_map = {"Daily": 1, "Weekly": max(1, ppy // 52), "Monthly": max(1, ppy // 12),
                          "Quarterly": max(1, ppy // 4), "Yearly": max(1, ppy)}

        period_return_table = {}
        if dates is not None:
            wealth_s = pd.Series(wealth, index=pd.DatetimeIndex(dates))
            for label, rule in freq_defs:
                try:
                    grp = wealth_s.resample(rule).last()
                    grp = grp.dropna()
                    if len(grp) < 2:
                        continue
                    period_rets = grp.pct_change().dropna()
                    rows_out = [{"period": str(idx.date() if hasattr(idx, "date") else idx), "return": _fin(float(v), 5)}
                                for idx, v in period_rets.items()]
                    if rows_out:
                        period_return_table[label] = rows_out
                except Exception:
                    continue
        else:
            for label, bsize in bucket_map.items():
                nb = n // bsize
                if nb < 2:
                    continue
                idxs = [wealth[min((i + 1) * bsize, n) - 1] for i in range(nb)]
                base = 1.0
                rows_out = []
                for i, w in enumerate(idxs):
                    r = float(w / base - 1.0)
                    rows_out.append({"period": f"{label} bucket {i+1}", "return": _fin(r, 5)})
                    base = w
                if rows_out:
                    period_return_table[label] = rows_out

        default_freq = next((f for f in ["Monthly", "Weekly", "Quarterly", "Daily", "Yearly"] if f in period_return_table), None)

        chart_period_return = None
        if default_freq and period_return_table.get(default_freq):
            try:
                pr = period_return_table[default_freq]
                vals = [row["return"] * 100 if row["return"] is not None else 0 for row in pr]
                labels = [row["period"] for row in pr]
                colors = [GREEN if v >= 0 else RED for v in vals]
                fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
                xs = np.arange(len(vals))
                ax.bar(xs, vals, color=colors, width=0.8)
                ax.axhline(0, color="#111827", lw=0.7)
                ax.set_title(f"Period Return ({default_freq})")
                ax.set_ylabel("Return (%)")
                step = max(1, len(xs) // 15)
                ax.set_xticks(xs[::step])
                ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right", fontsize=7)
                ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                chart_period_return = _png(fig)
            except Exception:
                plt.close("all"); chart_period_return = None

        # ─────────────────────────── Section 4: Annual Return ───────────────────────────
        annual_return_table = None
        annual_return_note = None
        chart_annual_return = None
        if dates is not None:
            wealth_s = pd.Series(wealth, index=pd.DatetimeIndex(dates))
            yearly = wealth_s.resample("YE").last().dropna()
            if len(yearly) >= 2:
                yr_rets = yearly.pct_change().dropna()
                annual_return_table = [{"year": int(idx.year), "return": _fin(float(v), 5)} for idx, v in yr_rets.items()]
                try:
                    vals = [row["return"] * 100 if row["return"] is not None else 0 for row in annual_return_table]
                    labels = [str(row["year"]) for row in annual_return_table]
                    colors = [GREEN if v >= 0 else RED for v in vals]
                    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
                    xs = np.arange(len(vals))
                    ax.bar(xs, vals, color=colors, width=0.6)
                    ax.axhline(0, color="#111827", lw=0.7)
                    ax.set_title("Annual Return")
                    ax.set_ylabel("Return (%)")
                    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=0)
                    ax.grid(alpha=0.2, axis="y")
                    fig.tight_layout()
                    chart_annual_return = _png(fig)
                except Exception:
                    plt.close("all"); chart_annual_return = None
            else:
                annual_return_note = "Not enough distinct years in the date range to compute annual returns."
        else:
            annual_return_note = "A date column is needed to compute year-by-year returns; none was provided."

        # ─────────────────────────── Section 5: Return Distribution ───────────────────────────
        q1 = float(np.percentile(rv, 25))
        q3 = float(np.percentile(rv, 75))
        distribution_stats = {
            "mean": _fin(mean, 6), "median": _fin(median_ret, 6),
            "minimum": _fin(worst, 5), "maximum": _fin(best, 5),
            "q1": _fin(q1, 6), "q3": _fin(q3, 6),
            "skewness": _fin(skew, 4), "kurtosis": _fin(kurt, 4),
        }
        chart_distribution = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.hist(rv * 100, bins=min(50, max(10, n // 8)), color=LIGHT_BLUE, edgecolor="white", density=True)
            if n > 3:
                xs = np.linspace(rv.min(), rv.max(), 100)
                ax.plot(xs * 100, sstats.norm.pdf(xs, mean, sd) / 100, color=RED, lw=1.5, label="Normal")
                ax.legend(fontsize=8, frameon=False)
            ax.set_title("Return Distribution")
            ax.set_xlabel("Return (%)")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_distribution = _png(fig)
        except Exception:
            plt.close("all"); chart_distribution = None

        # ─────────────────────────── Section 6: Positive / Negative Return ───────────────────────────
        best_streak = _longest_streak(pos_mask)
        worst_streak = _longest_streak(neg_mask)
        pos_neg_stats = {
            "positive_periods": n_pos, "negative_periods": n_neg,
            "win_rate": _fin(pos_pct, 2),
            "average_gain": _fin(avg_gain, 6), "average_loss": _fin(avg_loss, 6),
            "best_streak": best_streak, "worst_streak": worst_streak,
        }
        chart_pos_neg = None
        try:
            fig, ax = plt.subplots(figsize=(6, 4.6), dpi=115)
            ax.bar(["Positive", "Negative"], [n_pos, n_neg], color=[GREEN, RED], width=0.55)
            ax.set_title("Positive vs Negative Periods")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_pos_neg = _png(fig)
        except Exception:
            plt.close("all"); chart_pos_neg = None

        charts = {
            "cumulative": chart_cumulative,
            "period_return": chart_period_return,
            "annual_return": chart_annual_return,
            "distribution": chart_distribution,
            "pos_neg": chart_pos_neg,
        }

        results = {
            "status": "ok", "n_obs": n, "value_col": col, "is_returns": is_returns, "return_type": rtype,
            "periods_per_year": ppy, "rf_annual": _fin(rf_annual, 4), "roll_window": roll,
            "summary": summary, "interpretation": interpretation,
            "return_summary": return_summary,
            "lookback_table": lookback_table,
            "period_return_table": period_return_table,
            "period_return_frequencies": list(period_return_table.keys()),
            "default_period_frequency": default_freq,
            "annual_return_table": annual_return_table,
            "annual_return_note": annual_return_note,
            "distribution_stats": distribution_stats,
            "pos_neg_stats": pos_neg_stats,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
