#!/usr/bin/env python3
"""Return Time Series — descriptive analysis of a financial return series.
numpy / pandas / scipy.

Summarises the distribution (moments, skew, kurtosis), annualised return and
volatility, cumulative performance, drawdown, rolling volatility, and normality
(steps 1-5, unchanged), plus a step-6 report with the sections that are
DISTINCT from the Returns & Volatility page: the raw return time series,
rolling return at multiple windows, a calendar / monthly-return heatmap,
return streak analysis, outlier detection via z-scores, and a one-line
autocorrelation teaser. Sections that would duplicate Returns & Volatility
(period-return-by-frequency, lookback-window table, best/worst-streak-only
pos/neg breakdown) are intentionally NOT rebuilt here.

Input (from return-time-series-page.tsx):
    data             : list[dict]
    asset_col        : str
    date_col         : str | null   (optional, enables calendar heatmap)
    is_returns       : bool
    return_type      : "simple"|"log"
    periods_per_year : int   (default 252)
    roll_window      : int   (default 21) rolling volatility window (steps 1-5 chart)
Output: { results: {...}, plot } (cumulative + rolling vol + histogram, plus
         results.charts: {return_series, cumulative, distribution,
         rolling_return, calendar_heatmap, pos_neg, streak_timeline, outliers}).
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


def _streak_lengths(mask):
    """All run-lengths of consecutive True values in a boolean array."""
    lengths = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    return lengths


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        col = p.get("asset_col")
        date_col = p.get("date_col") or None
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        roll = int(p.get("roll_window") or 21)
        if not col or col not in df.columns:
            raise ValueError("Select the return column.")

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
        median_ret = float(np.median(rv))
        # drawdown
        wealth = np.cumprod(1 + rv); peak = np.maximum.accumulate(wealth)
        max_dd = float((wealth / peak - 1).min())
        # normality (Jarque-Bera)
        jb, jb_p = sstats.jarque_bera(rv)
        var5 = float(np.percentile(rv, 5))

        roll = max(2, min(roll, n // 2))
        roll_vol = pd.Series(rv).rolling(roll).std(ddof=1) * np.sqrt(ppy)

        # Original 3-panel plot (kept intact — other frontend steps rely on it).
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

        # ══════════════════════ Step-6 report: distinctive sections only ══════════════════════

        # ① Return Summary (brief) — companion to steps 1-5 output above, not a rebuild of R&V's table.
        return_summary = {
            "n_obs": n,
            "periods_per_year": ppy,
            "mean_return": _fin(mean, 6),
            "median_return": _fin(median_ret, 6),
            "annualized_return": _fin(ann_ret, 5),
            "std_deviation": _fin(sd, 6),
            "minimum_return": _fin(worst, 5),
            "maximum_return": _fin(best, 5),
            "positive_ratio": _fin(pos, 4),
            "negative_ratio": _fin(1 - pos, 4),
        }

        # ② Return Time Series — raw day-by-day return line (genuinely new).
        chart_return_series = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=115)
            xaxis = dates if dates is not None else np.arange(n)
            colors = np.where(rv >= 0, GREEN, RED)
            ax.bar(xaxis, rv * 100, color=colors, width=1.0 if dates is None else 1.0)
            ax.axhline(0, color="#111827", lw=0.8)
            ax.set_title("Return Time Series")
            ax.set_xlabel("Date" if dates is not None else "Period")
            ax.set_ylabel("Return (%)")
            ax.grid(alpha=0.2, axis="y")
            if dates is not None:
                fig.autofmt_xdate()
            fig.tight_layout()
            chart_return_series = _png(fig)
        except Exception:
            plt.close("all"); chart_return_series = None

        # ③ Cumulative Return — simple companion line (kept simple, no lookback-window table).
        chart_cumulative = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=115)
            xaxis = dates if dates is not None else np.arange(n)
            ax.plot(xaxis, (wealth - 1) * 100, color=BLUE, lw=1.5)
            ax.axhline(0, color="#111827", lw=0.7, ls="--")
            ax.set_title("Cumulative Return (%)")
            ax.set_xlabel("Date" if dates is not None else "Period")
            ax.set_ylabel("Cumulative Return (%)")
            ax.grid(alpha=0.2)
            if dates is not None:
                fig.autofmt_xdate()
            fig.tight_layout()
            chart_cumulative = _png(fig)
        except Exception:
            plt.close("all"); chart_cumulative = None

        # ④ Return Distribution — brief (mean/median/std/skew/kurt/JB), a legitimate complement
        # to R&V's distribution_stats (which excludes std-dev and JB).
        distribution_stats = {
            "mean": _fin(mean, 6), "median": _fin(median_ret, 6), "std_deviation": _fin(sd, 6),
            "skewness": _fin(skew, 4), "kurtosis": _fin(kurt, 4),
            "jarque_bera": _fin(float(jb), 4), "jb_p_value": _fin(float(jb_p), 6), "is_normal": bool(jb_p >= 0.05),
        }
        chart_distribution = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=115)
            ax.hist(rv * 100, bins=min(50, max(10, n // 8)), color=LIGHT_BLUE, edgecolor="white", density=True)
            xs = np.linspace(rv.min(), rv.max(), 100)
            ax.plot(xs * 100, sstats.norm.pdf(xs, mean, sd) / 100, color=RED, lw=1.5, label="Normal")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title(f"Return Distribution (JB p={jb_p:.3f})")
            ax.set_xlabel("Return (%)")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_distribution = _png(fig)
        except Exception:
            plt.close("all"); chart_distribution = None

        # ⑤ Rolling Return — GENUINELY NEW. Windows scaled to periods_per_year: the 30/90/252-day
        # convention is for daily (ppy=252) data; for other frequencies we scale proportionally
        # (e.g. weekly ppy=52 -> ~6/18/52), then drop anything >= n/2.
        base_windows_daily = [30, 90, 252]
        scale = ppy / 252.0
        candidate_windows = sorted(set(max(2, int(round(w * scale))) for w in base_windows_daily))
        rolling_windows = [w for w in candidate_windows if w < n // 2] or ([candidate_windows[0]] if candidate_windows and candidate_windows[0] < n else [])
        rolling_return_table = []
        rolling_series = {}
        for w in rolling_windows:
            rr = pd.Series(rv).rolling(w).apply(lambda x: np.prod(1 + x) - 1.0, raw=True)
            rolling_series[w] = rr
            valid = rr.dropna()
            if len(valid) == 0:
                continue
            rolling_return_table.append({
                "window": f"{w}p",
                "current": _fin(float(valid.iloc[-1]), 5),
                "mean": _fin(float(valid.mean()), 5),
                "min": _fin(float(valid.min()), 5),
                "max": _fin(float(valid.max()), 5),
            })

        chart_rolling_return = None
        if rolling_series:
            try:
                fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=115)
                xaxis = dates if dates is not None else np.arange(n)
                palette = [BLUE, AMBER, GREEN, RED]
                for i, (w, rr) in enumerate(rolling_series.items()):
                    ax.plot(xaxis, rr.values * 100, lw=1.3, label=f"{w}p", color=palette[i % len(palette)])
                ax.axhline(0, color="#111827", lw=0.7)
                ax.set_title("Rolling Return")
                ax.set_xlabel("Date" if dates is not None else "Period")
                ax.set_ylabel("Rolling Return (%)")
                ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=0.2)
                if dates is not None:
                    fig.autofmt_xdate()
                fig.tight_layout()
                chart_rolling_return = _png(fig)
            except Exception:
                plt.close("all"); chart_rolling_return = None

        # ⑥ Calendar Return — GENUINELY NEW, only if a usable date column is available.
        calendar_return_table = None
        calendar_return_note = None
        chart_calendar_heatmap = None
        if dates is not None:
            wealth_s = pd.Series(wealth, index=pd.DatetimeIndex(dates))
            ret_s = pd.Series(rv, index=pd.DatetimeIndex(dates))
            yearly = wealth_s.resample("YE").last().dropna()
            if len(yearly) >= 1:
                yearly_first = wealth_s.resample("YE").first().dropna()
                yr_rets = []
                prev_wealth = 1.0
                for idx in yearly.index:
                    yr_rets.append({"year": int(idx.year), "return": _fin(float(yearly[idx] / prev_wealth - 1.0), 5)})
                    prev_wealth = float(yearly[idx])
                calendar_return_table = yr_rets

                # Monthly return heatmap: rows=years, cols=Jan-Dec, cell=that month's simple return.
                try:
                    monthly_wealth = wealth_s.resample("ME").last()
                    monthly_ret = monthly_wealth.pct_change()
                    # first month's return relative to period start
                    if len(monthly_wealth) > 0:
                        monthly_ret.iloc[0] = monthly_wealth.iloc[0] - 1.0
                    monthly_ret = monthly_ret.dropna()
                    years = sorted(set(idx.year for idx in monthly_ret.index))
                    months = list(range(1, 13))
                    grid = np.full((len(years), 12), np.nan)
                    for idx, v in monthly_ret.items():
                        yi = years.index(idx.year)
                        grid[yi, idx.month - 1] = v * 100
                    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 * len(years) + 1.2)), dpi=115)
                    vmax = np.nanmax(np.abs(grid)) if np.any(~np.isnan(grid)) else 1.0
                    vmax = vmax if vmax > 0 else 1.0
                    im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
                    ax.set_xticks(range(12))
                    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                    ax.set_yticks(range(len(years)))
                    ax.set_yticklabels(years)
                    for yi in range(len(years)):
                        for mi in range(12):
                            val = grid[yi, mi]
                            if not np.isnan(val):
                                ax.text(mi, yi, f"{val:.1f}", ha="center", va="center", fontsize=7,
                                        color="white" if abs(val) > vmax * 0.6 else "#111827")
                    ax.set_title("Monthly Return Heatmap (%)")
                    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                    fig.tight_layout()
                    chart_calendar_heatmap = _png(fig)
                except Exception:
                    plt.close("all"); chart_calendar_heatmap = None
            else:
                calendar_return_note = "Not enough distinct years in the date range to compute calendar returns."
        else:
            calendar_return_note = "A date column is needed to build the calendar return table and monthly heatmap; none was provided."

        # ⑦ Positive / Negative Return — brief, reuses the same style of stats as R&V's pos_neg_stats
        # but without streak info (that is section ⑧'s job here).
        pos_mask = rv > 0
        neg_mask = rv < 0
        n_pos = int(np.sum(pos_mask)); n_neg = int(np.sum(neg_mask))
        avg_gain = float(np.mean(rv[pos_mask])) if n_pos else None
        avg_loss = float(np.mean(rv[neg_mask])) if n_neg else None
        max_gain = float(np.max(rv[pos_mask])) if n_pos else None
        max_loss = float(np.min(rv[neg_mask])) if n_neg else None
        pos_neg_stats = {
            "positive": {"number": n_pos, "frequency": _fin(n_pos / n, 4), "average": _fin(avg_gain, 6), "maximum": _fin(max_gain, 5)},
            "negative": {"number": n_neg, "frequency": _fin(n_neg / n, 4), "average": _fin(avg_loss, 6), "maximum": _fin(max_loss, 5)},
        }
        chart_pos_neg = None
        try:
            fig, ax = plt.subplots(figsize=(6, 4.4), dpi=115)
            ax.bar(["Positive", "Negative"], [n_pos, n_neg], color=[GREEN, RED], width=0.55)
            ax.set_title("Positive vs Negative Periods")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_pos_neg = _png(fig)
        except Exception:
            plt.close("all"); chart_pos_neg = None

        # ⑧ Return Streak Analysis — GENUINELY NEW: all streak lengths, not just the longest.
        pos_streaks = _streak_lengths(pos_mask)
        neg_streaks = _streak_lengths(neg_mask)
        streak_stats = {
            "longest_positive_streak": int(max(pos_streaks)) if pos_streaks else 0,
            "longest_negative_streak": int(max(neg_streaks)) if neg_streaks else 0,
            "average_positive_streak": _fin(float(np.mean(pos_streaks)), 3) if pos_streaks else None,
            "average_negative_streak": _fin(float(np.mean(neg_streaks)), 3) if neg_streaks else None,
        }
        chart_streak_timeline = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=115)
            xaxis = dates if dates is not None else np.arange(n)
            colors = np.where(pos_mask, GREEN, np.where(neg_mask, RED, "#9ca3af"))
            ax.bar(xaxis, np.ones(n), color=colors, width=1.0)
            ax.set_yticks([])
            ax.set_title("Return Streak Timeline (green = positive run, red = negative run)")
            ax.set_xlabel("Date" if dates is not None else "Period")
            if dates is not None:
                fig.autofmt_xdate()
            fig.tight_layout()
            chart_streak_timeline = _png(fig)
        except Exception:
            plt.close("all"); chart_streak_timeline = None

        # ⑨ Outlier Return Analysis — GENUINELY NEW: z-scores, top-N by |z|.
        z_scores = (rv - mean) / sd if sd > 0 else np.zeros_like(rv)
        outlier_n = min(10, n)
        order = np.argsort(-np.abs(z_scores))[:outlier_n]
        outlier_table = []
        for i in order:
            az = abs(float(z_scores[i]))
            cls = "Extreme" if az > 3 else "Notable" if az > 2 else None
            if cls is None:
                continue
            label = str(dates.iloc[i].date()) if dates is not None else f"Period {i + 1}"
            outlier_table.append({
                "date": label,
                "return": _fin(float(rv[i]), 5),
                "z_score": _fin(float(z_scores[i]), 3),
                "classification": cls,
            })
        outlier_table.sort(key=lambda row: -abs(row["z_score"]))

        chart_outliers = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=115)
            xaxis = dates if dates is not None else np.arange(n)
            extreme_mask = np.abs(z_scores) > 3
            notable_mask = (np.abs(z_scores) > 2) & ~extreme_mask
            normal_mask = ~extreme_mask & ~notable_mask
            xa = np.asarray(xaxis)
            ax.scatter(xa[normal_mask], rv[normal_mask] * 100, color="#9ca3af", s=10, label="Normal")
            ax.scatter(xa[notable_mask], rv[notable_mask] * 100, color=AMBER, s=22, label="Notable (|z|>2)")
            ax.scatter(xa[extreme_mask], rv[extreme_mask] * 100, color=RED, s=32, label="Extreme (|z|>3)")
            ax.axhline(0, color="#111827", lw=0.6)
            ax.set_title("Return Outliers")
            ax.set_xlabel("Date" if dates is not None else "Period")
            ax.set_ylabel("Return (%)")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.2)
            if dates is not None:
                fig.autofmt_xdate()
            fig.tight_layout()
            chart_outliers = _png(fig)
        except Exception:
            plt.close("all"); chart_outliers = None

        # ⑪ Autocorrelation preview — one-line teaser only, NOT a full ACF/PACF analysis.
        if n > 2:
            lag1 = float(np.corrcoef(rv[:-1], rv[1:])[0, 1])
        else:
            lag1 = None
        if lag1 is not None and np.isfinite(lag1):
            strength = "notable" if abs(lag1) > 0.2 else "weak"
            acf_preview_note = (
                f"Lag-1 autocorrelation is {lag1:.3f} ({strength}) — see the Autocorrelation Analysis page "
                "for full ACF/PACF diagnostics and formal independence tests."
            )
        else:
            acf_preview_note = "Lag-1 autocorrelation could not be computed — see the Autocorrelation Analysis page for full diagnostics."

        charts = {
            "return_series": chart_return_series,
            "cumulative": chart_cumulative,
            "distribution": chart_distribution,
            "rolling_return": chart_rolling_return,
            "calendar_heatmap": chart_calendar_heatmap,
            "pos_neg": chart_pos_neg,
            "streak_timeline": chart_streak_timeline,
            "outliers": chart_outliers,
        }

        results = {
            "status": "ok", "asset": col, "n_obs": n, "periods_per_year": ppy, "roll_window": roll,
            "mean": _fin(mean, 6), "std": _fin(sd, 6), "skew": _fin(skew, 4), "excess_kurtosis": _fin(kurt, 4),
            "annual_return": _fin(ann_ret, 5), "annual_volatility": _fin(ann_vol, 5), "sharpe_naive": _fin(sharpe0, 4),
            "cumulative_return": _fin(cum, 5), "pct_positive": _fin(pos, 4),
            "best_period": _fin(best, 5), "worst_period": _fin(worst, 5), "max_drawdown": _fin(max_dd, 5),
            "var_5pct": _fin(var5, 5),
            "jarque_bera": _fin(float(jb), 4), "jb_p_value": _fin(float(jb_p), 6), "is_normal": bool(jb_p >= 0.05),
            "interpretation": interpretation,
            "has_date_col": dates is not None,
            "return_summary": return_summary,
            "distribution_stats": distribution_stats,
            "rolling_windows": rolling_windows,
            "rolling_return_table": rolling_return_table,
            "calendar_return_table": calendar_return_table,
            "calendar_return_note": calendar_return_note,
            "pos_neg_stats": pos_neg_stats,
            "streak_stats": streak_stats,
            "outlier_table": outlier_table,
            "acf_preview_note": acf_preview_note,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
