#!/usr/bin/env python3
"""Event Study — market-model abnormal returns around an event. pandas/statsmodels/numpy.

Estimates the normal (expected) return from a market-model regression over an
estimation window, then measures abnormal returns (AR) and cumulative abnormal
returns (CAR) over an event window, with t-tests for significance.

Single event / single asset design:
Event Definition -> Estimation Window -> Event Window -> AR -> CAR -> Significance.

Input (from event-study-page.tsx):
    data              : list[dict]
    asset_col         : str   asset return column
    market_col        : str   market/benchmark return column
    event_index       : int   0-based row index of the event day
    estimation_window : int   periods to estimate the market model (default 120)
    event_pre         : int   periods before the event in the window (default 5)
    event_post        : int   periods after the event in the window (default 5)
    gap               : int   periods between estimation and event window (default 5)
    date_col          : str   optional — column to label the event date in the summary
    return_type       : str   optional — "simple" | "log", echoed back (default "simple")
    event_label       : str   optional — free-text label for the event (e.g. "Earnings"),
                              echoed back for the user's own record-keeping only
Output: { results: { study_summary, event_setup, abnormal_returns, car_table,
                     significance_table, window_comparison, ... }, charts: {...} }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED = "#dc2626"
BLUE = "#93c5fd"
AMBER = "#f59e0b"
GREEN = "#16a34a"


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


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        asset_col = p.get("asset_col"); market_col = p.get("market_col")
        if not asset_col or asset_col not in df.columns:
            raise ValueError("Select the asset return column.")
        if not market_col or market_col not in df.columns:
            raise ValueError("Select the market return column.")
        ev = int(p.get("event_index"))
        est_w = int(p.get("estimation_window") or 120)
        pre = int(p.get("event_pre") or 5)
        post = int(p.get("event_post") or 5)
        gap = int(p.get("gap") or 5)
        date_col = p.get("date_col")
        return_type = (p.get("return_type") or "simple").lower()
        event_label = p.get("event_label") or None

        a = pd.to_numeric(df[asset_col], errors="coerce").values
        m = pd.to_numeric(df[market_col], errors="coerce").values
        n = len(a)
        if not (0 <= ev < n):
            raise ValueError(f"Event index must be between 0 and {n-1}.")

        # ── Event Definition ──
        event_label_str = f"#{ev}"
        if date_col and date_col in df.columns:
            try:
                event_label_str = str(df[date_col].iloc[ev])
            except Exception:
                event_label_str = f"#{ev}"

        event_start = ev - pre
        event_end = ev + post
        est_end = event_start - gap                 # exclusive
        est_start = est_end - est_w
        if est_start < 0:
            raise ValueError(f"Not enough history before the event: need {est_w + gap + pre} rows before the event, "
                             f"have {ev}. Reduce the estimation window or move the event later.")
        if event_end >= n:
            raise ValueError(f"Not enough data after the event: need {post} rows after index {ev}, have {n-1-ev}.")

        # ── Estimation Window: market-model regression (OLS of asset returns on market returns) ──
        ae = a[est_start:est_end]; me = m[est_start:est_end]
        mask = np.isfinite(ae) & np.isfinite(me)
        ae, me = ae[mask], me[mask]
        if len(ae) < 10:
            raise ValueError("Estimation window has too few valid observations (need >= 10).")
        X = sm.add_constant(me)
        model = sm.OLS(ae, X).fit()
        alpha = float(model.params[0]); beta = float(model.params[1])
        resid_sd = float(np.std(model.resid, ddof=2))    # residual std of AR under H0
        M = len(ae)
        me_mean = float(np.mean(me))
        me_ssd = float(np.sum((me - me_mean) ** 2))      # sum of squared deviations, for Patell adjustment

        # ── Event Window: abnormal returns (AR) and cumulative abnormal returns (CAR) ──
        idxs = list(range(event_start, event_end + 1))
        rel = list(range(-pre, post + 1))
        ar_list = []
        car = 0.0
        car_series = []
        sar_list = []  # Patell standardized abnormal returns
        for k, i in enumerate(idxs):
            ai, mi = a[i], m[i]
            normal = alpha + beta * mi
            ar = ai - normal
            car += ar
            t_ar = ar / resid_sd if resid_sd > 0 else np.nan
            p_ar = 2 * (1 - sstats.norm.cdf(abs(t_ar))) if np.isfinite(t_ar) else None
            ar_list.append({
                "rel_day": rel[k], "actual": _fin(ai, 6), "normal": _fin(normal, 6),
                "abnormal_return": _fin(ar, 6), "car": _fin(car, 6),
                "t_stat": _fin(t_ar, 4), "p_value": _fin(p_ar, 6),
                "significant": bool(p_ar is not None and p_ar < 0.05),
            })
            car_series.append(car)
            # Patell (1976): forecast-error adjustment for out-of-sample prediction at mi
            if resid_sd > 0 and me_ssd > 0:
                adj = 1.0 + 1.0 / M + ((mi - me_mean) ** 2) / me_ssd
                sar_list.append(ar / (resid_sd * np.sqrt(adj)))
            else:
                sar_list.append(np.nan)

        L = len(idxs)
        car_total = car
        car_se = resid_sd * np.sqrt(L)
        car_t = car_total / car_se if car_se > 0 else np.nan
        car_p = 2 * (1 - sstats.norm.cdf(abs(car_t))) if np.isfinite(car_t) else None

        # event-day AR (rel day 0) — primary day-level statistic
        ar0 = next((x for x in ar_list if x["rel_day"] == 0), None)

        # ── Statistical Significance: AR t-test (day 0), CAR t-test, Patell Z ──
        significance_table = []
        if ar0 is not None:
            significance_table.append({
                "test": "AR t-test (event day)",
                "statistic": ar0["t_stat"], "p_value": ar0["p_value"],
                "significant": bool(ar0["significant"]),
                "note": "Single-day abnormal return vs. estimation-window residual std.",
            })
        significance_table.append({
            "test": "CAR t-test (event window)",
            "statistic": _fin(car_t, 4), "p_value": _fin(car_p, 6),
            "significant": bool(car_p is not None and car_p < 0.05),
            "note": f"Cumulative AR over {L} days vs. residual std scaled by sqrt({L}).",
        })
        sar_arr = np.array(sar_list, dtype=float)
        sar_valid = sar_arr[np.isfinite(sar_arr)]
        if len(sar_valid) == L and L > 0:
            patell_z = float(np.sum(sar_valid) / np.sqrt(L))
            patell_p = 2 * (1 - sstats.norm.cdf(abs(patell_z))) if np.isfinite(patell_z) else None
            significance_table.append({
                "test": "Patell Z (standardized AR)",
                "statistic": _fin(patell_z, 4), "p_value": _fin(patell_p, 6),
                "significant": bool(patell_p is not None and patell_p < 0.05),
                "note": "Each day's AR standardized by its own forecast-error-adjusted std before summing.",
            })

        # Window comparison: re-run the CAR computation at a few standard event windows,
        # always including the user's own choice, using the same market-model (alpha/beta/resid_sd).
        # Reused for both the Cumulative Abnormal Return sub-windows and the Event Window Comparison.
        def _car_for_window(w_pre, w_post):
            w_start = ev - w_pre
            w_end = ev + w_post
            if w_start < 0 or w_end >= n:
                return None
            w_idxs = list(range(w_start, w_end + 1))
            w_car = 0.0
            for i in w_idxs:
                ai, mi = a[i], m[i]
                normal = alpha + beta * mi
                w_car += (ai - normal)
            w_L = len(w_idxs)
            w_se = resid_sd * np.sqrt(w_L) if resid_sd > 0 else np.nan
            w_t = w_car / w_se if w_se and np.isfinite(w_se) and w_se > 0 else np.nan
            w_p = 2 * (1 - sstats.norm.cdf(abs(w_t))) if np.isfinite(w_t) else None
            return {
                "window": f"[-{w_pre},+{w_post}]", "pre": w_pre, "post": w_post,
                "car": _fin(w_car, 6), "car_t_stat": _fin(w_t, 4), "car_p_value": _fin(w_p, 6),
                "significant": bool(w_p is not None and w_p < 0.05),
            }

        candidate_windows = sorted({(pre, post), (1, 1), (3, 3), (5, 5), (10, 10)}, key=lambda w: w[0] + w[1])
        window_comparison = []
        for w_pre, w_post in candidate_windows:
            row = _car_for_window(w_pre, w_post)
            if row is not None:
                row["is_selected"] = bool(w_pre == pre and w_post == post)
                window_comparison.append(row)

        # ── Section ①: Event Study Summary ──
        study_summary = [{
            "event": event_label_str,
            "estimation_window": f"-{gap + est_w} to -{gap}",
            "event_window": f"-{pre} to +{post}",
            "event_day_ar": _fin(ar0["abnormal_return"], 6) if ar0 else None,
            "car": _fin(car_total, 6),
            "test_statistic": _fin(car_t, 4),
            "p_value": _fin(car_p, 6),
            "significant": bool(car_p is not None and car_p < 0.05),
        }]

        # ── Section ②: Event Setup (restatement of inputs, no new computation) ──
        event_setup = [{
            "event": event_label_str,
            "event_label": event_label,
            "estimation_window": f"-{gap + est_w} to -{gap} ({len(ae)} periods)",
            "event_window": f"-{pre} to +{post} ({L} periods)",
            "benchmark": market_col,
            "expected_return_model": "Market Model (OLS of asset returns on market returns over the estimation window)",
            "return_type": return_type,
        }]

        # ── Section ④: Cumulative Abnormal Return sub-windows (reuses window_comparison computation) ──
        car_table = window_comparison

        plot = None  # legacy combined chart, kept for backward compatibility if present
        charts = {}
        try:
            rels = [x["rel_day"] for x in ar_list]
            ars_pct = [x["abnormal_return"] * 100 for x in ar_list]
            cols = [RED if x["significant"] else BLUE for x in ar_list]

            # Section ③ chart: Abnormal Return Around Event
            fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=115)
            ax.bar(rels, ars_pct, color=cols, alpha=0.9)
            ax.axhline(0, color="#111827", lw=0.8)
            ax.axvline(0, color=AMBER, ls="--", lw=1.3, label="Event day")
            ax.set_xlabel("Days relative to event"); ax.set_ylabel("Abnormal return (%)")
            ax.set_title("Abnormal Return Around Event")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            charts["ar_around_event"] = _png(fig)

            # Section ④ chart: Cumulative Abnormal Return
            fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=115)
            ax.plot(rels, [c * 100 for c in car_series], "o-", color=GREEN, lw=2)
            ax.axhline(0, color="#111827", lw=0.8)
            ax.axvline(0, color=AMBER, ls="--", lw=1.3, label="Event day")
            ax.set_xlabel("Days relative to event"); ax.set_ylabel("Cumulative abnormal return (%)")
            ax.set_title(f"Cumulative Abnormal Return — CAR = {car_total*100:.2f}%")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["car_cumulative"] = _png(fig)

            # Section ⑥ chart: CAR by Event Window
            if window_comparison:
                labels = [row["window"] for row in window_comparison]
                cars_pct = [(row["car"] or 0) * 100 for row in window_comparison]
                bar_cols = [GREEN if row.get("is_selected") else BLUE for row in window_comparison]
                fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
                ax.bar(labels, cars_pct, color=bar_cols)
                ax.axhline(0, color="#111827", lw=0.8)
                ax.set_ylabel("CAR (%)")
                ax.set_title("CAR by Event Window")
                ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                charts["car_by_window"] = _png(fig)

            # legacy combined chart (AR bars + CAR line on twin axes), retained for compatibility
            fig, ax1 = plt.subplots(figsize=(11, 5.2), dpi=120)
            ax1.bar(rels, ars_pct, color=cols, alpha=0.85, label="Abnormal return")
            ax1.axhline(0, color="#111827", lw=0.8)
            ax1.axvline(0, color=AMBER, ls="--", lw=1.2, label="Event day")
            ax1.set_xlabel("Days relative to event"); ax1.set_ylabel("Abnormal return (%)", color="#2563eb")
            ax2 = ax1.twinx()
            ax2.plot(rels, [c * 100 for c in car_series], "o-", color=GREEN, lw=2, label="CAR")
            ax2.set_ylabel("Cumulative abnormal return (%)", color=GREEN)
            ax1.set_title(f"Event study: AR (bars) and CAR (line) — CAR = {car_total*100:.2f}%")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False, loc="best")
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all")

        direction = "positive" if car_total > 0 else "negative"
        sig_txt = ("statistically significant" if (car_p is not None and car_p < 0.05)
                   else "not statistically significant")
        interpretation = (
            f"Using a market model estimated over {len(ae)} pre-event periods (beta {beta:.2f}, alpha {alpha:.4f}), "
            f"the event window of {L} days shows a cumulative abnormal return (CAR) of {car_total:.2%}, which is "
            f"{sig_txt} (t = {car_t:.2f}). "
            + (f"On the event day itself the abnormal return was {ar0['abnormal_return']:.2%} "
               f"(t = {ar0['t_stat']:.2f}). " if ar0 else "")
            + f"A {direction} CAR means the asset {('out' if car_total > 0 else 'under')}performed what the market "
            "model predicted around the event — the market\'s assessment of the event\'s value, net of overall market moves."
        )

        results = {
            "status": "ok", "asset": asset_col, "market": market_col, "event_index": ev,
            "event_label_str": event_label_str, "event_label": event_label, "return_type": return_type,
            "estimation_window": len(ae), "event_pre": pre, "event_post": post, "gap": gap,
            "alpha": _fin(alpha, 6), "beta": _fin(beta, 6), "r_squared": _fin(model.rsquared, 4),
            "resid_sd": _fin(resid_sd, 6),
            "car_total": _fin(car_total, 6), "car_t": _fin(car_t, 4), "car_p": _fin(car_p, 6),
            "car_significant": bool(car_p is not None and car_p < 0.05),
            "event_day_ar": _fin(ar0["abnormal_return"], 6) if ar0 else None,
            "event_day_t": _fin(ar0["t_stat"], 4) if ar0 else None,
            "study_summary": study_summary,
            "event_setup": event_setup,
            "abnormal_returns": ar_list,
            "car_table": car_table,
            "significance_table": significance_table,
            "window_comparison": window_comparison,
            "charts": charts,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
