#!/usr/bin/env python3
"""Event Study — market-model abnormal returns around an event. pandas/statsmodels/numpy.

Estimates the normal (expected) return from a market-model regression over an
estimation window, then measures abnormal returns (AR) and cumulative abnormal
returns (CAR) over an event window, with t-tests for significance.

Input (from event-study-page.tsx):
    data              : list[dict]
    asset_col         : str   asset return column
    market_col        : str   market/benchmark return column
    event_index       : int   0-based row index of the event day
    estimation_window : int   periods to estimate the market model (default 120)
    event_pre         : int   periods before the event in the window (default 5)
    event_post        : int   periods after the event in the window (default 5)
    gap               : int   periods between estimation and event window (default 5)
Output: { results: {ar[], car[], stats}, plot } (AR bars + CAR line).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm
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

        a = pd.to_numeric(df[asset_col], errors="coerce").values
        m = pd.to_numeric(df[market_col], errors="coerce").values
        n = len(a)
        if not (0 <= ev < n):
            raise ValueError(f"Event index must be between 0 and {n-1}.")

        event_start = ev - pre
        event_end = ev + post
        est_end = event_start - gap                 # exclusive
        est_start = est_end - est_w
        if est_start < 0:
            raise ValueError(f"Not enough history before the event: need {est_w + gap + pre} rows before the event, "
                             f"have {ev}. Reduce the estimation window or move the event later.")
        if event_end >= n:
            raise ValueError(f"Not enough data after the event: need {post} rows after index {ev}, have {n-1-ev}.")

        # market-model regression on estimation window
        ae = a[est_start:est_end]; me = m[est_start:est_end]
        mask = np.isfinite(ae) & np.isfinite(me)
        ae, me = ae[mask], me[mask]
        if len(ae) < 10:
            raise ValueError("Estimation window has too few valid observations (need >= 10).")
        X = sm.add_constant(me)
        model = sm.OLS(ae, X).fit()
        alpha = float(model.params[0]); beta = float(model.params[1])
        resid_sd = float(np.std(model.resid, ddof=2))    # residual std of AR under H0

        # event window abnormal returns
        idxs = list(range(event_start, event_end + 1))
        rel = list(range(-pre, post + 1))
        ar_list = []
        car = 0.0
        car_series = []
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

        L = len(idxs)
        car_total = car
        car_se = resid_sd * np.sqrt(L)
        car_t = car_total / car_se if car_se > 0 else np.nan
        car_p = 2 * (1 - sstats.norm.cdf(abs(car_t))) if np.isfinite(car_t) else None

        # event-day AR (rel day 0)
        ar0 = next((x for x in ar_list if x["rel_day"] == 0), None)

        # Window comparison: re-run the CAR computation at a few standard event windows,
        # always including the user's own choice, using the same market-model (alpha/beta/resid_sd).
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

        candidate_windows = sorted({(pre, post), (1, 1), (5, 5), (10, 10)}, key=lambda w: w[0] + w[1])
        window_comparison = []
        for w_pre, w_post in candidate_windows:
            row = _car_for_window(w_pre, w_post)
            if row is not None:
                row["is_selected"] = bool(w_pre == pre and w_post == post)
                window_comparison.append(row)

        plot = None
        try:
            fig, ax1 = plt.subplots(figsize=(11, 5.2), dpi=120)
            rels = [x["rel_day"] for x in ar_list]
            ars = [x["abnormal_return"] * 100 for x in ar_list]
            cols = ["#dc2626" if x["significant"] else "#93c5fd" for x in ar_list]
            ax1.bar(rels, ars, color=cols, alpha=0.85, label="Abnormal return")
            ax1.axhline(0, color="#111827", lw=0.8)
            ax1.axvline(0, color="#f59e0b", ls="--", lw=1.2, label="Event day")
            ax1.set_xlabel("Days relative to event"); ax1.set_ylabel("Abnormal return (%)", color="#2563eb")
            ax2 = ax1.twinx()
            ax2.plot(rels, [c * 100 for c in car_series], "o-", color="#16a34a", lw=2, label="CAR")
            ax2.set_ylabel("Cumulative abnormal return (%)", color="#16a34a")
            ax1.set_title(f"Event study: AR (bars) and CAR (line) — CAR = {car_total*100:.2f}%")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False, loc="best")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

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
            "estimation_window": len(ae), "event_pre": pre, "event_post": post, "gap": gap,
            "alpha": _fin(alpha, 6), "beta": _fin(beta, 6), "r_squared": _fin(model.rsquared, 4),
            "resid_sd": _fin(resid_sd, 6),
            "car_total": _fin(car_total, 6), "car_t": _fin(car_t, 4), "car_p": _fin(car_p, 6),
            "car_significant": bool(car_p is not None and car_p < 0.05),
            "event_day_ar": _fin(ar0["abnormal_return"], 6) if ar0 else None,
            "event_day_t": _fin(ar0["t_stat"], 4) if ar0 else None,
            "abnormal_returns": ar_list,
            "window_comparison": window_comparison,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
