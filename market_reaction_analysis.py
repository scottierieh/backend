#!/usr/bin/env python3
"""Market Reaction Analysis — how fast and asymmetrically an asset reacts to the
market. statsmodels.

Two lenses:
  1. Distributed-lag regression of the asset on contemporaneous and lagged market
     returns -> immediate vs delayed reaction (price-discovery speed / lead-lag).
  2. Up/down asymmetry: separate betas in rising vs falling markets.

Also builds a full step-6 report with 8 additive "reaction" sections (NOT formal
AR/CAR — this page never computes abnormal-return test statistics; that is the
job of Event Study / Abnormal Return Analysis). Two different windowing schemes
are reused rather than invented:
  - Sections that talk about "lags" (Price Reaction, Reaction Speed, Reaction
    Reversal) reuse the existing distributed-lag structure itself: the
    cumulative sum of the lag-0..lag-n betas *is* the reaction building up over
    time after the market moves it. There is no negative-lag term in this
    design (the regression only has terms for the market move itself and its
    lags), so the "pre-event" point on that curve is defined as the baseline
    (zero) the cumulative reaction starts from — documented in `_note` fields.
  - Sections that talk about actual trading activity (Volume, Volatility,
    Market vs Asset) use a chronological three-way split of the aligned sample
    into equal Pre / Event / Post thirds (extending the previous two-bucket
    before/after split used by volume_reaction / volatility_reaction).

Input (from market-reaction-page.tsx):
    data        : list[dict]
    asset_col   : str
    market_col  : str
    is_returns  : bool
    return_type : "simple"|"log"
    n_lags      : int   (default 3)
    volume_col  : str (optional)
    group_col   : str (optional)
Output: { results: {..., reaction_summary, price_reaction_table, speed_table,
          volume_table, volatility_table, relative_table, reversal_table,
          absorption, charts: {...}}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
GRAY = "#94a3b8"


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


def _ret(s, is_returns, rtype):
    s = pd.to_numeric(s, errors="coerce")
    if is_returns:
        return s
    if rtype == "log":
        return np.log(s / s.shift(1))
    return s / s.shift(1) - 1.0


def _thirds(n):
    """Split n aligned observations into (pre, event, post) index ranges."""
    third = max(1, n // 3)
    pre = (0, third)
    event = (third, 2 * third)
    post = (2 * third, n)
    return pre, event, post


def _impact_label(value, ref, hi_mult=2.0, lo_mult=1.0):
    """Qualitative Low/Medium/High vs a reference scale. None-safe."""
    if value is None or ref is None or ref == 0:
        return "N/A"
    av = abs(value)
    if av > hi_mult * abs(ref):
        return "High"
    if av > lo_mult * abs(ref):
        return "Medium"
    return "Low"


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        asset_col = p.get("asset_col"); market_col = p.get("market_col")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        n_lags = int(p.get("n_lags") or 3)
        if not asset_col or asset_col not in df.columns:
            raise ValueError("Select the asset return column.")
        if not market_col or market_col not in df.columns:
            raise ValueError("Select the market return column.")
        n_lags = max(1, min(n_lags, 8))

        a = _ret(df[asset_col], is_returns, rtype)
        m = _ret(df[market_col], is_returns, rtype)
        base = pd.concat([a.rename("a"), m.rename("m")], axis=1).dropna().reset_index(drop=True)
        if len(base) < n_lags + 20:
            raise ValueError(f"Need at least {n_lags + 20} aligned observations.")

        # ---- distributed lag regression ----
        cols = {}
        cols["m_0"] = base["m"]
        for k in range(1, n_lags + 1):
            cols[f"m_{k}"] = base["m"].shift(k)
        X = pd.DataFrame(cols)
        reg = pd.concat([base["a"], X], axis=1).dropna().reset_index(drop=True)
        y = reg["a"].values
        Xv = sm.add_constant(reg[list(cols.keys())].values)
        fit = sm.OLS(y, Xv).fit()

        names = ["alpha"] + list(cols.keys())
        lag_betas = []
        contemp = float(fit.params[1])
        total_beta = float(np.sum(fit.params[1:]))
        for i, nm in enumerate(names):
            if nm == "alpha":
                continue
            lag = int(nm.split("_")[1])
            lag_betas.append({"lag": lag, "beta": _fin(float(fit.params[i]), 5),
                              "t_stat": _fin(float(fit.tvalues[i]), 4),
                              "p_value": _fin(float(fit.pvalues[i]), 6),
                              "significant": bool(fit.pvalues[i] < 0.05)})
        # speed of adjustment: share of total beta in the contemporaneous term
        speed = (contemp / total_beta) if total_beta != 0 else None
        delayed_share = (1 - speed) if speed is not None else None
        n_sig_lags = sum(1 for l in lag_betas if l["lag"] >= 1 and l["significant"])

        # ---- up/down asymmetry ----
        up = base["m"] > 0
        Xa = pd.DataFrame({
            "m": base["m"],
            "m_down": base["m"] * (~up).astype(float),   # extra slope in down markets
        })
        Xa_ = sm.add_constant(Xa.values)
        fit2 = sm.OLS(base["a"].values, Xa_).fit()
        beta_up = float(fit2.params[1])
        beta_down = float(fit2.params[1] + fit2.params[2])
        asym_coef = float(fit2.params[2]); asym_p = float(fit2.pvalues[2])
        asymmetric = bool(asym_p < 0.05)

        # ---- optional: volume reaction (extended to Pre / Event / Post thirds) ----
        volume_reaction = None
        volume_table = None
        try:
            volume_col = p.get("volume_col")
            if volume_col and volume_col in df.columns:
                vol = pd.to_numeric(df[volume_col], errors="coerce")
                vol_aligned = vol.loc[base.index].reset_index(drop=True) if len(vol) == len(base) else vol.reindex(range(len(base)))
                vol_aligned = vol_aligned.dropna()
                if len(vol_aligned) >= 20:
                    nv = len(vol_aligned)
                    (p0, p1), (e0, e1), (o0, o1) = _thirds(nv)
                    pre_v = float(vol_aligned.iloc[p0:p1].mean())
                    event_v = float(vol_aligned.iloc[e0:e1].mean())
                    post_v = float(vol_aligned.iloc[o0:o1].mean())
                    pct_event = (event_v / pre_v - 1.0) if pre_v else None
                    pct_post = (post_v / pre_v - 1.0) if pre_v else None
                    # legacy 2-bucket fields kept for backward compatibility
                    mid_v = nv // 2
                    baseline_avg = float(vol_aligned.iloc[:mid_v].mean())
                    event_avg = float(vol_aligned.iloc[mid_v:].mean())
                    legacy_pct = (event_avg / baseline_avg - 1.0) if baseline_avg else None
                    volume_reaction = {"baseline_avg": _fin(baseline_avg, 2), "event_avg": _fin(event_avg, 2),
                                       "pct_change": _fin(legacy_pct, 4) if legacy_pct is not None else None}
                    volume_table = {
                        "pre_avg": _fin(pre_v, 2), "event_avg": _fin(event_v, 2), "post_avg": _fin(post_v, 2),
                        "pct_change_event": _fin(pct_event, 4) if pct_event is not None else None,
                        "pct_change_post": _fin(pct_post, 4) if pct_post is not None else None,
                    }
        except Exception:
            volume_reaction = None; volume_table = None

        # ---- volatility reaction (always on; extended to Pre / Event / Post thirds) ----
        volatility_reaction = None
        volatility_table = None
        try:
            asset_series = base["a"].reset_index(drop=True)
            n_a = len(asset_series)
            (p0, p1), (e0, e1), (o0, o1) = _thirds(n_a)
            pre_seg = asset_series.iloc[p0:p1]; event_seg = asset_series.iloc[e0:e1]; post_seg = asset_series.iloc[o0:o1]
            if len(pre_seg) >= 3 and len(event_seg) >= 3 and len(post_seg) >= 3:
                pre_vol = float(pre_seg.std(ddof=1))
                event_vol = float(event_seg.std(ddof=1))
                post_vol = float(post_seg.std(ddof=1))
                pct_event_v = (event_vol / pre_vol - 1.0) if pre_vol else None
                pct_post_v = (post_vol / pre_vol - 1.0) if pre_vol else None
                # legacy before/after (sample midpoint) fields kept for backward compatibility
                mid_vol = n_a // 2
                vol_before = float(asset_series.iloc[:mid_vol].std(ddof=1))
                vol_after = float(asset_series.iloc[mid_vol:].std(ddof=1))
                legacy_pct_v = (vol_after / vol_before - 1.0) if vol_before else None
                volatility_reaction = {"before": _fin(vol_before, 6), "after": _fin(vol_after, 6),
                                       "pct_change": _fin(legacy_pct_v, 4) if legacy_pct_v is not None else None}
                volatility_table = {
                    "pre_vol": _fin(pre_vol, 6), "event_vol": _fin(event_vol, 6), "post_vol": _fin(post_vol, 6),
                    "pct_change_event": _fin(pct_event_v, 4) if pct_event_v is not None else None,
                    "pct_change_post": _fin(pct_post_v, 4) if pct_post_v is not None else None,
                }
        except Exception:
            volatility_reaction = None; volatility_table = None

        # ---- Market vs Asset (Pre / Event / Post cumulative return, both series) ----
        relative_table = None
        try:
            n_b = len(base)
            (p0, p1), (e0, e1), (o0, o1) = _thirds(n_b)

            def _cum(seg):
                return float(np.prod(1.0 + seg.values) - 1.0) if len(seg) else None

            asset_pre = _cum(base["a"].iloc[p0:p1]); asset_event = _cum(base["a"].iloc[e0:e1]); asset_post = _cum(base["a"].iloc[o0:o1])
            mkt_pre = _cum(base["m"].iloc[p0:p1]); mkt_event = _cum(base["m"].iloc[e0:e1]); mkt_post = _cum(base["m"].iloc[o0:o1])
            relative_table = {
                "pre": {"asset": _fin(asset_pre, 5), "market": _fin(mkt_pre, 5),
                        "relative": _fin(asset_pre - mkt_pre, 5) if asset_pre is not None and mkt_pre is not None else None},
                "event": {"asset": _fin(asset_event, 5), "market": _fin(mkt_event, 5),
                          "relative": _fin(asset_event - mkt_event, 5) if asset_event is not None and mkt_event is not None else None},
                "post": {"asset": _fin(asset_post, 5), "market": _fin(mkt_post, 5),
                         "relative": _fin(asset_post - mkt_post, 5) if asset_post is not None and mkt_post is not None else None},
            }
        except Exception:
            relative_table = None

        # ---- optional: industry/market grouping ----
        by_group = None
        try:
            group_col = p.get("group_col")
            if group_col and group_col in df.columns:
                grp_series = df[group_col]
                grp_aligned = grp_series.loc[base.index].reset_index(drop=True) if len(grp_series) == len(df) else None
                if grp_aligned is not None:
                    rows_out = []
                    for gval in pd.unique(grp_aligned.dropna()):
                        gmask = (grp_aligned == gval).values
                        sub = base.loc[gmask].reset_index(drop=True)
                        if len(sub) < n_lags + 20:
                            continue
                        gy = sub["a"].values
                        gX = sm.add_constant(sub["m"].values)
                        gfit = sm.OLS(gy, gX).fit()
                        g_beta0 = float(gfit.params[1]); g_p = float(gfit.pvalues[1])
                        rows_out.append({"group": str(gval), "beta_lag0": _fin(g_beta0, 5),
                                          "significant": bool(g_p < 0.05), "n": int(len(sub))})
                    if rows_out:
                        by_group = rows_out
        except Exception:
            by_group = None

        # ══════════════════════════════════════════════════════════════════
        # Reaction-window curve: cumulative sum of lag betas, from an implicit
        # pre-event baseline of 0 at lag -1 through lag 0 (event) to lag n_lags
        # (post-event). This is the "reaction" building up after the market
        # moves the asset — not an abnormal return, just the accumulated
        # regression coefficient.
        # ══════════════════════════════════════════════════════════════════
        lags_ext = [-1] + [l["lag"] for l in lag_betas]
        betas_by_lag = {l["lag"]: (l["beta"] or 0.0) for l in lag_betas}
        cum = 0.0
        cum_ext = [0.0]
        for lag in [l["lag"] for l in lag_betas]:
            cum += betas_by_lag[lag]
            cum_ext.append(cum)
        peak_abs_idx = int(np.argmax(np.abs(cum_ext)))
        peak_lag = lags_ext[peak_abs_idx]
        peak_value = cum_ext[peak_abs_idx]

        # first reaction: first lag (>=0) with a significant beta, else the
        # first lag whose |beta| exceeds 10% of the eventual total reaction
        first_reaction_lag = None
        for l in lag_betas:
            if l["significant"]:
                first_reaction_lag = l["lag"]
                break
        if first_reaction_lag is None:
            thresh = 0.1 * abs(total_beta) if total_beta else 0.0
            for l in lag_betas:
                if abs(l["beta"] or 0.0) >= thresh:
                    first_reaction_lag = l["lag"]
                    break
        if first_reaction_lag is None:
            first_reaction_lag = 0

        # 50%-reaction-reached: first lag where |cum| crosses half of |peak|
        half_lag = None
        if peak_value != 0:
            half_target = 0.5 * abs(peak_value)
            for lag, cval in zip(lags_ext, cum_ext):
                if lag < 0:
                    continue
                if abs(cval) >= half_target:
                    half_lag = lag
                    break
        if half_lag is None:
            half_lag = peak_lag

        # full reaction: first lag (>= peak lag) where the curve plateaus —
        # the step-to-step change drops below 5% of the peak magnitude
        full_reaction_lag = lags_ext[-1]  # default: still moving at window edge
        plateau_thresh = 0.05 * abs(peak_value) if peak_value else 0.0
        for i in range(1, len(cum_ext)):
            if lags_ext[i] < max(0, peak_lag):
                continue
            step = abs(cum_ext[i] - cum_ext[i - 1])
            if step <= plateau_thresh:
                full_reaction_lag = lags_ext[i]
                break

        reaction_lag_metric = peak_lag - 0  # peak lag minus event lag (0)

        speed_label = ("Immediate" if peak_lag <= 1 else "Fast" if peak_lag <= 2 else
                       "Gradual" if full_reaction_lag < n_lags else "Delayed")

        speed_table = {
            "first_reaction_lag": int(first_reaction_lag),
            "peak_reaction_lag": int(peak_lag),
            "peak_reaction_value": _fin(peak_value, 5),
            "half_reaction_lag": int(half_lag),
            "full_reaction_lag": int(full_reaction_lag),
            "reaction_lag": int(reaction_lag_metric),
            "_note": "Reaction speed is read off the cumulative sum of the distributed-lag betas "
                     "(lag -1 = baseline 0 before the market moves the asset, lag 0 = the event period, "
                     "lag 1..n = the delayed/post-event periods). 'Full reaction' is the first lag at or "
                     "after the peak where the step-to-step change falls below 5% of the peak magnitude "
                     "(a heuristic plateau threshold, not a statistical test).",
        }

        # ── ② Price Reaction table (reusing the same lag structure) ──
        pre_event_reaction = 0.0  # baseline before lag 0, by construction of this design
        event_reaction = contemp
        post_event_reaction = total_beta - contemp
        price_reaction_table = {
            "pre_event": _fin(pre_event_reaction, 5),
            "event_window": _fin(event_reaction, 5),
            "post_event": _fin(post_event_reaction, 5),
            "_note": "These are distributed-lag reaction coefficients (beta units), not raw returns. "
                     "'Pre-event' is the implicit zero baseline the cumulative reaction curve starts "
                     "from (this design has no negative-lag term); 'event window' is the contemporaneous "
                     "(lag 0) reaction; 'post-event' is the sum of the lag 1..n reactions.",
        }

        # ── ⑦ Reaction Reversal table (same lag curve) ──
        initial_reaction = event_reaction  # return at lag 0
        final_reaction = cum_ext[-1]       # cumulative reaction at last lag
        reversal = peak_value - final_reaction
        reversal_rate = (reversal / peak_value * 100.0) if peak_value not in (0, None) else None
        reversal_table = {
            "initial_reaction": _fin(initial_reaction, 5),
            "peak_reaction": _fin(peak_value, 5),
            "final_reaction": _fin(final_reaction, 5),
            "reversal": _fin(reversal, 5),
            "reversal_rate": _fin(reversal_rate, 2) if reversal_rate is not None else None,
            "_note": "Reversal = peak cumulative reaction minus the final (lag-n) cumulative reaction; "
                     "a negative reversal means the asset gave back reaction it had gained. Reversal rate "
                     "is only meaningful when the peak reaction is positive (division by a near-zero or "
                     "negative peak is reported as N/A).",
        }
        if peak_value in (0, None) or (isinstance(peak_value, float) and abs(peak_value) < 1e-9):
            reversal_table["reversal_rate"] = None

        # ── ⑧ Information Absorption (rule-based classification) ──
        absorption_label = speed_label
        if reversal_rate is not None and reversal_rate > 50:
            absorption_label = "Reversal"
        absorption = {
            "classification": absorption_label,
            "_note": "Rule-based, not a statistical test: 'Immediate' if the peak reaction is reached by "
                     "lag 0-1, 'Fast' by lag 2, 'Gradual' if the curve plateaus before the last lag, "
                     "'Delayed' if it is still trending at the window's edge, and 'Reversal' overrides "
                     "any of these if more than half of the peak reaction is given back by the last lag.",
        }

        # ── ① Market Reaction Summary (qualitative table) ──
        pre_vol_ref = volatility_table["pre_vol"] if volatility_table else (volatility_reaction["before"] if volatility_reaction else None)
        price_impact = _impact_label(total_beta, pre_vol_ref, hi_mult=2.0, lo_mult=1.0)
        volume_impact = "N/A"
        if volume_table and volume_table.get("pct_change_event") is not None:
            pc = abs(volume_table["pct_change_event"])
            volume_impact = "High" if pc > 0.5 else "Medium" if pc > 0.2 else "Low"
        volatility_impact = "N/A"
        if volatility_table and volatility_table.get("pct_change_event") is not None:
            pcv = abs(volatility_table["pct_change_event"])
            volatility_impact = "High" if pcv > 0.5 else "Medium" if pcv > 0.2 else "Low"
        reaction_direction = "Positive" if final_reaction >= 0 else "Negative"

        # optional heuristic composite score (0-100), clearly labeled
        impact_score_map = {"High": 3, "Medium": 2, "Low": 1, "N/A": 0}
        speed_score_map = {"Immediate": 3, "Fast": 2, "Gradual": 1, "Delayed": 0.5}
        raw_score = (impact_score_map[price_impact] + impact_score_map[volume_impact] +
                     impact_score_map[volatility_impact] + speed_score_map.get(speed_label, 1))
        overall_reaction_score = _fin(min(100.0, raw_score / 12.0 * 100.0), 1)

        reaction_summary = {
            "price_impact": price_impact,
            "volume_impact": volume_impact,
            "volatility_impact": volatility_impact,
            "reaction_speed": speed_label,
            "reaction_duration": int(full_reaction_lag),
            "reaction_direction": reaction_direction,
            "overall_reaction_score": overall_reaction_score,
            "_note": "Price/Volume/Volatility Impact are qualitative High/Medium/Low labels defined for "
                     "this report only, not universal financial standards: Price Impact compares the "
                     "total reaction beta to the pre-event volatility (|reaction| > 2x pre-event vol = "
                     "High, > 1x = Medium, else Low); Volume/Volatility Impact use the event-vs-pre-event "
                     "percent change (>50% = High, >20% = Medium, else Low). 'Overall reaction score' is "
                     "an internally-defined 0-100 heuristic combining these four labels — it is not a "
                     "standard or validated financial measure.",
        }

        # ══════════════════════════════════════════════════════════════════
        # Charts — one PNG per section (up to 6), each its own tab.
        # ══════════════════════════════════════════════════════════════════
        charts = {}

        # price_reaction
        try:
            fig, ax = plt.subplots(figsize=(9, 4.6), dpi=115)
            ax.axvspan(lags_ext[0] - 0.5, -0.5, color=GRAY, alpha=0.15, label="Pre-event")
            ax.axvspan(-0.5, 0.5, color=BLUE, alpha=0.15, label="Event")
            ax.axvspan(0.5, lags_ext[-1] + 0.5, color=AMBER, alpha=0.12, label="Post-event")
            ax.plot(lags_ext, cum_ext, color=BLUE, lw=1.6, marker="o", ms=4)
            ax.axhline(0, color="#111827", lw=0.7)
            ax.set_title("Cumulative Price Reaction")
            ax.set_xlabel("Lag (0 = event)"); ax.set_ylabel("Cumulative reaction (beta units)")
            ax.legend(fontsize=8, frameon=False, loc="best")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["price_reaction"] = _png(fig)
        except Exception:
            plt.close("all"); charts["price_reaction"] = None

        # reaction_speed
        try:
            fig, ax = plt.subplots(figsize=(9, 4.6), dpi=115)
            ax.plot(lags_ext, cum_ext, color=BLUE, lw=1.6, marker="o", ms=4)
            ax.axhline(0, color="#111827", lw=0.7)
            fi_idx = lags_ext.index(first_reaction_lag) if first_reaction_lag in lags_ext else None
            if fi_idx is not None:
                ax.scatter([lags_ext[fi_idx]], [cum_ext[fi_idx]], color=AMBER, zorder=5, s=45, label="First reaction")
            ax.scatter([peak_lag], [peak_value], color=GREEN, zorder=5, s=45, label="Peak reaction")
            fr_idx = lags_ext.index(full_reaction_lag) if full_reaction_lag in lags_ext else None
            if fr_idx is not None:
                ax.scatter([lags_ext[fr_idx]], [cum_ext[fr_idx]], color=RED, zorder=5, s=45, label="Full reaction")
            ax.set_title("Cumulative Market Reaction")
            ax.set_xlabel("Lag (0 = event)"); ax.set_ylabel("Cumulative reaction (beta units)")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["reaction_speed"] = _png(fig)
        except Exception:
            plt.close("all"); charts["reaction_speed"] = None

        # volume
        charts["volume"] = None
        if volume_table:
            try:
                fig, ax = plt.subplots(figsize=(7, 4.6), dpi=115)
                vals = [volume_table["pre_avg"], volume_table["event_avg"], volume_table["post_avg"]]
                ax.bar(["Pre", "Event", "Post"], vals, color=[GRAY, BLUE, AMBER], width=0.55)
                ax.set_ylabel("Avg volume"); ax.set_title("Volume Around Event")
                ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                charts["volume"] = _png(fig)
            except Exception:
                plt.close("all"); charts["volume"] = None

        # volatility
        try:
            fig, ax = plt.subplots(figsize=(7, 4.6), dpi=115)
            vals = [volatility_table["pre_vol"], volatility_table["event_vol"], volatility_table["post_vol"]] if volatility_table else [0, 0, 0]
            ax.bar(["Pre", "Event", "Post"], vals, color=[GRAY, BLUE, RED], width=0.55)
            ax.set_ylabel("Realized volatility"); ax.set_title("Volatility Around Event")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            charts["volatility"] = _png(fig)
        except Exception:
            plt.close("all"); charts["volatility"] = None

        # market_vs_asset
        try:
            fig, ax = plt.subplots(figsize=(7, 4.6), dpi=115)
            xs = np.arange(3); width = 0.35
            asset_vals = [relative_table["pre"]["asset"], relative_table["event"]["asset"], relative_table["post"]["asset"]] if relative_table else [0, 0, 0]
            mkt_vals = [relative_table["pre"]["market"], relative_table["event"]["market"], relative_table["post"]["market"]] if relative_table else [0, 0, 0]
            ax.bar(xs - width / 2, [v * 100 if v is not None else 0 for v in asset_vals], width, color=BLUE, label="Asset")
            ax.bar(xs + width / 2, [v * 100 if v is not None else 0 for v in mkt_vals], width, color=GRAY, label="Market")
            ax.axhline(0, color="#111827", lw=0.7)
            ax.set_xticks(xs); ax.set_xticklabels(["Pre", "Event", "Post"])
            ax.set_ylabel("Cumulative return (%)"); ax.set_title("Asset vs Benchmark")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            charts["market_vs_asset"] = _png(fig)
        except Exception:
            plt.close("all"); charts["market_vs_asset"] = None

        # reversal
        try:
            fig, ax = plt.subplots(figsize=(9, 4.6), dpi=115)
            ax.plot(lags_ext, cum_ext, color=BLUE, lw=1.6, marker="o", ms=4)
            ax.axhline(0, color="#111827", lw=0.7)
            ax.scatter([0], [initial_reaction], color=BLUE, zorder=5, s=45, label="Initial")
            ax.scatter([peak_lag], [peak_value], color=GREEN, zorder=5, s=45, label="Peak")
            ax.scatter([lags_ext[-1]], [final_reaction], color=RED, zorder=5, s=45, label="Final")
            ax.set_title(f"Reaction Reversal ({'reversed' if reversal > 0 else 'held'})")
            ax.set_xlabel("Lag (0 = event)"); ax.set_ylabel("Cumulative reaction (beta units)")
            ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["reversal"] = _png(fig)
        except Exception:
            plt.close("all"); charts["reversal"] = None

        # ---- original 2-panel (+ legacy volume/volatility panels) summary plot,
        # kept for the step-4/5 "Reaction Result" visual which other parts rely on ----
        n_extra_panels = (1 if volume_reaction is not None else 0) + (1 if volatility_reaction is not None else 0)
        plot = None
        try:
            n_panels = 2 + n_extra_panels
            fig = plt.figure(figsize=(12.5 if n_panels <= 2 else 12.5, 5 if n_panels <= 2 else 5 * ((n_panels + 1) // 2)), dpi=120)
            ncols = 2
            nrows = (n_panels + 1) // 2
            ax1 = fig.add_subplot(nrows, ncols, 1)
            ax2 = fig.add_subplot(nrows, ncols, 2)
            lags = [l["lag"] for l in lag_betas]
            betas = [l["beta"] for l in lag_betas]
            cols_c = ["#2563eb" if l["significant"] else "#94a3b8" for l in lag_betas]
            ax1.bar(lags, betas, color=cols_c)
            ax1.axhline(0, color="#111827", lw=0.7)
            ax1.set_xlabel("Lag (0 = same period)"); ax1.set_ylabel("Reaction (beta)")
            ax1.set_title("Reaction to market by lag (blue = significant)")
            ax1.set_xticks(lags)
            # up/down betas
            ax2.bar(["Up market", "Down market"], [beta_up, beta_down], color=["#16a34a", "#dc2626"])
            for i, v in enumerate([beta_up, beta_down]):
                ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
            ax2.axhline(0, color="#111827", lw=0.7)
            ax2.set_ylabel("Beta"); ax2.set_title(f"Up vs down market beta ({'asymmetric' if asymmetric else 'symmetric'})")

            panel_i = 3
            if volume_reaction is not None:
                ax3 = fig.add_subplot(nrows, ncols, panel_i); panel_i += 1
                ax3.bar(["Baseline", "Event window"], [volume_reaction["baseline_avg"], volume_reaction["event_avg"]], color=["#94a3b8", "#2563eb"])
                ax3.set_ylabel("Avg volume"); ax3.set_title("Trading volume reaction")
            if volatility_reaction is not None:
                ax4 = fig.add_subplot(nrows, ncols, panel_i); panel_i += 1
                ax4.bar(["Before", "After"], [volatility_reaction["before"], volatility_reaction["after"]], color=["#94a3b8", "#dc2626"])
                ax4.set_ylabel("Realized volatility"); ax4.set_title("Volatility reaction (before vs after midpoint)")

            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        speed_txt = ("almost all of the reaction happens in the same period (efficient, fast price discovery)"
                     if speed is not None and speed > 0.9 else
                     "most of the reaction is immediate, with some spilling into later periods"
                     if speed is not None and speed > 0.7 else
                     "a substantial part of the reaction is delayed, appearing at later lags — a sign of slow price "
                     "discovery, illiquidity, or that the asset lags the market")
        interpretation = (
            f"The asset's contemporaneous reaction to the market is a beta of {contemp:.2f}, and summing the lagged "
            f"reactions gives a total (long-run) beta of {total_beta:.2f}. "
            + (f"About {speed:.0%} of the total reaction occurs immediately — {speed_txt}. " if speed is not None else "")
            + (f"The reaction is asymmetric: the beta is {beta_up:.2f} in rising markets versus {beta_down:.2f} in "
               f"falling markets (difference significant, p = {asym_p:.3f}), so the asset "
               + ("falls harder than it rises with the market — a downside-amplifying profile. " if beta_down > beta_up else
                  "rises more than it falls with the market. ")
               if asymmetric else
               f"The up-market beta ({beta_up:.2f}) and down-market beta ({beta_down:.2f}) are not statistically "
               f"different (p = {asym_p:.3f}), so the reaction is symmetric. ")
        )

        results = {
            "status": "ok", "asset": asset_col, "market": market_col, "n_obs": int(len(reg)), "n_lags": n_lags,
            "contemporaneous_beta": _fin(contemp, 5), "total_beta": _fin(total_beta, 5),
            "speed_of_adjustment": _fin(speed, 4) if speed is not None else None,
            "delayed_share": _fin(delayed_share, 4) if delayed_share is not None else None,
            "n_significant_lags": n_sig_lags, "lag_betas": lag_betas,
            "beta_up": _fin(beta_up, 5), "beta_down": _fin(beta_down, 5),
            "asymmetry_coef": _fin(asym_coef, 5), "asymmetry_p": _fin(asym_p, 6), "asymmetric": asymmetric,
            "r_squared": _fin(float(fit.rsquared), 4),
            "interpretation": interpretation,
            "reaction_summary": reaction_summary,
            "price_reaction_table": price_reaction_table,
            "speed_table": speed_table,
            "volatility_table": volatility_table,
            "relative_table": relative_table,
            "reversal_table": reversal_table,
            "absorption": absorption,
            "charts": charts,
        }
        if volume_reaction is not None:
            results["volume_reaction"] = volume_reaction
        if volume_table is not None:
            results["volume_table"] = volume_table
        if volatility_reaction is not None:
            results["volatility_reaction"] = volatility_reaction
        if by_group is not None:
            results["by_group"] = by_group
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
