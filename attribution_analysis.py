#!/usr/bin/env python3
"""Attribution Analysis — multi-touch marketing attribution from user-level
touchpoint rows (customer_id, timestamp, channel, conversion marker).

pandas / numpy / matplotlib only.

Builds conversion paths per customer (or per designated conversion-group id),
computes channel credit under six attribution models (First Touch, Last
Touch, Linear, Time Decay, Position-Based U-shaped, and a Markov-chain
removal-effect data-driven model), then assembles a 13-section report:

  1. Attribution Overview (KPIs)
  2. Customer Journey / Touchpoints (flow viz + touchpoint table)
  3. Attribution Model Comparison (channel x model table + grouped bar)
  4. Channel Attribution (primary model) (table + chart)
  5. Touchpoint Contribution (Initiator / Converter / Assist roles)
  6. Conversion Path Analysis (top N full paths)
  7. Path Length Analysis (distribution of touchpoints per conversion)
  8. Time to Conversion (elapsed-time buckets)
  9. Assisted Conversions (direct vs assisted per channel)
  10. Attribution ROI (spend vs attributed revenue, if spend given)
  11. Attribution by Customer Segment (conditional)
  12. Attribution by Conversion Type (conditional)
  13. Attribution Model Sensitivity (variance of channel share across models)

Input (from attribution-analysis-page.tsx):
    data              : list[dict]           -- one row per touchpoint
    customer_id_col   : str                  -- groups touchpoints into a journey
    timestamp_col     : str
    channel_col       : str
    conversion_flag_col  : str | None        -- boolean/0-1, marks converting row
    conversion_value_col : str | None        -- numeric; >0 marks a conversion if no flag col
    primary_model     : str  (default "linear")
    segment_col       : str | None           -- optional, section 11
    conversion_type_col : str | None         -- optional, section 12
    spend             : dict[channel -> float] | None  -- optional, section 10

Output: { results: {...} }
"""
import sys
import json
import math
import warnings
import io
import base64
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#9333ea"
TEAL = "#0d9488"
PALETTE = [BLUE, GREEN, AMBER, PURPLE, RED, TEAL, "#64748b", "#db2777"]

MODELS = ["first_touch", "last_touch", "linear", "time_decay", "position_based", "data_driven"]
MODEL_LABELS = {
    "first_touch": "First Touch", "last_touch": "Last Touch", "linear": "Linear",
    "time_decay": "Time Decay", "position_based": "Position-Based (U-shaped)",
    "data_driven": "Data-Driven (Markov removal effect)",
}


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches='tight')
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_paths(df, cust_col, time_col, chan_col, conv_flag_col, conv_value_col):
    """Return list of dicts: {customer, channels[], times[], converted(bool), value(float)}"""
    paths = []
    for cust, grp in df.groupby(cust_col):
        g = grp.sort_values(time_col)
        channels = g[chan_col].astype(str).tolist()
        times = g[time_col].tolist()
        converted = False
        value = 0.0
        if conv_flag_col and conv_flag_col in g.columns:
            flags = g[conv_flag_col].tolist()
            converted = any(bool(f) and not pd.isna(f) and str(f).lower() not in ('0', 'false', 'nan', '') for f in flags)
            if conv_value_col and conv_value_col in g.columns:
                value = float(pd.to_numeric(g[conv_value_col], errors='coerce').fillna(0).sum())
        elif conv_value_col and conv_value_col in g.columns:
            vals = pd.to_numeric(g[conv_value_col], errors='coerce').fillna(0)
            converted = bool((vals > 0).any())
            value = float(vals.sum())
        else:
            converted = True
            value = 1.0
        paths.append({
            "customer": str(cust), "channels": channels, "times": times,
            "converted": converted, "value": value,
        })
    return paths


def markov_removal_effect(all_paths, all_channels):
    """First-order Markov-chain removal-effect attribution (data-driven).
    States = channels + START + CONVERSION + NULL. NULL is a second absorbing
    state reached by journeys that did not convert -- without it every path
    would eventually reach CONVERSION with probability 1 regardless of which
    channel is removed (since the chain is otherwise irreducible), making the
    removal effect degenerate. Transition probabilities are estimated from
    ALL observed journeys (converting -> CONVERSION, non-converting -> NULL).
    """
    channels = list(all_channels)
    if not channels:
        return {}

    trans = defaultdict(lambda: defaultdict(int))
    for p in all_paths:
        end = "CONVERSION" if p["converted"] else "NULL"
        seq = ["START"] + p["channels"] + [end]
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a][b] += 1

    def transition_probs(exclude=None):
        probs = {}
        for state, nxt in trans.items():
            if exclude and state == exclude:
                continue
            total = sum(v for k, v in nxt.items() if not (exclude and k == exclude))
            if total <= 0:
                continue
            probs[state] = {k: v / total for k, v in nxt.items() if not (exclude and k == exclude)}
        return probs

    def conversion_prob(exclude=None, max_iter=200):
        probs = transition_probs(exclude)
        states = set(["START", "CONVERSION", "NULL"]) | set(channels)
        if exclude:
            states.discard(exclude)
        dist = {s: 0.0 for s in states}
        dist["START"] = 1.0
        conv_mass = 0.0
        for _ in range(max_iter):
            new_dist = {s: 0.0 for s in states}
            moving = False
            for s, mass in dist.items():
                if mass <= 1e-12 or s in ("CONVERSION", "NULL"):
                    continue
                nxt = probs.get(s, {})
                if not nxt:
                    continue
                for k, pr in nxt.items():
                    if k == "CONVERSION":
                        conv_mass += mass * pr
                    elif k == "NULL":
                        pass  # absorbed, mass leaves the system
                    elif k in new_dist:
                        new_dist[k] += mass * pr
                        moving = True
            dist = new_dist
            if not moving:
                break
        return conv_mass

    base_conv = conversion_prob(None)
    removal_effects = {}
    for c in channels:
        with_c_removed = conversion_prob(exclude=c)
        effect = max(base_conv - with_c_removed, 0.0)
        removal_effects[c] = effect

    total_effect = sum(removal_effects.values())
    if total_effect <= 1e-9:
        return {c: 1.0 / len(channels) for c in channels}
    return {c: v / total_effect for c, v in removal_effects.items()}


def compute_model_credits(paths, all_paths_for_markov=None):
    """paths: list of converting journeys with channels/times/value (used for
    the $-credit rule-based models). all_paths_for_markov: full set of paths
    (converting + non-converting) used only for the data-driven Markov model,
    which needs both outcomes to estimate a meaningful removal effect.
    Returns dict[model] -> dict[channel] -> credit ($ amount)."""
    credit = {m: defaultdict(float) for m in MODELS}
    all_channels = set()
    for p in paths:
        for c in p["channels"]:
            all_channels.add(c)

    HALF_LIFE_SEC = 7 * 86400.0

    for p in paths:
        chans = p["channels"]
        times = p["times"]
        val = p["value"] if p["value"] > 0 else 1.0
        n = len(chans)
        if n == 0:
            continue

        credit["first_touch"][chans[0]] += val
        credit["last_touch"][chans[-1]] += val

        share = val / n
        for c in chans:
            credit["linear"][c] += share

        try:
            t_end = pd.Timestamp(times[-1]).timestamp()
            secs = [max((t_end - pd.Timestamp(t).timestamp()), 0.0) for t in times]
        except Exception:
            secs = [float(n - 1 - i) * 86400.0 for i in range(n)]
        weights = [2 ** (-(s / HALF_LIFE_SEC)) for s in secs]
        wsum = sum(weights) or 1.0
        for c, w in zip(chans, weights):
            credit["time_decay"][c] += val * (w / wsum)

        if n == 1:
            credit["position_based"][chans[0]] += val
        elif n == 2:
            credit["position_based"][chans[0]] += val * 0.5
            credit["position_based"][chans[1]] += val * 0.5
        else:
            credit["position_based"][chans[0]] += val * 0.4
            credit["position_based"][chans[-1]] += val * 0.4
            mid = val * 0.2 / (n - 2)
            for c in chans[1:-1]:
                credit["position_based"][c] += mid

    dd_shares = markov_removal_effect(all_paths_for_markov if all_paths_for_markov is not None else paths, all_channels)
    total_val = sum(p["value"] if p["value"] > 0 else 1.0 for p in paths)
    for c, share in dd_shares.items():
        credit["data_driven"][c] = share * total_val

    return credit, all_channels


def bucket_time_to_conv(seconds):
    days = seconds / 86400.0
    if days < 1:
        return "Same day"
    if days <= 3:
        return "1-3 days"
    if days <= 7:
        return "4-7 days"
    if days <= 30:
        return "8-30 days"
    return "30+ days"


def bucket_path_len(n):
    return str(n) if n <= 4 else "5+"


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        cust_col = payload.get('customer_id_col')
        time_col = payload.get('timestamp_col')
        chan_col = payload.get('channel_col')
        conv_flag_col = payload.get('conversion_flag_col') or None
        conv_value_col = payload.get('conversion_value_col') or None
        primary_model = payload.get('primary_model') or 'linear'
        segment_col = payload.get('segment_col') or None
        conv_type_col = payload.get('conversion_type_col') or None
        spend = payload.get('spend') or None

        if not all([data, cust_col, time_col, chan_col]):
            raise ValueError("Missing data, customer_id_col, timestamp_col, or channel_col.")
        if primary_model not in MODELS:
            primary_model = 'linear'

        df = pd.DataFrame(data)
        for need in (cust_col, time_col, chan_col):
            if need not in df.columns:
                raise ValueError(f"Column '{need}' not found in data.")

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[cust_col, time_col, chan_col])
        df[chan_col] = df[chan_col].astype(str)
        if df.empty:
            raise ValueError("No valid touchpoint rows after cleaning.")

        n_touchpoints = int(len(df))

        all_paths = build_paths(df, cust_col, time_col, chan_col, conv_flag_col, conv_value_col)
        converting_paths = [p for p in all_paths if p["converted"] and len(p["channels"]) > 0]
        n_conversions = len(converting_paths)
        n_customers = len(all_paths)
        if n_conversions == 0:
            raise ValueError("No conversions detected — check conversion_flag_col / conversion_value_col.")

        total_revenue = float(sum(p["value"] if p["value"] > 0 else 0.0 for p in converting_paths))
        coverage_pct = _fin(n_conversions / n_customers * 100, 2) if n_customers else None

        credit, all_channels_set = compute_model_credits(converting_paths, all_paths_for_markov=all_paths)
        all_channels = sorted(all_channels_set)

        # ---------------- Section 3: Model comparison table ----------------
        model_totals = {m: sum(credit[m].values()) or 1.0 for m in MODELS}
        model_comparison_table = []
        for c in all_channels:
            row = {"channel": c}
            for m in MODELS:
                amt = credit[m].get(c, 0.0)
                row[f"{m}_value"] = _fin(amt, 2)
                row[f"{m}_pct"] = _fin(amt / model_totals[m] * 100, 2)
            model_comparison_table.append(row)
        model_comparison_table.sort(key=lambda r: r.get(f"{primary_model}_value") or 0, reverse=True)

        top_channel_diff = None
        if all_channels:
            top_c = model_comparison_table[0]["channel"]
            top_c_pcts = {m: next(r[f"{m}_pct"] for r in model_comparison_table if r["channel"] == top_c) for m in MODELS}
            top_channel_diff = {"channel": top_c, "pct_by_model": top_c_pcts}

        chart_model_comparison = None
        try:
            fig, ax = plt.subplots(figsize=(10, 5.5), dpi=110)
            x = np.arange(len(all_channels))
            width = 0.13
            for i, m in enumerate(MODELS):
                vals = [next(r[f"{m}_pct"] or 0 for r in model_comparison_table if r["channel"] == c) for c in all_channels]
                ax.bar(x + i * width, vals, width, label=MODEL_LABELS[m], color=PALETTE[i % len(PALETTE)])
            ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
            ax.set_xticklabels(all_channels, rotation=20, ha='right')
            ax.set_ylabel("Attributed share (%)")
            ax.set_title("Attribution Model Comparison by Channel")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.2, axis='y')
            fig.tight_layout()
            chart_model_comparison = _png(fig)
        except Exception:
            plt.close('all')
            chart_model_comparison = None

        # ---------------- Section 4: Channel attribution (primary model) ----------------
        channel_attribution_table = []
        for c in all_channels:
            amt = credit[primary_model].get(c, 0.0)
            channel_attribution_table.append({
                "channel": c,
                "conversions": _fin(amt / (total_revenue / n_conversions) if total_revenue > 0 else 0, 2),
                "attribution_pct": _fin(amt / model_totals[primary_model] * 100, 2),
                "revenue": _fin(amt, 2),
            })
        channel_attribution_table.sort(key=lambda r: r["revenue"] or 0, reverse=True)
        top_channel = channel_attribution_table[0]["channel"] if channel_attribution_table else None

        chart_channel_attribution = None
        try:
            fig, ax = plt.subplots(figsize=(7.5, 5), dpi=110)
            sorted_rows = sorted(channel_attribution_table, key=lambda r: r["revenue"] or 0)
            colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(sorted_rows)))
            ax.barh([r["channel"] for r in sorted_rows], [r["revenue"] or 0 for r in sorted_rows], color=colors)
            ax.set_xlabel(f"Attributed revenue ({MODEL_LABELS[primary_model]})")
            ax.set_title("Channel Attribution")
            ax.grid(alpha=0.2, axis='x')
            fig.tight_layout()
            chart_channel_attribution = _png(fig)
        except Exception:
            plt.close('all')
            chart_channel_attribution = None

        # ---------------- Section 2: Journey / touchpoint interactions ----------------
        transition_counts = defaultdict(int)
        touchpoint_channel_stats = defaultdict(lambda: {"occurrences": 0, "conversions": 0})
        for p in all_paths:
            chans = p["channels"]
            for i, c in enumerate(chans):
                touchpoint_channel_stats[c]["occurrences"] += 1
                if p["converted"] and i == len(chans) - 1:
                    touchpoint_channel_stats[c]["conversions"] += 1
            seq = ["START"] + chans + (["CONVERSION"] if p["converted"] else ["EXIT"])
            for a, b in zip(seq[:-1], seq[1:]):
                transition_counts[(a, b)] += 1

        touchpoint_table = []
        for c, s in touchpoint_channel_stats.items():
            occ = s["occurrences"]
            touchpoint_table.append({
                "channel": c, "occurrences": occ, "conversions": s["conversions"],
                "conversion_rate_pct": _fin(s["conversions"] / occ * 100, 2) if occ else None,
            })
        touchpoint_table.sort(key=lambda r: r["occurrences"], reverse=True)

        chart_journey_flow = None
        try:
            # simple flow diagram: top transitions as an annotated bar of frequency
            top_trans = sorted(transition_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
            fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
            labels = [f"{a} -> {b}" for (a, b), _ in top_trans]
            vals = [v for _, v in top_trans]
            colors = plt.cm.cool(np.linspace(0.1, 0.9, len(labels)))
            ax.barh(labels[::-1], vals[::-1], color=colors)
            ax.set_xlabel("Occurrences")
            ax.set_title("Top Touchpoint Transitions (Journey Flow)")
            ax.grid(alpha=0.2, axis='x')
            fig.tight_layout()
            chart_journey_flow = _png(fig)
        except Exception:
            plt.close('all')
            chart_journey_flow = None

        # ---------------- Section 5: Touchpoint contribution (role) ----------------
        role_stats = defaultdict(lambda: {"initiator": 0, "converter": 0, "assist": 0})
        for p in converting_paths:
            chans = p["channels"]
            n = len(chans)
            for i, c in enumerate(chans):
                if i == 0 and n > 1:
                    role_stats[c]["initiator"] += 1
                if i == n - 1:
                    role_stats[c]["converter"] += 1
                if 0 < i < n - 1:
                    role_stats[c]["assist"] += 1
                if n == 1:
                    role_stats[c]["converter"] += 1
        touchpoint_role_table = []
        for c, s in role_stats.items():
            touchpoint_role_table.append({"channel": c, **s, "total": s["initiator"] + s["converter"] + s["assist"]})
        touchpoint_role_table.sort(key=lambda r: r["total"], reverse=True)

        chart_role = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 5), dpi=110)
            chs = [r["channel"] for r in touchpoint_role_table]
            x = np.arange(len(chs))
            width = 0.25
            ax.bar(x - width, [r["initiator"] for r in touchpoint_role_table], width, label="Initiator", color=BLUE)
            ax.bar(x, [r["assist"] for r in touchpoint_role_table], width, label="Assist", color=AMBER)
            ax.bar(x + width, [r["converter"] for r in touchpoint_role_table], width, label="Converter", color=GREEN)
            ax.set_xticks(x); ax.set_xticklabels(chs, rotation=20, ha='right')
            ax.set_ylabel("Touchpoint occurrences")
            ax.set_title("Touchpoint Contribution by Role")
            ax.legend()
            ax.grid(alpha=0.2, axis='y')
            fig.tight_layout()
            chart_role = _png(fig)
        except Exception:
            plt.close('all')
            chart_role = None

        # ---------------- Section 6: Conversion path analysis ----------------
        path_counts = defaultdict(int)
        path_revenue = defaultdict(float)
        for p in converting_paths:
            key = " > ".join(p["channels"])
            path_counts[key] += 1
            path_revenue[key] += (p["value"] if p["value"] > 0 else 0.0)
        top_paths = sorted(path_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
        conversion_path_table = [{
            "path": path, "conversions": cnt,
            "pct_of_conversions": _fin(cnt / n_conversions * 100, 2),
            "revenue": _fin(path_revenue[path], 2),
        } for path, cnt in top_paths]

        chart_top_paths = None
        try:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
            labels = [r["path"] if len(r["path"]) < 40 else r["path"][:37] + "..." for r in conversion_path_table][::-1]
            vals = [r["conversions"] for r in conversion_path_table][::-1]
            colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(labels)))
            ax.barh(labels, vals, color=colors)
            ax.set_xlabel("Conversions")
            ax.set_title("Top Conversion Paths")
            ax.grid(alpha=0.2, axis='x')
            fig.tight_layout()
            chart_top_paths = _png(fig)
        except Exception:
            plt.close('all')
            chart_top_paths = None

        # ---------------- Section 7: Path length analysis ----------------
        len_counts = defaultdict(int)
        for p in converting_paths:
            len_counts[bucket_path_len(len(p["channels"]))] += 1
        order = ["1", "2", "3", "4", "5+"]
        path_length_table = [{
            "touchpoints": k, "conversions": len_counts.get(k, 0),
            "pct": _fin(len_counts.get(k, 0) / n_conversions * 100, 2),
        } for k in order]
        lengths = [len(p["channels"]) for p in converting_paths]
        avg_path_length = _fin(np.mean(lengths), 2) if lengths else None
        median_path_length = _fin(np.median(lengths), 2) if lengths else None

        chart_path_length = None
        try:
            fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=110)
            ax.bar([r["touchpoints"] for r in path_length_table], [r["conversions"] for r in path_length_table], color=TEAL)
            ax.set_xlabel("Touchpoints before conversion")
            ax.set_ylabel("Conversions")
            ax.set_title("Path Length Distribution")
            ax.grid(alpha=0.2, axis='y')
            fig.tight_layout()
            chart_path_length = _png(fig)
        except Exception:
            plt.close('all')
            chart_path_length = None

        # ---------------- Section 8: Time to conversion ----------------
        ttc_buckets = defaultdict(int)
        ttc_seconds_list = []
        for p in converting_paths:
            if len(p["times"]) < 1:
                continue
            try:
                t0 = pd.Timestamp(p["times"][0]).timestamp()
                t1 = pd.Timestamp(p["times"][-1]).timestamp()
                secs = max(t1 - t0, 0.0)
            except Exception:
                continue
            ttc_seconds_list.append(secs)
            ttc_buckets[bucket_time_to_conv(secs)] += 1
        bucket_order = ["Same day", "1-3 days", "4-7 days", "8-30 days", "30+ days"]
        time_to_conversion_table = [{
            "bucket": b, "conversions": ttc_buckets.get(b, 0),
            "pct": _fin(ttc_buckets.get(b, 0) / n_conversions * 100, 2),
        } for b in bucket_order]
        avg_days_to_conversion = _fin(np.mean(ttc_seconds_list) / 86400.0, 2) if ttc_seconds_list else None
        median_days_to_conversion = _fin(np.median(ttc_seconds_list) / 86400.0, 2) if ttc_seconds_list else None

        chart_ttc = None
        try:
            fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=110)
            ax.bar([r["bucket"] for r in time_to_conversion_table], [r["conversions"] for r in time_to_conversion_table], color=PURPLE)
            ax.set_ylabel("Conversions")
            ax.set_title("Time to Conversion")
            plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
            ax.grid(alpha=0.2, axis='y')
            fig.tight_layout()
            chart_ttc = _png(fig)
        except Exception:
            plt.close('all')
            chart_ttc = None

        # ---------------- Section 9: Assisted conversions ----------------
        assisted_table = []
        for c in all_channels:
            direct = sum(1 for p in converting_paths if p["channels"] and p["channels"][-1] == c)
            assisted = sum(1 for p in converting_paths if c in p["channels"][:-1])
            assisted_table.append({
                "channel": c, "direct_conversions": direct, "assisted_conversions": assisted,
                "assist_ratio": _fin(assisted / direct, 2) if direct else None,
            })
        assisted_table.sort(key=lambda r: r["assisted_conversions"], reverse=True)

        chart_assisted = None
        try:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
            chs = [r["channel"] for r in assisted_table]
            x = np.arange(len(chs))
            width = 0.35
            ax.bar(x - width / 2, [r["direct_conversions"] for r in assisted_table], width, label="Direct (last-touch)", color=GREEN)
            ax.bar(x + width / 2, [r["assisted_conversions"] for r in assisted_table], width, label="Assisted", color=AMBER)
            ax.set_xticks(x); ax.set_xticklabels(chs, rotation=20, ha='right')
            ax.set_ylabel("Conversions")
            ax.set_title("Direct vs Assisted Conversions")
            ax.legend()
            ax.grid(alpha=0.2, axis='y')
            fig.tight_layout()
            chart_assisted = _png(fig)
        except Exception:
            plt.close('all')
            chart_assisted = None

        # ---------------- Section 10: Attribution ROI (conditional on spend) ----------------
        roi_table = None
        roi_note = None
        chart_roi = None
        if spend and isinstance(spend, dict):
            roi_table = []
            for c in all_channels:
                sp = float(spend.get(c, 0) or 0)
                rev = credit[primary_model].get(c, 0.0)
                roi_table.append({
                    "channel": c, "spend": _fin(sp, 2), "attributed_revenue": _fin(rev, 2),
                    "roas": _fin(rev / sp, 2) if sp > 0 else None,
                })
            roi_table.sort(key=lambda r: r["attributed_revenue"] or 0, reverse=True)
            try:
                fig, ax = plt.subplots(figsize=(7.5, 5), dpi=110)
                chs = [r["channel"] for r in roi_table]
                ax.bar(chs, [r["roas"] or 0 for r in roi_table], color=BLUE)
                ax.set_ylabel("ROAS (attributed revenue / spend)")
                ax.set_title("Attribution ROI by Channel")
                plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
                ax.grid(alpha=0.2, axis='y')
                fig.tight_layout()
                chart_roi = _png(fig)
            except Exception:
                plt.close('all')
                chart_roi = None
        else:
            roi_note = "No per-channel spend was provided, so ROAS cannot be computed — only attributed revenue per channel is available (see Channel Attribution)."

        # ---------------- Section 11: Attribution by customer segment (conditional) ----------------
        segment_table = None
        segment_note = None
        if segment_col and segment_col in df.columns:
            seg_map = df.drop_duplicates(subset=[cust_col]).set_index(cust_col)[segment_col].to_dict()
            seg_credit = defaultdict(lambda: defaultdict(float))
            for p in converting_paths:
                seg = str(seg_map.get(p["customer"], "Unknown"))
                n = len(p["channels"]) or 1
                val = p["value"] if p["value"] > 0 else 1.0
                share = val / n
                for c in p["channels"]:
                    seg_credit[seg][c] += share
            segment_table = []
            for seg, cmap in seg_credit.items():
                total = sum(cmap.values())
                top_c = max(cmap.items(), key=lambda kv: kv[1])[0] if cmap else None
                segment_table.append({"segment": seg, "revenue": _fin(total, 2), "top_channel": top_c})
            segment_table.sort(key=lambda r: r["revenue"] or 0, reverse=True)
        else:
            segment_note = "No customer-segment column was provided, so attribution-by-segment is skipped."

        # ---------------- Section 12: Attribution by conversion type (conditional) ----------------
        conv_type_table = None
        conv_type_note = None
        if conv_type_col and conv_type_col in df.columns:
            type_map = defaultdict(lambda: defaultdict(float))
            type_counts = defaultdict(int)
            row_type = df.drop_duplicates(subset=[cust_col], keep='last').set_index(cust_col)[conv_type_col].to_dict()
            for p in converting_paths:
                ctype = str(row_type.get(p["customer"], "Unknown"))
                type_counts[ctype] += 1
                n = len(p["channels"]) or 1
                val = p["value"] if p["value"] > 0 else 1.0
                share = val / n
                for c in p["channels"]:
                    type_map[ctype][c] += share
            conv_type_table = []
            for ctype, cnt in type_counts.items():
                cmap = type_map[ctype]
                top_c = max(cmap.items(), key=lambda kv: kv[1])[0] if cmap else None
                conv_type_table.append({
                    "conversion_type": ctype, "conversions": cnt,
                    "revenue": _fin(sum(cmap.values()), 2), "top_channel": top_c,
                })
            conv_type_table.sort(key=lambda r: r["conversions"], reverse=True)
        else:
            conv_type_note = "No conversion-type column was provided, so attribution-by-conversion-type is skipped."

        # ---------------- Section 13: Model sensitivity ----------------
        sensitivity_table = []
        for c in all_channels:
            pcts = [next(r[f"{m}_pct"] or 0 for r in model_comparison_table if r["channel"] == c) for m in MODELS]
            spread = max(pcts) - min(pcts) if pcts else 0
            verdict = "High" if spread >= 15 else "Low"
            sensitivity_table.append({
                "channel": c, "min_pct": _fin(min(pcts), 2) if pcts else None,
                "max_pct": _fin(max(pcts), 2) if pcts else None,
                "spread_pct": _fin(spread, 2), "sensitivity": verdict,
            })
        sensitivity_table.sort(key=lambda r: r["spread_pct"] or 0, reverse=True)

        chart_sensitivity = None
        try:
            top_n = sensitivity_table[:6]
            fig, axes = plt.subplots(1, len(top_n) or 1, figsize=(3 * (len(top_n) or 1), 4), dpi=110, sharey=True)
            if len(top_n) == 1:
                axes = [axes]
            for ax, row in zip(axes, top_n):
                c = row["channel"]
                pcts = [next(r[f"{m}_pct"] or 0 for r in model_comparison_table if r["channel"] == c) for m in MODELS]
                ax.bar(range(len(MODELS)), pcts, color=PALETTE[:len(MODELS)])
                ax.set_title(c, fontsize=9)
                ax.set_xticks([])
            axes[0].set_ylabel("Attributed share (%)")
            fig.suptitle("Model Sensitivity by Channel (small multiples)", fontsize=11)
            fig.tight_layout()
            chart_sensitivity = _png(fig)
        except Exception:
            plt.close('all')
            chart_sensitivity = None

        # ---------------- Overview ----------------
        overview = {
            "total_conversions": n_conversions,
            "total_customers": n_customers,
            "attributed_conversions": n_conversions,
            "attribution_coverage_pct": coverage_pct,
            "total_revenue": _fin(total_revenue, 2),
            "top_channel": top_channel,
            "primary_model": primary_model,
            "primary_model_label": MODEL_LABELS[primary_model],
            "n_touchpoints": n_touchpoints,
            "n_channels": len(all_channels),
        }

        charts = {
            "journey_flow": chart_journey_flow,
            "model_comparison": chart_model_comparison,
            "channel_attribution": chart_channel_attribution,
            "role_contribution": chart_role,
            "top_paths": chart_top_paths,
            "path_length": chart_path_length,
            "time_to_conversion": chart_ttc,
            "assisted": chart_assisted,
            "roi": chart_roi,
            "sensitivity": chart_sensitivity,
        }

        results = {
            "status": "ok",
            "overview": overview,
            "touchpoint_table": touchpoint_table,
            "model_comparison_table": model_comparison_table,
            "model_labels": MODEL_LABELS,
            "top_channel_diff": top_channel_diff,
            "channel_attribution_table": channel_attribution_table,
            "touchpoint_role_table": touchpoint_role_table,
            "conversion_path_table": conversion_path_table,
            "path_length_table": path_length_table,
            "avg_path_length": avg_path_length,
            "median_path_length": median_path_length,
            "time_to_conversion_table": time_to_conversion_table,
            "avg_days_to_conversion": avg_days_to_conversion,
            "median_days_to_conversion": median_days_to_conversion,
            "assisted_table": assisted_table,
            "roi_table": roi_table,
            "roi_note": roi_note,
            "segment_table": segment_table,
            "segment_note": segment_note,
            "conv_type_table": conv_type_table,
            "conv_type_note": conv_type_note,
            "sensitivity_table": sensitivity_table,
            "charts": charts,
        }

        print(json.dumps({"results": results}))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
