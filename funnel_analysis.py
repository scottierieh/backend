#!/usr/bin/env python3
"""Funnel Analysis — within-journey step-by-step drop-off for ONE defined
multi-step funnel. pandas / numpy / matplotlib / seaborn.

Each row is one event: a user reaching a named step of a fixed, ordered
funnel (e.g. Visit -> Product View -> Add to Cart -> Checkout -> Purchase) at
a given timestamp. Optional columns unlock conditional sections: an existing
segment column, a traffic-source/channel column, a device/platform column,
and a revenue/value column at the final step.

Produces 15 additive sections:
  1. Funnel Overview (KPI cards)
  2. Funnel Visualization (funnel/bar chart, centerpiece)
  3. Funnel Conversion Table
  4. Drop-off Analysis (table + chart, largest drop-off called out)
  5. Conversion Rate by Step (standalone bar chart)
  6. Time to Conversion (conditional on granular-enough timestamps)
  7. Funnel by Segment (conditional on existing segment column)
  8. Funnel by Traffic Source (conditional on channel column)
  9. Device / Platform Funnel (conditional on device column)
  10. Funnel Comparison (two time halves, or two segments/channels — whichever
      dimension is available)
  11. Funnel Trend Over Time (conditional on enough calendar range)
  12. Funnel Heatmap (segment/channel/device x step conversion matrix)
  13. Funnel Leakage Analysis (counterfactual: potential conversions if each
      transition matched the best-performing transition's rate)
  14. Funnel Optimization Simulator (+5pp per step, incremental final-step
      conversions, holding all other steps constant)
  15. Revenue Funnel (conditional on a revenue/value column at the final step)

Scope note: this is within-journey step-by-step drop-off analysis for ONE
defined funnel. It does NOT do multi-touch attribution modeling (separate
Attribution Analysis page), does NOT create new customer segments (RFM /
Customer Segmentation's job), and does NOT do A/B significance testing
(A/B Test Analysis's job).

Input (JSON via stdin):
    data          : list[dict]     one row per funnel event
    id_col        : str            user/session id
    step_col      : str            step/event name
    timestamp_col : str            event timestamp
    step_order    : list[str]      the funnel stages, first to last
    segment_col   : str | None     existing customer-segment column (section 7)
    channel_col   : str | None     traffic-source/channel column (section 8)
    device_col    : str | None     device/platform column (section 9)
    revenue_col   : str | None     revenue/value column, read from the final step's row (section 15)

Output: {"results": {..., "charts": {name: "data:image/png;base64,..."}}, "plot": <funnel chart, convenience>}
"""
import sys
import json
import warnings
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#9333ea"
TEAL = "#0d9488"
PALETTE = [BLUE, GREEN, AMBER, PURPLE, RED, TEAL, "#be185d", "#65a30d", "#0369a1", "#a16207"]


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def main():
    payload = json.load(sys.stdin)
    data = payload.get("data") or []
    id_col = payload["id_col"]
    step_col = payload["step_col"]
    ts_col = payload["timestamp_col"]
    step_order = payload["step_order"]
    segment_col = payload.get("segment_col") or None
    channel_col = payload.get("channel_col") or None
    device_col = payload.get("device_col") or None
    revenue_col = payload.get("revenue_col") or None

    if not data:
        print(json.dumps({"error": "No data provided."})); return
    if not step_order or len(step_order) < 2:
        print(json.dumps({"error": "Need at least 2 ordered funnel steps."})); return

    df = pd.DataFrame(data)
    for c in (id_col, step_col, ts_col):
        if c not in df.columns:
            print(json.dumps({"error": f"Column '{c}' not found."})); return

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col, id_col, step_col])
    df = df[df[step_col].isin(step_order)].copy()
    if df.empty:
        print(json.dumps({"error": "No rows matched the given step_order values."})); return

    step_rank = {s: i for i, s in enumerate(step_order)}
    df["_rank"] = df[step_col].map(step_rank)
    n_steps = len(step_order)

    has_segment = bool(segment_col) and segment_col in df.columns
    has_channel = bool(channel_col) and channel_col in df.columns
    has_device = bool(device_col) and device_col in df.columns
    has_revenue = bool(revenue_col) and revenue_col in df.columns

    # one row per user: the furthest step reached + the timestamp of each step reached (first occurrence)
    # build a user x step first-timestamp matrix
    first_ts = df.sort_values(ts_col).groupby([id_col, "_rank"])[ts_col].first().reset_index()
    pivot = first_ts.pivot(index=id_col, columns="_rank", values=ts_col)
    pivot = pivot.reindex(columns=range(n_steps))

    # attach user-level dims (first-seen value)
    dims = df.sort_values(ts_col).groupby(id_col).first()
    users = pivot.index

    def reached_mask(rank):
        return pivot[rank].notna()

    top_users = int(reached_mask(0).sum())
    final_users = int(reached_mask(n_steps - 1).sum())
    overall_conv = _fin(final_users / top_users) if top_users else None

    # ---- ③ Funnel Conversion Table -----------------------------------
    step_table = []
    prev_count = None
    for i, s in enumerate(step_order):
        cnt = int(reached_mask(i).sum())
        step_conv = _fin(cnt / prev_count) if prev_count else None
        overall = _fin(cnt / top_users) if top_users else None
        drop = _fin(1 - step_conv) if step_conv is not None else None
        dropped = int(prev_count - cnt) if prev_count is not None else 0
        step_table.append({
            "step": s, "order": i + 1, "users": cnt,
            "step_conversion": step_conv, "overall_conversion": overall,
            "drop_off": drop, "dropped": dropped,
        })
        prev_count = cnt

    # ---- ④ Drop-off Analysis -------------------------------------------
    dropoff_table = []
    for i in range(1, n_steps):
        a, b = step_order[i - 1], step_order[i]
        row = step_table[i]
        dropoff_table.append({
            "from_step": a, "to_step": b, "transition": f"{a} -> {b}",
            "drop_off": row["drop_off"], "users_lost": row["dropped"],
        })
    biggest = max(dropoff_table, key=lambda r: r["drop_off"] if r["drop_off"] is not None else -1) if dropoff_table else None

    # ---- ⑥ Time to Conversion -------------------------------------------
    completed_mask = reached_mask(0) & reached_mask(n_steps - 1)
    elapsed_total = (pivot.loc[completed_mask, n_steps - 1] - pivot.loc[completed_mask, 0]).dt.total_seconds() / 60.0
    avg_time_to_convert_min = _fin(elapsed_total.mean()) if len(elapsed_total) else None

    # granularity check: median gap between first and last event across all users > 0 and not everything at exact same instant
    span_seconds = (df[ts_col].max() - df[ts_col].min()).total_seconds()
    granular_enough = span_seconds > 0 and elapsed_total.notna().sum() >= 5 and elapsed_total.median() >= (1 / 60.0)

    step_time_table = None
    if granular_enough:
        step_time_table = []
        for i in range(1, n_steps):
            both = reached_mask(i - 1) & reached_mask(i)
            gaps = (pivot.loc[both, i] - pivot.loc[both, i - 1]).dt.total_seconds() / 60.0
            gaps = gaps[gaps >= 0]
            if len(gaps) == 0:
                continue
            step_time_table.append({
                "transition": f"{step_order[i-1]} -> {step_order[i]}",
                "n": int(len(gaps)),
                "median_minutes": _fin(gaps.median(), 2),
                "p25_minutes": _fin(gaps.quantile(0.25), 2),
                "p75_minutes": _fin(gaps.quantile(0.75), 2),
                "mean_minutes": _fin(gaps.mean(), 2),
            })

    # ---- helper: build a funnel table for an arbitrary boolean mask of users ----
    def funnel_for(mask_users):
        sub = pivot.loc[mask_users]
        out = []
        prev = None
        for i, s in enumerate(step_order):
            cnt = int(sub[i].notna().sum())
            conv = _fin(cnt / prev) if prev else None
            overall = _fin(cnt / int(sub[0].notna().sum())) if int(sub[0].notna().sum()) else None
            out.append({"step": s, "users": cnt, "step_conversion": conv, "overall_conversion": overall})
            prev = cnt
        return out

    # ---- ⑦ Funnel by Segment -------------------------------------------
    segment_funnel = None
    segment_note = None
    if has_segment:
        segments = sorted([str(x) for x in dims[segment_col].dropna().unique()])
        if 1 < len(segments) <= 12:
            segment_funnel = []
            for seg in segments:
                seg_users = dims.index[dims[segment_col].astype(str) == seg]
                mask = users.isin(seg_users)
                tbl = funnel_for(mask)
                final_conv = tbl[-1]["overall_conversion"]
                segment_funnel.append({"segment": seg, "top_users": tbl[0]["users"], "final_conversion": final_conv, "steps": tbl})
        else:
            segment_note = "Segment column has too few or too many distinct values to break out meaningfully."
    else:
        segment_note = "No segment column provided — section skipped."

    # ---- ⑧ Funnel by Traffic Source ------------------------------------
    channel_funnel = None
    channel_note = None
    if has_channel:
        channels = sorted([str(x) for x in dims[channel_col].dropna().unique()])
        if 1 < len(channels) <= 15:
            channel_funnel = []
            for ch in channels:
                ch_users = dims.index[dims[channel_col].astype(str) == ch]
                mask = users.isin(ch_users)
                tbl = funnel_for(mask)
                rev_per_user = None
                if has_revenue:
                    final_step_rows = df[(df[step_col] == step_order[-1]) & (df[id_col].isin(ch_users))]
                    vals = pd.to_numeric(final_step_rows[revenue_col], errors="coerce").dropna()
                    if tbl[0]["users"] > 0:
                        rev_per_user = _fin(vals.sum() / tbl[0]["users"], 2)
                channel_funnel.append({"channel": ch, "visitors": tbl[0]["users"], "final_conversion": tbl[-1]["overall_conversion"], "revenue_per_user": rev_per_user, "steps": tbl})
        else:
            channel_note = "Channel column has too few or too many distinct values to break out meaningfully."
    else:
        channel_note = "No traffic-source/channel column provided — section skipped."

    # ---- ⑨ Device / Platform Funnel -------------------------------------
    device_funnel = None
    device_note = None
    device_insight = None
    if has_device:
        devices = sorted([str(x) for x in dims[device_col].dropna().unique()])
        if 1 < len(devices) <= 10:
            device_funnel = []
            worst = None  # (device, transition, drop)
            for dv in devices:
                dv_users = dims.index[dims[device_col].astype(str) == dv]
                mask = users.isin(dv_users)
                tbl = funnel_for(mask)
                device_funnel.append({"device": dv, "top_users": tbl[0]["users"], "final_conversion": tbl[-1]["overall_conversion"], "steps": tbl})
                for i in range(1, n_steps):
                    sc = tbl[i]["step_conversion"]
                    if sc is not None:
                        drop = 1 - sc
                        if worst is None or drop > worst[2]:
                            worst = (dv, f"{step_order[i-1]} -> {step_order[i]}", drop)
            if worst:
                device_insight = f"The steepest single drop across devices is on {worst[0]} at the {worst[1]} step, losing {round(worst[2]*100,1)}% of that device's users who reached it."
        else:
            device_note = "Device column has too few or too many distinct values to break out meaningfully."
    else:
        device_note = "No device/platform column provided — section skipped."

    # ---- ⑩ Funnel Comparison --------------------------------------------
    comparison = None
    comparison_dimension = None
    if has_segment and segment_funnel and len(segment_funnel) >= 2:
        sf = sorted(segment_funnel, key=lambda r: r["top_users"], reverse=True)[:2]
        comparison_dimension = "segment"
        a_name, b_name = sf[0]["segment"], sf[1]["segment"]
        a_steps, b_steps = sf[0]["steps"], sf[1]["steps"]
    elif has_channel and channel_funnel and len(channel_funnel) >= 2:
        cf = sorted(channel_funnel, key=lambda r: r["visitors"], reverse=True)[:2]
        comparison_dimension = "channel"
        a_name, b_name = cf[0]["channel"], cf[1]["channel"]
        a_steps, b_steps = cf[0]["steps"], cf[1]["steps"]
    else:
        comparison_dimension = "time_period"
        mid = df[ts_col].min() + (df[ts_col].max() - df[ts_col].min()) / 2
        first_half_users = dims.index[dims[ts_col] < mid] if ts_col in dims.columns else None
        # dims holds first-seen row per user, which includes ts_col value at first event
        first_half_ids = dims.index[dims[ts_col] < mid]
        second_half_ids = dims.index[dims[ts_col] >= mid]
        a_name, b_name = "First half", "Second half"
        a_steps = funnel_for(users.isin(first_half_ids))
        b_steps = funnel_for(users.isin(second_half_ids))

    if comparison_dimension is not None:
        rows = []
        for i, s in enumerate(step_order):
            av = a_steps[i]["overall_conversion"]
            bv = b_steps[i]["overall_conversion"]
            diff = _fin(bv - av) if (av is not None and bv is not None) else None
            rows.append({"step": s, "a_conversion": av, "b_conversion": bv, "point_diff": diff})
        comparison = {"dimension": comparison_dimension, "a_label": a_name, "b_label": b_name, "rows": rows}

    # ---- ⑪ Funnel Trend Over Time ---------------------------------------
    trend = None
    span_days = (df[ts_col].max() - df[ts_col].min()).days
    if span_days >= 14:
        dims_w = dims.copy()
        dims_w["_week"] = dims_w[ts_col].dt.to_period("W").dt.start_time
        weeks = sorted(dims_w["_week"].dropna().unique())
        if len(weeks) >= 2:
            trend = []
            for wk in weeks:
                wk_ids = dims_w.index[dims_w["_week"] == wk]
                mask = users.isin(wk_ids)
                sub = pivot.loc[mask]
                top = int(sub[0].notna().sum())
                if top == 0:
                    continue
                final = int(sub[n_steps - 1].notna().sum())
                trend.append({"week": str(pd.Timestamp(wk).date()), "users": top, "final_conversion": _fin(final / top)})
    if trend is None or len(trend) < 2:
        trend = None
        trend_note = "Not enough calendar range/granularity to bucket a meaningful weekly trend."
    else:
        trend_note = None

    # ---- ⑫ Funnel Heatmap ------------------------------------------------
    heatmap = None
    heatmap_note = None
    heat_source = None
    if segment_funnel:
        heat_source = ("segment", segment_funnel, "segment")
    elif channel_funnel:
        heat_source = ("channel", channel_funnel, "channel")
    elif device_funnel:
        heat_source = ("device", device_funnel, "device")
    if heat_source:
        dim_name, rows_src, key = heat_source
        heatmap = {
            "dimension": dim_name,
            "steps": step_order,
            "rows": [{"label": r[key], "conversions": [s["overall_conversion"] for s in r["steps"]]} for r in rows_src],
        }
    else:
        heatmap_note = "No segment/channel/device column available — heatmap skipped."

    # ---- ⑬ Funnel Leakage Analysis ---------------------------------------
    valid_convs = [row["step_conversion"] for row in step_table[1:] if row["step_conversion"] is not None]
    best_rate = max(valid_convs) if valid_convs else None
    leakage = []
    for i in range(1, n_steps):
        row = step_table[i]
        prev_users = step_table[i - 1]["users"]
        actual_users = row["users"]
        potential_users = int(round(prev_users * best_rate)) if best_rate is not None else actual_users
        potential_gain = max(0, potential_users - actual_users)
        leakage.append({
            "transition": f"{step_order[i-1]} -> {step_order[i]}",
            "users_lost": row["dropped"],
            "actual_step_conversion": row["step_conversion"],
            "target_step_conversion": best_rate,
            "potential_additional_users_at_this_step": potential_gain,
        })

    # ---- ⑭ Funnel Optimization Simulator ---------------------------------
    simulator = []
    step_convs = [None] + [step_table[i]["step_conversion"] for i in range(1, n_steps)]  # index i = conv into step i
    for i in range(1, n_steps):
        base_conv = step_convs[i]
        if base_conv is None:
            continue
        improved_conv = min(1.0, base_conv + 0.05)
        # recompute downstream final count holding all other steps' conversion rates constant
        sim_counts = [step_table[0]["users"]]
        for j in range(1, n_steps):
            c = improved_conv if j == i else step_convs[j]
            sim_counts.append(sim_counts[-1] * (c if c is not None else 0))
        sim_final = sim_counts[-1]
        incremental = _fin(sim_final - final_users, 2)
        simulator.append({
            "step": f"{step_order[i-1]} -> {step_order[i]}",
            "current_conversion": base_conv,
            "simulated_conversion": _fin(improved_conv),
            "current_final_users": final_users,
            "simulated_final_users": _fin(sim_final, 2),
            "incremental_conversions": incremental,
        })
    simulator.sort(key=lambda r: r["incremental_conversions"], reverse=True)

    # ---- ⑮ Revenue Funnel --------------------------------------------------
    revenue_funnel = None
    revenue_note = None
    if has_revenue:
        final_rows = df[df[step_col] == step_order[-1]].copy()
        final_rows["_rev"] = pd.to_numeric(final_rows[revenue_col], errors="coerce")
        realized_total = _fin(final_rows["_rev"].sum(), 2)
        avg_rev = _fin(final_rows["_rev"].mean(), 2)
        # potential: if every user who reached the step BEFORE final had converted at the average realized value
        pre_final_users = step_table[-2]["users"] if n_steps >= 2 else final_users
        potential_total = _fin((avg_rev or 0) * pre_final_users, 2)
        lost_revenue = _fin((potential_total or 0) - (realized_total or 0), 2)
        revenue_funnel = {
            "realized_revenue": realized_total,
            "avg_revenue_per_converter": avg_rev,
            "users_before_final_step": pre_final_users,
            "potential_revenue_if_all_converted": potential_total,
            "lost_revenue_estimate": lost_revenue,
        }
    else:
        revenue_note = "No revenue/value column provided — section skipped."

    # ---- overview ---------------------------------------------------------
    overview = {
        "total_users": top_users,
        "final_users": final_users,
        "overall_conversion": overall_conv,
        "biggest_drop_off": biggest,
        "avg_time_to_convert_minutes": avg_time_to_convert_min,
        "n_steps": n_steps,
        "has_segment": bool(segment_funnel),
        "has_channel": bool(channel_funnel),
        "has_device": bool(device_funnel),
        "has_time_analysis": granular_enough,
        "has_trend": trend is not None,
        "has_revenue": has_revenue,
    }

    # ================= CHARTS =================
    charts = {}

    # ② funnel visualization (centerpiece)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    counts = [row["users"] for row in step_table]
    y = np.arange(n_steps)
    ax.barh(y, counts, color=PALETTE[:n_steps][::-1] if n_steps <= len(PALETTE) else BLUE)
    ax.set_yticks(y); ax.set_yticklabels(step_order); ax.invert_yaxis()
    for i, c in enumerate(counts):
        label = f"{c:,}"
        if i > 0 and step_table[i]["step_conversion"] is not None:
            label += f"  ({step_table[i]['step_conversion']*100:.1f}% step)"
        ax.text(c, i, "  " + label, va="center", fontsize=9)
    ax.set_xlabel("Users"); ax.set_title("Funnel")
    ax2 = axes[1]
    convs = [row["step_conversion"] * 100 if row["step_conversion"] is not None else 100 for row in step_table]
    ax2.bar(step_order, convs, color=GREEN)
    ax2.set_ylabel("Step conversion %"); ax2.set_title("Conversion Rate by Step")
    ax2.set_xticklabels(step_order, rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(convs):
        ax2.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    charts["funnel_overview"] = _png(fig)

    # ④ drop-off chart
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [r["transition"] for r in dropoff_table]
    vals = [r["drop_off"] * 100 if r["drop_off"] is not None else 0 for r in dropoff_table]
    colors = [RED if r is biggest else AMBER for r in dropoff_table]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Drop-off %"); ax.set_title("Drop-off by Transition (red = largest)")
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    charts["dropoff_analysis"] = _png(fig)

    # ⑥ time to conversion chart
    if step_time_table:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        labels = [r["transition"] for r in step_time_table]
        med = [r["median_minutes"] for r in step_time_table]
        p25 = [r["p25_minutes"] for r in step_time_table]
        p75 = [r["p75_minutes"] for r in step_time_table]
        ax.bar(labels, med, color=TEAL)
        err_low = [m - lo for m, lo in zip(med, p25)]
        err_high = [hi - m for m, hi in zip(med, p75)]
        ax.errorbar(labels, med, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4)
        ax.set_ylabel("Minutes (median, IQR)"); ax.set_title("Time to Convert Between Steps")
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        fig.tight_layout()
        charts["time_to_conversion"] = _png(fig)

    # ⑦/⑧/⑨ conditional funnels
    def dim_chart(rows, key, title, chartname):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for i, r in enumerate(rows):
            ys = [s["overall_conversion"] * 100 if s["overall_conversion"] is not None else 0 for s in r["steps"]]
            ax.plot(step_order, ys, marker="o", label=str(r[key]), color=PALETTE[i % len(PALETTE)])
        ax.set_ylabel("Overall conversion %"); ax.set_title(title)
        ax.set_xticklabels(step_order, rotation=25, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        charts[chartname] = _png(fig)

    if segment_funnel:
        dim_chart(segment_funnel, "segment", "Funnel by Segment", "by_segment")
    if channel_funnel:
        dim_chart(channel_funnel, "channel", "Funnel by Traffic Source", "by_channel")
    if device_funnel:
        dim_chart(device_funnel, "device", "Device / Platform Funnel", "by_device")

    # ⑩ comparison chart
    if comparison:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        idx = np.arange(n_steps)
        av = [row["a_conversion"] * 100 if row["a_conversion"] is not None else 0 for row in comparison["rows"]]
        bv = [row["b_conversion"] * 100 if row["b_conversion"] is not None else 0 for row in comparison["rows"]]
        ax.bar(idx - 0.2, av, 0.4, label=comparison["a_label"], color=BLUE)
        ax.bar(idx + 0.2, bv, 0.4, label=comparison["b_label"], color=AMBER)
        ax.set_xticks(idx); ax.set_xticklabels(step_order, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Overall conversion %"); ax.set_title(f"Funnel Comparison: {comparison['a_label']} vs {comparison['b_label']}")
        ax.legend()
        fig.tight_layout()
        charts["funnel_comparison"] = _png(fig)

    # ⑪ trend chart
    if trend:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        weeks_x = [r["week"] for r in trend]
        vals = [r["final_conversion"] * 100 if r["final_conversion"] is not None else 0 for r in trend]
        ax.plot(weeks_x, vals, marker="o", color=PURPLE)
        ax.set_ylabel("Final-step conversion %"); ax.set_title("Funnel Trend Over Time (weekly)")
        ax.set_xticklabels(weeks_x, rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        charts["funnel_trend"] = _png(fig)

    # ⑫ heatmap
    if heatmap:
        import matplotlib.colors as mcolors
        mat = np.array([[v if v is not None else np.nan for v in row["conversions"]] for row in heatmap["rows"]])
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * n_steps), max(3, 0.5 * len(heatmap["rows"]))))
        im = ax.imshow(mat * 100, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(n_steps)); ax.set_xticklabels(step_order, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(heatmap["rows"]))); ax.set_yticklabels([r["label"] for r in heatmap["rows"]])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]*100:.0f}%", ha="center", va="center", fontsize=7)
        ax.set_title(f"Funnel Heatmap by {heatmap['dimension'].title()}")
        fig.colorbar(im, ax=ax, label="Overall conversion %")
        fig.tight_layout()
        charts["funnel_heatmap"] = _png(fig)

    # ⑬ leakage chart
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [r["transition"] for r in leakage]
    vals = [r["potential_additional_users_at_this_step"] for r in leakage]
    ax.bar(labels, vals, color=AMBER)
    ax.set_ylabel("Potential additional users"); ax.set_title("Leakage: Potential Gain vs Best-Performing Transition")
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    charts["leakage_analysis"] = _png(fig)

    # ⑭ optimization simulator chart
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [r["step"] for r in simulator]
    vals = [r["incremental_conversions"] for r in simulator]
    ax.bar(labels, vals, color=GREEN)
    ax.set_ylabel("Incremental final-step conversions"); ax.set_title("Optimization Simulator: +5pp per Step")
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    charts["optimization_simulator"] = _png(fig)

    # ⑮ revenue funnel chart
    if revenue_funnel:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cats = ["Potential", "Realized"]
        vals = [revenue_funnel["potential_revenue_if_all_converted"] or 0, revenue_funnel["realized_revenue"] or 0]
        ax.bar(cats, vals, color=[AMBER, GREEN])
        ax.set_ylabel("Revenue"); ax.set_title("Revenue Funnel: Potential vs Realized")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        charts["revenue_funnel"] = _png(fig)

    results = {
        "setup": {
            "id_col": id_col, "step_col": step_col, "timestamp_col": ts_col, "step_order": step_order,
            "segment_col": segment_col, "channel_col": channel_col, "device_col": device_col, "revenue_col": revenue_col,
        },
        "overview": overview,
        "funnel_table": step_table,
        "dropoff_table": dropoff_table,
        "step_time_table": step_time_table,
        "time_note": None if granular_enough else "Timestamps are not granular enough (too coarse or too few completions) for a reliable time-to-convert breakdown — section skipped.",
        "segment_funnel": segment_funnel,
        "segment_note": segment_note,
        "channel_funnel": channel_funnel,
        "channel_note": channel_note,
        "device_funnel": device_funnel,
        "device_note": device_note,
        "device_insight": device_insight,
        "comparison": comparison,
        "trend": trend,
        "trend_note": trend_note,
        "heatmap": heatmap,
        "heatmap_note": heatmap_note,
        "leakage": leakage,
        "simulator": simulator,
        "revenue_funnel": revenue_funnel,
        "revenue_note": revenue_note,
        "charts": charts,
    }

    print(json.dumps({"results": results, "plot": charts.get("funnel_overview")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
