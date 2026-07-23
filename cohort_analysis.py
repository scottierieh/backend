#!/usr/bin/env python3
"""Cohort Analysis — customer retention over time by acquisition cohort.
pandas / numpy / matplotlib / seaborn.

Groups every customer into an acquisition cohort (default: the calendar month
of their first transaction; optionally a categorical column such as
acquisition channel/campaign/product/region/segment if the caller passes
`cohort_by`), then tracks, period by period (months since acquisition), what
fraction of each cohort is still active and how much revenue it generates.
Produces 13 additive sections:
  1. Cohort Overview (KPI cards)
  2. Cohort Definition (Step 1/2 table — cohort / definition / customer count)
  3. Retention Cohort Matrix (the retention triangle heatmap)
  4. Retention Curve (one line per cohort, capped to ~10 most recent)
  5. Cohort Retention Comparison (M1/M3/M6/M12 milestone table + grouped bar)
  6. Revenue Cohort Analysis (cumulative revenue line + per-customer revenue heatmap)
  7. Revenue Retention (customer-count vs revenue retention divergence)
  8. Purchase Frequency Cohort (avg orders/customer by cohort x period)
  9. Cohort by Acquisition Channel (conditional on a channel column)
  10. Cohort by Product (conditional on a first-purchase-product column)
  11. Cohort by Customer Segment (conditional on an existing segment column — no new segments created)
  12. Cohort Size Trend (new customers per acquisition cohort over time)
  13. Cohort Comparison (focused two-cohort side-by-side drill-down)

Scope note: this is customer-retention-over-time by acquisition cohort. It
does NOT do funnel/conversion-step analysis (separate Funnel Analysis page)
and does NOT create new customer segments (that is RFM / Customer
Segmentation's job) — section 11 only uses a segment column if one already
exists in the input data.

Input (JSON via stdin):
    data            : list[dict]           one row per transaction
    customer_id_col : str
    date_col        : str                  transaction date
    amount_col      : str | None           revenue/amount column (optional but needed for sections 6-7)
    cohort_by       : str | None           categorical column to define cohorts by instead of
                                            first-purchase-month (e.g. channel/campaign/product/region/segment)
    channel_col     : str | None           acquisition channel/source column (section 9)
    product_col     : str | None           first-purchase product column (section 10)
    segment_col     : str | None           existing customer-segment column (section 11)
    max_periods     : int                  columns in the retention matrix / curve (default 12)
    compare_cohort_a, compare_cohort_b : str | None   cohorts to drill into for section 13
                                                        (default: earliest vs. latest-with->=6mo history)

Output: {"results": {..., "charts": {name: "data:image/png;base64,..."}}, "plot": <matrix heatmap, convenience>}
"""
import sys
import json
import warnings
import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

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
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def month_diff(later, earlier):
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def main():
    raw = sys.stdin.read()
    payload = json.loads(raw)

    data = payload.get("data")
    customer_id_col = payload.get("customer_id_col") or payload.get("id_col")
    date_col = payload.get("date_col")
    amount_col = payload.get("amount_col")
    cohort_by = payload.get("cohort_by") or None  # None => first-purchase-month
    channel_col = payload.get("channel_col") or None
    product_col = payload.get("product_col") or None
    segment_col = payload.get("segment_col") or None
    max_periods = int(payload.get("max_periods") or 12)
    cmp_a = payload.get("compare_cohort_a") or None
    cmp_b = payload.get("compare_cohort_b") or None

    if not data or not customer_id_col or not date_col:
        print(json.dumps({"error": "data, customer_id_col and date_col are required"}))
        sys.exit(1)

    df = pd.DataFrame(data)
    if customer_id_col not in df.columns or date_col not in df.columns:
        print(json.dumps({"error": f"Columns not found: {customer_id_col}, {date_col}"}))
        sys.exit(1)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, customer_id_col])
    if amount_col and amount_col in df.columns:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    else:
        amount_col = None

    if df.empty:
        print(json.dumps({"error": "No valid rows after parsing dates"}))
        sys.exit(1)

    df = df.sort_values(date_col)
    df["_period_month"] = df[date_col].values.astype("datetime64[M]")

    # ---- Cohort assignment ----------------------------------------------
    first_purchase = df.groupby(customer_id_col)[date_col].min()
    first_month = first_purchase.dt.to_period("M").dt.to_timestamp()

    using_categorical = bool(cohort_by and cohort_by in df.columns)
    if using_categorical:
        # cohort by first observed value of the categorical column, per customer
        first_row_idx = df.groupby(customer_id_col)[date_col].idxmin()
        cat_by_cust = df.loc[first_row_idx].set_index(customer_id_col)[cohort_by]
        cohort_label = cat_by_cust.astype(str)
        cohort_month_of_cust = first_month  # still needed for revenue/period math
        cohort_definition_desc = f"acquisition {cohort_by}"
    else:
        cohort_label = first_month.dt.strftime("%Y-%m")
        cohort_month_of_cust = first_month
        cohort_definition_desc = "first purchase month"

    cust_meta = pd.DataFrame({
        "cohort": cohort_label,
        "cohort_month": cohort_month_of_cust,
    })
    df = df.merge(cust_meta, left_on=customer_id_col, right_index=True, how="left")

    # period-since-acquisition, always relative to the customer's actual first-purchase MONTH
    # (even when the cohort *label* is categorical, "months since acquisition" still needs a clock)
    df["_period"] = df.apply(lambda r: month_diff(r["_period_month"].to_pydatetime(), r["cohort_month"].to_pydatetime()), axis=1)
    df = df[df["_period"] >= 0]

    data_max_month = df["_period_month"].max()

    # ---- cohort ordering + reachable-periods per cohort ------------------
    if using_categorical:
        cohort_order = sorted(cust_meta["cohort"].dropna().unique().tolist())
        cohort_first_month = cust_meta.groupby("cohort")["cohort_month"].min()
    else:
        cohort_order = sorted(cust_meta["cohort"].dropna().unique().tolist())
        cohort_first_month = pd.to_datetime(pd.Series(cohort_order) + "-01").to_list()
        cohort_first_month = pd.Series(cohort_first_month, index=cohort_order)

    max_reachable = {
        c: max(0, month_diff(data_max_month.to_pydatetime(), cohort_first_month[c].to_pydatetime()))
        for c in cohort_order
    }

    periods = list(range(0, max_periods + 1))

    cohort_sizes = cust_meta.groupby("cohort").size().to_dict()

    # ---- ① Cohort Definition table ---------------------------------------
    cohort_definition_table = [
        {"cohort": c, "definition": cohort_definition_desc, "customer_count": int(cohort_sizes.get(c, 0))}
        for c in cohort_order
    ]

    # ---- ③ Retention Cohort Matrix (customer-count retention) -------------
    retention_matrix = []
    revenue_retention_lookup = {}
    frequency_matrix = []
    revenue_heatmap_matrix = []

    for c in cohort_order:
        size = cohort_sizes.get(c, 0)
        cust_ids_in_cohort = cust_meta.index[cust_meta["cohort"] == c]
        sub = df[df[customer_id_col].isin(cust_ids_in_cohort)]
        row_ret, row_freq, row_rev_pc = [], [], []
        reachable = max_reachable[c]
        for p in periods:
            if p > reachable:
                row_ret.append(None)
                row_freq.append(None)
                row_rev_pc.append(None)
                continue
            active = sub[sub["_period"] == p]
            n_active = active[customer_id_col].nunique()
            row_ret.append(_fin(n_active / size, 4) if size else None)
            n_orders = len(active)
            row_freq.append(_fin(n_orders / size, 3) if size else None)
            if amount_col:
                rev = active[amount_col].sum()
                row_rev_pc.append(_fin(rev / size, 2) if size else None)
            else:
                row_rev_pc.append(None)
        retention_matrix.append({"cohort": c, "size": int(size), "retention": row_ret})
        frequency_matrix.append({"cohort": c, "size": int(size), "avg_orders": row_freq})
        revenue_heatmap_matrix.append({"cohort": c, "size": int(size), "revenue_per_customer": row_rev_pc})

    # ---- ④ Retention Curve (cap to ~10 most recent cohorts) ---------------
    RETENTION_CURVE_CAP = 10
    curve_capped = len(cohort_order) > RETENTION_CURVE_CAP
    curve_cohorts = cohort_order[-RETENTION_CURVE_CAP:] if curve_capped else cohort_order
    retention_curve = []
    for c in curve_cohorts:
        row = next(r for r in retention_matrix if r["cohort"] == c)
        pts = [{"period": p, "retention": row["retention"][i]} for i, p in enumerate(periods) if row["retention"][i] is not None]
        retention_curve.append({"cohort": c, "points": pts})

    # ---- ⑤ Cohort Retention Comparison (M1/M3/M6/M12) ---------------------
    milestones = [1, 3, 6, 12]

    def milestone_val(row, m):
        if m >= len(row["retention"]):
            return None
        return row["retention"][m]

    retention_comparison = []
    for row in retention_matrix:
        retention_comparison.append({
            "cohort": row["cohort"],
            "size": row["size"],
            "m1": milestone_val(row, 1),
            "m3": milestone_val(row, 3),
            "m6": milestone_val(row, 6),
            "m12": milestone_val(row, 12),
        })

    # ---- ⑥ Revenue Cohort Analysis (cumulative revenue per cohort) --------
    revenue_cohort_cumulative = []
    if amount_col:
        for c in cohort_order:
            cust_ids_in_cohort = cust_meta.index[cust_meta["cohort"] == c]
            sub = df[df[customer_id_col].isin(cust_ids_in_cohort)]
            reachable = max_reachable[c]
            cum = 0.0
            pts = []
            for p in periods:
                if p > reachable:
                    break
                cum += sub[sub["_period"] == p][amount_col].sum()
                pts.append({"period": p, "cumulative_revenue": _fin(cum, 2)})
            revenue_cohort_cumulative.append({"cohort": c, "points": pts})

    # ---- ⑦ Revenue Retention (customer-count vs revenue retention) --------
    revenue_retention = []
    if amount_col:
        for c in cohort_order:
            size = cohort_sizes.get(c, 0)
            cust_ids_in_cohort = cust_meta.index[cust_meta["cohort"] == c]
            sub = df[df[customer_id_col].isin(cust_ids_in_cohort)]
            reachable = max_reachable[c]
            m0_rev = sub[sub["_period"] == 0][amount_col].sum()
            entry = {"cohort": c, "size": int(size)}
            for m in [1, 3, 6]:
                if m > reachable or m0_rev <= 0:
                    entry[f"m{m}_customer_retention"] = None
                    entry[f"m{m}_revenue_retention"] = None
                    continue
                act = sub[sub["_period"] == m]
                n_active = act[customer_id_col].nunique()
                rev_m = act[amount_col].sum()
                entry[f"m{m}_customer_retention"] = _fin(n_active / size, 4) if size else None
                entry[f"m{m}_revenue_retention"] = _fin(rev_m / m0_rev, 4) if m0_rev else None
            revenue_retention.append(entry)

    # ---- ⑧ Purchase Frequency Cohort (already computed: frequency_matrix) -

    # ---- ⑨ Cohort by Acquisition Channel (conditional) ---------------------
    channel_cohort = None
    channel_note = None
    if channel_col and channel_col in df.columns:
        first_row_idx = df.groupby(customer_id_col)[date_col].idxmin()
        chan_by_cust = df.loc[first_row_idx].set_index(customer_id_col)[channel_col].astype(str)
        channel_cohort = []
        for ch in sorted(chan_by_cust.dropna().unique().tolist()):
            custs = chan_by_cust.index[chan_by_cust == ch]
            size = len(custs)
            sub = df[df[customer_id_col].isin(custs)]
            # reachable periods for this group = min over its cohorts' reachable, conservative
            first_dates = first_month.loc[first_month.index.isin(custs)]
            oldest = first_dates.min()
            reach = max(0, month_diff(data_max_month.to_pydatetime(), oldest.to_pydatetime())) if pd.notna(oldest) else 0
            row = {"channel": ch, "customers": int(size)}
            for m in [1, 3, 6]:
                if m > reach or size == 0:
                    row[f"m{m}"] = None
                    continue
                n_active = sub[sub["_period"] == m][customer_id_col].nunique()
                row[f"m{m}"] = _fin(n_active / size, 4)
            channel_cohort.append(row)
    else:
        channel_note = "No acquisition channel/source column was provided or detected — section skipped."

    # ---- ⑩ Cohort by Product (conditional) ---------------------------------
    product_cohort = None
    product_note = None
    if product_col and product_col in df.columns:
        first_row_idx = df.groupby(customer_id_col)[date_col].idxmin()
        prod_by_cust = df.loc[first_row_idx].set_index(customer_id_col)[product_col].astype(str)
        product_cohort = []
        for pr in sorted(prod_by_cust.dropna().unique().tolist()):
            custs = prod_by_cust.index[prod_by_cust == pr]
            size = len(custs)
            sub = df[df[customer_id_col].isin(custs)]
            first_dates = first_month.loc[first_month.index.isin(custs)]
            oldest = first_dates.min()
            reach = max(0, month_diff(data_max_month.to_pydatetime(), oldest.to_pydatetime())) if pd.notna(oldest) else 0
            row = {"product": pr, "customers": int(size)}
            for m in [1, 3, 6]:
                if m > reach or size == 0:
                    row[f"m{m}"] = None
                    continue
                n_active = sub[sub["_period"] == m][customer_id_col].nunique()
                row[f"m{m}"] = _fin(n_active / size, 4)
            product_cohort.append(row)
    else:
        product_note = "No first-purchase-product column was provided or detected — section skipped."

    # ---- ⑪ Cohort by Customer Segment (conditional, no new segments) -------
    segment_cohort = None
    segment_note = None
    if segment_col and segment_col in df.columns:
        first_row_idx = df.groupby(customer_id_col)[date_col].idxmin()
        seg_by_cust = df.loc[first_row_idx].set_index(customer_id_col)[segment_col].astype(str)
        segment_cohort = []
        for sg in sorted(seg_by_cust.dropna().unique().tolist()):
            custs = seg_by_cust.index[seg_by_cust == sg]
            size = len(custs)
            sub = df[df[customer_id_col].isin(custs)]
            first_dates = first_month.loc[first_month.index.isin(custs)]
            oldest = first_dates.min()
            reach = max(0, month_diff(data_max_month.to_pydatetime(), oldest.to_pydatetime())) if pd.notna(oldest) else 0
            row = {"segment": sg, "customers": int(size)}
            for m in [1, 3, 6]:
                if m > reach or size == 0:
                    row[f"m{m}"] = None
                    continue
                n_active = sub[sub["_period"] == m][customer_id_col].nunique()
                row[f"m{m}"] = _fin(n_active / size, 4)
            segment_cohort.append(row)
    else:
        segment_note = "No existing customer-segment column was provided or detected — section skipped (this analysis does not create new segments)."

    # ---- ⑫ Cohort Size Trend -----------------------------------------------
    cohort_size_trend = [{"cohort": c, "new_customers": int(cohort_sizes.get(c, 0))} for c in cohort_order]

    # ---- ⑬ Cohort Comparison (two-cohort drill-down) ------------------------
    eligible_for_compare = [c for c in cohort_order if max_reachable[c] >= 6]
    if not cmp_a or cmp_a not in cohort_order:
        cmp_a = eligible_for_compare[0] if eligible_for_compare else cohort_order[0]
    if not cmp_b or cmp_b not in cohort_order:
        cmp_b = eligible_for_compare[-1] if len(eligible_for_compare) > 1 else cohort_order[-1]

    def cohort_snapshot(c):
        row = next(r for r in retention_matrix if r["cohort"] == c)
        size = row["size"]
        rev_row = next((r for r in revenue_retention if r["cohort"] == c), None) if amount_col else None
        rpu_m0 = None
        if amount_col:
            cust_ids_in_cohort = cust_meta.index[cust_meta["cohort"] == c]
            sub = df[df[customer_id_col].isin(cust_ids_in_cohort)]
            m0_rev = sub[sub["_period"] == 0][amount_col].sum()
            rpu_m0 = _fin(m0_rev / size, 2) if size else None
        return {
            "cohort": c, "size": size,
            "m1": milestone_val(row, 1), "m3": milestone_val(row, 3), "m6": milestone_val(row, 6),
            "revenue_per_user_m0": rpu_m0,
        }

    snap_a, snap_b = cohort_snapshot(cmp_a), cohort_snapshot(cmp_b)

    def diff_metric(label, key_a, key_b, is_pct):
        va, vb = snap_a.get(key_a), snap_b.get(key_b)
        if va is None or vb is None:
            d = None
        else:
            d = _fin(vb - va, 4 if is_pct else 2)
        return {"metric": label, "cohort_a_value": va, "cohort_b_value": vb, "diff": d}

    cohort_comparison = {
        "cohort_a": cmp_a, "cohort_b": cmp_b,
        "size_a": snap_a["size"], "size_b": snap_b["size"],
        "metrics": [
            diff_metric("M1 retention", "m1", "m1", True),
            diff_metric("M3 retention", "m3", "m3", True),
            diff_metric("M6 retention", "m6", "m6", True),
            diff_metric("Revenue per user (M0)", "revenue_per_user_m0", "revenue_per_user_m0", False),
        ],
    }

    # ---- ① Overview KPIs -----------------------------------------------------
    n_customers = int(cust_meta.shape[0])
    n_cohorts = len(cohort_order)
    latest_cohort = cohort_order[-1] if cohort_order else None
    new_customers_latest = int(cohort_sizes.get(latest_cohort, 0)) if latest_cohort else 0

    m3_vals = [rc["m3"] for rc in retention_comparison if rc["m3"] is not None]
    avg_retention_m3 = _fin(float(np.mean(m3_vals)), 4) if m3_vals else None
    m1_vals = [rc["m1"] for rc in retention_comparison if rc["m1"] is not None]
    m6_vals = [rc["m6"] for rc in retention_comparison if rc["m6"] is not None]
    m1_blended = _fin(float(np.mean(m1_vals)), 4) if m1_vals else None
    m6_blended = _fin(float(np.mean(m6_vals)), 4) if m6_vals else None

    best_cohort, best_cohort_val = None, None
    for rc in retention_comparison:
        v = rc["m3"] if rc["m3"] is not None else rc["m1"]
        if v is not None and (best_cohort_val is None or v > best_cohort_val):
            best_cohort, best_cohort_val = rc["cohort"], v

    overview = {
        "n_customers": n_customers,
        "n_cohorts": n_cohorts,
        "new_customers_latest_cohort": new_customers_latest,
        "latest_cohort": latest_cohort,
        "avg_retention_m3": avg_retention_m3,
        "m1_retention_blended": m1_blended,
        "m3_retention_blended": avg_retention_m3,
        "m6_retention_blended": m6_blended,
        "best_cohort": best_cohort,
        "best_cohort_retention": _fin(best_cohort_val, 4) if best_cohort_val is not None else None,
        "has_revenue": bool(amount_col),
        "cohort_definition": cohort_definition_desc,
    }

    setup = {
        "customer_id_col": customer_id_col, "date_col": date_col, "amount_col": amount_col,
        "cohort_by": cohort_by if using_categorical else None,
        "channel_col": channel_col if channel_cohort is not None else None,
        "product_col": product_col if product_cohort is not None else None,
        "segment_col": segment_col if segment_cohort is not None else None,
        "max_periods": max_periods,
    }

    # =========================================================================
    # Charts
    # =========================================================================
    charts = {}

    # ③ retention heatmap
    mat = np.array([[np.nan if v is None else v for v in row["retention"]] for row in retention_matrix], dtype=float)
    fig, ax = plt.subplots(figsize=(min(1.0 + 0.55 * len(periods), 14), max(3.0, 0.42 * len(cohort_order) + 1.2)))
    sns.heatmap(mat * 100, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Retention %"},
                xticklabels=[f"M{p}" for p in periods], yticklabels=cohort_order, ax=ax, vmin=0, vmax=100,
                linewidths=0.4, linecolor="white")
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Acquisition cohort")
    ax.set_title("Retention Cohort Matrix")
    charts["retention_heatmap"] = _png(fig)

    # ④ retention curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, rc in enumerate(retention_curve):
        xs = [p["period"] for p in rc["points"]]
        ys = [p["retention"] * 100 for p in rc["points"]]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=rc["cohort"], color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Retention %")
    ax.set_title("Retention Curve" + (" (10 most recent cohorts)" if curve_capped else ""))
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    charts["retention_curve"] = _png(fig)

    # ⑤ milestone grouped bar
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n_cohorts), 5))
    idx = np.arange(n_cohorts)
    width = 0.2
    for j, m in enumerate(["m1", "m3", "m6", "m12"]):
        vals = [(rc[m] * 100 if rc[m] is not None else 0) for rc in retention_comparison]
        ax.bar(idx + (j - 1.5) * width, vals, width, label=m.upper(), color=PALETTE[j])
    ax.set_xticks(idx)
    ax.set_xticklabels([rc["cohort"] for rc in retention_comparison], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Retention %")
    ax.set_title("Cohort Retention Comparison (M1/M3/M6/M12)")
    ax.legend()
    charts["retention_comparison"] = _png(fig)

    # ⑥ revenue: cumulative line + per-customer revenue heatmap
    if amount_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, rc in enumerate(revenue_cohort_cumulative):
            xs = [p["period"] for p in rc["points"]]
            ys = [p["cumulative_revenue"] for p in rc["points"]]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=rc["cohort"], color=PALETTE[i % len(PALETTE)])
        ax.set_xlabel("Months since acquisition")
        ax.set_ylabel("Cumulative revenue")
        ax.set_title("Cumulative Revenue by Cohort")
        ax.legend(fontsize=6, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)
        charts["revenue_cumulative"] = _png(fig)

        rmat = np.array([[np.nan if v is None else v for v in row["revenue_per_customer"]] for row in revenue_heatmap_matrix], dtype=float)
        fig, ax = plt.subplots(figsize=(min(1.0 + 0.55 * len(periods), 14), max(3.0, 0.42 * len(cohort_order) + 1.2)))
        sns.heatmap(rmat, annot=True, fmt=".0f", cmap="Greens", cbar_kws={"label": "Revenue / customer"},
                    xticklabels=[f"M{p}" for p in periods], yticklabels=cohort_order, ax=ax, linewidths=0.4, linecolor="white")
        ax.set_xlabel("Months since acquisition")
        ax.set_ylabel("Acquisition cohort")
        ax.set_title("Revenue per Customer by Cohort (Heatmap)")
        charts["revenue_heatmap"] = _png(fig)

    # ⑦ revenue retention vs customer retention divergence
    if amount_col and revenue_retention:
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(revenue_retention)), 5))
        idx2 = np.arange(len(revenue_retention))
        cust_m3 = [(r["m3_customer_retention"] * 100 if r["m3_customer_retention"] is not None else 0) for r in revenue_retention]
        rev_m3 = [(r["m3_revenue_retention"] * 100 if r["m3_revenue_retention"] is not None else 0) for r in revenue_retention]
        ax.bar(idx2 - 0.2, cust_m3, 0.4, label="Customer retention (M3)", color=BLUE)
        ax.bar(idx2 + 0.2, rev_m3, 0.4, label="Revenue retention (M3)", color=GREEN)
        ax.set_xticks(idx2)
        ax.set_xticklabels([r["cohort"] for r in revenue_retention], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("%")
        ax.set_title("Revenue vs. Customer Retention at M3")
        ax.legend()
        charts["revenue_vs_customer_retention"] = _png(fig)

    # ⑧ frequency heatmap
    fmat = np.array([[np.nan if v is None else v for v in row["avg_orders"]] for row in frequency_matrix], dtype=float)
    fig, ax = plt.subplots(figsize=(min(1.0 + 0.55 * len(periods), 14), max(3.0, 0.42 * len(cohort_order) + 1.2)))
    sns.heatmap(fmat, annot=True, fmt=".2f", cmap="Purples", cbar_kws={"label": "Avg orders/customer"},
                xticklabels=[f"M{p}" for p in periods], yticklabels=cohort_order, ax=ax, linewidths=0.4, linecolor="white")
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Acquisition cohort")
    ax.set_title("Purchase Frequency Cohort")
    charts["frequency_heatmap"] = _png(fig)

    # ⑨ channel retention
    if channel_cohort:
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(channel_cohort)), 5))
        idx3 = np.arange(len(channel_cohort))
        for j, m in enumerate(["m1", "m3", "m6"]):
            vals = [(row[m] * 100 if row[m] is not None else 0) for row in channel_cohort]
            ax.bar(idx3 + (j - 1) * 0.25, vals, 0.25, label=m.upper(), color=PALETTE[j])
        ax.set_xticks(idx3)
        ax.set_xticklabels([row["channel"] for row in channel_cohort], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Retention %")
        ax.set_title("Retention by Acquisition Channel")
        ax.legend()
        charts["channel_retention"] = _png(fig)

    # ⑩ product retention
    if product_cohort:
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(product_cohort)), 5))
        idx4 = np.arange(len(product_cohort))
        for j, m in enumerate(["m1", "m3", "m6"]):
            vals = [(row[m] * 100 if row[m] is not None else 0) for row in product_cohort]
            ax.bar(idx4 + (j - 1) * 0.25, vals, 0.25, label=m.upper(), color=PALETTE[j])
        ax.set_xticks(idx4)
        ax.set_xticklabels([row["product"] for row in product_cohort], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Retention %")
        ax.set_title("Retention by First Product")
        ax.legend()
        charts["product_retention"] = _png(fig)

    # ⑪ segment retention
    if segment_cohort:
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(segment_cohort)), 5))
        idx5 = np.arange(len(segment_cohort))
        for j, m in enumerate(["m1", "m3", "m6"]):
            vals = [(row[m] * 100 if row[m] is not None else 0) for row in segment_cohort]
            ax.bar(idx5 + (j - 1) * 0.25, vals, 0.25, label=m.upper(), color=PALETTE[j])
        ax.set_xticks(idx5)
        ax.set_xticklabels([row["segment"] for row in segment_cohort], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Retention %")
        ax.set_title("Retention by Customer Segment")
        ax.legend()
        charts["segment_retention"] = _png(fig)

    # ⑫ cohort size trend
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n_cohorts), 4.5))
    ax.bar([r["cohort"] for r in cohort_size_trend], [r["new_customers"] for r in cohort_size_trend], color=BLUE)
    ax.set_xticklabels([r["cohort"] for r in cohort_size_trend], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("New customers")
    ax.set_title("Cohort Size Trend")
    charts["cohort_size_trend"] = _png(fig)

    # ⑬ cohort comparison bar
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [m["metric"] for m in cohort_comparison["metrics"]]
    a_vals, b_vals = [], []
    for m in cohort_comparison["metrics"]:
        is_pct = "retention" in m["metric"].lower()
        a_vals.append((m["cohort_a_value"] or 0) * (100 if is_pct else 1))
        b_vals.append((m["cohort_b_value"] or 0) * (100 if is_pct else 1))
    idx6 = np.arange(len(labels))
    ax.bar(idx6 - 0.2, a_vals, 0.4, label=cmp_a, color=BLUE)
    ax.bar(idx6 + 0.2, b_vals, 0.4, label=cmp_b, color=AMBER)
    ax.set_xticks(idx6)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"Cohort Comparison: {cmp_a} vs. {cmp_b}")
    ax.legend()
    charts["cohort_comparison"] = _png(fig)

    results = {
        "setup": setup,
        "overview": overview,
        "cohort_definition_table": cohort_definition_table,
        "periods": periods,
        "retention_matrix": retention_matrix,
        "retention_curve": retention_curve,
        "retention_curve_capped": curve_capped,
        "retention_curve_cap_n": RETENTION_CURVE_CAP,
        "retention_comparison": retention_comparison,
        "revenue_cohort_cumulative": revenue_cohort_cumulative,
        "revenue_heatmap_matrix": revenue_heatmap_matrix,
        "revenue_retention": revenue_retention,
        "frequency_matrix": frequency_matrix,
        "channel_cohort": channel_cohort,
        "channel_note": channel_note,
        "product_cohort": product_cohort,
        "product_note": product_note,
        "segment_cohort": segment_cohort,
        "segment_note": segment_note,
        "cohort_size_trend": cohort_size_trend,
        "cohort_comparison": cohort_comparison,
        "charts": charts,
    }

    print(json.dumps({"results": results, "plot": charts.get("retention_heatmap")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
