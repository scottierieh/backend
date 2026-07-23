#!/usr/bin/env python3
"""Cash Flow Analysis — CFO/CFI/CFF, free cash flow, cash conversion, waterfall.

Input (from cash-flow-analysis-page.tsx):
    data                      : list[dict]  one row per fiscal period
    period_col                : str  required
    net_income_col            : str  required
    ocf_col                   : str  required (operating_cash_flow)
    da_col                    : str  optional (depreciation_amortization)
    wc_col                    : str  optional (working_capital_change)
    capex_col                 : str  optional
    acquisitions_col          : str  optional
    asset_sales_col           : str  optional
    icf_col                   : str  optional (investing_cash_flow, else summed)
    debt_issuance_col         : str  optional
    debt_repayment_col        : str  optional
    share_issuance_col        : str  optional
    share_buyback_col         : str  optional
    dividends_col             : str  optional
    fcf_activity_col          : str  optional (financing_cash_flow, else summed)
    beginning_cash_col        : str  optional
    ending_cash_col           : str  optional (or net_change_in_cash_col)
    net_change_col            : str  optional
    revenue_col               : str  optional (enables CapEx/Revenue %)

Output: { "results": {...(charts nested inside)...}, "plot": <optional> }

Sections (11 total):
  1 overview, 2 trend, 3 composition, 4 operating cash flow (bridge/reconciliation),
  5 free cash flow, 6 capex analysis, 7 cash conversion, 8 cash flow quality,
  9 cash flow adequacy (conditional), 10 forecast (conditional), 11 waterfall
"""
import sys, json, io, base64
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777"]


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v is None or not np.isfinite(v):
        return None
    return round(v, nd)


def _pct(cur, prev):
    try:
        cur = float(cur); prev = float(prev)
    except (TypeError, ValueError):
        return None
    if prev == 0 or not np.isfinite(cur) or not np.isfinite(prev):
        return None
    return round((cur - prev) / abs(prev) * 100.0, 4)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def col(row, name):
    if not name:
        return np.nan
    v = row.get(name)
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided")

        c_period = p.get("period_col")
        c_ni = p.get("net_income_col")
        c_ocf = p.get("ocf_col")
        c_da = p.get("da_col")
        c_wc = p.get("wc_col")
        c_capex = p.get("capex_col")
        c_acq = p.get("acquisitions_col")
        c_asale = p.get("asset_sales_col")
        c_icf = p.get("icf_col")
        c_debt_iss = p.get("debt_issuance_col")
        c_debt_rep = p.get("debt_repayment_col")
        c_share_iss = p.get("share_issuance_col")
        c_buyback = p.get("share_buyback_col")
        c_div = p.get("dividends_col")
        c_fcf_act = p.get("fcf_activity_col")
        c_cash_beg = p.get("beginning_cash_col")
        c_cash_end = p.get("ending_cash_col")
        c_net_change = p.get("net_change_col")
        c_rev = p.get("revenue_col")

        if not c_ni or not c_ocf or not c_period:
            raise ValueError("period_col, net_income_col and ocf_col are required")

        periods = []
        for i, row in enumerate(rows):
            label = str(row.get(c_period)) if c_period else f"Period {i+1}"
            ni = col(row, c_ni)
            ocf = col(row, c_ocf)
            if not np.isfinite(ni) or not np.isfinite(ocf):
                continue
            da = col(row, c_da)
            wc = col(row, c_wc)
            capex = col(row, c_capex)
            acq = col(row, c_acq)
            asale = col(row, c_asale)
            icf_direct = col(row, c_icf)
            icf = icf_direct if np.isfinite(icf_direct) else (
                np.nansum([-capex if np.isfinite(capex) else 0, -acq if np.isfinite(acq) else 0,
                           asale if np.isfinite(asale) else 0])
                if any(np.isfinite(x) for x in [capex, acq, asale]) else np.nan)
            debt_iss = col(row, c_debt_iss)
            debt_rep = col(row, c_debt_rep)
            share_iss = col(row, c_share_iss)
            buyback = col(row, c_buyback)
            div = col(row, c_div)
            fcf_direct = col(row, c_fcf_act)
            cff = fcf_direct if np.isfinite(fcf_direct) else (
                np.nansum([debt_iss if np.isfinite(debt_iss) else 0, -debt_rep if np.isfinite(debt_rep) else 0,
                           share_iss if np.isfinite(share_iss) else 0, -buyback if np.isfinite(buyback) else 0,
                           -div if np.isfinite(div) else 0])
                if any(np.isfinite(x) for x in [debt_iss, debt_rep, share_iss, buyback, div]) else np.nan)
            cash_beg = col(row, c_cash_beg)
            cash_end_direct = col(row, c_cash_end)
            net_change_direct = col(row, c_net_change)
            net_change = net_change_direct if np.isfinite(net_change_direct) else (
                cash_end_direct - cash_beg if np.isfinite(cash_end_direct) and np.isfinite(cash_beg) else
                (np.nansum([ocf, icf, cff]) if all(np.isfinite(x) for x in [ocf, icf, cff]) else np.nan))
            cash_end = cash_end_direct if np.isfinite(cash_end_direct) else (
                cash_beg + net_change if np.isfinite(cash_beg) and np.isfinite(net_change) else np.nan)
            rev = col(row, c_rev)
            fcf = ocf - capex if np.isfinite(capex) else np.nan

            periods.append(dict(label=label, net_income=ni, ocf=ocf, da=da, wc=wc, capex=capex, acq=acq,
                                 asset_sales=asale, icf=icf, debt_issuance=debt_iss, debt_repayment=debt_rep,
                                 share_issuance=share_iss, buyback=buyback, dividends=div, cff=cff,
                                 cash_begin=cash_beg, cash_end=cash_end, net_change=net_change, revenue=rev,
                                 fcf=fcf))

        if not periods:
            raise ValueError("No valid periods (need net income & operating cash flow)")

        n = len(periods)
        labels = [pd_["label"] for pd_ in periods]
        last, first = periods[-1], periods[0]
        prev = periods[-2] if n > 1 else None

        def yoy(field):
            if not prev:
                return None
            return _pct(last[field], prev[field])

        results = {}

        # ── 1. Overview ──
        overview = {
            "operating_cash_flow": _fin(last["ocf"]), "investing_cash_flow": _fin(last["icf"]),
            "financing_cash_flow": _fin(last["cff"]), "free_cash_flow": _fin(last["fcf"]),
            "net_change_in_cash": _fin(last["net_change"]),
            "yoy": {"ocf_pct": yoy("ocf"), "icf_pct": yoy("icf"), "cff_pct": yoy("cff"),
                    "fcf_pct": yoy("fcf") if np.isfinite(last["fcf"]) and prev and np.isfinite(prev["fcf"]) else None,
                    "net_change_pct": yoy("net_change")},
        }
        results["overview"] = overview

        # ── 2. Trend ──
        trend_table = [{"period": pd_["label"], "operating_cash_flow": _fin(pd_["ocf"]),
                         "investing_cash_flow": _fin(pd_["icf"]), "financing_cash_flow": _fin(pd_["cff"])}
                        for pd_ in periods]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(labels, [pd_["ocf"] for pd_ in periods], marker="o", color=PALETTE[0], label="CFO")
        ax.plot(labels, [pd_["icf"] for pd_ in periods], marker="o", color=PALETTE[3], label="CFI")
        ax.plot(labels, [pd_["cff"] for pd_ in periods], marker="o", color=PALETTE[4], label="CFF")
        ax.axhline(0, color="#94a3b8", linewidth=0.8)
        ax.set_title("Cash Flow Trend"); ax.legend(); ax.grid(alpha=0.3)
        chart_trend = _png(fig)
        results["trend"] = {"table": trend_table}

        # ── 3. Composition (latest period breakdown of CFO/CFI/CFF drivers) ──
        cfo_parts = [("net_income", last["net_income"]), ("depreciation_amortization", last["da"]),
                     ("working_capital_change", last["wc"])]
        cfo_parts = [(k, v) for k, v in cfo_parts if np.isfinite(v)]
        cfi_parts = [("capex", -last["capex"] if np.isfinite(last["capex"]) else np.nan),
                     ("acquisitions", -last["acq"] if np.isfinite(last["acq"]) else np.nan),
                     ("asset_sales", last["asset_sales"])]
        cfi_parts = [(k, v) for k, v in cfi_parts if np.isfinite(v)]
        cff_parts = [("debt_issuance", last["debt_issuance"]),
                     ("debt_repayment", -last["debt_repayment"] if np.isfinite(last["debt_repayment"]) else np.nan),
                     ("share_issuance", last["share_issuance"]),
                     ("share_buyback", -last["buyback"] if np.isfinite(last["buyback"]) else np.nan),
                     ("dividends", -last["dividends"] if np.isfinite(last["dividends"]) else np.nan)]
        cff_parts = [(k, v) for k, v in cff_parts if np.isfinite(v)]
        comp_table = ([{"category": "CFO", "item": k, "value": _fin(v)} for k, v in cfo_parts] +
                       [{"category": "CFI", "item": k, "value": _fin(v)} for k, v in cfi_parts] +
                       [{"category": "CFF", "item": k, "value": _fin(v)} for k, v in cff_parts])
        results["composition"] = {"table": comp_table, "period": last["label"]}
        chart_comp = None
        if comp_table:
            cfo_sum = sum(v for _, v in cfo_parts) if cfo_parts else 0
            cfi_sum = sum(v for _, v in cfi_parts) if cfi_parts else 0
            cff_sum = sum(v for _, v in cff_parts) if cff_parts else 0
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            cats = ["CFO", "CFI", "CFF"]
            groups = [cfo_parts, cfi_parts, cff_parts]
            bottoms_pos = [0.0, 0.0, 0.0]
            bottoms_neg = [0.0, 0.0, 0.0]
            for gi, grp in enumerate(groups):
                for ci, (k, v) in enumerate(grp):
                    color = PALETTE[ci % len(PALETTE)]
                    if v >= 0:
                        ax.bar(cats[gi], v, bottom=bottoms_pos[gi], color=color, label=k if gi == 0 or k not in [x[0] for x in groups[0]] else None)
                        bottoms_pos[gi] += v
                    else:
                        bottoms_neg[gi] += v
                        ax.bar(cats[gi], v, bottom=bottoms_neg[gi] - v, color=color)
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set_title(f"Cash Flow Composition — {last['label']}"); ax.grid(alpha=0.3, axis="y")
            handles, hlabels = ax.get_legend_handles_labels()
            if hlabels:
                ax.legend(fontsize=7, loc="best")
            chart_comp = _png(fig)

        # ── 4. Operating cash flow analysis + NI→OCF bridge ──
        ocf_table = [{"period": pd_["label"], "net_income": _fin(pd_["net_income"]), "da": _fin(pd_["da"]),
                       "working_capital_change": _fin(pd_["wc"]), "operating_cash_flow": _fin(pd_["ocf"])}
                      for pd_ in periods]
        bridge_available = bool(np.isfinite(last["da"]) and np.isfinite(last["wc"]))
        reconciled_ocf = None
        reconciliation_gap = None
        chart_bridge = None
        if bridge_available:
            reconciled_ocf = last["net_income"] + last["da"] + last["wc"]
            reconciliation_gap = reconciled_ocf - last["ocf"]
            steps_name = ["Net Income", "+ D&A", "± Working Capital", "Operating CF"]
            steps_val = [last["net_income"], last["da"], last["wc"], last["ocf"]]
            running = 0.0
            bottoms = []
            heights = []
            for i, v in enumerate(steps_val):
                if steps_name[i] in ("Net Income", "Operating CF"):
                    bottoms.append(0); heights.append(v); running = v
                else:
                    lo = min(running, running + v)
                    bottoms.append(lo); heights.append(abs(v)); running += v
            fig, ax = plt.subplots(figsize=(7, 4.2))
            colors = [PALETTE[0], PALETTE[1] if last["da"] >= 0 else PALETTE[3],
                      PALETTE[1] if last["wc"] >= 0 else PALETTE[3], PALETTE[0]]
            ax.bar(steps_name, heights, bottom=bottoms, color=colors)
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set_title(f"Net Income → Operating Cash Flow Bridge — {last['label']}"); ax.grid(alpha=0.3, axis="y")
            chart_bridge = _png(fig)
        results["operating_cash_flow"] = {
            "table": ocf_table, "bridge_available": bridge_available,
            "reconciled_ocf": _fin(reconciled_ocf), "reported_ocf": _fin(last["ocf"]),
            "reconciliation_gap": _fin(reconciliation_gap),
            "reconciliation_note": (
                "net_income + D&A + working_capital_change reconciles closely with reported OCF."
                if bridge_available and reconciliation_gap is not None and abs(reconciliation_gap) < max(1.0, abs(last["ocf"]) * 0.02)
                else ("A material gap exists between the reconciled and reported OCF — investigate other adjustments (stock comp, deferred tax, etc.)."
                      if bridge_available else "D&A and/or working-capital-change columns not provided — bridge not computed.")),
        }

        # ── 5. Free cash flow analysis ⭐ ──
        fcf_table = [{"period": pd_["label"], "operating_cash_flow": _fin(pd_["ocf"]), "capex": _fin(pd_["capex"]),
                       "free_cash_flow": _fin(pd_["fcf"])} for pd_ in periods]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(labels, [pd_["ocf"] for pd_ in periods], marker="o", color=PALETTE[0], label="OCF")
        ax.plot(labels, [pd_["capex"] for pd_ in periods], marker="o", color=PALETTE[3], label="CapEx")
        ax.plot(labels, [pd_["fcf"] for pd_ in periods], marker="o", color=PALETTE[4], label="FCF")
        ax.axhline(0, color="#94a3b8", linewidth=0.8)
        ax.set_title("Free Cash Flow Trend"); ax.legend(); ax.grid(alpha=0.3)
        chart_fcf = _png(fig)
        results["free_cash_flow"] = {"table": fcf_table}

        # ── 6. CapEx analysis ──
        has_capex = bool(any(np.isfinite(pd_["capex"]) for pd_ in periods))
        capex_table = []
        for pd_ in periods:
            capex_to_rev = _fin(pd_["capex"] / pd_["revenue"] * 100) if np.isfinite(pd_["capex"]) and np.isfinite(pd_["revenue"]) and pd_["revenue"] else None
            capex_to_da = _fin(pd_["capex"] / pd_["da"]) if np.isfinite(pd_["capex"]) and np.isfinite(pd_["da"]) and pd_["da"] else None
            capex_table.append({"period": pd_["label"], "capex": _fin(pd_["capex"]),
                                 "capex_to_revenue_pct": capex_to_rev, "capex_to_depreciation": capex_to_da})
        chart_capex = None
        if has_capex and any(np.isfinite(pd_["da"]) for pd_ in periods):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(labels, [pd_["capex"] for pd_ in periods], marker="o", color=PALETTE[3], label="CapEx")
            ax.plot(labels, [pd_["da"] for pd_ in periods], marker="o", color=PALETTE[2], label="D&A")
            ax.set_title("CapEx vs. Depreciation"); ax.legend(); ax.grid(alpha=0.3)
            chart_capex = _png(fig)
        capex_da_last = capex_table[-1]["capex_to_depreciation"] if capex_table else None
        capex_note = (
            "CapEx > D&A: net investment implies capacity growth." if capex_da_last is not None and capex_da_last > 1
            else "CapEx < D&A: spending trails depreciation, a possible under-maintenance signal worth watching."
            if capex_da_last is not None and capex_da_last < 1 else None)
        results["capex_analysis"] = {"table": capex_table, "revenue_available": any(np.isfinite(pd_["revenue"]) for pd_ in periods),
                                       "capex_vs_depreciation_note": capex_note}

        # ── 7. Cash conversion analysis ⭐ ──
        cc_table = []
        for i, pd_ in enumerate(periods):
            ratio = _fin(pd_["ocf"] / pd_["net_income"]) if pd_["net_income"] else None
            cc_table.append({"period": pd_["label"], "net_income": _fin(pd_["net_income"]),
                              "operating_cash_flow": _fin(pd_["ocf"]), "cash_conversion_ratio": ratio})
        divergence_flag = False
        divergence_note = None
        if prev and np.isfinite(last["net_income"]) and np.isfinite(prev["net_income"]) and prev["net_income"] != 0:
            ni_g = (last["net_income"] - prev["net_income"]) / abs(prev["net_income"])
            ocf_g = (last["ocf"] - prev["ocf"]) / abs(prev["ocf"]) if prev["ocf"] else None
            if ocf_g is not None and ni_g - ocf_g > 0.10:
                divergence_flag = True
                divergence_note = f"Net income growth ({ni_g*100:.1f}%) notably outpaced OCF growth ({ocf_g*100:.1f}%) in the latest period."
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(labels, [pd_["net_income"] for pd_ in periods], marker="o", color=PALETTE[2], label="Net Income")
        ax.plot(labels, [pd_["ocf"] for pd_ in periods], marker="o", color=PALETTE[0], label="OCF")
        ax.set_title("Net Income vs. Operating Cash Flow"); ax.legend(); ax.grid(alpha=0.3)
        chart_cc = _png(fig)
        results["cash_conversion"] = {"table": cc_table, "divergence_flag": divergence_flag, "divergence_note": divergence_note}

        # ── 8. Cash flow quality (driver breakdown of OCF) ──
        quality = {}
        chart_quality = None
        if bridge_available:
            total_abs = abs(last["net_income"]) + abs(last["da"]) + abs(last["wc"])
            quality = {
                "net_income_pct_of_ocf": _fin(last["net_income"] / last["ocf"] * 100) if last["ocf"] else None,
                "da_pct_of_ocf": _fin(last["da"] / last["ocf"] * 100) if last["ocf"] else None,
                "working_capital_pct_of_ocf": _fin(last["wc"] / last["ocf"] * 100) if last["ocf"] else None,
            }
            fig, ax = plt.subplots(figsize=(6, 4.2))
            names = ["Net Income", "D&A", "Working Capital Δ"]
            vals = [last["net_income"], last["da"], last["wc"]]
            colors = [PALETTE[0], PALETTE[1] if last["da"] >= 0 else PALETTE[3], PALETTE[1] if last["wc"] >= 0 else PALETTE[3]]
            ax.bar(names, vals, color=colors)
            ax.axhline(0, color="#334155", linewidth=0.8)
            ax.set_title(f"OCF Quality / Composition — {last['label']}"); ax.grid(alpha=0.3, axis="y")
            chart_quality = _png(fig)
        results["cash_flow_quality"] = {"available": bridge_available, **quality}

        # ── 9. Cash flow adequacy (conditional) ──
        adequacy = {}
        has_capex_last = bool(np.isfinite(last["capex"]) and last["capex"] != 0)
        has_div_last = bool(np.isfinite(last["dividends"]))
        has_debt_rep_last = bool(np.isfinite(last["debt_repayment"]) and last["debt_repayment"] != 0)
        if has_capex_last:
            adequacy["ocf_to_capex_coverage"] = _fin(last["ocf"] / last["capex"])
        else:
            adequacy["ocf_to_capex_coverage"] = None
            adequacy["ocf_to_capex_skipped_note"] = "CapEx not provided — OCF/CapEx coverage skipped."
        if has_div_last and last["dividends"]:
            adequacy["fcf_to_dividends_coverage"] = _fin(last["fcf"] / last["dividends"]) if np.isfinite(last["fcf"]) else None
        else:
            adequacy["fcf_to_dividends_coverage"] = None
            adequacy["fcf_to_dividends_skipped_note"] = "Dividends column not provided — FCF/Dividends coverage skipped."
        if has_debt_rep_last:
            adequacy["fcf_to_debt_repayment_coverage"] = _fin(last["fcf"] / last["debt_repayment"]) if np.isfinite(last["fcf"]) else None
        else:
            adequacy["fcf_to_debt_repayment_coverage"] = None
            adequacy["fcf_to_debt_repayment_skipped_note"] = "Debt repayment column not provided — FCF/Debt-repayment coverage skipped."
        results["cash_flow_adequacy"] = adequacy

        # ── 10. Cash flow forecast (conditional) ──
        forecast = {}
        min_periods_for_trend = 4
        ocf_series = np.array([pd_["ocf"] for pd_ in periods], dtype=float)
        fcf_series = np.array([pd_["fcf"] for pd_ in periods if np.isfinite(pd_["fcf"])], dtype=float)
        if n >= min_periods_for_trend:
            x = np.arange(n)
            ocf_slope, ocf_intercept = np.polyfit(x, ocf_series, 1)
            ocf_diffs = np.diff(ocf_series)
            ocf_vol = float(np.std(ocf_diffs, ddof=1)) if len(ocf_diffs) > 1 else 0.0
            proj_periods = [n, n + 1, n + 2]
            ocf_forecast = [_fin(ocf_slope * xp + ocf_intercept) for xp in proj_periods]
            ocf_band = [_fin(ocf_vol * (i + 1) ** 0.5) for i in range(3)]
            fcf_forecast = None
            fcf_band = None
            if len(fcf_series) == n:
                fcf_slope, fcf_intercept = np.polyfit(x, fcf_series, 1)
                fcf_diffs = np.diff(fcf_series)
                fcf_vol = float(np.std(fcf_diffs, ddof=1)) if len(fcf_diffs) > 1 else 0.0
                fcf_forecast = [_fin(fcf_slope * xp + fcf_intercept) for xp in proj_periods]
                fcf_band = [_fin(fcf_vol * (i + 1) ** 0.5) for i in range(3)]
            forecast = {
                "method": "linear_trend",
                "formula_note": f"OLS linear trend fit on {n} historical periods; confidence band = ± historical period-over-period volatility × sqrt(horizon).",
                "ocf_forecast": ocf_forecast, "ocf_band": ocf_band,
                "fcf_forecast": fcf_forecast, "fcf_band": fcf_band,
                "horizons": [f"{last['label']}+{i+1}" for i in range(3)],
            }
        else:
            avg_growth = np.mean([np.diff(ocf_series)]) if n > 1 else 0.0
            flat_val = float(ocf_series[-1])
            vol = float(np.std(np.diff(ocf_series), ddof=1)) if n > 2 else abs(flat_val) * 0.1
            forecast = {
                "method": "flat_extrapolation",
                "formula_note": f"Fewer than {min_periods_for_trend} historical periods ({n} available) — degraded to a flat extrapolation from the latest OCF rather than a fitted trend, to avoid false precision.",
                "ocf_forecast": [_fin(flat_val)] * 3, "ocf_band": [_fin(vol), _fin(vol * 1.4), _fin(vol * 1.7)],
                "fcf_forecast": None, "fcf_band": None,
                "horizons": [f"{last['label']}+{i+1}" for i in range(3)],
            }
        results["forecast"] = forecast

        # ── 11. Cash flow waterfall ⭐ (centerpiece) ──
        wf_steps = []
        running = last["cash_begin"] if np.isfinite(last["cash_begin"]) else 0.0
        start_val = running
        wf_steps.append(("Beginning Cash", start_val, True))
        if np.isfinite(last["ocf"]):
            wf_steps.append(("+ OCF", last["ocf"], False)); running += last["ocf"]
        if np.isfinite(last["capex"]) and last["capex"] != 0:
            wf_steps.append(("− CapEx", -last["capex"], False)); running -= last["capex"]
        if np.isfinite(last["acq"]) and last["acq"] != 0:
            wf_steps.append(("− Acquisitions", -last["acq"], False)); running -= last["acq"]
        if np.isfinite(last["asset_sales"]) and last["asset_sales"] != 0:
            wf_steps.append(("+ Asset Sales", last["asset_sales"], False)); running += last["asset_sales"]
        if np.isfinite(last["debt_repayment"]) and last["debt_repayment"] != 0:
            wf_steps.append(("− Debt Repayment", -last["debt_repayment"], False)); running -= last["debt_repayment"]
        if np.isfinite(last["debt_issuance"]) and last["debt_issuance"] != 0:
            wf_steps.append(("+ Debt Issuance", last["debt_issuance"], False)); running += last["debt_issuance"]
        if np.isfinite(last["share_issuance"]) and last["share_issuance"] != 0:
            wf_steps.append(("+ Share Issuance", last["share_issuance"], False)); running += last["share_issuance"]
        if np.isfinite(last["dividends"]) and last["dividends"] != 0:
            wf_steps.append(("− Dividends", -last["dividends"], False)); running -= last["dividends"]
        if np.isfinite(last["buyback"]) and last["buyback"] != 0:
            wf_steps.append(("− Share Buyback", -last["buyback"], False)); running -= last["buyback"]
        # if no financing/investing sub-lines were available at all, fall back to aggregate ICF/CFF bars
        used_sublines = len(wf_steps) > 2
        if not used_sublines:
            if np.isfinite(last["icf"]):
                wf_steps.append(("+/− CFI", last["icf"], False)); running += last["icf"]
            if np.isfinite(last["cff"]):
                wf_steps.append(("+/− CFF", last["cff"], False)); running += last["cff"]
        wf_steps.append(("Ending Cash", running, True))

        reported_ending = last["cash_end"] if np.isfinite(last["cash_end"]) else None
        waterfall_gap = _fin(running - reported_ending) if reported_ending is not None else None
        waterfall_reconciled = (reported_ending is not None and waterfall_gap is not None and abs(waterfall_gap) < max(1.0, abs(reported_ending) * 0.02))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = [s[0] for s in wf_steps]
        cum = 0.0
        bar_bottoms = []
        bar_heights = []
        colors_wf = []
        for name, val, is_total in wf_steps:
            if is_total:
                bar_bottoms.append(0); bar_heights.append(val); colors_wf.append(PALETTE[0])
                cum = val
            else:
                lo = min(cum, cum + val)
                bar_bottoms.append(lo); bar_heights.append(abs(val)); colors_wf.append(PALETTE[1] if val >= 0 else PALETTE[3])
                cum += val
        ax.bar(names, bar_heights, bottom=bar_bottoms, color=colors_wf)
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.set_title(f"Cash Flow Waterfall — {last['label']}"); ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        chart_waterfall = _png(fig)

        results["waterfall"] = {
            "period": last["label"],
            "steps": [{"name": nm, "value": _fin(v), "is_total": it} for nm, v, it in wf_steps],
            "computed_ending_cash": _fin(running), "reported_ending_cash": _fin(reported_ending),
            "reconciliation_gap": waterfall_gap, "reconciled": waterfall_reconciled,
            "used_sublines": used_sublines,
        }

        # ── charts (nested inside results) ──
        charts = {"trend_chart": chart_trend, "fcf_chart": chart_fcf, "cash_conversion_chart": chart_cc,
                  "waterfall_chart": chart_waterfall}
        if chart_comp: charts["composition_chart"] = chart_comp
        if chart_bridge: charts["ocf_bridge_chart"] = chart_bridge
        if chart_capex: charts["capex_chart"] = chart_capex
        if chart_quality: charts["quality_chart"] = chart_quality
        results["charts"] = charts
        results["n_periods"] = n

        print(json.dumps({"results": results, "plot": chart_waterfall}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
