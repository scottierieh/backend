#!/usr/bin/env python3
"""Balance Sheet Analysis — structure, leverage, working capital, asset quality.

Input (from balance-sheet-analysis-page.tsx):
    data                  : list[dict]  one row per fiscal period
    period_col            : str
    cash_col, ar_col, inventory_col, ppe_col, other_ca_col, intangibles_col : str (optional composition)
    total_assets_col      : str  required
    ap_col, other_cl_col, short_debt_col, long_debt_col, other_ncl_col     : str (optional)
    total_liabilities_col : str  required (or omitted -> assets - equity)
    common_equity_col, retained_earnings_col, apic_col, treasury_col      : str (optional)
    total_equity_col      : str  required
    current_assets_col, current_liabilities_col                          : str (optional, else summed)
    revenue_col           : str | null  (optional, enables asset-quality growth compare)
    ebitda_col            : str | null  (optional, enables Net Debt/EBITDA)

Output: { "results": {...(charts nested inside)...}, "plot": <optional> }

Sections (11 total):
  1 overview, 2 trend, 3 asset structure, 4 liability structure, 5 equity structure,
  6 common-size balance sheet, 7 debt & leverage, 8 working capital snapshot,
  9 balance sheet composition (A = L + E), 10 change analysis, 11 asset quality (conditional)
"""
import sys, json, io, base64
import numpy as np
import pandas as pd

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

        period_col = p.get("period_col")
        c_cash, c_ar, c_inv, c_ppe, c_oca, c_intan = (
            p.get("cash_col"), p.get("ar_col"), p.get("inventory_col"),
            p.get("ppe_col"), p.get("other_ca_col"), p.get("intangibles_col"))
        c_ta = p.get("total_assets_col")
        c_ap, c_ocl, c_std, c_ltd, c_oncl = (
            p.get("ap_col"), p.get("other_cl_col"), p.get("short_debt_col"),
            p.get("long_debt_col"), p.get("other_ncl_col"))
        c_tl = p.get("total_liabilities_col")
        c_ce, c_re, c_apic, c_treas = (
            p.get("common_equity_col"), p.get("retained_earnings_col"),
            p.get("apic_col"), p.get("treasury_col"))
        c_te = p.get("total_equity_col")
        c_ca, c_cl = p.get("current_assets_col"), p.get("current_liabilities_col")
        c_rev, c_ebitda = p.get("revenue_col"), p.get("ebitda_col")

        if not c_ta or not c_te:
            raise ValueError("total_assets_col and total_equity_col are required")

        periods = []
        for i, row in enumerate(rows):
            label = str(row.get(period_col)) if period_col else f"Period {i+1}"
            ta = col(row, c_ta)
            te = col(row, c_te)
            tl_direct = col(row, c_tl)
            tl = tl_direct if np.isfinite(tl_direct) else (ta - te if np.isfinite(ta) and np.isfinite(te) else np.nan)
            if not (np.isfinite(ta) and np.isfinite(te)):
                continue

            cash, ar, inv, ppe, oca, intan = (col(row, c_cash), col(row, c_ar), col(row, c_inv),
                                                col(row, c_ppe), col(row, c_oca), col(row, c_intan))
            ap, ocl, std, ltd, oncl = (col(row, c_ap), col(row, c_ocl), col(row, c_std),
                                        col(row, c_ltd), col(row, c_oncl))
            ce, re, apic, treas = col(row, c_ce), col(row, c_re), col(row, c_apic), col(row, c_treas)
            ca_direct = col(row, c_ca)
            cl_direct = col(row, c_cl)
            ca = ca_direct if np.isfinite(ca_direct) else np.nansum([cash, ar, inv, oca]) if any(np.isfinite(x) for x in [cash, ar, inv, oca]) else np.nan
            cl = cl_direct if np.isfinite(cl_direct) else np.nansum([ap, ocl, std]) if any(np.isfinite(x) for x in [ap, ocl, std]) else np.nan
            total_debt = np.nansum([x for x in [std, ltd] if np.isfinite(x)]) if (np.isfinite(std) or np.isfinite(ltd)) else np.nan
            net_debt = total_debt - cash if np.isfinite(total_debt) and np.isfinite(cash) else np.nan
            rev = col(row, c_rev)
            ebitda = col(row, c_ebitda)

            periods.append(dict(label=label, total_assets=ta, total_equity=te, total_liabilities=tl,
                                 cash=cash, ar=ar, inventory=inv, ppe=ppe, other_ca=oca, intangibles=intan,
                                 ap=ap, other_cl=ocl, short_debt=std, long_debt=ltd, other_ncl=oncl,
                                 common_equity=ce, retained_earnings=re, apic=apic, treasury=treas,
                                 current_assets=ca, current_liabilities=cl,
                                 total_debt=total_debt, net_debt=net_debt, revenue=rev, ebitda=ebitda))

        if not periods:
            raise ValueError("No valid periods (need total assets & total equity)")

        n = len(periods)
        last, first = periods[-1], periods[0]
        prev = periods[-2] if n > 1 else None

        results = {}

        # ── 1. Overview ──
        overview = {
            "total_assets": _fin(last["total_assets"]),
            "total_liabilities": _fin(last["total_liabilities"]),
            "total_equity": _fin(last["total_equity"]),
            "cash": _fin(last["cash"]),
            "total_debt": _fin(last["total_debt"]),
            "net_debt": _fin(last["net_debt"]),
        }
        if prev:
            overview["yoy"] = {
                "total_assets_pct": _pct(last["total_assets"], prev["total_assets"]),
                "total_liabilities_pct": _pct(last["total_liabilities"], prev["total_liabilities"]),
                "total_equity_pct": _pct(last["total_equity"], prev["total_equity"]),
            }
        results["overview"] = overview

        # ── 2. Trend ──
        trend_table = [{"period": pd_["label"], "total_assets": _fin(pd_["total_assets"]),
                         "total_liabilities": _fin(pd_["total_liabilities"]), "total_equity": _fin(pd_["total_equity"])}
                        for pd_ in periods]
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [pd_["label"] for pd_ in periods]
        ax.plot(labels, [pd_["total_assets"] for pd_ in periods], marker="o", color=PALETTE[0], label="Total Assets")
        ax.plot(labels, [pd_["total_liabilities"] for pd_ in periods], marker="o", color=PALETTE[3], label="Total Liabilities")
        ax.plot(labels, [pd_["total_equity"] for pd_ in periods], marker="o", color=PALETTE[1], label="Total Equity")
        ax.set_title("Balance Sheet Trend"); ax.legend(); ax.grid(alpha=0.3)
        chart_trend = _png(fig)
        results["trend"] = {"table": trend_table}

        # ── 3. Asset structure (latest) ──
        asset_parts = [("cash", last["cash"]), ("accounts_receivable", last["ar"]), ("inventory", last["inventory"]),
                        ("ppe", last["ppe"]), ("intangibles", last["intangibles"]), ("other_current_assets", last["other_ca"])]
        asset_parts = [(k, v) for k, v in asset_parts if np.isfinite(v)]
        asset_table = [{"item": k, "value": _fin(v), "pct_of_total_assets": _fin(v / last["total_assets"] * 100 if last["total_assets"] else None)}
                        for k, v in asset_parts]
        results["asset_structure"] = {"table": asset_table}
        chart_asset = None
        if asset_parts:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.pie([v for _, v in asset_parts], labels=[k for k, _ in asset_parts], autopct="%1.1f%%",
                   colors=PALETTE[:len(asset_parts)])
            ax.set_title(f"Asset Composition — {last['label']}")
            chart_asset = _png(fig)

        # ── 4. Liability structure (latest) ──
        liab_parts = [("accounts_payable", last["ap"]), ("short_term_debt", last["short_debt"]),
                       ("long_term_debt", last["long_debt"]), ("other_current_liabilities", last["other_cl"]),
                       ("other_noncurrent_liabilities", last["other_ncl"])]
        liab_parts = [(k, v) for k, v in liab_parts if np.isfinite(v)]
        liab_table = [{"item": k, "value": _fin(v), "pct_of_total_liabilities": _fin(v / last["total_liabilities"] * 100 if last["total_liabilities"] else None)}
                       for k, v in liab_parts]
        results["liability_structure"] = {"table": liab_table}
        chart_liab = None
        if liab_parts:
            st_total = np.nansum([last["ap"], last["short_debt"], last["other_cl"]])
            lt_total = np.nansum([last["long_debt"], last["other_ncl"]])
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.bar(["Short-term", "Long-term"], [st_total, lt_total], color=[PALETTE[2], PALETTE[3]])
            ax.set_title("Short-term vs Long-term Liabilities"); ax.grid(alpha=0.3, axis="y")
            chart_liab = _png(fig)

        # ── 5. Equity structure (latest) ──
        eq_parts = [("common_equity", last["common_equity"]), ("retained_earnings", last["retained_earnings"]),
                     ("additional_paid_in_capital", last["apic"]), ("treasury_stock", last["treasury"])]
        eq_parts = [(k, v) for k, v in eq_parts if np.isfinite(v)]
        eq_table = [{"item": k, "value": _fin(v), "pct_of_total_equity": _fin(v / last["total_equity"] * 100 if last["total_equity"] else None)}
                    for k, v in eq_parts]
        results["equity_structure"] = {"table": eq_table}
        chart_equity = None
        if eq_parts:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(eq_parts))]
            ax.bar([k for k, _ in eq_parts], [v for _, v in eq_parts], color=colors)
            ax.set_title(f"Equity Composition — {last['label']}"); ax.grid(alpha=0.3, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            chart_equity = _png(fig)

        # ── 6. Common-size balance sheet (every period, % of total assets) ──
        common_size_rows = []
        identity_check = []
        for pd_ in periods:
            ta = pd_["total_assets"]
            if not ta:
                continue
            def pct_(v):
                return _fin(v / ta * 100) if np.isfinite(v) else None
            row_out = {
                "period": pd_["label"],
                "cash_pct": pct_(pd_["cash"]), "ar_pct": pct_(pd_["ar"]), "inventory_pct": pct_(pd_["inventory"]),
                "ppe_pct": pct_(pd_["ppe"]), "intangibles_pct": pct_(pd_["intangibles"]),
                "total_liabilities_pct": pct_(pd_["total_liabilities"]), "total_equity_pct": pct_(pd_["total_equity"]),
            }
            common_size_rows.append(row_out)
            recon = (row_out["total_liabilities_pct"] or 0) + (row_out["total_equity_pct"] or 0)
            identity_check.append({"period": pd_["label"], "liab_plus_equity_pct": _fin(recon), "deviation_from_100": _fin(recon - 100)})
        results["common_size"] = {"table": common_size_rows, "identity_check": identity_check}

        # ── 7. Debt & leverage ──
        lev_table = []
        for pd_ in periods:
            de = pd_["total_debt"] / pd_["total_equity"] if np.isfinite(pd_["total_debt"]) and pd_["total_equity"] else np.nan
            da = pd_["total_debt"] / pd_["total_assets"] if np.isfinite(pd_["total_debt"]) and pd_["total_assets"] else np.nan
            nde = pd_["net_debt"] / pd_["ebitda"] if np.isfinite(pd_["net_debt"]) and np.isfinite(pd_["ebitda"]) and pd_["ebitda"] else np.nan
            lev_table.append({"period": pd_["label"], "total_debt": _fin(pd_["total_debt"]), "net_debt": _fin(pd_["net_debt"]),
                               "debt_to_equity": _fin(de), "debt_to_assets_pct": _fin(da * 100 if np.isfinite(da) else None),
                               "net_debt_to_ebitda": _fin(nde) if np.isfinite(nde) else None})
        has_ebitda = any(np.isfinite(pd_["ebitda"]) for pd_ in periods)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(labels, [r["debt_to_equity"] for r in lev_table], marker="o", color=PALETTE[3], label="Debt/Equity")
        ax.set_title("Leverage Trend"); ax.legend(); ax.grid(alpha=0.3)
        chart_leverage = _png(fig)
        results["leverage"] = {"table": lev_table, "net_debt_to_ebitda_available": has_ebitda}

        # ── 8. Working capital snapshot ──
        wc_table = []
        for pd_ in periods:
            wc = pd_["current_assets"] - pd_["current_liabilities"] if np.isfinite(pd_["current_assets"]) and np.isfinite(pd_["current_liabilities"]) else np.nan
            wc_table.append({"period": pd_["label"], "current_assets": _fin(pd_["current_assets"]),
                              "current_liabilities": _fin(pd_["current_liabilities"]), "net_working_capital": _fin(wc)})
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(labels, [r["net_working_capital"] for r in wc_table], marker="o", color=PALETTE[1])
        ax.set_title("Net Working Capital Trend"); ax.grid(alpha=0.3)
        chart_wc = _png(fig)
        results["working_capital"] = {"table": wc_table}

        # ── 9. Balance sheet composition (A = L + E), latest period ──
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.bar(["Assets"], [last["total_assets"]], color=PALETTE[0], label="Total Assets")
        ax.bar(["Liabilities + Equity"], [last["total_liabilities"]], color=PALETTE[3], label="Liabilities")
        ax.bar(["Liabilities + Equity"], [last["total_equity"]], bottom=[last["total_liabilities"]], color=PALETTE[1], label="Equity")
        ax.set_title(f"Assets = Liabilities + Equity — {last['label']}"); ax.legend(); ax.grid(alpha=0.3, axis="y")
        chart_composition = _png(fig)
        results["composition"] = {
            "period": last["label"], "total_assets": _fin(last["total_assets"]),
            "total_liabilities": _fin(last["total_liabilities"]), "total_equity": _fin(last["total_equity"]),
        }

        # ── 10. Change analysis (period-over-period) ──
        change_table = []
        change_items = ["cash", "inventory", "ar", "total_debt", "total_equity", "total_assets", "total_liabilities"]
        for i in range(1, n):
            cur, pr = periods[i], periods[i - 1]
            row_out = {"period": cur["label"]}
            for item in change_items:
                cv, pv = cur.get(item), pr.get(item)
                row_out[f"{item}_change"] = _fin(cv - pv) if np.isfinite(cv) and np.isfinite(pv) else None
                row_out[f"{item}_change_pct"] = _pct(cv, pv)
            change_table.append(row_out)
        chart_change = None
        if change_table:
            last_chg = change_table[-1]
            movers = [(k.replace("_change", ""), last_chg.get(k)) for k in last_chg if k.endswith("_change") and last_chg.get(k) is not None]
            movers.sort(key=lambda x: abs(x[1]), reverse=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            names = [m[0] for m in movers]; vals = [m[1] for m in movers]
            colors = [PALETTE[1] if v >= 0 else PALETTE[3] for v in vals]
            ax.bar(names, vals, color=colors)
            ax.set_title(f"Biggest Movers — {last_chg['period']} vs prior"); ax.grid(alpha=0.3, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            chart_change = _png(fig)
        results["change_analysis"] = {"table": change_table}

        # ── 11. Asset quality (conditional on revenue) ──
        asset_quality = {}
        has_revenue = prev is not None and np.isfinite(last["revenue"]) and np.isfinite(prev["revenue"]) and prev["revenue"] != 0
        if has_revenue:
            rev_growth = _pct(last["revenue"], prev["revenue"])
            ar_growth = _pct(last["ar"], prev["ar"]) if np.isfinite(last["ar"]) and np.isfinite(prev["ar"]) else None
            inv_growth = _pct(last["inventory"], prev["inventory"]) if np.isfinite(last["inventory"]) and np.isfinite(prev["inventory"]) else None
            asset_quality["revenue_growth_pct"] = rev_growth
            asset_quality["ar_growth_pct"] = ar_growth
            asset_quality["ar_growing_faster_than_revenue"] = (ar_growth is not None and rev_growth is not None and ar_growth > rev_growth)
            asset_quality["inventory_growth_pct"] = inv_growth
            asset_quality["inventory_growing_faster_than_revenue"] = (inv_growth is not None and rev_growth is not None and inv_growth > rev_growth)
        else:
            asset_quality["revenue_comparison_available"] = False
        if np.isfinite(last["intangibles"]) and last["total_assets"]:
            gw_pct = last["intangibles"] / last["total_assets"] * 100
            asset_quality["goodwill_intangibles_pct_of_assets"] = _fin(gw_pct)
            asset_quality["goodwill_flag_high"] = bool(gw_pct > 30)
        results["asset_quality"] = asset_quality

        # ── charts (nested inside results) ──
        charts = {"trend_chart": chart_trend, "leverage_chart": chart_leverage, "working_capital_chart": chart_wc,
                  "composition_chart": chart_composition}
        if chart_asset: charts["asset_structure_chart"] = chart_asset
        if chart_liab: charts["liability_structure_chart"] = chart_liab
        if chart_equity: charts["equity_structure_chart"] = chart_equity
        if chart_change: charts["change_analysis_chart"] = chart_change
        results["charts"] = charts
        results["n_periods"] = n

        print(json.dumps({"results": results, "plot": chart_trend}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
