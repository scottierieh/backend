#!/usr/bin/env python3
"""Income Statement Analysis — "How is the company earning money, where is
it spending it, and how much is left?"

Input (from income-statement-analysis-page.tsx):
    data              : list[dict]   one row per fiscal period
    period_col        : str          period label column (e.g. "period")
    revenue_col       : str
    cogs_col          : str
    sga_col           : str          SG&A (may already exclude R&D/marketing)
    rnd_col           : str | null   R&D expense (optional)
    marketing_col     : str | null   Marketing expense (optional)
    da_col            : str          Depreciation & amortization
    interest_col      : str          Interest expense
    tax_col           : str          Income tax expense
    net_income_col    : str | null   If omitted, computed as pretax - tax
    ocf_col           : str | null   Operating cash flow (optional — enables ⑫)
    segment_cols      : list[{name, revenue_col, operating_income_col}] (optional — enables ③/⑪)

Output: { "results": {...(charts nested inside)...}, "plot": <optional> }

Sections (12 total):
  1 overview (KPI + YoY), 2 trend (multi-line + table), 3 revenue-by-segment (conditional),
  4 cost structure, 5 profit bridge (waterfall), 6 margin structure, 7 EBITDA/EBIT cascade,
  8 opex analysis (+ faster-than-revenue check), 9 common-size statement, 10 YoY change,
  11 segment profitability (conditional), 12 earnings-quality snapshot (conditional)
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


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)

        period_col = p.get("period_col") or df.columns[0]
        revenue_col = p["revenue_col"]
        cogs_col = p["cogs_col"]
        sga_col = p["sga_col"]
        rnd_col = p.get("rnd_col") or None
        marketing_col = p.get("marketing_col") or None
        da_col = p["da_col"]
        interest_col = p["interest_col"]
        tax_col = p["tax_col"]
        net_income_col = p.get("net_income_col") or None
        ocf_col = p.get("ocf_col") or None
        segment_cols = p.get("segment_cols") or []

        for c in [revenue_col, cogs_col, sga_col, da_col, interest_col, tax_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in [rnd_col, marketing_col, net_income_col, ocf_col]:
            if c:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.sort_values(by=period_col).reset_index(drop=True)
        df[period_col] = df[period_col].astype(str)
        n = len(df)
        if n < 2:
            raise ValueError("Need at least 2 periods.")

        rnd = df[rnd_col] if rnd_col else pd.Series(0.0, index=df.index)
        mkt = df[marketing_col] if marketing_col else pd.Series(0.0, index=df.index)

        gross_profit = df[revenue_col] - df[cogs_col]
        opex = df[sga_col] + rnd + mkt + df[da_col]
        operating_income = gross_profit - opex
        pretax = operating_income - df[interest_col]
        if net_income_col:
            net_income = df[net_income_col]
        else:
            net_income = pretax - df[tax_col]
        ebitda = operating_income + df[da_col]
        ebit = operating_income  # EBIT == operating income here (no non-operating items besides interest/tax)

        periods = df[period_col].tolist()

        # ---------------------------------------------------------------
        # ① Overview — latest period KPIs + YoY
        # ---------------------------------------------------------------
        def kpi(series):
            cur = series.iloc[-1]
            prev = series.iloc[-2] if n >= 2 else None
            return {"value": _fin(cur, 2), "yoy_pct": _pct(cur, prev)}

        overview = {
            "period": periods[-1],
            "revenue": kpi(df[revenue_col]),
            "gross_profit": kpi(gross_profit),
            "operating_income": kpi(operating_income),
            "net_income": kpi(net_income),
            "ebitda": kpi(ebitda),
        }

        # ---------------------------------------------------------------
        # ② Trend table + chart
        # ---------------------------------------------------------------
        trend_table = [
            {
                "period": periods[i],
                "revenue": _fin(df[revenue_col].iloc[i], 2),
                "gross_profit": _fin(gross_profit.iloc[i], 2),
                "operating_income": _fin(operating_income.iloc[i], 2),
                "net_income": _fin(net_income.iloc[i], 2),
                "ebitda": _fin(ebitda.iloc[i], 2),
            }
            for i in range(n)
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        for label, series, color in [
            ("Revenue", df[revenue_col], PALETTE[0]),
            ("Gross Profit", gross_profit, PALETTE[1]),
            ("Operating Income", operating_income, PALETTE[2]),
            ("Net Income", net_income, PALETTE[3]),
        ]:
            ax.plot(periods, series, marker="o", label=label, color=color)
        ax.set_title("Income Statement Trend")
        ax.set_ylabel("Amount")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        chart_trend = _png(fig)

        # ---------------------------------------------------------------
        # ③ Revenue by segment (conditional)
        # ---------------------------------------------------------------
        has_segments = len(segment_cols) >= 2 and all(
            s.get("revenue_col") in df.columns for s in segment_cols
        )
        chart_revenue_segment = None
        revenue_segment_table = None
        if has_segments:
            for s in segment_cols:
                df[s["revenue_col"]] = pd.to_numeric(df[s["revenue_col"]], errors="coerce")
                if s.get("operating_income_col"):
                    df[s["operating_income_col"]] = pd.to_numeric(df[s["operating_income_col"]], errors="coerce")
            revenue_segment_table = []
            for i in range(n):
                row = {"period": periods[i]}
                for s in segment_cols:
                    cur = df[s["revenue_col"]].iloc[i]
                    prev = df[s["revenue_col"]].iloc[i - 1] if i > 0 else None
                    row[f"{s['name']}_revenue"] = _fin(cur, 2)
                    row[f"{s['name']}_change_pct"] = _pct(cur, prev) if i > 0 else None
                revenue_segment_table.append(row)

            fig, ax = plt.subplots(figsize=(9, 5))
            bottom = np.zeros(n)
            for idx, s in enumerate(segment_cols):
                vals = df[s["revenue_col"]].to_numpy(dtype=float)
                ax.bar(periods, vals, bottom=bottom, label=s["name"], color=PALETTE[idx % len(PALETTE)])
                bottom += vals
            ax.set_title("Revenue by Segment")
            ax.legend(fontsize=8)
            plt.xticks(rotation=30, ha="right")
            chart_revenue_segment = _png(fig)

        # ---------------------------------------------------------------
        # ④ Cost structure (% of revenue) + trend
        # ---------------------------------------------------------------
        cost_items = [("cogs", df[cogs_col]), ("sga", df[sga_col]), ("rnd", rnd), ("marketing", mkt), ("da", df[da_col])]
        cost_structure_table = []
        for i in range(n):
            row = {"period": periods[i]}
            rev_i = df[revenue_col].iloc[i]
            for name, series in cost_items:
                row[f"{name}_pct_revenue"] = _fin((series.iloc[i] / rev_i * 100.0) if rev_i else None, 3)
            cost_structure_table.append(row)

        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(n)
        for idx, (name, series) in enumerate(cost_items):
            pct_vals = np.array([r[f"{name}_pct_revenue"] or 0 for r in cost_structure_table])
            ax.bar(periods, pct_vals, bottom=bottom, label=name.upper(), color=PALETTE[idx % len(PALETTE)])
            bottom += pct_vals
        ax.set_title("Cost Structure (% of Revenue)")
        ax.set_ylabel("% of Revenue")
        ax.legend(fontsize=8)
        plt.xticks(rotation=30, ha="right")
        chart_cost_structure = _png(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for idx, (name, series) in enumerate(cost_items):
            pct_vals = [r[f"{name}_pct_revenue"] for r in cost_structure_table]
            ax.plot(periods, pct_vals, marker="o", label=name.upper(), color=PALETTE[idx % len(PALETTE)])
        ax.set_title("Cost Structure Over Time")
        ax.set_ylabel("% of Revenue")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        chart_cost_structure_trend = _png(fig)

        # ---------------------------------------------------------------
        # ⑤ Profit bridge (waterfall) — latest period, real $ reconciliation
        # ---------------------------------------------------------------
        li = n - 1
        rev_l = float(df[revenue_col].iloc[li])
        cogs_l = float(df[cogs_col].iloc[li])
        gp_l = float(gross_profit.iloc[li])
        opex_l = float(opex.iloc[li])
        oi_l = float(operating_income.iloc[li])
        interest_l = float(df[interest_col].iloc[li])
        tax_l = float(df[tax_col].iloc[li])
        ni_l = float(net_income.iloc[li])
        reconciled_net_income = rev_l - cogs_l - opex_l - interest_l - tax_l
        reconciliation_diff = round(reconciled_net_income - ni_l, 2)

        bridge_steps = [
            {"label": "Revenue", "value": _fin(rev_l, 2), "type": "total"},
            {"label": "− COGS", "value": _fin(-cogs_l, 2), "type": "decrease"},
            {"label": "Gross Profit", "value": _fin(gp_l, 2), "type": "subtotal"},
            {"label": "− OpEx (SG&A+R&D+Mkt+D&A)", "value": _fin(-opex_l, 2), "type": "decrease"},
            {"label": "Operating Income", "value": _fin(oi_l, 2), "type": "subtotal"},
            {"label": "− Interest & Tax", "value": _fin(-(interest_l + tax_l), 2), "type": "decrease"},
            {"label": "Net Income", "value": _fin(ni_l, 2), "type": "total"},
        ]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        labels = [s["label"] for s in bridge_steps]
        vals = [s["value"] for s in bridge_steps]
        running = 0.0
        for i2, (lab, v) in enumerate(zip(labels, vals)):
            if bridge_steps[i2]["type"] in ("total", "subtotal"):
                base = 0
                height = v
                running = v
            else:
                base = running + min(v, 0)
                height = abs(v)
                running += v
            color = PALETTE[1] if bridge_steps[i2]["type"] != "decrease" else PALETTE[3]
            ax.bar(i2, height, bottom=base, color=color, width=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Profit Bridge — {periods[-1]}")
        ax.grid(alpha=0.3, axis="y")
        chart_profit_bridge = _png(fig)

        # ---------------------------------------------------------------
        # ⑥ Margin structure
        # ---------------------------------------------------------------
        margin_table = []
        for i in range(n):
            rev_i = df[revenue_col].iloc[i]
            margin_table.append({
                "period": periods[i],
                "gross_margin_pct": _fin((gross_profit.iloc[i] / rev_i * 100.0) if rev_i else None, 3),
                "operating_margin_pct": _fin((operating_income.iloc[i] / rev_i * 100.0) if rev_i else None, 3),
                "net_margin_pct": _fin((net_income.iloc[i] / rev_i * 100.0) if rev_i else None, 3),
            })

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(periods, [r["gross_margin_pct"] for r in margin_table], marker="o", label="Gross Margin %", color=PALETTE[1])
        ax.plot(periods, [r["operating_margin_pct"] for r in margin_table], marker="o", label="Operating Margin %", color=PALETTE[0])
        ax.plot(periods, [r["net_margin_pct"] for r in margin_table], marker="o", label="Net Margin %", color=PALETTE[3])
        ax.set_title("Margin Trend")
        ax.set_ylabel("%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        chart_margin_trend = _png(fig)

        # ---------------------------------------------------------------
        # ⑦ EBITDA & EBIT analysis
        # ---------------------------------------------------------------
        ebitda_ebit_table = []
        for i in range(n):
            ebitda_prev = ebitda.iloc[i - 1] if i > 0 else None
            ebit_prev = ebit.iloc[i - 1] if i > 0 else None
            ni_prev = net_income.iloc[i - 1] if i > 0 else None
            ebitda_ebit_table.append({
                "period": periods[i],
                "ebitda": _fin(ebitda.iloc[i], 2),
                "ebitda_yoy_pct": _pct(ebitda.iloc[i], ebitda_prev) if i > 0 else None,
                "ebit": _fin(ebit.iloc[i], 2),
                "ebit_yoy_pct": _pct(ebit.iloc[i], ebit_prev) if i > 0 else None,
                "net_income": _fin(net_income.iloc[i], 2),
                "net_income_yoy_pct": _pct(net_income.iloc[i], ni_prev) if i > 0 else None,
            })

        cascade_steps = [
            {"label": "EBITDA", "value": _fin(ebitda.iloc[li], 2)},
            {"label": "− D&A", "value": _fin(-df[da_col].iloc[li], 2)},
            {"label": "= EBIT", "value": _fin(ebit.iloc[li], 2)},
            {"label": "− Interest & Tax", "value": _fin(-(interest_l + tax_l), 2)},
            {"label": "= Net Income", "value": _fin(ni_l, 2)},
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        cascade_vals = [ebitda.iloc[li], df[da_col].iloc[li], ebit.iloc[li], interest_l + tax_l, ni_l]
        cascade_labels = ["EBITDA", "D&A", "EBIT", "Interest+Tax", "Net Income"]
        ax.bar(cascade_labels, cascade_vals, color=[PALETTE[1], PALETTE[3], PALETTE[0], PALETTE[3], PALETTE[2]])
        ax.set_title(f"EBITDA → EBIT → Net Income — {periods[-1]}")
        plt.xticks(rotation=20, ha="right")
        chart_ebitda_cascade = _png(fig)

        # ---------------------------------------------------------------
        # ⑧ Operating expense analysis + faster-than-revenue check
        # ---------------------------------------------------------------
        rev_cagr = None
        if n >= 2 and df[revenue_col].iloc[0] > 0:
            rev_cagr = (df[revenue_col].iloc[-1] / df[revenue_col].iloc[0]) ** (1 / (n - 1)) - 1

        opex_lines = [("SG&A", df[sga_col])]
        if rnd_col:
            opex_lines.append(("R&D", rnd))
        if marketing_col:
            opex_lines.append(("Marketing", mkt))

        opex_table = []
        for i in range(n):
            row = {"period": periods[i]}
            for name, series in opex_lines:
                prev = series.iloc[i - 1] if i > 0 else None
                row[f"{name.lower().replace('&','').replace(' ', '_')}"] = _fin(series.iloc[i], 2)
                row[f"{name.lower().replace('&','').replace(' ', '_')}_yoy_pct"] = _pct(series.iloc[i], prev) if i > 0 else None
            opex_table.append(row)

        faster_than_revenue = []
        for name, series in opex_lines:
            if series.iloc[0] > 0:
                line_cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / (n - 1)) - 1
                faster_than_revenue.append({
                    "line": name,
                    "cagr_pct": _fin(line_cagr * 100, 3),
                    "revenue_cagr_pct": _fin(rev_cagr * 100, 3) if rev_cagr is not None else None,
                    "growing_faster_than_revenue": bool(line_cagr > (rev_cagr or 0)),
                })

        fig, ax = plt.subplots(figsize=(9, 5))
        for idx, (name, series) in enumerate(opex_lines):
            ax.plot(periods, series, marker="o", label=name, color=PALETTE[idx % len(PALETTE)])
        ax.set_title("Operating Expense Trend")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        chart_opex_trend = _png(fig)

        # ---------------------------------------------------------------
        # ⑨ Common-size income statement (every line as % of revenue)
        # ---------------------------------------------------------------
        common_size_table = []
        for i in range(n):
            rev_i = df[revenue_col].iloc[i]
            def pctf(v):
                return _fin((v / rev_i * 100.0) if rev_i else None, 3)
            common_size_table.append({
                "period": periods[i],
                "revenue_pct": 100.0,
                "cogs_pct": pctf(df[cogs_col].iloc[i]),
                "gross_profit_pct": pctf(gross_profit.iloc[i]),
                "sga_pct": pctf(df[sga_col].iloc[i]),
                "rnd_pct": pctf(rnd.iloc[i]) if rnd_col else None,
                "marketing_pct": pctf(mkt.iloc[i]) if marketing_col else None,
                "da_pct": pctf(df[da_col].iloc[i]),
                "operating_income_pct": pctf(operating_income.iloc[i]),
                "interest_pct": pctf(df[interest_col].iloc[i]),
                "tax_pct": pctf(df[tax_col].iloc[i]),
                "net_income_pct": pctf(net_income.iloc[i]),
            })

        # ---------------------------------------------------------------
        # ⑩ YoY change table + chart
        # ---------------------------------------------------------------
        yoy_items = [("Revenue", df[revenue_col]), ("Gross Profit", gross_profit), ("Operating Income", operating_income),
                     ("Net Income", net_income), ("EBITDA", ebitda), ("COGS", df[cogs_col]), ("SG&A", df[sga_col])]
        yoy_table = []
        for i in range(1, n):
            row = {"period": periods[i]}
            for name, series in yoy_items:
                cur, prev = series.iloc[i], series.iloc[i - 1]
                row[f"{name}_change_abs"] = _fin(cur - prev, 2)
                row[f"{name}_change_pct"] = _pct(cur, prev)
            yoy_table.append(row)

        li_yoy = yoy_table[-1] if yoy_table else {}
        fig, ax = plt.subplots(figsize=(9, 5))
        names = [nm for nm, _ in yoy_items]
        vals = [li_yoy.get(f"{nm}_change_pct") or 0 for nm in names]
        colors = [PALETTE[1] if v >= 0 else PALETTE[3] for v in vals]
        ax.bar(names, vals, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"YoY % Change — {periods[-1]} vs {periods[-2]}")
        ax.set_ylabel("% change")
        plt.xticks(rotation=30, ha="right")
        chart_yoy = _png(fig)

        # ---------------------------------------------------------------
        # ⑪ Segment profitability (conditional)
        # ---------------------------------------------------------------
        segment_profitability = None
        chart_segment_scatter = None
        if has_segments:
            segment_profitability = []
            for s in segment_cols:
                rev_series = df[s["revenue_col"]]
                oi_col = s.get("operating_income_col")
                oi_series = df[oi_col] if oi_col else None
                rev_latest = float(rev_series.iloc[-1])
                oi_latest = float(oi_series.iloc[-1]) if oi_series is not None else None
                margin = (oi_latest / rev_latest * 100.0) if (oi_latest is not None and rev_latest) else None
                segment_profitability.append({
                    "segment": s["name"],
                    "revenue": _fin(rev_latest, 2),
                    "operating_income": _fin(oi_latest, 2) if oi_latest is not None else None,
                    "operating_margin_pct": _fin(margin, 3) if margin is not None else None,
                })
            fig, ax = plt.subplots(figsize=(7.5, 6))
            for idx, s in enumerate(segment_profitability):
                if s["operating_margin_pct"] is None:
                    continue
                size = max(200, (s["revenue"] or 0) / 2_000_000)
                ax.scatter(s["revenue"], s["operating_margin_pct"], s=size, color=PALETTE[idx % len(PALETTE)], label=s["segment"], alpha=0.75)
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Operating Margin %")
            ax.set_title(f"Segment Profitability — {periods[-1]} (bubble = revenue)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            chart_segment_scatter = _png(fig)

        # ---------------------------------------------------------------
        # ⑫ Earnings quality snapshot (conditional)
        # ---------------------------------------------------------------
        earnings_quality = None
        chart_earnings_quality = None
        has_ocf = bool(ocf_col) and ocf_col in df.columns
        if has_ocf:
            df[ocf_col] = pd.to_numeric(df[ocf_col], errors="coerce")
            eq_table = []
            for i in range(n):
                ni_growth = _pct(net_income.iloc[i], net_income.iloc[i - 1]) if i > 0 else None
                ocf_growth = _pct(df[ocf_col].iloc[i], df[ocf_col].iloc[i - 1]) if i > 0 else None
                divergence = None
                if ni_growth is not None and ocf_growth is not None:
                    divergence = ni_growth - ocf_growth
                eq_table.append({
                    "period": periods[i],
                    "net_income": _fin(net_income.iloc[i], 2),
                    "operating_cash_flow": _fin(df[ocf_col].iloc[i], 2),
                    "net_income_yoy_pct": ni_growth,
                    "ocf_yoy_pct": ocf_growth,
                    "divergence_pct_pts": _fin(divergence, 3) if divergence is not None else None,
                    "income_outpaces_cash_flag": bool(divergence is not None and divergence > 10),
                })
            earnings_quality = {"table": eq_table, "note_en": "Snapshot only — a simple net-income-vs-operating-cash-flow comparison, not a full cash-flow analysis.",
                                 "note_ko": "스냅샷일 뿐입니다 — 단순 순이익 대 영업현금흐름 비교이며, 전체 현금흐름 분석이 아닙니다."}
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(periods, net_income, marker="o", label="Net Income", color=PALETTE[0])
            ax.plot(periods, df[ocf_col], marker="o", label="Operating Cash Flow", color=PALETTE[1])
            ax.set_title("Earnings Quality Snapshot: Net Income vs Operating Cash Flow")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            chart_earnings_quality = _png(fig)

        charts = {
            "trend_chart": chart_trend,
            "cost_structure_chart": chart_cost_structure,
            "cost_structure_trend_chart": chart_cost_structure_trend,
            "profit_bridge_chart": chart_profit_bridge,
            "margin_trend_chart": chart_margin_trend,
            "ebitda_cascade_chart": chart_ebitda_cascade,
            "opex_trend_chart": chart_opex_trend,
            "yoy_chart": chart_yoy,
        }
        if chart_revenue_segment:
            charts["revenue_segment_chart"] = chart_revenue_segment
        if chart_segment_scatter:
            charts["segment_scatter_chart"] = chart_segment_scatter
        if chart_earnings_quality:
            charts["earnings_quality_chart"] = chart_earnings_quality

        results = {
            "n_periods": n,
            "periods": periods,
            "latest_period": periods[-1],
            "overview": overview,
            "trend_table": trend_table,
            "has_segments": has_segments,
            "revenue_segment_table": revenue_segment_table,
            "cost_structure_table": cost_structure_table,
            "profit_bridge": {
                "period": periods[-1],
                "steps": bridge_steps,
                "reconciled_net_income": _fin(reconciled_net_income, 2),
                "reported_net_income": _fin(ni_l, 2),
                "reconciliation_diff": reconciliation_diff,
            },
            "margin_table": margin_table,
            "ebitda_ebit_table": ebitda_ebit_table,
            "ebitda_cascade": cascade_steps,
            "opex_table": opex_table,
            "opex_faster_than_revenue": faster_than_revenue,
            "revenue_cagr_pct": _fin(rev_cagr * 100, 3) if rev_cagr is not None else None,
            "common_size_table": common_size_table,
            "yoy_table": yoy_table,
            "has_ocf": has_ocf,
            "segment_profitability": segment_profitability,
            "earnings_quality": earnings_quality,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": chart_trend}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
