from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional
import numpy as np
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────────────

class DCFRequest(BaseModel):
    # Revenue & Growth
    baseRevenue: float = Field(..., description="Current annual revenue (in $M or chosen unit)")
    revenueGrowthRates: list[float] = Field(..., description="Projected revenue growth rates (%) for each year, length = projectionYears")
    projectionYears: int = Field(default=5, description="Number of projection years (default 5)")

    # Margins & Operating
    ebitdaMargin: float = Field(..., description="Initial EBITDA margin (%)")
    ebitdaMarginTerminal: float = Field(..., description="Terminal-year EBITDA margin (%)")
    depreciationPctRevenue: float = Field(default=3.0, description="D&A as % of revenue")
    capexPctRevenue: float = Field(default=5.0, description="CapEx as % of revenue")
    nwcPctRevenue: float = Field(default=10.0, description="Net Working Capital as % of revenue")
    taxRate: float = Field(default=25.0, description="Corporate tax rate (%)")

    # Discount & Terminal
    wacc: float = Field(..., description="Weighted Average Cost of Capital (%)")
    terminalGrowthRate: float = Field(..., description="Perpetuity growth rate (%)")

    # Balance Sheet
    totalDebt: float = Field(default=0.0, description="Total debt ($M)")
    cashEquivalents: float = Field(default=0.0, description="Cash & equivalents ($M)")
    sharesOutstanding: float = Field(default=1.0, description="Shares outstanding (M)")

    # Optional: uploaded financial data for historical context
    data: Optional[list[dict[str, Any]]] = Field(default=None, description="Optional uploaded financial data")
    revenueColumn: Optional[str] = Field(default=None, description="Column name for revenue in uploaded data")
    ebitdaColumn: Optional[str] = Field(default=None, description="Column name for EBITDA in uploaded data")


# ─── Helper Functions ─────────────────────────────────────────────────────────

def compute_dcf(req: DCFRequest) -> dict:
    """Core DCF computation engine."""
    n = req.projectionYears
    growth_rates = req.revenueGrowthRates[:n]

    # Pad growth rates if fewer than projection years
    while len(growth_rates) < n:
        growth_rates.append(growth_rates[-1] if growth_rates else 5.0)

    wacc = req.wacc / 100.0
    tgr = req.terminalGrowthRate / 100.0

    if wacc <= tgr:
        raise ValueError(
            f"WACC ({req.wacc}%) must be greater than terminal growth rate ({req.terminalGrowthRate}%). "
            f"Otherwise the Gordon Growth Model produces infinite/negative terminal value."
        )

    # ── Year-by-year projections ──────────────────────────────────────────
    projections = []
    prev_revenue = req.baseRevenue
    prev_nwc = req.baseRevenue * (req.nwcPctRevenue / 100.0)

    for i in range(n):
        g = growth_rates[i] / 100.0
        revenue = prev_revenue * (1 + g)

        # Linearly interpolate EBITDA margin from initial to terminal
        if n > 1:
            margin_weight = i / (n - 1)
        else:
            margin_weight = 1.0
        ebitda_margin = req.ebitdaMargin + (req.ebitdaMarginTerminal - req.ebitdaMargin) * margin_weight

        ebitda = revenue * (ebitda_margin / 100.0)
        depreciation = revenue * (req.depreciationPctRevenue / 100.0)
        ebit = ebitda - depreciation
        taxes = max(0.0, ebit * (req.taxRate / 100.0))
        nopat = ebit - taxes
        capex = revenue * (req.capexPctRevenue / 100.0)
        current_nwc = revenue * (req.nwcPctRevenue / 100.0)
        change_nwc = current_nwc - prev_nwc
        fcf = nopat + depreciation - capex - change_nwc

        discount_factor = 1.0 / ((1 + wacc) ** (i + 1))
        pv_fcf = fcf * discount_factor

        projections.append({
            "year": i + 1,
            "revenue": float(revenue),
            "growthRate": float(growth_rates[i]),
            "ebitda": float(ebitda),
            "ebitdaMargin": float(ebitda_margin),
            "depreciation": float(depreciation),
            "ebit": float(ebit),
            "taxes": float(taxes),
            "nopat": float(nopat),
            "capex": float(capex),
            "changeNWC": float(change_nwc),
            "fcf": float(fcf),
            "discountFactor": float(discount_factor),
            "pvFCF": float(pv_fcf),
        })

        prev_revenue = revenue
        prev_nwc = current_nwc

    # ── Terminal Value (Gordon Growth Model) ──────────────────────────────
    last_fcf = projections[-1]["fcf"]
    terminal_value = (last_fcf * (1 + tgr)) / (wacc - tgr)
    pv_terminal_value = terminal_value / ((1 + wacc) ** n)

    # ── Enterprise & Equity Value ─────────────────────────────────────────
    sum_pv_fcf = sum(p["pvFCF"] for p in projections)
    enterprise_value = sum_pv_fcf + pv_terminal_value
    net_debt = req.totalDebt - req.cashEquivalents
    equity_value = enterprise_value - net_debt
    implied_share_price = equity_value / req.sharesOutstanding if req.sharesOutstanding > 0 else 0.0

    # ── Implied Multiples ─────────────────────────────────────────────────
    last_ebitda = projections[-1]["ebitda"]
    last_revenue = projections[-1]["revenue"]
    ev_to_ebitda = enterprise_value / last_ebitda if last_ebitda > 0 else None
    ev_to_revenue = enterprise_value / last_revenue if last_revenue > 0 else None

    # Year-1 multiples (current)
    y1_ebitda = projections[0]["ebitda"]
    y1_revenue = projections[0]["revenue"]
    ev_to_ebitda_y1 = enterprise_value / y1_ebitda if y1_ebitda > 0 else None
    ev_to_revenue_y1 = enterprise_value / y1_revenue if y1_revenue > 0 else None

    # ── FCF Yield ─────────────────────────────────────────────────────────
    fcf_yield = (projections[0]["fcf"] / equity_value * 100) if equity_value > 0 else None

    # ── TV as % of EV ─────────────────────────────────────────────────────
    tv_pct_of_ev = (pv_terminal_value / enterprise_value * 100) if enterprise_value > 0 else 0.0

    # ── CAGR ──────────────────────────────────────────────────────────────
    revenue_cagr = ((projections[-1]["revenue"] / req.baseRevenue) ** (1 / n) - 1) * 100
    fcf_cagr = None
    if projections[0]["fcf"] > 0 and projections[-1]["fcf"] > 0:
        fcf_cagr = ((projections[-1]["fcf"] / projections[0]["fcf"]) ** (1 / max(n - 1, 1)) - 1) * 100

    # ── Sensitivity Analysis (WACC × Terminal Growth) ─────────────────────
    wacc_range = [req.wacc + d for d in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    growth_range = [req.terminalGrowthRate + d for d in [-1.0, -0.5, 0.0, 0.5, 1.0]]

    sensitivity_matrix = []
    for w in wacc_range:
        row = []
        for g in growth_range:
            w_dec = w / 100.0
            g_dec = g / 100.0
            if w_dec <= g_dec:
                row.append(None)  # Invalid: WACC must exceed growth
            else:
                tv_s = (last_fcf * (1 + g_dec)) / (w_dec - g_dec)
                pv_tv_s = tv_s / ((1 + w_dec) ** n)
                # Recalculate PV of FCFs with new WACC
                sum_pv_s = sum(
                    p["fcf"] / ((1 + w_dec) ** p["year"])
                    for p in projections
                )
                ev_s = sum_pv_s + pv_tv_s
                eq_s = ev_s - net_debt
                price_s = eq_s / req.sharesOutstanding if req.sharesOutstanding > 0 else 0.0
                row.append(float(round(price_s, 2)))
        sensitivity_matrix.append(row)

    # ── Scenario Analysis (Bull / Base / Bear) ────────────────────────────
    scenarios = {}
    for scenario_name, wacc_adj, growth_adj, margin_adj in [
        ("bull", -1.0, 0.5, 2.0),
        ("base", 0.0, 0.0, 0.0),
        ("bear", 1.5, -0.5, -3.0),
    ]:
        s_wacc = (req.wacc + wacc_adj) / 100.0
        s_tgr = (req.terminalGrowthRate + growth_adj) / 100.0
        if s_wacc <= s_tgr:
            scenarios[scenario_name] = {"impliedSharePrice": None, "enterpriseValue": None, "equityValue": None}
            continue

        s_projections = []
        s_prev_rev = req.baseRevenue
        s_prev_nwc = req.baseRevenue * (req.nwcPctRevenue / 100.0)
        for i in range(n):
            g_s = growth_rates[i] / 100.0
            rev_s = s_prev_rev * (1 + g_s)
            if n > 1:
                mw = i / (n - 1)
            else:
                mw = 1.0
            m_s = (req.ebitdaMargin + margin_adj) + ((req.ebitdaMarginTerminal + margin_adj) - (req.ebitdaMargin + margin_adj)) * mw
            ebitda_s = rev_s * (m_s / 100.0)
            dep_s = rev_s * (req.depreciationPctRevenue / 100.0)
            ebit_s = ebitda_s - dep_s
            tax_s = max(0.0, ebit_s * (req.taxRate / 100.0))
            nopat_s = ebit_s - tax_s
            capex_s = rev_s * (req.capexPctRevenue / 100.0)
            nwc_s = rev_s * (req.nwcPctRevenue / 100.0)
            dnwc_s = nwc_s - s_prev_nwc
            fcf_s = nopat_s + dep_s - capex_s - dnwc_s
            s_projections.append({"year": i + 1, "fcf": fcf_s})
            s_prev_rev = rev_s
            s_prev_nwc = nwc_s

        s_last_fcf = s_projections[-1]["fcf"]
        s_tv = (s_last_fcf * (1 + s_tgr)) / (s_wacc - s_tgr)
        s_pv_tv = s_tv / ((1 + s_wacc) ** n)
        s_sum_pv = sum(p["fcf"] / ((1 + s_wacc) ** p["year"]) for p in s_projections)
        s_ev = s_sum_pv + s_pv_tv
        s_eq = s_ev - net_debt
        s_price = s_eq / req.sharesOutstanding if req.sharesOutstanding > 0 else 0.0

        scenarios[scenario_name] = {
            "impliedSharePrice": float(round(s_price, 2)),
            "enterpriseValue": float(round(s_ev, 2)),
            "equityValue": float(round(s_eq, 2)),
            "wacc": float(req.wacc + wacc_adj),
            "terminalGrowth": float(req.terminalGrowthRate + growth_adj),
            "ebitdaMarginAdj": float(margin_adj),
        }

    # ── Interpretation ────────────────────────────────────────────────────
    interpretation_parts = []
    interpretation_parts.append(
        f"A {n}-year Discounted Cash Flow analysis was performed with a base revenue of "
        f"${req.baseRevenue:,.1f}M, discounted at a WACC of {req.wacc:.1f}%."
    )
    interpretation_parts.append(
        f"Revenue is projected to grow from ${req.baseRevenue:,.1f}M to "
        f"${projections[-1]['revenue']:,.1f}M (CAGR: {revenue_cagr:.1f}%), "
        f"with EBITDA margins converging from {req.ebitdaMargin:.1f}% to {req.ebitdaMarginTerminal:.1f}%."
    )
    interpretation_parts.append(
        f"The model yields an Enterprise Value of ${enterprise_value:,.1f}M and "
        f"an Equity Value of ${equity_value:,.1f}M, "
        f"implying a share price of ${implied_share_price:,.2f}."
    )
    interpretation_parts.append(
        f"Terminal value accounts for {tv_pct_of_ev:.1f}% of Enterprise Value "
        f"(Gordon Growth at {req.terminalGrowthRate:.1f}% perpetuity growth)."
    )
    if tv_pct_of_ev > 75:
        interpretation_parts.append(
            "⚠ Terminal value represents over 75% of EV, indicating high sensitivity to terminal assumptions. "
            "Consider extending the projection period or stress-testing terminal growth."
        )
    interpretation = " ".join(interpretation_parts)

    # ── Warnings ──────────────────────────────────────────────────────────
    warnings = []
    if tv_pct_of_ev > 75:
        warnings.append({
            "type": "high_tv_dependency",
            "message": f"Terminal value is {tv_pct_of_ev:.1f}% of EV. Consider extending projection period.",
            "severity": "warning"
        })
    if any(p["fcf"] < 0 for p in projections):
        neg_years = [p["year"] for p in projections if p["fcf"] < 0]
        warnings.append({
            "type": "negative_fcf",
            "message": f"Negative FCF in Year(s) {neg_years}. Check margin and CapEx assumptions.",
            "severity": "warning"
        })
    if equity_value < 0:
        warnings.append({
            "type": "negative_equity",
            "message": "Equity value is negative. The company's debt exceeds its enterprise value under these assumptions.",
            "severity": "critical"
        })
    if req.wacc - req.terminalGrowthRate < 2.0:
        warnings.append({
            "type": "narrow_spread",
            "message": f"WACC-growth spread is only {req.wacc - req.terminalGrowthRate:.1f}%. Terminal value is highly sensitive.",
            "severity": "warning"
        })

    return {
        "projections": projections,
        "terminalValue": float(round(terminal_value, 2)),
        "pvTerminalValue": float(round(pv_terminal_value, 2)),
        "sumPVFCF": float(round(sum_pv_fcf, 2)),
        "enterpriseValue": float(round(enterprise_value, 2)),
        "netDebt": float(round(net_debt, 2)),
        "equityValue": float(round(equity_value, 2)),
        "impliedSharePrice": float(round(implied_share_price, 2)),
        "tvPctOfEV": float(round(tv_pct_of_ev, 2)),
        "multiples": {
            "evToEbitda_exit": float(round(ev_to_ebitda, 2)) if ev_to_ebitda else None,
            "evToRevenue_exit": float(round(ev_to_revenue, 2)) if ev_to_revenue else None,
            "evToEbitda_y1": float(round(ev_to_ebitda_y1, 2)) if ev_to_ebitda_y1 else None,
            "evToRevenue_y1": float(round(ev_to_revenue_y1, 2)) if ev_to_revenue_y1 else None,
            "fcfYield": float(round(fcf_yield, 2)) if fcf_yield else None,
        },
        "cagr": {
            "revenue": float(round(revenue_cagr, 2)),
            "fcf": float(round(fcf_cagr, 2)) if fcf_cagr is not None else None,
        },
        "sensitivity": {
            "waccRange": [float(round(w, 2)) for w in wacc_range],
            "growthRange": [float(round(g, 2)) for g in growth_range],
            "matrix": sensitivity_matrix,
        },
        "scenarios": scenarios,
        "interpretation": interpretation,
        "warnings": warnings,
        "inputs_echo": {
            "baseRevenue": req.baseRevenue,
            "projectionYears": n,
            "wacc": req.wacc,
            "terminalGrowthRate": req.terminalGrowthRate,
            "ebitdaMargin": req.ebitdaMargin,
            "ebitdaMarginTerminal": req.ebitdaMarginTerminal,
            "taxRate": req.taxRate,
            "totalDebt": req.totalDebt,
            "cashEquivalents": req.cashEquivalents,
            "sharesOutstanding": req.sharesOutstanding,
        },
    }


def generate_dcf_plot(result: dict, req: DCFRequest) -> str:
    """Generate a 2×2 plot panel for the DCF analysis."""
    projections = result["projections"]
    years = [f"Y{p['year']}" for p in projections]
    n = len(projections)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('DCF Valuation Model', fontsize=16, fontweight='bold')

    colors = sns.color_palette("crest", n_colors=n)

    # ── Plot 1: Revenue & EBITDA Bars ─────────────────────────────────────
    ax1 = axes[0, 0]
    revenues = [p["revenue"] for p in projections]
    ebitdas = [p["ebitda"] for p in projections]

    x = np.arange(n)
    width = 0.35
    bars1 = ax1.bar(x - width / 2, revenues, width, label='Revenue', color=colors[0], alpha=0.85)
    bars2 = ax1.bar(x + width / 2, ebitdas, width, label='EBITDA', color=colors[-1], alpha=0.85)

    ax1.set_title('Revenue & EBITDA Projection')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Amount ($M)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f'${val:,.0f}'))

    # Add growth rate labels
    for i, p in enumerate(projections):
        ax1.annotate(f'{p["growthRate"]:.0f}%',
                     xy=(x[i] - width / 2, revenues[i]),
                     ha='center', va='bottom', fontsize=8, color='gray')

    # ── Plot 2: FCF Waterfall ─────────────────────────────────────────────
    ax2 = axes[0, 1]
    fcfs = [p["fcf"] for p in projections]
    pv_fcfs = [p["pvFCF"] for p in projections]

    ax2.bar(x - width / 2, fcfs, width, label='FCF (Nominal)', color='#2196F3', alpha=0.8)
    ax2.bar(x + width / 2, pv_fcfs, width, label='PV of FCF', color='#FF9800', alpha=0.8)

    ax2.set_title('Free Cash Flow: Nominal vs Present Value')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Amount ($M)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.legend()
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f'${val:,.0f}'))

    # ── Plot 3: Valuation Bridge (Horizontal Bar) ─────────────────────────
    ax3 = axes[1, 0]
    bridge_labels = ['PV of FCFs', 'PV of TV', 'Enterprise Value', '(−) Net Debt', 'Equity Value']
    bridge_values = [
        result["sumPVFCF"],
        result["pvTerminalValue"],
        result["enterpriseValue"],
        -result["netDebt"],
        result["equityValue"],
    ]
    bridge_colors = ['#4CAF50', '#8BC34A', '#2196F3', '#F44336' if result["netDebt"] > 0 else '#4CAF50', '#1565C0']

    bars = ax3.barh(bridge_labels, bridge_values, color=bridge_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax3.set_title('Valuation Bridge')
    ax3.set_xlabel('Amount ($M)')
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f'${val:,.0f}'))

    # Add value labels
    for bar, val in zip(bars, bridge_values):
        ax3.text(bar.get_width() + max(abs(v) for v in bridge_values) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'${val:,.1f}M', va='center', fontsize=9)

    # ── Plot 4: EBITDA Margin Trend + FCF Margin ──────────────────────────
    ax4 = axes[1, 1]
    margins = [p["ebitdaMargin"] for p in projections]
    fcf_margins = [(p["fcf"] / p["revenue"] * 100) if p["revenue"] > 0 else 0 for p in projections]

    ax4.plot(years, margins, 'o-', color='#2196F3', linewidth=2, markersize=8, label='EBITDA Margin')
    ax4.plot(years, fcf_margins, 's--', color='#FF9800', linewidth=2, markersize=7, label='FCF Margin')
    ax4.fill_between(years, margins, alpha=0.1, color='#2196F3')
    ax4.fill_between(years, fcf_margins, alpha=0.1, color='#FF9800')

    ax4.set_title('Margin Trends')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Margin (%)')
    ax4.legend()
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f'{val:.0f}%'))

    # Add annotations
    for i, (m, f) in enumerate(zip(margins, fcf_margins)):
        ax4.annotate(f'{m:.1f}%', xy=(i, m), ha='center', va='bottom', fontsize=8, color='#1565C0')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def generate_sensitivity_heatmap(result: dict) -> str:
    """Generate a standalone sensitivity heatmap."""
    sensitivity = result["sensitivity"]
    matrix = np.array(sensitivity["matrix"], dtype=float)
    wacc_labels = [f'{w:.1f}%' for w in sensitivity["waccRange"]]
    growth_labels = [f'{g:.1f}%' for g in sensitivity["growthRange"]]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Mask invalid cells (None → NaN)
    masked = np.where(matrix == None, np.nan, matrix).astype(float)

    sns.heatmap(
        masked,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=growth_labels,
        yticklabels=wacc_labels,
        ax=ax,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Implied Share Price ($)'},
    )
    ax.set_title('Sensitivity Analysis: Implied Share Price\n(WACC vs Terminal Growth Rate)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Terminal Growth Rate', fontsize=12)
    ax.set_ylabel('WACC', fontsize=12)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


# ─── API Endpoint ─────────────────────────────────────────────────────────────

@router.post("/dcf")
def dcf_valuation(req: DCFRequest):
    """
    Perform a Discounted Cash Flow (DCF) valuation.
    
    Returns:
        - results: Full DCF output including projections, valuation bridge,
          sensitivity analysis, scenario analysis, and interpretation
        - plot: Base64-encoded 2x2 visualization panel
        - sensitivityPlot: Base64-encoded sensitivity heatmap
    """
    try:
        # ── Input Validation ──────────────────────────────────────────────
        if req.baseRevenue <= 0:
            raise HTTPException(status_code=400, detail="Base revenue must be positive.")

        if req.wacc <= 0:
            raise HTTPException(status_code=400, detail="WACC must be positive.")

        if req.wacc <= req.terminalGrowthRate:
            raise HTTPException(
                status_code=400,
                detail=f"WACC ({req.wacc}%) must exceed terminal growth rate ({req.terminalGrowthRate}%)."
            )

        if req.sharesOutstanding <= 0:
            raise HTTPException(status_code=400, detail="Shares outstanding must be positive.")

        if req.projectionYears < 1 or req.projectionYears > 20:
            raise HTTPException(status_code=400, detail="Projection years must be between 1 and 20.")

        if len(req.revenueGrowthRates) == 0:
            raise HTTPException(status_code=400, detail="At least one revenue growth rate is required.")

        if req.terminalGrowthRate < 0:
            raise HTTPException(status_code=400, detail="Terminal growth rate should not be negative.")

        if req.terminalGrowthRate > 5:
            # Not a hard error, but flag it
            pass  # Warning added in compute_dcf

        # ── Compute ──────────────────────────────────────────────────────
        result = compute_dcf(req)

        # ── Generate Plots ───────────────────────────────────────────────
        plot = generate_dcf_plot(result, req)
        sensitivity_plot = generate_sensitivity_heatmap(result)

        return {
            "results": result,
            "plot": plot,
            "sensitivityPlot": sensitivity_plot,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DCF computation error: {str(e)}")


# ─── Lightweight Health / Info Endpoint ───────────────────────────────────────

@router.get("/dcf/info")
def dcf_info():
    """Return metadata about the DCF endpoint."""
    return {
        "name": "DCF Valuation Model",
        "version": "1.0.0",
        "description": "Discounted Cash Flow intrinsic valuation with sensitivity & scenario analysis.",
        "inputs": {
            "required": ["baseRevenue", "revenueGrowthRates", "ebitdaMargin", "ebitdaMarginTerminal", "wacc", "terminalGrowthRate"],
            "optional": ["projectionYears", "depreciationPctRevenue", "capexPctRevenue", "nwcPctRevenue", "taxRate", "totalDebt", "cashEquivalents", "sharesOutstanding"],
        },
        "outputs": ["projections", "terminalValue", "enterpriseValue", "equityValue", "impliedSharePrice", "sensitivity", "scenarios", "plot", "sensitivityPlot"],
    }
