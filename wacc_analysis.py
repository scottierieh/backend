#!/usr/bin/env python3
"""WACC — weighted average cost of capital. numpy only.

Cost of equity is taken directly, or built from CAPM if capm inputs are given.

Input (from wacc-analysis-page.tsx):
    market_value_equity : float   E (market cap)
    market_value_debt   : float   D
    cost_of_debt        : float   pre-tax Kd (e.g. 0.06)
    tax_rate            : float    corporate tax rate (e.g. 0.25)
    cost_of_equity      : float   (optional) Ke directly, else use CAPM below
    risk_free_rate      : float   (optional, CAPM) Rf
    beta                : float   (optional, CAPM)
    market_return       : float   (optional, CAPM) E[Rm]
Output: { results: {...}, plot } (capital structure + cost bar).
"""
import sys, json, io, base64
import numpy as np

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
        E = float(p.get("market_value_equity") or 0.0)
        D = float(p.get("market_value_debt") or 0.0)
        kd = float(p.get("cost_of_debt") or 0.0)
        tax = float(p.get("tax_rate") or 0.0)
        if E <= 0 and D <= 0:
            raise ValueError("Provide the market value of equity and/or debt (at least one must be positive).")
        if not (0 <= tax < 1):
            raise ValueError("Tax rate must be between 0 and 1 (e.g. 0.25 for 25%).")

        # cost of equity: direct, or CAPM
        ke_direct = p.get("cost_of_equity")
        capm = None
        if ke_direct is not None and str(ke_direct) != "":
            ke = float(ke_direct)
        else:
            rf = float(p.get("risk_free_rate"))
            beta = float(p.get("beta"))
            rm = float(p.get("market_return"))
            ke = rf + beta * (rm - rf)
            capm = {"risk_free_rate": _fin(rf), "beta": _fin(beta), "market_return": _fin(rm),
                    "equity_risk_premium": _fin(rm - rf)}

        V = E + D
        we = E / V
        wd = D / V
        kd_after_tax = kd * (1 - tax)
        wacc = we * ke + wd * kd_after_tax

        # plot: capital structure pie + cost components bar
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=120)
            ax1.pie([E, D], labels=[f"Equity\n{we:.0%}", f"Debt\n{wd:.0%}"],
                    colors=["#2563eb", "#f59e0b"], autopct=lambda v: f"{v:.0f}%",
                    startangle=90, wedgeprops={"edgecolor": "white"})
            ax1.set_title("Capital structure (market value)")
            names = ["Cost of\nequity", "Cost of debt\n(after-tax)", "WACC"]
            vals = [ke * 100, kd_after_tax * 100, wacc * 100]
            bars = ax2.bar(names, vals, color=["#2563eb", "#f59e0b", "#16a34a"])
            for b, v in zip(bars, vals):
                ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
            ax2.set_ylabel("Cost (%)"); ax2.set_title("Cost of capital components")
            ax2.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"With {we:.0%} equity and {wd:.0%} debt, the blended cost of capital (WACC) is {wacc:.2%}. "
            f"Equity is the pricier source at {ke:.2%}; debt costs {kd:.2%} before tax but only "
            f"{kd_after_tax:.2%} after the {tax:.0%} tax shield, which is why leverage can lower the average "
            f"cost — up to the point where the added financial risk pushes both costs back up. This WACC is the "
            f"discount rate you would use to value the firm's cash flows (e.g. in a DCF)."
        )

        results = {
            "status": "ok",
            "market_value_equity": _fin(E, 2), "market_value_debt": _fin(D, 2), "total_value": _fin(V, 2),
            "weight_equity": _fin(we, 6), "weight_debt": _fin(wd, 6),
            "cost_of_equity": _fin(ke, 6), "cost_of_debt": _fin(kd, 6),
            "cost_of_debt_after_tax": _fin(kd_after_tax, 6), "tax_rate": _fin(tax, 6),
            "wacc": _fin(wacc, 6), "capm": capm,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
