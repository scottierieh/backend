#!/usr/bin/env python3
"""Enterprise Value Analysis — the bridge from equity to enterprise value. numpy.

EV = market cap + total debt + preferred equity + minority interest - cash.
Also derives common EV and equity multiples if the operating/earnings figures
are supplied.

Input (from enterprise-value-page.tsx):
    market_cap        : float   (share price x shares, or enter directly)
    share_price       : float   (optional, with shares_outstanding)
    shares_outstanding: float   (optional)
    total_debt        : float
    cash              : float
    preferred_equity  : float   (optional)
    minority_interest : float   (optional)
    ebitda, ebit, revenue, net_income : float (optional, for multiples)
Output: { results: {ev bridge, multiples}, plot }
"""
import sys, json, io, base64
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def main():
    try:
        p = json.load(sys.stdin)
        sp = p.get("share_price"); sh = p.get("shares_outstanding")
        mc = p.get("market_cap")
        if sp is not None and sh is not None and str(sp) != "" and str(sh) != "":
            market_cap = float(sp) * float(sh)
        elif mc is not None and str(mc) != "":
            market_cap = float(mc)
        else:
            raise ValueError("Provide either market cap, or share price and shares outstanding.")
        if market_cap <= 0:
            raise ValueError("Market cap must be positive.")

        total_debt = float(p.get("total_debt") or 0.0)
        cash = float(p.get("cash") or 0.0)
        preferred = float(p.get("preferred_equity") or 0.0)
        minority = float(p.get("minority_interest") or 0.0)

        net_debt = total_debt - cash
        ev = market_cap + total_debt + preferred + minority - cash

        # bridge components (signed contributions to EV from market cap)
        bridge = [
            {"label": "Market cap", "value": _fin(market_cap, 2), "kind": "base"},
            {"label": "+ Total debt", "value": _fin(total_debt, 2), "kind": "add"},
            {"label": "+ Preferred", "value": _fin(preferred, 2), "kind": "add"},
            {"label": "+ Minority interest", "value": _fin(minority, 2), "kind": "add"},
            {"label": "- Cash", "value": _fin(-cash, 2), "kind": "sub"},
            {"label": "Enterprise value", "value": _fin(ev, 2), "kind": "total"},
        ]

        def mult(numer, denom):
            try:
                d = float(denom)
                return _fin(numer / d, 4) if d not in (0, None) else None
            except (TypeError, ValueError):
                return None

        ebitda = p.get("ebitda"); ebit = p.get("ebit"); revenue = p.get("revenue"); ni = p.get("net_income")
        def val(x):
            try:
                return float(x) if x is not None and str(x) != "" else None
            except (TypeError, ValueError):
                return None
        ebitda, ebit, revenue, ni = val(ebitda), val(ebit), val(revenue), val(ni)
        multiples = {
            "ev_ebitda": mult(ev, ebitda) if ebitda else None,
            "ev_ebit": mult(ev, ebit) if ebit else None,
            "ev_revenue": mult(ev, revenue) if revenue else None,
            "pe": mult(market_cap, ni) if ni else None,
        }
        net_debt_ebitda = mult(net_debt, ebitda) if ebitda else None

        # plot: EV bridge waterfall
        plot = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 5), dpi=120)
            steps = [("Market\ncap", market_cap, "#2563eb"),
                     ("+ Debt", total_debt, "#16a34a"),
                     ("+ Pref", preferred, "#16a34a"),
                     ("+ Minority", minority, "#16a34a"),
                     ("- Cash", -cash, "#dc2626")]
            running = 0.0
            xs = []
            for i, (name, delta, col) in enumerate(steps):
                if delta == 0 and name not in ("Market\ncap",):
                    continue
                bottom = running if delta >= 0 else running + delta
                ax.bar(i, abs(delta), bottom=bottom, color=col, edgecolor="white")
                ax.text(i, running + delta + (0.01 * ev if ev else 0), f"{delta:,.0f}", ha="center", va="bottom", fontsize=8)
                running += delta
                xs.append(name)
            ax.bar(len(steps), ev, color="#f59e0b", edgecolor="white")
            ax.text(len(steps), ev, f"{ev:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            xs.append("Enterprise\nvalue")
            ax.set_xticks(range(len(steps) + 1)); ax.set_xticklabels(xs, fontsize=8)
            ax.set_ylabel("Value"); ax.set_title(f"Equity → enterprise value bridge (EV = {ev:,.0f})")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The enterprise value is {ev:,.0f}, built from a market capitalisation of {market_cap:,.0f} by adding "
            f"{total_debt:,.0f} of debt"
            + (f", {preferred:,.0f} preferred and {minority:,.0f} minority interest" if (preferred or minority) else "")
            + f" and subtracting {cash:,.0f} of cash (net debt {net_debt:,.0f}). Enterprise value measures the cost "
            "to acquire the whole business regardless of how it is financed — it is what you pay to control all the "
            "operating assets, which is why acquisition and comparison multiples are usually built on EV rather than "
            "equity value. "
            + (f"At these figures the business trades at {multiples['ev_ebitda']:.1f}x EV/EBITDA. " if multiples["ev_ebitda"] else "")
        )

        results = {
            "status": "ok", "market_cap": _fin(market_cap, 2), "total_debt": _fin(total_debt, 2),
            "cash": _fin(cash, 2), "net_debt": _fin(net_debt, 2), "preferred_equity": _fin(preferred, 2),
            "minority_interest": _fin(minority, 2), "enterprise_value": _fin(ev, 2),
            "bridge": bridge, "multiples": multiples, "net_debt_ebitda": net_debt_ebitda,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
