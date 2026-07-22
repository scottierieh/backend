#!/usr/bin/env python3
"""Inventory Optimization — Economic Order Quantity (EOQ) and reorder point.

Input (from inventory-optimization-page.tsx):
    annual_demand   : float  D  (units/year)
    ordering_cost   : float  S  (cost per order)
    holding_cost    : float  H  (holding cost per unit per year)
    lead_time_days  : float  L  (days)
    safety_stock    : float  SS (units, default 0)
Output: { results: {eoq, reorder_point, ...}, plot }
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
        D = float(p.get("annual_demand"))
        S = float(p.get("ordering_cost"))
        H = float(p.get("holding_cost"))
        L = float(p.get("lead_time_days") or 0.0)
        SS = float(p.get("safety_stock") or 0.0)
        if D <= 0 or S <= 0 or H <= 0:
            raise ValueError("Annual demand, ordering cost and holding cost must all be positive.")

        eoq = float(np.sqrt(2 * D * S / H))
        orders_per_year = D / eoq
        cycle_time_days = 365.0 / orders_per_year
        daily_demand = D / 365.0
        reorder_point = daily_demand * L + SS
        ordering_total = orders_per_year * S
        holding_total = (eoq / 2.0 + SS) * H
        total_cost = ordering_total + holding_total

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=120)
            q = np.linspace(max(eoq * 0.15, 1), eoq * 2.5, 300)
            oc = D / q * S
            hc = (q / 2.0 + SS) * H
            tc = oc + hc
            ax.plot(q, oc, "--", color="#f59e0b", label="Ordering cost")
            ax.plot(q, hc, "--", color="#16a34a", label="Holding cost")
            ax.plot(q, tc, color="#2563eb", lw=2.2, label="Total cost")
            ax.axvline(eoq, color="#dc2626", ls=":", lw=1.4, label=f"EOQ = {eoq:,.0f}")
            ax.scatter([eoq], [total_cost], color="#dc2626", s=60, zorder=6)
            ax.set_xlabel("Order quantity (units)"); ax.set_ylabel("Annual cost")
            ax.set_title("EOQ — total cost minimised where ordering = holding")
            ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The economic order quantity is {eoq:,.1f} units — the order size that minimises the sum of ordering "
            f"and holding costs, which are equal at the optimum ({ordering_total:,.2f} each region). That means placing "
            f"about {orders_per_year:,.1f} orders per year, one roughly every {cycle_time_days:,.1f} days, for a total "
            f"annual inventory cost of {total_cost:,.2f}. With a {L:g}-day lead time"
            + (f" and {SS:,.0f} units of safety stock" if SS > 0 else "")
            + f", you should reorder when stock falls to {reorder_point:,.1f} units. Ordering more than the EOQ raises "
            "holding cost faster than it saves on ordering; ordering less does the reverse."
        )

        results = {
            "status": "ok",
            "eoq": _fin(eoq, 2), "reorder_point": _fin(reorder_point, 2),
            "orders_per_year": _fin(orders_per_year, 3), "cycle_time_days": _fin(cycle_time_days, 2),
            "ordering_cost_total": _fin(ordering_total, 2), "holding_cost_total": _fin(holding_total, 2),
            "total_cost": _fin(total_cost, 2), "safety_stock": _fin(SS, 2),
            "annual_demand": _fin(D, 2), "lead_time_days": _fin(L, 2),
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
