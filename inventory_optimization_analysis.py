#!/usr/bin/env python3
"""Inventory Optimization — Economic Order Quantity (EOQ) + reorder point.
Closed-form, no solver needed.

Input: annual_demand D, ordering_cost S, holding_cost H (per unit/year),
       lead_time_days L (optional), safety_stock SS (optional),
       working_days (optional, default 365)
Output: results{eoq, reorder_point, orders_per_year, cycle_time_days,
                ordering_cost_total, holding_cost_total, total_cost,
                safety_stock, interpretation}, plot
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
        L = float(p.get("lead_time_days") or 0)
        SS = float(p.get("safety_stock") or 0)
        WD = float(p.get("working_days") or 365)
        if D <= 0 or S < 0 or H <= 0:
            raise ValueError("Annual demand and holding cost must be positive; ordering cost non-negative.")

        eoq = float(np.sqrt(2 * D * S / H))
        orders = D / eoq
        cycle_days = WD / orders if orders > 0 else None
        daily_demand = D / WD
        rop = daily_demand * L + SS
        order_cost_total = (D / eoq) * S
        hold_cost_total = (eoq / 2 + SS) * H
        total = order_cost_total + hold_cost_total

        plot = None
        try:
            # total-cost curve vs order quantity
            qs = np.linspace(max(eoq * 0.2, 1), eoq * 2.2, 200)
            oc = (D / qs) * S
            hc = (qs / 2) * H
            tc = oc + hc
            fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
            ax.plot(qs, oc, "--", color="#f59e0b", label="Ordering cost")
            ax.plot(qs, hc, "--", color="#10b981", label="Holding cost")
            ax.plot(qs, tc, color="#2563eb", lw=2, label="Total cost")
            ax.axvline(eoq, color="#dc2626", ls=":", lw=1.5)
            ax.annotate(f"EOQ = {eoq:.0f}", (eoq, tc.min()), textcoords="offset points",
                        xytext=(8, 20), color="#dc2626", fontsize=9)
            ax.set_xlabel("Order quantity"); ax.set_ylabel("Annual cost")
            ax.set_title("EOQ — total cost is minimised where ordering = holding cost")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The economic order quantity is {eoq:,.1f} units — order this much each time to "
            f"minimise combined ordering and holding cost at {total:,.2f}/year. That means about "
            f"{orders:.1f} orders per year, roughly one every {cycle_days:.0f} days. "
            + (f"With a {L:g}-day lead time"
               + (f" and {SS:g} units of safety stock" if SS else "")
               + f", reorder when stock falls to {rop:,.1f} units. " if L or SS else "")
            + "At the EOQ the ordering and holding costs are equal — that balance is what makes it optimal."
        )
        results = {
            "status": "optimal", "unsolved": False,
            "eoq": _fin(eoq, 2), "reorder_point": _fin(rop, 2),
            "orders_per_year": _fin(orders, 2), "cycle_time_days": _fin(cycle_days, 1),
            "ordering_cost_total": _fin(order_cost_total, 2), "holding_cost_total": _fin(hold_cost_total, 2),
            "total_cost": _fin(total, 2), "safety_stock": _fin(SS, 2),
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
