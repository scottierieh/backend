#!/usr/bin/env python3
"""Production Planning — multi-period lot-sizing at min production+holding cost.
LP via scipy.optimize.linprog.

Input: period_names[], demand[], prod_cost[], hold_cost[], capacity[],
       start_inventory (scalar)
Output: results{status, unsolved, message, n_periods, total_cost,
                total_production, periods:[{name,demand,production,inventory,
                prod_cost,hold_cost}], interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import linprog

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
        names = [str(x) for x in (p.get("period_names") or [])]
        demand = [float(x) for x in (p.get("demand") or [])]
        pc = [float(x) for x in (p.get("prod_cost") or [])]
        hc = [float(x) for x in (p.get("hold_cost") or [])]
        cap = [float(x) for x in (p.get("capacity") or [])]
        I0 = float(p.get("start_inventory") or 0)
        T = len(names)
        if T < 1:
            raise ValueError("Need at least one period.")
        if not (len(demand) == len(pc) == len(hc) == len(cap) == T):
            raise ValueError("All period arrays must have the same length.")

        # vars: p_0..p_{T-1} (production), I_0..I_{T-1} (end inventory) -> 2T vars
        # inventory balance: I_t = I_{t-1} + p_t - d_t  (I_{-1}=I0)
        # => p_t - I_t + I_{t-1} = d_t ; as equality A_eq v = b
        nv = 2 * T
        c = np.array(pc + hc, dtype=float)
        A_eq = np.zeros((T, nv)); b_eq = np.zeros(T)
        for t in range(T):
            A_eq[t, t] = 1.0          # p_t
            A_eq[t, T + t] = -1.0     # -I_t
            if t > 0:
                A_eq[t, T + t - 1] = 1.0   # +I_{t-1}
            b_eq[t] = demand[t] - (I0 if t == 0 else 0.0)
        bounds = [(0, cap[t]) for t in range(T)] + [(0, None)] * T
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            results = {"status": "infeasible", "unsolved": True,
                       "message": f"Demand cannot be met within capacity: {res.message}"}
            print(json.dumps({"results": results, "plot": None}))
            return
        v = res.x
        prod = v[:T]; inv = v[T:]
        total = float(np.dot(c, v))
        periods = []
        for t in range(T):
            periods.append({"name": names[t], "demand": _fin(demand[t], 2),
                            "production": _fin(prod[t], 2), "inventory": _fin(inv[t], 2),
                            "prod_cost": _fin(prod[t] * pc[t], 2), "hold_cost": _fin(inv[t] * hc[t], 2)})
        total_prod = float(np.sum(prod))

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(7, T * 0.9), 4.6), dpi=120)
            x = np.arange(T)
            ax.bar(x - 0.2, demand, width=0.4, color="#93c5fd", label="Demand")
            ax.bar(x + 0.2, prod, width=0.4, color="#2563eb", label="Production")
            ax.plot(x, inv, "-o", color="#dc2626", label="End inventory")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"Production plan — total cost {total:,.2f}")
            ax.legend(fontsize=8, frameon=False); fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The least-cost plan produces {total_prod:,.0f} units across {T} periods for a total of "
            f"{total:,.2f}. It trades off producing early (and paying to hold stock) against producing "
            "in cheaper or higher-capacity periods — building inventory only when that is cheaper than "
            "producing later. Periods where end inventory is zero are met exactly on time."
        )
        results = {"status": "optimal", "unsolved": False, "n_periods": T,
                   "total_cost": _fin(total, 2), "total_production": _fin(total_prod, 2),
                   "periods": periods, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
