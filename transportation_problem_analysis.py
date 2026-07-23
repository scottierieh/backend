#!/usr/bin/env python3
"""Transportation Problem — min-cost shipping from sources to destinations.

CLI contract: read one JSON object from stdin, print one JSON object to stdout,
or {"error": ...} to stderr + exit(1).

Input (from transportation-problem-page.tsx):
    source_names : string[m]
    supply       : number[m]
    dest_names   : string[n]
    demand       : number[n]
    cost_matrix  : number[m][n]   per-unit shipping cost source x dest

Output: { results: { status, unsolved, message, n_sources, n_destinations,
                     total_cost, flows:[{from,to,amount,cost}], interpretation },
          plot } (allocation heatmap, base64).
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
        src = [str(x) for x in (p.get("source_names") or [])]
        dst = [str(x) for x in (p.get("dest_names") or [])]
        supply = [float(x) for x in (p.get("supply") or [])]
        demand = [float(x) for x in (p.get("demand") or [])]
        C = [[float(v) for v in row] for row in (p.get("cost_matrix") or [])]
        m, n = len(src), len(dst)
        if m < 1 or n < 1:
            raise ValueError("Need at least one source and one destination.")
        if len(supply) != m or len(demand) != n:
            raise ValueError("supply/demand length must match source/destination counts.")
        if len(C) != m or any(len(r) != n for r in C):
            raise ValueError(f"cost_matrix must be {m} x {n}.")
        if any(s < 0 for s in supply) or any(d < 0 for d in demand):
            raise ValueError("Supply and demand must be non-negative.")

        tot_s, tot_d = sum(supply), sum(demand)
        if tot_s + 1e-9 < tot_d:
            results = {"status": "infeasible", "unsolved": True,
                       "message": f"Total supply ({tot_s:g}) is less than total demand ({tot_d:g}) — demand cannot be met."}
            print(json.dumps({"results": results, "plot": None}))
            return

        # variables x_ij flattened row-major
        c = np.array(C, dtype=float).flatten()
        # supply constraints: sum_j x_ij <= supply_i  (<=)
        A_ub, b_ub = [], []
        for i in range(m):
            row = np.zeros(m * n); row[i*n:(i+1)*n] = 1.0
            A_ub.append(row); b_ub.append(supply[i])
        # demand constraints: sum_i x_ij >= demand_j  ->  -sum <= -demand
        for j in range(n):
            row = np.zeros(m * n); row[j::n] = 1.0
            A_ub.append(-row); b_ub.append(-demand[j])
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=[(0, None)] * (m * n), method="highs")
        if not res.success:
            results = {"status": "infeasible", "unsolved": True,
                       "message": f"No feasible shipping plan: {res.message}"}
            print(json.dumps({"results": results, "plot": None}))
            return

        X = res.x.reshape(m, n)
        total_cost = float(np.sum(X * np.array(C)))
        tol = 1e-6
        flows = []
        for i in range(m):
            for j in range(n):
                if X[i, j] > tol:
                    flows.append({"from": src[i], "to": dst[j],
                                  "amount": _fin(X[i, j], 4), "cost": _fin(X[i, j] * C[i][j], 4)})
        n_routes = len(flows)

        # plot: allocation heatmap
        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(6, n * 1.1), max(4, m * 0.8)), dpi=120)
            im = ax.imshow(X, cmap="Blues", aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels(dst, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(m)); ax.set_yticklabels(src, fontsize=8)
            for i in range(m):
                for j in range(n):
                    if X[i, j] > tol:
                        ax.text(j, i, f"{X[i,j]:g}", ha="center", va="center",
                                color="white" if X[i, j] > X.max()*0.6 else "#1e3a8a", fontsize=8)
            ax.set_title("Optimal shipment quantities")
            fig.colorbar(im, ax=ax, shrink=0.8, label="units shipped")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        balanced = abs(tot_s - tot_d) < 1e-6
        interpretation = (
            f"The cheapest way to meet all demand ships on {n_routes} of the {m*n} possible "
            f"lanes for a total cost of {total_cost:,.2f}. "
            + ("Supply equals demand, so the plan is balanced and every unit is used. "
               if balanced else
               f"Total supply ({tot_s:g}) exceeds demand ({tot_d:g}) by {tot_s-tot_d:g}, which stays unshipped at the cheapest sources. ")
            + "Because the transportation structure guarantees integer solutions for integer "
            "supply/demand, these quantities are whole units with no rounding."
        )

        results = {
            "status": "optimal", "unsolved": False,
            "n_sources": m, "n_destinations": n,
            "total_cost": _fin(total_cost, 4),
            "flows": flows, "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
