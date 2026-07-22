#!/usr/bin/env python3
"""Transportation Problem — minimum-cost shipping plan. scipy linprog.

Input (from transportation-problem-page.tsx):
    source_names : string[]
    supply       : number[]
    dest_names   : string[]
    demand       : number[]
    cost_matrix  : number[][]   rows = sources, cols = destinations
Output: { results: {flows[], total_cost, ...}, plot }
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
        sources = p.get("source_names") or []
        supply = [float(x) for x in (p.get("supply") or [])]
        dests = p.get("dest_names") or []
        demand = [float(x) for x in (p.get("demand") or [])]
        cost = p.get("cost_matrix")
        if cost is None or not sources or not dests:
            raise ValueError("Provide source/destination names, supply, demand and a cost matrix.")
        C = np.array(cost, dtype=float)
        m, n = len(sources), len(dests)
        if C.shape != (m, n):
            raise ValueError("Cost matrix must be sources (rows) by destinations (columns).")
        if len(supply) != m or len(demand) != n:
            raise ValueError("Supply and demand lengths must match sources and destinations.")

        total_supply, total_demand = sum(supply), sum(demand)
        if total_supply + 1e-9 < total_demand:
            results = {"status": "unsolved", "unsolved": True,
                       "message": f"Total supply ({total_supply:g}) is less than total demand ({total_demand:g}); the problem is infeasible.",
                       "n_sources": m, "n_destinations": n,
                       "interpretation": "Demand cannot be met because total supply is insufficient. Increase supply or reduce demand."}
            print(json.dumps({"results": results, "plot": None})); return

        # variables x_ij flattened (i*n + j); minimise sum c_ij x_ij
        c = C.reshape(-1)
        # supply: sum_j x_ij <= supply_i
        A_ub, b_ub = [], []
        for i in range(m):
            row = np.zeros(m * n); row[i * n:(i + 1) * n] = 1.0
            A_ub.append(row); b_ub.append(supply[i])
        # demand: sum_i x_ij >= demand_j  ->  -sum_i x_ij <= -demand_j
        for j in range(n):
            row = np.zeros(m * n); row[j::n] = -1.0
            A_ub.append(row); b_ub.append(-demand[j])
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * (m * n), method="highs")
        if not res.success or res.x is None:
            results = {"status": "unsolved", "unsolved": True, "message": res.message or "infeasible",
                       "n_sources": m, "n_destinations": n,
                       "interpretation": f"No feasible shipping plan: {res.message}"}
            print(json.dumps({"results": results, "plot": None})); return

        X = np.array(res.x).reshape(m, n)
        total_cost = float((C * X).sum())
        flows = []
        for i in range(m):
            for j in range(n):
                if X[i, j] > 1e-7:
                    flows.append({"from": sources[i], "to": dests[j],
                                  "amount": _fin(X[i, j], 4), "cost": _fin(X[i, j] * C[i, j], 4)})

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(6, n * 0.9 + 2), max(4.5, m * 0.7 + 1.5)), dpi=120)
            im = ax.imshow(X, cmap="Greens", aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels(dests, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(m)); ax.set_yticklabels(sources, fontsize=8)
            for i in range(m):
                for j in range(n):
                    if X[i, j] > 1e-7:
                        ax.text(j, i, f"{X[i,j]:g}", ha="center", va="center", fontsize=8,
                                color="white" if X[i, j] > X.max() * 0.5 else "#111827")
            ax.set_title(f"Optimal shipment plan (total cost = {total_cost:g})")
            fig.colorbar(im, ax=ax, label="Units shipped")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The minimum-cost plan ships all {total_demand:g} units of demand for a total transport cost of "
            f"{total_cost:,.4g}, using {len(flows)} of the {m*n} possible routes. "
            + (f"Total supply ({total_supply:g}) exceeds demand ({total_demand:g}), so {total_supply-total_demand:g} units "
               "stay unshipped at the cheapest-to-hold sources. " if total_supply > total_demand + 1e-9 else
               "Supply and demand are balanced, so every unit is shipped. ")
            + "The solution favours low-cost lanes but respects each source's capacity and each destination's requirement, "
            "which is why some shipments do not take the single cheapest route."
        )

        results = {
            "status": "ok", "unsolved": False,
            "n_sources": m, "n_destinations": n, "total_cost": _fin(total_cost, 4),
            "flows": flows, "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
