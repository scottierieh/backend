#!/usr/bin/env python3
"""Network Flow Optimization — minimum-cost flow. scipy linprog.

Input (from network-flow-optimization-page.tsx):
    node_names    : string[]
    node_balance  : number[]   supply (+) / demand (-) / transshipment (0)
    edge_from     : string[]
    edge_to       : string[]
    edge_capacity : number[]
    edge_cost     : number[]
Output: { results: {edges[], total_cost, ...}, plot }
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
        nodes = p.get("node_names") or []
        balance = [float(x) for x in (p.get("node_balance") or [])]
        efrom = p.get("edge_from") or []
        eto = p.get("edge_to") or []
        cap = [float(x) for x in (p.get("edge_capacity") or [])]
        cost = [float(x) for x in (p.get("edge_cost") or [])]
        E = len(efrom)
        if not nodes or E == 0:
            raise ValueError("Provide nodes and at least one edge.")
        if not (len(eto) == len(cap) == len(cost) == E):
            raise ValueError("Edge arrays must all have the same length.")
        if len(balance) != len(nodes):
            raise ValueError("node_balance must have one entry per node.")
        if abs(sum(balance)) > 1e-6:
            raise ValueError(f"Node balances must sum to zero (supply = demand); they sum to {sum(balance):g}.")

        idx = {nm: i for i, nm in enumerate(nodes)}
        for a, b in zip(efrom, eto):
            if a not in idx or b not in idx:
                raise ValueError(f"Edge endpoint not found among nodes: {a} -> {b}.")

        # conservation: for node k, outflow - inflow = balance_k
        A_eq = np.zeros((len(nodes), E))
        for e in range(E):
            A_eq[idx[efrom[e]], e] += 1.0
            A_eq[idx[eto[e]], e] -= 1.0
        b_eq = np.array(balance)
        res = linprog(np.array(cost), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0, cap[e]) for e in range(E)], method="highs")
        if not res.success or res.x is None:
            results = {"status": "unsolved", "unsolved": True, "message": res.message or "infeasible",
                       "n_nodes": len(nodes), "n_edges": E,
                       "interpretation": f"No feasible flow exists: {res.message}. Check capacities and that supply can reach demand."}
            print(json.dumps({"results": results, "plot": None})); return

        f = np.array(res.x)
        total_cost = float(np.dot(cost, f))
        edges = []
        for e in range(E):
            edges.append({"from": efrom[e], "to": eto[e], "flow": _fin(f[e], 4),
                          "capacity": _fin(cap[e], 4), "cost": _fin(cost[e], 4)})

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(7, E * 0.7 + 2), 5), dpi=120)
            lbls = [f"{efrom[e]}→{eto[e]}" for e in range(E)]
            xs = np.arange(E)
            ax.bar(xs, cap, color="#e5e7eb", label="Capacity")
            used = ["#2563eb" if f[e] > 1e-7 else "#cbd5e1" for e in range(E)]
            ax.bar(xs, f, color=used, label="Flow")
            for e in range(E):
                if f[e] > 1e-7:
                    ax.text(e, f[e], f"{f[e]:g}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(xs); ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("Flow / capacity"); ax.set_title(f"Optimal flow per edge (total cost = {total_cost:g})")
            ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        n_used = sum(1 for e in range(E) if f[e] > 1e-7)
        n_sat = sum(1 for e in range(E) if abs(f[e] - cap[e]) < 1e-6 and cap[e] > 0)
        interpretation = (
            f"The minimum-cost flow routes all supply to demand for a total cost of {total_cost:,.4g}, using "
            f"{n_used} of the {E} available edges. {n_sat} edge(s) are at full capacity — these are bottlenecks where "
            "extra capacity would most likely reduce cost. The solver balances cheap-but-limited routes against more "
            "expensive alternatives so that every node's supply/demand balance is met exactly while total cost is minimised."
        )

        results = {
            "status": "ok", "unsolved": False,
            "n_nodes": len(nodes), "n_edges": E, "total_cost": _fin(total_cost, 4),
            "edges": edges, "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
