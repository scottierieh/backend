#!/usr/bin/env python3
"""Network Flow Optimization — min-cost flow satisfying per-node supply/demand.
LP via scipy.optimize.linprog.

Input: node_names[], node_balance[] (>0 supply, <0 demand, 0 transship),
       edge_from[], edge_to[], edge_capacity[], edge_cost[]
Output: results{status, unsolved, message, n_nodes, n_edges, total_cost,
                edges:[{from,to,capacity,cost,flow}], interpretation}, plot
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
        nodes = [str(x) for x in (p.get("node_names") or [])]
        bal = [float(x) for x in (p.get("node_balance") or [])]
        ef = [str(x) for x in (p.get("edge_from") or [])]
        et = [str(x) for x in (p.get("edge_to") or [])]
        ecap = [float(x) for x in (p.get("edge_capacity") or [])]
        ecost = [float(x) for x in (p.get("edge_cost") or [])]
        N, E = len(nodes), len(ef)
        if N < 2 or E < 1:
            raise ValueError("Need at least 2 nodes and one edge.")
        if len(bal) != N:
            raise ValueError("node_balance length must match node_names.")
        if not (len(et) == len(ecap) == len(ecost) == E):
            raise ValueError("edge arrays must all have the same length.")
        if abs(sum(bal)) > 1e-6:
            raise ValueError(f"Total supply must equal total demand (sum of balances = {sum(bal):g}, should be 0).")
        idx = {n: i for i, n in enumerate(nodes)}
        A_eq = np.zeros((N, E))
        for e, (a, b) in enumerate(zip(ef, et)):
            A_eq[idx[a], e] += 1.0
            A_eq[idx[b], e] -= 1.0
        b_eq = np.array(bal)   # net outflow at node = its balance
        res = linprog(np.array(ecost), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0, c) for c in ecap], method="highs")
        if not res.success:
            results = {"status": "infeasible", "unsolved": True,
                       "message": f"No feasible flow meets all node balances within capacities: {res.message}"}
            print(json.dumps({"results": results, "plot": None}))
            return
        f = res.x
        total = float(np.dot(ecost, f))
        edges = [{"from": ef[e], "to": et[e], "capacity": _fin(ecap[e], 4),
                  "cost": _fin(ecost[e], 4), "flow": _fin(max(f[e], 0), 4)} for e in range(E)]
        used = [e for e in edges if (e["flow"] or 0) > 1e-6]

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(7, E * 0.55), 4.5), dpi=120)
            ys = np.arange(E)
            ax.barh(ys, [e["capacity"] for e in edges], color="#e5e7eb", label="capacity")
            ax.barh(ys, [e["flow"] for e in edges], color="#2563eb", label="flow")
            ax.set_yticks(ys); ax.set_yticklabels([f"{e['from']}→{e['to']}" for e in edges], fontsize=8)
            ax.invert_yaxis(); ax.set_xlabel("units")
            ax.set_title(f"Min-cost network flow — total cost {total:,.2f}")
            ax.legend(fontsize=8, frameon=False); fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        n_sup = sum(1 for b in bal if b > 1e-9); n_dem = sum(1 for b in bal if b < -1e-9)
        interpretation = (
            f"The cheapest flow satisfying {n_sup} supply node(s) and {n_dem} demand node(s) costs "
            f"{total:,.2f}, using {len(used)} of {E} edges. Every node's inflow minus outflow equals "
            "its balance exactly; cheap edges carry as much as capacity allows before costlier ones are used."
        )
        results = {"status": "optimal", "unsolved": False, "n_nodes": N, "n_edges": E,
                   "total_cost": _fin(total, 4), "edges": edges, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
