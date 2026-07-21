#!/usr/bin/env python3
"""Minimum Cost Flow — cheapest way to push a required flow through a network.
LP via scipy.optimize.linprog.

Input: node_names[], edge_from[], edge_to[], edge_capacity[], edge_cost[],
       required_flow, source(optional -> first), sink(optional -> last)
Output: results{status, unsolved, message, source, sink, required_flow,
                total_cost, edges:[{from,to,capacity,cost,flow}], interpretation}, plot
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
        ef = [str(x) for x in (p.get("edge_from") or [])]
        et = [str(x) for x in (p.get("edge_to") or [])]
        ecap = [float(x) for x in (p.get("edge_capacity") or [])]
        ecost = [float(x) for x in (p.get("edge_cost") or [])]
        req = float(p.get("required_flow"))
        N, E = len(nodes), len(ef)
        if N < 2 or E < 1:
            raise ValueError("Need at least 2 nodes and one edge.")
        if not (len(et) == len(ecap) == len(ecost) == E):
            raise ValueError("edge arrays must all have the same length.")
        idx = {n: i for i, n in enumerate(nodes)}
        src = p.get("source") or nodes[0]
        snk = p.get("sink") or nodes[-1]
        if src not in idx or snk not in idx or src == snk:
            raise ValueError("source and sink must be distinct valid nodes.")
        s, k = idx[src], idx[snk]

        # flow conservation: A_eq f = b, b[src]=+req (net out), b[sink]=-req, else 0
        A_eq = np.zeros((N, E)); b_eq = np.zeros(N)
        for e, (a, b) in enumerate(zip(ef, et)):
            A_eq[idx[a], e] += 1.0    # leaves a
            A_eq[idx[b], e] -= 1.0    # enters b
        b_eq[s] = req; b_eq[k] = -req
        res = linprog(np.array(ecost), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0, cap) for cap in ecap], method="highs")
        if not res.success:
            results = {"status": "infeasible", "unsolved": True,
                       "message": f"Cannot route {req:g} units from {src} to {snk} within capacities: {res.message}"}
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
            ax.set_title(f"Min-cost flow {src}→{snk}: {req:g} units, cost {total:,.2f}")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Routing the required {req:g} units from {src} to {snk} costs at minimum {total:,.2f}, "
            f"using {len(used)} of {E} edges. The solver balances low per-unit cost against capacity "
            "limits — cheap edges fill first, and dearer ones are used only when a bottleneck forces it."
        )
        results = {"status": "optimal", "unsolved": False, "source": src, "sink": snk,
                   "required_flow": _fin(req, 4), "total_cost": _fin(total, 4),
                   "edges": edges, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
