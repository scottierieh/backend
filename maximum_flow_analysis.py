#!/usr/bin/env python3
"""Maximum Flow — most units routable from source to sink respecting capacities.
scipy.sparse.csgraph.maximum_flow (integer capacities).

scipy needs integer capacities: when any capacity has decimals every capacity
is multiplied by 1000 and rounded to an integer before solving (flows are
divided back). The response says so: capacity_scale (1 or 1000) and
capacities_rounded (True when that rounding changed at least one capacity,
i.e. a capacity had more than three decimals).

Parallel arcs (several input arcs with the same from -> to) are solved as one
arc whose capacity is their SUM, and the pair's flow is then split back over
the input arcs in input order (each takes min(its capacity, what is left)), so
every input arc is reported with its own capacity and a flow <= that capacity.

Input: node_names[], edge_from[], edge_to[], edge_capacity[],
       source(optional -> first), sink(optional -> last)
Output: results{status, unsolved, message, source, sink, max_flow,
                capacity_scale, capacities_rounded, parallel_arcs,
                edges:[{from,to,capacity,flow}], interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


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
        ec = [float(x) for x in (p.get("edge_capacity") or [])]
        N = len(nodes)
        if N < 2 or not ef:
            raise ValueError("Need at least 2 nodes and one edge.")
        if not (len(ef) == len(et) == len(ec)):
            raise ValueError("edge_from / edge_to / edge_capacity lengths must match.")
        if any(c < 0 for c in ec):
            raise ValueError("Capacities must be non-negative.")
        idx = {n: i for i, n in enumerate(nodes)}
        src = p.get("source") or nodes[0]
        snk = p.get("sink") or nodes[-1]
        if src not in idx or snk not in idx or src == snk:
            raise ValueError("source and sink must be distinct valid nodes.")

        # scipy maximum_flow needs integer capacities
        scale = 1
        if any(abs(c - round(c)) > 1e-9 for c in ec):
            scale = 1000  # scale fractional capacities to integers
        M = np.zeros((N, N), dtype=np.int64)
        arcs = []            # (input position, a, b, original capacity, scaled int capacity)
        rounded = False
        for e, (a, b, c) in enumerate(zip(ef, et, ec)):
            if a in idx and b in idx:
                ci = int(round(c * scale))
                if abs(c * scale - ci) > 1e-9:
                    rounded = True
                M[idx[a], idx[b]] += ci     # parallel arcs: capacities add up
                arcs.append((e, idx[a], idx[b], c, ci))
        g = csr_matrix(M)
        res = maximum_flow(g, idx[src], idx[snk])
        flow_matrix = res.flow.toarray()
        max_flow = res.flow_value / scale

        # Split each pair's (net, non-negative) flow over its input arcs in input order.
        left = {}
        for _, a, b, _, _ in arcs:
            left.setdefault((a, b), max(int(flow_matrix[a, b]), 0))
        n_per_pair = {}
        for _, a, b, _, _ in arcs:
            n_per_pair[(a, b)] = n_per_pair.get((a, b), 0) + 1
        edges = []
        for _, a, b, cap, ci in arcs:
            take = min(ci, left[(a, b)])
            left[(a, b)] -= take
            edges.append({"from": nodes[a], "to": nodes[b],
                          "capacity": _fin(cap, 4), "flow": _fin(take / scale, 4)})
        parallel = sorted({f"{nodes[a]}→{nodes[b]}" for (a, b), n in n_per_pair.items() if n > 1})

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(7, len(edges) * 0.6), 4.5), dpi=120)
            ys = np.arange(len(edges))
            caps = [e["capacity"] for e in edges]
            flows = [e["flow"] for e in edges]
            labels = [f"{e['from']}→{e['to']}" for e in edges]
            ax.barh(ys, caps, color="#dbeafe", label="capacity")
            ax.barh(ys, flows, color="#2563eb", label="flow used")
            ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("units"); ax.set_title(f"Max flow {src} → {snk} = {max_flow:g}")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        saturated = sum(1 for e in edges if e["flow"] and e["capacity"] and abs(e["flow"] - e["capacity"]) < 1e-6)
        interpretation = (
            f"At most {max_flow:g} units can flow from {src} to {snk} through this network. "
            f"{saturated} edge(s) are saturated (flow = capacity) — these form the bottleneck "
            "(the minimum cut). Adding capacity anywhere else will not raise the maximum; only "
            "widening a saturated edge can."
            + (f" Parallel arcs ({', '.join(parallel)}) were solved as one arc with their summed capacity; "
               "the pair's flow is split over them in input order." if parallel else "")
            + (" Capacities with more than three decimals were rounded to 1/1000 because the max-flow "
               "solver needs integer capacities." if rounded else "")
        )
        results = {"status": "optimal", "unsolved": False, "source": src, "sink": snk,
                   "max_flow": _fin(max_flow, 4), "capacity_scale": scale,
                   "capacities_rounded": bool(rounded), "parallel_arcs": parallel,
                   "edges": edges, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
