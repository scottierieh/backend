#!/usr/bin/env python3
"""Maximum Flow — most units routable from source to sink respecting capacities.
scipy.sparse.csgraph.maximum_flow (integer capacities).

Input: node_names[], edge_from[], edge_to[], edge_capacity[],
       source(optional -> first), sink(optional -> last)
Output: results{status, unsolved, message, source, sink, max_flow,
                edges:[{from,to,capacity,flow}], interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

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
        cap_lookup = {}
        for a, b, c in zip(ef, et, ec):
            if a in idx and b in idx:
                M[idx[a], idx[b]] += int(round(c * scale))
                cap_lookup[(idx[a], idx[b])] = c
        g = csr_matrix(M)
        res = maximum_flow(g, idx[src], idx[snk])
        flow_matrix = res.flow.toarray()
        max_flow = res.flow_value / scale

        edges = []
        for (a, b), cap in cap_lookup.items():
            f = flow_matrix[a, b] / scale
            edges.append({"from": nodes[a], "to": nodes[b],
                          "capacity": _fin(cap, 4), "flow": _fin(max(f, 0), 4)})

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
        )
        results = {"status": "optimal", "unsolved": False, "source": src, "sink": snk,
                   "max_flow": _fin(max_flow, 4), "edges": edges, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
