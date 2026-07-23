#!/usr/bin/env python3
"""Shortest Path — least-weight route from a source to a sink node.
Dijkstra via scipy.sparse.csgraph (weights must be non-negative).

Input: node_names[], edge_from[], edge_to[], edge_weight[],
       source(name, optional -> first node), sink(name, optional -> last),
       directed(bool, default True)
Output: results{status, unsolved, message, source, sink, total_distance,
                path:[names], edges:[{from,to,weight}], interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

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
        ew = [float(x) for x in (p.get("edge_weight") or [])]
        directed = bool(p.get("directed", True))
        N = len(nodes)
        if N < 2 or not ef:
            raise ValueError("Need at least 2 nodes and one edge.")
        if not (len(ef) == len(et) == len(ew)):
            raise ValueError("edge_from / edge_to / edge_weight lengths must match.")
        if any(w < 0 for w in ew):
            raise ValueError("Dijkstra requires non-negative edge weights.")
        idx = {n: i for i, n in enumerate(nodes)}
        src = p.get("source") or nodes[0]
        snk = p.get("sink") or nodes[-1]
        if src not in idx or snk not in idx:
            raise ValueError("source/sink must be valid node names.")
        s, k = idx[src], idx[snk]

        M = np.zeros((N, N))
        for a, b, w in zip(ef, et, ew):
            if a in idx and b in idx:
                M[idx[a], idx[b]] = w
                if not directed:
                    M[idx[b], idx[a]] = w
        g = csr_matrix(M)
        dist, pred = shortest_path(g, method="D", directed=directed,
                                   indices=s, return_predecessors=True)
        if not np.isfinite(dist[k]):
            results = {"status": "unreachable", "unsolved": True,
                       "message": f"No path exists from {src} to {snk}."}
            print(json.dumps({"results": results, "plot": None}))
            return
        # reconstruct path
        path_idx = []
        cur = k
        while cur != -9999 and cur != s:
            path_idx.append(cur); cur = pred[cur]
            if len(path_idx) > N:
                break
        path_idx.append(s); path_idx.reverse()
        path = [nodes[i] for i in path_idx]
        edges = []
        for a, b in zip(path_idx[:-1], path_idx[1:]):
            edges.append({"from": nodes[a], "to": nodes[b], "weight": _fin(M[a, b], 4)})
        total = float(dist[k])

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
            xs = np.arange(len(path))
            ax.plot(xs, [0]*len(path), "-o", color="#2563eb", lw=2, markersize=10)
            for i, nm in enumerate(path):
                ax.annotate(nm, (i, 0), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
            for i, e in enumerate(edges):
                ax.annotate(f"{e['weight']:g}", ((i+0.5), 0), textcoords="offset points",
                            xytext=(0, -16), ha="center", fontsize=8, color="#dc2626")
            ax.set_title(f"Shortest path {src} → {snk}  (distance {total:g})")
            ax.axis("off")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The shortest route from {src} to {snk} is {' → '.join(path)}, with a total "
            f"distance of {total:g} across {len(edges)} edge(s). Dijkstra's algorithm proves no "
            "cheaper route exists given non-negative weights."
        )
        results = {"status": "optimal", "unsolved": False, "source": src, "sink": snk,
                   "total_distance": _fin(total, 4), "path": path, "edges": edges,
                   "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
