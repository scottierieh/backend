"""
routers/network.py

POST /api/network/shortestpath    — Dijkstra shortest path on spatial graph
POST /api/network/centrality      — Betweenness / closeness centrality
POST /api/network/community       — Louvain community detection
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import haversine, centroid, to_native

router = APIRouter()

CLUSTER_COLORS = [
    "#3b82f6","#ef4444","#22c55e","#a855f7","#f97316",
    "#ec4899","#14b8a6","#eab308","#6366f1","#84cc16",
]


class BaseMapRequest(BaseModel):
    data: List[Dict[str, Any]]


def _build_graph(pts: list, max_dist_m: float):
    """Build networkx graph from points within distance threshold."""
    import networkx as nx
    G = nx.Graph()
    for i, p in enumerate(pts):
        G.add_node(i, lat=p["lat"], lng=p["lng"], **{k: v for k, v in p.items() if k not in ("lat", "lng")})
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = haversine(pts[i]["lat"], pts[i]["lng"], pts[j]["lat"], pts[j]["lng"])
            if d <= max_dist_m:
                G.add_edge(i, j, weight=d)
    return G


# ══════════════════════════════════════════════════════════
# Shortest Path
# ══════════════════════════════════════════════════════════

class ShortestPathRequest(BaseMapRequest):
    sourceId: str         # point id
    targetId: str         # point id
    maxEdgeM: float = 5000   # max edge length to include


@router.post("/api/network/shortestpath")
def run_shortest_path(req: ShortestPathRequest):
    import networkx as nx

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    id_to_idx = {str(p.get("id", i)): i for i, p in enumerate(pts)}

    src = id_to_idx.get(req.sourceId)
    tgt = id_to_idx.get(req.targetId)
    if src is None or tgt is None:
        raise HTTPException(400, "Source or target id not found.")

    G = _build_graph(pts, req.maxEdgeM)

    if not nx.has_path(G, src, tgt):
        raise HTTPException(400, "No path found between these points.")

    path_nodes = nx.shortest_path(G, src, tgt, weight="weight")
    path_length = nx.shortest_path_length(G, src, tgt, weight="weight")

    path_points = [pts[n] for n in path_nodes]

    # Build edges for visualization
    edges = []
    for i in range(len(path_nodes) - 1):
        a, b = path_nodes[i], path_nodes[i + 1]
        edges.append({
            "from": pts[a],
            "to": pts[b],
            "distanceM": round(G[a][b]["weight"], 1),
        })

    return to_native({
        "results": {
            "path": path_points,
            "edges": edges,
            "totalDistanceM": round(path_length, 1),
            "totalDistanceKm": round(path_length / 1000, 3),
            "nHops": len(path_nodes) - 1,
            "source": pts[src],
            "target": pts[tgt],
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Network Centrality
# ══════════════════════════════════════════════════════════

class CentralityRequest(BaseMapRequest):
    maxEdgeM: float = 2000
    metric: str = "betweenness"   # betweenness | closeness | degree | eigenvector


@router.post("/api/network/centrality")
def run_centrality(req: CentralityRequest):
    import networkx as nx

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < 3:
        raise HTTPException(400, "Need at least 3 points.")

    G = _build_graph(pts, req.maxEdgeM)

    if req.metric == "betweenness":
        scores = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif req.metric == "closeness":
        scores = nx.closeness_centrality(G, distance="weight")
    elif req.metric == "degree":
        scores = nx.degree_centrality(G)
    elif req.metric == "eigenvector":
        try:
            scores = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            scores = nx.degree_centrality(G)
    else:
        raise HTTPException(400, f"Unknown metric: {req.metric}")

    score_vals = list(scores.values())
    min_s = min(score_vals) if score_vals else 0
    max_s = max(score_vals) if score_vals else 1
    s_range = max_s - min_s or 1

    results = []
    for i, p in enumerate(pts):
        raw = scores.get(i, 0.0)
        norm = (raw - min_s) / s_range
        size = 4 + norm * 16
        # color: high = red, low = blue
        r_c = int(59 + (239 - 59) * norm)
        g_c = int(130 - (130 - 68) * norm)
        b_c = int(246 - (246 - 68) * norm)
        results.append({
            "row": p,
            "score": round(raw, 6),
            "normalizedScore": round(norm, 4),
            "size": round(size, 1),
            "color": f"rgb({r_c},{g_c},{b_c})",
            "degree": G.degree(i),
        })

    results.sort(key=lambda x: -x["score"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    # Build edges for display
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "from": pts[u],
            "to": pts[v],
            "distanceM": round(d["weight"], 1),
        })

    return to_native({
        "results": {
            "points": results,
            "edges": edges,
            "metric": req.metric,
            "nNodes": G.number_of_nodes(),
            "nEdges": G.number_of_edges(),
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Community Detection (Louvain)
# ══════════════════════════════════════════════════════════

class CommunityRequest(BaseMapRequest):
    maxEdgeM: float = 2000
    resolution: float = 1.0   # Louvain resolution parameter


@router.post("/api/network/community")
def run_community(req: CommunityRequest):
    import networkx as nx
    try:
        from networkx.algorithms.community import louvain_communities
    except ImportError:
        raise HTTPException(500, "networkx >= 2.7 required for Louvain.")

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < 4:
        raise HTTPException(400, "Need at least 4 points.")

    G = _build_graph(pts, req.maxEdgeM)

    if G.number_of_edges() == 0:
        raise HTTPException(400, "No edges found — try increasing maxEdgeM.")

    communities = louvain_communities(G, weight="weight",
                                       resolution=req.resolution, seed=42)

    # Build node → community map
    node_community = {}
    for cid, community in enumerate(communities):
        for node in community:
            node_community[node] = cid

    # Build result clusters
    clusters = []
    for cid, community in enumerate(communities):
        members = [pts[n] for n in community]
        c = centroid(members)
        radius = max(
            haversine(c["lat"], c["lng"], p["lat"], p["lng"]) for p in members
        ) if len(members) > 1 else 0
        clusters.append({
            "id": cid,
            "color": CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            "center": c,
            "count": len(members),
            "radius": radius,
            "points": members,
            "visible": True,
        })

    # Edges with community info
    edges = []
    for u, v, d in G.edges(data=True):
        same_community = node_community.get(u) == node_community.get(v)
        edges.append({
            "from": pts[u],
            "to": pts[v],
            "distanceM": round(d["weight"], 1),
            "sameCommunity": same_community,
            "color": CLUSTER_COLORS[node_community.get(u, 0) % len(CLUSTER_COLORS)]
            if same_community else "#d1d5db",
        })

    modularity = nx.community.modularity(G, communities, weight="weight")

    return to_native({
        "results": {
            "clusters": clusters,
            "noise": [],
            "noiseColor": "#9ca3af",
            "showNoise": False,
            "nClusters": len(clusters),
            "nNoise": 0,
            "edges": edges,
            "modularity": round(modularity, 4),
            "nNodes": G.number_of_nodes(),
            "nEdges": G.number_of_edges(),
            "visible": True,
        }
    })
