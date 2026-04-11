"""
Social Network Analysis Router — FastAPI
Improved: bug fixes + new fields + consistent role logic + approx warnings
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import networkx as nx
except ImportError:
    raise ImportError("pip install networkx")

router = APIRouter()

# ─── Input / Output Models ────────────────────────────────────────────────────

class EdgeInput(BaseModel):
    source: str
    target: str
    weight: Optional[float] = 1.0
    timestamp: Optional[str] = None

class NetworkInput(BaseModel):
    edges: List[EdgeInput]
    directed: bool = True
    analysis_type: str   # overview | centrality | role | community | influence | dynamic
    params: Optional[Dict[str, Any]] = {}

class NodeResult(BaseModel):
    id: str
    value: Optional[float] = None
    role: Optional[str] = None
    community: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = {}

class AnalysisResult(BaseModel):
    analysis_type: str
    summary: Dict[str, Any]
    nodes: List[NodeResult]
    edges: Optional[List[Dict[str, Any]]] = []
    insights: List[Dict[str, str]]

# ─── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(edges: List[EdgeInput], directed: bool) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    for e in edges:
        G.add_edge(
            e.source, e.target,
            weight=e.weight or 1.0,
            timestamp=e.timestamp,
        )
    return G

# ─── Shared Role Classification (single source of truth) ─────────────────────
# FIX #5: was duplicated between analyze_role and graph_viz_get — now shared.

def compute_role_thresholds(G: nx.Graph, deg: dict, bet: dict, eig: dict, clo: dict) -> dict:
    nodes_list = list(G.nodes())
    if not nodes_list:
        return {}
    deg_vals = np.array([deg[n] for n in nodes_list])
    bet_vals = np.array([bet[n] for n in nodes_list])
    eig_vals = np.array([eig[n] for n in nodes_list])
    clo_vals = np.array([clo[n] for n in nodes_list])
    return {
        "deg_p75": float(np.percentile(deg_vals, 75)),
        "deg_p40": float(np.percentile(deg_vals, 40)),
        "bet_p75": float(np.percentile(bet_vals, 75)),
        "eig_p70": float(np.percentile(eig_vals, 70)),
        "clo_p40": float(np.percentile(clo_vals, 40)),
    }

def classify_node_role(node: str, G: nx.Graph, deg: dict, bet: dict, eig: dict, clo: dict, thresholds: dict) -> str:
    if G.degree(node) == 0:
        return "Isolate"
    d, b, e, c = deg[node], bet[node], eig[node], clo[node]
    if d >= thresholds["deg_p75"] and e >= thresholds["eig_p70"]:
        return "Hub"
    if b >= thresholds["bet_p75"] and d < thresholds["deg_p75"]:
        return "Bridge/Broker"
    if d < thresholds["deg_p40"] and c < thresholds["clo_p40"]:
        return "Peripheral"
    return "Regular"

def structural_hole_score(node: str, UG: nx.Graph, bet: dict) -> float:
    """Betweenness × (1 - local neighbourhood density) — proxy for structural hole."""
    neighbors = list(UG.neighbors(node))
    if len(neighbors) < 2:
        return 0.0
    sub = UG.subgraph(neighbors)
    local_density = nx.density(sub)
    return round(bet[node] * (1 - local_density), 4)

def compute_betweenness(G: nx.Graph) -> tuple[dict, str]:
    """Returns (betweenness_dict, method_label). Uses approximation for large graphs."""
    n = G.number_of_nodes()
    if n <= 300:
        return nx.betweenness_centrality(G, normalized=True, weight="weight"), "exact"
    k = min(200, n)
    return nx.betweenness_centrality(G, normalized=True, weight="weight", k=k), f"approx(k={k})"

# ─── Module 1: Network Overview ───────────────────────────────────────────────

def analyze_overview(G: nx.Graph) -> AnalysisResult:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    is_directed = G.is_directed()
    UG = G.to_undirected() if is_directed else G

    density = nx.density(G)

    if is_directed:
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    num_components = len(components)
    largest_cc_size = max(len(c) for c in components) if components else 0

    isolates = list(nx.isolates(G))
    avg_clustering = nx.average_clustering(UG)

    try:
        if is_directed:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
        sub = UG.subgraph(largest_cc).copy()
        avg_path = nx.average_shortest_path_length(sub)
        diameter = nx.diameter(sub)

        # ── Path length distribution histogram ──
        # Only compute for small/medium graphs (expensive for large)
        path_histogram: Dict[str, Any] = {}
        longest_pairs: List[Dict[str, Any]] = []
        path_concentration_index: Optional[float] = None

        sub_n = sub.number_of_nodes()
        if sub_n <= 500:
            path_lengths: List[int] = []
            # Collect all pairwise shortest path lengths
            for source in sub.nodes():
                lengths = nx.single_source_shortest_path_length(sub, source)
                for target, dist in lengths.items():
                    if target != source:
                        path_lengths.append(dist)

            if path_lengths:
                max_dist = max(path_lengths)
                # Histogram: count of pairs at each distance 1..diameter
                hist_counts = [0] * (max_dist + 1)
                for d in path_lengths:
                    hist_counts[d] += 1

                path_histogram = {
                    "distances": list(range(max_dist + 1)),
                    "counts":    hist_counts,
                    "total_pairs": len(path_lengths),
                }

                # Path concentration index: % of pairs at distance 1 or 2
                short_pairs = hist_counts[1] + (hist_counts[2] if len(hist_counts) > 2 else 0)
                path_concentration_index = round(short_pairs / len(path_lengths), 4) if path_lengths else None

                # Reachability ratio (within largest CC it's always 1.0, but across full graph)
                reachable_pairs = len(path_lengths)
                total_possible_pairs = n * (n - 1) if is_directed else n * (n - 1) // 2
                reachability_ratio = round(reachable_pairs / total_possible_pairs, 4) if total_possible_pairs > 0 else 1.0

                # Longest pairs — sample node pairs at diameter distance
                if diameter and diameter > 0:
                    found = []
                    for src in list(sub.nodes())[:50]:   # cap search
                        lengths_src = nx.single_source_shortest_path_length(sub, src)
                        for tgt, dist in lengths_src.items():
                            if dist == diameter and src != tgt:
                                found.append({"nodeA": str(src), "nodeB": str(tgt), "distance": dist})
                            if len(found) >= 5:
                                break
                        if len(found) >= 5:
                            break
                    longest_pairs = found
            else:
                reachability_ratio = 1.0
        else:
            # For large graphs: skip full distribution, just compute reachability
            reachable_pairs = sum(1 for _ in nx.connected_components(UG)) if not is_directed else None
            total_possible_pairs = n * (n - 1) if is_directed else n * (n - 1) // 2
            reachability_ratio = round(largest_cc_size * (largest_cc_size - 1) / max(total_possible_pairs, 1), 4)
            path_histogram = {}
            longest_pairs = []
            path_concentration_index = None

    except Exception:
        avg_path = None
        diameter = None
        path_histogram = {}
        longest_pairs = []
        reachability_ratio = None
        path_concentration_index = None

    degrees = dict(G.degree())
    avg_degree = float(np.mean(list(degrees.values()))) if degrees else 0

    # ── Per-component stats ──
    component_stats = []
    for i, comp in enumerate(sorted(components, key=len, reverse=True)):
        comp_sub = UG.subgraph(comp).copy()
        comp_size = len(comp)
        comp_edges = comp_sub.number_of_edges()
        comp_density = round(nx.density(comp_sub), 4)
        comp_degrees = dict(comp_sub.degree())
        top_node = max(comp_degrees, key=lambda x: comp_degrees[x]) if comp_degrees else None
        component_stats.append({
            "id":             i + 1,
            "size":           comp_size,
            "internal_edges": comp_edges,
            "density":        comp_density,
            "top_node":       str(top_node) if top_node else None,
        })

    # Fragmentation index: 1 - (largest_cc_size / n)^2  (Herfindahl-style)
    frag_index = round(1 - (largest_cc_size / n) ** 2, 4) if n > 0 else 0
    connected_coverage = round(largest_cc_size / n * 100, 1) if n > 0 else 0
    isolate_ratio = round(len(isolates) / n * 100, 1) if n > 0 else 0

    # Degree distribution histogram (10 buckets) — NEW
    deg_vals = np.array(list(degrees.values()), dtype=float)
    if len(deg_vals) > 0:
        counts, bin_edges = np.histogram(deg_vals, bins=min(10, n))
        degree_histogram = {
            "counts": counts.tolist(),
            "bins":   [round(float(b), 2) for b in bin_edges.tolist()],
        }
    else:
        degree_histogram = {"counts": [], "bins": []}

    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortativity = None

    reciprocity = nx.reciprocity(G) if is_directed else None

    # ── Mutual ties & degree pairing (for Assortativity/Reciprocity page) ──
    mutual_edges: List[Dict[str, Any]] = []
    one_way_edges: int = 0
    degree_pairs: List[Dict[str, Any]] = []

    if is_directed:
        for u, v, d in list(G.edges(data=True))[:200]:   # cap for payload
            has_reverse = G.has_edge(v, u)
            mutual_edges.append({
                "source": str(u), "target": str(v),
                "weight": round(float(d.get("weight", 1)), 3),
                "mutual": has_reverse,
            })
        total_edges_d = G.number_of_edges()
        mutual_count  = sum(1 for u, v in G.edges() if G.has_edge(v, u)) // 2
        one_way_count = total_edges_d - mutual_count * 2
        mutual_tie_ratio    = round(mutual_count * 2 / total_edges_d, 4) if total_edges_d > 0 else 0
        asymmetry_ratio     = round(one_way_count / total_edges_d, 4)   if total_edges_d > 0 else 0
    else:
        mutual_tie_ratio  = 1.0
        asymmetry_ratio   = 0.0
        mutual_count      = m
        one_way_count     = 0

    # Degree pairing: sample edges and record (src_degree, tgt_degree)
    deg_in  = dict(G.in_degree())  if is_directed else {}
    deg_out = dict(G.out_degree()) if is_directed else {}
    for u, v in list(G.edges())[:300]:
        src_deg = degrees.get(u, 0)
        tgt_deg = degrees.get(v, 0)
        degree_pairs.append({"src_deg": src_deg, "tgt_deg": tgt_deg})

    # High-high tendency: % of edges where both nodes are above avg degree
    avg_deg_val = avg_degree
    high_high = sum(1 for p in degree_pairs if p["src_deg"] > avg_deg_val and p["tgt_deg"] > avg_deg_val)
    hub_peripheral = sum(1 for p in degree_pairs if (p["src_deg"] > avg_deg_val * 2 and p["tgt_deg"] < avg_deg_val * 0.5)
                                                   or (p["tgt_deg"] > avg_deg_val * 2 and p["src_deg"] < avg_deg_val * 0.5))
    high_high_ratio    = round(high_high    / len(degree_pairs), 4) if degree_pairs else 0
    hub_peripheral_ratio = round(hub_peripheral / len(degree_pairs), 4) if degree_pairs else 0

    summary = {
        "node_count":               n,
        "edge_count":               m,
        "density":                  round(density, 4),
        "avg_degree":               round(avg_degree, 2),
        "avg_clustering":           round(avg_clustering, 4),
        "avg_path_length":          round(avg_path, 3) if avg_path is not None else None,
        "diameter":                 diameter,
        "num_components":           num_components,
        "largest_cc":               largest_cc_size,
        "num_isolates":             len(isolates),
        "isolates":                 isolates[:10],
        "isolate_ratio":            isolate_ratio,
        "fragmentation_index":      frag_index,
        "connected_coverage":       connected_coverage,
        "component_stats":          component_stats[:20],
        "assortativity":            round(assortativity, 4) if assortativity is not None else None,
        "reciprocity":              round(reciprocity, 4) if reciprocity is not None else None,
        "mutual_tie_ratio":         mutual_tie_ratio,
        "asymmetry_ratio":          asymmetry_ratio,
        "mutual_count":             mutual_count,
        "one_way_count":            one_way_count,
        "high_high_ratio":          high_high_ratio,
        "hub_peripheral_ratio":     hub_peripheral_ratio,
        "mutual_edges":             mutual_edges[:100],
        "degree_pairs":             degree_pairs[:200],
        "is_directed":              is_directed,
        "degree_histogram":         degree_histogram,
        "path_histogram":           path_histogram,
        "longest_pairs":            longest_pairs,
        "reachability_ratio":       reachability_ratio,
        "path_concentration_index": path_concentration_index,
    }

    # All nodes sorted by degree (no cap — show all)
    nodes = [
        NodeResult(
            id=node,
            value=float(deg),
            metadata={"degree": deg},
        )
        for node, deg in sorted(degrees.items(), key=lambda x: -x[1])
    ]

    insights: List[Dict[str, str]] = []

    # ── Density ──
    possible = n * (n - 1) / 2 if n > 1 else 1
    actual_pct = round(density * 100, 1)
    if density < 0.1:
        insights.append({"title": "Network Density", "text": 
            f"Density measures what fraction of all possible connections actually exist. "
            f"Density {density:.3f} means only {actual_pct}% of possible connections are active — "
            f"out of {int(possible)} possible links, only {m} exist. "
            f"Most nodes rely on a small number of well-connected hubs to pass information along. "
            f"If a key hub goes down, large parts of the network lose contact with each other."
        })
    elif density > 0.5:
        insights.append({"title": "Network Density", "text": 
            f"Density measures what fraction of all possible connections actually exist. "
            f"Density {density:.3f} means {actual_pct}% of all possible connections are active — "
            f"a tightly woven network where almost every node can reach every other in just one or two steps. "
            f"Information or changes spread quickly and broadly across the entire network."
        })
    else:
        insights.append({"title": "Network Density", "text": 
            f"Density measures what fraction of all possible connections actually exist. "
            f"Density {density:.3f} means {actual_pct}% of possible connections are active — "
            f"a balanced structure where nodes are reasonably well connected without being overcrowded."
        })

    # ── Components ──
    if num_components > 1:
        isolated_pct = round((1 - largest_cc_size / n) * 100, 1) if n > 0 else 0
        insights.append({"title": "Disconnected Components", "text": 
            f"The network is split into {num_components} disconnected groups. "
            f"The largest group contains {largest_cc_size} out of {n} nodes ({100 - isolated_pct:.0f}%). "
            f"The remaining {isolated_pct:.0f}% of nodes ({n - largest_cc_size} nodes) are in separate islands "
            f"with no path to the main group — they cannot exchange information with it at all."
        })

    # ── Isolates ──
    if isolates:
        insights.append({"title": "Isolated Nodes", "text": 
            f"{len(isolates)} node(s) have zero connections: {', '.join(isolates[:3])}{'...' if len(isolates) > 3 else ''}. "
            f"These nodes are completely cut off — nothing flows to or from them. "
            f"Check whether this is intentional or a data issue."
        })

    # ── Clustering ──
    if avg_clustering > 0.5:
        insights.append({"title": "Clustering Coefficient", "text": 
            f"Clustering coefficient ({avg_clustering:.3f}) measures how often two neighbours of a node "
            f"are also connected to each other. "
            f"At {avg_clustering:.3f}, roughly {avg_clustering*100:.0f}% of neighbour pairs are also directly linked — "
            f"the network forms tight local clusters. Information tends to circulate within groups "
            f"rather than spreading across the whole network."
        })
    elif avg_clustering > 0.2:
        insights.append({"title": "Clustering Coefficient", "text": 
            f"Clustering coefficient {avg_clustering:.3f} — about {avg_clustering*100:.0f}% of neighbour pairs "
            f"are also directly connected to each other. "
            f"Moderate local grouping: some cluster structure exists but the network is not heavily siloed."
        })

    # ── Assortativity ──
    if assortativity is not None:
        if assortativity > 0.3:
            insights.append({"title": "Assortativity", "text": 
                f"Assortativity {assortativity:.3f} — highly connected nodes tend to connect with other "
                f"highly connected nodes. The network has a dense, well-connected core surrounded by a sparser periphery. "
                f"This makes the core resilient, but peripheral nodes depend entirely on their few links to reach the core."
            })
        elif assortativity < -0.3:
            insights.append({"title": "Assortativity", "text": 
                f"Assortativity {assortativity:.3f} — highly connected nodes tend to connect with weakly connected ones. "
                f"This creates a hub-and-spoke structure: a few hubs hold the network together. "
                f"Removing even one major hub can rapidly fragment the network."
            })

    # ── Diameter ──
    if diameter and diameter > 6:
        insights.append({"title": "Network Diameter", "text": 
            f"Diameter {diameter} — the two furthest nodes in this network are {diameter} steps apart. "
            f"In practical terms, some nodes require passing through {diameter} intermediaries to reach each other. "
            f"This creates slow information paths and potential bottlenecks at key intermediary nodes."
        })

    # ── Degree concentration ──
    top3_deg = sorted(degrees.values(), reverse=True)[:3]
    top3_sum = sum(top3_deg)
    total_deg = sum(degrees.values())
    if total_deg > 0 and n > 5:
        top3_pct = top3_sum / total_deg * 100
        if top3_pct > 50:
            top3_nodes = [nd for nd, _ in sorted(degrees.items(), key=lambda x: -x[1])[:3]]
            insights.append({"title": "Degree Concentration", "text": 
                f"The top 3 nodes — {', '.join(top3_nodes)} — hold {top3_pct:.0f}% of all connections in the network. "
                f"More than half of all activity flows through just these 3 nodes. "
                f"This level of concentration creates a critical dependency risk: "
                f"if these nodes become unavailable, the rest of the network is severely impaired."
            })

    return AnalysisResult(
        analysis_type="overview",
        summary=summary,
        nodes=nodes,
        insights=insights,
    )

# ─── Module 2: Centrality Analysis ───────────────────────────────────────────

def analyze_centrality(G: nx.Graph, params: dict) -> AnalysisResult:
    UG = G.to_undirected() if G.is_directed() else G

    deg = nx.degree_centrality(G)
    bet, bet_method = compute_betweenness(G)   # FIX #3: shared helper, returns method label
    clo = nx.closeness_centrality(G)
    pr  = nx.pagerank(G, weight="weight")

    try:
        eig = nx.eigenvector_centrality(UG, max_iter=1000, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    top_k = 5   # Increased from 3 to 5 for richer summary
    top_degree      = sorted(G.nodes(), key=lambda n: deg[n],  reverse=True)[:top_k]
    top_betweenness = sorted(G.nodes(), key=lambda n: bet[n],  reverse=True)[:top_k]
    top_closeness   = sorted(G.nodes(), key=lambda n: clo[n],  reverse=True)[:top_k]
    top_pagerank    = sorted(G.nodes(), key=lambda n: pr[n],   reverse=True)[:top_k]
    top_eigenvector = sorted(G.nodes(), key=lambda n: eig[n],  reverse=True)[:top_k]

    # NEW: centrality correlation matrix (deg vs bet vs clo vs pr)
    nodes_list = list(G.nodes())
    if len(nodes_list) >= 4:
        mat = np.array([[deg[n], bet[n], clo[n], pr[n]] for n in nodes_list])
        try:
            corr = np.corrcoef(mat.T)
            centrality_correlation = {
                "deg_bet": round(float(corr[0, 1]), 3),
                "deg_clo": round(float(corr[0, 2]), 3),
                "deg_pr":  round(float(corr[0, 3]), 3),
                "bet_clo": round(float(corr[1, 2]), 3),
            }
        except Exception:
            centrality_correlation = {}
    else:
        centrality_correlation = {}

    nodes = [
        NodeResult(
            id=node,
            value=round(deg[node], 4),
            metadata={
                "degree":      round(deg[node],  4),
                "betweenness": round(bet[node],  4),
                "closeness":   round(clo[node],  4),
                "eigenvector": round(eig[node],  4),
                "pagerank":    round(pr[node],   4),
            },
        )
        for node in sorted(G.nodes(), key=lambda n: -deg[n])
    ]

    summary = {
        "top_degree":              {n: round(deg[n],  4) for n in top_degree},
        "top_betweenness":         {n: round(bet[n],  4) for n in top_betweenness},
        "top_closeness":           {n: round(clo[n],  4) for n in top_closeness},
        "top_pagerank":            {n: round(pr[n],   4) for n in top_pagerank},
        "top_eigenvector":         {n: round(eig[n],  4) for n in top_eigenvector},
        "betweenness_method":      bet_method,
        "centrality_correlation":  centrality_correlation,
        # ── Degree detail ──
        "avg_degree_centrality":   round(float(np.mean(list(deg.values()))), 4),
        "degree_skewness":         round(float(
            np.mean([(v - np.mean(list(deg.values())))**3 for v in deg.values()]) /
            max(np.std(list(deg.values()))**3, 1e-9)
        ), 4) if len(deg) > 2 else 0,
        "top10_concentration":     round(
            sum(sorted(deg.values(), reverse=True)[:10]) / max(sum(deg.values()), 1e-9), 4
        ),
        "in_degree":  {n: round(dict(G.in_degree())[n]  / max(G.number_of_nodes() - 1, 1), 4)
                       for n in top_degree} if G.is_directed() else {},
        "out_degree": {n: round(dict(G.out_degree())[n] / max(G.number_of_nodes() - 1, 1), 4)
                       for n in top_degree} if G.is_directed() else {},
        "in_degree_leader":  str(max(G.nodes(), key=lambda n: G.in_degree(n)))  if G.is_directed() and G.nodes() else None,
        "out_degree_leader": str(max(G.nodes(), key=lambda n: G.out_degree(n))) if G.is_directed() and G.nodes() else None,
    }

    insights: List[Dict[str, str]] = []

    # ── Approx warning ──
    if bet_method != "exact":
        insights.append({"title": "Approximation Warning", "text": 
            f"This network has {G.number_of_nodes()} nodes, so betweenness centrality is estimated "
            f"using a random sample of paths ({bet_method}). "
            f"The relative rankings are reliable — who is higher or lower — "
            f"but the exact decimal values are approximations."
        })

    # ── Degree ──
    top_deg_node = top_degree[0]
    top_deg_val  = deg[top_deg_node]
    insights.append({"title": "Degree Centrality", "text": 
        f"Degree centrality measures how directly connected each node is. "
        f"'{top_deg_node}' has the highest degree centrality at {top_deg_val:.3f}, "
        f"meaning it is directly linked to {top_deg_val*100:.0f}% of all other nodes — "
        f"1 in every {round(1/top_deg_val) if top_deg_val > 0 else '?'} nodes connects directly to it. "
        f"Top 3 by direct connections: {', '.join(top_degree[:3])}."
    })

    # ── Betweenness ──
    top_bet_node = top_betweenness[0]
    top_bet_val  = bet[top_bet_node]
    insights.append({"title": "Betweenness Centrality", "text": 
        f"Betweenness centrality measures how often a node sits on the shortest path between others — "
        f"essentially, which node every route has to pass through. "
        f"'{top_bet_node}' scores {top_bet_val:.3f}, meaning {top_bet_val*100:.0f}% of all paths in the network "
        f"run through this single node. "
        f"Remove it, and {top_bet_val*100:.0f}% of routes lose their most direct connection. "
        f"Top 3 flow controllers: {', '.join(top_betweenness[:3])}."
    })

    # ── Degree vs Betweenness split ──
    if set(top_degree[:3]) != set(top_betweenness[:3]):
        overlap = set(top_degree[:3]) & set(top_betweenness[:3])
        if overlap:
            insights.append({"title": "Dominant Nodes", "text": 
                f"{', '.join(overlap)} rank highly on both direct connections and flow control — "
                f"the most strategically powerful nodes in this network. "
                f"They are both widely connected and sit on critical paths."
            })
        else:
            insights.append({"title": "Degree vs Betweenness Split", "text": 
                f"The most-connected nodes and the flow-controlling nodes are entirely different. "
                f"Most connections: {', '.join(top_degree[:2])}. "
                f"Flow controllers: {', '.join(top_betweenness[:2])}. "
                f"This means there are hidden bottlenecks — nodes that control the flow "
                f"without appearing prominent by connection count alone."
            })

    # ── PageRank ──
    pr_top     = top_pagerank[0]
    pr_top_val = pr[pr_top]
    insights.append({"title": "PageRank", "text": 
        f"PageRank measures influence by the quality of incoming connections, not just quantity — "
        f"being chosen by an already-influential node counts for more. "
        f"'{pr_top}' leads with a PageRank of {pr_top_val:.4f}, "
        f"meaning it receives endorsement from the most influential sources in the network. "
        f"This is the node best positioned to propagate influence through high-credibility channels."
    })

    # ── Correlation ──
    if centrality_correlation:
        dc = centrality_correlation.get("deg_bet", 0)
        if abs(dc) < 0.4:
            insights.append({"title": "Hidden Brokers Detected", "text": 
                f"Degree–betweenness correlation is low ({dc:.2f}). "
                f"In this network, having many connections does NOT guarantee controlling the flow. "
                f"There are hidden brokers — nodes that quietly control critical paths "
                f"despite appearing unremarkable by connection count. "
                f"Betweenness rankings reveal these hidden power nodes."
            })
        elif dc > 0.8:
            insights.append({"title": "Centrality Correlation", "text": 
                f"Degree–betweenness correlation is high ({dc:.2f}). "
                f"In this network, the most connected nodes are also the most critical flow controllers. "
                f"Connection count is a reliable indicator of strategic importance."
            })

    # ── Low-centrality nodes ──
    low_all = [
        nd for nd in G.nodes()
        if deg[nd] < 0.05 and bet[nd] < 0.01 and clo[nd] < 0.2
    ]
    if low_all:
        insights.append({"title": "Marginal Nodes", "text": 
            f"{len(low_all)} node(s) score low across all centrality measures: "
            f"{', '.join(low_all[:3])}{'...' if len(low_all) > 3 else ''}. "
            f"These nodes have few connections, control no paths, and are far from the network centre. "
            f"They are effectively on the margins — receiving information last and influencing little."
        })

    return AnalysisResult(
        analysis_type="centrality",
        summary=summary,
        nodes=nodes,
        insights=insights,
    )

# ─── Module 3: Role & Position ────────────────────────────────────────────────

def analyze_role(G: nx.Graph) -> AnalysisResult:
    UG = G.to_undirected() if G.is_directed() else G

    deg = nx.degree_centrality(G)
    bet, bet_method = compute_betweenness(G)   # FIX #5: uses shared helper
    clo = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(UG, max_iter=1000, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    nodes_list = list(G.nodes())
    if not nodes_list:
        raise ValueError("Graph has no nodes.")

    # FIX #5: use shared threshold + classification functions
    thresholds = compute_role_thresholds(G, deg, bet, eig, clo)

    role_counts: Dict[str, int] = {}
    nodes_out: List[NodeResult] = []

    for node in nodes_list:
        role   = classify_node_role(node, G, deg, bet, eig, clo, thresholds)
        sh     = structural_hole_score(node, UG, bet)
        role_counts[role] = role_counts.get(role, 0) + 1

        nodes_out.append(NodeResult(
            id=node,
            role=role,
            value=round(bet[node], 4),
            metadata={
                "role":              role,
                "degree":            round(deg[node], 4),
                "betweenness":       round(bet[node], 4),
                "closeness":         round(clo[node], 4),
                "eigenvector":       round(eig[node], 4),
                "structural_hole":   sh,
                "actual_degree":     G.degree(node),
            },
        ))

    role_order = {"Hub": 0, "Bridge/Broker": 1, "Regular": 2, "Peripheral": 3, "Isolate": 4}
    nodes_out.sort(key=lambda x: (role_order.get(x.role or "", 5), -(x.value or 0)))

    hubs        = [n.id for n in nodes_out if n.role == "Hub"]
    bridges     = [n.id for n in nodes_out if n.role == "Bridge/Broker"]
    peripherals = [n.id for n in nodes_out if n.role == "Peripheral"]
    isolates    = [n.id for n in nodes_out if n.role == "Isolate"]

    top_sh = sorted(nodes_out, key=lambda x: x.metadata.get("structural_hole", 0), reverse=True)[:5]

    # NEW: role classification criteria exposed to frontend
    summary = {
        "role_distribution":    role_counts,
        "hubs":                 hubs[:10],
        "bridges":              bridges[:10],
        "peripherals":          peripherals[:10],
        "isolates":             isolates[:10],
        "top_structural_holes": [{"node": n.id, "score": n.metadata["structural_hole"]} for n in top_sh],
        "node_count":           len(nodes_list),
        "betweenness_method":   bet_method,   # NEW
        "thresholds": {          # NEW — expose percentile thresholds used
            "hub_min_degree":      round(thresholds["deg_p75"], 4),
            "hub_min_eigenvector": round(thresholds["eig_p70"], 4),
            "broker_min_bet":      round(thresholds["bet_p75"], 4),
            "peripheral_max_deg":  round(thresholds["deg_p40"], 4),
        },
    }

    insights: List[Dict[str, str]] = []

    if bet_method != "exact":
        insights.append({"title": "Approximation Warning", "text": 
            f"Betweenness is estimated ({bet_method}) for this {len(nodes_list)}-node network. "
            f"Bridge/Broker classifications are based on these estimates — treat rankings as directionally correct."
        })

    if hubs:
        top_hub = hubs[0]
        top_hub_node = next(n for n in nodes_out if n.id == top_hub)
        insights.append({"title": "Hub Nodes", "text": 
            f"A Hub node is broadly connected AND connected to other well-connected nodes — "
            f"the coordinators at the centre of activity. "
            f"'{top_hub}' leads as the top hub with degree {top_hub_node.metadata['degree']:.3f} "
            f"(directly linked to {top_hub_node.metadata['degree']*100:.0f}% of all nodes) "
            f"and eigenvector {top_hub_node.metadata['eigenvector']:.3f}. "
            f"{len(hubs)} hub node(s) total: {', '.join(hubs[:3])}{'...' if len(hubs) > 3 else ''}."
        })

    if bridges:
        top_br = bridges[0]
        top_br_node = next(n for n in nodes_out if n.id == top_br)
        bet_pct = top_br_node.metadata['betweenness'] * 100
        insights.append({"title": "Bridge / Broker Nodes", "text": 
            f"A Bridge/Broker node sits between groups, controlling the flow between them "
            f"— often with surprisingly few direct connections. "
            f"'{top_br}' has betweenness {top_br_node.metadata['betweenness']:.3f}, "
            f"meaning {bet_pct:.0f}% of all paths pass through it, "
            f"yet it has only {top_br_node.metadata['actual_degree']} direct connections. "
            f"Remove this node and those groups lose their link to each other. "
            f"{len(bridges)} broker node(s) total: {', '.join(bridges[:3])}{'...' if len(bridges) > 3 else ''}."
        })

    if peripherals:
        peri_pct = round(len(peripherals) / len(nodes_list) * 100, 0)
        insights.append({"title": "Peripheral Nodes", "text": 
            f"Peripheral nodes have few connections and sit far from the network centre — "
            f"they are the last to receive information and the first to be cut off. "
            f"{len(peripherals)} peripheral node(s) found ({peri_pct:.0f}% of the network): "
            f"{', '.join(peripherals[:3])}{'...' if len(peripherals) > 3 else ''}."
        })

    if isolates:
        insights.append({"title": "Isolated Nodes", "text": 
            f"{len(isolates)} node(s) have zero connections: "
            f"{', '.join(isolates[:3])}{'...' if len(isolates) > 3 else ''}. "
            f"Nothing flows to or from these nodes — they are completely outside the network. "
            f"Verify whether this is intentional or a data entry issue."
        })

    if not bridges and not isolates:
        insights.append({"title": "Distributed Connectivity", "text": 
            f"No bridge or isolate nodes detected. "
            f"Connectivity is distributed across the network — "
            f"no single node removal would cause the network to fragment."
        })

    if top_sh and top_sh[0].metadata.get("structural_hole", 0) > 0.1:
        sh_node  = top_sh[0].id
        sh_score = top_sh[0].metadata['structural_hole']
        insights.append({"title": "Structural Hole Nodes", "text": 
            f"A structural hole node's neighbours do not know each other — "
            f"it is the only bridge between otherwise disconnected groups. "
            f"'{sh_node}' has the highest structural hole score ({sh_score:.4f}): "
            f"its neighbours are largely unconnected to one another, making it "
            f"the sole gateway between those groups and giving it exclusive access to information from all sides."
        })

    return AnalysisResult(
        analysis_type="role",
        summary=summary,
        nodes=nodes_out,
        insights=insights,
    )


# ─── Module 4: Community Analysis ────────────────────────────────────────────

def analyze_community(G: nx.Graph, params: dict) -> AnalysisResult:
    from networkx.algorithms import community as nx_comm

    algorithm = params.get("algorithm", "louvain")
    _n_comm = G.number_of_nodes()
    if algorithm == "girvan_newman" and _n_comm > 150:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Girvan-Newman cannot run on {_n_comm} nodes (limit: 150). "
                "Use 'louvain' or 'label_propagation' instead."
            ),
        )
    UG = G.to_undirected() if G.is_directed() else G

    try:
        if algorithm == "louvain":
            communities = list(nx_comm.louvain_communities(UG, seed=42))
        elif algorithm == "label_propagation":
            communities = list(nx_comm.label_propagation_communities(UG))
        elif algorithm == "girvan_newman":
            comp = nx_comm.girvan_newman(UG)
            communities = [set(c) for c in next(iter(comp))]
        else:
            communities = list(nx_comm.louvain_communities(UG, seed=42))
    except Exception:
        if G.is_directed():
            communities = [set(c) for c in nx.weakly_connected_components(G)]
        else:
            communities = [set(c) for c in nx.connected_components(G)]

    try:
        modularity = nx_comm.modularity(UG, communities)
    except Exception:
        modularity = None

    node_comm: Dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_comm[node] = idx

    # FIX #1: external_edges — O(E) single pass, correct for directed graphs
    comm_stats = []
    # Build comm sets for O(1) lookup
    comm_sets = [set(c) for c in communities]

    # Pre-count external edges per community via single pass over all edges
    ext_edge_counts = [0] * len(communities)
    for u, v in G.edges():
        cu = node_comm.get(u, -1)
        cv = node_comm.get(v, -1)
        if cu != cv:
            if cu >= 0:
                ext_edge_counts[cu] += 1
            if cv >= 0 and not G.is_directed():
                ext_edge_counts[cv] += 1

    for idx, comm in enumerate(communities):
        sub = UG.subgraph(comm)
        deg_in = dict(sub.degree())
        top_node = max(deg_in, key=deg_in.get) if deg_in else None
        internal_edges = sub.number_of_edges()
        external_edges = ext_edge_counts[idx]

        # NEW: inter-community density
        possible_external = len(comm) * (_n_comm - len(comm))
        inter_density = round(external_edges / possible_external, 4) if possible_external > 0 else 0.0

        # NEW: cohesion score = internal_density / (internal_density + inter_density + ε)
        internal_density_val = nx.density(sub)
        cohesion = round(
            internal_density_val / (internal_density_val + inter_density + 1e-9), 4
        )

        comm_stats.append({
            "id":               idx,
            "size":             len(comm),
            "top_node":         top_node,
            "internal_edges":   internal_edges,
            "external_edges":   external_edges,   # FIX #1: now correct
            "density":          round(internal_density_val, 4),
            "inter_density":    inter_density,    # NEW
            "cohesion":         cohesion,         # NEW
            "nodes":            list(comm)[:20],
        })

    # Inter-community edges
    inter_edges: List[Dict[str, Any]] = []
    for u, v, data in G.edges(data=True):
        cu, cv = node_comm.get(u, -1), node_comm.get(v, -1)
        if cu != cv:
            inter_edges.append({
                "source":         u,
                "target":         v,
                "from_community": cu,
                "to_community":   cv,
                "weight":         data.get("weight", 1),
            })

    # Bridge nodes
    node_comm_neighbors: Dict[str, set] = {}
    for u, v in UG.edges():
        cu, cv = node_comm.get(u, -1), node_comm.get(v, -1)
        if cu != cv:
            node_comm_neighbors.setdefault(u, set()).add(cv)
            node_comm_neighbors.setdefault(v, set()).add(cu)
    bridge_nodes = [
        {"node": n, "connects": len(comms), "communities": list(comms)}
        for n, comms in node_comm_neighbors.items()
        if len(comms) >= 2
    ]
    bridge_nodes.sort(key=lambda x: -x["connects"])

    nodes_out = [
        NodeResult(
            id=node,
            community=node_comm.get(node, -1),
            value=float(node_comm.get(node, -1)),
            metadata={
                "community":      node_comm.get(node, -1),
                "community_size": len(communities[node_comm[node]]) if node in node_comm else 0,
            },
        )
        for node in G.nodes()
    ]

    # ── Community pair matrix & inter-edge details ──
    n_comm_ids = len(communities)
    pair_counts: Dict[str, int]   = {}
    pair_weights: Dict[str, float] = {}
    for e in inter_edges:
        key = f"{min(e['from_community'], e['to_community'])}_{max(e['from_community'], e['to_community'])}"
        pair_counts[key]  = pair_counts.get(key, 0) + 1
        pair_weights[key] = pair_weights.get(key, 0) + float(e['weight'])

    community_pairs = [
        {
            "from_comm":   int(k.split("_")[0]),
            "to_comm":     int(k.split("_")[1]),
            "edge_count":  v,
            "total_weight": round(pair_weights.get(k, 0), 3),
        }
        for k, v in sorted(pair_counts.items(), key=lambda x: -x[1])
    ]

    # Full flow matrix (for heatmap)
    comm_ids = sorted(set(node_comm.values()))
    flow_matrix: List[List[int]] = [
        [pair_counts.get(f"{min(r,c)}_{max(r,c)}", 0) for c in comm_ids]
        for r in comm_ids
    ]

    summary = {
        "num_communities":  len(communities),
        "modularity":       round(modularity, 4) if modularity is not None else None,
        "algorithm":        algorithm,
        "communities":      comm_stats,
        "inter_edges":      len(inter_edges),
        "inter_edges_list": inter_edges[:100],     # NEW — per-edge detail
        "community_pairs":  community_pairs[:20],  # NEW — sorted pair counts
        "flow_matrix":      flow_matrix,           # NEW — for heatmap
        "comm_ids":         comm_ids,              # NEW — community id list
        "bridge_nodes":     bridge_nodes[:10],
        "node_count":       G.number_of_nodes(),
    }

    insights: List[Dict[str, str]] = []
    n_comm = len(communities)

    # ── What is community detection ──
    insights.append({"title": "What is Community Detection", "text": 
        f"Community detection groups nodes that are more densely connected to each other "
        f"than to the rest of the network — like finding natural clusters or sub-groups. "
        f"{n_comm} distinct group(s) were identified using the {algorithm} algorithm"
        + (f" (modularity = {modularity:.3f})." if modularity is not None else ".")
    })

    # ── Modularity ──
    if modularity is not None:
        if modularity > 0.5:
            insights.append({"title": "Modularity Score", "text": 
                f"Modularity {modularity:.3f} — the group boundaries are strong. "
                f"Nodes within each group are tightly connected, while connections between groups are sparse. "
                f"Information flows easily within groups but struggles to cross between them. "
                f"The bridge nodes connecting groups become disproportionately critical."
            })
        elif modularity > 0.3:
            insights.append({"title": "Modularity Score", "text": 
                f"Modularity {modularity:.3f} — groups are visible but boundaries are permeable. "
                f"There is meaningful clustering, but cross-group connections are common enough "
                f"that information can still travel between groups without relying on a single bridge."
            })
        else:
            insights.append({"title": "Modularity Score", "text": 
                f"Modularity {modularity:.3f} — weak group structure. "
                f"The network does not naturally divide into distinct clusters. "
                f"Connections are fairly evenly distributed, so information spreads "
                f"across the whole network without strong barriers."
            })

    # ── Largest community ──
    largest  = max(comm_stats, key=lambda c: c["size"])
    smallest = min(comm_stats, key=lambda c: c["size"])
    largest_pct = round(largest['size'] / G.number_of_nodes() * 100, 0) if G.number_of_nodes() > 0 else 0
    insights.append({"title": "Largest Community", "text": 
        f"The largest group (C{largest['id']}) contains {largest['size']} nodes "
        f"({largest_pct:.0f}% of the network), centred around '{largest['top_node']}'. "
        f"Its internal density is {largest['density']:.3f} and cohesion score is {largest['cohesion']:.2f} — "
        f"{'a tightly bound cluster that rarely exchanges with outsiders' if largest['cohesion'] > 0.6 else 'moderately cohesive with regular cross-group exchange'}."
    })

    # ── Size imbalance ──
    if len(comm_stats) > 1:
        size_ratio = largest["size"] / max(smallest["size"], 1)
        if size_ratio > 3:
            insights.append({"title": "Group Size Imbalance", "text": 
                f"Group sizes are unequal — the largest group ({largest['size']} nodes) is "
                f"{size_ratio:.1f}× bigger than the smallest ({smallest['size']} nodes). "
                f"Smaller groups may be isolated clusters or peripheral sub-networks "
                f"with limited access to the main network."
            })

    # ── Cohesion ──
    avg_cohesion = np.mean([c["cohesion"] for c in comm_stats])
    if avg_cohesion > 0.7:
        insights.append({"title": "Community Cohesion", "text": 
            f"Average cohesion score {avg_cohesion:.2f} — groups are tightly sealed. "
            f"Nodes within each group rarely connect outside it. "
            f"This means the nodes that bridge multiple groups carry enormous responsibility: "
            f"remove one bridge node and entire groups lose contact with each other."
        })
    elif avg_cohesion < 0.4:
        insights.append({"title": "Community Cohesion", "text": 
            f"Average cohesion score {avg_cohesion:.2f} — group boundaries are open. "
            f"Nodes regularly connect across groups, so information flows freely throughout the network "
            f"without depending on specific bridge nodes."
        })

    # ── Bridge nodes ──
    if bridge_nodes:
        top_bridge = bridge_nodes[0]
        insights.append({"title": "Top Bridge Node", "text": 
            f"'{top_bridge['node']}' connects {top_bridge['connects']} different groups, "
            f"making it the most critical cross-group connector in the network. "
            f"All inter-group flow that passes through this node would be disrupted if it were removed. "
            f"{len(bridge_nodes)} cross-group connector(s) detected in total."
        })
    else:
        insights.append({"title": "No Bridge Nodes", "text": 
            f"No single node bridges multiple groups. "
            f"Each group is either self-contained or connected to others through multiple redundant paths — "
            f"a resilient structure with no cross-group single points of failure."
        })

    # ── Inter-edges ──
    if len(inter_edges) == 0:
        insights.append({"title": "Fully Isolated Communities", "text": 
            f"Zero connections exist between groups. "
            f"Each group is a completely isolated island — "
            f"nothing flows between them at all."
        })
    elif len(inter_edges) < n_comm:
        insights.append({"title": "Finding", "text": 
            f"Only {len(inter_edges)} connection(s) exist across {n_comm} groups — extremely sparse cross-group flow. "
            f"The groups are nearly isolated from each other, relying on just a handful of links."
        })

    return AnalysisResult(
        analysis_type="community",
        summary=summary,
        nodes=nodes_out,
        edges=inter_edges[:100],
        insights=insights,
    )


# ─── Module 5: Influence & Diffusion ─────────────────────────────────────────

def analyze_influence(G: nx.Graph, params: dict) -> AnalysisResult:
    model          = params.get("model",          "sir")
    infection_rate = float(params.get("infection_rate", 0.3))
    recovery_rate  = float(params.get("recovery_rate",  0.1))
    steps          = int(params.get("steps",          15))
    # FIX #4: seed_k is now a proper param, not hardcoded
    seed_k         = int(params.get("seed_k", 3))

    UG = G.to_undirected() if G.is_directed() else G
    nodes_list = list(G.nodes())
    if not nodes_list:
        raise ValueError("Graph has no nodes.")

    deg = nx.degree_centrality(G)
    bet, bet_method = compute_betweenness(G)
    pr  = nx.pagerank(G, weight="weight")
    try:
        eig = nx.eigenvector_centrality(UG, max_iter=1000, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    # Influence score: weighted composite
    influence_scores: Dict[str, float] = {
        n: round(0.35 * deg[n] + 0.25 * bet[n] + 0.25 * pr[n] + 0.15 * eig[n], 6)
        for n in nodes_list
    }

    top_seeds = sorted(nodes_list, key=lambda n: influence_scores[n], reverse=True)[:seed_k]

    def reach_at_depth(G_: nx.Graph, seed: str, depth: int) -> int:
        visited = {seed}
        frontier = {seed}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(G_.neighbors(node))
            frontier = next_frontier - visited
            visited.update(frontier)
        return len(visited) - 1

    top_seed = top_seeds[0] if top_seeds else None
    reach_depths = {}
    if top_seed:
        for d in [1, 2, 3, 4]:
            reach_depths[d] = reach_at_depth(UG, top_seed, d)

    np.random.seed(42)

    def sir_simulation(
        G_: nx.Graph, seed: str, beta: float, gamma: float, max_steps: int
    ) -> List[Dict[str, Any]]:
        S = set(G_.nodes()) - {seed}
        I = {seed}
        R: set = set()
        history = [{"step": 0, "S": len(S), "I": len(I), "R": len(R), "total_infected": 1}]
        for step in range(1, max_steps + 1):
            new_I: set = set()
            new_R: set = set()
            for node in list(I):
                for nb in G_.neighbors(node):
                    if nb in S and np.random.random() < beta:
                        new_I.add(nb)
                if np.random.random() < gamma:
                    new_R.add(node)
            S -= new_I
            I  = (I | new_I) - new_R
            R |= new_R
            history.append({
                "step":          step,
                "S":             len(S),
                "I":             len(I),
                "R":             len(R),
                "total_infected": len(I) + len(R),
            })
            if not I:
                break
        return history

    sim_history: List[Dict[str, Any]] = []
    if top_seed:
        sim_history = sir_simulation(UG, top_seed, infection_rate, recovery_rate, steps)

    final    = sim_history[-1] if sim_history else {"S": 0, "I": 0, "R": 0, "total_infected": 0}
    n_total  = len(nodes_list)
    n_reached = final.get("total_infected", 0)
    reach_rate = round(n_reached / n_total, 4) if n_total else 0

    # NEW: peak infection step
    peak_step = 0
    peak_infected = 0
    for h in sim_history:
        if h["I"] > peak_infected:
            peak_infected = h["I"]
            peak_step = h["step"]

    # NEW: multi-seed comparison (run SIR for each top seed, compare reach rates)
    multi_seed_results = []
    if len(top_seeds) > 1:
        for seed_node in top_seeds:
            hist = sir_simulation(UG, seed_node, infection_rate, recovery_rate, steps)
            final_h = hist[-1]
            multi_seed_results.append({
                "seed":         seed_node,
                "score":        round(influence_scores[seed_node], 6),
                "n_reached":    final_h.get("total_infected", 0),
                "reach_rate":   round(final_h.get("total_infected", 0) / n_total, 4) if n_total else 0,
                "steps_to_end": len(hist) - 1,
            })

    # Bottleneck node (unchanged logic, small graphs only)
    bottleneck = None
    if top_seed and n_total <= 200:
        base_reach = reach_at_depth(UG, top_seed, 3)
        best_drop  = 0
        for node in nodes_list:
            if node == top_seed:
                continue
            tmp = UG.copy()
            tmp.remove_node(node)
            if top_seed not in tmp:
                continue
            drop = base_reach - reach_at_depth(tmp, top_seed, 3)
            if drop > best_drop:
                best_drop = drop
                bottleneck = {"node": node, "reach_drop": drop}

    nodes_out = [
        NodeResult(
            id=node,
            value=round(influence_scores[node], 6),
            role="Seed" if node in top_seeds else None,
            metadata={
                "influence_score": round(influence_scores[node], 6),
                "degree":          round(deg[node],  4),
                "betweenness":     round(bet[node],  4),
                "pagerank":        round(pr[node],   4),
                "eigenvector":     round(eig[node],  4),
            },
        )
        for node in sorted(nodes_list, key=lambda n: -influence_scores[n])
    ]

    summary = {
        "top_seeds":          {n: round(influence_scores[n], 6) for n in top_seeds},
        "model":              model.upper(),
        "infection_rate":     infection_rate,
        "recovery_rate":      recovery_rate,
        "simulation_steps":   len(sim_history),
        "n_reached":          n_reached,
        "reach_rate":         reach_rate,
        "reach_by_depth":     reach_depths,      # was already there, now guaranteed
        "sim_history":        sim_history,
        "bottleneck_node":    bottleneck,
        "node_count":         n_total,
        "peak_step":          peak_step,          # NEW
        "peak_infected":      peak_infected,      # NEW
        "multi_seed_results": multi_seed_results, # NEW
        "influence_weights":  {"degree": 0.35, "betweenness": 0.25, "pagerank": 0.25, "eigenvector": 0.15},  # NEW
        "betweenness_method": bet_method,         # NEW — FIX #3
    }

    insights: List[Dict[str, str]] = []

    if bet_method != "exact":
        insights.append({"title": "Approximation Warning", "text": 
            f"Betweenness is estimated ({bet_method}) for this {n_total}-node network. "
            f"Influence score rankings are directionally reliable; exact values are approximations."
        })

    # ── Seed nodes ──
    if top_seeds:
        top_s      = top_seeds[0]
        top_s_score = influence_scores[top_s]
        insights.append({"title": "Influence Score & Seed Nodes", "text": 
            f"Influence score combines four centrality measures to find the best starting point "
            f"for spreading something across the network (information, a campaign, a risk). "
            f"Weights: 35% direct connections + 25% flow control + 25% PageRank + 15% neighbourhood quality. "
            f"'{top_s}' has the highest composite score ({top_s_score:.4f}) — "
            f"it is the single most effective node to start from."
        })

    # ── Diffusion result ──
    if reach_rate > 0.7:
        insights.append({"title": "Approximation Warning", "text": 
            f"Starting from '{top_seed}', the spread reached {n_reached} out of {n_total} nodes "
            f"({reach_rate*100:.1f}%) in {len(sim_history)-1} steps. "
            f"This network has strong propagation — once something starts here, "
            f"it reaches the vast majority quickly. "
            f"Peak simultaneous spread: {peak_infected} nodes at step {peak_step}."
        })
    elif reach_rate > 0.3:
        insights.append({"title": "Diffusion Result", "text": 
            f"Starting from '{top_seed}', the spread reached {n_reached} out of {n_total} nodes "
            f"({reach_rate*100:.1f}%) — moderate reach. "
            f"The network's group structure likely limits full propagation: "
            f"some groups are hard to reach from this starting point. "
            f"Peak simultaneous spread: {peak_infected} nodes at step {peak_step}."
        })
    else:
        insights.append({"title": "Sparse Cross-Group Flow", "text": 
            f"Starting from '{top_seed}', the spread only reached {n_reached} out of {n_total} nodes "
            f"({reach_rate*100:.1f}%). "
            f"Strong structural barriers — isolated groups or sparse connectivity — "
            f"prevent the spread from reaching most of the network even from the best seed node."
        })

    # ── Reach by depth ──
    if reach_depths:
        d1, d2, d3 = reach_depths.get(1, 0), reach_depths.get(2, 0), reach_depths.get(3, 0)
        d1_pct = round(d1 / n_total * 100, 0) if n_total else 0
        d2_pct = round(d2 / n_total * 100, 0) if n_total else 0
        insights.append({"title": "Reach by Depth", "text": 
            f"From '{top_seed}': 1 step away — {d1} nodes ({d1_pct:.0f}%); "
            f"2 steps — {d2} nodes ({d2_pct:.0f}%); 3 steps — {d3} nodes. "
            f"{'Rapid expansion: each step more than doubles the reach — a high-degree hub effect.' if d2 > d1 * 2 else 'Moderate expansion: the network grows steadily but not explosively from this node.'}"
        })

    # ── Bottleneck ──
    if bottleneck:
        insights.append({"title": "Cascade Bottleneck", "text": 
            f"'{bottleneck['node']}' is the cascade bottleneck: "
            f"removing it cuts the 3-step reach from the seed by {bottleneck['reach_drop']} nodes. "
            f"In practical terms, {bottleneck['reach_drop']} nodes become unreachable within 3 steps "
            f"if this node is unavailable. "
            f"It is the single highest-impact node for either containment or resilience planning."
        })

    # ── Multi-seed comparison ──
    if len(multi_seed_results) > 1:
        best   = multi_seed_results[0]
        second = multi_seed_results[1]
        diff   = best["reach_rate"] - second["reach_rate"]
        if diff > 0.05:
            insights.append({"title": "Seed Comparison", "text": 
                f"Seed comparison: '{best['seed']}' reaches {best['reach_rate']*100:.1f}% of the network, "
                f"versus {second['reach_rate']*100:.1f}% from '{second['seed']}' — "
                f"a {diff*100:.1f} percentage point difference. "
                f"Choosing the right starting node meaningfully changes the outcome."
            })
        else:
            insights.append({"title": "Similar Seed Performance", "text": 
                f"The top {len(multi_seed_results)} seed candidates produce similar reach "
                f"(within {diff*100:.1f} percentage points of each other). "
                f"Any of them would be an effective starting point."
            })

    return AnalysisResult(
        analysis_type="influence",
        summary=summary,
        nodes=nodes_out,
        insights=insights,
    )


# ─── Module 6: Dynamic Network ────────────────────────────────────────────────

def analyze_dynamic(edges_input: List[EdgeInput], directed: bool) -> AnalysisResult:
    from networkx.algorithms import community as nx_comm

    rows = [
        {
            "source":    e.source,
            "target":    e.target,
            "weight":    e.weight or 1.0,
            "timestamp": e.timestamp or "T1",
        }
        for e in edges_input
    ]
    df = pd.DataFrame(rows)
    timestamps = sorted(df["timestamp"].unique().tolist())

    if len(timestamps) < 2:
        return AnalysisResult(
            analysis_type="dynamic",
            summary={
                "error":      "Temporal analysis requires at least 2 distinct timestamp values.",
                "timestamps": timestamps,
                "hint":       "Add a 'timestamp' column to your edge list (e.g. 2024-01, 2024-02).",
            },
            nodes=[],
            insights=[
                "No temporal data found. Add a 'timestamp' column to enable Dynamic Network analysis.",
                "Supported timestamp formats: year (2024), year-month (2024-01), any sortable string.",
            ],
        )

    ts_metrics: List[Dict[str, Any]] = []
    centrality_trend: Dict[str, Dict[str, float]] = {}

    for ts in timestamps:
        sub_df = df[df["timestamp"] == ts]
        G_t = nx.DiGraph() if directed else nx.Graph()
        for _, row in sub_df.iterrows():
            G_t.add_edge(row["source"], row["target"], weight=row["weight"])

        n, m = G_t.number_of_nodes(), G_t.number_of_edges()
        density = nx.density(G_t) if n > 1 else 0.0

        deg_t = nx.degree_centrality(G_t)
        for node, val in deg_t.items():
            centrality_trend.setdefault(node, {})[ts] = round(val, 4)

        try:
            UG_t = G_t.to_undirected() if directed else G_t
            comms = list(nx_comm.label_propagation_communities(UG_t))
            n_comms = len(comms)
        except Exception:
            n_comms = 0

        ts_metrics.append({
            "timestamp":  ts,
            "node_count": n,
            "edge_count": m,
            "density":    round(density, 4),
            "communities": n_comms,
        })

    # FIX #2: emerging hub detection with absolute minimum threshold
    MIN_ABSOLUTE_DEG = 0.05   # must have at least 5% degree centrality at peak
    MIN_CHANGE_PCT   = 50.0   # must have grown by at least 50%

    emerging_hubs: List[Dict[str, Any]] = []
    disappearing_bridges: List[Dict[str, Any]] = []

    for node, trend in centrality_trend.items():
        vals = [trend.get(ts, 0.0) for ts in timestamps]
        first_val = next((v for v in vals if v > 0), None)
        last_val  = vals[-1]

        if (
            first_val and first_val > 0
            and last_val > first_val
            and last_val >= MIN_ABSOLUTE_DEG          # FIX #2: must be truly significant
        ):
            change_pct = (last_val - first_val) / first_val * 100
            if change_pct >= MIN_CHANGE_PCT:
                emerging_hubs.append({
                    "node":       node,
                    "start":      round(first_val, 4),
                    "end":        round(last_val, 4),
                    "change_pct": round(change_pct, 1),
                    "trend":      vals,
                })
        elif (
            first_val
            and first_val >= MIN_ABSOLUTE_DEG         # FIX #2: must have been significant
            and last_val < first_val * 0.5
        ):
            change_pct = (last_val - first_val) / first_val * 100
            disappearing_bridges.append({
                "node":       node,
                "start":      round(first_val, 4),
                "end":        round(last_val, 4),
                "change_pct": round(change_pct, 1),
                "trend":      vals,
            })

    emerging_hubs.sort(key=lambda x: -x["change_pct"])
    disappearing_bridges.sort(key=lambda x: x["change_pct"])

    # Tie events
    all_edge_sets: List[set] = []
    for ts in timestamps:
        sub = df[df["timestamp"] == ts]
        edge_set = set(zip(sub["source"], sub["target"]))
        all_edge_sets.append(edge_set)

    tie_events: List[Dict[str, Any]] = []
    for i in range(1, len(timestamps)):
        formed  = all_edge_sets[i] - all_edge_sets[i - 1]
        decayed = all_edge_sets[i - 1] - all_edge_sets[i]
        tie_events.append({
            "from_ts": timestamps[i - 1],
            "to_ts":   timestamps[i],
            "formed":  len(formed),
            "decayed": len(decayed),
            # NEW: net change
            "net":     len(formed) - len(decayed),
        })

    # Change scores
    change_scores: List[Dict[str, Any]] = []
    for i in range(1, len(ts_metrics)):
        prev, curr = ts_metrics[i - 1], ts_metrics[i]
        change_scores.append({
            "timestamp":     curr["timestamp"],
            "delta_density": round(curr["density"] - prev["density"], 4),
            "delta_nodes":   curr["node_count"] - prev["node_count"],
            "delta_edges":   curr["edge_count"]  - prev["edge_count"],
        })

    # NEW: most volatile period (largest absolute density change)
    if change_scores:
        most_volatile = max(change_scores, key=lambda x: abs(x["delta_density"]))
    else:
        most_volatile = None

    all_nodes = sorted(
        centrality_trend.keys(),
        key=lambda n: -(centrality_trend[n].get(timestamps[-1], 0))
    )
    nodes_out = [
        NodeResult(
            id=node,
            value=round(centrality_trend[node].get(timestamps[-1], 0), 4),
            metadata={
                "centrality_trend": centrality_trend[node],
                "first_seen":       next((ts for ts in timestamps if ts in centrality_trend[node]), None),
                "last_seen":        timestamps[-1] if timestamps[-1] in centrality_trend[node] else None,
            },
        )
        for node in all_nodes
    ]

    summary = {
        "timestamps":           timestamps,
        "n_timestamps":         len(timestamps),
        "ts_metrics":           ts_metrics,
        "centrality_trend":     centrality_trend,
        "emerging_hubs":        emerging_hubs[:10],
        "disappearing_bridges": disappearing_bridges[:10],
        "tie_events":           tie_events,
        "change_scores":        change_scores,
        "node_count_total":     len(all_nodes),
        "most_volatile_period": most_volatile,          # NEW
        "emerging_hub_threshold": {                     # NEW — expose thresholds
            "min_degree_centrality": MIN_ABSOLUTE_DEG,
            "min_change_pct":        MIN_CHANGE_PCT,
        },
    }

    insights: List[Dict[str, str]] = []

    # ── What is this ──
    insights.append({"title": "What is Dynamic Analysis", "text": 
        f"Dynamic network analysis tracks how connections change over time. "
        f"This dataset covers {len(timestamps)} time snapshots from {timestamps[0]} to {timestamps[-1]}, "
        f"with {len(all_nodes)} unique nodes appearing across all snapshots."
    })

    # ── Density trend ──
    d_first = ts_metrics[0]["density"]
    d_last  = ts_metrics[-1]["density"]
    d_change_pct = round((d_last - d_first) / d_first * 100, 1) if d_first > 0 else 0
    if d_last > d_first * 1.2:
        insights.append({"title": "Density Trend: Increasing", "text": 
            f"Network density grew from {d_first:.3f} to {d_last:.3f} (+{d_change_pct:.0f}%) over the period. "
            f"Connections are forming faster than new nodes are joining — "
            f"the network is becoming more tightly woven over time."
        })
    elif d_last < d_first * 0.8:
        insights.append({"title": "Density Trend: Declining", "text": 
            f"Network density dropped from {d_first:.3f} to {d_last:.3f} ({d_change_pct:.0f}%) over the period. "
            f"Connections are dissolving or the network is expanding without proportional new links — "
            f"the structure is loosening over time."
        })
    else:
        insights.append({"title": "Density Trend: Stable", "text": 
            f"Network density stayed stable ({d_first:.3f} → {d_last:.3f}, {d_change_pct:+.0f}%). "
            f"Connection patterns are consistent across time — no major structural shifts."
        })

    # ── Most volatile period ──
    if most_volatile:
        ts_label = most_volatile.get('timestamp', most_volatile.get('from_ts', ''))
        insights.append({"title": "Most Volatile Period", "text": 
            f"The most disruptive transition was at {ts_label}: "
            f"density changed by {most_volatile['delta_density']:+.4f}, "
            f"{most_volatile['delta_nodes']:+d} nodes, "
            f"{most_volatile['delta_edges']:+d} edges in a single step. "
            f"Something structurally significant happened here — worth investigating what event drove this change."
        })

    # ── Emerging hubs ──
    if emerging_hubs:
        top = emerging_hubs[0]
        insights.append({"title": "Emerging Hub", "text": 
            f"'{top['node']}' is the fastest-rising node in the network: "
            f"its centrality grew {top['change_pct']:.0f}% — "
            f"from {top['start']:.4f} at the start to {top['end']:.4f} at the end. "
            f"This node is rapidly becoming more central to the network's flow. "
            f"Monitor it — fast-rising nodes can become single points of failure if growth is unchecked."
        })

    # ── Disappearing bridges ──
    if disappearing_bridges:
        top = disappearing_bridges[0]
        insights.append({"title": "Declining Node", "text": 
            f"'{top['node']}' has lost the most influence over the period: "
            f"centrality dropped {abs(top['change_pct']):.0f}% "
            f"({top['start']:.4f} → {top['end']:.4f}). "
            f"If this node previously played a bridging role, that connection may now be at risk. "
            f"Determine whether this decline is planned or an unintended disconnection."
        })

    # ── Tie events ──
    if tie_events:
        total_formed  = sum(e["formed"]  for e in tie_events)
        total_decayed = sum(e["decayed"] for e in tie_events)
        net = total_formed - total_decayed
        insights.append({"title": "Edge Dynamics", "text": 
            f"Across all transitions: {total_formed} new connections formed, "
            f"{total_decayed} connections dissolved (net: {net:+d}). "
            f"{'The network is growing — more connections are being added than lost.' if net > 0 else 'The network is contracting — more connections are being lost than added.' if net < 0 else 'Edge churn is balanced — as many connections form as dissolve.'}"
        })

    # ── Stable core ──
    nodes_all_ts = [nd for nd, t in centrality_trend.items() if len(t) == len(timestamps)]
    if nodes_all_ts:
        insights.append({"title": "Stable Core Nodes", "text": 
            f"{len(nodes_all_ts)} node(s) appear in every single snapshot — "
            f"the stable backbone of the network: {', '.join(nodes_all_ts[:3])}{'...' if len(nodes_all_ts) > 3 else ''}. "
            f"These nodes are consistently present regardless of how the rest of the network changes."
        })

    return AnalysisResult(
        analysis_type="dynamic",
        summary=summary,
        nodes=nodes_out,
        insights=insights,
    )


# ─── Main Endpoint ────────────────────────────────────────────────────────────

@router.post("/sna", response_model=AnalysisResult)
async def run_sna(request: NetworkInput):
    try:
        if not request.edges:
            raise HTTPException(status_code=400, detail="Edge list is empty.")

        G = build_graph(request.edges, request.directed)
        if G.number_of_nodes() > 5000:
            raise HTTPException(
                status_code=400,
                detail=f"Graph has {G.number_of_nodes()} nodes — too large for in-memory analysis. Sample to ≤5000 nodes first.",
            )

        if G.number_of_nodes() == 0:
            raise HTTPException(status_code=400, detail="No nodes found. Check edge data.")

        t = request.analysis_type
        p = request.params or {}

        if t == "overview":
            return analyze_overview(G)
        elif t == "centrality":
            return analyze_centrality(G, p)
        elif t == "role":
            return analyze_role(G)
        elif t == "community":
            return analyze_community(G, p)
        elif t == "influence":
            return analyze_influence(G, p)
        elif t == "dynamic":
            return analyze_dynamic(request.edges, request.directed)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported analysis_type: '{t}'. "
                       f"Available: overview, centrality, role, community, influence, dynamic",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/sna/health")
async def health():
    return {
        "status":  "ok",
        "phase":   6,
        "version": "improved",
        "modules": ["overview", "centrality", "role", "community", "influence", "dynamic"],
        "fixes":   [
            "external_edges O(E) single-pass (was O(N*E))",
            "emerging hub min absolute threshold (was % only)",
            "betweenness approx warning propagated to frontend",
            "seed_k param respected (was hardcoded=3)",
            "role classification single source of truth",
        ],
        "new_fields": [
            "overview.degree_histogram",
            "centrality.betweenness_method + centrality_correlation",
            "role.thresholds",
            "community.cohesion + inter_density per community",
            "influence.peak_step + peak_infected + multi_seed_results + influence_weights",
            "dynamic.most_volatile_period + tie_events.net + emerging_hub_threshold",
        ],
    }


# ─── Graph Visualization ─────────────────────────────────────────────────────

from fastapi.responses import HTMLResponse
from fastapi import Query as QParam
import json as _json

class GraphVizInput(BaseModel):
    edges:    List[EdgeInput]
    directed: bool = True
    colorBy:  str  = "degree"
    params:   Optional[Dict[str, Any]] = {}

@router.post("/sna/graph", response_class=HTMLResponse)
async def graph_viz_post(request: GraphVizInput):
    edges_str = _json.dumps([{"source": e.source, "target": e.target, "weight": e.weight} for e in request.edges])
    return await graph_viz_get(edges=edges_str, directed=request.directed, colorBy=request.colorBy)

@router.get("/sna/graph", response_class=HTMLResponse)
async def graph_viz_get(
    edges:    str  = QParam(..., description="JSON array of edge objects"),
    directed: bool = True,
    colorBy:  str  = "degree",
):
    try:
        edge_list = _json.loads(edges)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid edges JSON")

    edge_inputs = [EdgeInput(
        source=str(e.get("source", "")),
        target=str(e.get("target", "")),
        weight=float(e.get("weight", 1)),
    ) for e in edge_list if e.get("source") and e.get("target")]

    if not edge_inputs:
        raise HTTPException(status_code=400, detail="No valid edges")

    G = build_graph(edge_inputs, directed)
    n = G.number_of_nodes()
    if n == 0:
        raise HTTPException(status_code=400, detail="No nodes found.")
    if n > 2000:
        raise HTTPException(status_code=400, detail="Too many nodes (max 2000).")

    deg = nx.degree_centrality(G)
    bet, _  = compute_betweenness(G)   # FIX #5: use shared helper
    pr  = nx.pagerank(G, weight="weight")
    UG  = G.to_undirected() if G.is_directed() else G
    clo = nx.closeness_centrality(G)

    try:
        eig = nx.eigenvector_centrality(UG, max_iter=500, weight="weight")
    except Exception:
        eig = {node: deg[node] for node in G.nodes()}

    try:
        from networkx.algorithms import community as nx_comm
        comms = list(nx_comm.louvain_communities(UG, seed=42))
        node_comm: Dict[str, int] = {}
        for idx, comm in enumerate(comms):
            for node in comm:
                node_comm[node] = idx
    except Exception:
        node_comm = {node: 0 for node in G.nodes()}

    # FIX #5: use shared threshold + classification — no more duplicate logic
    thresholds = compute_role_thresholds(G, deg, bet, eig, clo)

    ROLE_COLORS = {
        "Hub": "#3b82f6", "Bridge/Broker": "#a855f7",
        "Regular": "#14b8a6", "Peripheral": "#94a3b8", "Isolate": "#ef4444",
    }
    COMM_COLORS = [
        "#3b82f6", "#a855f7", "#14b8a6", "#f97316",
        "#ec4899", "#f59e0b", "#06b6d4", "#22c55e", "#ef4444", "#6b7280"
    ]
    DEGREE_STOPS = ["#94a3b8", "#60a5fa", "#34d399", "#fbbf24", "#f87171"]
    max_deg = max(deg.values()) if deg else 1

    def get_color(node: str) -> str:
        if colorBy == "role":
            role = classify_node_role(node, G, deg, bet, eig, clo, thresholds)
            return ROLE_COLORS.get(role, "#14b8a6")
        if colorBy == "community":
            return COMM_COLORS[node_comm.get(node, 0) % len(COMM_COLORS)]
        t = deg[node] / max_deg if max_deg > 0 else 0
        return DEGREE_STOPS[round(t * (len(DEGREE_STOPS) - 1))]

    def get_size(node: str) -> int:
        return 12 + int(deg[node] / max(max_deg, 0.001) * 28)

    nodes_data = []
    for node in G.nodes():
        role  = classify_node_role(node, G, deg, bet, eig, clo, thresholds)
        color = get_color(node)
        nodes_data.append({
            "id": node, "label": node,
            "size": get_size(node),
            "color": {
                "background": color, "border": "#ffffff",
                "highlight": {"background": color, "border": "#facc15"},
                "hover":     {"background": color, "border": "#facc15"},
            },
            "font": {"size": 13, "color": "#1e293b"},
            "title": (
                f"<div style='font:13px sans-serif;padding:8px;line-height:1.6'>"
                f"<b>{node}</b><br>"
                f"Role: {role}<br>"
                f"Degree: {deg[node]:.4f}<br>"
                f"Betweenness: {bet[node]:.4f}<br>"
                f"PageRank: {pr[node]:.4f}<br>"
                f"Community: C{node_comm.get(node, 0)}"
                f"</div>"
            ),
        })

    edges_data = []
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        w = data.get("weight", 1)
        edges_data.append({
            "id": i, "from": u, "to": v, "value": w,
            "title": f"weight: {w}",
            "color": {"color": "#94a3b8", "opacity": 0.55},
            "arrows": {"to": {"enabled": G.is_directed(), "scaleFactor": 0.5}},
        })

    nodes_json = _json.dumps(nodes_data)
    edges_json = _json.dumps(edges_data)

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#fff; width:100vw; height:100vh; overflow:hidden; }
#graph { width:100%; height:100%; }
</style>
</head>
<body>
<div id="graph"></div>
<script>
var nodes = new vis.DataSet(""" + nodes_json + """);
var edges = new vis.DataSet(""" + edges_json + """);
var net = new vis.Network(
  document.getElementById('graph'),
  { nodes: nodes, edges: edges },
  {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant:-80, centralGravity:0.005, springLength:120, springConstant:0.08, damping:0.4, avoidOverlap:0.8 },
      stabilization: { iterations:150 }
    },
    interaction: { hover:true, tooltipDelay:100, hideEdgesOnDrag:true },
    nodes: { shape:'dot', borderWidth:2, shadow:{ enabled:true, size:6 } },
    edges: { smooth:{ type:'dynamic' }, scaling:{ min:1, max:6 } }
  }
);
</script>
</body>
</html>"""

    return HTMLResponse(content=html)
