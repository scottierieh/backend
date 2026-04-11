"""
routers/clustering_advanced.py

POST /api/clustering/hdbscan     — HDBSCAN (better density clustering)
POST /api/clustering/kmeans      — K-Means spatial clustering
POST /api/clustering/som         — Self-Organizing Map clustering
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import centroid, haversine, to_native

router = APIRouter()

CLUSTER_COLORS = [
    "#3b82f6","#ef4444","#22c55e","#a855f7","#f97316",
    "#ec4899","#14b8a6","#eab308","#6366f1","#84cc16",
    "#0ea5e9","#f43f5e","#10b981","#8b5cf6","#fb923c",
]


class BaseMapRequest(BaseModel):
    data: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════
# HDBSCAN
# ══════════════════════════════════════════════════════════

class HDBSCANRequest(BaseMapRequest):
    minClusterSize: int = 5
    minSamples: Optional[int] = None


@router.post("/api/clustering/hdbscan")
def run_hdbscan(req: HDBSCANRequest):
    try:
        import hdbscan as hdb
    except ImportError:
        raise HTTPException(500, "hdbscan not installed")

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < req.minClusterSize:
        raise HTTPException(400, f"Need at least {req.minClusterSize} points.")

    coords = np.array([[p["lat"], p["lng"]] for p in pts])
    # Convert to radians for haversine metric
    coords_rad = np.radians(coords)

    clusterer = hdb.HDBSCAN(
        min_cluster_size=req.minClusterSize,
        min_samples=req.minSamples,
        metric="haversine",
    )
    labels = clusterer.fit_predict(coords_rad)
    probabilities = clusterer.probabilities_

    clusters_map: dict[int, list] = {}
    noise = []
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        row = {**pts[i], "_probability": float(prob)}
        if label == -1:
            noise.append(row)
        else:
            clusters_map.setdefault(int(label), []).append(row)

    clusters = []
    for cid, members in sorted(clusters_map.items()):
        c = centroid(members)
        radius = max(
            haversine(c["lat"], c["lng"], p["lat"], p["lng"]) for p in members
        ) if len(members) > 1 else 0
        avg_prob = float(np.mean([m["_probability"] for m in members]))
        clusters.append({
            "id": cid,
            "color": CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            "center": c,
            "count": len(members),
            "radius": radius,
            "avgProbability": avg_prob,
            "points": members,
            "visible": True,
        })

    return to_native({
        "results": {
            "clusters": clusters,
            "noise": noise,
            "noiseColor": "#9ca3af",
            "showNoise": True,
            "nClusters": len(clusters),
            "nNoise": len(noise),
            "minClusterSize": req.minClusterSize,
        }
    })


# ══════════════════════════════════════════════════════════
# K-Means
# ══════════════════════════════════════════════════════════

class KMeansRequest(BaseMapRequest):
    k: int = 5
    nInit: int = 10


@router.post("/api/clustering/kmeans")
def run_kmeans(req: KMeansRequest):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < req.k:
        raise HTTPException(400, f"Need at least {req.k} points for k={req.k}.")

    coords = np.array([[p["lat"], p["lng"]] for p in pts])
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    km = KMeans(n_clusters=req.k, n_init=req.nInit, random_state=42)
    labels = km.fit_predict(coords_scaled)

    # Cluster centers back to lat/lng
    centers_scaled = km.cluster_centers_
    centers_latlon = scaler.inverse_transform(centers_scaled)

    clusters = []
    for cid in range(req.k):
        members = [pts[i] for i, l in enumerate(labels) if l == cid]
        c_lat, c_lng = centers_latlon[cid]
        radius = max(
            haversine(c_lat, c_lng, p["lat"], p["lng"]) for p in members
        ) if members else 0
        clusters.append({
            "id": cid,
            "color": CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            "center": {"lat": float(c_lat), "lng": float(c_lng)},
            "count": len(members),
            "radius": radius,
            "points": members,
            "visible": True,
        })

    inertia = float(km.inertia_)

    return to_native({
        "results": {
            "clusters": clusters,
            "noise": [],
            "noiseColor": "#9ca3af",
            "showNoise": False,
            "nClusters": req.k,
            "nNoise": 0,
            "inertia": inertia,
        }
    })


# ══════════════════════════════════════════════════════════
# SOM (Self-Organizing Map)
# ══════════════════════════════════════════════════════════

class SOMRequest(BaseMapRequest):
    gridX: int = 4
    gridY: int = 3
    iterations: int = 1000
    featureCols: Optional[List[str]] = None   # cols to cluster on (besides lat/lng)


@router.post("/api/clustering/som")
def run_som(req: SOMRequest):
    try:
        from minisom import MiniSom
    except ImportError:
        raise HTTPException(500, "minisom not installed")
    from sklearn.preprocessing import StandardScaler

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < 4:
        raise HTTPException(400, "Need at least 4 points.")

    # Build feature matrix: lat + lng + optional numeric cols
    feat_cols = req.featureCols or []
    valid_cols = [c for c in feat_cols if all(isinstance(p.get(c), (int, float)) for p in pts)]
    feature_matrix = []
    for p in pts:
        row = [p["lat"], p["lng"]] + [float(p[c]) for c in valid_cols]
        feature_matrix.append(row)

    X = np.array(feature_matrix, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = X_scaled.shape[1]
    som = MiniSom(req.gridX, req.gridY, n_features,
                  sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, req.iterations)

    # Assign each point to its BMU (best matching unit)
    bmu_map: dict[tuple, list] = {}
    for i, x in enumerate(X_scaled):
        bmu = som.winner(x)
        bmu_map.setdefault(bmu, []).append(pts[i])

    clusters = []
    for idx, ((gx, gy), members) in enumerate(sorted(bmu_map.items())):
        c = centroid(members)
        radius = max(
            haversine(c["lat"], c["lng"], p["lat"], p["lng"]) for p in members
        ) if len(members) > 1 else 0
        clusters.append({
            "id": idx,
            "gridX": gx,
            "gridY": gy,
            "color": CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
            "center": c,
            "count": len(members),
            "radius": radius,
            "points": members,
            "visible": True,
        })

    return to_native({
        "results": {
            "clusters": clusters,
            "noise": [],
            "noiseColor": "#9ca3af",
            "showNoise": False,
            "nClusters": len(clusters),
            "nNoise": 0,
            "gridX": req.gridX,
            "gridY": req.gridY,
        }
    })
