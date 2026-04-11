"""
routers/clustering.py
클러스터링 & 공간 분할 엔드포인트

POST /api/clustering/dbscan       — DBSCAN 밀집 군집
POST /api/clustering/voronoi      — Voronoi 세력권 분할
POST /api/clustering/hull         — Convex Hull (그룹별 외곽선)
POST /api/clustering/gridhex      — Grid / Hexbin 밀도 셀
"""

import math
import random
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import (
    centroid,
    convex_hull,
    haversine,
    polygon_area_km2,
    to_native,
)

router = APIRouter()


# ══════════════════════════════════════════════════════════
# 공통 모델
# ══════════════════════════════════════════════════════════

class GeoRow(BaseModel):
    id: str
    lat: float
    lng: float
    # 나머지 필드는 dict로
    class Config:
        extra = "allow"


class BaseMapRequest(BaseModel):
    data: List[Dict[str, Any]]  # MapDataRow[]


# ══════════════════════════════════════════════════════════
# DBSCAN
# ══════════════════════════════════════════════════════════

class DBSCANRequest(BaseMapRequest):
    epsilonM: float = 500       # 반경 (미터)
    minPoints: int = 3          # 최소 포인트 수


def _dbscan(points: list, eps_m: float, min_pts: int):
    """순수 Python DBSCAN (scikit-learn 없이)"""
    n = len(points)
    labels = [-2] * n   # -2 = 미방문

    def neighbors(idx):
        return [
            j for j in range(n)
            if haversine(points[idx]["lat"], points[idx]["lng"],
                         points[j]["lat"], points[j]["lng"]) <= eps_m
        ]

    cluster_id = 0
    for i in range(n):
        if labels[i] != -2:
            continue
        nbs = neighbors(i)
        if len(nbs) < min_pts:
            labels[i] = -1  # noise
            continue
        labels[i] = cluster_id
        queue = list(nbs)
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -2:
                continue
            labels[j] = cluster_id
            nb2 = neighbors(j)
            if len(nb2) >= min_pts:
                queue.extend(nb2)
        cluster_id += 1

    return labels, cluster_id


CLUSTER_COLORS = [
    "#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f97316",
    "#ec4899", "#14b8a6", "#eab308", "#6366f1", "#84cc16",
]


@router.post("/api/clustering/dbscan")
def run_dbscan(req: DBSCANRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < req.minPoints:
        raise HTTPException(400, "포인트 수가 너무 적습니다.")

    labels, n_clusters = _dbscan(pts, req.epsilonM, req.minPoints)

    # 클러스터별 집계
    clusters_map: dict[int, list] = {}
    noise = []
    for i, label in enumerate(labels):
        if label == -1:
            noise.append(pts[i])
        else:
            clusters_map.setdefault(label, []).append(pts[i])

    clusters = []
    for cid, members in clusters_map.items():
        c = centroid(members)
        # 반경 = 센터에서 가장 먼 포인트까지
        radius = max(haversine(c["lat"], c["lng"], p["lat"], p["lng"]) for p in members)
        clusters.append({
            "id": cid,
            "color": CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            "center": c,
            "count": len(members),
            "radius": radius,
            "points": members,
            "visible": True,
        })

    return to_native({
        "results": {
            "clusters": clusters,
            "noise": noise,
            "noiseColor": "#9ca3af",
            "showNoise": True,
            "nClusters": n_clusters,
            "nNoise": len(noise),
            "epsilonM": req.epsilonM,
            "minPoints": req.minPoints,
        }
    })


# ══════════════════════════════════════════════════════════
# Voronoi (Fortune's — 근사: 각 포인트에서 가장 가까운 센터)
# ══════════════════════════════════════════════════════════

class VoronoiRequest(BaseMapRequest):
    centers: List[Dict[str, Any]]  # {id, label, lat, lng} 리스트


@router.post("/api/clustering/voronoi")
def run_voronoi(req: VoronoiRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    centers = [c for c in req.centers if c.get("lat") and c.get("lng")]

    if not centers:
        raise HTTPException(400, "센터가 없습니다.")

    # 각 포인트를 가장 가까운 센터에 배정
    assignments: dict[int, list] = {i: [] for i in range(len(centers))}
    for pt in pts:
        best = min(
            range(len(centers)),
            key=lambda i: haversine(pt["lat"], pt["lng"], centers[i]["lat"], centers[i]["lng"])
        )
        assignments[best].append(pt)

    # 센터에 colorIdx 추가
    enriched_centers = []
    for i, c in enumerate(centers):
        enriched_centers.append({**c, "colorIdx": i})

    colors = [
        {"fill": f"rgba({r},{g},{b},0.12)", "border": f"rgb({r},{g},{b})"}
        for r, g, b in [
            (59,130,246), (239,68,68), (34,197,94), (168,85,247),
            (249,115,22), (236,72,153), (20,184,166), (234,179,8),
        ]
    ]

    # assignments를 index → rows 형태로
    assign_out = {str(k): v for k, v in assignments.items()}

    return to_native({
        "results": {
            "centers": enriched_centers,
            "assignments": assign_out,
            "colors": colors,
            "stats": [
                {
                    "centerIdx": i,
                    "label": centers[i].get("label", f"Center {i+1}"),
                    "count": len(assignments[i]),
                    "center": {"lat": centers[i]["lat"], "lng": centers[i]["lng"]},
                }
                for i in range(len(centers))
            ],
        }
    })


# ══════════════════════════════════════════════════════════
# Convex Hull (그룹별)
# ══════════════════════════════════════════════════════════

class HullRequest(BaseMapRequest):
    groupCol: Optional[str] = None   # None이면 전체 하나의 hull


@router.post("/api/clustering/hull")
def run_hull(req: HullRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]

    if req.groupCol:
        groups: dict[str, list] = {}
        for p in pts:
            key = str(p.get(req.groupCol, "Unknown"))
            groups.setdefault(key, []).append(p)
    else:
        groups = {"All": pts}

    hull_colors = [
        "#3b82f6", "#ef4444", "#22c55e", "#a855f7",
        "#f97316", "#ec4899", "#14b8a6", "#eab308",
    ]
    hulls = []
    for i, (label, members) in enumerate(groups.items()):
        hull = convex_hull(members)
        hulls.append({
            "id": i,
            "label": label,
            "hull": hull,
            "pointCount": len(members),
            "areaKm2": polygon_area_km2(hull),
            "colorIdx": i % len(hull_colors),
            "color": hull_colors[i % len(hull_colors)],
            "visible": True,
        })

    return to_native({"results": {"hulls": hulls}})


# ══════════════════════════════════════════════════════════
# Grid / Hex Bin
# ══════════════════════════════════════════════════════════

class GridHexRequest(BaseMapRequest):
    mode: str = "grid"          # "grid" | "hex"
    cellSizeKm: float = 1.0
    aggregation: str = "count"  # "count" | 컬럼명
    colorScheme: str = "heat"


def _hex_vertices(center_lat: float, center_lng: float, size_km: float) -> list[dict]:
    """정육각형 꼭짓점 반환 (flat-top)"""
    pts = []
    size_m = size_km * 1000
    for i in range(6):
        angle = math.radians(60 * i)
        from map.utils import destination
        pts.append(destination(center_lat, center_lng, math.degrees(angle), size_m))
    return pts


@router.post("/api/clustering/gridhex")
def run_gridhex(req: GridHexRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if not pts:
        raise HTTPException(400, "포인트가 없습니다.")

    cell_deg = req.cellSizeKm / 111.0  # km → 위도 도(degree) 근사

    if req.mode == "grid":
        # 격자 셀 집계
        grid: dict[tuple, list] = {}
        for p in pts:
            gi = math.floor(p["lat"] / cell_deg)
            gj = math.floor(p["lng"] / cell_deg)
            grid.setdefault((gi, gj), []).append(p)

        cells = []
        for (gi, gj), members in grid.items():
            min_lat = gi * cell_deg
            max_lat = min_lat + cell_deg
            min_lng = gj * cell_deg
            max_lng = min_lng + cell_deg

            value: float
            if req.aggregation == "count":
                value = float(len(members))
            else:
                vals = [m.get(req.aggregation) for m in members if m.get(req.aggregation) is not None]
                value = float(sum(vals) / len(vals)) if vals else 0.0

            cells.append({
                "id": f"{gi}_{gj}",
                "count": len(members),
                "value": value,
                "bounds": {"minLat": min_lat, "maxLat": max_lat, "minLng": min_lng, "maxLng": max_lng},
            })

    else:  # hex
        hex_map: dict[tuple, list] = {}
        hex_size = cell_deg

        for p in pts:
            # Offset hex grid 근사
            col = round(p["lng"] / (hex_size * 1.5))
            row_offset = 0.5 if col % 2 else 0.0
            row = round((p["lat"] - row_offset * hex_size) / (hex_size * math.sqrt(3)))
            hex_map.setdefault((col, row), []).append(p)

        cells = []
        for (col, row), members in hex_map.items():
            row_offset = 0.5 if col % 2 else 0.0
            c_lat = (row + row_offset) * hex_size * math.sqrt(3)
            c_lng = col * hex_size * 1.5

            value: float
            if req.aggregation == "count":
                value = float(len(members))
            else:
                vals = [m.get(req.aggregation) for m in members if m.get(req.aggregation) is not None]
                value = float(sum(vals) / len(vals)) if vals else 0.0

            cells.append({
                "id": f"{col}_{row}",
                "count": len(members),
                "value": value,
                "vertices": _hex_vertices(c_lat, c_lng, req.cellSizeKm * 0.5),
            })

    # 정규화
    if cells:
        max_val = max(c["value"] for c in cells) or 1.0
        for c in cells:
            c["normalizedValue"] = c["value"] / max_val
    
    return to_native({
        "results": {
            "cells": cells,
            "mode": req.mode,
            "aggregation": req.aggregation,
            "colorScheme": req.colorScheme,
            "visible": True,
        }
    })
