"""
routers/spatial.py
공간 분석 엔드포인트

POST /api/spatial/radius         — 반경 내 포인트 집계
POST /api/spatial/buffer         — 포인트별 버퍼 존 생성
POST /api/spatial/isochrone      — 이동 시간 등시선 근사
POST /api/spatial/outlier        — 공간적 이상치 탐지
POST /api/spatial/cannibalization — 매장 간 상권 겹침 분석
POST /api/spatial/locationscore  — 입지 점수 산출
POST /api/spatial/spatialjoin    — 사용자 정의 영역 포인트 집계
"""

import math
import statistics
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import (
    centroid,
    circle_polygon,
    haversine,
    to_native,
)

router = APIRouter()


# ══════════════════════════════════════════════════════════
# 공통
# ══════════════════════════════════════════════════════════

class BaseMapRequest(BaseModel):
    data: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════
# 반경 분석
# ══════════════════════════════════════════════════════════

class RadiusRequest(BaseModel):
    center: Dict[str, float]     # {lat, lng}
    radiusM: float
    data: List[Dict[str, Any]]
    numericCols: Optional[List[str]] = None


@router.post("/api/spatial/radius")
def run_radius(req: RadiusRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    inside = [
        p for p in pts
        if haversine(req.center["lat"], req.center["lng"], p["lat"], p["lng"]) <= req.radiusM
    ]

    # 수치형 컬럼 통계
    numeric_stats: dict[str, dict] = {}
    cols = req.numericCols or []
    for col in cols:
        vals = [p[col] for p in inside if isinstance(p.get(col), (int, float))]
        if vals:
            numeric_stats[col] = {
                "count": len(vals),
                "sum": sum(vals),
                "mean": statistics.mean(vals),
                "min": min(vals),
                "max": max(vals),
            }

    return to_native({
        "results": {
            "center": req.center,
            "radiusM": req.radiusM,
            "totalPoints": len(pts),
            "insideCount": len(inside),
            "insidePoints": inside,
            "stats": {
                "totalPoints": len(inside),
                **numeric_stats,
            },
        }
    })


# ══════════════════════════════════════════════════════════
# 버퍼 존
# ══════════════════════════════════════════════════════════

class BufferRequest(BaseMapRequest):
    radiusM: float = 500
    labelCol: Optional[str] = None


BUFFER_COLORS = [
    "#3b82f6","#ef4444","#22c55e","#a855f7","#f97316",
    "#ec4899","#14b8a6","#eab308","#6366f1","#84cc16",
]


@router.post("/api/spatial/buffer")
def run_buffer(req: BufferRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]

    buffers = []
    for i, row in enumerate(pts):
        polygon = circle_polygon(row["lat"], row["lng"], req.radiusM)
        # 다른 포인트와 겹치는 수
        overlapping = sum(
            1 for j, other in enumerate(pts)
            if j != i and haversine(row["lat"], row["lng"], other["lat"], other["lng"]) <= req.radiusM * 2
        )
        buffers.append({
            "center": {"lat": row["lat"], "lng": row["lng"]},
            "polygon": polygon,
            "radiusM": req.radiusM,
            "color": BUFFER_COLORS[i % len(BUFFER_COLORS)],
            "row": row,
            "overlappingWith": overlapping,
        })

    return to_native({"results": {"buffers": buffers, "visible": True}})


# ══════════════════════════════════════════════════════════
# 이소크론 (등시선 — 직선 거리 기반 근사)
# ══════════════════════════════════════════════════════════

class IsochroneRequest(BaseModel):
    center: Dict[str, float]
    minutesList: List[int]       # ex) [5, 10, 15, 30]
    speedKmh: float = 40.0       # 평균 속도
    data: List[Dict[str, Any]]


@router.post("/api/spatial/isochrone")
def run_isochrone(req: IsochroneRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    bands = []

    for minutes in sorted(req.minutesList, reverse=True):  # 큰 것부터 (레이어링)
        radius_m = (req.speedKmh * 1000 / 60) * minutes
        polygon = circle_polygon(req.center["lat"], req.center["lng"], radius_m)
        reachable = [
            p for p in pts
            if haversine(req.center["lat"], req.center["lng"], p["lat"], p["lng"]) <= radius_m
        ]
        # 색상 그라데이션 (초록 → 노랑 → 빨강)
        t = (minutes - min(req.minutesList)) / max((max(req.minutesList) - min(req.minutesList)), 1)
        r = int(34 + (239 - 34) * t)
        g = int(197 - (197 - 68) * t)
        b = int(94 - (94 - 68) * t)
        bands.append({
            "minutes": minutes,
            "radiusM": radius_m,
            "polygon": polygon,
            "color": f"rgb({r},{g},{b})",
            "reachablePoints": reachable,
        })

    return to_native({
        "results": {
            "center": req.center,
            "bands": bands,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# 공간적 이상치 탐지 (Local Outlier Factor 근사)
# ══════════════════════════════════════════════════════════

class OutlierRequest(BaseMapRequest):
    k: int = 5                  # 이웃 수
    threshold: float = 2.0      # LOF 임계값
    showOutliersOnly: bool = False


@router.post("/api/spatial/outlier")
def run_outlier(req: OutlierRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    n = len(pts)
    if n < req.k + 1:
        raise HTTPException(400, f"포인트가 최소 {req.k + 1}개 필요합니다.")

    k = min(req.k, n - 1)

    # k-NN 거리 계산
    def k_distances(idx):
        dists = sorted(
            haversine(pts[idx]["lat"], pts[idx]["lng"], pts[j]["lat"], pts[j]["lng"])
            for j in range(n) if j != idx
        )
        return dists[:k]

    # LOF 근사: k-NN 평균 거리의 상대적 비율
    avg_dists = [sum(k_distances(i)) / k for i in range(n)]
    global_avg = statistics.mean(avg_dists) or 1.0

    results = []
    for i, p in enumerate(pts):
        score = avg_dists[i] / global_avg
        results.append({
            "row": p,
            "score": score,
            "isOutlier": score > req.threshold,
        })

    results.sort(key=lambda x: -x["score"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return to_native({
        "results": {
            "points": results,
            "threshold": req.threshold,
            "nOutliers": sum(1 for r in results if r["isOutlier"]),
            "showOutliersOnly": req.showOutliersOnly,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# 자기잠식 분석 (Cannibalization)
# ══════════════════════════════════════════════════════════

class CannibalizationRequest(BaseMapRequest):
    radiusM: float = 1000
    labelCol: Optional[str] = None


@router.post("/api/spatial/cannibalization")
def run_cannibalization(req: CannibalizationRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]

    circles = []
    for i, p in enumerate(pts):
        circles.append({
            "center": {"lat": p["lat"], "lng": p["lng"]},
            "radiusM": req.radiusM,
            "color": BUFFER_COLORS[i % len(BUFFER_COLORS)],
            "label": str(p.get(req.labelCol, f"Store {i+1}")) if req.labelCol else f"Store {i+1}",
        })

    pairs = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = haversine(pts[i]["lat"], pts[i]["lng"], pts[j]["lat"], pts[j]["lng"])
            if dist < req.radiusM * 2:
                overlap_pct = max(0.0, (req.radiusM * 2 - dist) / (req.radiusM * 2) * 100)
                shared = [
                    p for p in pts
                    if haversine(pts[i]["lat"], pts[i]["lng"], p["lat"], p["lng"]) <= req.radiusM
                    and haversine(pts[j]["lat"], pts[j]["lng"], p["lat"], p["lng"]) <= req.radiusM
                    and p is not pts[i] and p is not pts[j]
                ]
                severity = (
                    "critical" if overlap_pct >= 60
                    else "high" if overlap_pct >= 40
                    else "medium" if overlap_pct >= 20
                    else "low"
                )
                pairs.append({
                    "storeA": {**pts[i], "label": circles[i]["label"]},
                    "storeB": {**pts[j], "label": circles[j]["label"]},
                    "distance": dist,
                    "overlapPercent": overlap_pct,
                    "sharedPoints": shared,
                    "severity": severity,
                })

    return to_native({
        "results": {
            "circles": circles,
            "pairs": pairs,
            "nConflicts": len(pairs),
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# 입지 점수 (Location Score)
# ══════════════════════════════════════════════════════════

class LocationScoreRequest(BaseMapRequest):
    criteria: List[Dict[str, Any]]  # [{col, weight, direction}]


@router.post("/api/spatial/locationscore")
def run_location_score(req: LocationScoreRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if not req.criteria:
        raise HTTPException(400, "기준(criteria)이 없습니다.")

    # 각 기준 정규화
    scored = []
    for p in pts:
        total = 0.0
        total_weight = sum(c.get("weight", 1.0) for c in req.criteria)
        for crit in req.criteria:
            col = crit.get("col")
            weight = crit.get("weight", 1.0)
            direction = crit.get("direction", "higher")  # higher | lower
            val = p.get(col)
            if val is None:
                continue
            # 전체 컬럼 범위 계산
            col_vals = [r.get(col) for r in pts if isinstance(r.get(col), (int, float))]
            if not col_vals:
                continue
            min_v, max_v = min(col_vals), max(col_vals)
            rng = max_v - min_v or 1.0
            norm = (val - min_v) / rng
            if direction == "lower":
                norm = 1.0 - norm
            total += norm * weight

        total_score = (total / total_weight) * 100 if total_weight else 0.0
        scored.append({"row": p, "totalScore": total_score})

    scored.sort(key=lambda x: -x["totalScore"])
    for rank, s in enumerate(scored, 1):
        s["rank"] = rank

    return to_native({
        "results": {
            "points": scored,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Spatial Join (사용자 정의 영역)
# ══════════════════════════════════════════════════════════

class SpatialArea(BaseModel):
    id: str
    label: str
    polygon: List[List[float]]   # [[lat, lng], ...]
    color: Optional[str] = "#3b82f6"


class SpatialJoinRequest(BaseMapRequest):
    areas: List[SpatialArea]
    aggregateCols: Optional[List[str]] = None


def _point_in_polygon(lat: float, lng: float, polygon: list[list[float]]) -> bool:
    """Ray casting 알고리즘"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][1], polygon[i][0]
        xj, yj = polygon[j][1], polygon[j][0]
        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@router.post("/api/spatial/spatialjoin")
def run_spatial_join(req: SpatialJoinRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    results = []

    for area in req.areas:
        members = [
            p for p in pts
            if _point_in_polygon(p["lat"], p["lng"], area.polygon)
        ]

        agg: dict[str, Any] = {"count": len(members)}
        if req.aggregateCols:
            for col in req.aggregateCols:
                vals = [m[col] for m in members if isinstance(m.get(col), (int, float))]
                if vals:
                    agg[f"{col}_sum"] = sum(vals)
                    agg[f"{col}_mean"] = statistics.mean(vals)

        results.append({
            "id": area.id,
            "label": area.label,
            "polygon": area.polygon,
            "color": area.color,
            "pointCount": len(members),
            "center": centroid(members) if members else {"lat": 0, "lng": 0},
            "aggregations": agg,
        })

    return to_native({
        "results": {
            "areas": results,
            "visible": True,
        }
    })
