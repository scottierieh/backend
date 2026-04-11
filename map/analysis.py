"""
routers/analysis.py
데이터 분석 엔드포인트

POST /api/analysis/heatmap      — 히트맵 포인트 생성
POST /api/analysis/bivariate    — 이변량 공간 분석
POST /api/analysis/timeseries   — 시계열 프레임 분리
POST /api/analysis/filter       — 서버사이드 필터링
POST /api/analysis/distancematrix — 거리 매트릭스 (전체)
"""

import statistics
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import haversine, to_native

router = APIRouter()


# ══════════════════════════════════════════════════════════
# 히트맵
# ══════════════════════════════════════════════════════════

class HeatmapRequest(BaseModel):
    data: List[Dict[str, Any]]
    weightCol: Optional[str] = None   # None → 모든 포인트 가중치 1
    radius: int = 25
    blur: int = 15
    maxZoom: int = 17
    gradient: Optional[Dict[str, str]] = None


@router.post("/api/analysis/heatmap")
def run_heatmap(req: HeatmapRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]

    # Leaflet.heat 형식: [lat, lng, intensity]
    points: list[list[float]] = []
    for p in pts:
        weight = 1.0
        if req.weightCol:
            w = p.get(req.weightCol)
            if isinstance(w, (int, float)) and w > 0:
                weight = float(w)
        points.append([p["lat"], p["lng"], weight])

    # 강도 정규화 (0~1)
    if points:
        max_w = max(p[2] for p in points) or 1.0
        points = [[p[0], p[1], p[2] / max_w] for p in points]

    return to_native({
        "results": {
            "points": points,
            "radius": req.radius,
            "blur": req.blur,
            "maxZoom": req.maxZoom,
            "gradient": req.gradient,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# 이변량 분석
# ══════════════════════════════════════════════════════════

class BivariateRequest(BaseModel):
    data: List[Dict[str, Any]]
    xColumn: str
    yColumn: str


# 3×3 Bivariate 색상 매트릭스 (x: low→high, y: low→high)
BIVARIATE_MATRIX = [
    ["#e8e8e8", "#ace4e4", "#5ac8c8"],   # y=low
    ["#dfb0d6", "#a5add3", "#5698b9"],   # y=mid
    ["#be64ac", "#8c62aa", "#3b4994"],   # y=high
]


@router.post("/api/analysis/bivariate")
def run_bivariate(req: BivariateRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]

    x_vals = [p.get(req.xColumn) for p in pts if isinstance(p.get(req.xColumn), (int, float))]
    y_vals = [p.get(req.yColumn) for p in pts if isinstance(p.get(req.yColumn), (int, float))]

    if not x_vals or not y_vals:
        raise HTTPException(400, "선택한 컬럼에 수치형 데이터가 없습니다.")

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_rng = (x_max - x_min) or 1.0
    y_rng = (y_max - y_min) or 1.0

    result_pts = []
    for p in pts:
        xv = p.get(req.xColumn)
        yv = p.get(req.yColumn)
        if not isinstance(xv, (int, float)) or not isinstance(yv, (int, float)):
            continue

        xi = min(int((xv - x_min) / x_rng * 3), 2)
        yi = min(int((yv - y_min) / y_rng * 3), 2)
        color = BIVARIATE_MATRIX[yi][xi]

        result_pts.append({
            "row": p,
            "xValue": xv,
            "yValue": yv,
            "xClass": xi,
            "yClass": yi,
            "color": color,
        })

    return to_native({
        "results": {
            "points": result_pts,
            "xColumn": req.xColumn,
            "yColumn": req.yColumn,
            "colorMatrix": BIVARIATE_MATRIX,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# 시계열 프레임 분리
# ══════════════════════════════════════════════════════════

class TimeSeriesRequest(BaseModel):
    data: List[Dict[str, Any]]
    dateCol: str
    valueCol: Optional[str] = None
    granularity: str = "month"   # "day" | "week" | "month" | "year"


def _truncate_date(date_str: str, gran: str) -> str:
    """날짜 문자열을 granularity로 자름"""
    try:
        import re
        # 날짜만 추출 (YYYY-MM-DD 형식 근사)
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(date_str))
        if not match:
            return str(date_str)[:7]
        y, m, d = match.groups()
        if gran == "year":
            return y
        elif gran == "month":
            return f"{y}-{m}"
        elif gran == "week":
            from datetime import date, timedelta
            dt = date(int(y), int(m), int(d))
            week_start = dt - timedelta(days=dt.weekday())
            return str(week_start)
        else:  # day
            return f"{y}-{m}-{d}"
    except Exception:
        return str(date_str)[:10]


@router.post("/api/analysis/timeseries")
def run_timeseries(req: TimeSeriesRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]

    frames_map: dict[str, list] = {}
    for p in pts:
        date_val = p.get(req.dateCol)
        if date_val is None:
            continue
        key = _truncate_date(str(date_val), req.granularity)
        frames_map.setdefault(key, []).append(p)

    frames = []
    for label in sorted(frames_map.keys()):
        members = frames_map[label]
        weights = None
        if req.valueCol:
            weights = [
                float(p[req.valueCol]) if isinstance(p.get(req.valueCol), (int, float)) else 1.0
                for p in members
            ]
        frames.append({
            "label": label,
            "points": [{"lat": p["lat"], "lng": p["lng"]} for p in members],
            "weights": weights,
            "count": len(members),
        })

    return to_native({"results": {"frames": frames}})


# ══════════════════════════════════════════════════════════
# 서버사이드 필터
# ══════════════════════════════════════════════════════════

class FilterCondition(BaseModel):
    col: str
    op: str        # "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "contains"
    value: Any


class FilterRequest(BaseModel):
    data: List[Dict[str, Any]]
    filters: List[FilterCondition]


def _apply_filter(row: dict, f: FilterCondition) -> bool:
    val = row.get(f.col)
    target = f.value
    try:
        if f.op == "eq":      return str(val) == str(target)
        if f.op == "neq":     return str(val) != str(target)
        if f.op == "gt":      return float(val) > float(target)
        if f.op == "gte":     return float(val) >= float(target)
        if f.op == "lt":      return float(val) < float(target)
        if f.op == "lte":     return float(val) <= float(target)
        if f.op == "contains":return str(target).lower() in str(val).lower()
    except (TypeError, ValueError):
        return False
    return True


@router.post("/api/analysis/filter")
def run_filter(req: FilterRequest):
    result = [
        r for r in req.data
        if all(_apply_filter(r, f) for f in req.filters)
    ]
    return to_native({"results": {"rows": result, "count": len(result)}})


# ══════════════════════════════════════════════════════════
# 거리 매트릭스 (전체 n×n)
# ══════════════════════════════════════════════════════════

class DistanceMatrixRequest(BaseModel):
    data: List[Dict[str, Any]]
    labelCol: Optional[str] = None
    maxPoints: int = 30


@router.post("/api/analysis/distancematrix")
def run_distance_matrix(req: DistanceMatrixRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")][:req.maxPoints]
    n = len(pts)

    labels = [
        str(p.get(req.labelCol, p.get("id", f"P{i}")))
        for i, p in enumerate(pts)
    ]

    matrix: list[list[float]] = []
    for i in range(n):
        row = []
        for j in range(n):
            dist = haversine(pts[i]["lat"], pts[i]["lng"], pts[j]["lat"], pts[j]["lng"]) if i != j else 0.0
            row.append(round(dist, 1))
        matrix.append(row)

    return to_native({
        "results": {
            "labels": labels,
            "matrix": matrix,
            "unit": "meters",
        }
    })
