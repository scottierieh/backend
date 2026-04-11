"""
utils.py
공통 유틸리티: 타입 변환, 지리 수학 함수
"""

import math
import numpy as np
from typing import Any


# ══════════════════════════════════════════════════════════
# 직렬화
# ══════════════════════════════════════════════════════════

def to_native(obj: Any) -> Any:
    """numpy/pandas 타입 → Python 기본 타입 (JSON 직렬화용)"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    return obj


# ══════════════════════════════════════════════════════════
# 지리 수학
# ══════════════════════════════════════════════════════════

EARTH_RADIUS_M = 6_371_000  # 미터


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 좌표 간 거리 (미터, Haversine 공식)"""
    r = EARTH_RADIUS_M
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def centroid(points: list[dict]) -> dict:
    """포인트 리스트의 무게중심 반환 {'lat': ..., 'lng': ...}"""
    if not points:
        return {"lat": 0.0, "lng": 0.0}
    return {
        "lat": sum(p["lat"] for p in points) / len(points),
        "lng": sum(p["lng"] for p in points) / len(points),
    }


def destination(lat: float, lng: float, bearing_deg: float, distance_m: float) -> dict:
    """
    시작 좌표에서 방위각(degrees) + 거리(m)만큼 이동한 좌표 반환.
    버퍼 원 근사, 이소크론 근사에 사용.
    """
    R = EARTH_RADIUS_M
    d = distance_m / R
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lng2 = lng1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return {"lat": math.degrees(lat2), "lng": math.degrees(lng2)}


def circle_polygon(lat: float, lng: float, radius_m: float, steps: int = 64) -> list[list[float]]:
    """
    원을 다각형으로 근사 (Leaflet Polygon 용).
    반환: [[lat, lng], ...] (닫힌 링)
    """
    pts = []
    for i in range(steps):
        bearing = 360 * i / steps
        p = destination(lat, lng, bearing, radius_m)
        pts.append([p["lat"], p["lng"]])
    pts.append(pts[0])  # 닫기
    return pts


def convex_hull(points: list[dict]) -> list[dict]:
    """
    Graham scan 볼록 껍질.
    입력: [{"lat": ..., "lng": ...}, ...]
    반환: 같은 형식의 hull 꼭짓점 리스트
    """
    pts = [(p["lng"], p["lat"]) for p in points]
    pts = list(set(pts))
    if len(pts) < 3:
        return [{"lat": p[1], "lng": p[0]} for p in pts]

    pts.sort()

    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [{"lat": p[1], "lng": p[0]} for p in hull]


def polygon_area_km2(hull: list[dict]) -> float:
    """Shoelace 공식으로 다각형 넓이 추정 (km²)"""
    if len(hull) < 3:
        return 0.0
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        # 위도/경도를 미터로 근사 변환 후 넓이
        x1 = math.radians(hull[i]["lng"]) * EARTH_RADIUS_M * math.cos(math.radians(hull[i]["lat"]))
        y1 = math.radians(hull[i]["lat"]) * EARTH_RADIUS_M
        x2 = math.radians(hull[j]["lng"]) * EARTH_RADIUS_M * math.cos(math.radians(hull[j]["lat"]))
        y2 = math.radians(hull[j]["lat"]) * EARTH_RADIUS_M
        area += x1 * y2 - x2 * y1
    return abs(area) / 2 / 1_000_000  # m² → km²
