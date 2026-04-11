"""
routers/route.py
경로 & 네트워크 분석 엔드포인트

POST /api/route/directions   — OSRM 경로 계산 프록시
POST /api/route/tsp          — 외판원 문제 (Nearest Neighbor 근사)
POST /api/route/odmatrix     — Origin-Destination 거리 매트릭스
POST /api/route/flowmap      — Flow Map (연결선 + 곡선)
POST /api/route/spider       — Spider Map (한 센터 → 모든 포인트)
POST /api/route/nearestfacility — 가장 가까운 시설 매칭
"""

import math
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import haversine, to_native

router = APIRouter()

OSRM_BASE = "http://router.project-osrm.org"


# ══════════════════════════════════════════════════════════
# OSRM 프록시
# ══════════════════════════════════════════════════════════

class DirectionsRequest(BaseModel):
    waypoints: List[Dict[str, float]]   # [{lat, lng}, ...]
    profile: str = "driving"            # driving | walking | cycling


@router.post("/api/route/directions")
async def get_directions(req: DirectionsRequest):
    if len(req.waypoints) < 2:
        raise HTTPException(400, "최소 2개의 경유지가 필요합니다.")

    coords = ";".join(f"{w['lng']},{w['lat']}" for w in req.waypoints)
    url = f"{OSRM_BASE}/route/v1/{req.profile}/{coords}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "OSRM 서버 응답 시간 초과")
    except Exception as e:
        raise HTTPException(502, f"OSRM 오류: {str(e)}")

    if data.get("code") != "Ok":
        raise HTTPException(400, f"경로를 찾을 수 없습니다: {data.get('message', '')}")

    route = data["routes"][0]
    coords_list = route["geometry"]["coordinates"]   # [[lng, lat], ...]
    path = [{"lat": c[1], "lng": c[0]} for c in coords_list]

    return {
        "path": path,
        "distanceKm": route["distance"] / 1000,
        "durationMin": route["duration"] / 60,
    }


# ══════════════════════════════════════════════════════════
# TSP — Nearest Neighbor + 2-opt
# ══════════════════════════════════════════════════════════

class TSPRequest(BaseModel):
    data: List[Dict[str, Any]]
    returnToStart: bool = True
    useOSRM: bool = False


def _nn_tour(pts: list) -> list[int]:
    """Nearest Neighbor 휴리스틱"""
    n = len(pts)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best = min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: haversine(pts[last]["lat"], pts[last]["lng"],
                                    pts[j]["lat"],  pts[j]["lng"])
        )
        tour.append(best)
        visited[best] = True
    return tour


def _two_opt(pts: list, tour: list[int]) -> list[int]:
    """2-opt 개선"""
    def tour_dist(t):
        return sum(
            haversine(pts[t[i]]["lat"], pts[t[i]]["lng"],
                      pts[t[(i+1) % len(t)]]["lat"], pts[t[(i+1) % len(t)]]["lng"])
            for i in range(len(t))
        )

    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                if tour_dist(new_tour) < tour_dist(tour) - 0.01:
                    tour = new_tour
                    improved = True
    return tour


@router.post("/api/route/tsp")
async def run_tsp(req: TSPRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < 2:
        raise HTTPException(400, "최소 2개 포인트가 필요합니다.")

    tour_idx = _nn_tour(pts)
    if len(pts) <= 12:   # 작은 경우만 2-opt (큰 경우 너무 느림)
        tour_idx = _two_opt(pts, tour_idx)

    ordered = [pts[i] for i in tour_idx]
    if req.returnToStart:
        ordered.append(ordered[0])

    total_dist = sum(
        haversine(ordered[i]["lat"], ordered[i]["lng"],
                  ordered[i+1]["lat"], ordered[i+1]["lng"])
        for i in range(len(ordered) - 1)
    )

    segments = [
        {
            "from": ordered[i],
            "to": ordered[i+1],
            "distanceM": haversine(ordered[i]["lat"], ordered[i]["lng"],
                                   ordered[i+1]["lat"], ordered[i+1]["lng"]),
        }
        for i in range(len(ordered) - 1)
    ]

    route_geometry = None
    if req.useOSRM and len(pts) <= 20:
        try:
            waypoints = ordered[:-1] if req.returnToStart else ordered
            coords = ";".join(f"{w['lng']},{w['lat']}" for w in waypoints)
            if req.returnToStart:
                coords += f";{ordered[0]['lng']},{ordered[0]['lat']}"
            url = f"{OSRM_BASE}/route/v1/driving/{coords}?overview=full&geometries=geojson"
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url)
                r.raise_for_status()
                geo = r.json()
            if geo.get("code") == "Ok":
                c_list = geo["routes"][0]["geometry"]["coordinates"]
                route_geometry = [[c[1], c[0]] for c in c_list]
        except Exception:
            pass  # OSRM 실패 시 직선으로 fallback

    return to_native({
        "results": {
            "tour": ordered,
            "segments": segments,
            "totalDistanceM": total_dist,
            "totalDistanceKm": total_dist / 1000,
            "returnToStart": req.returnToStart,
            "routeGeometry": route_geometry,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# OD Matrix
# ══════════════════════════════════════════════════════════

class ODMatrixRequest(BaseModel):
    origins: List[Dict[str, Any]]
    destinations: Optional[List[Dict[str, Any]]] = None   # None → origins == destinations
    maxLines: int = 200


@router.post("/api/route/odmatrix")
def run_od_matrix(req: ODMatrixRequest):
    origins = [r for r in req.origins if r.get("lat") and r.get("lng")]
    dests = (
        [r for r in req.destinations if r.get("lat") and r.get("lng")]
        if req.destinations else origins
    )

    lines = []
    for o in origins:
        for d in dests:
            if o is d:
                continue
            dist = haversine(o["lat"], o["lng"], d["lat"], d["lng"])
            lines.append({"from": o, "to": d, "distanceM": dist})

    lines.sort(key=lambda l: l["distanceM"])
    lines = lines[:req.maxLines]

    max_dist = max((l["distanceM"] for l in lines), default=1.0)
    for l in lines:
        l["colorValue"] = l["distanceM"] / max_dist

    return to_native({
        "results": {
            "lines": lines,
            "origins": origins,
            "destinations": dests,
            "maxValue": max_dist,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Flow Map
# ══════════════════════════════════════════════════════════

class FlowRequest(BaseModel):
    flows: List[Dict[str, Any]]   # [{fromId, toId, value}]
    data: List[Dict[str, Any]]
    idCol: str
    maxFlows: int = 50


def _bezier_points(p1: dict, p2: dict, steps: int = 20) -> list[list[float]]:
    """단순 곡선 근사 (중점을 살짝 위로 올려서 arc 효과)"""
    mid_lat = (p1["lat"] + p2["lat"]) / 2
    mid_lng = (p1["lng"] + p2["lng"]) / 2
    # 수직 오프셋
    dlat = p2["lat"] - p1["lat"]
    dlng = p2["lng"] - p1["lng"]
    dist_deg = math.sqrt(dlat**2 + dlng**2)
    ctrl_lat = mid_lat - dlng * 0.2
    ctrl_lng = mid_lng + dlat * 0.2

    pts = []
    for i in range(steps + 1):
        t = i / steps
        lat = (1-t)**2 * p1["lat"] + 2*(1-t)*t * ctrl_lat + t**2 * p2["lat"]
        lng = (1-t)**2 * p1["lng"] + 2*(1-t)*t * ctrl_lng + t**2 * p2["lng"]
        pts.append([lat, lng])
    return pts


@router.post("/api/route/flowmap")
def run_flow_map(req: FlowRequest):
    id_map = {str(r.get(req.idCol)): r for r in req.data if r.get("lat") and r.get("lng")}

    flows_out = []
    for f in req.flows[:req.maxFlows]:
        from_row = id_map.get(str(f.get("fromId")))
        to_row   = id_map.get(str(f.get("toId")))
        if not from_row or not to_row:
            continue
        dist = haversine(from_row["lat"], from_row["lng"], to_row["lat"], to_row["lng"])
        flows_out.append({
            "from": from_row,
            "to": to_row,
            "value": f.get("value", 1),
            "distanceM": dist,
            "curvePoints": _bezier_points(
                {"lat": from_row["lat"], "lng": from_row["lng"]},
                {"lat": to_row["lat"], "lng": to_row["lng"]},
            ),
        })

    max_val = max((f["value"] for f in flows_out), default=1)
    return to_native({
        "results": {
            "flows": flows_out,
            "maxValue": max_val,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Spider Map
# ══════════════════════════════════════════════════════════

class SpiderRequest(BaseModel):
    center: Dict[str, Any]
    data: List[Dict[str, Any]]
    maxLines: int = 100
    valueCol: Optional[str] = None


@router.post("/api/route/spider")
def run_spider(req: SpiderRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    center = req.center

    lines = []
    for p in pts[:req.maxLines]:
        dist = haversine(center["lat"], center["lng"], p["lat"], p["lng"])
        value = p.get(req.valueCol, dist) if req.valueCol else dist
        lines.append({
            "from": center,
            "to": p,
            "distanceM": dist,
            "value": value,
            "colorValue": 0.0,  # 정규화 후 채움
        })

    if lines:
        max_v = max(l["value"] for l in lines) or 1.0
        for l in lines:
            l["colorValue"] = l["value"] / max_v

    return to_native({
        "results": {
            "center": center,
            "lines": lines,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Nearest Facility
# ══════════════════════════════════════════════════════════

class NearestFacilityRequest(BaseModel):
    sources: List[Dict[str, Any]]
    facilities: List[Dict[str, Any]]


@router.post("/api/route/nearestfacility")
def run_nearest_facility(req: NearestFacilityRequest):
    sources    = [r for r in req.sources    if r.get("lat") and r.get("lng")]
    facilities = [r for r in req.facilities if r.get("lat") and r.get("lng")]

    if not facilities:
        raise HTTPException(400, "시설(facilities) 데이터가 없습니다.")

    matches = []
    for src in sources:
        nearest = min(
            facilities,
            key=lambda f: haversine(src["lat"], src["lng"], f["lat"], f["lng"])
        )
        dist = haversine(src["lat"], src["lng"], nearest["lat"], nearest["lng"])
        matches.append({"source": src, "facility": nearest, "distanceM": dist})

    matches.sort(key=lambda m: m["distanceM"])
    return to_native({
        "results": {
            "matches": matches,
            "avgDistanceM": sum(m["distanceM"] for m in matches) / len(matches) if matches else 0,
            "visible": True,
        }
    })
