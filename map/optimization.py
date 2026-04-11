"""
routers/optimization.py

POST /api/optimization/vrp           — Vehicle Routing Problem (OR-Tools)
POST /api/optimization/pmedian       — P-Median optimal facility placement
POST /api/optimization/mclp          — Max Coverage Location Problem
POST /api/optimization/territory     — Territory design (balanced zone assignment)
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
]


class BaseMapRequest(BaseModel):
    data: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════
# VRP — Vehicle Routing Problem
# ══════════════════════════════════════════════════════════

class VRPRequest(BaseModel):
    depot: Dict[str, float]          # {lat, lng} — start/end point
    data: List[Dict[str, Any]]       # delivery points
    nVehicles: int = 3
    maxDistanceM: Optional[float] = None   # per-vehicle max distance


@router.post("/api/optimization/vrp")
def run_vrp(req: VRPRequest):
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        raise HTTPException(500, "ortools not installed")

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if not pts:
        raise HTTPException(400, "No valid points.")

    # Build distance matrix (depot = index 0)
    all_points = [req.depot] + pts
    n = len(all_points)
    dist_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            d = haversine(all_points[i]["lat"], all_points[i]["lng"],
                          all_points[j]["lat"], all_points[j]["lng"])
            row.append(int(d))
        dist_matrix.append(row)

    manager = pywrapcp.RoutingIndexManager(n, req.nVehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        return dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Max distance per vehicle
    if req.maxDistanceM:
        dim_name = "Distance"
        routing.AddDimension(transit_cb, 0, int(req.maxDistanceM), True, dim_name)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 5

    solution = routing.SolveWithParameters(params)
    if not solution:
        raise HTTPException(400, "No VRP solution found.")

    routes = []
    total_dist = 0
    for v in range(req.nVehicles):
        idx = routing.Start(v)
        route_pts = []
        route_dist = 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_pts.append(all_points[node])
            next_idx = solution.Value(routing.NextVar(idx))
            route_dist += dist_matrix[manager.IndexToNode(idx)][manager.IndexToNode(next_idx)]
            idx = next_idx
        route_pts.append(all_points[0])  # return to depot
        if len(route_pts) > 2:  # skip empty routes
            routes.append({
                "vehicleId": v,
                "color": CLUSTER_COLORS[v % len(CLUSTER_COLORS)],
                "points": route_pts,
                "distanceM": route_dist,
                "distanceKm": round(route_dist / 1000, 2),
                "stops": len(route_pts) - 2,
            })
        total_dist += route_dist

    return to_native({
        "results": {
            "routes": routes,
            "depot": req.depot,
            "totalDistanceM": total_dist,
            "totalDistanceKm": round(total_dist / 1000, 2),
            "nVehicles": len(routes),
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# P-Median — Optimal facility placement
# ══════════════════════════════════════════════════════════

class PMedianRequest(BaseMapRequest):
    p: int = 3                        # number of facilities
    demandCol: Optional[str] = None   # demand weight column
    maxIterations: int = 100


@router.post("/api/optimization/pmedian")
def run_pmedian(req: PMedianRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < req.p:
        raise HTTPException(400, f"Need at least {req.p} points.")

    # Demand weights
    demands = []
    for p in pts:
        w = p.get(req.demandCol) if req.demandCol else None
        demands.append(float(w) if isinstance(w, (int, float)) and w > 0 else 1.0)

    n = len(pts)
    # Distance matrix
    dist = [[haversine(pts[i]["lat"], pts[i]["lng"],
                       pts[j]["lat"], pts[j]["lng"])
             for j in range(n)] for i in range(n)]

    # Greedy init: pick p points that minimize weighted distance
    import random
    random.seed(42)

    def total_cost(facility_ids):
        cost = 0.0
        for i in range(n):
            min_d = min(dist[i][f] for f in facility_ids)
            cost += demands[i] * min_d
        return cost

    # Greedy initialization
    facilities = set()
    remaining = list(range(n))
    for _ in range(req.p):
        best, best_cost = None, float("inf")
        for candidate in remaining:
            trial = facilities | {candidate}
            c = total_cost(trial)
            if c < best_cost:
                best, best_cost = candidate, c
        facilities.add(best)
        remaining.remove(best)

    # Local search
    improved = True
    iterations = 0
    while improved and iterations < req.maxIterations:
        improved = False
        iterations += 1
        for f in list(facilities):
            for candidate in range(n):
                if candidate in facilities:
                    continue
                trial = (facilities - {f}) | {candidate}
                if total_cost(trial) < total_cost(facilities):
                    facilities = trial
                    improved = True
                    break
            if improved:
                break

    # Assign each demand point to nearest facility
    facility_list = sorted(facilities)
    assignments: dict[int, list] = {f: [] for f in facility_list}
    for i in range(n):
        nearest = min(facility_list, key=lambda f: dist[i][f])
        assignments[nearest].append(pts[i])

    result_facilities = []
    for idx, f in enumerate(facility_list):
        members = assignments[f]
        result_facilities.append({
            "id": idx,
            "location": pts[f],
            "color": CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
            "assignedCount": len(members),
            "assignedPoints": members,
            "totalWeightedDist": sum(demands[i] * dist[i][f]
                                     for i in range(n) if pts[i] in members),
        })

    return to_native({
        "results": {
            "facilities": result_facilities,
            "totalCost": total_cost(facilities),
            "p": req.p,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# MCLP — Maximum Coverage Location Problem
# ══════════════════════════════════════════════════════════

class MCLPRequest(BaseMapRequest):
    p: int = 3             # number of facilities
    coverageM: float = 1000   # coverage radius in meters
    demandCol: Optional[str] = None


@router.post("/api/optimization/mclp")
def run_mclp(req: MCLPRequest):
    try:
        from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, PULP_CBC_CMD
    except ImportError:
        raise HTTPException(500, "pulp not installed")

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < req.p:
        raise HTTPException(400, f"Need at least {req.p} points.")

    n = len(pts)
    demands = []
    for p in pts:
        w = p.get(req.demandCol) if req.demandCol else None
        demands.append(float(w) if isinstance(w, (int, float)) and w > 0 else 1.0)

    # Coverage matrix
    coverage = [[
        haversine(pts[i]["lat"], pts[i]["lng"],
                  pts[j]["lat"], pts[j]["lng"]) <= req.coverageM
        for j in range(n)] for i in range(n)]

    prob = LpProblem("MCLP", LpMaximize)
    x = [LpVariable(f"x_{j}", cat="Binary") for j in range(n)]  # facility at j?
    y = [LpVariable(f"y_{i}", cat="Binary") for i in range(n)]  # demand i covered?

    # Objective: maximize covered demand
    prob += lpSum(demands[i] * y[i] for i in range(n))

    # Exactly p facilities
    prob += lpSum(x) == req.p

    # Demand covered only if at least one covering facility selected
    for i in range(n):
        covering = [j for j in range(n) if coverage[i][j]]
        if covering:
            prob += y[i] <= lpSum(x[j] for j in covering)
        else:
            prob += y[i] == 0

    prob.solve(PULP_CBC_CMD(msg=0))

    selected = [j for j in range(n) if value(x[j]) and value(x[j]) > 0.5]
    covered = [i for i in range(n) if value(y[i]) and value(y[i]) > 0.5]
    uncovered = [i for i in range(n) if i not in covered]

    facilities = []
    for idx, j in enumerate(selected):
        assigned = [i for i in range(n) if coverage[i][j] and i in covered]
        facilities.append({
            "id": idx,
            "location": pts[j],
            "color": CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
            "coverageM": req.coverageM,
            "coveredCount": len(assigned),
            "coveredPoints": [pts[i] for i in assigned],
        })

    total_demand = sum(demands)
    covered_demand = sum(demands[i] for i in covered)

    return to_native({
        "results": {
            "facilities": facilities,
            "coveredPoints": [pts[i] for i in covered],
            "uncoveredPoints": [pts[i] for i in uncovered],
            "coverageRate": round(covered_demand / total_demand * 100, 1) if total_demand else 0,
            "coveredCount": len(covered),
            "uncoveredCount": len(uncovered),
            "p": req.p,
            "coverageM": req.coverageM,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Territory Design — balanced zone assignment
# ══════════════════════════════════════════════════════════

class TerritoryRequest(BaseMapRequest):
    nTerritories: int = 4
    balanceCol: Optional[str] = None    # metric to balance (e.g. revenue, visits)


@router.post("/api/optimization/territory")
def run_territory(req: TerritoryRequest):
    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if len(pts) < req.nTerritories:
        raise HTTPException(400, f"Need at least {req.nTerritories} points.")

    n = req.nTerritories
    workloads = []
    for p in pts:
        w = p.get(req.balanceCol) if req.balanceCol else None
        workloads.append(float(w) if isinstance(w, (int, float)) and w > 0 else 1.0)

    total_work = sum(workloads)
    target = total_work / n

    # K-Means on lat/lng for spatial balance, then rebalance workload
    from sklearn.cluster import KMeans
    import numpy as np

    coords = np.array([[p["lat"], p["lng"]] for p in pts])
    km = KMeans(n_clusters=n, n_init=20, random_state=42)
    labels = km.fit_predict(coords)

    # Iterative rebalancing: move border points to equalize workload
    assignments = list(labels)
    for _ in range(50):
        zone_work = [0.0] * n
        for i, z in enumerate(assignments):
            zone_work[z] += workloads[i]

        changed = False
        for i in range(len(pts)):
            current = assignments[i]
            # find nearest other zone center
            center_lat = np.mean([pts[j]["lat"] for j in range(len(pts)) if assignments[j] == current])
            center_lng = np.mean([pts[j]["lng"] for j in range(len(pts)) if assignments[j] == current])

            for z in range(n):
                if z == current:
                    continue
                # move if this zone is overloaded and target is under
                if zone_work[current] > target * 1.15 and zone_work[z] < target * 0.85:
                    # check proximity to zone z
                    z_pts = [pts[j] for j in range(len(pts)) if assignments[j] == z]
                    if z_pts:
                        z_center = centroid(z_pts)
                        dist_current = haversine(pts[i]["lat"], pts[i]["lng"], center_lat, center_lng)
                        dist_z = haversine(pts[i]["lat"], pts[i]["lng"], z_center["lat"], z_center["lng"])
                        if dist_z < dist_current * 1.5:
                            assignments[i] = z
                            zone_work[current] -= workloads[i]
                            zone_work[z] += workloads[i]
                            changed = True
                            break
        if not changed:
            break

    # Build result
    territories = []
    for z in range(n):
        members = [pts[i] for i, a in enumerate(assignments) if a == z]
        zone_workload = sum(workloads[i] for i, a in enumerate(assignments) if a == z)
        territories.append({
            "id": z,
            "color": CLUSTER_COLORS[z % len(CLUSTER_COLORS)],
            "center": centroid(members) if members else {"lat": 0, "lng": 0},
            "count": len(members),
            "workload": round(zone_workload, 2),
            "workloadPct": round(zone_workload / total_work * 100, 1) if total_work else 0,
            "targetPct": round(100 / n, 1),
            "balance": round(abs(zone_workload - target) / target * 100, 1),  # % deviation
            "points": members,
            "visible": True,
        })

    return to_native({
        "results": {
            "territories": territories,
            "nTerritories": n,
            "totalWorkload": round(total_work, 2),
            "targetPerTerritory": round(target, 2),
            "visible": True,
        }
    })
