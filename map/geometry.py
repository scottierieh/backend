"""
routers/geometry.py

POST /api/geometry/buffer         — Precise Shapely buffer polygons
POST /api/geometry/intersection   — Exact overlap area between buffers
POST /api/geometry/voronoi        — True Voronoi polygons (scipy + shapely)
POST /api/geometry/spatialjoin    — GeoDataFrame spatial join
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


def _latlon_to_meters(lat: float, lng: float, ref_lat: float, ref_lng: float):
    """Convert lat/lng offset to approximate meters (local projection)."""
    x = (lng - ref_lng) * math.cos(math.radians(ref_lat)) * 111320
    y = (lat - ref_lat) * 110540
    return x, y


def _meters_to_latlon(x: float, y: float, ref_lat: float, ref_lng: float):
    lat = ref_lat + y / 110540
    lng = ref_lng + x / (111320 * math.cos(math.radians(ref_lat)))
    return lat, lng


def _shapely_polygon_to_leaflet(poly, ref_lat: float, ref_lng: float) -> list:
    """Convert shapely polygon in local meters back to [[lat,lng],...]."""
    coords = list(poly.exterior.coords)
    result = []
    for x, y in coords:
        lat, lng = _meters_to_latlon(x, y, ref_lat, ref_lng)
        result.append([lat, lng])
    return result


# ══════════════════════════════════════════════════════════
# Precise buffer polygons (Shapely)
# ══════════════════════════════════════════════════════════

class PreciseBufferRequest(BaseMapRequest):
    radiusM: float = 500
    resolution: int = 32      # polygon approximation steps
    labelCol: Optional[str] = None


@router.post("/api/geometry/buffer")
def run_precise_buffer(req: PreciseBufferRequest):
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except ImportError:
        raise HTTPException(500, "shapely not installed")

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if not pts:
        raise HTTPException(400, "No valid points.")

    # Use mean center as local projection reference
    ref_lat = sum(p["lat"] for p in pts) / len(pts)
    ref_lng = sum(p["lng"] for p in pts) / len(pts)

    buffers = []
    shapely_polys = []
    for i, row in enumerate(pts):
        x, y = _latlon_to_meters(row["lat"], row["lng"], ref_lat, ref_lng)
        circle = Point(x, y).buffer(req.radiusM, resolution=req.resolution)
        shapely_polys.append(circle)
        polygon = _shapely_polygon_to_leaflet(circle, ref_lat, ref_lng)
        buffers.append({
            "center": {"lat": row["lat"], "lng": row["lng"]},
            "polygon": polygon,
            "radiusM": req.radiusM,
            "color": CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            "row": row,
            "label": str(row.get(req.labelCol, f"Point {i+1}")) if req.labelCol else f"Point {i+1}",
        })

    # Compute pairwise overlaps
    for i in range(len(shapely_polys)):
        overlaps = []
        for j in range(len(shapely_polys)):
            if i == j:
                continue
            if shapely_polys[i].intersects(shapely_polys[j]):
                overlap = shapely_polys[i].intersection(shapely_polys[j])
                overlap_area = overlap.area
                self_area = shapely_polys[i].area
                pct = round(overlap_area / self_area * 100, 1) if self_area > 0 else 0
                overlaps.append({
                    "withIdx": j,
                    "withLabel": buffers[j]["label"],
                    "overlapPct": pct,
                    "overlapAreaM2": round(overlap_area, 1),
                })
        buffers[i]["overlaps"] = overlaps
        buffers[i]["overlappingWith"] = len(overlaps)

    return to_native({"results": {"buffers": buffers, "visible": True}})


# ══════════════════════════════════════════════════════════
# Area intersection — exact overlap matrix
# ══════════════════════════════════════════════════════════

class IntersectionRequest(BaseMapRequest):
    radiusM: float = 500
    labelCol: Optional[str] = None


@router.post("/api/geometry/intersection")
def run_intersection(req: IntersectionRequest):
    try:
        from shapely.geometry import Point
    except ImportError:
        raise HTTPException(500, "shapely not installed")

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    if not pts:
        raise HTTPException(400, "No valid points.")

    ref_lat = sum(p["lat"] for p in pts) / len(pts)
    ref_lng = sum(p["lng"] for p in pts) / len(pts)

    labels = [str(p.get(req.labelCol, f"P{i+1}")) if req.labelCol else f"P{i+1}" for i, p in enumerate(pts)]

    polys = []
    for row in pts:
        x, y = _latlon_to_meters(row["lat"], row["lng"], ref_lat, ref_lng)
        polys.append(Point(x, y).buffer(req.radiusM, resolution=32))

    # Build overlap matrix
    n = len(pts)
    matrix = [[0.0] * n for _ in range(n)]
    pairs = []
    for i in range(n):
        matrix[i][i] = 100.0
        for j in range(i + 1, n):
            if polys[i].intersects(polys[j]):
                inter_area = polys[i].intersection(polys[j]).area
                pct_i = inter_area / polys[i].area * 100
                pct_j = inter_area / polys[j].area * 100
                avg_pct = (pct_i + pct_j) / 2
                matrix[i][j] = round(pct_i, 1)
                matrix[j][i] = round(pct_j, 1)
                severity = "critical" if avg_pct >= 60 else "high" if avg_pct >= 40 else "medium" if avg_pct >= 20 else "low"
                pairs.append({
                    "storeA": {**pts[i], "label": labels[i]},
                    "storeB": {**pts[j], "label": labels[j]},
                    "overlapPctA": round(pct_i, 1),
                    "overlapPctB": round(pct_j, 1),
                    "overlapAvgPct": round(avg_pct, 1),
                    "overlapAreaM2": round(inter_area, 1),
                    "severity": severity,
                    "distance": round(haversine(pts[i]["lat"], pts[i]["lng"], pts[j]["lat"], pts[j]["lng"]), 1),
                })

    pairs.sort(key=lambda x: -x["overlapAvgPct"])

    # Build circles for map display
    circles = []
    for i, row in enumerate(pts):
        poly = polys[i]
        polygon = _shapely_polygon_to_leaflet(poly, ref_lat, ref_lng)
        circles.append({
            "center": {"lat": row["lat"], "lng": row["lng"]},
            "radiusM": req.radiusM,
            "color": CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            "label": labels[i],
            "polygon": polygon,
        })

    return to_native({
        "results": {
            "circles": circles,
            "pairs": pairs,
            "matrix": matrix,
            "labels": labels,
            "nConflicts": len(pairs),
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# True Voronoi polygons (scipy Delaunay + shapely)
# ══════════════════════════════════════════════════════════

class TrueVoronoiRequest(BaseMapRequest):
    centers: List[Dict[str, Any]]   # {id, label, lat, lng}


@router.post("/api/geometry/voronoi")
def run_true_voronoi(req: TrueVoronoiRequest):
    try:
        from scipy.spatial import Voronoi
        from shapely.geometry import Polygon, MultiPolygon, box
        from shapely.ops import unary_union
    except ImportError:
        raise HTTPException(500, "scipy/shapely not installed")

    pts_data = [r for r in req.data if r.get("lat") and r.get("lng")]
    centers = [c for c in req.centers if c.get("lat") and c.get("lng")]
    if len(centers) < 2:
        raise HTTPException(400, "Need at least 2 centers for Voronoi.")

    ref_lat = sum(c["lat"] for c in centers) / len(centers)
    ref_lng = sum(c["lng"] for c in centers) / len(centers)

    # Project to local meters
    center_xy = [_latlon_to_meters(c["lat"], c["lng"], ref_lat, ref_lng) for c in centers]

    # Add far-away mirror points to close open Voronoi regions
    far = 1_000_000
    mirror_points = []
    for x, y in center_xy:
        mirror_points += [(x + far, y), (x - far, y), (x, y + far), (x, y - far)]

    all_points = center_xy + mirror_points
    vor = Voronoi(all_points)

    # Bounding box from data points
    all_lats = [p["lat"] for p in pts_data] + [c["lat"] for c in centers]
    all_lngs = [p["lng"] for p in pts_data] + [c["lng"] for c in centers]
    margin = 0.05 * max(max(all_lats) - min(all_lats), max(all_lngs) - min(all_lngs))
    bbox_x1, bbox_y1 = _latlon_to_meters(min(all_lats) - margin, min(all_lngs) - margin, ref_lat, ref_lng)
    bbox_x2, bbox_y2 = _latlon_to_meters(max(all_lats) + margin, max(all_lngs) + margin, ref_lat, ref_lng)
    bounding_box = box(bbox_x1, bbox_y1, bbox_x2, bbox_y2)

    VORONOI_COLORS = [
        {"fill": "rgba(59,130,246,0.12)", "border": "#3b82f6"},
        {"fill": "rgba(239,68,68,0.12)",  "border": "#ef4444"},
        {"fill": "rgba(34,197,94,0.12)",  "border": "#22c55e"},
        {"fill": "rgba(168,85,247,0.12)", "border": "#a855f7"},
        {"fill": "rgba(249,115,22,0.12)", "border": "#f97316"},
        {"fill": "rgba(236,72,153,0.12)", "border": "#ec4899"},
        {"fill": "rgba(20,184,166,0.12)", "border": "#14b8a6"},
        {"fill": "rgba(234,179,8,0.12)",  "border": "#eab308"},
    ]

    result_cells = []
    for i, center in enumerate(centers):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or not region:
            continue

        vertices = [vor.vertices[v] for v in region]
        try:
            poly = Polygon(vertices).intersection(bounding_box)
        except Exception:
            continue

        if poly.is_empty:
            continue

        # Convert back to lat/lng
        if isinstance(poly, Polygon):
            polygon_latlon = [list(_meters_to_latlon(x, y, ref_lat, ref_lng)) for x, y in poly.exterior.coords]
        else:
            # MultiPolygon — take largest
            largest = max(poly.geoms, key=lambda g: g.area)
            polygon_latlon = [list(_meters_to_latlon(x, y, ref_lat, ref_lng)) for x, y in largest.exterior.coords]

        # Points inside this cell
        members = []
        for pt in pts_data:
            px, py = _latlon_to_meters(pt["lat"], pt["lng"], ref_lat, ref_lng)
            if poly.contains(__import__("shapely.geometry", fromlist=["Point"]).Point(px, py)):
                members.append(pt)

        color = VORONOI_COLORS[i % len(VORONOI_COLORS)]
        result_cells.append({
            "id": i,
            "label": center.get("label", f"Zone {i+1}"),
            "center": {"lat": center["lat"], "lng": center["lng"]},
            "polygon": polygon_latlon,
            "fill": color["fill"],
            "border": color["border"],
            "colorIdx": i,
            "pointCount": len(members),
            "points": members,
        })

    # Build enriched centers + colors for LeafletMap compatibility
    enriched_centers = [{**c, "colorIdx": i, "point": {"lat": c["lat"], "lng": c["lng"]}} for i, c in enumerate(centers)]
    assignments_map = {str(i): cell["points"] for i, cell in enumerate(result_cells)}

    return to_native({
        "results": {
            "cells": result_cells,
            "centers": enriched_centers,
            "assignments": assignments_map,
            "colors": VORONOI_COLORS,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# GeoDataFrame Spatial Join
# ══════════════════════════════════════════════════════════

class GeoSpatialJoinRequest(BaseModel):
    points: List[Dict[str, Any]]
    areas: List[Dict[str, Any]]   # [{id, label, polygon: [[lat,lng],...], color?}]
    aggregateCols: Optional[List[str]] = None


@router.post("/api/geometry/spatialjoin")
def run_geo_spatialjoin(req: GeoSpatialJoinRequest):
    try:
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
    except ImportError:
        raise HTTPException(500, "geopandas/shapely not installed")

    import statistics

    pts = [r for r in req.points if r.get("lat") and r.get("lng")]
    if not pts or not req.areas:
        raise HTTPException(400, "Need points and areas.")

    # Build GeoDataFrames
    pt_geoms = [Point(p["lng"], p["lat"]) for p in pts]
    gdf_pts = gpd.GeoDataFrame(pts, geometry=pt_geoms, crs="EPSG:4326")

    area_geoms = []
    valid_areas = []
    for area in req.areas:
        poly_coords = area.get("polygon", [])
        if len(poly_coords) >= 3:
            try:
                shapely_poly = Polygon([(c[1], c[0]) for c in poly_coords])  # lng, lat
                area_geoms.append(shapely_poly)
                valid_areas.append(area)
            except Exception:
                pass

    if not valid_areas:
        raise HTTPException(400, "No valid area polygons.")

    gdf_areas = gpd.GeoDataFrame(valid_areas, geometry=area_geoms, crs="EPSG:4326")
    joined = gpd.sjoin(gdf_pts, gdf_areas, how="left", predicate="within")

    results = []
    for area in valid_areas:
        area_id = area.get("id")
        mask = joined["id"] == area_id
        members_df = joined[mask]
        members = [pts[i] for i in members_df.index if i < len(pts)]

        agg: dict[str, Any] = {"count": len(members)}
        if req.aggregateCols:
            for col in req.aggregateCols:
                vals = [m[col] for m in members if isinstance(m.get(col), (int, float))]
                if vals:
                    agg[f"{col}_sum"] = round(sum(vals), 2)
                    agg[f"{col}_mean"] = round(statistics.mean(vals), 2)
                    agg[f"{col}_min"] = round(min(vals), 2)
                    agg[f"{col}_max"] = round(max(vals), 2)

        results.append({
            "id": area_id,
            "label": area.get("label", str(area_id)),
            "polygon": area.get("polygon", []),
            "color": area.get("color", "#3b82f6"),
            "pointCount": len(members),
            "center": centroid(members) if members else {"lat": 0, "lng": 0},
            "aggregations": agg,
        })

    return to_native({"results": {"areas": results, "visible": True}})
