"""
P-Median Problem FastAPI Endpoint using Spopt
Professional spatial optimization with PySAL ecosystem
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings

# Optimization solver
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    warnings.warn("PuLP not installed. Install with: pip install pulp")

# Spopt imports
try:
    from spopt.locate import PMedian
    from spopt.locate import PMedian
    SPOPT_AVAILABLE = True
except ImportError:
    SPOPT_AVAILABLE = False
    warnings.warn("Spopt not installed. Install with: pip install spopt")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

router = APIRouter()


class Location(BaseModel):
    """Location point with coordinates and demand"""
    name: str
    x: float
    y: float
    demand: float = 1.0


class PMedianRequest(BaseModel):
    """Request model for P-Median problem"""
    locations: List[Location] = Field(..., min_items=2)
    p: int = Field(..., gt=0, description="Number of facilities to locate")
    distance_type: str = Field(default="euclidean", pattern="^(euclidean|manhattan)$")
    crs: Optional[str] = Field(
        default=None,
        description="Coordinate Reference System. If provided, coordinates will be projected for accurate distance calculation. Examples: 'EPSG:5179' (Korea), 'EPSG:3857' (Web Mercator), 'EPSG:32633' (UTM Zone 33N)"
    )
    input_crs: str = Field(
        default="EPSG:4326",
        description="Input coordinate system. Default is EPSG:4326 (WGS84 lat/lon)"
    )


class PMedianResponse(BaseModel):
    """Response model for P-Median problem"""
    success: bool
    total_distance: float
    avg_distance: float
    max_distance: float
    selected_facilities: List[str]
    facility_indices: List[int]
    assignments: Dict[str, str]
    facility_loads: Dict[str, int]
    problem: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]
    solver_info: Dict[str, Any]


def create_geodataframe(
    locations: List[Location],
    input_crs: str = "EPSG:4326",
    target_crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Convert locations to GeoDataFrame with proper CRS handling
    
    Args:
        locations: List of location points
        input_crs: Input coordinate reference system (default: EPSG:4326 - WGS84)
        target_crs: Target CRS for projection. If None, uses input_crs.
                   For accurate distance calculation, use projected CRS:
                   - EPSG:5179 (Korea)
                   - EPSG:3857 (Web Mercator)
                   - EPSG:326XX (UTM zones)
    
    Returns:
        GeoDataFrame with proper CRS
    """
    data = {
        'name': [loc.name for loc in locations],
        'demand': [loc.demand for loc in locations],
        'geometry': [Point(loc.x, loc.y) for loc in locations]
    }
    
    # Create GeoDataFrame with input CRS
    gdf = gpd.GeoDataFrame(data, crs=input_crs)
    
    # Project to target CRS if specified
    if target_crs and target_crs != input_crs:
        gdf = gdf.to_crs(target_crs)
    
    return gdf


def solve_pmedian_spopt(
    locations: List[Location],
    p: int,
    distance_type: str,
    input_crs: str = "EPSG:4326",
    target_crs: Optional[str] = None
) -> tuple:
    """
    Solve P-Median problem using spopt library
    
    Args:
        locations: List of location points
        p: Number of facilities to locate
        distance_type: 'euclidean' or 'manhattan'
        input_crs: Input coordinate reference system
        target_crs: Target CRS for projection (for accurate distance)
    
    Returns:
        tuple: (facility_indices, assignments, total_distance, solver_status)
    """
    if not SPOPT_AVAILABLE:
        raise ImportError("Spopt not available. Falling back to greedy heuristic.")
    
    if not PULP_AVAILABLE:
        raise ImportError("PuLP not available. Cannot use exact solver.")
    
    # Create GeoDataFrame with proper CRS
    gdf = create_geodataframe(locations, input_crs, target_crs)
    
    # Use PMedian.from_geodataframe (simpler API)
    try:
        pmedian = PMedian.from_geodataframe(
            gdf,
            gdf,
            "demand",
            "name",
            p_facilities=p,
            distance_metric=distance_type
        )
        
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        facility_indices = list(pmedian.fac2iloc.values())
        
        # Get assignments
        cli2fac = pmedian.cli2fac
        assignments = [cli2fac.get(i, facility_indices[0]) for i in range(len(locations))]
        
        # Calculate total distance
        total_distance = pmedian.problem.objective.value()
        
        solver_status = {
            'status': str(pmedian.problem.status),
            'solver': 'PULP_CBC',
            'method': 'exact_ip',  # Integer Programming
            'library': 'spopt',
            'crs_used': target_crs if target_crs else input_crs,
            'projected': target_crs is not None
        }
        
        return facility_indices, assignments, total_distance, solver_status
        
    except Exception as e:
        # Fallback to greedy heuristic if IP solver fails
        print(f"IP solver failed: {e}, falling back to greedy")
        
        # Calculate cost matrix manually
        coords = np.array([[p.x, p.y] for p in gdf.geometry])
        if distance_type == 'euclidean':
            cost_matrix = np.sqrt(((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2).sum(axis=2))
        else:  # manhattan
            cost_matrix = np.abs(coords[:, np.newaxis] - coords[np.newaxis, :]).sum(axis=2)
        
        return solve_greedy_fallback(cost_matrix, gdf['demand'].values, p, input_crs, target_crs)


def solve_greedy_fallback(
    cost_matrix: np.ndarray,
    demands: np.ndarray,
    p: int,
    input_crs: str = "EPSG:4326",
    target_crs: Optional[str] = None
) -> tuple:
    """Greedy heuristic fallback"""
    n = len(demands)
    selected_facilities = []
    
    # Start with point that minimizes total weighted distance
    min_total = float('inf')
    first_facility = 0
    
    for i in range(n):
        total = sum(cost_matrix[j, i] * demands[j] for j in range(n))
        if total < min_total:
            min_total = total
            first_facility = i
    
    selected_facilities.append(first_facility)
    
    # Greedily add remaining facilities
    for _ in range(p - 1):
        best_reduction = -float('inf')
        best_candidate = None
        
        for candidate in range(n):
            if candidate in selected_facilities:
                continue
            
            # Calculate improvement
            reduction = 0
            for i in range(n):
                current_min = min(cost_matrix[i, f] for f in selected_facilities)
                new_min = min(current_min, cost_matrix[i, candidate])
                reduction += (current_min - new_min) * demands[i]
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_candidate = candidate
        
        if best_candidate is not None:
            selected_facilities.append(best_candidate)
    
    # Assign all points to closest facility
    assignments = []
    total_dist = 0
    
    for i in range(n):
        min_dist = float('inf')
        closest = selected_facilities[0]
        
        for f in selected_facilities:
            dist = cost_matrix[i, f] * demands[i]
            if dist < min_dist:
                min_dist = dist
                closest = f
        
        assignments.append(closest)
        total_dist += min_dist
    
    solver_status = {
        'status': 'Optimal (Greedy Heuristic)',
        'solver': 'greedy_heuristic',
        'method': 'heuristic',
        'library': 'custom',
        'crs_used': target_crs if target_crs else input_crs,
        'projected': target_crs is not None
    }
    
    return selected_facilities, assignments, total_dist, solver_status


def create_location_map(
    locations: List[Location],
    facility_indices: List[int],
    assignments: List[int]
) -> str:
    """Create location map visualization"""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create GeoDataFrame for plotting
    gdf = create_geodataframe(locations)
    
    # Color map for assignments
    n_facilities = len(facility_indices)
    colors = plt.cm.Set3(np.linspace(0, 1, n_facilities))
    facility_color_map = {f: colors[i] for i, f in enumerate(facility_indices)}
    
    # Plot demand points by assignment
    for i, loc in enumerate(locations):
        assigned_facility = assignments[i]
        color = facility_color_map[assigned_facility]
        
        if i in facility_indices:
            # Facility point - star marker
            ax.scatter(loc.x, loc.y, c=[color], s=500, marker='*', 
                      edgecolors='black', linewidths=2.5, zorder=5, label='_nolegend_')
            ax.scatter(loc.x, loc.y, c=[color], s=200, marker='s', 
                      edgecolors='black', linewidths=2, zorder=4, alpha=0.8)
        else:
            # Demand point - circle with size based on demand
            ax.scatter(loc.x, loc.y, c=[color], s=100 + loc.demand * 50, 
                      alpha=0.6, edgecolors='black', linewidths=1, zorder=3)
        
        # Label with offset
        ax.annotate(loc.name, (loc.x, loc.y), xytext=(6, 6), 
                   textcoords='offset points', fontsize=9, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.7))
    
    # Draw assignment lines
    for i, loc in enumerate(locations):
        if i not in facility_indices:
            facility_idx = assignments[i]
            facility_loc = locations[facility_idx]
            ax.plot([loc.x, facility_loc.x], [loc.y, facility_loc.y], 
                   'k--', alpha=0.15, linewidth=1, zorder=1)
    
    # Add Voronoi-like regions (convex hulls) for visual clarity
    from scipy.spatial import ConvexHull
    for fac_idx, color in facility_color_map.items():
        assigned_points = [(locations[i].x, locations[i].y) 
                          for i in range(len(locations)) 
                          if assignments[i] == fac_idx]
        
        if len(assigned_points) >= 3:
            try:
                points = np.array(assigned_points)
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 
                           color=color, alpha=0.2, linewidth=1.5, zorder=0)
            except:
                pass
    
    ax.set_xlabel('X Coordinate', fontsize=12, weight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
    ax.set_title('P-Median Solution: Optimal Facility Locations', 
                fontsize=14, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markersize=15, markeredgecolor='black', markeredgewidth=2,
                  label='Selected Facilities'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=10, markeredgecolor='black', label='Demand Points', alpha=0.7),
        plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.3,
                  label='Assignments')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_distance_distribution(
    cost_matrix: np.ndarray,
    locations: List[Location],
    facility_indices: List[int],
    assignments: List[int]
) -> str:
    """Create distance distribution chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate distances
    distances = []
    demands = []
    
    for i, loc in enumerate(locations):
        if i not in facility_indices:
            dist = cost_matrix[i, assignments[i]]
            distances.append(dist)
            demands.append(loc.demand)
    
    # Histogram of distances
    if distances:
        ax1.hist(distances, bins=min(20, max(5, len(distances)//2)), 
                color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        
        ax1.axvline(mean_dist, color='red', linestyle='--', 
                   linewidth=2.5, label=f'Mean: {mean_dist:.2f}')
        ax1.axvline(median_dist, color='orange', linestyle=':', 
                   linewidth=2.5, label=f'Median: {median_dist:.2f}')
        
        ax1.set_xlabel('Distance to Nearest Facility', fontsize=11, weight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax1.set_title('Distance Distribution', fontsize=13, weight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Weighted distances scatter
    weighted_distances = [d * dem for d, dem in zip(distances, demands)]
    
    if weighted_distances:
        indices = list(range(len(weighted_distances)))
        colors_scatter = plt.cm.viridis(np.array(weighted_distances) / max(weighted_distances))
        
        ax2.scatter(indices, weighted_distances, c=colors_scatter, s=100, 
                   alpha=0.7, edgecolors='black', linewidths=1)
        ax2.axhline(np.mean(weighted_distances), color='darkred', 
                   linestyle='--', linewidth=2.5, 
                   label=f'Mean: {np.mean(weighted_distances):.2f}')
        
        ax2.set_xlabel('Demand Point Index', fontsize=11, weight='bold')
        ax2.set_ylabel('Weighted Distance (Distance × Demand)', fontsize=11, weight='bold')
        ax2.set_title('Weighted Distance Distribution', fontsize=13, weight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_facility_load_chart(
    locations: List[Location],
    facility_indices: List[int],
    facility_loads: Dict[str, int]
) -> str:
    """Create facility load comparison chart"""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    facility_names = [locations[i].name for i in facility_indices]
    loads = [facility_loads[name] for name in facility_names]
    
    # Create color gradient
    colors_gradient = plt.cm.RdYlGn_r(np.array(loads) / max(loads))
    
    bars = ax.bar(facility_names, loads, color=colors_gradient, 
                 alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, load in zip(bars, loads):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(load)}',
               ha='center', va='bottom', fontsize=11, weight='bold')
    
    mean_load = np.mean(loads)
    ax.axhline(mean_load, color='red', linestyle='--', 
              linewidth=2.5, label=f'Average: {mean_load:.1f}', zorder=10)
    
    ax.set_xlabel('Facility', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Assigned Demand Points', fontsize=12, weight='bold')
    ax.set_title('Facility Load Distribution', fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_coverage_analysis(
    cost_matrix: np.ndarray,
    locations: List[Location],
    facility_indices: List[int],
    assignments: List[int]
) -> str:
    """Create coverage analysis visualization"""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Calculate coverage at different distance thresholds
    max_dist = np.max(cost_matrix)
    thresholds = np.linspace(0, max_dist * 1.1, 100)
    coverage_percentages = []
    
    demands = np.array([loc.demand for loc in locations])
    total_demand = sum(demands)
    
    for threshold in thresholds:
        covered_demand = 0
        for i in range(len(locations)):
            if cost_matrix[i, assignments[i]] <= threshold:
                covered_demand += demands[i]
        coverage_percentages.append((covered_demand / total_demand) * 100)
    
    # Plot coverage curve
    ax.plot(thresholds, coverage_percentages, linewidth=3.5, 
           color='royalblue', label='Coverage Curve')
    ax.fill_between(thresholds, coverage_percentages, alpha=0.3, color='royalblue')
    
    # Mark current solution
    current_max_dist = max(cost_matrix[i, assignments[i]] for i in range(len(locations)))
    current_coverage = coverage_percentages[np.argmin(np.abs(thresholds - current_max_dist))]
    
    ax.axvline(current_max_dist, color='red', linestyle='--', 
              linewidth=2.5, label=f'Max Distance in Solution: {current_max_dist:.2f}')
    ax.scatter([current_max_dist], [current_coverage], color='red', s=200, 
              zorder=5, edgecolors='black', linewidths=2)
    
    # Mark percentiles
    percentiles = [50, 80, 95]
    colors_perc = ['green', 'orange', 'purple']
    
    for pct, color in zip(percentiles, colors_perc):
        threshold_idx = np.argmin(np.abs(np.array(coverage_percentages) - pct))
        threshold_val = thresholds[threshold_idx]
        ax.axhline(pct, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
        ax.axvline(threshold_val, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
        ax.scatter([threshold_val], [pct], color=color, s=100, zorder=4,
                  edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Distance Threshold', fontsize=12, weight='bold')
    ax.set_ylabel('Coverage (% of Total Demand)', fontsize=12, weight='bold')
    ax.set_title('Service Coverage Analysis', fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max_dist * 1.1)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def generate_interpretation(
    total_distance: float,
    avg_distance: float,
    max_distance: float,
    facility_loads: Dict[str, int],
    p: int,
    n_locations: int,
    solver_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Insight 1: Solution quality
    if avg_distance < max_distance * 0.5:
        key_insights.append({
            "title": "Excellent Service Coverage",
            "description": f"Average distance ({avg_distance:.2f}) is less than half the maximum ({max_distance:.2f}), indicating well-balanced facility placement.",
            "status": "positive"
        })
    elif avg_distance < max_distance * 0.7:
        key_insights.append({
            "title": "Good Service Distribution",
            "description": f"Most demand points are well-served, though some outliers exist with longer distances.",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Coverage Gap Detected",
            "description": f"Significant variation in service distance. Some areas may need additional facilities.",
            "status": "warning"
        })
        recommendations.append("Consider increasing the number of facilities (P) to improve coverage uniformity")
    
    # Insight 2: Facility utilization
    loads = list(facility_loads.values())
    load_std = np.std(loads)
    load_mean = np.mean(loads)
    load_cv = load_std / load_mean if load_mean > 0 else 0
    
    if load_cv < 0.3:
        key_insights.append({
            "title": "Well-Balanced Facility Load",
            "description": f"Demand is evenly distributed across facilities (CV: {load_cv:.2f}). All facilities are utilized efficiently.",
            "status": "positive"
        })
    elif load_cv < 0.5:
        key_insights.append({
            "title": "Moderate Load Variation",
            "description": f"Some facilities handle more demand than others (CV: {load_cv:.2f}), but within acceptable range.",
            "status": "neutral"
        })
    else:
        key_insights.append({
            "title": "Imbalanced Facility Utilization",
            "description": f"Significant load imbalance detected (CV: {load_cv:.2f}). Some facilities may be overutilized.",
            "status": "warning"
        })
        recommendations.append("Review facility placements to balance workload more evenly")
    
    # Insight 3: Solver performance
    if solver_info['method'] == 'exact_ip':
        key_insights.append({
            "title": "Optimal Solution Found",
            "description": f"Integer programming solver ({solver_info['solver']}) guaranteed the globally optimal solution.",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Heuristic Solution",
            "description": f"Solution obtained using {solver_info['method']}. Result is high-quality but may not be globally optimal.",
            "status": "neutral"
        })
    
    # Additional recommendations
    coverage_ratio = p / n_locations
    
    if coverage_ratio < 0.2:
        recommendations.append(f"Low facility ratio ({p}/{n_locations} = {coverage_ratio*100:.1f}%) - solution is highly optimized but may benefit from additional facilities")
    
    if max_distance > avg_distance * 2.5:
        recommendations.append(f"Maximum service distance ({max_distance:.2f}) is exceptionally high - investigate underserved regions")
    
    recommendations.append(f"Total weighted distance minimized to {total_distance:.2f} units")
    recommendations.append(f"Solved using {solver_info['library']} library with {solver_info['solver']}")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/p-median", response_model=PMedianResponse)
async def solve_p_median_problem(request: PMedianRequest):
    """
    Solve the P-Median problem using spopt (PySAL spatial optimization)
    
    Finds P facility locations that minimize total weighted distance to demand points
    
    Important: For accurate distance calculation with lat/lon coordinates (EPSG:4326),
    consider specifying a projected CRS in the 'crs' parameter:
    - Korea: EPSG:5179 or EPSG:5186
    - Japan: EPSG:6668 or EPSG:2451
    - USA: EPSG:5070 (Albers) or appropriate UTM zone
    - Europe: EPSG:3035 (LAEA) or appropriate UTM zone
    - Web Mercator: EPSG:3857
    """
    try:
        locations = request.locations
        p = request.p
        distance_type = request.distance_type
        input_crs = request.input_crs
        target_crs = request.crs
        
        n = len(locations)
        
        if p > n:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot select {p} facilities from {n} locations"
            )
        
        # Check for required libraries
        if not PULP_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="PuLP library not installed. Install with: pip install pulp"
            )
        
        if not SPOPT_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Spopt library not installed. Install with: pip install spopt geopandas"
            )
        
        # Solve using spopt (which handles distance matrix internally)
        facility_indices, assignments, total_distance, solver_info = solve_pmedian_spopt(
            locations, p, distance_type, input_crs, target_crs
        )
        
        # Calculate distances for statistics
        gdf = create_geodataframe(locations, input_crs, target_crs)
        coords = np.array([[loc.x, loc.y] for loc in locations])
        if target_crs and target_crs != input_crs:
            coords = np.array([[gdf.geometry.iloc[i].x, gdf.geometry.iloc[i].y] for i in range(len(locations))])
        
        # Build cost matrix for visualization functions
        if distance_type == 'euclidean':
            cost_matrix = np.sqrt(((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2).sum(axis=2))
        else:  # manhattan
            cost_matrix = np.abs(coords[:, np.newaxis] - coords[np.newaxis, :]).sum(axis=2)
        
        distances = []
        for i in range(len(locations)):
            fac_idx = assignments[i]
            distances.append(cost_matrix[i, fac_idx])
        
        # Calculate statistics
        demands = np.array([loc.demand for loc in locations])
        avg_distance = total_distance / sum(demands)
        max_distance = max(distances)
        
        # Build results
        selected_names = [locations[i].name for i in facility_indices]
        assignment_dict = {
            locations[i].name: locations[assignments[i]].name
            for i in range(n)
        }
        
        # Calculate facility loads
        facility_loads = {locations[f].name: 0 for f in facility_indices}
        for assignment in assignments:
            facility_loads[locations[assignment].name] += 1
        
        # Generate visualizations
        plots = {
            "location_map": create_location_map(locations, facility_indices, assignments),
            "distance_distribution": create_distance_distribution(
                cost_matrix, locations, facility_indices, assignments
            ),
            "facility_loads": create_facility_load_chart(
                locations, facility_indices, facility_loads
            ),
            "coverage_analysis": create_coverage_analysis(
                cost_matrix, locations, facility_indices, assignments
            )
        }
        
        # Generate interpretation
        interpretation = generate_interpretation(
            total_distance, avg_distance, max_distance,
            facility_loads, p, n, solver_info
        )
        
        # Add CRS warning to interpretation if using unprojected coordinates
        if input_crs == "EPSG:4326" and not target_crs:
            interpretation["recommendations"].insert(0,
                "⚠️ Using unprojected lat/lon coordinates (EPSG:4326). For accurate distance calculation, "
                "consider specifying a projected CRS (e.g., EPSG:5179 for Korea, EPSG:3857 for Web Mercator)"
            )
        
        return PMedianResponse(
            success=True,
            total_distance=float(total_distance),
            avg_distance=float(avg_distance),
            max_distance=float(max_distance),
            selected_facilities=selected_names,
            facility_indices=facility_indices,
            assignments=assignment_dict,
            facility_loads=facility_loads,
            problem={
                "n_locations": n,
                "n_facilities": p,
                "distance_type": distance_type,
                "total_demand": float(sum(demands)),
                "input_crs": input_crs,
                "target_crs": target_crs,
                "projected": target_crs is not None
            },
            plots=plots,
            interpretation=interpretation,
            solver_info=solver_info
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required library: {str(e)}. Install with: pip install spopt geopandas pulp"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")
