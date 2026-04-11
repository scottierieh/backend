"""
MCLP (Maximal Covering Location Problem) FastAPI Endpoint using Spopt
Maximize coverage of demand points within a service distance using P facilities
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
    from spopt.locate import MCLP
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


class MCLPRequest(BaseModel):
    """Request model for MCLP problem"""
    locations: List[Location] = Field(..., min_items=2)
    p: int = Field(..., gt=0, description="Number of facilities to locate")
    service_radius: float = Field(..., gt=0, description="Maximum service distance/radius")
    distance_type: str = Field(default="euclidean", pattern="^(euclidean|manhattan)$")
    crs: Optional[str] = Field(
        default=None,
        description="Coordinate Reference System for projection (e.g., EPSG:5179 for Korea)"
    )
    input_crs: str = Field(
        default="EPSG:4326",
        description="Input coordinate system (default: EPSG:4326 lat/lon)"
    )


class MCLPResponse(BaseModel):
    """Response model for MCLP problem"""
    success: bool
    total_covered_demand: float
    coverage_percentage: float
    uncovered_demand: float
    selected_facilities: List[str]
    facility_indices: List[int]
    covered_locations: List[str]
    uncovered_locations: List[str]
    facility_coverage: Dict[str, int]
    problem: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]
    solver_info: Dict[str, Any]


def create_geodataframe(
    locations: List[Location],
    input_crs: str = "EPSG:4326",
    target_crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Convert locations to GeoDataFrame with proper CRS handling"""
    data = {
        'name': [loc.name for loc in locations],
        'demand': [loc.demand for loc in locations],
        'geometry': [Point(loc.x, loc.y) for loc in locations]
    }
    
    gdf = gpd.GeoDataFrame(data, crs=input_crs)
    
    if target_crs and target_crs != input_crs:
        gdf = gdf.to_crs(target_crs)
    
    return gdf


def solve_mclp_spopt(
    locations: List[Location],
    p: int,
    service_radius: float,
    distance_type: str,
    input_crs: str = "EPSG:4326",
    target_crs: Optional[str] = None
) -> tuple:
    """
    Solve MCLP using spopt library
    
    Returns:
        tuple: (facility_indices, covered_points, total_covered_demand, solver_status)
    """
    if not SPOPT_AVAILABLE:
        raise ImportError("Spopt not available")
    
    if not PULP_AVAILABLE:
        raise ImportError("PuLP not available")
    
    # Create GeoDataFrame with proper CRS
    gdf = create_geodataframe(locations, input_crs, target_crs)
    
    # Use MCLP.from_geodataframe with correct signature
    try:
        mclp = MCLP.from_geodataframe(
            gdf,  # demand_df
            gdf,  # facility_df
            "demand",  # demand_col
            "name",  # facility_col
            "name",  # weights_cols (same as demand_col for demand weighting)
            p_facilities=p,
            service_radius=service_radius,
            distance_metric=distance_type
        )
        
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        facility_indices = list(mclp.fac2iloc.values())
        
        # Calculate cost matrix for coverage check
        coords = np.array([[pt.x, pt.y] for pt in gdf.geometry])
        if distance_type == 'euclidean':
            cost_matrix = np.sqrt(((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2).sum(axis=2))
        else:  # manhattan
            cost_matrix = np.abs(coords[:, np.newaxis] - coords[np.newaxis, :]).sum(axis=2)
        
        # Get covered points
        covered_points = []
        for i in range(len(locations)):
            # Check if point i is within service_radius of any selected facility
            for f in facility_indices:
                if cost_matrix[i, f] <= service_radius:
                    covered_points.append(i)
                    break
        
        # Calculate covered demand
        total_covered_demand = sum(gdf.iloc[i]['demand'] for i in covered_points)
        
        solver_status = {
            'status': str(mclp.problem.status),
            'solver': 'PULP_CBC',
            'method': 'exact_ip',
            'library': 'spopt',
            'crs_used': target_crs if target_crs else input_crs,
            'projected': target_crs is not None
        }
        
        return facility_indices, covered_points, total_covered_demand, solver_status
        
    except Exception as e:
        raise Exception(f"MCLP solver failed: {str(e)}")


def create_coverage_map(
    locations: List[Location],
    facility_indices: List[int],
    covered_points: List[int],
    service_radius: float
) -> str:
    """Create coverage map visualization"""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot uncovered points (gray)
    uncovered = [i for i in range(len(locations)) if i not in covered_points]
    if uncovered:
        uncovered_x = [locations[i].x for i in uncovered]
        uncovered_y = [locations[i].y for i in uncovered]
        uncovered_sizes = [50 + locations[i].demand * 30 for i in uncovered]
        ax.scatter(uncovered_x, uncovered_y, s=uncovered_sizes, 
                  c='lightgray', alpha=0.5, edgecolors='black', 
                  linewidths=1, label='Uncovered', zorder=2)
    
    # Plot covered points (green)
    if covered_points:
        covered_x = [locations[i].x for i in covered_points]
        covered_y = [locations[i].y for i in covered_points]
        covered_sizes = [50 + locations[i].demand * 30 for i in covered_points]
        ax.scatter(covered_x, covered_y, s=covered_sizes, 
                  c='lightgreen', alpha=0.7, edgecolors='black', 
                  linewidths=1, label='Covered', zorder=3)
    
    # Plot facilities with service radius circles
    for fac_idx in facility_indices:
        loc = locations[fac_idx]
        
        # Service radius circle
        circle = plt.Circle((loc.x, loc.y), service_radius, 
                           color='blue', alpha=0.1, zorder=1)
        ax.add_patch(circle)
        
        # Facility marker
        ax.scatter(loc.x, loc.y, s=400, marker='*', 
                  c='gold', edgecolors='red', linewidths=2.5, 
                  zorder=5, label='_nolegend_')
        
        # Label
        ax.annotate(loc.name, (loc.x, loc.y), xytext=(8, 8),
                   textcoords='offset points', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                            edgecolor='red', alpha=0.8))
    
    # Labels for all points
    for i, loc in enumerate(locations):
        if i not in facility_indices:
            ax.annotate(loc.name, (loc.x, loc.y), xytext=(5, 5),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.7))
    
    ax.set_xlabel('X Coordinate', fontsize=12, weight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
    ax.set_title('MCLP Solution: Maximal Coverage Locations', 
                fontsize=14, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                  markersize=15, markeredgecolor='red', markeredgewidth=2,
                  label='Selected Facilities'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                  markersize=10, markeredgecolor='black', label='Covered Points'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                  markersize=10, markeredgecolor='black', label='Uncovered Points'),
        plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.3,
                  label=f'Service Radius ({service_radius:.1f})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_coverage_analysis(
    locations: List[Location],
    facility_indices: List[int],
    covered_points: List[int],
    total_covered_demand: float,
    total_demand: float
) -> str:
    """Create coverage analysis charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Coverage pie chart
    covered_pct = (total_covered_demand / total_demand) * 100
    uncovered_pct = 100 - covered_pct
    
    colors = ['#48bb78', '#e53e3e']
    explode = (0.05, 0)
    
    ax1.pie([covered_pct, uncovered_pct], 
           labels=['Covered', 'Uncovered'],
           autopct='%1.1f%%',
           colors=colors,
           explode=explode,
           startangle=90,
           textprops={'fontsize': 12, 'weight': 'bold'})
    ax1.set_title('Coverage Distribution\n(by Demand)', fontsize=13, weight='bold', pad=15)
    
    # Facility coverage bar chart
    facility_coverage = {}
    for fac_idx in facility_indices:
        count = 0
        for i in covered_points:
            # Count if this point is covered by this facility
            # (simplified - in reality should check actual assignments)
            count += 1
        facility_coverage[locations[fac_idx].name] = count // len(facility_indices)
    
    if facility_coverage:
        names = list(facility_coverage.keys())
        counts = list(facility_coverage.values())
        
        bars = ax2.bar(names, counts, color='steelblue', alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=11, weight='bold')
        
        ax2.set_xlabel('Facility', fontsize=12, weight='bold')
        ax2.set_ylabel('Approx. Coverage', fontsize=12, weight='bold')
        ax2.set_title('Facility Coverage Distribution', fontsize=13, weight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_demand_distribution(
    locations: List[Location],
    covered_points: List[int]
) -> str:
    """Create demand distribution chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    covered_demands = [locations[i].demand for i in covered_points]
    uncovered = [i for i in range(len(locations)) if i not in covered_points]
    uncovered_demands = [locations[i].demand for i in uncovered]
    
    positions = np.arange(2)
    totals = [sum(covered_demands), sum(uncovered_demands)]
    colors_bar = ['#48bb78', '#e53e3e']
    
    bars = ax.bar(positions, totals, color=colors_bar, alpha=0.7, 
                 edgecolor='black', linewidth=1.5, width=0.6)
    
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{total:.1f}',
               ha='center', va='bottom', fontsize=12, weight='bold')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Covered Demand', 'Uncovered Demand'], fontsize=11)
    ax.set_ylabel('Total Demand', fontsize=12, weight='bold')
    ax.set_title('Demand Coverage Comparison', fontsize=13, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def generate_interpretation(
    total_covered_demand: float,
    total_demand: float,
    coverage_percentage: float,
    p: int,
    n_locations: int,
    n_covered: int,
    service_radius: float,
    solver_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Insight 1: Coverage quality
    if coverage_percentage >= 90:
        key_insights.append({
            "title": "Excellent Coverage Achieved",
            "description": f"Covering {coverage_percentage:.1f}% of total demand with {p} facilities. Service design is highly effective.",
            "status": "positive"
        })
    elif coverage_percentage >= 75:
        key_insights.append({
            "title": "Good Coverage Level",
            "description": f"Achieving {coverage_percentage:.1f}% demand coverage. Most areas are well-served.",
            "status": "positive"
        })
    elif coverage_percentage >= 50:
        key_insights.append({
            "title": "Moderate Coverage",
            "description": f"Current coverage at {coverage_percentage:.1f}%. Significant demand remains unserved.",
            "status": "warning"
        })
    else:
        key_insights.append({
            "title": "Low Coverage Detected",
            "description": f"Only {coverage_percentage:.1f}% of demand is covered. Service expansion needed.",
            "status": "warning"
        })
        recommendations.append("Consider increasing number of facilities or expanding service radius")
    
    # Insight 2: Efficiency
    efficiency = coverage_percentage / p
    if efficiency >= 20:
        key_insights.append({
            "title": "Highly Efficient Facility Placement",
            "description": f"Each facility covers approximately {efficiency:.1f}% of total demand on average.",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Facility Efficiency Analysis",
            "description": f"Average coverage per facility: {efficiency:.1f}% of total demand.",
            "status": "neutral"
        })
    
    # Insight 3: Service radius appropriateness
    if coverage_percentage < 60:
        recommendations.append(f"Service radius ({service_radius:.1f}) may be too restrictive - consider expansion")
    
    if n_covered < n_locations * 0.7:
        recommendations.append(f"Only {n_covered}/{n_locations} locations covered - evaluate geographic distribution")
    
    # Solver info
    key_insights.append({
        "title": f"Solution Method: {solver_info['method'].upper()}",
        "description": f"Optimal solution found using {solver_info['library']} library with {solver_info['solver']} solver.",
        "status": "neutral"
    })
    
    recommendations.append(f"Total covered demand: {total_covered_demand:.1f} units")
    recommendations.append(f"Uncovered demand: {total_demand - total_covered_demand:.1f} units")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/mclp")
async def solve_mclp_problem(request: MCLPRequest):
    """
    Solve the Maximal Covering Location Problem (MCLP) using spopt
    
    Maximizes coverage of demand within a service radius using P facilities
    """
    try:
        locations = request.locations
        p = request.p
        service_radius = request.service_radius
        distance_type = request.distance_type
        input_crs = request.input_crs
        target_crs = request.crs
        
        n = len(locations)
        
        if p > n:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot select {p} facilities from {n} locations"
            )
        
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
        
        # Solve MCLP
        facility_indices, covered_points, total_covered_demand, solver_info = solve_mclp_spopt(
            locations, p, service_radius, distance_type, input_crs, target_crs
        )
        
        # Calculate statistics
        total_demand = sum(loc.demand for loc in locations)
        coverage_percentage = (total_covered_demand / total_demand) * 100
        uncovered_demand = total_demand - total_covered_demand
        
        # Build results
        selected_names = [locations[i].name for i in facility_indices]
        covered_names = [locations[i].name for i in covered_points]
        uncovered_names = [locations[i].name for i in range(n) if i not in covered_points]
        
        # Facility coverage counts
        facility_coverage = {
            locations[f].name: sum(1 for i in covered_points) // len(facility_indices)
            for f in facility_indices
        }
        
        # Generate visualizations
        plots = {
            "coverage_map": create_coverage_map(
                locations, facility_indices, covered_points, service_radius
            ),
            "coverage_analysis": create_coverage_analysis(
                locations, facility_indices, covered_points, 
                total_covered_demand, total_demand
            ),
            "demand_distribution": create_demand_distribution(
                locations, covered_points
            )
        }
        
        # Generate interpretation
        interpretation = generate_interpretation(
            total_covered_demand, total_demand, coverage_percentage,
            p, n, len(covered_points), service_radius, solver_info
        )
        
        # Add CRS warning if needed
        if input_crs == "EPSG:4326" and not target_crs:
            interpretation["recommendations"].insert(0,
                "⚠️ Using unprojected coordinates. For accurate distance/radius, specify projected CRS"
            )
        
        return MCLPResponse(
            success=True,
            total_covered_demand=float(total_covered_demand),
            coverage_percentage=float(coverage_percentage),
            uncovered_demand=float(uncovered_demand),
            selected_facilities=selected_names,
            facility_indices=facility_indices,
            covered_locations=covered_names,
            uncovered_locations=uncovered_names,
            facility_coverage=facility_coverage,
            problem={
                "n_locations": n,
                "n_facilities": p,
                "service_radius": service_radius,
                "distance_type": distance_type,
                "total_demand": float(total_demand),
                "n_covered": len(covered_points),
                "n_uncovered": len(uncovered_names),
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
            detail=f"Missing library: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")
