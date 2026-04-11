"""
Location-Allocation Model FastAPI Endpoint
Optimizes facility locations and demand assignments to minimize total weighted distance
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

router = APIRouter()


class DemandPoint(BaseModel):
    """Demand point with location and demand weight"""
    name: str
    x: float
    y: float
    demand: float = 1.0


class LocationAllocationRequest(BaseModel):
    """Request model for Location-Allocation problem"""
    demand_points: List[DemandPoint] = Field(..., min_items=2)
    n_facilities: int = Field(..., gt=0, description="Number of facilities to locate")
    max_iterations: int = Field(default=100, gt=0, le=500)
    distance_type: str = Field(default="euclidean", pattern="^(euclidean|manhattan)$")
    algorithm: str = Field(default="kmeans", pattern="^(kmeans|alternate)$")


class LocationAllocationResponse(BaseModel):
    """Response model for Location-Allocation problem"""
    success: bool
    total_distance: float
    avg_distance_per_demand: float
    facilities: List[Dict[str, Any]]
    assignments: Dict[str, str]
    facility_loads: Dict[str, int]
    facility_total_demand: Dict[str, float]
    iterations: int
    converged: bool
    problem: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]
    algorithm_info: Dict[str, Any]


def calculate_distance_matrix(
    points1: np.ndarray,
    points2: np.ndarray,
    distance_type: str = 'euclidean'
) -> np.ndarray:
    """Calculate distance matrix between two sets of points"""
    if distance_type == 'euclidean':
        return cdist(points1, points2, metric='euclidean')
    else:  # manhattan
        return cdist(points1, points2, metric='cityblock')


def solve_kmeans(
    demand_coords: np.ndarray,
    demands: np.ndarray,
    n_facilities: int,
    max_iterations: int,
    distance_type: str
) -> tuple:
    """
    Solve using K-Means clustering (weighted)
    """
    # Weight coordinates by demand for clustering
    weighted_coords = np.repeat(demand_coords, demands.astype(int).clip(1, 100), axis=0)
    
    kmeans = KMeans(
        n_clusters=n_facilities,
        max_iter=max_iterations,
        n_init=10,
        random_state=42
    )
    
    kmeans.fit(weighted_coords)
    facility_locations = kmeans.cluster_centers_
    
    # Assign demands to nearest facility
    distances = calculate_distance_matrix(demand_coords, facility_locations, distance_type)
    assignments = np.argmin(distances, axis=1)
    
    # Calculate total weighted distance
    total_distance = sum(
        distances[i, assignments[i]] * demands[i]
        for i in range(len(demand_coords))
    )
    
    return facility_locations, assignments, total_distance, kmeans.n_iter_


def solve_alternate_location_allocation(
    demand_coords: np.ndarray,
    demands: np.ndarray,
    n_facilities: int,
    max_iterations: int,
    distance_type: str
) -> tuple:
    """
    Solve using Alternate Location-Allocation algorithm
    1. Start with random facility locations
    2. Allocate: Assign demands to nearest facility
    3. Locate: Move facilities to weighted centroid of assigned demands
    4. Repeat until convergence
    """
    # Initialize facilities randomly from demand points
    np.random.seed(42)
    initial_indices = np.random.choice(len(demand_coords), n_facilities, replace=False)
    facility_locations = demand_coords[initial_indices].copy()
    
    prev_total_distance = float('inf')
    converged = False
    
    for iteration in range(max_iterations):
        # ALLOCATION STEP: Assign demands to nearest facility
        distances = calculate_distance_matrix(demand_coords, facility_locations, distance_type)
        assignments = np.argmin(distances, axis=1)
        
        # Calculate total weighted distance
        total_distance = sum(
            distances[i, assignments[i]] * demands[i]
            for i in range(len(demand_coords))
        )
        
        # Check convergence
        if abs(prev_total_distance - total_distance) < 0.001:
            converged = True
            break
        
        prev_total_distance = total_distance
        
        # LOCATION STEP: Move facilities to weighted centroid
        new_facility_locations = np.zeros_like(facility_locations)
        for f in range(n_facilities):
            assigned_indices = np.where(assignments == f)[0]
            
            if len(assigned_indices) > 0:
                # Weighted centroid
                assigned_coords = demand_coords[assigned_indices]
                assigned_demands = demands[assigned_indices]
                
                total_demand = assigned_demands.sum()
                if total_demand > 0:
                    new_facility_locations[f] = (
                        (assigned_coords * assigned_demands[:, np.newaxis]).sum(axis=0) / total_demand
                    )
                else:
                    new_facility_locations[f] = facility_locations[f]
            else:
                # No demands assigned, keep current location
                new_facility_locations[f] = facility_locations[f]
        
        facility_locations = new_facility_locations
    
    # Final allocation
    distances = calculate_distance_matrix(demand_coords, facility_locations, distance_type)
    assignments = np.argmin(distances, axis=1)
    total_distance = sum(
        distances[i, assignments[i]] * demands[i]
        for i in range(len(demand_coords))
    )
    
    return facility_locations, assignments, total_distance, iteration + 1


def create_location_allocation_map(
    demand_points: List[DemandPoint],
    facility_locations: np.ndarray,
    assignments: np.ndarray
) -> str:
    """Create location-allocation visualization map"""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    n_facilities = len(facility_locations)
    colors = plt.cm.Set3(np.linspace(0, 1, n_facilities))
    
    # Plot assignments by color
    for f in range(n_facilities):
        assigned_indices = np.where(assignments == f)[0]
        
        if len(assigned_indices) > 0:
            assigned_x = [demand_points[i].x for i in assigned_indices]
            assigned_y = [demand_points[i].y for i in assigned_indices]
            assigned_sizes = [50 + demand_points[i].demand * 30 for i in assigned_indices]
            
            ax.scatter(assigned_x, assigned_y, s=assigned_sizes, 
                      c=[colors[f]], alpha=0.6, edgecolors='black', 
                      linewidths=1, label=f'Facility {f+1} Area', zorder=3)
            
            # Draw lines from demands to facility
            for i in assigned_indices:
                ax.plot([demand_points[i].x, facility_locations[f, 0]], 
                       [demand_points[i].y, facility_locations[f, 1]], 
                       color=colors[f], alpha=0.2, linewidth=1, zorder=1)
    
    # Plot facilities
    ax.scatter(facility_locations[:, 0], facility_locations[:, 1], 
              s=500, marker='*', c='gold', edgecolors='red', 
              linewidths=2.5, zorder=5, label='Facilities')
    
    # Labels for demand points
    for i, dp in enumerate(demand_points):
        ax.annotate(dp.name, (dp.x, dp.y), xytext=(5, 5),
                   textcoords='offset points', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.7))
    
    # Labels for facilities
    for f in range(n_facilities):
        ax.annotate(f'F{f+1}', facility_locations[f], xytext=(8, 8),
                   textcoords='offset points', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                            edgecolor='red', alpha=0.9))
    
    ax.set_xlabel('X Coordinate', fontsize=12, weight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
    ax.set_title('Location-Allocation Solution', fontsize=14, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_facility_analysis(
    demand_points: List[DemandPoint],
    facility_locations: np.ndarray,
    assignments: np.ndarray,
    facility_loads: Dict[str, int],
    facility_demands: Dict[str, float]
) -> str:
    """Create facility analysis charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    n_facilities = len(facility_locations)
    facility_names = [f'Facility {i+1}' for i in range(n_facilities)]
    
    # Facility loads (number of assigned points)
    loads = [facility_loads.get(name, 0) for name in facility_names]
    
    bars1 = ax1.bar(facility_names, loads, color='steelblue', alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    
    for bar, load in zip(bars1, loads):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(load)}',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax1.axhline(np.mean(loads), color='red', linestyle='--',
               linewidth=2, label=f'Average: {np.mean(loads):.1f}')
    ax1.set_xlabel('Facility', fontsize=11, weight='bold')
    ax1.set_ylabel('Number of Assigned Demand Points', fontsize=11, weight='bold')
    ax1.set_title('Facility Load Distribution', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Total demand served by each facility
    demands = [facility_demands.get(name, 0) for name in facility_names]
    
    bars2 = ax2.bar(facility_names, demands, color='coral', alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    
    for bar, demand in zip(bars2, demands):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{demand:.1f}',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax2.axhline(np.mean(demands), color='darkred', linestyle='--',
               linewidth=2, label=f'Average: {np.mean(demands):.1f}')
    ax2.set_xlabel('Facility', fontsize=11, weight='bold')
    ax2.set_ylabel('Total Demand Served', fontsize=11, weight='bold')
    ax2.set_title('Demand Distribution by Facility', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_distance_analysis(
    demand_points: List[DemandPoint],
    facility_locations: np.ndarray,
    assignments: np.ndarray,
    demand_coords: np.ndarray,
    distance_type: str
) -> str:
    """Create distance analysis chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate distances
    distances = calculate_distance_matrix(demand_coords, facility_locations, distance_type)
    assigned_distances = [distances[i, assignments[i]] for i in range(len(demand_points))]
    demands = [dp.demand for dp in demand_points]
    weighted_distances = [d * w for d, w in zip(assigned_distances, demands)]
    
    # Distance histogram
    ax1.hist(assigned_distances, bins=min(20, len(assigned_distances)), 
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    mean_dist = np.mean(assigned_distances)
    median_dist = np.median(assigned_distances)
    
    ax1.axvline(mean_dist, color='red', linestyle='--',
               linewidth=2.5, label=f'Mean: {mean_dist:.2f}')
    ax1.axvline(median_dist, color='orange', linestyle=':',
               linewidth=2.5, label=f'Median: {median_dist:.2f}')
    
    ax1.set_xlabel('Distance to Assigned Facility', fontsize=11, weight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
    ax1.set_title('Distance Distribution', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Weighted distance by point
    colors_scatter = plt.cm.viridis(np.array(weighted_distances) / max(weighted_distances))
    
    ax2.scatter(range(len(weighted_distances)), weighted_distances,
               c=colors_scatter, s=100, alpha=0.7, edgecolors='black', linewidths=1)
    ax2.axhline(np.mean(weighted_distances), color='darkred',
               linestyle='--', linewidth=2.5,
               label=f'Mean: {np.mean(weighted_distances):.2f}')
    
    ax2.set_xlabel('Demand Point Index', fontsize=11, weight='bold')
    ax2.set_ylabel('Weighted Distance (Distance × Demand)', fontsize=11, weight='bold')
    ax2.set_title('Weighted Distance by Point', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
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
    facility_loads: Dict[str, int],
    n_facilities: int,
    n_demands: int,
    iterations: int,
    converged: bool,
    algorithm: str
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Insight 1: Algorithm performance
    if converged:
        key_insights.append({
            "title": "Optimal Solution Achieved",
            "description": f"Algorithm converged after {iterations} iterations, finding the optimal facility locations.",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Maximum Iterations Reached",
            "description": f"Solution found after {iterations} iterations. May benefit from additional iterations.",
            "status": "warning"
        })
        recommendations.append("Consider increasing max_iterations for potential improvement")
    
    # Insight 2: Load balance
    loads = list(facility_loads.values())
    load_std = np.std(loads)
    load_mean = np.mean(loads)
    load_cv = load_std / load_mean if load_mean > 0 else 0
    
    if load_cv < 0.3:
        key_insights.append({
            "title": "Well-Balanced Facility Allocation",
            "description": f"Demand is evenly distributed across facilities (CV: {load_cv:.2f}). Efficient resource utilization.",
            "status": "positive"
        })
    elif load_cv < 0.5:
        key_insights.append({
            "title": "Moderate Load Variation",
            "description": f"Some facilities handle more demand than others (CV: {load_cv:.2f}).",
            "status": "neutral"
        })
    else:
        key_insights.append({
            "title": "Imbalanced Facility Load",
            "description": f"Significant load imbalance detected (CV: {load_cv:.2f}). Some facilities may be overutilized.",
            "status": "warning"
        })
        recommendations.append("Consider adjusting number of facilities or adding capacity constraints")
    
    # Insight 3: Distance efficiency
    avg_per_demand = avg_distance
    if avg_per_demand < 20:
        key_insights.append({
            "title": "Excellent Distance Efficiency",
            "description": f"Average distance per demand unit is {avg_per_demand:.2f}, indicating close proximity to facilities.",
            "status": "positive"
        })
    elif avg_per_demand < 50:
        key_insights.append({
            "title": "Good Service Proximity",
            "description": f"Average distance per demand is {avg_per_demand:.2f}. Most demands are reasonably close to facilities.",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Long Average Distance",
            "description": f"Average distance of {avg_per_demand:.2f} suggests demands are far from facilities.",
            "status": "warning"
        })
        recommendations.append("Consider increasing number of facilities to reduce travel distances")
    
    # Algorithm info
    algo_name = "K-Means Clustering" if algorithm == "kmeans" else "Alternate Location-Allocation"
    key_insights.append({
        "title": f"Algorithm: {algo_name}",
        "description": f"Solution optimized using {algo_name} method for {n_facilities} facilities serving {n_demands} demand points.",
        "status": "neutral"
    })
    
    recommendations.append(f"Total weighted distance: {total_distance:.2f} units")
    recommendations.append(f"Average distance per demand unit: {avg_distance:.2f}")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/location-allocation")
async def solve_location_allocation(request: LocationAllocationRequest):
    """
    Solve Location-Allocation problem to optimize facility locations and assignments
    
    Minimizes total weighted distance between demand points and facilities
    """
    try:
        demand_points = request.demand_points
        n_facilities = request.n_facilities
        max_iterations = request.max_iterations
        distance_type = request.distance_type
        algorithm = request.algorithm
        
        n_demands = len(demand_points)
        
        if n_facilities > n_demands:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot create {n_facilities} facilities for {n_demands} demand points"
            )
        
        # Prepare data
        demand_coords = np.array([[dp.x, dp.y] for dp in demand_points])
        demands = np.array([dp.demand for dp in demand_points])
        total_demand = demands.sum()
        
        # Solve
        if algorithm == "kmeans":
            facility_locations, assignments, total_distance, iterations = solve_kmeans(
                demand_coords, demands, n_facilities, max_iterations, distance_type
            )
            converged = True
        else:  # alternate
            facility_locations, assignments, total_distance, iterations = solve_alternate_location_allocation(
                demand_coords, demands, n_facilities, max_iterations, distance_type
            )
            converged = iterations < max_iterations
        
        # Calculate statistics
        avg_distance = total_distance / total_demand
        
        # Build facilities list
        facilities = []
        for i, loc in enumerate(facility_locations):
            facilities.append({
                "name": f"Facility {i+1}",
                "x": float(loc[0]),
                "y": float(loc[1]),
                "index": i
            })
        
        # Build assignments
        assignment_dict = {
            demand_points[i].name: f"Facility {assignments[i]+1}"
            for i in range(n_demands)
        }
        
        # Calculate facility loads and demands
        facility_loads = {}
        facility_total_demand = {}
        
        for i in range(n_facilities):
            facility_name = f"Facility {i+1}"
            assigned_indices = np.where(assignments == i)[0]
            facility_loads[facility_name] = len(assigned_indices)
            facility_total_demand[facility_name] = float(demands[assigned_indices].sum())
        
        # Generate visualizations
        plots = {
            "location_map": create_location_allocation_map(
                demand_points, facility_locations, assignments
            ),
            "facility_analysis": create_facility_analysis(
                demand_points, facility_locations, assignments,
                facility_loads, facility_total_demand
            ),
            "distance_analysis": create_distance_analysis(
                demand_points, facility_locations, assignments,
                demand_coords, distance_type
            )
        }
        
        # Generate interpretation
        interpretation = generate_interpretation(
            total_distance, avg_distance, facility_loads,
            n_facilities, n_demands, iterations, converged, algorithm
        )
        
        return LocationAllocationResponse(
            success=True,
            total_distance=float(total_distance),
            avg_distance_per_demand=float(avg_distance),
            facilities=facilities,
            assignments=assignment_dict,
            facility_loads=facility_loads,
            facility_total_demand=facility_total_demand,
            iterations=iterations,
            converged=converged,
            problem={
                "n_demand_points": n_demands,
                "n_facilities": n_facilities,
                "total_demand": float(total_demand),
                "distance_type": distance_type,
                "algorithm": algorithm,
                "max_iterations": max_iterations
            },
            plots=plots,
            interpretation=interpretation,
            algorithm_info={
                "name": "K-Means" if algorithm == "kmeans" else "Alternate Location-Allocation",
                "iterations": iterations,
                "converged": converged,
                "method": "clustering" if algorithm == "kmeans" else "iterative"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")
