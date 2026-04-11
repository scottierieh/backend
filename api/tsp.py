"""
Traveling Salesman Problem (TSP) Router for FastAPI
Using Google OR-Tools routing library
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import time
import warnings
import math

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class TSPRequest(BaseModel):
    data: List[Dict[str, Any]]
    location_id_col: str
    name_col: Optional[str] = None
    x_col: str
    y_col: str
    algorithm: Literal["automatic", "greedy", "christofides", "savings"] = "automatic"
    return_to_start: bool = True


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


NODE_COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def calculate_distance_matrix(locations: List[Dict]) -> List[List[int]]:
    """Calculate Euclidean distance matrix (scaled to integers)"""
    n = len(locations)
    scale = 100  # Scale for integer conversion
    
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                dx = locations[i]['x'] - locations[j]['x']
                dy = locations[i]['y'] - locations[j]['y']
                dist = int(math.sqrt(dx*dx + dy*dy) * scale)
                row.append(dist)
        matrix.append(row)
    
    return matrix


def solve_tsp(distance_matrix: List[List[int]], algorithm: str) -> List[int]:
    """Solve TSP using OR-Tools"""
    n = len(distance_matrix)
    
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Select search parameters based on algorithm
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    if algorithm == "greedy":
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
    elif algorithm == "christofides":
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES
        )
    elif algorithm == "savings":
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        )
    else:  # automatic
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 5
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return list(range(n))  # Return naive order if no solution
    
    # Extract route
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    
    return route


def calculate_naive_distance(locations: List[Dict]) -> float:
    """Calculate total distance for naive sequential ordering"""
    total = 0
    for i in range(len(locations) - 1):
        dx = locations[i+1]['x'] - locations[i]['x']
        dy = locations[i+1]['y'] - locations[i]['y']
        total += math.sqrt(dx*dx + dy*dy)
    # Return to start
    dx = locations[0]['x'] - locations[-1]['x']
    dy = locations[0]['y'] - locations[-1]['y']
    total += math.sqrt(dx*dx + dy*dy)
    return total


def create_route_map(locations: List[Dict], route_order: List[int]) -> str:
    """Create route visualization"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get ordered locations
    ordered = [locations[i] for i in route_order]
    
    # Plot route lines
    for i in range(len(ordered)):
        j = (i + 1) % len(ordered)
        ax.plot([ordered[i]['x'], ordered[j]['x']], 
               [ordered[i]['y'], ordered[j]['y']],
               'b-', linewidth=2, alpha=0.6, zorder=1)
        
        # Add arrow in middle
        mid_x = (ordered[i]['x'] + ordered[j]['x']) / 2
        mid_y = (ordered[i]['y'] + ordered[j]['y']) / 2
        dx = ordered[j]['x'] - ordered[i]['x']
        dy = ordered[j]['y'] - ordered[i]['y']
        ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                   xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
    
    # Plot locations
    for i, loc in enumerate(ordered):
        color = NODE_COLORS[i % len(NODE_COLORS)]
        ax.scatter(loc['x'], loc['y'], s=200, c=color, zorder=3, edgecolors='white', linewidths=2)
        ax.annotate(str(i + 1), (loc['x'], loc['y']), ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white', zorder=4)
        
        # Location name
        name = loc.get('name', loc['location_id'])
        ax.annotate(name, (loc['x'], loc['y']), xytext=(5, 5),
                   textcoords='offset points', fontsize=8, alpha=0.8)
    
    # Highlight start
    ax.scatter(ordered[0]['x'], ordered[0]['y'], s=300, c='none', 
              edgecolors='green', linewidths=3, zorder=5)
    
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)
    ax.set_title('Optimal TSP Route', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_distance_chart(distances: List[float]) -> str:
    """Create leg distance chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"Leg {i+1}" for i in range(len(distances))]
    colors = [NODE_COLORS[i % len(NODE_COLORS)] for i in range(len(distances))]
    
    bars = ax.bar(labels, distances, color=colors, edgecolor='white', linewidth=1)
    
    avg_dist = np.mean(distances)
    ax.axhline(y=avg_dist, color='gray', linestyle='--', alpha=0.7, label=f'Avg: {avg_dist:.1f}')
    
    for bar, dist in zip(bars, distances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{dist:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Distance', fontsize=11)
    ax.set_xlabel('Leg', fontsize=11)
    ax.set_title('Distance by Leg', fontsize=14, fontweight='bold')
    ax.legend()
    
    if len(labels) > 10:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_leg_analysis(distances: List[float]) -> str:
    """Create leg analysis chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(distances, bins=min(10, len(distances)), color='#3b82f6', 
             edgecolor='white', linewidth=1)
    ax1.axvline(x=np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.1f}')
    ax1.axvline(x=np.median(distances), color='green', linestyle='--', label=f'Median: {np.median(distances):.1f}')
    ax1.set_xlabel('Distance', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Leg Distance Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Cumulative distance
    ax2 = axes[1]
    cumulative = np.cumsum(distances)
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 'b-', linewidth=2, marker='o')
    ax2.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.3)
    ax2.set_xlabel('Leg Number', fontsize=11)
    ax2.set_ylabel('Cumulative Distance', fontsize=11)
    ax2.set_title('Cumulative Distance Progress', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_comparison_chart(optimized_dist: float, naive_dist: float) -> str:
    """Create comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Optimized\n(OR-Tools)', 'Naive\n(Sequential)']
    distances = [optimized_dist, naive_dist]
    colors = ['#22c55e', '#ef4444']
    
    bars = ax.bar(methods, distances, color=colors, edgecolor='white', linewidth=2)
    
    for bar, dist in zip(bars, distances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{dist:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    improvement = ((naive_dist - optimized_dist) / naive_dist) * 100
    ax.annotate(f'{improvement:.0f}% savings', 
               xy=(0, optimized_dist), xytext=(0.5, (optimized_dist + naive_dist) / 2),
               ha='center', fontsize=11, color='green',
               arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    
    ax.set_ylabel('Total Distance', fontsize=11)
    ax.set_title('Optimized vs Naive Route Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(total_distance: float, distances: List[float],
                          improvement: float, num_locations: int) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Improvement insight
    if improvement > 20:
        insights.append({
            'title': f'Significant Optimization: {improvement:.0f}% Improvement',
            'description': 'The optimized route is significantly shorter than naive sequential ordering.',
            'status': 'positive'
        })
    elif improvement > 0:
        insights.append({
            'title': f'Route Optimized: {improvement:.0f}% Improvement',
            'description': 'Optimization achieved measurable distance savings.',
            'status': 'positive'
        })
    
    # Leg variance insight
    if distances:
        variance = np.std(distances)
        avg = np.mean(distances)
        cv = (variance / avg) * 100 if avg > 0 else 0
        
        if cv > 50:
            insights.append({
                'title': 'High Leg Variance',
                'description': f'Leg distances vary significantly (CV: {cv:.0f}%). Some legs are much longer than others.',
                'status': 'warning'
            })
        else:
            insights.append({
                'title': 'Balanced Route',
                'description': f'Leg distances are relatively uniform (CV: {cv:.0f}%).',
                'status': 'neutral'
            })
    
    # Scale insight
    insights.append({
        'title': f'{num_locations} Locations Optimized',
        'description': f'Total route distance: {total_distance:.1f} units across {len(distances)} legs.',
        'status': 'neutral'
    })
    
    return insights


@router.post("/tsp")
async def run_tsp(request: TSPRequest) -> Dict[str, Any]:
    """Run TSP optimization"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.location_id_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Location ID column '{request.location_id_col}' not found")
        if request.x_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"X column '{request.x_col}' not found")
        if request.y_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Y column '{request.y_col}' not found")
        
        # Prepare locations
        locations = []
        for _, row in df.iterrows():
            loc = {
                'location_id': str(row[request.location_id_col]),
                'x': float(row[request.x_col]),
                'y': float(row[request.y_col])
            }
            if request.name_col and request.name_col in df.columns:
                loc['name'] = str(row[request.name_col])
            locations.append(loc)
        
        if len(locations) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 locations for TSP")
        
        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(locations)
        
        # Solve TSP
        route_order = solve_tsp(distance_matrix, request.algorithm)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Calculate distances for each leg
        scale = 100
        distances = []
        for i in range(len(route_order)):
            j = (i + 1) % len(route_order)
            dist = distance_matrix[route_order[i]][route_order[j]] / scale
            distances.append(dist)
        
        if not request.return_to_start:
            distances = distances[:-1]
        
        total_distance = sum(distances)
        
        # Calculate naive distance for comparison
        naive_distance = calculate_naive_distance(locations)
        improvement = ((naive_distance - total_distance) / naive_distance * 100) if naive_distance > 0 else 0
        
        # Create ordered route
        route = [locations[i] for i in route_order]
        for i, loc in enumerate(route):
            loc['order'] = i + 1
        
        # Metrics
        metrics = {
            'avg_leg_distance': np.mean(distances) if distances else 0,
            'min_leg_distance': min(distances) if distances else 0,
            'max_leg_distance': max(distances) if distances else 0,
            'improvement_vs_naive': improvement
        }
        
        # Visualizations
        visualizations = {
            'route_map': create_route_map(locations, route_order),
            'distance_chart': create_distance_chart(distances),
            'leg_analysis': create_leg_analysis(distances),
            'comparison_chart': create_comparison_chart(total_distance, naive_distance)
        }
        
        # Key insights
        key_insights = generate_key_insights(total_distance, distances, improvement, len(locations))
        
        # Results
        results = {
            'route': [{k: _to_native_type(v) for k, v in loc.items()} for loc in route],
            'total_distance': total_distance,
            'num_locations': len(locations),
            'distances': [_to_native_type(d) for d in distances],
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'algorithm': request.algorithm.title(),
            'num_locations': len(locations),
            'total_distance': total_distance,
            'solve_time_ms': solve_time_ms
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TSP optimization failed: {str(e)}")
