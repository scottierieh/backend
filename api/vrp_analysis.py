"""
VRP (Vehicle Routing Problem) Analysis Router for FastAPI
Using Google OR-Tools for optimization
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
import math
import time
import warnings

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class VRPRequest(BaseModel):
    data: List[Dict[str, Any]]
    name_col: Optional[str] = None
    lat_col: str
    lng_col: str
    demand_col: Optional[str] = None
    time_start_col: Optional[str] = None
    time_end_col: Optional[str] = None
    service_time_col: Optional[str] = None
    problem_type: Literal["vrp", "cvrp", "vrptw"] = "vrp"
    num_vehicles: int = 4
    vehicle_capacity: int = 100
    depot_index: int = 0
    max_distance: Optional[int] = None
    max_time: Optional[int] = None


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


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate Haversine distance between two points in km"""
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def create_distance_matrix(locations: List[Dict]) -> List[List[int]]:
    """Create distance matrix from location coordinates"""
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = haversine_distance(
                    locations[i]['lat'], locations[i]['lng'],
                    locations[j]['lat'], locations[j]['lng']
                )
                matrix[i][j] = int(dist * 1000)  # meters for OR-Tools
    return matrix


def create_time_matrix(distance_matrix: List[List[int]], avg_speed_kmh: float = 40) -> List[List[int]]:
    """Create time matrix from distance matrix (in minutes)"""
    n = len(distance_matrix)
    time_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_km = distance_matrix[i][j] / 1000
            time_min = (dist_km / avg_speed_kmh) * 60
            time_matrix[i][j] = int(time_min)
    return time_matrix


def parse_time_window(time_str: str) -> int:
    """Convert time string (HH:MM) to minutes from midnight"""
    try:
        parts = str(time_str).split(':')
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        return hours * 60 + minutes
    except:
        return 0


def solve_vrp(data_model: Dict) -> Dict:
    """Solve basic VRP"""
    manager = pywrapcp.RoutingIndexManager(
        len(data_model['distance_matrix']),
        data_model['num_vehicles'],
        data_model['depot']
    )
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data_model['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    routing.AddDimension(
        transit_callback_index, 0,
        data_model.get('max_distance', 100000000),
        True, 'Distance'
    )
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30
    
    solution = routing.SolveWithParameters(search_parameters)
    return extract_solution(manager, routing, solution, data_model)


def solve_cvrp(data_model: Dict) -> Dict:
    """Solve Capacitated VRP"""
    manager = pywrapcp.RoutingIndexManager(
        len(data_model['distance_matrix']),
        data_model['num_vehicles'],
        data_model['depot']
    )
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data_model['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data_model['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0,
        data_model['vehicle_capacities'],
        True, 'Capacity'
    )
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30
    
    solution = routing.SolveWithParameters(search_parameters)
    return extract_solution(manager, routing, solution, data_model)


def solve_vrptw(data_model: Dict) -> Dict:
    """Solve VRP with Time Windows"""
    manager = pywrapcp.RoutingIndexManager(
        len(data_model['distance_matrix']),
        data_model['num_vehicles'],
        data_model['depot']
    )
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data_model['time_matrix'][from_node][to_node] + data_model['service_times'][from_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    routing.AddDimension(
        transit_callback_index, 60, 1440, False, 'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    for location_idx, time_window in enumerate(data_model['time_windows']):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    
    depot_idx = data_model['depot']
    for vehicle_id in range(data_model['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data_model['time_windows'][depot_idx][0],
            data_model['time_windows'][depot_idx][1]
        )
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30
    
    solution = routing.SolveWithParameters(search_parameters)
    return extract_solution(manager, routing, solution, data_model, time_dimension)


def extract_solution(manager, routing, solution, data_model, time_dimension=None) -> Dict:
    """Extract solution from OR-Tools solver"""
    if not solution:
        return {'routes': [], 'status': 'NO_SOLUTION'}
    
    routes = []
    total_distance = 0
    total_time = 0
    total_load = 0
    
    for vehicle_id in range(data_model['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_indices = []
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_indices.append(node_index)
            if 'demands' in data_model:
                route_load += data_model['demands'][node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        route_indices.append(manager.IndexToNode(index))
        
        if len(route_indices) > 2:
            route_distance_km = route_distance / 1000
            route_time = round(route_distance_km / 40 * 60, 1)  # 반올림하여 소수점 1자리까지
            
            routes.append({
                'vehicle_id': vehicle_id + 1,
                'route_indices': route_indices,
                'route_names': [data_model['names'][i] for i in route_indices],
                'total_distance': route_distance_km,
                'total_time': route_time,
                'total_load': route_load,
                'num_stops': len(route_indices)
            })
            
            total_distance += route_distance_km
            total_time += route_time
            total_load += route_load
    
    return {
        'routes': routes,
        'total_distance': total_distance,
        'total_time': total_time,
        'total_load': total_load,
        'status': 'SUCCESS'
    }


def create_route_map(locations: List[Dict], routes: List[Dict]) -> str:
    """Create route visualization"""
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
              '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
    
    lats = [loc['lat'] for loc in locations]
    lngs = [loc['lng'] for loc in locations]
    names = [loc['name'] for loc in locations]
    
    ax.scatter(lngs[0], lats[0], c='red', s=200, marker='s', zorder=5, label='Depot')
    ax.annotate(names[0], (lngs[0], lats[0]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.scatter(lngs[1:], lats[1:], c='gray', s=80, alpha=0.5, zorder=3)
    
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        route_lats = [locations[i]['lat'] for i in route['route_indices']]
        route_lngs = [locations[i]['lng'] for i in route['route_indices']]
        ax.plot(route_lngs, route_lats, 'o-', color=color, linewidth=2, markersize=8,
                label=f"V{route['vehicle_id']} ({route['total_distance']:.1f}km)", zorder=4)
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Optimized Vehicle Routes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_vehicle_distance_chart(routes: List[Dict]) -> str:
    """Create vehicle distance comparison chart"""
    if not routes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No routes', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    vehicles = [f"V{r['vehicle_id']}" for r in routes]
    distances = [r['total_distance'] for r in routes]
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
              '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(vehicles, distances, color=[colors[i % len(colors)] for i in range(len(routes))])
    
    for bar, dist in zip(bars, distances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{dist:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Distance (km)', fontsize=11)
    ax.set_title('Distance by Vehicle', fontsize=14, fontweight='bold')
    avg_dist = np.mean(distances)
    ax.axhline(y=avg_dist, color='gray', linestyle='--', alpha=0.7, label=f'Avg: {avg_dist:.1f}km')
    ax.legend()
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_vehicle_load_chart(routes: List[Dict], capacity: int) -> str:
    """Create vehicle load comparison chart"""
    if not routes or all(r['total_load'] == 0 for r in routes):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No load data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    vehicles = [f"V{r['vehicle_id']}" for r in routes]
    loads = [r['total_load'] for r in routes]
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
              '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(vehicles, loads, color=[colors[i % len(colors)] for i in range(len(routes))])
    
    for bar, load in zip(bars, loads):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{load}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=capacity, color='red', linestyle='--', alpha=0.7, label=f'Capacity: {capacity}')
    ax.set_ylabel('Load (units)', fontsize=11)
    ax.set_title('Load by Vehicle', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_route_timeline(routes: List[Dict]) -> str:
    """Create route timeline chart"""
    if not routes:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No routes', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
              '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        stops = route['num_stops']
        times = np.linspace(0, route['total_time'], stops)
        ax.barh(idx, route['total_time'], left=0, height=0.5, color=color, alpha=0.7,
                label=f"V{route['vehicle_id']}: {route['num_stops']} stops")
        
        for i, t in enumerate(times[:-1]):
            ax.plot(t, idx, 'o', color='white', markersize=6)
    
    ax.set_yticks(range(len(routes)))
    ax.set_yticklabels([f"Vehicle {r['vehicle_id']}" for r in routes])
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_title('Route Timeline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(routes: List[Dict], metrics: Dict, num_locations: int, num_vehicles: int) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    vehicles_used = len(routes)
    if vehicles_used < num_vehicles:
        insights.append({
            'title': f'Fleet Efficiency: {vehicles_used}/{num_vehicles} vehicles used',
            'description': f'Only {vehicles_used} vehicles needed. Consider reducing fleet size to save costs.',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': f'Full Fleet Utilized: {vehicles_used} vehicles',
            'description': 'All available vehicles are being used efficiently.',
            'status': 'neutral'
        })
    
    if metrics['distance_balance'] < 0.2:
        insights.append({
            'title': 'Well-Balanced Routes',
            'description': f"Route distances are evenly distributed (balance score: {metrics['distance_balance']:.2f}).",
            'status': 'positive'
        })
    else:
        insights.append({
            'title': 'Unbalanced Routes',
            'description': f"Consider redistributing stops for better balance (score: {metrics['distance_balance']:.2f}).",
            'status': 'warning'
        })
    
    avg_stops = metrics['avg_stops_per_vehicle']
    if avg_stops > 5:
        insights.append({
            'title': f'High Stop Density: {avg_stops:.1f} stops/vehicle',
            'description': 'Each vehicle serves many locations efficiently.',
            'status': 'positive'
        })
    
    return insights


@router.post("/vrp")
async def run_vrp_optimization(request: VRPRequest) -> Dict[str, Any]:
    """Run VRP optimization using Google OR-Tools"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        if request.lat_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Latitude column '{request.lat_col}' not found")
        if request.lng_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Longitude column '{request.lng_col}' not found")
        
        # Prepare locations
        locations = []
        for idx, row in df.iterrows():
            name = str(row[request.name_col]) if request.name_col and request.name_col in df.columns else f"Location {idx}"
            locations.append({
                'name': name,
                'lat': float(row[request.lat_col]),
                'lng': float(row[request.lng_col])
            })
        
        num_locations = len(locations)
        if num_locations < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 locations")
        
        # Create distance matrix
        distance_matrix = create_distance_matrix(locations)
        
        # Prepare data model
        data_model = {
            'distance_matrix': distance_matrix,
            'num_vehicles': request.num_vehicles,
            'depot': request.depot_index,
            'names': [loc['name'] for loc in locations],
            'max_distance': request.max_distance * 1000 if request.max_distance else 100000000
        }
        
        # Add demands for CVRP
        if request.problem_type in ['cvrp', 'vrptw'] and request.demand_col:
            demands = [int(row[request.demand_col]) if pd.notna(row[request.demand_col]) else 0 
                      for _, row in df.iterrows()]
            data_model['demands'] = demands
            data_model['vehicle_capacities'] = [request.vehicle_capacity] * request.num_vehicles
        
        # Add time windows for VRPTW
        if request.problem_type == 'vrptw' and request.time_start_col and request.time_end_col:
            time_windows = []
            service_times = []
            for _, row in df.iterrows():
                start = parse_time_window(str(row[request.time_start_col]))
                end = parse_time_window(str(row[request.time_end_col]))
                time_windows.append((start, end))
                if request.service_time_col and request.service_time_col in df.columns:
                    service_times.append(int(row[request.service_time_col]) if pd.notna(row[request.service_time_col]) else 10)
                else:
                    service_times.append(10)
            data_model['time_windows'] = time_windows
            data_model['service_times'] = service_times
            data_model['time_matrix'] = create_time_matrix(distance_matrix)
        
        # Solve
        if request.problem_type == 'cvrp' and 'demands' in data_model:
            solution = solve_cvrp(data_model)
        elif request.problem_type == 'vrptw' and 'time_windows' in data_model:
            solution = solve_vrptw(data_model)
        else:
            solution = solve_vrp(data_model)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        if solution['status'] != 'SUCCESS' or not solution['routes']:
            raise HTTPException(status_code=400, detail="No feasible solution found. Try adding more vehicles or relaxing constraints.")
        
        routes = solution['routes']
        vehicles_used = len(routes)
        
        # Calculate metrics
        distances = [r['total_distance'] for r in routes]
        metrics = {
            'avg_distance_per_vehicle': np.mean(distances) if distances else 0,
            'avg_time_per_vehicle': np.mean([r['total_time'] for r in routes]) if routes else 0,
            'avg_stops_per_vehicle': np.mean([r['num_stops'] for r in routes]) if routes else 0,
            'max_distance': max(distances) if distances else 0,
            'min_distance': min(distances) if distances else 0,
            'utilization_rate': (vehicles_used / request.num_vehicles) * 100,
            'distance_balance': (max(distances) - min(distances)) / max(distances) if distances and max(distances) > 0 else 0
        }
        
        # Find unassigned locations
        assigned = set()
        for route in routes:
            assigned.update(route['route_indices'])
        unassigned = [locations[i]['name'] for i in range(num_locations) if i not in assigned and i != request.depot_index]
        
        # Create visualizations
        visualizations = {
            'route_map': create_route_map(locations, routes),
            'vehicle_distances': create_vehicle_distance_chart(routes),
            'vehicle_loads': create_vehicle_load_chart(routes, request.vehicle_capacity),
            'route_timeline': create_route_timeline(routes)
        }
        
        # Generate insights
        key_insights = generate_key_insights(routes, metrics, num_locations, request.num_vehicles)
        
        # Prepare results
        results = {
            'routes': [{k: _to_native_type(v) for k, v in r.items()} for r in routes],
            'total_distance': _to_native_type(solution['total_distance']),
            'total_time': _to_native_type(solution['total_time']),
            'total_load': _to_native_type(solution['total_load']),
            'vehicles_used': vehicles_used,
            'unassigned': unassigned,
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'problem_type': request.problem_type,
            'num_locations': num_locations,
            'num_vehicles': request.num_vehicles,
            'depot': locations[request.depot_index]['name'],
            'total_distance': _to_native_type(solution['total_distance']),
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
        raise HTTPException(status_code=500, detail=f"VRP optimization failed: {str(e)}")
