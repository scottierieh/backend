"""
Ant Colony Optimization (ACO) Router for FastAPI
Solving Traveling Salesman Problem with Folium map visualization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from folium import plugins
import io
import base64
import time
import warnings
import math

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()

# ============ PRESET CITIES ============
US_MAJOR_CITIES = [
    {"name": "New York", "lat": 40.7128, "lng": -74.0060, "state": "NY"},
    {"name": "Los Angeles", "lat": 34.0522, "lng": -118.2437, "state": "CA"},
    {"name": "Chicago", "lat": 41.8781, "lng": -87.6298, "state": "IL"},
    {"name": "Houston", "lat": 29.7604, "lng": -95.3698, "state": "TX"},
    {"name": "Phoenix", "lat": 33.4484, "lng": -112.0740, "state": "AZ"},
    {"name": "Philadelphia", "lat": 39.9526, "lng": -75.1652, "state": "PA"},
    {"name": "San Antonio", "lat": 29.4241, "lng": -98.4936, "state": "TX"},
    {"name": "San Diego", "lat": 32.7157, "lng": -117.1611, "state": "CA"},
    {"name": "Dallas", "lat": 32.7767, "lng": -96.7970, "state": "TX"},
    {"name": "San Francisco", "lat": 37.7749, "lng": -122.4194, "state": "CA"},
]

US_WEST_COAST = [
    {"name": "Seattle", "lat": 47.6062, "lng": -122.3321, "state": "WA"},
    {"name": "Portland", "lat": 45.5152, "lng": -122.6784, "state": "OR"},
    {"name": "San Francisco", "lat": 37.7749, "lng": -122.4194, "state": "CA"},
    {"name": "Los Angeles", "lat": 34.0522, "lng": -118.2437, "state": "CA"},
    {"name": "San Diego", "lat": 32.7157, "lng": -117.1611, "state": "CA"},
    {"name": "Las Vegas", "lat": 36.1699, "lng": -115.1398, "state": "NV"},
    {"name": "Phoenix", "lat": 33.4484, "lng": -112.0740, "state": "AZ"},
]

US_EAST_COAST = [
    {"name": "Boston", "lat": 42.3601, "lng": -71.0589, "state": "MA"},
    {"name": "New York", "lat": 40.7128, "lng": -74.0060, "state": "NY"},
    {"name": "Philadelphia", "lat": 39.9526, "lng": -75.1652, "state": "PA"},
    {"name": "Baltimore", "lat": 39.2904, "lng": -76.6122, "state": "MD"},
    {"name": "Washington DC", "lat": 38.9072, "lng": -77.0369, "state": "DC"},
    {"name": "Richmond", "lat": 37.5407, "lng": -77.4360, "state": "VA"},
    {"name": "Charlotte", "lat": 35.2271, "lng": -80.8431, "state": "NC"},
    {"name": "Atlanta", "lat": 33.7490, "lng": -84.3880, "state": "GA"},
    {"name": "Miami", "lat": 25.7617, "lng": -80.1918, "state": "FL"},
]

TEXAS_CITIES = [
    {"name": "Houston", "lat": 29.7604, "lng": -95.3698, "state": "TX"},
    {"name": "San Antonio", "lat": 29.4241, "lng": -98.4936, "state": "TX"},
    {"name": "Dallas", "lat": 32.7767, "lng": -96.7970, "state": "TX"},
    {"name": "Austin", "lat": 30.2672, "lng": -97.7431, "state": "TX"},
    {"name": "Fort Worth", "lat": 32.7555, "lng": -97.3308, "state": "TX"},
    {"name": "El Paso", "lat": 31.7619, "lng": -106.4850, "state": "TX"},
    {"name": "Corpus Christi", "lat": 27.8006, "lng": -97.3964, "state": "TX"},
]

PRESET_OPTIONS = {
    "us_major": {"name": "US Major Cities", "cities": US_MAJOR_CITIES},
    "us_west": {"name": "US West Coast", "cities": US_WEST_COAST},
    "us_east": {"name": "US East Coast", "cities": US_EAST_COAST},
    "texas": {"name": "Texas Cities", "cities": TEXAS_CITIES},
}


# ============ PYDANTIC MODELS ============
class City(BaseModel):
    name: str
    lat: float
    lng: float
    state: Optional[str] = None


class ACOParams(BaseModel):
    n_ants: int = 20
    n_iterations: int = 100
    alpha: float = 1.0  # Pheromone importance
    beta: float = 2.0   # Heuristic importance (distance)
    evaporation_rate: float = 0.5
    q: float = 100.0    # Pheromone deposit factor


class ACORequest(BaseModel):
    cities: List[City]
    params: ACOParams = ACOParams()
    preset: Optional[str] = None  # If provided, use preset cities


class CompareRequest(BaseModel):
    cities: List[City]
    param_sets: List[ACOParams]  # Multiple parameter sets to compare


# ============ UTILITY FUNCTIONS ============
def _to_native_type(obj):
    """Convert numpy types to native Python types"""
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
    """Convert matplotlib figure to base64"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in miles using Haversine formula"""
    R = 3959  # Earth's radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def calculate_distance_matrix(cities: List[Dict]) -> np.ndarray:
    """Calculate distance matrix between all cities"""
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = haversine_distance(
                    cities[i]['lat'], cities[i]['lng'],
                    cities[j]['lat'], cities[j]['lng']
                )
    
    return dist_matrix


# ============ ACO ALGORITHM ============
class AntColonyOptimizer:
    def __init__(self, dist_matrix: np.ndarray, params: ACOParams):
        self.dist_matrix = dist_matrix
        self.n_cities = len(dist_matrix)
        self.params = params
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((self.n_cities, self.n_cities))
        
        # Heuristic information (inverse of distance)
        with np.errstate(divide='ignore'):
            self.heuristic = 1 / (dist_matrix + np.eye(self.n_cities))
        self.heuristic[self.heuristic == np.inf] = 0
        
        # Track convergence
        self.convergence_history = []
        self.best_path = None
        self.best_distance = float('inf')
        
    def _select_next_city(self, current: int, visited: set) -> int:
        """Select next city based on probability"""
        unvisited = [c for c in range(self.n_cities) if c not in visited]
        
        if not unvisited:
            return -1
        
        # Calculate probabilities
        probs = []
        for city in unvisited:
            pheromone = self.pheromone[current][city] ** self.params.alpha
            heuristic = self.heuristic[current][city] ** self.params.beta
            probs.append(pheromone * heuristic)
        
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return np.random.choice(unvisited, p=probs)
    
    def _construct_path(self, start: int) -> tuple:
        """Construct a path for one ant"""
        path = [start]
        visited = {start}
        
        while len(path) < self.n_cities:
            next_city = self._select_next_city(path[-1], visited)
            if next_city == -1:
                break
            path.append(next_city)
            visited.add(next_city)
        
        # Calculate total distance (including return to start)
        total_dist = sum(self.dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        total_dist += self.dist_matrix[path[-1]][path[0]]  # Return to start
        
        return path, total_dist
    
    def _update_pheromone(self, all_paths: List[tuple]):
        """Update pheromone levels"""
        # Evaporation
        self.pheromone *= (1 - self.params.evaporation_rate)
        
        # Add new pheromone
        for path, distance in all_paths:
            pheromone_deposit = self.params.q / distance
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i+1]] += pheromone_deposit
                self.pheromone[path[i+1]][path[i]] += pheromone_deposit
            # Return edge
            self.pheromone[path[-1]][path[0]] += pheromone_deposit
            self.pheromone[path[0]][path[-1]] += pheromone_deposit
    
    def optimize(self) -> Dict:
        """Run ACO optimization"""
        for iteration in range(self.params.n_iterations):
            all_paths = []
            
            # Each ant constructs a path
            for ant in range(self.params.n_ants):
                start = np.random.randint(self.n_cities)
                path, distance = self._construct_path(start)
                all_paths.append((path, distance))
                
                # Update best
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # Update pheromone
            self._update_pheromone(all_paths)
            
            # Track convergence
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_distance': self.best_distance,
                'avg_distance': np.mean([d for _, d in all_paths]),
                'worst_distance': max(d for _, d in all_paths)
            })
        
        return {
            'best_path': self.best_path,
            'best_distance': self.best_distance,
            'convergence_history': self.convergence_history,
            'final_pheromone': self.pheromone
        }


# ============ VISUALIZATION FUNCTIONS ============
def create_folium_map(cities: List[Dict], best_path: List[int], total_distance: float) -> str:
    """Create interactive Folium map with route"""
    # Calculate center
    center_lat = np.mean([c['lat'] for c in cities])
    center_lng = np.mean([c['lng'] for c in cities])
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=4,
        tiles='cartodbpositron'
    )
    
    # Color palette for route segments
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
              '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
    
    # Add route lines
    path_coords = []
    for i, idx in enumerate(best_path):
        city = cities[idx]
        path_coords.append([city['lat'], city['lng']])
    # Close the loop
    path_coords.append([cities[best_path[0]]['lat'], cities[best_path[0]]['lng']])
    
    # Draw route with gradient colors
    for i in range(len(path_coords) - 1):
        folium.PolyLine(
            [path_coords[i], path_coords[i + 1]],
            weight=4,
            color=colors[i % len(colors)],
            opacity=0.8,
            popup=f"Segment {i + 1}"
        ).add_to(m)
    
    # Add city markers
    for i, idx in enumerate(best_path):
        city = cities[idx]
        
        # Create custom icon with order number
        icon_html = f'''
            <div style="
                background-color: {'#22c55e' if i == 0 else '#3b82f6'};
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">{i + 1}</div>
        '''
        
        folium.Marker(
            [city['lat'], city['lng']],
            popup=folium.Popup(
                f"<b>{i + 1}. {city['name']}</b><br>"
                f"State: {city.get('state', 'N/A')}<br>"
                f"Lat: {city['lat']:.4f}<br>"
                f"Lng: {city['lng']:.4f}",
                max_width=200
            ),
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(30, 30),
                icon_anchor=(15, 15)
            )
        ).add_to(m)
    
    # Add title
    title_html = f'''
        <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
                    background: white; padding: 10px 15px; border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-family: Arial;">
            <b>ACO Optimal Route</b><br>
            <span style="color: #666;">Total Distance: {total_distance:,.1f} miles</span><br>
            <span style="color: #666;">Cities: {len(cities)}</span>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    return m._repr_html_()


def create_convergence_chart(history: List[Dict]) -> str:
    """Create convergence graph showing optimization progress"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = [h['iteration'] for h in history]
    best = [h['best_distance'] for h in history]
    avg = [h['avg_distance'] for h in history]
    worst = [h['worst_distance'] for h in history]
    
    ax.fill_between(iterations, worst, best, alpha=0.2, color='#3b82f6', label='Range')
    ax.plot(iterations, best, 'g-', linewidth=2, label='Best Distance', marker='o', markersize=3)
    ax.plot(iterations, avg, 'b--', linewidth=1.5, label='Average Distance', alpha=0.7)
    ax.plot(iterations, worst, 'r:', linewidth=1, label='Worst Distance', alpha=0.5)
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Distance (miles)', fontsize=11)
    ax.set_title('ACO Convergence Over Iterations', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = ((history[0]['best_distance'] - history[-1]['best_distance']) / 
                   history[0]['best_distance'] * 100)
    ax.annotate(f'Improvement: {improvement:.1f}%',
                xy=(iterations[-1], best[-1]),
                xytext=(iterations[-1] * 0.7, best[0] * 0.95),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_pheromone_heatmap(pheromone: np.ndarray, city_names: List[str]) -> str:
    """Create pheromone intensity heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize pheromone for better visualization
    pheromone_norm = pheromone / pheromone.max()
    
    im = ax.imshow(pheromone_norm, cmap='YlOrRd', aspect='auto')
    
    # Add labels
    ax.set_xticks(range(len(city_names)))
    ax.set_yticks(range(len(city_names)))
    ax.set_xticklabels(city_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(city_names, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pheromone Intensity (normalized)', fontsize=10)
    
    ax.set_title('Pheromone Trail Intensity Between Cities', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_distance_matrix_chart(dist_matrix: np.ndarray, city_names: List[str]) -> str:
    """Create distance matrix visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(dist_matrix, cmap='Blues', aspect='auto')
    
    # Add labels
    ax.set_xticks(range(len(city_names)))
    ax.set_yticks(range(len(city_names)))
    ax.set_xticklabels(city_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(city_names, fontsize=9)
    
    # Add distance values
    for i in range(len(city_names)):
        for j in range(len(city_names)):
            if i != j:
                ax.text(j, i, f'{dist_matrix[i][j]:.0f}',
                       ha='center', va='center', fontsize=7,
                       color='white' if dist_matrix[i][j] > dist_matrix.max()/2 else 'black')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (miles)', fontsize=10)
    
    ax.set_title('Distance Matrix Between Cities', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_route_bar_chart(cities: List[Dict], best_path: List[int], dist_matrix: np.ndarray) -> str:
    """Create bar chart showing segment distances"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    segments = []
    distances = []
    
    for i in range(len(best_path)):
        from_idx = best_path[i]
        to_idx = best_path[(i + 1) % len(best_path)]
        from_city = cities[from_idx]['name']
        to_city = cities[to_idx]['name']
        distance = dist_matrix[from_idx][to_idx]
        
        segments.append(f"{from_city} → {to_city}")
        distances.append(distance)
    
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
              '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
    bar_colors = [colors[i % len(colors)] for i in range(len(segments))]
    
    bars = ax.barh(segments, distances, color=bar_colors, edgecolor='white', linewidth=1)
    
    for bar, dist in zip(bars, distances):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{dist:.0f} mi', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Distance (miles)', fontsize=11)
    ax.set_title('Route Segment Distances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(cities: List[Dict], best_path: List[int], 
                          total_distance: float, convergence: List[Dict],
                          dist_matrix: np.ndarray) -> List[Dict]:
    """Generate key insights from ACO results"""
    insights = []
    
    # Distance improvement
    if convergence:
        initial = convergence[0]['best_distance']
        final = convergence[-1]['best_distance']
        improvement = ((initial - final) / initial * 100)
        
        if improvement > 20:
            insights.append({
                'title': f'Significant Optimization: {improvement:.1f}% improvement',
                'description': f'Distance reduced from {initial:,.0f} to {final:,.0f} miles.',
                'status': 'positive'
            })
        elif improvement > 5:
            insights.append({
                'title': f'Moderate Optimization: {improvement:.1f}% improvement',
                'description': 'Algorithm found better routes through iterations.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Quick Convergence',
                'description': 'Algorithm found near-optimal solution early. Consider more iterations for complex problems.',
                'status': 'warning'
            })
    
    # Route efficiency
    avg_segment = total_distance / len(cities)
    insights.append({
        'title': f'Average Segment: {avg_segment:,.0f} miles',
        'description': f'Total {len(cities)} cities connected with {total_distance:,.0f} miles total distance.',
        'status': 'neutral'
    })
    
    # Longest/shortest segments
    segment_distances = []
    for i in range(len(best_path)):
        from_idx = best_path[i]
        to_idx = best_path[(i + 1) % len(best_path)]
        segment_distances.append({
            'from': cities[from_idx]['name'],
            'to': cities[to_idx]['name'],
            'distance': dist_matrix[from_idx][to_idx]
        })
    
    longest = max(segment_distances, key=lambda x: x['distance'])
    shortest = min(segment_distances, key=lambda x: x['distance'])
    
    insights.append({
        'title': f'Longest Segment: {longest["distance"]:,.0f} miles',
        'description': f'{longest["from"]} → {longest["to"]}',
        'status': 'warning' if longest['distance'] > avg_segment * 2 else 'neutral'
    })
    
    insights.append({
        'title': f'Shortest Segment: {shortest["distance"]:,.0f} miles',
        'description': f'{shortest["from"]} → {shortest["to"]}',
        'status': 'positive'
    })
    
    return insights


# ============ API ENDPOINTS ============
@router.get("/presets")
async def get_presets() -> Dict[str, Any]:
    """Get available preset city configurations"""
    return {
        'presets': {
            key: {
                'name': value['name'],
                'city_count': len(value['cities']),
                'cities': value['cities']
            }
            for key, value in PRESET_OPTIONS.items()
        }
    }


@router.post("/ant-colony")
async def run_aco(request: ACORequest) -> Dict[str, Any]:
    """Run Ant Colony Optimization for TSP"""
    try:
        start_time = time.time()
        
        # Use preset cities if specified
        if request.preset and request.preset in PRESET_OPTIONS:
            cities_data = PRESET_OPTIONS[request.preset]['cities']
        else:
            cities_data = [c.dict() for c in request.cities]
        
        if len(cities_data) < 3:
            raise HTTPException(status_code=400, detail="At least 3 cities required")
        
        if len(cities_data) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 cities allowed")
        
        # Calculate distance matrix
        dist_matrix = calculate_distance_matrix(cities_data)
        
        # Run ACO
        aco = AntColonyOptimizer(dist_matrix, request.params)
        result = aco.optimize()
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        best_path = result['best_path']
        best_distance = result['best_distance']
        
        # Create ordered city list
        ordered_cities = [cities_data[i] for i in best_path]
        city_names = [c['name'] for c in cities_data]
        
        # Generate visualizations
        visualizations = {
            'map_html': create_folium_map(cities_data, best_path, best_distance),
            'convergence_chart': create_convergence_chart(result['convergence_history']),
            'pheromone_heatmap': create_pheromone_heatmap(result['final_pheromone'], city_names),
            'distance_matrix': create_distance_matrix_chart(dist_matrix, city_names),
            'segment_chart': create_route_bar_chart(cities_data, best_path, dist_matrix)
        }
        
        # Generate insights
        key_insights = generate_key_insights(
            cities_data, best_path, best_distance,
            result['convergence_history'], dist_matrix
        )
        
        # Prepare results
        results = {
            'best_path': [cities_data[i]['name'] for i in best_path],
            'best_path_indices': [int(i) for i in best_path],
            'ordered_cities': [{k: _to_native_type(v) for k, v in c.items()} for c in ordered_cities],
            'total_distance': _to_native_type(best_distance),
            'num_cities': len(cities_data),
            'convergence_history': [
                {k: _to_native_type(v) for k, v in h.items()}
                for h in result['convergence_history']
            ],
            'segment_distances': [
                {
                    'from': cities_data[best_path[i]]['name'],
                    'to': cities_data[best_path[(i + 1) % len(best_path)]]['name'],
                    'distance': _to_native_type(dist_matrix[best_path[i]][best_path[(i + 1) % len(best_path)]])
                }
                for i in range(len(best_path))
            ]
        }
        
        summary = {
            'algorithm': 'Ant Colony Optimization',
            'n_ants': request.params.n_ants,
            'n_iterations': request.params.n_iterations,
            'alpha': request.params.alpha,
            'beta': request.params.beta,
            'evaporation_rate': request.params.evaporation_rate,
            'total_distance': _to_native_type(best_distance),
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
        raise HTTPException(status_code=500, detail=f"ACO optimization failed: {str(e)}")


@router.post("/ant-colony/compare")
async def compare_aco_params(request: CompareRequest) -> Dict[str, Any]:
    """Compare multiple ACO parameter configurations"""
    try:
        cities_data = [c.dict() for c in request.cities]
        
        if len(cities_data) < 3:
            raise HTTPException(status_code=400, detail="At least 3 cities required")
        
        dist_matrix = calculate_distance_matrix(cities_data)
        
        comparison_results = []
        
        for i, params in enumerate(request.param_sets):
            start_time = time.time()
            
            aco = AntColonyOptimizer(dist_matrix, params)
            result = aco.optimize()
            
            solve_time_ms = int((time.time() - start_time) * 1000)
            
            comparison_results.append({
                'config_id': i + 1,
                'params': {
                    'n_ants': params.n_ants,
                    'n_iterations': params.n_iterations,
                    'alpha': params.alpha,
                    'beta': params.beta,
                    'evaporation_rate': params.evaporation_rate
                },
                'best_distance': _to_native_type(result['best_distance']),
                'best_path': [cities_data[idx]['name'] for idx in result['best_path']],
                'solve_time_ms': solve_time_ms,
                'convergence_final': result['convergence_history'][-1] if result['convergence_history'] else None
            })
        
        # Find best configuration
        best_config = min(comparison_results, key=lambda x: x['best_distance'])
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distance comparison
        ax1 = axes[0]
        configs = [f"Config {r['config_id']}" for r in comparison_results]
        distances = [r['best_distance'] for r in comparison_results]
        colors = ['#22c55e' if r['config_id'] == best_config['config_id'] else '#3b82f6' 
                  for r in comparison_results]
        
        bars = ax1.bar(configs, distances, color=colors, edgecolor='white', linewidth=1)
        ax1.set_ylabel('Best Distance (miles)', fontsize=11)
        ax1.set_title('Distance Comparison', fontsize=12, fontweight='bold')
        
        for bar, dist in zip(bars, distances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{dist:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Time comparison
        ax2 = axes[1]
        times = [r['solve_time_ms'] for r in comparison_results]
        ax2.bar(configs, times, color='#f59e0b', edgecolor='white', linewidth=1)
        ax2.set_ylabel('Solve Time (ms)', fontsize=11)
        ax2.set_title('Computation Time Comparison', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        comparison_chart = _fig_to_base64(fig)
        
        return {
            'success': True,
            'comparison_results': comparison_results,
            'best_config': best_config,
            'comparison_chart': comparison_chart,
            'summary': {
                'num_configs': len(request.param_sets),
                'best_distance': best_config['best_distance'],
                'best_config_id': best_config['config_id']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
