"""
Network Optimization FastAPI Endpoint
Road network-based flow and routing optimization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
import networkx as nx
from collections import defaultdict

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

router = APIRouter()


class Node(BaseModel):
    """Network node (intersection/location)"""
    id: str
    name: str
    x: float
    y: float
    node_type: str = "regular"  # regular, source, sink, facility


class Edge(BaseModel):
    """Network edge (road segment)"""
    from_node: str
    to_node: str
    distance: float
    capacity: Optional[float] = None  # Maximum flow capacity
    cost: Optional[float] = None  # Cost per unit flow
    speed_limit: Optional[float] = None  # km/h


class NetworkOptimizationRequest(BaseModel):
    """Request model for Network Optimization"""
    nodes: List[Node] = Field(..., min_items=2)
    edges: List[Edge] = Field(..., min_items=1)
    problem_type: str = Field(..., pattern="^(shortest_path|max_flow|min_cost_flow|tsp)$")
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    flow_demand: Optional[float] = None


class NetworkOptimizationResponse(BaseModel):
    """Response model for Network Optimization"""
    success: bool
    problem_type: str
    solution: Dict[str, Any]
    network_stats: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]


def build_graph(nodes: List[Node], edges: List[Edge], directed: bool = True) -> nx.DiGraph:
    """Build NetworkX graph from nodes and edges"""
    G = nx.DiGraph() if directed else nx.Graph()
    
    # Add nodes with attributes
    for node in nodes:
        G.add_node(node.id, 
                  name=node.name, 
                  pos=(node.x, node.y),
                  node_type=node.node_type)
    
    # Add edges with attributes
    for edge in edges:
        G.add_edge(edge.from_node, edge.to_node,
                  distance=edge.distance,
                  capacity=edge.capacity or float('inf'),
                  cost=edge.cost or edge.distance,
                  speed_limit=edge.speed_limit)
    
    return G


def solve_shortest_path(
    G: nx.DiGraph,
    source: str,
    target: str
) -> Dict[str, Any]:
    """
    Solve shortest path problem using Dijkstra's algorithm
    """
    try:
        # Shortest path by distance
        path = nx.shortest_path(G, source, target, weight='distance')
        path_length = nx.shortest_path_length(G, source, target, weight='distance')
        
        # Calculate path details
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        total_cost = sum(G[u][v].get('cost', 0) for u, v in path_edges)
        
        # Alternative paths (k-shortest paths)
        try:
            k_paths = list(nx.shortest_simple_paths(G, source, target, weight='distance'))[:3]
            alternative_paths = []
            for alt_path in k_paths[1:]:  # Skip first (already found)
                alt_length = sum(G[alt_path[i]][alt_path[i+1]]['distance'] 
                               for i in range(len(alt_path)-1))
                alternative_paths.append({
                    'path': alt_path,
                    'length': alt_length
                })
        except:
            alternative_paths = []
        
        return {
            'path': path,
            'path_length': path_length,
            'path_cost': total_cost,
            'num_nodes': len(path),
            'alternative_paths': alternative_paths
        }
    except nx.NetworkXNoPath:
        raise ValueError(f"No path exists between {source} and {target}")


def solve_max_flow(
    G: nx.DiGraph,
    source: str,
    target: str
) -> Dict[str, Any]:
    """
    Solve maximum flow problem using Ford-Fulkerson algorithm
    """
    try:
        # Maximum flow
        flow_value, flow_dict = nx.maximum_flow(G, source, target, capacity='capacity')
        
        # Get flow edges
        flow_edges = []
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow > 0:
                    capacity = G[u][v].get('capacity', float('inf'))
                    flow_edges.append({
                        'from': u,
                        'to': v,
                        'flow': flow,
                        'capacity': capacity,
                        'utilization': (flow / capacity * 100) if capacity != float('inf') else 0
                    })
        
        # Find minimum cut
        cut_value, partition = nx.minimum_cut(G, source, target, capacity='capacity')
        reachable, non_reachable = partition
        cut_edges = [(u, v) for u, v in G.edges() if u in reachable and v in non_reachable]
        
        return {
            'max_flow_value': flow_value,
            'flow_edges': flow_edges,
            'min_cut_value': cut_value,
            'cut_edges': [(u, v) for u, v in cut_edges],
            'bottleneck_edges': [e for e in flow_edges if e['utilization'] > 90]
        }
    except nx.NetworkXError as e:
        raise ValueError(f"Flow problem error: {str(e)}")


def solve_min_cost_flow(
    G: nx.DiGraph,
    source: str,
    target: str,
    flow_demand: float
) -> Dict[str, Any]:
    """
    Solve minimum cost flow problem
    """
    try:
        # Set demand attributes
        G_copy = G.copy()
        for node in G_copy.nodes():
            if node == source:
                G_copy.nodes[node]['demand'] = -flow_demand  # Supply
            elif node == target:
                G_copy.nodes[node]['demand'] = flow_demand   # Demand
            else:
                G_copy.nodes[node]['demand'] = 0
        
        # Solve min cost flow
        flow_dict = nx.min_cost_flow(G_copy, demand='demand', capacity='capacity', weight='cost')
        
        # Calculate total cost
        total_cost = 0
        flow_edges = []
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow > 0:
                    cost = G_copy[u][v].get('cost', 0)
                    total_cost += flow * cost
                    flow_edges.append({
                        'from': u,
                        'to': v,
                        'flow': flow,
                        'cost_per_unit': cost,
                        'total_cost': flow * cost
                    })
        
        return {
            'total_cost': total_cost,
            'flow_value': flow_demand,
            'flow_edges': flow_edges,
            'avg_cost_per_unit': total_cost / flow_demand if flow_demand > 0 else 0
        }
    except nx.NetworkXUnfeasible:
        raise ValueError("No feasible flow exists for the given demand")
    except Exception as e:
        raise ValueError(f"Min cost flow error: {str(e)}")


def solve_tsp(
    G: nx.Graph,
    start_node: str
) -> Dict[str, Any]:
    """
    Solve Traveling Salesman Problem using approximation algorithm
    """
    try:
        # Use Christofides algorithm for TSP approximation
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=True, weight='distance')
        
        # Calculate tour length
        tour_length = sum(G[tsp_path[i]][tsp_path[i+1]]['distance'] 
                         for i in range(len(tsp_path)-1))
        
        # Reorder to start from specified node
        if start_node in tsp_path:
            start_idx = tsp_path.index(start_node)
            tsp_path = tsp_path[start_idx:] + tsp_path[:start_idx]
        
        return {
            'tour': tsp_path,
            'tour_length': tour_length,
            'num_nodes': len(set(tsp_path)) - 1,  # -1 because start=end
            'avg_distance': tour_length / (len(tsp_path) - 1)
        }
    except Exception as e:
        raise ValueError(f"TSP error: {str(e)}")


def create_network_visualization(
    nodes: List[Node],
    edges: List[Edge],
    solution: Dict[str, Any],
    problem_type: str
) -> str:
    """Create network visualization with solution overlay"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Build graph for layout
    G = build_graph(nodes, edges, directed=True)
    pos = {node.id: (node.x, node.y) for node in nodes}
    
    # Draw all edges (gray)
    edge_list = [(e.from_node, e.to_node) for e in edges]
    nx.draw_networkx_edges(G, pos, edge_list, edge_color='lightgray', 
                          width=1, alpha=0.5, arrows=True, 
                          arrowsize=10, ax=ax)
    
    # Highlight solution
    if problem_type == 'shortest_path':
        path = solution['path']
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, path_edges, edge_color='red', 
                              width=3, arrows=True, arrowsize=15, ax=ax)
        
        # Draw alternative paths
        for alt in solution.get('alternative_paths', [])[:2]:
            alt_path = alt['path']
            alt_edges = [(alt_path[i], alt_path[i+1]) for i in range(len(alt_path)-1)]
            nx.draw_networkx_edges(G, pos, alt_edges, edge_color='orange', 
                                  width=2, style='dashed', alpha=0.6, 
                                  arrows=True, arrowsize=12, ax=ax)
    
    elif problem_type == 'max_flow':
        # Draw edges with flow
        for flow_edge in solution['flow_edges']:
            u, v = flow_edge['from'], flow_edge['to']
            utilization = flow_edge['utilization']
            if utilization > 90:
                color = 'red'
                width = 4
            elif utilization > 70:
                color = 'orange'
                width = 3
            else:
                color = 'green'
                width = 2
            
            nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=color,
                                  width=width, arrows=True, arrowsize=12, ax=ax)
    
    elif problem_type == 'min_cost_flow':
        # Draw edges with flow
        max_flow = max(e['flow'] for e in solution['flow_edges']) if solution['flow_edges'] else 1
        for flow_edge in solution['flow_edges']:
            u, v = flow_edge['from'], flow_edge['to']
            width = 1 + (flow_edge['flow'] / max_flow) * 4
            nx.draw_networkx_edges(G, pos, [(u, v)], edge_color='blue',
                                  width=width, arrows=True, arrowsize=12, ax=ax)
    
    elif problem_type == 'tsp':
        tour = solution['tour']
        tour_edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
        nx.draw_networkx_edges(G, pos, tour_edges, edge_color='purple',
                              width=3, arrows=True, arrowsize=15, ax=ax)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in nodes:
        if node.node_type == 'source':
            node_colors.append('lightgreen')
            node_sizes.append(800)
        elif node.node_type == 'sink' or node.node_type == 'target':
            node_colors.append('lightcoral')
            node_sizes.append(800)
        elif node.node_type == 'facility':
            node_colors.append('gold')
            node_sizes.append(700)
        else:
            node_colors.append('lightblue')
            node_sizes.append(500)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          edgecolors='black', linewidths=2, ax=ax)
    
    # Draw labels
    labels = {node.id: node.name for node in nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title(f'Network Optimization: {problem_type.replace("_", " ").title()}',
                fontsize=14, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_statistics_chart(
    solution: Dict[str, Any],
    problem_type: str
) -> str:
    """Create statistics visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if problem_type == 'shortest_path':
        # Path comparison
        paths = ['Main Path'] + [f'Alt {i+1}' for i in range(len(solution.get('alternative_paths', [])))]
        lengths = [solution['path_length']] + [alt['length'] for alt in solution.get('alternative_paths', [])]
        
        bars = ax.bar(paths, lengths, color=['green'] + ['orange'] * (len(paths)-1),
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, length in zip(bars, lengths):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{length:.1f}',
                   ha='center', va='bottom', fontsize=11, weight='bold')
        
        ax.set_ylabel('Path Length', fontsize=12, weight='bold')
        ax.set_title('Path Comparison', fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    elif problem_type == 'max_flow':
        # Flow utilization
        flow_edges = solution['flow_edges']
        if flow_edges:
            utilizations = [e['utilization'] for e in flow_edges if e['utilization'] > 0]
            
            ax.hist(utilizations, bins=10, color='steelblue', alpha=0.7,
                   edgecolor='black', linewidth=1.5)
            ax.axvline(np.mean(utilizations), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(utilizations):.1f}%')
            
            ax.set_xlabel('Edge Utilization (%)', fontsize=12, weight='bold')
            ax.set_ylabel('Frequency', fontsize=12, weight='bold')
            ax.set_title('Edge Utilization Distribution', fontsize=13, weight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    elif problem_type == 'min_cost_flow':
        # Cost distribution
        flow_edges = solution['flow_edges']
        if flow_edges:
            edge_names = [f"{e['from']}-{e['to']}" for e in flow_edges[:10]]
            costs = [e['total_cost'] for e in flow_edges[:10]]
            
            bars = ax.barh(edge_names, costs, color='coral', alpha=0.7,
                          edgecolor='black', linewidth=1.5)
            
            for bar, cost in zip(bars, costs):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{cost:.1f}',
                       ha='left', va='center', fontsize=10, weight='bold')
            
            ax.set_xlabel('Total Cost', fontsize=12, weight='bold')
            ax.set_title('Edge Cost Distribution (Top 10)', fontsize=13, weight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def generate_interpretation(
    solution: Dict[str, Any],
    problem_type: str,
    network_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    if problem_type == 'shortest_path':
        path_length = solution['path_length']
        num_nodes = solution['num_nodes']
        
        key_insights.append({
            "title": "Optimal Route Found",
            "description": f"Shortest path with length {path_length:.2f} using {num_nodes} nodes.",
            "status": "positive"
        })
        
        if solution.get('alternative_paths'):
            alt_diff = solution['alternative_paths'][0]['length'] - path_length
            pct_diff = (alt_diff / path_length) * 100
            key_insights.append({
                "title": "Alternative Routes Available",
                "description": f"Next best route is {pct_diff:.1f}% longer. Consider for redundancy planning.",
                "status": "neutral"
            })
        
        recommendations.append(f"Total travel distance: {path_length:.2f} units")
        recommendations.append(f"Route passes through {num_nodes} intersections")
    
    elif problem_type == 'max_flow':
        max_flow = solution['max_flow_value']
        bottlenecks = solution['bottleneck_edges']
        
        key_insights.append({
            "title": "Maximum Flow Capacity",
            "description": f"Network can handle maximum flow of {max_flow:.2f} units between source and target.",
            "status": "positive"
        })
        
        if bottlenecks:
            key_insights.append({
                "title": "Bottleneck Edges Detected",
                "description": f"{len(bottlenecks)} edge(s) operating at >90% capacity. These limit overall flow.",
                "status": "warning"
            })
            recommendations.append("Consider increasing capacity of bottleneck edges")
        else:
            key_insights.append({
                "title": "Well-Balanced Network",
                "description": "No severe bottlenecks detected. Network capacity is well-distributed.",
                "status": "positive"
            })
        
        recommendations.append(f"Maximum throughput: {max_flow:.2f} units")
        recommendations.append(f"Minimum cut value: {solution['min_cut_value']:.2f}")
    
    elif problem_type == 'min_cost_flow':
        total_cost = solution['total_cost']
        avg_cost = solution['avg_cost_per_unit']
        
        key_insights.append({
            "title": "Cost-Optimal Flow",
            "description": f"Minimum cost solution: {total_cost:.2f} total cost for {solution['flow_value']:.2f} flow units.",
            "status": "positive"
        })
        
        key_insights.append({
            "title": "Unit Economics",
            "description": f"Average cost per flow unit: {avg_cost:.2f}. Monitor for cost reduction opportunities.",
            "status": "neutral"
        })
        
        recommendations.append(f"Total cost: {total_cost:.2f}")
        recommendations.append(f"Cost per unit: {avg_cost:.2f}")
        recommendations.append("Review high-cost edges for potential savings")
    
    elif problem_type == 'tsp':
        tour_length = solution['tour_length']
        num_nodes = solution['num_nodes']
        avg_dist = solution['avg_distance']
        
        key_insights.append({
            "title": "Optimal Tour Found",
            "description": f"TSP tour visiting {num_nodes} locations with total distance {tour_length:.2f}.",
            "status": "positive"
        })
        
        key_insights.append({
            "title": "Tour Efficiency",
            "description": f"Average distance between consecutive stops: {avg_dist:.2f}.",
            "status": "neutral"
        })
        
        recommendations.append(f"Total tour length: {tour_length:.2f}")
        recommendations.append(f"Locations visited: {num_nodes}")
        recommendations.append("Consider time windows for practical implementation")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/network-optimization")
async def solve_network_optimization(request: NetworkOptimizationRequest):
    """
    Solve network optimization problems on road networks
    
    Supports: shortest path, max flow, min cost flow, TSP
    """
    try:
        nodes = request.nodes
        edges = request.edges
        problem_type = request.problem_type
        
        # Build graph
        G = build_graph(nodes, edges, directed=(problem_type != 'tsp'))
        
        # Network statistics
        network_stats = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "avg_degree": sum(dict(G.degree()).values()) / len(nodes) if nodes else 0,
            "is_connected": nx.is_strongly_connected(G) if problem_type != 'tsp' else nx.is_connected(G),
            "diameter": nx.diameter(G.to_undirected()) if nx.is_connected(G.to_undirected()) else None
        }
        
        # Solve based on problem type
        if problem_type == 'shortest_path':
            if not request.source_node or not request.target_node:
                raise HTTPException(status_code=400, detail="source_node and target_node required for shortest path")
            solution = solve_shortest_path(G, request.source_node, request.target_node)
        
        elif problem_type == 'max_flow':
            if not request.source_node or not request.target_node:
                raise HTTPException(status_code=400, detail="source_node and target_node required for max flow")
            solution = solve_max_flow(G, request.source_node, request.target_node)
        
        elif problem_type == 'min_cost_flow':
            if not request.source_node or not request.target_node or request.flow_demand is None:
                raise HTTPException(status_code=400, detail="source_node, target_node, and flow_demand required for min cost flow")
            solution = solve_min_cost_flow(G, request.source_node, request.target_node, request.flow_demand)
        
        elif problem_type == 'tsp':
            if not request.source_node:
                raise HTTPException(status_code=400, detail="source_node required for TSP")
            G_undirected = G.to_undirected()
            solution = solve_tsp(G_undirected, request.source_node)
        
        # Generate visualizations
        plots = {
            "network_graph": create_network_visualization(nodes, edges, solution, problem_type),
            "statistics": create_statistics_chart(solution, problem_type)
        }
        
        # Generate interpretation
        interpretation = generate_interpretation(solution, problem_type, network_stats)
        
        return NetworkOptimizationResponse(
            success=True,
            problem_type=problem_type,
            solution=solution,
            network_stats=network_stats,
            plots=plots,
            interpretation=interpretation
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")
