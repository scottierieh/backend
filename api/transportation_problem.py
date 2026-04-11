"""
Transportation Problem Solver Router for FastAPI
Solve and visualize transportation/distribution optimization problems
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class TransportationRequest(BaseModel):
    """Transportation Problem solver request parameters"""
    costs: List[List[float]] = Field(
        default=[[4, 6, 9], [5, 2, 7]],
        description="Cost matrix (sources x destinations)"
    )
    supply: List[float] = Field(
        default=[120, 150],
        description="Supply at each source"
    )
    demand: List[float] = Field(
        default=[80, 90, 100],
        description="Demand at each destination"
    )


def _to_native_type(obj):
    """Convert numpy types to JSON-serializable Python types"""
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
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def solve_transportation(
    costs: List[List[float]],
    supply: List[float],
    demand: List[float]
) -> Dict[str, Any]:
    """
    Solve transportation problem using linear programming.
    Minimizes total transportation cost.
    """
    
    m = len(supply)  # number of sources
    n = len(demand)  # number of destinations
    
    total_supply = sum(supply)
    total_demand = sum(demand)
    
    # Check balance
    is_balanced = abs(total_supply - total_demand) < 1e-6
    
    # If unbalanced, add dummy source or destination
    if total_supply > total_demand:
        # Add dummy destination
        demand = demand + [total_supply - total_demand]
        costs = [row + [0] for row in costs]
        n += 1
        dummy_added = "destination"
    elif total_demand > total_supply:
        # Add dummy source
        supply = supply + [total_demand - total_supply]
        costs = costs + [[0] * n]
        m += 1
        dummy_added = "source"
    else:
        dummy_added = None
    
    # Flatten cost matrix for linprog
    c = []
    for i in range(m):
        for j in range(n):
            c.append(costs[i][j])
    
    # Equality constraints
    # Supply constraints: sum over j of x_ij = supply_i
    # Demand constraints: sum over i of x_ij = demand_j
    
    A_eq = []
    b_eq = []
    
    # Supply constraints
    for i in range(m):
        row = [0] * (m * n)
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(supply[i])
    
    # Demand constraints
    for j in range(n):
        row = [0] * (m * n)
        for i in range(m):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(demand[j])
    
    # Bounds (non-negative)
    bounds = [(0, None) for _ in range(m * n)]
    
    # Solve
    result = linprog(
        c,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method='highs'
    )
    
    if not result.success:
        return {
            "success": False,
            "total_cost": None,
            "shipments": [],
            "allocation_matrix": [],
            "message": result.message if hasattr(result, 'message') else "Optimization failed"
        }
    
    # Extract solution
    allocation = np.array(result.x).reshape(m, n)
    
    # Calculate total cost (excluding dummy)
    original_m = len(costs) - (1 if dummy_added == "source" else 0)
    original_n = len(costs[0]) - (1 if dummy_added == "destination" else 0)
    
    total_cost = 0
    shipments = []
    
    for i in range(original_m):
        for j in range(original_n):
            amount = allocation[i, j]
            if amount > 1e-6:
                cost = costs[i][j]
                total_cost += amount * cost
                shipments.append({
                    "source": f"Source {i+1}",
                    "source_idx": i,
                    "destination": f"Dest. {j+1}",
                    "dest_idx": j,
                    "amount": _to_native_type(amount),
                    "unit_cost": _to_native_type(cost),
                    "total_cost": _to_native_type(amount * cost)
                })
    
    # Handle unmet demand/excess supply
    unmet_demand = []
    excess_supply = []
    
    if dummy_added == "source":
        for j in range(original_n):
            if allocation[m-1, j] > 1e-6:
                unmet_demand.append({
                    "destination": f"Dest. {j+1}",
                    "amount": _to_native_type(allocation[m-1, j])
                })
    elif dummy_added == "destination":
        for i in range(original_m):
            if allocation[i, n-1] > 1e-6:
                excess_supply.append({
                    "source": f"Source {i+1}",
                    "amount": _to_native_type(allocation[i, n-1])
                })
    
    return {
        "success": True,
        "total_cost": _to_native_type(total_cost),
        "shipments": shipments,
        "allocation_matrix": [[_to_native_type(allocation[i, j]) for j in range(original_n)] for i in range(original_m)],
        "is_balanced": is_balanced,
        "total_supply": _to_native_type(sum(supply[:original_m])),
        "total_demand": _to_native_type(sum(demand[:original_n])),
        "unmet_demand": unmet_demand,
        "excess_supply": excess_supply,
        "message": "Optimal solution found"
    }


def generate_allocation_heatmap(
    allocation_matrix: List[List[float]],
    costs: List[List[float]]
) -> str:
    """Generate heatmap of allocation matrix"""
    
    m = len(allocation_matrix)
    n = len(allocation_matrix[0])
    
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(6, m * 1.2)))
    
    allocation = np.array(allocation_matrix)
    
    # Create heatmap
    im = ax.imshow(allocation, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Shipment Amount')
    
    # Add text annotations
    for i in range(m):
        for j in range(n):
            amount = allocation[i, j]
            cost = costs[i][j]
            text_color = 'white' if amount > allocation.max() * 0.5 else 'black'
            ax.text(j, i, f'{amount:.0f}\n(c={cost})', ha='center', va='center', 
                   color=text_color, fontsize=10)
    
    # Labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels([f'Dest. {j+1}' for j in range(n)])
    ax.set_yticklabels([f'Source {i+1}' for i in range(m)])
    ax.set_xlabel('Destinations')
    ax.set_ylabel('Sources')
    ax.set_title('Transportation Allocation Matrix')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_flow_diagram(
    shipments: List[Dict],
    supply: List[float],
    demand: List[float]
) -> str:
    """Generate flow diagram showing transportation routes"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    m = len(supply)
    n = len(demand)
    
    # Position sources on left, destinations on right
    source_y = np.linspace(0.9, 0.1, m)
    dest_y = np.linspace(0.9, 0.1, n)
    
    source_x = 0.15
    dest_x = 0.85
    
    # Draw sources
    for i in range(m):
        circle = plt.Circle((source_x, source_y[i]), 0.05, color='#4299e1', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(source_x - 0.1, source_y[i], f'S{i+1}\n({supply[i]:.0f})', 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw destinations
    for j in range(n):
        circle = plt.Circle((dest_x, dest_y[j]), 0.05, color='#48bb78', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(dest_x + 0.1, dest_y[j], f'D{j+1}\n({demand[j]:.0f})', 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw flows
    max_amount = max(s['amount'] for s in shipments) if shipments else 1
    
    for ship in shipments:
        i = ship['source_idx']
        j = ship['dest_idx']
        amount = ship['amount']
        
        # Line width proportional to amount
        lw = 1 + (amount / max_amount) * 5
        
        # Draw arrow
        ax.annotate('', xy=(dest_x - 0.05, dest_y[j]), xytext=(source_x + 0.05, source_y[i]),
                   arrowprops=dict(arrowstyle='->', color='#718096', lw=lw, alpha=0.7))
        
        # Label
        mid_x = (source_x + dest_x) / 2
        mid_y = (source_y[i] + dest_y[j]) / 2
        ax.text(mid_x, mid_y, f'{amount:.0f}', ha='center', va='center',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Transportation Flow Diagram', fontsize=14, pad=20)
    
    # Legend
    ax.plot([], [], 'o', color='#4299e1', markersize=10, label='Sources')
    ax.plot([], [], 'o', color='#48bb78', markersize=10, label='Destinations')
    ax.legend(loc='lower center', ncol=2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_cost_breakdown(shipments: List[Dict], total_cost: float) -> str:
    """Generate cost breakdown bar chart"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by cost contribution
    shipments_sorted = sorted(shipments, key=lambda x: x['total_cost'], reverse=True)
    
    labels = [f"{s['source']} → {s['destination']}" for s in shipments_sorted]
    costs = [s['total_cost'] for s in shipments_sorted]
    amounts = [s['amount'] for s in shipments_sorted]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(costs)))
    
    bars = ax.barh(labels, costs, color=colors, edgecolor='black')
    
    # Add labels
    for bar, amount, cost in zip(bars, amounts, costs):
        width = bar.get_width()
        ax.text(width + total_cost * 0.02, bar.get_y() + bar.get_height()/2,
               f'{cost:.0f} ({amount:.0f} units)', va='center', fontsize=9)
    
    ax.set_xlabel('Transportation Cost')
    ax.set_title(f'Cost Breakdown (Total: {total_cost:.2f})')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of transportation results"""
    key_insights = []
    
    if result['success']:
        key_insights.append({
            'title': 'Optimal Solution Found',
            'description': f"Minimum total transportation cost is {result['total_cost']:.2f}.",
            'status': 'positive'
        })
        
        # Balance check
        if result['is_balanced']:
            key_insights.append({
                'title': 'Balanced Problem',
                'description': f"Total supply ({result['total_supply']:.0f}) equals total demand ({result['total_demand']:.0f}).",
                'status': 'positive'
            })
        else:
            if result['unmet_demand']:
                unmet_total = sum(u['amount'] for u in result['unmet_demand'])
                key_insights.append({
                    'title': 'Unmet Demand',
                    'description': f"Demand exceeds supply by {unmet_total:.0f} units. Some destinations have unmet demand.",
                    'status': 'warning'
                })
            if result['excess_supply']:
                excess_total = sum(e['amount'] for e in result['excess_supply'])
                key_insights.append({
                    'title': 'Excess Supply',
                    'description': f"Supply exceeds demand by {excess_total:.0f} units. Some sources have unused inventory.",
                    'status': 'warning'
                })
        
        # Route analysis
        n_routes = len(result['shipments'])
        key_insights.append({
            'title': 'Active Routes',
            'description': f"{n_routes} transportation routes are used in the optimal solution.",
            'status': 'neutral'
        })
        
        # Highest cost route
        if result['shipments']:
            max_cost_route = max(result['shipments'], key=lambda x: x['total_cost'])
            key_insights.append({
                'title': 'Largest Cost Route',
                'description': f"{max_cost_route['source']} → {max_cost_route['destination']}: {max_cost_route['amount']:.0f} units at cost {max_cost_route['total_cost']:.2f}",
                'status': 'neutral'
            })
    else:
        key_insights.append({
            'title': 'No Solution Found',
            'description': result['message'],
            'status': 'warning'
        })
    
    # Recommendations
    recommendations = []
    if result['success']:
        if result['unmet_demand']:
            recommendations.append("Consider increasing supply capacity or finding additional sources.")
        if result['excess_supply']:
            recommendations.append("Excess inventory at some sources. Consider reducing production or finding new markets.")
        recommendations.append("Review high-cost routes for potential cost reduction opportunities.")
    else:
        recommendations.append("Check that supply and demand data are correctly entered.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


@router.post("/transportation-problem")
async def solve_transportation_problem(request: TransportationRequest) -> Dict[str, Any]:
    """
    Solve the Transportation Problem.
    
    The Transportation Problem finds the minimum cost way to ship goods
    from sources (factories, warehouses) to destinations (customers, stores).
    
    Minimize: sum(cost_ij * x_ij)
    Subject to: sum_j(x_ij) = supply_i (supply constraints)
               sum_i(x_ij) = demand_j (demand constraints)
               x_ij >= 0
    """
    try:
        # Validate inputs
        m = len(request.supply)
        n = len(request.demand)
        
        if len(request.costs) != m:
            raise HTTPException(status_code=400, detail="Cost matrix rows must match number of sources")
        
        for i, row in enumerate(request.costs):
            if len(row) != n:
                raise HTTPException(status_code=400, detail=f"Cost matrix row {i+1} must have {n} columns")
        
        if any(s < 0 for s in request.supply):
            raise HTTPException(status_code=400, detail="Supply values must be non-negative")
        
        if any(d < 0 for d in request.demand):
            raise HTTPException(status_code=400, detail="Demand values must be non-negative")
        
        # Solve
        result = solve_transportation(request.costs, request.supply, request.demand)
        
        # Generate plots
        plots = {}
        
        if result['success'] and result['allocation_matrix']:
            plots['heatmap'] = generate_allocation_heatmap(
                result['allocation_matrix'], request.costs
            )
            plots['flow'] = generate_flow_diagram(
                result['shipments'], request.supply, request.demand
            )
            if result['shipments']:
                plots['cost_breakdown'] = generate_cost_breakdown(
                    result['shipments'], result['total_cost']
                )
        
        # Generate interpretation
        params = {
            'costs': request.costs,
            'supply': request.supply,
            'demand': request.demand
        }
        interpretation = generate_interpretation(result, params)
        
        return {
            'success': result['success'],
            'message': result['message'],
            'total_cost': result['total_cost'],
            'shipments': result['shipments'],
            'allocation_matrix': result['allocation_matrix'],
            'is_balanced': result.get('is_balanced', True),
            'total_supply': result.get('total_supply'),
            'total_demand': result.get('total_demand'),
            'unmet_demand': result.get('unmet_demand', []),
            'excess_supply': result.get('excess_supply', []),
            'problem': {
                'n_sources': m,
                'n_destinations': n,
                'n_routes': len(result['shipments']) if result['shipments'] else 0
            },
            'plots': plots,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transportation problem solver failed: {str(e)}")
