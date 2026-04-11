"""
Knapsack Problem Router for FastAPI
Using Google OR-Tools for 0/1, bounded, and unbounded knapsack
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

from ortools.algorithms.python import knapsack_solver

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class KnapsackRequest(BaseModel):
    data: List[Dict[str, Any]]
    item_id_col: str
    value_col: str
    weight_col: str
    quantity_col: Optional[str] = None
    problem_type: Literal["0_1", "bounded", "unbounded"] = "0_1"
    capacity: float = 100


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


ITEM_COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def solve_knapsack(items: List[Dict], capacity: int, problem_type: str) -> Dict:
    """Solve knapsack using OR-Tools"""
    if not items:
        return {'selected_indices': [], 'total_value': 0}
    
    # Scale to integers (OR-Tools requires integers)
    scale_factor = 100
    
    values = [int(item['value'] * scale_factor) for item in items]
    weights = [[int(item['weight'] * scale_factor) for item in items]]
    capacities = [int(capacity * scale_factor)]
    
    # Select solver based on problem type
    if problem_type == '0_1':
        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            'KnapsackSolver'
        )
    else:
        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
            'KnapsackSolver'
        )
    
    solver.init(values, weights, capacities)
    computed_value = solver.solve()
    
    selected_indices = []
    for i in range(len(items)):
        if solver.best_solution_contains(i):
            selected_indices.append(i)
    
    return {
        'selected_indices': selected_indices,
        'total_value': computed_value / scale_factor
    }


def create_selection_chart(selected: List[Dict], excluded: List[Dict]) -> str:
    """Create selection comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Combine and sort by efficiency
    all_items = [(item, True) for item in selected] + [(item, False) for item in excluded]
    all_items.sort(key=lambda x: -x[0]['efficiency'])
    all_items = all_items[:15]  # Top 15
    
    labels = [item[0]['item_id'] for item in all_items]
    values = [item[0]['value'] for item in all_items]
    colors = ['#22c55e' if item[1] else '#ef4444' for item in all_items]
    
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1)
    
    for bar, item in zip(bars, all_items):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${item[0]["value"]}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Value ($)', fontsize=11)
    ax.set_xlabel('Item', fontsize=11)
    ax.set_title('Item Selection (Green=Selected, Red=Excluded)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_efficiency_chart(selected: List[Dict], excluded: List[Dict]) -> str:
    """Create efficiency comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by efficiency
    all_items = sorted(selected + excluded, key=lambda x: -x['efficiency'])[:12]
    
    labels = [item['item_id'] for item in all_items]
    efficiencies = [item['efficiency'] for item in all_items]
    colors = [ITEM_COLORS[i % len(ITEM_COLORS)] for i in range(len(all_items))]
    
    bars = ax.barh(labels, efficiencies, color=colors, edgecolor='white', linewidth=1)
    
    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'${eff:.1f}/kg', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Efficiency ($/kg)', fontsize=11)
    ax.set_title('Item Efficiency Ranking', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_capacity_chart(total_weight: float, capacity: float, selected: List[Dict]) -> str:
    """Create capacity utilization chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    ax1 = axes[0]
    sizes = [total_weight, capacity - total_weight]
    labels = ['Used', 'Remaining']
    colors = ['#3b82f6', '#e5e7eb']
    explode = (0.05, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.set_title('Capacity Utilization', fontsize=12, fontweight='bold')
    
    # Weight breakdown by item
    ax2 = axes[1]
    
    items_sorted = sorted(selected, key=lambda x: -x['weight'])[:10]
    labels = [item['item_id'] for item in items_sorted]
    weights = [item['weight'] for item in items_sorted]
    colors = [ITEM_COLORS[i % len(ITEM_COLORS)] for i in range(len(items_sorted))]
    
    bars = ax2.barh(labels, weights, color=colors, edgecolor='white', linewidth=1)
    
    for bar, w in zip(bars, weights):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}kg', ha='left', va='center', fontsize=9)
    
    ax2.set_xlabel('Weight (kg)', fontsize=11)
    ax2.set_title('Weight Distribution', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_value_weight_scatter(selected: List[Dict], excluded: List[Dict]) -> str:
    """Create value vs weight scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Selected items
    if selected:
        sel_values = [item['value'] for item in selected]
        sel_weights = [item['weight'] for item in selected]
        ax.scatter(sel_weights, sel_values, c='#22c55e', s=100, alpha=0.7, 
                  label='Selected', edgecolors='white', linewidths=1)
    
    # Excluded items
    if excluded:
        exc_values = [item['value'] for item in excluded]
        exc_weights = [item['weight'] for item in excluded]
        ax.scatter(exc_weights, exc_values, c='#ef4444', s=100, alpha=0.7,
                  label='Excluded', edgecolors='white', linewidths=1)
    
    # Add efficiency lines
    max_weight = max([item['weight'] for item in selected + excluded]) if selected + excluded else 1
    max_value = max([item['value'] for item in selected + excluded]) if selected + excluded else 1
    
    for eff in [100, 500, 1000]:
        x = np.linspace(0, max_weight * 1.1, 100)
        y = eff * x
        ax.plot(x, y, '--', alpha=0.3, label=f'${eff}/kg efficiency')
    
    ax.set_xlabel('Weight (kg)', fontsize=11)
    ax.set_ylabel('Value ($)', fontsize=11)
    ax.set_title('Value vs Weight (with Efficiency Lines)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, max_weight * 1.2)
    ax.set_ylim(0, max_value * 1.2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(selected: List[Dict], excluded: List[Dict], 
                          total_value: float, total_weight: float, 
                          capacity: float) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    utilization = (total_weight / capacity * 100) if capacity > 0 else 0
    
    # Utilization insight
    if utilization >= 95:
        insights.append({
            'title': f'Excellent Capacity Utilization: {utilization:.1f}%',
            'description': 'Nearly optimal packing achieved with minimal wasted capacity.',
            'status': 'positive'
        })
    elif utilization >= 80:
        insights.append({
            'title': f'Good Capacity Utilization: {utilization:.1f}%',
            'description': 'Reasonable packing with some room for smaller items.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Capacity Utilization: {utilization:.1f}%',
            'description': 'Significant unused capacity. Consider adding smaller high-value items.',
            'status': 'warning'
        })
    
    # Selection ratio
    selection_ratio = len(selected) / (len(selected) + len(excluded)) * 100 if (selected or excluded) else 0
    if selection_ratio > 50:
        insights.append({
            'title': f'High Selection Rate: {selection_ratio:.0f}%',
            'description': f'{len(selected)} of {len(selected) + len(excluded)} items selected.',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': f'{len(selected)} Items Selected',
            'description': f'{len(excluded)} items excluded due to weight constraints.',
            'status': 'neutral'
        })
    
    # Efficiency analysis
    if selected:
        avg_eff = np.mean([item['efficiency'] for item in selected])
        insights.append({
            'title': f'Average Efficiency: ${avg_eff:.1f}/kg',
            'description': 'Value gained per unit weight for selected items.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/knapsack")
async def run_knapsack(request: KnapsackRequest) -> Dict[str, Any]:
    """Run knapsack optimization"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.item_id_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item ID column '{request.item_id_col}' not found")
        if request.value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Value column '{request.value_col}' not found")
        if request.weight_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Weight column '{request.weight_col}' not found")
        
        # Prepare items
        items = []
        for _, row in df.iterrows():
            weight = float(row[request.weight_col])
            value = float(row[request.value_col])
            
            if weight <= 0 or value <= 0:
                continue
            
            item = {
                'item_id': str(row[request.item_id_col]),
                'value': value,
                'weight': weight,
                'efficiency': value / weight if weight > 0 else 0
            }
            
            if request.quantity_col and request.quantity_col in df.columns:
                item['quantity'] = int(row[request.quantity_col])
            else:
                item['quantity'] = 1
            
            items.append(item)
        
        if not items:
            raise HTTPException(status_code=400, detail="No valid items found")
        
        # Solve knapsack
        solution = solve_knapsack(items, request.capacity, request.problem_type)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Separate selected and excluded
        selected_indices = set(solution['selected_indices'])
        selected_items = [items[i] for i in range(len(items)) if i in selected_indices]
        excluded_items = [items[i] for i in range(len(items)) if i not in selected_indices]
        
        total_value = sum(item['value'] for item in selected_items)
        total_weight = sum(item['weight'] for item in selected_items)
        utilization = (total_weight / request.capacity * 100) if request.capacity > 0 else 0
        
        # Calculate metrics
        avg_efficiency = np.mean([item['efficiency'] for item in selected_items]) if selected_items else 0
        value_density = total_value / total_weight if total_weight > 0 else 0
        
        # Theoretical max (fractional knapsack upper bound)
        sorted_by_eff = sorted(items, key=lambda x: -x['efficiency'])
        theoretical_max = 0
        remaining = request.capacity
        for item in sorted_by_eff:
            if remaining >= item['weight']:
                theoretical_max += item['value']
                remaining -= item['weight']
            else:
                theoretical_max += item['efficiency'] * remaining
                break
        
        metrics = {
            'avg_efficiency': avg_efficiency,
            'value_density': value_density,
            'weight_utilization': utilization,
            'theoretical_max': theoretical_max
        }
        
        # Create visualizations
        visualizations = {
            'selection_chart': create_selection_chart(selected_items, excluded_items),
            'efficiency_chart': create_efficiency_chart(selected_items, excluded_items),
            'capacity_chart': create_capacity_chart(total_weight, request.capacity, selected_items),
            'value_weight_scatter': create_value_weight_scatter(selected_items, excluded_items)
        }
        
        # Generate insights
        key_insights = generate_key_insights(selected_items, excluded_items,
                                             total_value, total_weight, request.capacity)
        
        # Prepare results
        results = {
            'selected_items': [{k: _to_native_type(v) for k, v in item.items()} for item in selected_items],
            'excluded_items': [{k: _to_native_type(v) for k, v in item.items()} for item in excluded_items],
            'total_value': total_value,
            'total_weight': total_weight,
            'capacity': request.capacity,
            'utilization': utilization,
            'num_selected': len(selected_items),
            'num_total': len(items),
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'problem_type': request.problem_type,
            'algorithm': 'OR-Tools Dynamic Programming',
            'capacity': request.capacity,
            'total_value': total_value,
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
        raise HTTPException(status_code=500, detail=f"Knapsack optimization failed: {str(e)}")
