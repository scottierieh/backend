"""
Dynamic Programming Router for FastAPI
Solve and visualize the 0/1 Knapsack Problem using Dynamic Programming
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class KnapsackItem(BaseModel):
    name: str = Field(description="Item name")
    weight: int = Field(ge=1, description="Item weight (positive integer)")
    value: float = Field(ge=0, description="Item value")


class DynamicProgrammingRequest(BaseModel):
    """Knapsack Problem request parameters"""
    items: List[KnapsackItem] = Field(
        default=[
            KnapsackItem(name="Item 1", weight=10, value=60),
            KnapsackItem(name="Item 2", weight=20, value=100),
            KnapsackItem(name="Item 3", weight=30, value=120)
        ],
        description="List of items with name, weight, and value"
    )
    capacity: int = Field(
        default=50,
        ge=1,
        description="Knapsack capacity"
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


def solve_knapsack(items: List[KnapsackItem], capacity: int) -> Dict[str, Any]:
    """
    Solve 0/1 Knapsack Problem using Dynamic Programming.
    Time complexity: O(n * W) where n = number of items, W = capacity
    """
    
    n = len(items)
    weights = [item.weight for item in items]
    values = [item.value for item in items]
    names = [item.name for item in items]
    
    # DP table: dp[i][w] = max value using items 0..i-1 with capacity w
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    # Backtrack to find selected items
    selected_indices = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_indices.append(i - 1)
            w -= weights[i - 1]
    
    selected_indices.reverse()
    selected_items = [names[i] for i in selected_indices]
    
    total_value = dp[n][capacity]
    total_weight = sum(weights[i] for i in selected_indices)
    
    # Calculate efficiency for each item
    item_details = []
    for i, item in enumerate(items):
        efficiency = item.value / item.weight if item.weight > 0 else 0
        item_details.append({
            "name": item.name,
            "weight": item.weight,
            "value": _to_native_type(item.value),
            "efficiency": _to_native_type(efficiency),
            "selected": i in selected_indices
        })
    
    # Sort by efficiency for analysis
    item_details_sorted = sorted(item_details, key=lambda x: x['efficiency'], reverse=True)
    
    return {
        "success": True,
        "total_value": _to_native_type(total_value),
        "total_weight": total_weight,
        "remaining_capacity": capacity - total_weight,
        "selected_items": selected_items,
        "selected_indices": selected_indices,
        "item_details": item_details,
        "item_details_by_efficiency": item_details_sorted,
        "dp_table": [[_to_native_type(dp[i][j]) for j in range(min(capacity + 1, 51))] for i in range(n + 1)],  # Limit for display
        "capacity": capacity,
        "n_items": n,
        "utilization": _to_native_type(total_weight / capacity * 100) if capacity > 0 else 0
    }


def generate_items_comparison_plot(item_details: List[Dict]) -> str:
    """Generate bar chart comparing items by value and weight"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [item['name'] for item in item_details]
    values = [item['value'] for item in item_details]
    weights = [item['weight'] for item in item_details]
    selected = [item['selected'] for item in item_details]
    
    colors = ['#4CAF50' if s else '#BBDEFB' for s in selected]
    
    # Value chart
    ax1 = axes[0]
    bars1 = ax1.bar(names, values, color=colors, edgecolor='black')
    ax1.set_xlabel('Items')
    ax1.set_ylabel('Value')
    ax1.set_title('Item Values')
    ax1.tick_params(axis='x', rotation=45)
    
    # Weight chart
    ax2 = axes[1]
    bars2 = ax2.bar(names, weights, color=colors, edgecolor='black')
    ax2.set_xlabel('Items')
    ax2.set_ylabel('Weight')
    ax2.set_title('Item Weights')
    ax2.tick_params(axis='x', rotation=45)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Selected'),
        Patch(facecolor='#BBDEFB', label='Not Selected')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_efficiency_plot(item_details: List[Dict]) -> str:
    """Generate efficiency (value/weight ratio) plot"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by efficiency
    sorted_items = sorted(item_details, key=lambda x: x['efficiency'], reverse=True)
    
    names = [item['name'] for item in sorted_items]
    efficiencies = [item['efficiency'] for item in sorted_items]
    selected = [item['selected'] for item in sorted_items]
    
    colors = ['#4CAF50' if s else '#BBDEFB' for s in selected]
    
    bars = ax.barh(names, efficiencies, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
               f'{eff:.2f}', va='center', fontsize=9)
    
    ax.set_xlabel('Efficiency (Value / Weight)')
    ax.set_title('Item Efficiency Ranking')
    ax.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Selected'),
        Patch(facecolor='#BBDEFB', label='Not Selected')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_capacity_utilization_plot(total_weight: int, capacity: int, selected_items: List[Dict]) -> str:
    """Generate capacity utilization pie/donut chart"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Capacity utilization donut
    ax1 = axes[0]
    used = total_weight
    remaining = capacity - total_weight
    
    sizes = [used, remaining]
    labels = [f'Used\n{used}', f'Remaining\n{remaining}']
    colors = ['#4CAF50', '#E0E0E0']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, pctdistance=0.75)
    
    # Draw center circle for donut
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax1.add_patch(centre_circle)
    ax1.set_title(f'Capacity Utilization\n(Total: {capacity})')
    
    # Selected items breakdown
    ax2 = axes[1]
    if selected_items:
        item_names = [item['name'] for item in selected_items if item['selected']]
        item_weights = [item['weight'] for item in selected_items if item['selected']]
        
        if item_weights:
            colors2 = plt.cm.Set3(np.linspace(0, 1, len(item_weights)))
            wedges2, texts2, autotexts2 = ax2.pie(item_weights, labels=item_names, colors=colors2,
                                                   autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(item_weights))})',
                                                   startangle=90)
            ax2.set_title('Weight Distribution of Selected Items')
        else:
            ax2.text(0.5, 0.5, 'No items selected', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Weight Distribution')
    else:
        ax2.text(0.5, 0.5, 'No items selected', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Weight Distribution')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_dp_table_heatmap(dp_table: List[List[float]], items: List[KnapsackItem], capacity: int) -> Optional[str]:
    """Generate DP table heatmap (limited size for visualization)"""
    
    # Limit size for visualization
    max_capacity_display = min(capacity + 1, 30)
    max_items_display = min(len(items) + 1, 15)
    
    if capacity > 50 or len(items) > 20:
        return None  # Too large to visualize nicely
    
    fig, ax = plt.subplots(figsize=(max(10, max_capacity_display * 0.4), max(6, max_items_display * 0.5)))
    
    dp_array = np.array(dp_table[:max_items_display])[:, :max_capacity_display]
    
    im = ax.imshow(dp_array, cmap='YlGn', aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(max_capacity_display))
    ax.set_yticks(np.arange(max_items_display))
    ax.set_xticklabels([str(i) for i in range(max_capacity_display)])
    ax.set_yticklabels(['∅'] + [items[i].name if i < len(items) else '' for i in range(max_items_display - 1)])
    
    ax.set_xlabel('Capacity')
    ax.set_ylabel('Items (cumulative)')
    ax.set_title('Dynamic Programming Table')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Max Value')
    
    # Add text annotations for small tables
    if max_capacity_display <= 15 and max_items_display <= 10:
        for i in range(dp_array.shape[0]):
            for j in range(dp_array.shape[1]):
                text = ax.text(j, i, f'{dp_array[i, j]:.0f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if dp_array[i, j] > dp_array.max() * 0.5 else 'black')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict) -> Dict[str, Any]:
    """Generate interpretation of knapsack results"""
    key_insights = []
    
    key_insights.append({
        'title': 'Optimal Solution Found',
        'description': f"Maximum value of {result['total_value']:.2f} achieved using {len(result['selected_items'])} items.",
        'status': 'positive'
    })
    
    # Utilization
    util = result['utilization']
    if util >= 90:
        key_insights.append({
            'title': 'Excellent Utilization',
            'description': f"Capacity utilization is {util:.1f}%. The knapsack is nearly full.",
            'status': 'positive'
        })
    elif util >= 70:
        key_insights.append({
            'title': 'Good Utilization',
            'description': f"Capacity utilization is {util:.1f}%. {result['remaining_capacity']} units of capacity remain.",
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'Low Utilization',
            'description': f"Only {util:.1f}% of capacity used. Consider if item weights match capacity well.",
            'status': 'warning'
        })
    
    # Efficiency analysis
    if result['item_details_by_efficiency']:
        most_efficient = result['item_details_by_efficiency'][0]
        key_insights.append({
            'title': 'Most Efficient Item',
            'description': f"'{most_efficient['name']}' has the highest efficiency ({most_efficient['efficiency']:.2f} value/weight).",
            'status': 'neutral'
        })
        
        # Check if most efficient items were selected
        efficient_selected = [item for item in result['item_details_by_efficiency'][:3] if item['selected']]
        if len(efficient_selected) >= 2:
            key_insights.append({
                'title': 'Efficiency-Based Selection',
                'description': f"Top efficient items are well represented in the selection.",
                'status': 'positive'
            })
    
    # Recommendations
    recommendations = []
    recommendations.append("Dynamic Programming guarantees the optimal solution for 0/1 Knapsack.")
    
    if result['remaining_capacity'] > 0:
        min_weight = min(item['weight'] for item in result['item_details'] if not item['selected']) if any(not item['selected'] for item in result['item_details']) else 0
        if min_weight > 0 and min_weight <= result['remaining_capacity']:
            recommendations.append(f"Remaining capacity ({result['remaining_capacity']}) could fit smaller items if available.")
    
    if result['utilization'] < 70:
        recommendations.append("Consider adding smaller items or adjusting capacity for better utilization.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


@router.post("/dynamic-programming")
async def solve_knapsack_problem(request: DynamicProgrammingRequest) -> Dict[str, Any]:
    """
    Solve the 0/1 Knapsack Problem using Dynamic Programming.
    
    Given a set of items with weights and values, and a knapsack capacity,
    find the combination of items that maximizes total value without
    exceeding the capacity. Each item can only be selected once.
    
    Time Complexity: O(n * W)
    Space Complexity: O(n * W)
    """
    try:
        # Validate inputs
        if not request.items:
            raise HTTPException(status_code=400, detail="At least one item is required")
        
        if request.capacity <= 0:
            raise HTTPException(status_code=400, detail="Capacity must be positive")
        
        # Check for reasonable problem size
        if len(request.items) * request.capacity > 1000000:
            raise HTTPException(status_code=400, detail="Problem size too large. Reduce items or capacity.")
        
        # Solve
        result = solve_knapsack(request.items, request.capacity)
        
        # Generate plots
        plots = {}
        
        plots['items_comparison'] = generate_items_comparison_plot(result['item_details'])
        plots['efficiency'] = generate_efficiency_plot(result['item_details'])
        plots['utilization'] = generate_capacity_utilization_plot(
            result['total_weight'], result['capacity'], result['item_details']
        )
        
        # DP table heatmap (only for small problems)
        dp_heatmap = generate_dp_table_heatmap(result['dp_table'], request.items, request.capacity)
        if dp_heatmap:
            plots['dp_table'] = dp_heatmap
        
        # Generate interpretation
        interpretation = generate_interpretation(result)
        
        return {
            'success': result['success'],
            'total_value': result['total_value'],
            'total_weight': result['total_weight'],
            'remaining_capacity': result['remaining_capacity'],
            'utilization': result['utilization'],
            'selected_items': result['selected_items'],
            'item_details': result['item_details'],
            'item_details_by_efficiency': result['item_details_by_efficiency'],
            'problem': {
                'n_items': result['n_items'],
                'capacity': result['capacity'],
                'n_selected': len(result['selected_items'])
            },
            'plots': plots,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knapsack solver failed: {str(e)}")
