"""
Inventory Optimization Router for FastAPI
EOQ, Safety Stock, Reorder Point, ABC Analysis, Inventory Turnover
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
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class InventoryRequest(BaseModel):
    # Single item mode
    annual_demand: Optional[float] = None
    ordering_cost: Optional[float] = None
    holding_cost_rate: Optional[float] = None  # As percentage of unit cost
    unit_cost: Optional[float] = None
    lead_time_days: Optional[float] = None
    demand_std_dev: Optional[float] = None  # Standard deviation of demand
    service_level: float = 0.95  # Target service level (e.g., 95%)
    
    # Multi-item mode (from data)
    data: Optional[List[Dict[str, Any]]] = None
    sku_col: Optional[str] = None
    demand_col: Optional[str] = None
    unit_cost_col: Optional[str] = None
    quantity_col: Optional[str] = None  # Current inventory
    sales_col: Optional[str] = None  # For turnover calculation
    
    # Common parameters
    working_days: int = 365
    ordering_cost_default: float = 50  # Default ordering cost if not provided
    holding_rate_default: float = 0.25  # 25% of unit cost per year


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
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


def calculate_eoq(annual_demand: float, ordering_cost: float, 
                  holding_cost: float) -> Dict[str, Any]:
    """Calculate Economic Order Quantity"""
    if holding_cost <= 0 or annual_demand <= 0:
        return {'error': 'Invalid parameters'}
    
    # EOQ formula: sqrt(2 * D * S / H)
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    
    # Number of orders per year
    orders_per_year = annual_demand / eoq
    
    # Time between orders (days)
    order_cycle_days = 365 / orders_per_year
    
    # Total annual costs
    annual_ordering_cost = (annual_demand / eoq) * ordering_cost
    annual_holding_cost = (eoq / 2) * holding_cost
    total_inventory_cost = annual_ordering_cost + annual_holding_cost
    
    return {
        'eoq': _to_native_type(eoq),
        'orders_per_year': _to_native_type(orders_per_year),
        'order_cycle_days': _to_native_type(order_cycle_days),
        'annual_ordering_cost': _to_native_type(annual_ordering_cost),
        'annual_holding_cost': _to_native_type(annual_holding_cost),
        'total_inventory_cost': _to_native_type(total_inventory_cost),
        'average_inventory': _to_native_type(eoq / 2)
    }


def calculate_safety_stock(demand_std_dev: float, lead_time_days: float,
                           service_level: float, daily_demand: float) -> Dict[str, Any]:
    """Calculate safety stock for target service level"""
    # Z-score for service level
    z_score = stats.norm.ppf(service_level)
    
    # Safety stock formula: Z * σ * sqrt(L)
    # Where σ is daily demand std dev and L is lead time in days
    safety_stock = z_score * demand_std_dev * np.sqrt(lead_time_days)
    
    # Alternative: if we have demand variability during lead time
    lead_time_demand_std = demand_std_dev * np.sqrt(lead_time_days)
    
    # Days of supply
    days_of_supply = safety_stock / daily_demand if daily_demand > 0 else 0
    
    return {
        'safety_stock': _to_native_type(safety_stock),
        'z_score': _to_native_type(z_score),
        'service_level': _to_native_type(service_level),
        'lead_time_demand_std': _to_native_type(lead_time_demand_std),
        'days_of_supply': _to_native_type(days_of_supply)
    }


def calculate_reorder_point(daily_demand: float, lead_time_days: float,
                            safety_stock: float) -> Dict[str, Any]:
    """Calculate reorder point"""
    # ROP = (Daily Demand × Lead Time) + Safety Stock
    lead_time_demand = daily_demand * lead_time_days
    reorder_point = lead_time_demand + safety_stock
    
    return {
        'reorder_point': _to_native_type(reorder_point),
        'lead_time_demand': _to_native_type(lead_time_demand),
        'safety_stock': _to_native_type(safety_stock),
        'lead_time_days': _to_native_type(lead_time_days)
    }


def perform_abc_analysis(df: pd.DataFrame, sku_col: str, 
                         value_col: str) -> Dict[str, Any]:
    """Perform ABC analysis based on value"""
    # Calculate total value per SKU
    sku_values = df.groupby(sku_col)[value_col].sum().reset_index()
    sku_values.columns = ['sku', 'total_value']
    
    # Sort by value descending
    sku_values = sku_values.sort_values('total_value', ascending=False)
    
    # Calculate cumulative percentage
    total = sku_values['total_value'].sum()
    sku_values['value_pct'] = sku_values['total_value'] / total * 100
    sku_values['cumulative_pct'] = sku_values['value_pct'].cumsum()
    
    # Assign ABC categories
    def assign_category(cum_pct):
        if cum_pct <= 80:
            return 'A'
        elif cum_pct <= 95:
            return 'B'
        else:
            return 'C'
    
    sku_values['category'] = sku_values['cumulative_pct'].apply(assign_category)
    
    # Summary by category
    category_summary = []
    for cat in ['A', 'B', 'C']:
        cat_df = sku_values[sku_values['category'] == cat]
        category_summary.append({
            'category': cat,
            'sku_count': len(cat_df),
            'sku_pct': _to_native_type(len(cat_df) / len(sku_values) * 100),
            'value_total': _to_native_type(cat_df['total_value'].sum()),
            'value_pct': _to_native_type(cat_df['total_value'].sum() / total * 100)
        })
    
    return {
        'items': sku_values.to_dict('records'),
        'category_summary': category_summary,
        'total_value': _to_native_type(total),
        'total_skus': len(sku_values)
    }


def calculate_inventory_turnover(df: pd.DataFrame, sku_col: str,
                                  sales_col: str, inventory_col: str) -> Dict[str, Any]:
    """Calculate inventory turnover metrics"""
    # Aggregate by SKU
    sku_metrics = df.groupby(sku_col).agg({
        sales_col: 'sum',
        inventory_col: 'mean'
    }).reset_index()
    sku_metrics.columns = ['sku', 'total_sales', 'avg_inventory']
    
    # Calculate turnover
    sku_metrics['turnover'] = sku_metrics['total_sales'] / sku_metrics['avg_inventory']
    sku_metrics['turnover'] = sku_metrics['turnover'].replace([np.inf, -np.inf], 0)
    
    # Days of inventory
    sku_metrics['days_of_inventory'] = 365 / sku_metrics['turnover']
    sku_metrics['days_of_inventory'] = sku_metrics['days_of_inventory'].replace([np.inf, -np.inf], 999)
    
    # Overall metrics
    total_sales = sku_metrics['total_sales'].sum()
    total_inventory = sku_metrics['avg_inventory'].sum()
    overall_turnover = total_sales / total_inventory if total_inventory > 0 else 0
    
    # Categorize turnover
    def turnover_category(t):
        if t >= 12:
            return 'Excellent'
        elif t >= 6:
            return 'Good'
        elif t >= 3:
            return 'Average'
        else:
            return 'Low'
    
    sku_metrics['turnover_category'] = sku_metrics['turnover'].apply(turnover_category)
    
    return {
        'sku_metrics': sku_metrics.to_dict('records'),
        'overall_turnover': _to_native_type(overall_turnover),
        'overall_days_of_inventory': _to_native_type(365 / overall_turnover) if overall_turnover > 0 else None,
        'total_sales': _to_native_type(total_sales),
        'total_inventory_value': _to_native_type(total_inventory)
    }


def calculate_total_cost_curve(annual_demand: float, ordering_cost: float,
                                holding_cost: float, eoq: float) -> Dict[str, Any]:
    """Generate total cost curve data"""
    # Range of order quantities
    q_range = np.linspace(max(1, eoq * 0.2), eoq * 3, 100)
    
    ordering_costs = (annual_demand / q_range) * ordering_cost
    holding_costs = (q_range / 2) * holding_cost
    total_costs = ordering_costs + holding_costs
    
    return {
        'quantities': [_to_native_type(q) for q in q_range],
        'ordering_costs': [_to_native_type(c) for c in ordering_costs],
        'holding_costs': [_to_native_type(c) for c in holding_costs],
        'total_costs': [_to_native_type(c) for c in total_costs],
        'eoq': _to_native_type(eoq),
        'min_cost': _to_native_type(total_costs.min())
    }


def simulate_inventory_levels(eoq: float, reorder_point: float, daily_demand: float,
                               lead_time_days: float, days: int = 180) -> Dict[str, Any]:
    """Simulate inventory levels over time"""
    inventory = []
    orders = []
    stockouts = []
    
    current_inventory = eoq
    order_in_transit = False
    order_arrival_day = -1
    
    for day in range(days):
        # Check for order arrival
        if order_in_transit and day >= order_arrival_day:
            current_inventory += eoq
            order_in_transit = False
        
        # Daily demand (with some randomness)
        demand = daily_demand * (0.8 + np.random.random() * 0.4)
        current_inventory -= demand
        
        # Check for stockout
        if current_inventory < 0:
            stockouts.append(day)
            current_inventory = 0
        
        # Check reorder point
        if current_inventory <= reorder_point and not order_in_transit:
            order_in_transit = True
            order_arrival_day = day + int(lead_time_days)
            orders.append(day)
        
        inventory.append(_to_native_type(current_inventory))
    
    return {
        'days': list(range(days)),
        'inventory_levels': inventory,
        'order_days': orders,
        'stockout_days': stockouts,
        'reorder_point': _to_native_type(reorder_point),
        'service_level_achieved': _to_native_type(1 - len(stockouts) / days)
    }


def create_eoq_chart(cost_curve: Dict, eoq_result: Dict) -> str:
    """Create EOQ cost curve chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    quantities = cost_curve['quantities']
    ordering = cost_curve['ordering_costs']
    holding = cost_curve['holding_costs']
    total = cost_curve['total_costs']
    eoq = cost_curve['eoq']
    
    # Cost curves
    ax1.plot(quantities, ordering, 'b-', linewidth=2, label='Ordering Cost')
    ax1.plot(quantities, holding, 'r-', linewidth=2, label='Holding Cost')
    ax1.plot(quantities, total, 'g-', linewidth=2.5, label='Total Cost')
    ax1.axvline(x=eoq, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.scatter([eoq], [cost_curve['min_cost']], color='green', s=150, zorder=5, 
                edgecolors='white', linewidth=2)
    ax1.annotate(f'EOQ = {eoq:,.0f}\nMin Cost = ${cost_curve["min_cost"]:,.0f}',
                xy=(eoq, cost_curve['min_cost']), xytext=(eoq * 1.3, cost_curve['min_cost'] * 1.2),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))
    
    ax1.set_xlabel('Order Quantity')
    ax1.set_ylabel('Annual Cost ($)')
    ax1.set_title('Economic Order Quantity Analysis', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Cost breakdown pie
    costs = [eoq_result['annual_ordering_cost'], eoq_result['annual_holding_cost']]
    labels = [f'Ordering\n${costs[0]:,.0f}', f'Holding\n${costs[1]:,.0f}']
    colors = ['#3b82f6', '#ef4444']
    
    wedges, texts, autotexts = ax2.pie(costs, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        explode=[0.05, 0.05])
    ax2.set_title(f'Cost Breakdown (Total: ${sum(costs):,.0f})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_inventory_simulation_chart(simulation: Dict, rop: Dict) -> str:
    """Create inventory simulation chart"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    days = simulation['days']
    levels = simulation['inventory_levels']
    reorder_point = simulation['reorder_point']
    
    # Inventory levels
    ax.plot(days, levels, 'b-', linewidth=1.5, label='Inventory Level')
    ax.axhline(y=reorder_point, color='orange', linestyle='--', linewidth=2, 
               label=f'Reorder Point ({reorder_point:,.0f})')
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1)
    
    # Mark order points
    for order_day in simulation['order_days']:
        ax.axvline(x=order_day, color='green', alpha=0.3, linewidth=1)
    
    # Mark stockouts
    for stockout_day in simulation['stockout_days']:
        ax.axvline(x=stockout_day, color='red', alpha=0.5, linewidth=2)
    
    ax.fill_between(days, 0, levels, alpha=0.3, color='blue')
    ax.fill_between(days, 0, [min(l, reorder_point) for l in levels], 
                   alpha=0.2, color='orange')
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Inventory Units')
    ax.set_title(f'Inventory Simulation (Service Level: {simulation["service_level_achieved"]*100:.1f}%)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_abc_chart(abc_data: Dict) -> str:
    """Create ABC analysis chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    items = abc_data['items']
    summary = abc_data['category_summary']
    
    # Pareto chart
    skus = [item['sku'] for item in items[:20]]  # Top 20
    values = [item['total_value'] for item in items[:20]]
    cum_pct = [item['cumulative_pct'] for item in items[:20]]
    colors = ['#22c55e' if item['category'] == 'A' else '#f59e0b' if item['category'] == 'B' else '#6b7280' 
              for item in items[:20]]
    
    ax1.bar(range(len(skus)), values, color=colors, edgecolor='white', linewidth=1)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(skus)), cum_pct, 'r-o', linewidth=2, markersize=4)
    ax1_twin.axhline(y=80, color='red', linestyle='--', alpha=0.5)
    ax1_twin.axhline(y=95, color='orange', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('SKU Rank')
    ax1.set_ylabel('Value ($)')
    ax1_twin.set_ylabel('Cumulative %')
    ax1.set_title('Pareto Analysis (Top 20 SKUs)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(skus)))
    ax1.set_xticklabels([str(i+1) for i in range(len(skus))], fontsize=8)
    ax1.spines['top'].set_visible(False)
    
    # Category summary
    categories = [s['category'] for s in summary]
    sku_pcts = [s['sku_pct'] for s in summary]
    value_pcts = [s['value_pct'] for s in summary]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, sku_pcts, width, label='% of SKUs', color='#3b82f6', edgecolor='white')
    bars2 = ax2.bar(x + width/2, value_pcts, width, label='% of Value', color='#22c55e', edgecolor='white')
    
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Percentage')
    ax2.set_title('ABC Category Distribution', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, pct in zip(bars1, sku_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.0f}%', 
                ha='center', fontsize=9)
    for bar, pct in zip(bars2, value_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.0f}%', 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_turnover_chart(turnover_data: Dict) -> str:
    """Create inventory turnover chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = turnover_data['sku_metrics']
    
    # Top/Bottom performers
    sorted_metrics = sorted(metrics, key=lambda x: x['turnover'], reverse=True)
    top_10 = sorted_metrics[:10]
    
    skus = [m['sku'][:15] for m in top_10]
    turnovers = [m['turnover'] for m in top_10]
    colors = ['#22c55e' if t >= 12 else '#3b82f6' if t >= 6 else '#f59e0b' if t >= 3 else '#ef4444' 
              for t in turnovers]
    
    bars = ax1.barh(skus, turnovers, color=colors, edgecolor='white', linewidth=2)
    ax1.axvline(x=12, color='green', linestyle='--', alpha=0.5, label='Excellent (12+)')
    ax1.axvline(x=6, color='blue', linestyle='--', alpha=0.5, label='Good (6+)')
    ax1.set_xlabel('Inventory Turnover')
    ax1.set_title('Top 10 SKUs by Turnover', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Turnover distribution
    all_turnovers = [m['turnover'] for m in metrics if m['turnover'] < 50]
    ax2.hist(all_turnovers, bins=20, color='#3b82f6', edgecolor='white', linewidth=1)
    ax2.axvline(x=turnover_data['overall_turnover'], color='red', linestyle='--', 
                linewidth=2, label=f"Overall: {turnover_data['overall_turnover']:.1f}")
    ax2.set_xlabel('Turnover Ratio')
    ax2.set_ylabel('Number of SKUs')
    ax2.set_title('Turnover Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(eoq: Optional[Dict], safety: Optional[Dict], 
                      abc: Optional[Dict], turnover: Optional[Dict]) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    if eoq:
        insights.append({
            'title': f'Optimal Order: {eoq["eoq"]:,.0f} units',
            'description': f'Order {eoq["orders_per_year"]:.1f}x/year. Total cost: ${eoq["total_inventory_cost"]:,.0f}/year.',
            'status': 'positive'
        })
    
    if safety:
        insights.append({
            'title': f'Safety Stock: {safety["safety_stock"]:,.0f} units',
            'description': f'{safety["days_of_supply"]:.1f} days of supply at {safety["service_level"]*100:.0f}% service level.',
            'status': 'neutral'
        })
    
    if abc:
        a_summary = next((s for s in abc['category_summary'] if s['category'] == 'A'), None)
        if a_summary:
            insights.append({
                'title': f'A Items: {a_summary["sku_pct"]:.0f}% SKUs = {a_summary["value_pct"]:.0f}% Value',
                'description': 'Focus inventory management efforts on A items.',
                'status': 'positive'
            })
    
    if turnover:
        overall = turnover['overall_turnover']
        if overall >= 12:
            insights.append({
                'title': f'Excellent Turnover: {overall:.1f}x',
                'description': 'Inventory is moving efficiently.',
                'status': 'positive'
            })
        elif overall >= 6:
            insights.append({
                'title': f'Good Turnover: {overall:.1f}x',
                'description': 'Healthy inventory movement.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': f'Low Turnover: {overall:.1f}x',
                'description': 'Consider reducing inventory levels or boosting sales.',
                'status': 'warning'
            })
    
    return insights


@router.post("/inventory")
async def run_inventory_optimization(request: InventoryRequest) -> Dict[str, Any]:
    """
    Perform Inventory Optimization analysis.
    """
    try:
        results = {}
        visualizations = {}
        
        # Determine mode
        if request.data and request.sku_col:
            # Multi-item mode
            df = pd.DataFrame(request.data)
            
            # ABC Analysis
            if request.demand_col and request.unit_cost_col:
                df['value'] = df[request.demand_col] * df[request.unit_cost_col]
                abc = perform_abc_analysis(df, request.sku_col, 'value')
                results['abc_analysis'] = abc
                visualizations['abc_chart'] = create_abc_chart(abc)
            
            # Turnover Analysis
            if request.sales_col and request.quantity_col:
                turnover = calculate_inventory_turnover(
                    df, request.sku_col, request.sales_col, request.quantity_col
                )
                results['turnover_analysis'] = turnover
                visualizations['turnover_chart'] = create_turnover_chart(turnover)
        
        # Single item EOQ analysis
        if request.annual_demand and request.unit_cost:
            ordering_cost = request.ordering_cost or request.ordering_cost_default
            holding_rate = request.holding_cost_rate or request.holding_rate_default
            holding_cost = request.unit_cost * holding_rate
            
            # EOQ
            eoq_result = calculate_eoq(request.annual_demand, ordering_cost, holding_cost)
            if 'error' not in eoq_result:
                results['eoq'] = eoq_result
                
                # Cost curve
                cost_curve = calculate_total_cost_curve(
                    request.annual_demand, ordering_cost, holding_cost, eoq_result['eoq']
                )
                results['cost_curve'] = cost_curve
                visualizations['eoq_chart'] = create_eoq_chart(cost_curve, eoq_result)
                
                # Daily demand
                daily_demand = request.annual_demand / request.working_days
                
                # Safety stock
                if request.demand_std_dev and request.lead_time_days:
                    safety = calculate_safety_stock(
                        request.demand_std_dev, request.lead_time_days,
                        request.service_level, daily_demand
                    )
                    results['safety_stock'] = safety
                    
                    # Reorder point
                    rop = calculate_reorder_point(
                        daily_demand, request.lead_time_days, safety['safety_stock']
                    )
                    results['reorder_point'] = rop
                    
                    # Simulation
                    simulation = simulate_inventory_levels(
                        eoq_result['eoq'], rop['reorder_point'],
                        daily_demand, request.lead_time_days
                    )
                    results['simulation'] = simulation
                    visualizations['simulation_chart'] = create_inventory_simulation_chart(simulation, rop)
        
        # Generate insights
        insights = generate_insights(
            results.get('eoq'),
            results.get('safety_stock'),
            results.get('abc_analysis'),
            results.get('turnover_analysis')
        )
        
        # Summary
        summary = {
            'eoq': results.get('eoq', {}).get('eoq'),
            'total_cost': results.get('eoq', {}).get('total_inventory_cost'),
            'safety_stock': results.get('safety_stock', {}).get('safety_stock'),
            'reorder_point': results.get('reorder_point', {}).get('reorder_point'),
            'overall_turnover': results.get('turnover_analysis', {}).get('overall_turnover'),
            'abc_a_pct': next((s['value_pct'] for s in results.get('abc_analysis', {}).get('category_summary', []) 
                             if s['category'] == 'A'), None)
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory optimization failed: {str(e)}")
