"""
Inventory Analysis Router for FastAPI
Implements ABC/XYZ classification, EOQ, Safety Stock, and Turnover analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import io
import base64
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class InventoryRequest(BaseModel):
    data: List[Dict[str, Any]]
    item_id_col: str
    item_name_col: Optional[str] = None
    demand_col: str
    price_col: str
    demand_std_col: Optional[str] = None
    lead_time_col: Optional[str] = None
    order_cost_col: Optional[str] = None
    holding_cost_col: Optional[str] = None
    current_stock_col: Optional[str] = None
    service_level: float = 0.95


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
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


def calculate_abc_classification(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(value_col, ascending=False)
    df['cumulative_value'] = df[value_col].cumsum()
    total_value = df[value_col].sum()
    df['cumulative_pct'] = df['cumulative_value'] / total_value if total_value > 0 else 0
    
    def assign_abc(pct):
        if pct <= 0.80:
            return 'A'
        elif pct <= 0.95:
            return 'B'
        else:
            return 'C'
    
    df['abc_class'] = df['cumulative_pct'].apply(assign_abc)
    return df


def calculate_xyz_classification(df: pd.DataFrame, demand_col: str, demand_std_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    if demand_std_col and demand_std_col in df.columns:
        df['demand_std'] = pd.to_numeric(df[demand_std_col], errors='coerce').fillna(0)
    else:
        df['demand_std'] = df[demand_col] * 0.3
    
    monthly_demand = df[demand_col] / 12
    df['cv'] = np.where(monthly_demand > 0, df['demand_std'] / monthly_demand, 0)
    df['cv'] = df['cv'].replace([np.inf, -np.inf], 0).fillna(0)
    
    def assign_xyz(cv):
        if cv < 0.5:
            return 'X'
        elif cv < 1.0:
            return 'Y'
        else:
            return 'Z'
    
    df['xyz_class'] = df['cv'].apply(assign_xyz)
    return df


def calculate_eoq(annual_demand: float, order_cost: float, holding_cost_pct: float, unit_price: float) -> Dict:
    H = holding_cost_pct * unit_price
    if H <= 0 or annual_demand <= 0:
        return {'eoq': 0, 'orders_per_year': 0, 'total_cost': 0}
    
    eoq = np.sqrt((2 * annual_demand * order_cost) / H)
    orders_per_year = annual_demand / eoq if eoq > 0 else 0
    ordering_cost = orders_per_year * order_cost
    holding_cost_total = (eoq / 2) * H
    total_cost = ordering_cost + holding_cost_total
    
    return {'eoq': float(eoq), 'orders_per_year': float(orders_per_year), 'total_cost': float(total_cost)}


def calculate_safety_stock(demand_std: float, lead_time: float, service_level: float) -> Dict:
    z = stats.norm.ppf(service_level)
    safety_stock = z * demand_std * np.sqrt(lead_time)
    return {'safety_stock': float(max(0, safety_stock)), 'z_score': float(z)}


def calculate_reorder_point(avg_daily_demand: float, lead_time: float, safety_stock: float) -> float:
    return (avg_daily_demand * lead_time) + safety_stock


def calculate_turnover(cogs: float, avg_inventory: float) -> Dict:
    if avg_inventory <= 0:
        return {'turnover_ratio': 0, 'days_on_hand': 365, 'turnover_class': 'Slow'}
    turnover = cogs / avg_inventory
    days_on_hand = 365 / turnover if turnover > 0 else 365
    turnover_class = 'Fast' if turnover >= 12 else 'Normal' if turnover >= 6 else 'Slow'
    return {'turnover_ratio': float(turnover), 'days_on_hand': float(days_on_hand), 'turnover_class': turnover_class}


def determine_stock_status(current_stock: float, reorder_point: float, safety_stock: float, avg_demand: float) -> str:
    if current_stock <= safety_stock * 0.5:
        return 'Stockout Risk'
    elif current_stock < reorder_point:
        return 'Low'
    elif current_stock > avg_demand * 60:
        return 'Overstock'
    else:
        return 'Optimal'


def create_abc_pareto_chart(abc_data: List[Dict]) -> str:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    items = list(range(len(abc_data)))
    values = [d['annual_value'] for d in abc_data]
    cum_pcts = [d['cumulative_pct'] * 100 for d in abc_data]
    classes = [d['abc_class'] for d in abc_data]
    colors = ['#22c55e' if c == 'A' else '#f59e0b' if c == 'B' else '#ef4444' for c in classes]
    
    ax1.bar(items, values, color=colors, alpha=0.7, edgecolor='white')
    ax1.set_xlabel('Items (sorted by value)', fontsize=11)
    ax1.set_ylabel('Annual Value ($)', fontsize=11, color='#3b82f6')
    
    ax2 = ax1.twinx()
    ax2.plot(items, cum_pcts, color='#ef4444', linewidth=2)
    ax2.axhline(y=80, color='#22c55e', linestyle='--', alpha=0.7, label='80% (A)')
    ax2.axhline(y=95, color='#f59e0b', linestyle='--', alpha=0.7, label='95% (B)')
    ax2.set_ylabel('Cumulative %', fontsize=11, color='#ef4444')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='right')
    
    plt.title('ABC Pareto Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_abc_pie_chart(abc_summary: List[Dict]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    classes = [d['class'] for d in abc_summary]
    item_counts = [d['item_count'] for d in abc_summary]
    value_pcts = [d['value_pct'] * 100 for d in abc_summary]
    colors = ['#22c55e', '#f59e0b', '#ef4444']
    
    ax1.pie(item_counts, labels=classes, autopct='%1.0f%%', colors=colors, startangle=90)
    ax1.set_title('Items by Class', fontsize=12, fontweight='bold')
    ax2.pie(value_pcts, labels=classes, autopct='%1.0f%%', colors=colors, startangle=90)
    ax2.set_title('Value by Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_turnover_distribution_chart(turnover_data: List[Dict]) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    turnovers = [d['turnover_ratio'] for d in turnover_data]
    bins = [0, 3, 6, 9, 12, 15, max(20, max(turnovers) + 1)]
    colors = ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#22c55e', '#22c55e']
    
    n, bins_out, patches = ax.hist(turnovers, bins=bins, edgecolor='white', rwidth=0.8)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    ax.axvline(x=6, color='#f59e0b', linestyle='--', alpha=0.7)
    ax.axvline(x=12, color='#22c55e', linestyle='--', alpha=0.7)
    ax.set_xlabel('Turnover Ratio', fontsize=11)
    ax.set_ylabel('Number of Items', fontsize=11)
    ax.set_title('Inventory Turnover Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_stock_status_chart(safety_stock_data: List[Dict]) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    statuses = {}
    for d in safety_stock_data:
        status = d['stock_status']
        statuses[status] = statuses.get(status, 0) + 1
    
    labels = list(statuses.keys())
    sizes = list(statuses.values())
    color_map = {'Optimal': '#22c55e', 'Low': '#f59e0b', 'Stockout Risk': '#ef4444', 'Overstock': '#3b82f6'}
    colors = [color_map.get(l, '#64748b') for l in labels]
    
    ax.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors, startangle=90)
    ax.set_title('Stock Status Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_recommendations(summary: Dict, abc_data: List, safety_stock_data: List, turnover_data: List) -> List[Dict]:
    recommendations = []
    
    if summary['stockout_risk_items'] > 0:
        recommendations.append({
            'priority': 'High', 'category': 'Stock Levels',
            'recommendation': f"Reorder {summary['stockout_risk_items']} items immediately - below reorder point",
            'impact': 'Prevent stockouts and lost sales',
        })
    
    if summary['overstock_items'] > 0:
        recommendations.append({
            'priority': 'Medium', 'category': 'Excess Inventory',
            'recommendation': f"Review {summary['overstock_items']} overstocked items for markdown or return",
            'impact': 'Free up capital and storage space',
        })
    
    slow_movers = [d for d in turnover_data if d.get('turnover_class') == 'Slow']
    if len(slow_movers) > 5:
        recommendations.append({
            'priority': 'Medium', 'category': 'Slow Movers',
            'recommendation': f"Analyze {len(slow_movers)} slow-moving items (turnover < 6x)",
            'impact': 'Improve inventory efficiency',
        })
    
    if summary['potential_savings'] > 1000:
        recommendations.append({
            'priority': 'Medium', 'category': 'Order Optimization',
            'recommendation': f"Implement EOQ ordering to save ${summary['potential_savings']:,.0f} annually",
            'impact': 'Reduce total inventory costs',
        })
    
    class_a_count = len([d for d in abc_data if d['abc_class'] == 'A'])
    recommendations.append({
        'priority': 'Low', 'category': 'Focus Management',
        'recommendation': f"Prioritize tight control of {class_a_count} Class A items (80% of value)",
        'impact': 'Maximize return on inventory investment',
    })
    
    return recommendations


def generate_key_insights(summary: Dict, abc_summary: List, turnover_data: List) -> List[Dict]:
    insights = []
    
    class_a = next((a for a in abc_summary if a['class'] == 'A'), None)
    if class_a:
        insights.append({
            'title': f"Class A: {class_a['item_count']} items = {class_a['value_pct']*100:.0f}% value",
            'description': "Focus inventory management efforts on these high-value items.",
            'status': 'positive'
        })
    
    avg_turnover = summary['avg_turnover_ratio']
    if avg_turnover >= 10:
        insights.append({'title': f"Strong Turnover ({avg_turnover:.1f}x)", 'description': "Inventory is moving efficiently.", 'status': 'positive'})
    elif avg_turnover >= 6:
        insights.append({'title': f"Normal Turnover ({avg_turnover:.1f}x)", 'description': "Turnover is acceptable but can be improved.", 'status': 'neutral'})
    else:
        insights.append({'title': f"Low Turnover ({avg_turnover:.1f}x)", 'description': "Inventory is moving slowly. Review slow movers.", 'status': 'warning'})
    
    if summary['stockout_risk_items'] > 0:
        insights.append({'title': f"{summary['stockout_risk_items']} Items at Stockout Risk", 'description': "Immediate reorder needed to prevent lost sales.", 'status': 'warning'})
    
    if summary['potential_savings'] > 0:
        insights.append({'title': f"EOQ Savings: ${summary['potential_savings']:,.0f}/year", 'description': "Optimizing order quantities can reduce total costs.", 'status': 'positive'})
    
    return insights


@router.post("/inventory")
async def run_inventory_analysis(request: InventoryRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        for col in [request.item_id_col, request.demand_col, request.price_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        df['_demand'] = pd.to_numeric(df[request.demand_col], errors='coerce').fillna(0)
        df['_price'] = pd.to_numeric(df[request.price_col], errors='coerce').fillna(0)
        df['_annual_value'] = df['_demand'] * df['_price']
        
        df['_name'] = df[request.item_name_col].astype(str) if request.item_name_col and request.item_name_col in df.columns else df[request.item_id_col].astype(str)
        
        df['_demand_std'] = pd.to_numeric(df[request.demand_std_col], errors='coerce').fillna(df['_demand'] * 0.3 / 12) if request.demand_std_col and request.demand_std_col in df.columns else df['_demand'] * 0.3 / 12
        df['_lead_time'] = pd.to_numeric(df[request.lead_time_col], errors='coerce').fillna(7) if request.lead_time_col and request.lead_time_col in df.columns else 7
        df['_order_cost'] = pd.to_numeric(df[request.order_cost_col], errors='coerce').fillna(50) if request.order_cost_col and request.order_cost_col in df.columns else 50
        df['_holding_cost'] = pd.to_numeric(df[request.holding_cost_col], errors='coerce').fillna(0.2) if request.holding_cost_col and request.holding_cost_col in df.columns else 0.2
        df['_current_stock'] = pd.to_numeric(df[request.current_stock_col], errors='coerce').fillna(df['_demand'] / 12) if request.current_stock_col and request.current_stock_col in df.columns else df['_demand'] / 12
        
        df = calculate_abc_classification(df, '_annual_value')
        df = calculate_xyz_classification(df, '_demand', request.demand_std_col)
        
        abc_analysis, eoq_analysis, safety_stock_results, turnover_analysis = [], [], [], []
        total_savings, stockout_risk_count, overstock_count = 0, 0, 0
        
        for _, row in df.iterrows():
            item_id = str(row[request.item_id_col])
            item_name = str(row['_name'])
            
            abc_analysis.append({
                'item_id': item_id, 'item_name': item_name, 'annual_value': float(row['_annual_value']),
                'cumulative_pct': float(row['cumulative_pct']), 'abc_class': row['abc_class'], 'xyz_class': row['xyz_class'],
            })
            
            eoq_result = calculate_eoq(row['_demand'], row['_order_cost'], row['_holding_cost'], row['_price'])
            current_order_qty = row['_demand'] / 6
            
            if eoq_result['eoq'] > 0 and current_order_qty > 0:
                current_orders = row['_demand'] / current_order_qty
                current_cost = (current_orders * row['_order_cost']) + (current_order_qty / 2 * row['_holding_cost'] * row['_price'])
                savings = max(0, current_cost - eoq_result['total_cost'])
                total_savings += savings
            else:
                savings = 0
            
            eoq_analysis.append({
                'item_id': item_id, 'item_name': item_name, 'annual_demand': float(row['_demand']),
                'order_cost': float(row['_order_cost']), 'holding_cost': float(row['_holding_cost']),
                'eoq': float(eoq_result['eoq']), 'orders_per_year': float(eoq_result['orders_per_year']),
                'total_cost': float(eoq_result['total_cost']), 'current_order_qty': float(current_order_qty), 'savings': float(savings),
            })
            
            ss_result = calculate_safety_stock(row['_demand_std'], row['_lead_time'], request.service_level)
            avg_daily_demand = row['_demand'] / 365
            reorder_point = calculate_reorder_point(avg_daily_demand, row['_lead_time'], ss_result['safety_stock'])
            stock_status = determine_stock_status(row['_current_stock'], reorder_point, ss_result['safety_stock'], avg_daily_demand)
            
            if stock_status == 'Stockout Risk':
                stockout_risk_count += 1
            elif stock_status == 'Overstock':
                overstock_count += 1
            
            safety_stock_results.append({
                'item_id': item_id, 'item_name': item_name, 'avg_demand': float(avg_daily_demand),
                'demand_std': float(row['_demand_std']), 'lead_time': float(row['_lead_time']),
                'service_level': float(request.service_level), 'safety_stock': float(ss_result['safety_stock']),
                'reorder_point': float(reorder_point), 'current_stock': float(row['_current_stock']), 'stock_status': stock_status,
            })
            
            cogs = row['_demand'] * row['_price'] * 0.6
            avg_inventory = row['_current_stock'] * row['_price']
            turnover_result = calculate_turnover(cogs, avg_inventory)
            
            turnover_analysis.append({
                'item_id': item_id, 'item_name': item_name, 'cogs': float(cogs), 'avg_inventory': float(avg_inventory),
                'turnover_ratio': float(turnover_result['turnover_ratio']), 'days_on_hand': float(turnover_result['days_on_hand']),
                'turnover_class': turnover_result['turnover_class'],
            })
        
        abc_summary = []
        total_items = len(df)
        total_value = df['_annual_value'].sum()
        
        for abc_class in ['A', 'B', 'C']:
            class_df = df[df['abc_class'] == abc_class]
            abc_summary.append({
                'class': abc_class, 'item_count': len(class_df),
                'value_pct': float(class_df['_annual_value'].sum() / total_value) if total_value > 0 else 0,
                'item_pct': float(len(class_df) / total_items) if total_items > 0 else 0,
            })
        
        avg_turnover = np.mean([t['turnover_ratio'] for t in turnover_analysis]) if turnover_analysis else 0
        
        summary_data = {
            'total_items': total_items, 'total_inventory_value': float(total_value),
            'avg_turnover_ratio': float(avg_turnover), 'stockout_risk_items': stockout_risk_count,
            'overstock_items': overstock_count, 'potential_savings': float(total_savings),
        }
        
        visualizations = {
            'abc_pareto': create_abc_pareto_chart(abc_analysis),
            'abc_pie': create_abc_pie_chart(abc_summary),
            'turnover_distribution': create_turnover_distribution_chart(turnover_analysis),
            'stock_status': create_stock_status_chart(safety_stock_results),
        }
        
        recommendations = generate_recommendations(summary_data, abc_analysis, safety_stock_results, turnover_analysis)
        key_insights = generate_key_insights(summary_data, abc_summary, turnover_analysis)
        highest_turnover = max(turnover_analysis, key=lambda x: x['turnover_ratio'])
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'results': {
                'summary': {k: _to_native_type(v) for k, v in summary_data.items()},
                'abc_analysis': abc_analysis, 'eoq_analysis': eoq_analysis,
                'safety_stock': safety_stock_results, 'turnover_analysis': turnover_analysis,
                'abc_summary': abc_summary, 'recommendations': recommendations,
            },
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'top_abc_a_items': len([a for a in abc_analysis if a['abc_class'] == 'A']),
                'highest_turnover_item': highest_turnover['item_id'],
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory analysis failed: {str(e)}")
