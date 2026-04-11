"""
Break-even Analysis Router for FastAPI
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

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class BreakevenRequest(BaseModel):
    data: List[Dict[str, Any]]
    product_col: Optional[str] = None
    fixed_cost_col: str
    variable_cost_col: str
    selling_price_col: str
    sales_mix_col: Optional[str] = None
    current_units_col: Optional[str] = None
    target_profit: float = 0
    analysis_type: Literal["single", "multi", "target", "sensitivity"] = "single"


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


COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def calculate_breakeven(fixed_cost: float, variable_cost: float, 
                        selling_price: float, target_profit: float = 0) -> Dict:
    contribution_margin = selling_price - variable_cost
    
    if contribution_margin <= 0:
        return {
            'breakeven_units': float('inf'),
            'breakeven_revenue': float('inf'),
            'contribution_margin': contribution_margin,
            'contribution_margin_ratio': 0
        }
    
    cm_ratio = contribution_margin / selling_price
    breakeven_units = (fixed_cost + target_profit) / contribution_margin
    breakeven_revenue = (fixed_cost + target_profit) / cm_ratio
    
    return {
        'breakeven_units': int(np.ceil(breakeven_units)),
        'breakeven_revenue': breakeven_revenue,
        'contribution_margin': contribution_margin,
        'contribution_margin_ratio': cm_ratio
    }


def create_breakeven_chart(products: List[Dict], total_fixed_cost: float) -> str:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(products) == 1:
        p = products[0]
        max_units = int(p['breakeven_units'] * 1.5)
        units = np.linspace(0, max_units, 100)
        
        total_cost = p['fixed_cost'] + p['variable_cost'] * units
        revenue = p['selling_price'] * units
        profit = revenue - total_cost
        
        ax.plot(units, total_cost, 'r-', linewidth=2, label='Total Cost')
        ax.plot(units, revenue, 'b-', linewidth=2, label='Revenue')
        ax.fill_between(units, total_cost, revenue, where=(revenue >= total_cost),
                        alpha=0.3, color='green', label='Profit Zone')
        ax.fill_between(units, total_cost, revenue, where=(revenue < total_cost),
                        alpha=0.3, color='red', label='Loss Zone')
        
        ax.axvline(x=p['breakeven_units'], color='gray', linestyle='--', linewidth=1.5)
        ax.axhline(y=p['breakeven_revenue'], color='gray', linestyle='--', linewidth=1.5)
        
        ax.scatter([p['breakeven_units']], [p['breakeven_revenue']], 
                   color='black', s=100, zorder=5, label=f"Break-even: {p['breakeven_units']:,} units")
        
        ax.set_xlabel('Units Sold', fontsize=11)
        ax.set_ylabel('Amount ($)', fontsize=11)
    else:
        products_sorted = sorted(products, key=lambda x: -x['breakeven_units'])[:8]
        
        names = [p['product'] for p in products_sorted]
        be_units = [p['breakeven_units'] for p in products_sorted]
        colors = [COLORS[i % len(COLORS)] for i in range(len(products_sorted))]
        
        bars = ax.barh(names, be_units, color=colors, edgecolor='white', linewidth=1)
        
        for bar, units in zip(bars, be_units):
            ax.text(bar.get_width() + max(be_units) * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{units:,}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Break-even Units', fontsize=11)
        ax.invert_yaxis()
    
    ax.set_title('Break-even Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_contribution_chart(products: List[Dict]) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [p['product'] for p in products]
    cm_ratios = [p['contribution_margin_ratio'] * 100 for p in products]
    colors = [COLORS[i % len(COLORS)] for i in range(len(products))]
    
    bars = ax.bar(names, cm_ratios, color=colors, edgecolor='white', linewidth=1)
    
    for bar, ratio in zip(bars, cm_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='30% benchmark')
    
    ax.set_ylabel('Contribution Margin Ratio (%)', fontsize=11)
    ax.set_xlabel('Product', fontsize=11)
    ax.set_title('Contribution Margin by Product', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_sensitivity_chart(sensitivity: List[Dict], base_breakeven: int) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    variables = [s['variable'] for s in sensitivity]
    changes = [s['change_from_base'] for s in sensitivity]
    colors = ['#ef4444' if c > 0 else '#22c55e' for c in changes]
    
    bars = ax.barh(variables, changes, color=colors, edgecolor='white', linewidth=1)
    
    for bar, change, s in zip(bars, changes, sensitivity):
        label = f"+{change:,.0f}" if change > 0 else f"{change:,.0f}"
        x_pos = bar.get_width() + (max(abs(c) for c in changes) * 0.05) if change >= 0 else bar.get_width() - (max(abs(c) for c in changes) * 0.05)
        ha = 'left' if change >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label, ha=ha, va='center', fontsize=9)
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Change in Break-even Units', fontsize=11)
    ax.set_title(f'Sensitivity Analysis (Base: {base_breakeven:,} units)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_profit_volume_chart(products: List[Dict]) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, p in enumerate(products[:5]):
        max_units = int(p['breakeven_units'] * 2)
        units = np.linspace(0, max_units, 100)
        profit = p['contribution_margin'] * units - p['fixed_cost']
        
        ax.plot(units, profit, color=COLORS[i % len(COLORS)], linewidth=2, label=p['product'])
        ax.scatter([p['breakeven_units']], [0], color=COLORS[i % len(COLORS)], s=80, zorder=5)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Units Sold', fontsize=11)
    ax.set_ylabel('Profit ($)', fontsize=11)
    ax.set_title('Profit-Volume Relationship', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def calculate_sensitivity(base_be: int, fixed_cost: float, variable_cost: float, 
                          selling_price: float) -> List[Dict]:
    sensitivity = []
    
    scenarios = [
        ('Price +10%', 'selling_price', 1.10),
        ('Price -10%', 'selling_price', 0.90),
        ('Variable Cost +10%', 'variable_cost', 1.10),
        ('Variable Cost -10%', 'variable_cost', 0.90),
        ('Fixed Cost +10%', 'fixed_cost', 1.10),
        ('Fixed Cost -10%', 'fixed_cost', 0.90),
    ]
    
    for name, var, factor in scenarios:
        new_fc = fixed_cost * factor if var == 'fixed_cost' else fixed_cost
        new_vc = variable_cost * factor if var == 'variable_cost' else variable_cost
        new_sp = selling_price * factor if var == 'selling_price' else selling_price
        
        new_cm = new_sp - new_vc
        if new_cm > 0:
            new_be = int(np.ceil(new_fc / new_cm))
        else:
            new_be = base_be * 10
        
        sensitivity.append({
            'variable': name,
            'change_percent': int((factor - 1) * 100),
            'new_breakeven': new_be,
            'change_from_base': new_be - base_be
        })
    
    return sensitivity


def generate_key_insights(products: List[Dict], total_fixed: float, 
                          overall_be_units: int, overall_be_revenue: float) -> List[Dict]:
    insights = []
    
    avg_cm_ratio = np.mean([p['contribution_margin_ratio'] for p in products])
    
    if avg_cm_ratio > 0.4:
        insights.append({
            'title': f'Strong Contribution Margin: {avg_cm_ratio*100:.0f}%',
            'description': 'High CM ratio indicates good pricing power and cost control.',
            'status': 'positive'
        })
    elif avg_cm_ratio < 0.2:
        insights.append({
            'title': f'Low Contribution Margin: {avg_cm_ratio*100:.0f}%',
            'description': 'Consider raising prices or reducing variable costs.',
            'status': 'warning'
        })
    
    profitable = [p for p in products if p.get('current_profit') and p['current_profit'] > 0]
    if profitable:
        insights.append({
            'title': f'{len(profitable)} of {len(products)} Products Above Break-even',
            'description': 'These products are currently generating profit.',
            'status': 'positive'
        })
    
    high_safety = [p for p in products if p.get('margin_of_safety') and p['margin_of_safety'] > 0.25]
    if high_safety:
        insights.append({
            'title': 'Healthy Margin of Safety',
            'description': f"{len(high_safety)} product(s) have >25% margin of safety.",
            'status': 'positive'
        })
    
    low_cm = [p for p in products if p['contribution_margin_ratio'] < 0.2]
    if low_cm:
        insights.append({
            'title': f'{len(low_cm)} Low-Margin Products',
            'description': 'Consider repricing or discontinuing low-margin items.',
            'status': 'warning'
        })
    
    return insights


@router.post("/breakeven")
async def run_breakeven_analysis(request: BreakevenRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        if request.fixed_cost_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Fixed cost column '{request.fixed_cost_col}' not found")
        if request.variable_cost_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Variable cost column '{request.variable_cost_col}' not found")
        if request.selling_price_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Selling price column '{request.selling_price_col}' not found")
        
        products = []
        total_fixed_cost = 0
        
        for idx, row in df.iterrows():
            fixed_cost = float(row[request.fixed_cost_col])
            variable_cost = float(row[request.variable_cost_col])
            selling_price = float(row[request.selling_price_col])
            
            total_fixed_cost += fixed_cost
            
            be = calculate_breakeven(fixed_cost, variable_cost, selling_price, request.target_profit)
            
            product_name = row[request.product_col] if request.product_col and request.product_col in df.columns else f"Product {idx + 1}"
            
            product = {
                'product': str(product_name),
                'fixed_cost': fixed_cost,
                'variable_cost': variable_cost,
                'selling_price': selling_price,
                'contribution_margin': be['contribution_margin'],
                'contribution_margin_ratio': be['contribution_margin_ratio'],
                'breakeven_units': be['breakeven_units'],
                'breakeven_revenue': be['breakeven_revenue'],
            }
            
            if request.current_units_col and request.current_units_col in df.columns:
                current_units = float(row[request.current_units_col])
                product['current_units'] = current_units
                product['current_profit'] = be['contribution_margin'] * current_units - fixed_cost
                if be['breakeven_units'] > 0 and be['breakeven_units'] != float('inf'):
                    product['margin_of_safety'] = (current_units - be['breakeven_units']) / current_units if current_units > 0 else 0
            
            products.append(product)
        
        if request.sales_mix_col and request.sales_mix_col in df.columns:
            weighted_cm = sum(p['contribution_margin'] * float(df.iloc[i][request.sales_mix_col]) 
                            for i, p in enumerate(products))
            weighted_cm_ratio = sum(p['contribution_margin_ratio'] * float(df.iloc[i][request.sales_mix_col]) 
                                   for i, p in enumerate(products))
        else:
            weighted_cm = np.mean([p['contribution_margin'] for p in products])
            weighted_cm_ratio = np.mean([p['contribution_margin_ratio'] for p in products])
        
        if weighted_cm > 0:
            overall_be_units = int(np.ceil((total_fixed_cost + request.target_profit) / weighted_cm))
            overall_be_revenue = (total_fixed_cost + request.target_profit) / weighted_cm_ratio if weighted_cm_ratio > 0 else 0
        else:
            overall_be_units = 0
            overall_be_revenue = 0
        
        sensitivity = []
        if products:
            p = products[0]
            sensitivity = calculate_sensitivity(
                p['breakeven_units'], p['fixed_cost'], p['variable_cost'], p['selling_price']
            )
        
        scenarios = []
        if products:
            p = products[0]
            for name, price_mult, cost_mult in [('Base', 1, 1), ('10% Price Increase', 1.1, 1), ('10% Cost Reduction', 1, 0.9)]:
                new_sp = p['selling_price'] * price_mult
                new_vc = p['variable_cost'] * cost_mult
                new_cm = new_sp - new_vc
                if new_cm > 0:
                    be_u = int(np.ceil(p['fixed_cost'] / new_cm))
                    be_r = p['fixed_cost'] / (new_cm / new_sp)
                    profit = new_cm * (p.get('current_units', be_u * 1.2)) - p['fixed_cost']
                else:
                    be_u = 0
                    be_r = 0
                    profit = -p['fixed_cost']
                scenarios.append({
                    'name': name,
                    'breakeven_units': be_u,
                    'breakeven_revenue': be_r,
                    'profit_at_target': profit
                })
        
        metrics = {
            'avg_contribution_margin': np.mean([p['contribution_margin'] for p in products]),
            'total_breakeven_units': sum(p['breakeven_units'] for p in products if p['breakeven_units'] != float('inf')),
            'days_to_breakeven': None
        }
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        visualizations = {
            'breakeven_chart': create_breakeven_chart(products, total_fixed_cost),
            'contribution_chart': create_contribution_chart(products),
            'profit_volume_chart': create_profit_volume_chart(products),
        }
        
        if sensitivity:
            visualizations['sensitivity_chart'] = create_sensitivity_chart(sensitivity, products[0]['breakeven_units'])
        
        key_insights = generate_key_insights(products, total_fixed_cost, overall_be_units, overall_be_revenue)
        
        results = {
            'products': [{k: _to_native_type(v) for k, v in p.items()} for p in products],
            'total_fixed_cost': total_fixed_cost,
            'weighted_avg_cm_ratio': weighted_cm_ratio,
            'overall_breakeven_revenue': overall_be_revenue,
            'sensitivity_analysis': [{k: _to_native_type(v) for k, v in s.items()} for s in sensitivity],
            'scenarios': [{k: _to_native_type(v) for k, v in s.items()} for s in scenarios],
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'analysis_type': request.analysis_type,
            'breakeven_units': overall_be_units,
            'breakeven_revenue': overall_be_revenue,
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
        raise HTTPException(status_code=500, detail=f"Break-even analysis failed: {str(e)}")
