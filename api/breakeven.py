"""
Break-even Analysis Router for FastAPI
Cost-Volume-Profit Analysis, Sensitivity Analysis, Multi-product Break-even
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
from scipy.optimize import brentq
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class BreakevenRequest(BaseModel):
    # Single product mode
    selling_price: Optional[float] = None
    variable_cost: Optional[float] = None
    fixed_costs: Optional[float] = None
    
    # Multi-product mode (from data)
    data: Optional[List[Dict[str, Any]]] = None
    product_col: Optional[str] = None
    price_col: Optional[str] = None
    variable_cost_col: Optional[str] = None
    sales_mix_col: Optional[str] = None  # Percentage or quantity
    
    # Additional parameters
    target_profit: Optional[float] = None
    tax_rate: float = 0
    
    # Sensitivity analysis ranges
    price_range_pct: float = 20  # +/- percentage for sensitivity
    cost_range_pct: float = 20
    volume_range_pct: float = 50


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


def calculate_breakeven_units(fixed_costs: float, selling_price: float, 
                               variable_cost: float) -> Dict[str, Any]:
    """Calculate break-even point in units"""
    contribution_margin = selling_price - variable_cost
    
    if contribution_margin <= 0:
        return {
            'error': 'Contribution margin must be positive',
            'contribution_margin': _to_native_type(contribution_margin)
        }
    
    breakeven_units = fixed_costs / contribution_margin
    breakeven_revenue = breakeven_units * selling_price
    
    # Contribution margin ratio
    cm_ratio = contribution_margin / selling_price
    
    return {
        'breakeven_units': _to_native_type(breakeven_units),
        'breakeven_revenue': _to_native_type(breakeven_revenue),
        'contribution_margin': _to_native_type(contribution_margin),
        'contribution_margin_ratio': _to_native_type(cm_ratio),
        'fixed_costs': _to_native_type(fixed_costs),
        'selling_price': _to_native_type(selling_price),
        'variable_cost': _to_native_type(variable_cost)
    }


def calculate_target_profit_volume(fixed_costs: float, selling_price: float,
                                    variable_cost: float, target_profit: float,
                                    tax_rate: float = 0) -> Dict[str, Any]:
    """Calculate volume needed for target profit"""
    contribution_margin = selling_price - variable_cost
    
    if contribution_margin <= 0:
        return {'error': 'Contribution margin must be positive'}
    
    # Adjust for taxes if applicable
    if tax_rate > 0:
        pre_tax_profit = target_profit / (1 - tax_rate)
    else:
        pre_tax_profit = target_profit
    
    required_units = (fixed_costs + pre_tax_profit) / contribution_margin
    required_revenue = required_units * selling_price
    
    return {
        'target_profit': _to_native_type(target_profit),
        'pre_tax_profit_needed': _to_native_type(pre_tax_profit),
        'required_units': _to_native_type(required_units),
        'required_revenue': _to_native_type(required_revenue),
        'tax_rate': _to_native_type(tax_rate)
    }


def calculate_margin_of_safety(actual_sales: float, breakeven_sales: float) -> Dict[str, Any]:
    """Calculate margin of safety"""
    margin_of_safety = actual_sales - breakeven_sales
    margin_of_safety_ratio = margin_of_safety / actual_sales if actual_sales > 0 else 0
    
    return {
        'margin_of_safety': _to_native_type(margin_of_safety),
        'margin_of_safety_ratio': _to_native_type(margin_of_safety_ratio),
        'margin_of_safety_pct': _to_native_type(margin_of_safety_ratio * 100)
    }


def calculate_operating_leverage(contribution_margin_total: float, 
                                  operating_income: float) -> Dict[str, Any]:
    """Calculate degree of operating leverage"""
    if operating_income == 0:
        return {'degree_of_operating_leverage': None, 'interpretation': 'At break-even point'}
    
    dol = contribution_margin_total / operating_income
    
    interpretation = ""
    if dol > 3:
        interpretation = "High operating leverage - profits very sensitive to sales changes"
    elif dol > 1.5:
        interpretation = "Moderate operating leverage"
    else:
        interpretation = "Low operating leverage - profits less sensitive to sales changes"
    
    return {
        'degree_of_operating_leverage': _to_native_type(dol),
        'interpretation': interpretation
    }


def perform_sensitivity_analysis(fixed_costs: float, selling_price: float,
                                  variable_cost: float, 
                                  price_range_pct: float = 20,
                                  cost_range_pct: float = 20) -> Dict[str, Any]:
    """Perform sensitivity analysis on key variables"""
    base_cm = selling_price - variable_cost
    base_be = fixed_costs / base_cm if base_cm > 0 else float('inf')
    
    results = {
        'base_case': {
            'breakeven_units': _to_native_type(base_be),
            'selling_price': _to_native_type(selling_price),
            'variable_cost': _to_native_type(variable_cost),
            'fixed_costs': _to_native_type(fixed_costs)
        },
        'price_sensitivity': [],
        'variable_cost_sensitivity': [],
        'fixed_cost_sensitivity': []
    }
    
    # Price sensitivity
    for pct in np.linspace(-price_range_pct, price_range_pct, 9):
        new_price = selling_price * (1 + pct/100)
        cm = new_price - variable_cost
        be = fixed_costs / cm if cm > 0 else float('inf')
        results['price_sensitivity'].append({
            'change_pct': _to_native_type(pct),
            'new_value': _to_native_type(new_price),
            'breakeven_units': _to_native_type(be),
            'change_from_base': _to_native_type((be - base_be) / base_be * 100) if base_be > 0 and be != float('inf') else None
        })
    
    # Variable cost sensitivity
    for pct in np.linspace(-cost_range_pct, cost_range_pct, 9):
        new_vc = variable_cost * (1 + pct/100)
        cm = selling_price - new_vc
        be = fixed_costs / cm if cm > 0 else float('inf')
        results['variable_cost_sensitivity'].append({
            'change_pct': _to_native_type(pct),
            'new_value': _to_native_type(new_vc),
            'breakeven_units': _to_native_type(be),
            'change_from_base': _to_native_type((be - base_be) / base_be * 100) if base_be > 0 and be != float('inf') else None
        })
    
    # Fixed cost sensitivity
    for pct in np.linspace(-cost_range_pct, cost_range_pct, 9):
        new_fc = fixed_costs * (1 + pct/100)
        be = new_fc / base_cm if base_cm > 0 else float('inf')
        results['fixed_cost_sensitivity'].append({
            'change_pct': _to_native_type(pct),
            'new_value': _to_native_type(new_fc),
            'breakeven_units': _to_native_type(be),
            'change_from_base': _to_native_type((be - base_be) / base_be * 100) if base_be > 0 else None
        })
    
    return results


def calculate_profit_volume_data(fixed_costs: float, selling_price: float,
                                  variable_cost: float, 
                                  max_volume: Optional[float] = None) -> Dict[str, Any]:
    """Generate data for profit-volume chart"""
    cm = selling_price - variable_cost
    breakeven_units = fixed_costs / cm if cm > 0 else 0
    
    if max_volume is None:
        max_volume = breakeven_units * 2.5
    
    volumes = np.linspace(0, max_volume, 100)
    
    revenues = volumes * selling_price
    total_costs = fixed_costs + (volumes * variable_cost)
    profits = revenues - total_costs
    
    return {
        'volumes': [_to_native_type(v) for v in volumes],
        'revenues': [_to_native_type(r) for r in revenues],
        'total_costs': [_to_native_type(c) for c in total_costs],
        'variable_costs': [_to_native_type(volumes[i] * variable_cost) for i in range(len(volumes))],
        'fixed_costs': [_to_native_type(fixed_costs)] * len(volumes),
        'profits': [_to_native_type(p) for p in profits],
        'breakeven_volume': _to_native_type(breakeven_units),
        'breakeven_revenue': _to_native_type(breakeven_units * selling_price)
    }


def analyze_multi_product(df: pd.DataFrame, product_col: str, price_col: str,
                          variable_cost_col: str, sales_mix_col: str,
                          fixed_costs: float) -> Dict[str, Any]:
    """Analyze multi-product break-even"""
    
    # Calculate contribution margin for each product
    df['contribution_margin'] = df[price_col] - df[variable_cost_col]
    df['cm_ratio'] = df['contribution_margin'] / df[price_col]
    
    # Normalize sales mix
    total_mix = df[sales_mix_col].sum()
    df['sales_mix_pct'] = df[sales_mix_col] / total_mix
    
    # Weighted average contribution margin
    weighted_cm = (df['contribution_margin'] * df['sales_mix_pct']).sum()
    weighted_price = (df[price_col] * df['sales_mix_pct']).sum()
    weighted_cm_ratio = weighted_cm / weighted_price if weighted_price > 0 else 0
    
    # Break-even in weighted units
    total_breakeven_units = fixed_costs / weighted_cm if weighted_cm > 0 else float('inf')
    
    # Break-even by product
    product_breakeven = []
    for _, row in df.iterrows():
        product_units = total_breakeven_units * row['sales_mix_pct']
        product_revenue = product_units * row[price_col]
        product_breakeven.append({
            'product': row[product_col],
            'selling_price': _to_native_type(row[price_col]),
            'variable_cost': _to_native_type(row[variable_cost_col]),
            'contribution_margin': _to_native_type(row['contribution_margin']),
            'cm_ratio': _to_native_type(row['cm_ratio']),
            'sales_mix_pct': _to_native_type(row['sales_mix_pct'] * 100),
            'breakeven_units': _to_native_type(product_units),
            'breakeven_revenue': _to_native_type(product_revenue)
        })
    
    return {
        'products': product_breakeven,
        'weighted_contribution_margin': _to_native_type(weighted_cm),
        'weighted_cm_ratio': _to_native_type(weighted_cm_ratio),
        'total_breakeven_units': _to_native_type(total_breakeven_units),
        'total_breakeven_revenue': _to_native_type(total_breakeven_units * weighted_price),
        'fixed_costs': _to_native_type(fixed_costs)
    }


def create_breakeven_chart(pv_data: Dict, breakeven: Dict) -> str:
    """Create break-even chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    volumes = pv_data['volumes']
    revenues = pv_data['revenues']
    total_costs = pv_data['total_costs']
    fixed_costs = pv_data['fixed_costs']
    be_volume = pv_data['breakeven_volume']
    be_revenue = pv_data['breakeven_revenue']
    
    # Cost-Volume-Profit chart
    ax1.fill_between(volumes, 0, fixed_costs, alpha=0.3, color='#ef4444', label='Fixed Costs')
    ax1.fill_between(volumes, fixed_costs, total_costs, alpha=0.3, color='#f59e0b', label='Variable Costs')
    ax1.plot(volumes, revenues, 'b-', linewidth=2.5, label='Revenue')
    ax1.plot(volumes, total_costs, 'r-', linewidth=2.5, label='Total Costs')
    
    # Break-even point
    ax1.scatter([be_volume], [be_revenue], color='green', s=150, zorder=5, edgecolors='white', linewidth=2)
    ax1.annotate(f'Break-even\n{be_volume:,.0f} units\n${be_revenue:,.0f}', 
                xy=(be_volume, be_revenue), xytext=(be_volume * 1.1, be_revenue * 1.1),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Profit/Loss regions
    ax1.fill_between(volumes, revenues, total_costs, where=[r > c for r, c in zip(revenues, total_costs)],
                    alpha=0.2, color='green', label='Profit Zone')
    ax1.fill_between(volumes, revenues, total_costs, where=[r <= c for r, c in zip(revenues, total_costs)],
                    alpha=0.2, color='red', label='Loss Zone')
    
    ax1.set_xlabel('Volume (Units)')
    ax1.set_ylabel('Amount ($)')
    ax1.set_title('Cost-Volume-Profit Analysis', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Profit chart
    profits = pv_data['profits']
    colors = ['#22c55e' if p >= 0 else '#ef4444' for p in profits]
    
    ax2.fill_between(volumes, 0, profits, where=[p >= 0 for p in profits], 
                    alpha=0.4, color='#22c55e', label='Profit')
    ax2.fill_between(volumes, 0, profits, where=[p < 0 for p in profits], 
                    alpha=0.4, color='#ef4444', label='Loss')
    ax2.plot(volumes, profits, 'k-', linewidth=2)
    ax2.axhline(y=0, color='black', linewidth=1.5)
    ax2.scatter([be_volume], [0], color='green', s=150, zorder=5, edgecolors='white', linewidth=2)
    
    ax2.set_xlabel('Volume (Units)')
    ax2.set_ylabel('Profit/Loss ($)')
    ax2.set_title('Profit Analysis', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_sensitivity_chart(sensitivity: Dict) -> str:
    """Create sensitivity analysis chart"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Price sensitivity
    ax = axes[0]
    price_data = sensitivity['price_sensitivity']
    changes = [d['change_pct'] for d in price_data]
    be_units = [d['breakeven_units'] if d['breakeven_units'] and d['breakeven_units'] < 1e10 else None for d in price_data]
    
    valid_idx = [i for i, v in enumerate(be_units) if v is not None]
    valid_changes = [changes[i] for i in valid_idx]
    valid_be = [be_units[i] for i in valid_idx]
    
    colors = ['#22c55e' if c > 0 else '#ef4444' for c in valid_changes]
    ax.bar(valid_changes, valid_be, color=colors, edgecolor='white', linewidth=2, width=4)
    ax.axhline(y=sensitivity['base_case']['breakeven_units'], color='blue', linestyle='--', linewidth=2, label='Base Case')
    ax.set_xlabel('Price Change (%)')
    ax.set_ylabel('Break-even Units')
    ax.set_title('Price Sensitivity', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Variable cost sensitivity
    ax = axes[1]
    vc_data = sensitivity['variable_cost_sensitivity']
    changes = [d['change_pct'] for d in vc_data]
    be_units = [d['breakeven_units'] if d['breakeven_units'] and d['breakeven_units'] < 1e10 else None for d in vc_data]
    
    valid_idx = [i for i, v in enumerate(be_units) if v is not None]
    valid_changes = [changes[i] for i in valid_idx]
    valid_be = [be_units[i] for i in valid_idx]
    
    colors = ['#ef4444' if c > 0 else '#22c55e' for c in valid_changes]
    ax.bar(valid_changes, valid_be, color=colors, edgecolor='white', linewidth=2, width=4)
    ax.axhline(y=sensitivity['base_case']['breakeven_units'], color='blue', linestyle='--', linewidth=2, label='Base Case')
    ax.set_xlabel('Variable Cost Change (%)')
    ax.set_ylabel('Break-even Units')
    ax.set_title('Variable Cost Sensitivity', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Fixed cost sensitivity
    ax = axes[2]
    fc_data = sensitivity['fixed_cost_sensitivity']
    changes = [d['change_pct'] for d in fc_data]
    be_units = [d['breakeven_units'] for d in fc_data]
    
    colors = ['#ef4444' if c > 0 else '#22c55e' for c in changes]
    ax.bar(changes, be_units, color=colors, edgecolor='white', linewidth=2, width=4)
    ax.axhline(y=sensitivity['base_case']['breakeven_units'], color='blue', linestyle='--', linewidth=2, label='Base Case')
    ax.set_xlabel('Fixed Cost Change (%)')
    ax.set_ylabel('Break-even Units')
    ax.set_title('Fixed Cost Sensitivity', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_contribution_chart(breakeven: Dict) -> str:
    """Create contribution margin chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Unit economics waterfall
    price = breakeven['selling_price']
    vc = breakeven['variable_cost']
    cm = breakeven['contribution_margin']
    
    categories = ['Selling Price', 'Variable Cost', 'Contribution Margin']
    values = [price, -vc, cm]
    colors = ['#3b82f6', '#ef4444', '#22c55e']
    
    # Waterfall
    cumulative = 0
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        if i == 0:
            ax1.bar(cat, val, color=color, edgecolor='white', linewidth=2)
            cumulative = val
        elif i == 1:
            ax1.bar(cat, abs(val), bottom=cumulative + val, color=color, edgecolor='white', linewidth=2)
            ax1.plot([i-0.4, i+0.4], [cumulative, cumulative], 'k--', linewidth=1)
            cumulative += val
        else:
            ax1.bar(cat, val, color=color, edgecolor='white', linewidth=2)
    
    ax1.set_ylabel('Amount ($)')
    ax1.set_title('Unit Economics', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for i, val in enumerate([price, vc, cm]):
        ax1.text(i, val/2 if i != 1 else price - vc/2, f'${val:,.2f}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # CM ratio pie
    cm_ratio = breakeven['contribution_margin_ratio']
    vc_ratio = 1 - cm_ratio
    
    wedges, texts, autotexts = ax2.pie([cm_ratio, vc_ratio], 
                                        labels=['Contribution\nMargin', 'Variable\nCost'],
                                        colors=['#22c55e', '#ef4444'],
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        explode=[0.05, 0],
                                        textprops={'fontsize': 10})
    ax2.set_title(f'CM Ratio: {cm_ratio*100:.1f}%', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_multi_product_chart(mp_data: Dict) -> str:
    """Create multi-product break-even chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    products = mp_data['products']
    names = [p['product'] for p in products]
    cms = [p['contribution_margin'] for p in products]
    be_units = [p['breakeven_units'] for p in products]
    mix_pcts = [p['sales_mix_pct'] for p in products]
    
    # Contribution margin by product
    colors = plt.cm.Set2(np.linspace(0, 1, len(products)))
    bars = ax1.bar(names, cms, color=colors, edgecolor='white', linewidth=2)
    ax1.axhline(y=mp_data['weighted_contribution_margin'], color='red', linestyle='--', 
                linewidth=2, label=f"Weighted Avg: ${mp_data['weighted_contribution_margin']:.2f}")
    ax1.set_ylabel('Contribution Margin ($)')
    ax1.set_title('Contribution Margin by Product', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, cm in zip(bars, cms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cms) * 0.02,
                f'${cm:.2f}', ha='center', fontsize=9)
    
    # Break-even units by product (stacked)
    bottom = 0
    for i, (name, units, color) in enumerate(zip(names, be_units, colors)):
        ax2.bar(['Break-even Mix'], [units], bottom=[bottom], color=color, 
               edgecolor='white', linewidth=2, label=f'{name}: {units:,.0f}')
        bottom += units
    
    ax2.set_ylabel('Units')
    ax2.set_title(f'Total Break-even: {mp_data["total_breakeven_units"]:,.0f} units', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(breakeven: Dict, sensitivity: Optional[Dict] = None,
                      target_profit: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    # Break-even insight
    be_units = breakeven.get('breakeven_units', 0)
    cm_ratio = breakeven.get('contribution_margin_ratio', 0)
    
    insights.append({
        'title': f'Break-even Point: {be_units:,.0f} units',
        'description': f'Need ${breakeven.get("breakeven_revenue", 0):,.0f} in revenue to cover all costs.',
        'status': 'neutral'
    })
    
    # CM ratio insight
    if cm_ratio >= 0.4:
        insights.append({
            'title': f'Strong CM Ratio: {cm_ratio*100:.1f}%',
            'description': 'High contribution margin provides good profit potential.',
            'status': 'positive'
        })
    elif cm_ratio >= 0.2:
        insights.append({
            'title': f'Moderate CM Ratio: {cm_ratio*100:.1f}%',
            'description': 'Average margin. Consider cost reduction strategies.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low CM Ratio: {cm_ratio*100:.1f}%',
            'description': 'Thin margins. Focus on volume or price increases.',
            'status': 'warning'
        })
    
    # Sensitivity insight
    if sensitivity:
        price_sens = sensitivity['price_sensitivity']
        # Find 10% price increase effect
        for p in price_sens:
            if abs(p['change_pct'] - 10) < 1:
                be_change = p.get('change_from_base', 0)
                if be_change:
                    insights.append({
                        'title': f'Price +10% → BE {be_change:.1f}%',
                        'description': f'10% price increase would change break-even by {be_change:.1f}%.',
                        'status': 'positive' if be_change < 0 else 'warning'
                    })
                break
    
    # Target profit insight
    if target_profit and 'required_units' in target_profit:
        units_above_be = target_profit['required_units'] - be_units
        insights.append({
            'title': f'Target Profit: {target_profit["required_units"]:,.0f} units',
            'description': f'Need {units_above_be:,.0f} units above break-even for target profit.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/breakeven")
async def run_breakeven_analysis(request: BreakevenRequest) -> Dict[str, Any]:
    """
    Perform Break-even Analysis.
    """
    try:
        results = {}
        visualizations = {}
        
        # Determine analysis mode
        if request.data and request.product_col:
            # Multi-product mode
            df = pd.DataFrame(request.data)
            
            if not all(col in df.columns for col in [request.product_col, request.price_col, 
                                                       request.variable_cost_col, request.sales_mix_col]):
                raise HTTPException(status_code=400, detail="Missing required columns for multi-product analysis")
            
            if not request.fixed_costs:
                raise HTTPException(status_code=400, detail="Fixed costs required")
            
            mp_analysis = analyze_multi_product(
                df, request.product_col, request.price_col,
                request.variable_cost_col, request.sales_mix_col,
                request.fixed_costs
            )
            results['multi_product'] = mp_analysis
            visualizations['multi_product_chart'] = create_multi_product_chart(mp_analysis)
            
            # Use weighted averages for other analyses
            selling_price = sum(p['selling_price'] * p['sales_mix_pct'] / 100 for p in mp_analysis['products'])
            variable_cost = sum(p['variable_cost'] * p['sales_mix_pct'] / 100 for p in mp_analysis['products'])
            fixed_costs = request.fixed_costs
            
        else:
            # Single product mode
            if not all([request.selling_price, request.variable_cost, request.fixed_costs]):
                raise HTTPException(status_code=400, detail="Selling price, variable cost, and fixed costs are required")
            
            selling_price = request.selling_price
            variable_cost = request.variable_cost
            fixed_costs = request.fixed_costs
        
        # Basic break-even calculation
        breakeven = calculate_breakeven_units(fixed_costs, selling_price, variable_cost)
        if 'error' in breakeven:
            raise HTTPException(status_code=400, detail=breakeven['error'])
        results['breakeven'] = breakeven
        
        # Profit-volume data
        pv_data = calculate_profit_volume_data(fixed_costs, selling_price, variable_cost)
        results['profit_volume_data'] = pv_data
        visualizations['breakeven_chart'] = create_breakeven_chart(pv_data, breakeven)
        visualizations['contribution_chart'] = create_contribution_chart(breakeven)
        
        # Target profit analysis
        if request.target_profit:
            target = calculate_target_profit_volume(
                fixed_costs, selling_price, variable_cost,
                request.target_profit, request.tax_rate
            )
            results['target_profit'] = target
        
        # Sensitivity analysis
        sensitivity = perform_sensitivity_analysis(
            fixed_costs, selling_price, variable_cost,
            request.price_range_pct, request.cost_range_pct
        )
        results['sensitivity'] = sensitivity
        visualizations['sensitivity_chart'] = create_sensitivity_chart(sensitivity)
        
        # Generate insights
        insights = generate_insights(
            breakeven, sensitivity,
            results.get('target_profit')
        )
        
        # Summary
        summary = {
            'breakeven_units': breakeven['breakeven_units'],
            'breakeven_revenue': breakeven['breakeven_revenue'],
            'contribution_margin': breakeven['contribution_margin'],
            'contribution_margin_ratio': breakeven['contribution_margin_ratio'],
            'selling_price': selling_price,
            'variable_cost': variable_cost,
            'fixed_costs': fixed_costs,
            'target_profit_units': results.get('target_profit', {}).get('required_units')
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
        raise HTTPException(status_code=500, detail=f"Break-even analysis failed: {str(e)}")
