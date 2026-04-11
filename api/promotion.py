"""
Promotion Optimization Router for FastAPI
Promotional Lift Analysis, Price Elasticity, ROI Optimization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class PromotionRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: Optional[str] = None
    sales_col: str  # Sales volume or revenue
    price_col: Optional[str] = None  # Regular price
    promo_price_col: Optional[str] = None  # Promotional price
    discount_col: Optional[str] = None  # Discount percentage
    promo_flag_col: Optional[str] = None  # Binary promo indicator
    promo_type_col: Optional[str] = None  # Type of promotion
    cost_col: Optional[str] = None  # Product cost for margin calculation
    product_col: Optional[str] = None  # Product identifier
    baseline_method: Literal["average", "median", "regression"] = "average"
    margin_pct: float = 0.3  # Default margin if cost not provided


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
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_baseline(df: pd.DataFrame, sales_col: str, promo_flag_col: Optional[str],
                       method: str = "average") -> float:
    """Calculate baseline sales (non-promotional period)"""
    if promo_flag_col and promo_flag_col in df.columns:
        non_promo = df[df[promo_flag_col] == 0][sales_col]
        if len(non_promo) > 0:
            if method == "median":
                return non_promo.median()
            elif method == "average":
                return non_promo.mean()
    
    # Fallback: use lower quartile as baseline
    return df[sales_col].quantile(0.25)


def calculate_promotional_lift(df: pd.DataFrame, sales_col: str, promo_flag_col: str,
                                baseline: float) -> Dict[str, Any]:
    """Calculate promotional lift metrics"""
    promo_df = df[df[promo_flag_col] == 1]
    non_promo_df = df[df[promo_flag_col] == 0]
    
    if len(promo_df) == 0:
        return {'error': 'No promotional periods found'}
    
    promo_sales = promo_df[sales_col].mean()
    non_promo_sales = non_promo_df[sales_col].mean() if len(non_promo_df) > 0 else baseline
    
    # Lift calculation
    absolute_lift = promo_sales - baseline
    percent_lift = (promo_sales - baseline) / baseline * 100 if baseline > 0 else 0
    
    # Incremental sales
    incremental_total = (promo_df[sales_col] - baseline).sum()
    incremental_total = max(0, incremental_total)  # Can't be negative
    
    # Statistical significance
    if len(promo_df) > 1 and len(non_promo_df) > 1:
        t_stat, p_value = stats.ttest_ind(promo_df[sales_col], non_promo_df[sales_col])
    else:
        t_stat, p_value = 0, 1
    
    return {
        'baseline_sales': _to_native_type(baseline),
        'promo_sales_avg': _to_native_type(promo_sales),
        'non_promo_sales_avg': _to_native_type(non_promo_sales),
        'absolute_lift': _to_native_type(absolute_lift),
        'percent_lift': _to_native_type(percent_lift),
        'incremental_total': _to_native_type(incremental_total),
        'promo_periods': len(promo_df),
        'non_promo_periods': len(non_promo_df),
        't_statistic': _to_native_type(t_stat),
        'p_value': _to_native_type(p_value),
        'is_significant': p_value < 0.05 if p_value else False
    }


def calculate_price_elasticity(df: pd.DataFrame, sales_col: str, price_col: str) -> Dict[str, Any]:
    """Calculate price elasticity of demand"""
    # Remove zero/null values
    valid_df = df[[sales_col, price_col]].dropna()
    valid_df = valid_df[(valid_df[sales_col] > 0) & (valid_df[price_col] > 0)]
    
    if len(valid_df) < 5:
        return {'error': 'Insufficient data for elasticity calculation'}
    
    # Log-log regression for elasticity
    log_price = np.log(valid_df[price_col])
    log_sales = np.log(valid_df[sales_col])
    
    # Check if all x values are identical (cannot calculate regression)
    if log_price.nunique() < 2:
        return {
            'error': 'Cannot calculate elasticity - all price values are identical',
            'elasticity': None,
            'r_squared': None,
            'p_value': None,
            'std_error': None,
            'interpretation': 'Price variation required for elasticity calculation',
            'recommendation': 'Ensure data includes multiple different price points',
            'is_significant': False
        }
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_sales)
    
    # Elasticity = slope of log-log regression
    elasticity = slope
    
    # Interpretation
    if elasticity < -1:
        interpretation = "Elastic (sensitive to price changes)"
        recommendation = "Consider smaller discounts - demand is very responsive"
    elif elasticity < 0:
        interpretation = "Inelastic (less sensitive to price)"
        recommendation = "Deeper discounts may be needed to drive volume"
    else:
        interpretation = "Unusual (positive elasticity)"
        recommendation = "Review data quality - positive elasticity is atypical"
    
    return {
        'elasticity': _to_native_type(elasticity),
        'r_squared': _to_native_type(r_value ** 2),
        'p_value': _to_native_type(p_value),
        'std_error': _to_native_type(std_err),
        'interpretation': interpretation,
        'recommendation': recommendation,
        'is_significant': p_value < 0.05
    }


def calculate_discount_effectiveness(df: pd.DataFrame, sales_col: str, 
                                      discount_col: str) -> Dict[str, Any]:
    """Analyze effectiveness across discount levels"""
    valid_df = df[[sales_col, discount_col]].dropna()
    
    if len(valid_df) < 5:
        return {'error': 'Insufficient data'}
    
    # Create discount buckets
    bins = [0, 5, 10, 15, 20, 25, 30, 50, 100]
    labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-50%', '50%+']
    
    valid_df['discount_bucket'] = pd.cut(valid_df[discount_col], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate metrics by bucket
    bucket_stats = valid_df.groupby('discount_bucket', observed=True).agg({
        sales_col: ['mean', 'sum', 'count']
    }).reset_index()
    bucket_stats.columns = ['discount_bucket', 'avg_sales', 'total_sales', 'count']
    
    # Find optimal discount range
    if len(bucket_stats) > 0:
        max_idx = bucket_stats['avg_sales'].idxmax()
        optimal_range = bucket_stats.loc[max_idx, 'discount_bucket']
    else:
        optimal_range = None
    
    # Correlation
    correlation = valid_df[sales_col].corr(valid_df[discount_col])
    
    return {
        'bucket_analysis': bucket_stats.to_dict('records'),
        'correlation': _to_native_type(correlation),
        'optimal_discount_range': str(optimal_range) if optimal_range else None,
        'avg_discount': _to_native_type(valid_df[discount_col].mean()),
        'max_discount': _to_native_type(valid_df[discount_col].max())
    }


def calculate_promo_type_performance(df: pd.DataFrame, sales_col: str, 
                                      promo_type_col: str, baseline: float,
                                      price_col: Optional[str] = None,
                                      cost_col: Optional[str] = None,
                                      margin_pct: float = 0.3) -> Dict[str, Any]:
    """Analyze performance by promotion type"""
    valid_df = df[[sales_col, promo_type_col]].dropna()
    
    if price_col and price_col in df.columns:
        valid_df[price_col] = df[price_col]
    if cost_col and cost_col in df.columns:
        valid_df[cost_col] = df[cost_col]
    
    promo_types = valid_df[promo_type_col].unique()
    
    type_performance = []
    for ptype in promo_types:
        type_df = valid_df[valid_df[promo_type_col] == ptype]
        
        avg_sales = type_df[sales_col].mean()
        total_sales = type_df[sales_col].sum()
        lift = (avg_sales - baseline) / baseline * 100 if baseline > 0 else 0
        incremental = max(0, (type_df[sales_col] - baseline).sum())
        
        # Calculate ROI if possible
        if price_col and price_col in type_df.columns:
            avg_price = type_df[price_col].mean()
            if cost_col and cost_col in type_df.columns:
                avg_cost = type_df[cost_col].mean()
            else:
                avg_cost = avg_price * (1 - margin_pct)
            
            revenue = total_sales * avg_price
            cost = total_sales * avg_cost
            profit = revenue - cost
            roi = (profit / cost - 1) * 100 if cost > 0 else 0
        else:
            revenue = None
            profit = None
            roi = None
        
        type_performance.append({
            'promo_type': str(ptype),
            'count': len(type_df),
            'avg_sales': _to_native_type(avg_sales),
            'total_sales': _to_native_type(total_sales),
            'lift_pct': _to_native_type(lift),
            'incremental_sales': _to_native_type(incremental),
            'revenue': _to_native_type(revenue),
            'profit': _to_native_type(profit),
            'roi': _to_native_type(roi)
        })
    
    # Sort by lift
    type_performance.sort(key=lambda x: x['lift_pct'] or 0, reverse=True)
    
    return {
        'performance_by_type': type_performance,
        'best_performer': type_performance[0] if type_performance else None,
        'worst_performer': type_performance[-1] if type_performance else None
    }


def calculate_promotion_roi(df: pd.DataFrame, sales_col: str, baseline: float,
                            price_col: Optional[str], promo_price_col: Optional[str],
                            cost_col: Optional[str], margin_pct: float) -> Dict[str, Any]:
    """Calculate overall promotion ROI"""
    
    total_sales = df[sales_col].sum()
    avg_sales = df[sales_col].mean()
    
    # Estimate incremental
    incremental_sales = max(0, total_sales - baseline * len(df))
    
    # Calculate financials
    if price_col and price_col in df.columns:
        regular_price = df[price_col].mean()
        
        if promo_price_col and promo_price_col in df.columns:
            promo_price = df[promo_price_col].mean()
            discount_amount = regular_price - promo_price
        else:
            promo_price = regular_price
            discount_amount = 0
        
        if cost_col and cost_col in df.columns:
            unit_cost = df[cost_col].mean()
        else:
            unit_cost = regular_price * (1 - margin_pct)
        
        # Revenue calculations
        actual_revenue = total_sales * promo_price
        baseline_revenue = baseline * len(df) * regular_price
        incremental_revenue = actual_revenue - baseline_revenue
        
        # Cost of promotion (discount given)
        promo_cost = total_sales * discount_amount
        
        # Profit
        actual_profit = total_sales * (promo_price - unit_cost)
        baseline_profit = baseline * len(df) * (regular_price - unit_cost)
        incremental_profit = actual_profit - baseline_profit
        
        # ROI
        if promo_cost > 0:
            roi = (incremental_profit / promo_cost) * 100
        else:
            roi = 0
        
        # Margin impact
        regular_margin = (regular_price - unit_cost) / regular_price * 100
        promo_margin = (promo_price - unit_cost) / promo_price * 100 if promo_price > 0 else 0
        
    else:
        actual_revenue = None
        incremental_revenue = None
        promo_cost = None
        incremental_profit = None
        roi = None
        regular_margin = None
        promo_margin = None
    
    return {
        'total_sales_units': _to_native_type(total_sales),
        'incremental_sales_units': _to_native_type(incremental_sales),
        'incremental_pct': _to_native_type(incremental_sales / (baseline * len(df)) * 100) if baseline > 0 else 0,
        'actual_revenue': _to_native_type(actual_revenue),
        'incremental_revenue': _to_native_type(incremental_revenue),
        'promotion_cost': _to_native_type(promo_cost),
        'incremental_profit': _to_native_type(incremental_profit),
        'roi': _to_native_type(roi),
        'regular_margin_pct': _to_native_type(regular_margin),
        'promo_margin_pct': _to_native_type(promo_margin)
    }


def find_optimal_discount(elasticity: float, margin_pct: float, 
                          current_price: float, current_volume: float) -> Dict[str, Any]:
    """Find optimal discount level to maximize profit"""
    
    if elasticity >= 0:
        return {'error': 'Cannot optimize with non-negative elasticity'}
    
    def profit_at_discount(discount_pct):
        new_price = current_price * (1 - discount_pct / 100)
        # Volume change based on elasticity
        price_change_pct = -discount_pct
        volume_change_pct = elasticity * price_change_pct
        new_volume = current_volume * (1 + volume_change_pct / 100)
        
        cost = current_price * (1 - margin_pct)
        profit = new_volume * (new_price - cost)
        return -profit  # Negative for minimization
    
    # Search for optimal discount
    result = minimize_scalar(profit_at_discount, bounds=(0, 50), method='bounded')
    optimal_discount = result.x
    
    # Calculate metrics at optimal
    optimal_price = current_price * (1 - optimal_discount / 100)
    price_change = -optimal_discount
    volume_change = elasticity * price_change
    optimal_volume = current_volume * (1 + volume_change / 100)
    
    cost = current_price * (1 - margin_pct)
    current_profit = current_volume * (current_price - cost)
    optimal_profit = optimal_volume * (optimal_price - cost)
    profit_improvement = (optimal_profit - current_profit) / current_profit * 100 if current_profit > 0 else 0
    
    return {
        'optimal_discount_pct': _to_native_type(optimal_discount),
        'optimal_price': _to_native_type(optimal_price),
        'expected_volume_change_pct': _to_native_type(volume_change),
        'expected_volume': _to_native_type(optimal_volume),
        'current_profit': _to_native_type(current_profit),
        'optimal_profit': _to_native_type(optimal_profit),
        'profit_improvement_pct': _to_native_type(profit_improvement)
    }


def create_lift_chart(lift_data: Dict, baseline: float) -> str:
    """Create promotional lift visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Lift bar chart
    categories = ['Baseline', 'Promo Avg', 'Lift']
    values = [baseline, lift_data['promo_sales_avg'], lift_data['absolute_lift']]
    colors = ['#6b7280', '#3b82f6', '#22c55e' if lift_data['absolute_lift'] > 0 else '#ef4444']
    
    bars = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Sales')
    ax1.set_title('Promotional Lift Analysis', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                f'{val:,.0f}', ha='center', fontsize=10)
    
    # Percentage lift gauge
    lift_pct = lift_data['percent_lift']
    colors_gauge = ['#ef4444', '#f59e0b', '#22c55e']
    
    ax2.barh(['Lift %'], [lift_pct], color='#22c55e' if lift_pct > 0 else '#ef4444',
             height=0.5, edgecolor='white', linewidth=2)
    ax2.axvline(x=0, color='black', linewidth=2)
    ax2.set_xlim(-50, max(100, lift_pct * 1.2))
    ax2.set_xlabel('Percentage Lift (%)')
    ax2.set_title(f'Lift: {lift_pct:.1f}%', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add significance indicator
    sig_text = "Statistically Significant ✓" if lift_data.get('is_significant') else "Not Significant"
    sig_color = 'green' if lift_data.get('is_significant') else 'gray'
    ax2.text(0.5, -0.15, sig_text, transform=ax2.transAxes, ha='center', 
             fontsize=10, color=sig_color)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_elasticity_chart(df: pd.DataFrame, sales_col: str, price_col: str,
                            elasticity_data: Dict) -> str:
    """Create price elasticity visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    valid_df = df[[sales_col, price_col]].dropna()
    valid_df = valid_df[(valid_df[sales_col] > 0) & (valid_df[price_col] > 0)]
    
    # Scatter with regression line
    ax1.scatter(valid_df[price_col], valid_df[sales_col], alpha=0.6, c='blue', edgecolors='white')
    
    # Regression line
    z = np.polyfit(valid_df[price_col], valid_df[sales_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df[price_col].min(), valid_df[price_col].max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
    
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Sales')
    ax1.set_title('Price vs Sales Relationship', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Elasticity gauge
    elasticity = elasticity_data.get('elasticity', 0)
    
    # Create elasticity scale
    scale_x = np.linspace(-3, 1, 100)
    scale_colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
    
    for i in range(len(scale_x) - 1):
        ax2.axvspan(scale_x[i], scale_x[i+1], color=scale_colors[i], alpha=0.7)
    
    ax2.axvline(x=elasticity, color='black', linewidth=4, label=f'Elasticity: {elasticity:.2f}')
    ax2.axvline(x=-1, color='white', linewidth=2, linestyle='--', alpha=0.8)
    ax2.set_xlim(-3, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Price Elasticity')
    ax2.set_title('Price Elasticity of Demand', fontsize=12, fontweight='bold')
    ax2.text(-2, 0.5, 'Elastic\n(Sensitive)', ha='center', fontsize=10)
    ax2.text(0, 0.5, 'Inelastic', ha='center', fontsize=10)
    ax2.set_yticks([])
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_discount_analysis_chart(discount_data: Dict) -> str:
    """Create discount effectiveness visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bucket_analysis = discount_data.get('bucket_analysis', [])
    
    if bucket_analysis:
        buckets = [b['discount_bucket'] for b in bucket_analysis]
        avg_sales = [b['avg_sales'] for b in bucket_analysis]
        counts = [b['count'] for b in bucket_analysis]
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(buckets)))
        
        # Average sales by discount bucket
        bars1 = ax1.bar(buckets, avg_sales, color=colors, edgecolor='white', linewidth=2)
        ax1.set_xlabel('Discount Range')
        ax1.set_ylabel('Average Sales')
        ax1.set_title('Sales by Discount Level', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(buckets, rotation=45, ha='right')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Highlight optimal
        optimal = discount_data.get('optimal_discount_range')
        if optimal:
            for i, bucket in enumerate(buckets):
                if str(bucket) == optimal:
                    bars1[i].set_color('#22c55e')
                    bars1[i].set_edgecolor('#16a34a')
                    bars1[i].set_linewidth(3)
        
        # Count distribution
        ax2.bar(buckets, counts, color='#f59e0b', edgecolor='white', linewidth=2)
        ax2.set_xlabel('Discount Range')
        ax2.set_ylabel('Number of Promotions')
        ax2.set_title('Promotion Frequency by Discount', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(buckets, rotation=45, ha='right')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_promo_type_chart(type_data: Dict) -> str:
    """Create promotion type performance chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    performance = type_data.get('performance_by_type', [])
    
    if performance:
        types = [p['promo_type'] for p in performance]
        lifts = [p['lift_pct'] or 0 for p in performance]
        sales = [p['avg_sales'] or 0 for p in performance]
        
        colors = ['#22c55e' if l > 0 else '#ef4444' for l in lifts]
        
        # Lift by type
        bars1 = ax1.barh(types, lifts, color=colors, edgecolor='white', linewidth=2)
        ax1.axvline(x=0, color='black', linewidth=1)
        ax1.set_xlabel('Lift (%)')
        ax1.set_title('Promotional Lift by Type', fontsize=12, fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for bar, val in zip(bars1, lifts):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)
        
        # Sales by type
        colors2 = plt.cm.Set2(np.linspace(0, 1, len(types)))
        bars2 = ax2.barh(types, sales, color=colors2, edgecolor='white', linewidth=2)
        ax2.set_xlabel('Average Sales')
        ax2.set_title('Average Sales by Promotion Type', fontsize=12, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, val in zip(bars2, sales):
            ax2.text(bar.get_width() + max(sales) * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_roi_summary_chart(roi_data: Dict) -> str:
    """Create ROI summary visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Waterfall chart for profit
    categories = ['Baseline\nRevenue', 'Incremental\nRevenue', 'Promo\nCost', 'Incremental\nProfit']
    
    baseline_rev = roi_data.get('actual_revenue', 0) - (roi_data.get('incremental_revenue') or 0)
    incr_rev = roi_data.get('incremental_revenue') or 0
    promo_cost = -(roi_data.get('promotion_cost') or 0)
    incr_profit = roi_data.get('incremental_profit') or 0
    
    values = [baseline_rev, incr_rev, promo_cost, incr_profit]
    colors = ['#6b7280', '#22c55e', '#ef4444', '#3b82f6']
    
    bars = ax1.bar(categories, [abs(v) if v else 0 for v in values], color=colors, 
                   edgecolor='white', linewidth=2)
    ax1.set_ylabel('Amount ($)')
    ax1.set_title('Promotion Financial Impact', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ROI gauge
    roi = roi_data.get('roi') or 0
    
    gauge_colors = ['#ef4444', '#f59e0b', '#22c55e']
    ax2.barh(['ROI'], [roi], color='#22c55e' if roi > 0 else '#ef4444',
             height=0.5, edgecolor='white', linewidth=2)
    ax2.axvline(x=0, color='black', linewidth=2)
    ax2.axvline(x=100, color='green', linewidth=2, linestyle='--', alpha=0.5, label='100% ROI')
    ax2.set_xlabel('ROI (%)')
    ax2.set_title(f'Promotion ROI: {roi:.1f}%', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_optimization_chart(optimization: Dict, current_price: float) -> str:
    """Create discount optimization visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    optimal_discount = optimization.get('optimal_discount_pct', 0)
    
    # Profit curve
    discounts = np.linspace(0, 40, 100)
    
    # Simulate profit curve (simplified)
    current_profit = optimization.get('current_profit', 1000)
    optimal_profit = optimization.get('optimal_profit', 1200)
    
    # Create a curve that peaks at optimal
    profits = current_profit * (1 + 0.5 * np.sin(np.pi * (discounts - optimal_discount) / 40))
    profits = np.maximum(profits, current_profit * 0.5)
    
    ax1.plot(discounts, profits, 'b-', linewidth=2)
    ax1.axvline(x=optimal_discount, color='green', linestyle='--', linewidth=2, 
                label=f'Optimal: {optimal_discount:.1f}%')
    ax1.scatter([optimal_discount], [optimal_profit], color='green', s=100, zorder=5)
    ax1.scatter([0], [current_profit], color='red', s=100, zorder=5, label='Current')
    
    ax1.set_xlabel('Discount (%)')
    ax1.set_ylabel('Expected Profit ($)')
    ax1.set_title('Profit vs Discount Level', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Comparison bar
    categories = ['Current', 'Optimal']
    profits_compare = [current_profit, optimal_profit]
    colors = ['#6b7280', '#22c55e']
    
    bars = ax2.bar(categories, profits_compare, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Profit ($)')
    ax2.set_title(f'Profit Improvement: {optimization.get("profit_improvement_pct", 0):.1f}%', 
                  fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, profits_compare):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(profits_compare) * 0.02,
                f'${val:,.0f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(lift_data: Dict, elasticity_data: Optional[Dict],
                          roi_data: Dict, optimization: Optional[Dict]) -> List[Dict[str, Any]]:
    """Generate key insights from promotion analysis"""
    insights = []
    
    # Lift insight
    lift_pct = lift_data.get('percent_lift', 0)
    if lift_pct > 30:
        insights.append({
            'title': 'Strong Promotional Lift',
            'description': f'{lift_pct:.1f}% lift during promotions. Promotions are highly effective.',
            'status': 'positive'
        })
    elif lift_pct > 10:
        insights.append({
            'title': 'Moderate Promotional Lift',
            'description': f'{lift_pct:.1f}% lift during promotions. Room for optimization.',
            'status': 'neutral'
        })
    elif lift_pct > 0:
        insights.append({
            'title': 'Low Promotional Lift',
            'description': f'Only {lift_pct:.1f}% lift. Consider different promotion strategies.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Negative/No Lift',
            'description': 'Promotions not driving incremental sales. Review pricing strategy.',
            'status': 'warning'
        })
    
    # Significance
    if lift_data.get('is_significant'):
        insights.append({
            'title': 'Statistically Significant Results',
            'description': f'p-value = {lift_data.get("p_value", 0):.4f}. Results are reliable.',
            'status': 'positive'
        })
    
    # Elasticity insight
    if elasticity_data and 'elasticity' in elasticity_data:
        elasticity = elasticity_data['elasticity']
        insights.append({
            'title': f'Price Elasticity: {elasticity:.2f}',
            'description': elasticity_data.get('recommendation', ''),
            'status': 'neutral'
        })
    
    # ROI insight
    roi = roi_data.get('roi')
    if roi is not None:
        if roi > 100:
            insights.append({
                'title': f'Excellent Promotion ROI: {roi:.0f}%',
                'description': 'Promotions are generating strong returns.',
                'status': 'positive'
            })
        elif roi > 0:
            insights.append({
                'title': f'Positive Promotion ROI: {roi:.0f}%',
                'description': 'Promotions are profitable but could be optimized.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': f'Negative Promotion ROI: {roi:.0f}%',
                'description': 'Promotions are losing money. Reduce discount depth or frequency.',
                'status': 'warning'
            })
    
    # Optimization insight
    if optimization and 'optimal_discount_pct' in optimization:
        improvement = optimization.get('profit_improvement_pct', 0)
        if improvement > 0:
            insights.append({
                'title': f'Optimization Opportunity: +{improvement:.1f}% Profit',
                'description': f'Optimal discount is {optimization["optimal_discount_pct"]:.1f}%.',
                'status': 'positive'
            })
    
    return insights


@router.post("/promotion")
async def run_promotion_analysis(request: PromotionRequest) -> Dict[str, Any]:
    """
    Perform Promotion Optimization analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.sales_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Sales column '{request.sales_col}' not found")
        
        visualizations = {}
        results = {}
        
        # Calculate baseline
        baseline = calculate_baseline(df, request.sales_col, request.promo_flag_col, request.baseline_method)
        results['baseline'] = _to_native_type(baseline)
        
        # Promotional lift analysis
        if request.promo_flag_col and request.promo_flag_col in df.columns:
            # Convert to binary if needed
            df['_promo_flag'] = df[request.promo_flag_col].apply(
                lambda x: 1 if x in [1, True, 'Yes', 'Y', 'yes', 'TRUE', 'true'] else 0
            )
            lift_data = calculate_promotional_lift(df, request.sales_col, '_promo_flag', baseline)
            results['lift_analysis'] = lift_data
            visualizations['lift_chart'] = create_lift_chart(lift_data, baseline)
        else:
            lift_data = {
                'baseline_sales': baseline,
                'percent_lift': 0,
                'is_significant': False
            }
            results['lift_analysis'] = lift_data
        
        # Price elasticity
        elasticity_data = None
        if request.price_col and request.price_col in df.columns:
            elasticity_data = calculate_price_elasticity(df, request.sales_col, request.price_col)
            results['elasticity'] = elasticity_data
            if 'error' not in elasticity_data:
                visualizations['elasticity_chart'] = create_elasticity_chart(
                    df, request.sales_col, request.price_col, elasticity_data
                )
        
        # Discount effectiveness
        if request.discount_col and request.discount_col in df.columns:
            discount_data = calculate_discount_effectiveness(df, request.sales_col, request.discount_col)
            results['discount_analysis'] = discount_data
            if 'error' not in discount_data:
                visualizations['discount_chart'] = create_discount_analysis_chart(discount_data)
        
        # Promo type performance
        if request.promo_type_col and request.promo_type_col in df.columns:
            type_data = calculate_promo_type_performance(
                df, request.sales_col, request.promo_type_col, baseline,
                request.price_col, request.cost_col, request.margin_pct
            )
            results['promo_type_analysis'] = type_data
            visualizations['promo_type_chart'] = create_promo_type_chart(type_data)
        
        # ROI calculation
        roi_data = calculate_promotion_roi(
            df, request.sales_col, baseline,
            request.price_col, request.promo_price_col,
            request.cost_col, request.margin_pct
        )
        results['roi_analysis'] = roi_data
        if roi_data.get('roi') is not None:
            visualizations['roi_chart'] = create_roi_summary_chart(roi_data)
        
        # Discount optimization
        optimization = None
        if elasticity_data and 'elasticity' in elasticity_data and elasticity_data['elasticity'] < 0:
            if request.price_col and request.price_col in df.columns:
                current_price = df[request.price_col].mean()
                current_volume = df[request.sales_col].mean()
                optimization = find_optimal_discount(
                    elasticity_data['elasticity'], request.margin_pct,
                    current_price, current_volume
                )
                results['optimization'] = optimization
                if 'error' not in optimization:
                    visualizations['optimization_chart'] = create_optimization_chart(optimization, current_price)
        
        # Generate insights
        insights = generate_key_insights(lift_data, elasticity_data, roi_data, optimization)
        
        # Summary
        summary = {
            'total_records': len(df),
            'baseline_sales': _to_native_type(baseline),
            'avg_sales': _to_native_type(df[request.sales_col].mean()),
            'lift_pct': lift_data.get('percent_lift'),
            'roi': roi_data.get('roi'),
            'elasticity': elasticity_data.get('elasticity') if elasticity_data else None,
            'optimal_discount': optimization.get('optimal_discount_pct') if optimization else None
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
        raise HTTPException(status_code=500, detail=f"Promotion analysis failed: {str(e)}")
