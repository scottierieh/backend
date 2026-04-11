"""
Cost-Benefit Analysis (CBA) Router for FastAPI
Evaluate projects/investments by comparing costs and benefits over time
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
from scipy.optimize import brentq
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CostBenefitRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str  # Time period column (year, month, etc.)
    cost_cols: List[str]  # Cost columns
    benefit_cols: List[str]  # Benefit columns
    discount_rate: float = 0.1  # Default 10%
    project_name: Optional[str] = "Project"


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


def calculate_npv(cash_flows: np.ndarray, discount_rate: float) -> float:
    """Calculate Net Present Value"""
    periods = np.arange(len(cash_flows))
    discount_factors = 1 / (1 + discount_rate) ** periods
    return np.sum(cash_flows * discount_factors)


def calculate_irr(cash_flows: np.ndarray) -> Optional[float]:
    """Calculate Internal Rate of Return"""
    try:
        # IRR is the rate where NPV = 0
        def npv_at_rate(rate):
            if rate <= -1:
                return float('inf')
            periods = np.arange(len(cash_flows))
            return np.sum(cash_flows / (1 + rate) ** periods)
        
        # Try to find IRR between -99% and 1000%
        try:
            irr = brentq(npv_at_rate, -0.99, 10.0)
            return irr
        except ValueError:
            # No sign change found, try different range
            return None
    except Exception:
        return None


def calculate_payback_period(cash_flows: np.ndarray) -> Optional[float]:
    """Calculate Payback Period (simple, undiscounted)"""
    cumulative = np.cumsum(cash_flows)
    
    # Find where cumulative becomes positive
    positive_indices = np.where(cumulative >= 0)[0]
    
    if len(positive_indices) == 0:
        return None  # Never pays back
    
    first_positive = positive_indices[0]
    
    if first_positive == 0:
        return 0.0
    
    # Linear interpolation for exact period
    prev_cumulative = cumulative[first_positive - 1]
    curr_cumulative = cumulative[first_positive]
    
    fraction = -prev_cumulative / (curr_cumulative - prev_cumulative)
    
    return first_positive - 1 + fraction


def calculate_discounted_payback(cash_flows: np.ndarray, discount_rate: float) -> Optional[float]:
    """Calculate Discounted Payback Period"""
    periods = np.arange(len(cash_flows))
    discounted_flows = cash_flows / (1 + discount_rate) ** periods
    cumulative = np.cumsum(discounted_flows)
    
    positive_indices = np.where(cumulative >= 0)[0]
    
    if len(positive_indices) == 0:
        return None
    
    first_positive = positive_indices[0]
    
    if first_positive == 0:
        return 0.0
    
    prev_cumulative = cumulative[first_positive - 1]
    curr_cumulative = cumulative[first_positive]
    
    fraction = -prev_cumulative / (curr_cumulative - prev_cumulative)
    
    return first_positive - 1 + fraction


def calculate_bcr(benefits: np.ndarray, costs: np.ndarray, discount_rate: float) -> float:
    """Calculate Benefit-Cost Ratio"""
    periods = np.arange(len(benefits))
    discount_factors = 1 / (1 + discount_rate) ** periods
    
    pv_benefits = np.sum(benefits * discount_factors)
    pv_costs = np.sum(costs * discount_factors)
    
    if pv_costs == 0:
        return float('inf') if pv_benefits > 0 else 0
    
    return pv_benefits / pv_costs


def calculate_roi(total_benefits: float, total_costs: float) -> float:
    """Calculate Return on Investment"""
    if total_costs == 0:
        return float('inf') if total_benefits > 0 else 0
    return (total_benefits - total_costs) / total_costs * 100


def sensitivity_analysis(cash_flows: np.ndarray, base_discount_rate: float) -> Dict[str, Any]:
    """Perform sensitivity analysis on discount rate"""
    rates = np.arange(0.0, 0.31, 0.02)  # 0% to 30%
    npvs = []
    
    for rate in rates:
        npv = calculate_npv(cash_flows, rate)
        npvs.append(_to_native_type(npv))
    
    return {
        'discount_rates': [_to_native_type(r * 100) for r in rates],
        'npvs': npvs
    }


def generate_cash_flow_chart(periods: List, costs: np.ndarray, benefits: np.ndarray, 
                              net_flows: np.ndarray, project_name: str) -> str:
    """Generate cash flow bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(periods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, -costs, width, label='Costs', color='#ef4444', alpha=0.8)
    bars2 = ax.bar(x + width/2, benefits, width, label='Benefits', color='#22c55e', alpha=0.8)
    
    # Add net flow line
    ax.plot(x, net_flows, 'ko-', linewidth=2, markersize=8, label='Net Cash Flow')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Amount', fontsize=12)
    ax.set_title(f'{project_name} - Cash Flow Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if abs(height) > 0:
            ax.annotate(f'{abs(height):,.0f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, -10 if height < 0 else 5),
                       textcoords="offset points",
                       ha='center', va='top' if height < 0 else 'bottom',
                       fontsize=8, color='darkred')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_cumulative_chart(periods: List, cumulative_flows: np.ndarray, 
                               cumulative_discounted: np.ndarray, project_name: str) -> str:
    """Generate cumulative cash flow chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(periods))
    
    ax.fill_between(x, cumulative_flows, where=(cumulative_flows >= 0), 
                    color='#22c55e', alpha=0.3, label='Positive (Undiscounted)')
    ax.fill_between(x, cumulative_flows, where=(cumulative_flows < 0), 
                    color='#ef4444', alpha=0.3, label='Negative (Undiscounted)')
    
    ax.plot(x, cumulative_flows, 'b-', linewidth=2, marker='o', label='Cumulative (Undiscounted)')
    ax.plot(x, cumulative_discounted, 'g--', linewidth=2, marker='s', label='Cumulative (Discounted)')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    # Find payback point
    for i, val in enumerate(cumulative_flows):
        if val >= 0 and (i == 0 or cumulative_flows[i-1] < 0):
            ax.axvline(x=i, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax.annotate('Payback', xy=(i, 0), xytext=(i+0.5, max(cumulative_flows)*0.1),
                       fontsize=10, color='orange')
            break
    
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Cumulative Amount', fontsize=12)
    ax.set_title(f'{project_name} - Cumulative Cash Flow', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_sensitivity_chart(sensitivity_data: Dict, base_rate: float, project_name: str) -> str:
    """Generate NPV sensitivity to discount rate chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rates = sensitivity_data['discount_rates']
    npvs = sensitivity_data['npvs']
    
    ax.plot(rates, npvs, 'b-', linewidth=2, marker='o', markersize=6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='NPV = 0')
    ax.axvline(x=base_rate * 100, color='green', linestyle=':', linewidth=2, 
               label=f'Base Rate ({base_rate*100:.0f}%)')
    
    # Find and mark IRR (where NPV crosses zero)
    for i in range(len(npvs) - 1):
        if npvs[i] is not None and npvs[i+1] is not None:
            if (npvs[i] > 0 and npvs[i+1] < 0) or (npvs[i] < 0 and npvs[i+1] > 0):
                irr_approx = rates[i] + (rates[i+1] - rates[i]) * npvs[i] / (npvs[i] - npvs[i+1])
                ax.plot(irr_approx, 0, 'r*', markersize=15)
                ax.annotate(f'IRR ≈ {irr_approx:.1f}%', xy=(irr_approx, 0),
                           xytext=(irr_approx + 2, max(filter(None, npvs)) * 0.1),
                           fontsize=10, color='red')
                break
    
    ax.fill_between(rates, npvs, where=[n > 0 if n is not None else False for n in npvs], 
                    color='green', alpha=0.2)
    ax.fill_between(rates, npvs, where=[n < 0 if n is not None else False for n in npvs], 
                    color='red', alpha=0.2)
    
    ax.set_xlabel('Discount Rate (%)', fontsize=12)
    ax.set_ylabel('Net Present Value', fontsize=12)
    ax.set_title(f'{project_name} - NPV Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_breakdown_pie(total_costs: float, total_benefits: float, project_name: str) -> str:
    """Generate cost vs benefit pie chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall comparison
    values = [total_costs, total_benefits]
    labels = [f'Total Costs\n${total_costs:,.0f}', f'Total Benefits\n${total_benefits:,.0f}']
    colors = ['#ef4444', '#22c55e']
    explode = (0.02, 0.02)
    
    axes[0].pie(values, labels=labels, colors=colors, explode=explode,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    axes[0].set_title('Cost vs Benefit Distribution', fontsize=14, fontweight='bold')
    
    # Net benefit visualization
    net = total_benefits - total_costs
    if net >= 0:
        axes[1].barh(['Investment', 'Return', 'Net Gain'], 
                     [total_costs, total_benefits, net],
                     color=['#ef4444', '#22c55e', '#3b82f6'])
    else:
        axes[1].barh(['Investment', 'Return', 'Net Loss'], 
                     [total_costs, total_benefits, abs(net)],
                     color=['#ef4444', '#22c55e', '#f97316'])
    
    axes[1].set_xlabel('Amount', fontsize=12)
    axes[1].set_title('Investment Summary', fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate([total_costs, total_benefits, abs(net)]):
        axes[1].text(v + max(total_costs, total_benefits) * 0.02, i, 
                     f'${v:,.0f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(metrics: Dict, project_name: str) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    key_insights = []
    
    # NPV interpretation
    npv = metrics.get('npv', 0)
    if npv > 0:
        key_insights.append({
            'title': 'NPV Positive',
            'description': f'NPV = ${npv:,.2f}. The project adds value and should be considered.',
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'NPV Negative',
            'description': f'NPV = ${npv:,.2f}. The project destroys value at the given discount rate.',
            'status': 'negative'
        })
    
    # BCR interpretation
    bcr = metrics.get('bcr', 0)
    if bcr > 1:
        key_insights.append({
            'title': 'BCR > 1',
            'description': f'BCR = {bcr:.2f}. Benefits exceed costs - project is economically viable.',
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'BCR < 1',
            'description': f'BCR = {bcr:.2f}. Costs exceed benefits - reconsider the project.',
            'status': 'negative'
        })
    
    # IRR interpretation
    irr = metrics.get('irr')
    discount_rate = metrics.get('discount_rate', 0.1)
    if irr is not None:
        if irr > discount_rate:
            key_insights.append({
                'title': 'IRR Exceeds Hurdle Rate',
                'description': f'IRR = {irr*100:.1f}% > {discount_rate*100:.0f}% discount rate. Good investment.',
                'status': 'positive'
            })
        else:
            key_insights.append({
                'title': 'IRR Below Hurdle Rate',
                'description': f'IRR = {irr*100:.1f}% < {discount_rate*100:.0f}% discount rate. Consider alternatives.',
                'status': 'negative'
            })
    
    # Payback interpretation
    payback = metrics.get('payback_period')
    if payback is not None:
        key_insights.append({
            'title': 'Payback Period',
            'description': f'Investment recovers in {payback:.1f} periods.',
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'No Payback',
            'description': 'Investment does not recover within the analysis period.',
            'status': 'negative'
        })
    
    # Overall recommendation
    positive_count = sum(1 for i in key_insights if i['status'] == 'positive')
    if positive_count >= 3:
        recommendation = 'PROCEED - Strong economic case for the project.'
    elif positive_count >= 2:
        recommendation = 'CONSIDER - Project shows promise but review risks.'
    else:
        recommendation = 'RECONSIDER - Economic case is weak.'
    
    return {
        'key_insights': key_insights,
        'recommendation': recommendation,
        'decision': 'Accept' if npv > 0 and bcr > 1 else 'Reject'
    }


@router.post("/cost-benefit")
async def run_cost_benefit_analysis(request: CostBenefitRequest) -> Dict[str, Any]:
    """
    Perform Cost-Benefit Analysis (CBA).
    
    Calculates NPV, IRR, BCR, Payback Period and provides visualizations
    for investment/project evaluation.
    """
    try:
        data = request.data
        period_col = request.period_col
        cost_cols = request.cost_cols
        benefit_cols = request.benefit_cols
        discount_rate = request.discount_rate
        project_name = request.project_name or "Project"
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        if not cost_cols and not benefit_cols:
            raise HTTPException(status_code=400, detail="At least one cost or benefit column required.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        all_cols = [period_col] + cost_cols + benefit_cols
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_cols)}")
        
        # Sort by period
        df = df.sort_values(period_col).reset_index(drop=True)
        periods = df[period_col].tolist()
        
        # Calculate total costs and benefits per period
        costs = np.zeros(len(df))
        benefits = np.zeros(len(df))
        
        for col in cost_cols:
            costs += pd.to_numeric(df[col], errors='coerce').fillna(0).values
        
        for col in benefit_cols:
            benefits += pd.to_numeric(df[col], errors='coerce').fillna(0).values
        
        # Net cash flows (benefits - costs)
        net_flows = benefits - costs
        
        # Calculate metrics
        npv = calculate_npv(net_flows, discount_rate)
        irr = calculate_irr(net_flows)
        bcr = calculate_bcr(benefits, costs, discount_rate)
        payback = calculate_payback_period(net_flows)
        discounted_payback = calculate_discounted_payback(net_flows, discount_rate)
        
        total_costs = float(np.sum(costs))
        total_benefits = float(np.sum(benefits))
        roi = calculate_roi(total_benefits, total_costs)
        
        # Present values
        n_periods = len(df)
        discount_factors = 1 / (1 + discount_rate) ** np.arange(n_periods)
        pv_costs = float(np.sum(costs * discount_factors))
        pv_benefits = float(np.sum(benefits * discount_factors))
        
        # Cumulative flows
        cumulative_flows = np.cumsum(net_flows)
        cumulative_discounted = np.cumsum(net_flows * discount_factors)
        
        # Sensitivity analysis
        sensitivity_data = sensitivity_analysis(net_flows, discount_rate)
        
        # Generate visualizations
        cash_flow_chart = generate_cash_flow_chart(periods, costs, benefits, net_flows, project_name)
        cumulative_chart = generate_cumulative_chart(periods, cumulative_flows, cumulative_discounted, project_name)
        sensitivity_chart = generate_sensitivity_chart(sensitivity_data, discount_rate, project_name)
        breakdown_chart = generate_breakdown_pie(total_costs, total_benefits, project_name)
        
        # Prepare period-by-period breakdown
        period_breakdown = []
        for i in range(len(df)):
            period_breakdown.append({
                'period': periods[i],
                'costs': _to_native_type(costs[i]),
                'benefits': _to_native_type(benefits[i]),
                'net_flow': _to_native_type(net_flows[i]),
                'cumulative': _to_native_type(cumulative_flows[i]),
                'discount_factor': _to_native_type(discount_factors[i]),
                'pv_net_flow': _to_native_type(net_flows[i] * discount_factors[i])
            })
        
        # Metrics summary
        metrics = {
            'npv': _to_native_type(npv),
            'irr': _to_native_type(irr) if irr is not None else None,
            'bcr': _to_native_type(bcr),
            'roi': _to_native_type(roi),
            'payback_period': _to_native_type(payback) if payback is not None else None,
            'discounted_payback': _to_native_type(discounted_payback) if discounted_payback is not None else None,
            'total_costs': _to_native_type(total_costs),
            'total_benefits': _to_native_type(total_benefits),
            'net_benefit': _to_native_type(total_benefits - total_costs),
            'pv_costs': _to_native_type(pv_costs),
            'pv_benefits': _to_native_type(pv_benefits),
            'discount_rate': _to_native_type(discount_rate)
        }
        
        # Generate interpretation
        interpretation = generate_interpretation(metrics, project_name)
        
        return {
            'project_name': project_name,
            'metrics': metrics,
            'period_breakdown': period_breakdown,
            'sensitivity_analysis': sensitivity_data,
            'cash_flow_chart': cash_flow_chart,
            'cumulative_chart': cumulative_chart,
            'sensitivity_chart': sensitivity_chart,
            'breakdown_chart': breakdown_chart,
            'interpretation': interpretation,
            'n_periods': n_periods
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost-Benefit analysis failed: {str(e)}")
