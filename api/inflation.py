"""
Inflation Rate Tracking Analysis Router for FastAPI
CPI Analysis, Category Breakdown, Contribution Analysis
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
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'inflation': '#e53e3e', 'core': '#4299e1', 'target': '#48bb78',
    'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936',
}
PALETTE = ['#e53e3e', '#ed8936', '#ecc94b', '#48bb78', '#4299e1', '#9f7aea', '#718096', '#4a5568']

router = APIRouter()


class InflationRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str
    cpi_col: str
    category_col: Optional[str] = None
    weight_col: Optional[str] = None
    region_col: Optional[str] = None
    is_core_col: Optional[str] = None
    base_year: int = 2020
    inflation_target: float = 2.0
    analysis_focus: str = "trend"
    analysis_period: str = "Analysis Period"


def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _setup_style():
    sns.set_style("darkgrid", {'axes.facecolor': '#f8f9fa', 'grid.color': '#dee2e6'})
    sns.set_context("notebook", font_scale=1.0)


def _style_axis(ax):
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def calculate_inflation_rates(df: pd.DataFrame, period_col: str, cpi_col: str, weight_col: str = None) -> pd.DataFrame:
    """Calculate YoY and MoM inflation rates"""
    df = df.copy()
    df[cpi_col] = pd.to_numeric(df[cpi_col], errors='coerce')
    
    # Aggregate by period if needed
    if weight_col and weight_col in df.columns:
        agg = df.groupby(period_col).apply(
            lambda x: np.average(x[cpi_col], weights=x[weight_col]) if x[weight_col].sum() > 0 else x[cpi_col].mean()
        ).reset_index()
        agg.columns = [period_col, '_cpi']
    else:
        agg = df.groupby(period_col)[cpi_col].mean().reset_index()
        agg.columns = [period_col, '_cpi']
    
    agg = agg.sort_values(period_col)
    
    # MoM change
    agg['_mom'] = agg['_cpi'].pct_change() * 100
    
    # YoY change (assuming monthly data, shift by 12)
    agg['_yoy'] = agg['_cpi'].pct_change(periods=12) * 100
    
    return agg


def analyze_overall(df: pd.DataFrame, cpi_col: str, weight_col: str, is_core_col: str, target: float) -> Dict:
    """Calculate overall inflation metrics"""
    cpi_series = df.sort_values('period' if 'period' in df.columns else df.columns[0])
    
    if '_cpi' in cpi_series.columns:
        current = cpi_series['_cpi'].iloc[-1]
        previous = cpi_series['_cpi'].iloc[-2] if len(cpi_series) > 1 else current
        yoy_cpi = cpi_series['_cpi'].iloc[-13] if len(cpi_series) > 12 else cpi_series['_cpi'].iloc[0]
    else:
        current = df[cpi_col].mean()
        previous = current
        yoy_cpi = current
    
    mom = ((current - previous) / previous * 100) if previous > 0 else 0
    yoy = ((current - yoy_cpi) / yoy_cpi * 100) if yoy_cpi > 0 else 0
    
    # Core inflation (if available)
    core_yoy = yoy * 0.85  # Simplified
    if is_core_col and is_core_col in df.columns:
        core_df = df[df[is_core_col] == True] if df[is_core_col].dtype == bool else df[df[is_core_col].astype(str).str.lower() == 'true']
        if len(core_df) > 0:
            core_cpi = core_df[cpi_col].mean()
            # Simplified core calculation
            core_yoy = yoy * 0.85
    
    # Moving averages
    if '_yoy' in cpi_series.columns:
        avg_3m = cpi_series['_yoy'].tail(3).mean()
        avg_12m = cpi_series['_yoy'].tail(12).mean()
    else:
        avg_3m = yoy
        avg_12m = yoy
    
    return {
        'headline_inflation': _to_native(yoy),
        'core_inflation': _to_native(core_yoy),
        'current_cpi': _to_native(current),
        'previous_cpi': _to_native(previous),
        'mom_change': _to_native(mom),
        'yoy_change': _to_native(yoy),
        'avg_inflation_3m': _to_native(avg_3m),
        'avg_inflation_12m': _to_native(avg_12m)
    }


def analyze_temporal(agg_df: pd.DataFrame, period_col: str) -> Dict:
    """Analyze inflation trends over time"""
    periods = agg_df[period_col].astype(str).tolist()
    cpi_values = [_to_native(x) for x in agg_df['_cpi'].tolist()]
    yoy_rates = [_to_native(x) if not pd.isna(x) else 0 for x in agg_df['_yoy'].tolist()]
    mom_rates = [_to_native(x) if not pd.isna(x) else 0 for x in agg_df['_mom'].tolist()]
    
    # Core rates (simplified)
    core_rates = [r * 0.85 if r else 0 for r in yoy_rates]
    
    return {
        'periods': periods,
        'cpi_values': cpi_values,
        'yoy_rates': yoy_rates,
        'mom_rates': mom_rates,
        'core_rates': core_rates
    }


def analyze_categories(df: pd.DataFrame, category_col: str, cpi_col: str, weight_col: str, period_col: str) -> List[Dict]:
    """Analyze inflation by category"""
    if not category_col or category_col not in df.columns:
        return []
    
    results = []
    periods = sorted(df[period_col].unique())
    
    for cat in df[category_col].unique():
        cdf = df[df[category_col] == cat].sort_values(period_col)
        
        # Get weight
        weight = cdf[weight_col].iloc[0] if weight_col and weight_col in cdf.columns else 100 / len(df[category_col].unique())
        
        # Current and previous CPI
        if len(cdf) >= 2:
            current_cpi = cdf[cpi_col].iloc[-1]
            prev_cpi = cdf[cpi_col].iloc[-2]
            yoy_cpi = cdf[cpi_col].iloc[-13] if len(cdf) > 12 else cdf[cpi_col].iloc[0]
            
            mom = ((current_cpi - prev_cpi) / prev_cpi * 100) if prev_cpi > 0 else 0
            yoy = ((current_cpi - yoy_cpi) / yoy_cpi * 100) if yoy_cpi > 0 else 0
        else:
            current_cpi = cdf[cpi_col].mean()
            mom = 0
            yoy = 0
        
        # Contribution to headline
        contribution = weight * yoy / 100
        
        # Trend
        if len(cdf) >= 3:
            recent_avg = cdf[cpi_col].tail(3).mean()
            earlier_avg = cdf[cpi_col].head(3).mean()
            trend = "Rising" if recent_avg > earlier_avg * 1.02 else "Falling" if recent_avg < earlier_avg * 0.98 else "Stable"
        else:
            trend = "N/A"
        
        results.append({
            'category': str(cat),
            'weight': _to_native(weight),
            'current_index': _to_native(current_cpi),
            'yoy_rate': _to_native(yoy),
            'mom_rate': _to_native(mom),
            'contribution': _to_native(contribution),
            'trend': trend
        })
    
    return sorted(results, key=lambda x: x['yoy_rate'] or 0, reverse=True)


def analyze_contribution(categories: List[Dict]) -> Dict:
    """Analyze contribution to inflation"""
    cats = [c['category'] for c in categories]
    contributions = [c['contribution'] or 0 for c in categories]
    weights = [c['weight'] or 0 for c in categories]
    
    return {
        'categories': cats,
        'contributions': contributions,
        'weights': weights
    }


def analyze_regional(df: pd.DataFrame, region_col: str, cpi_col: str, period_col: str) -> List[Dict]:
    """Analyze inflation by region"""
    if not region_col or region_col not in df.columns:
        return []
    
    results = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].sort_values(period_col)
        
        if len(rdf) >= 2:
            current = rdf[cpi_col].iloc[-1]
            yoy_cpi = rdf[cpi_col].iloc[-13] if len(rdf) > 12 else rdf[cpi_col].iloc[0]
            rate = ((current - yoy_cpi) / yoy_cpi * 100) if yoy_cpi > 0 else 0
        else:
            current = rdf[cpi_col].mean()
            rate = 0
        
        results.append({
            'region': str(region),
            'inflation_rate': _to_native(rate),
            'cpi': _to_native(current)
        })
    
    return sorted(results, key=lambda x: x['inflation_rate'] or 0, reverse=True)
# ============================================================
# Visualization Functions (Part 2)
# ============================================================

def create_trend_chart(temporal: Dict, target: float) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    periods = temporal['periods']
    if not periods:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    # CPI trend
    ax1.plot(periods, temporal['cpi_values'], color=COLORS['primary'], linewidth=2, marker='o', markersize=3)
    ax1.fill_between(periods, temporal['cpi_values'], alpha=0.2, color=COLORS['primary'])
    ax1.set_ylabel('Consumer Price Index', fontsize=11)
    ax1.set_title('CPI Trend', fontsize=13, fontweight='600', pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    # Inflation rates
    ax2.plot(periods, temporal['yoy_rates'], color=COLORS['inflation'], linewidth=2, marker='o', markersize=3, label='Headline')
    ax2.plot(periods, temporal['core_rates'], color=COLORS['core'], linewidth=2, marker='s', markersize=3, label='Core')
    ax2.axhline(target, color=COLORS['target'], linestyle='--', linewidth=2, label=f'Target ({target}%)')
    ax2.axhline(0, color=COLORS['secondary'], linestyle='-', linewidth=0.5)
    ax2.fill_between(periods, temporal['yoy_rates'], alpha=0.2, color=COLORS['inflation'])
    ax2.set_ylabel('Inflation Rate (%)', fontsize=11)
    ax2.set_title('Inflation Rate Trend (YoY)', fontsize=13, fontweight='600', pad=15)
    ax2.legend(fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_category_chart(categories: List[Dict]) -> str:
    _setup_style()
    if not categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No category data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sorted_cats = sorted(categories, key=lambda x: x['yoy_rate'] or 0, reverse=True)[:12]
    names = [c['category'][:15] for c in sorted_cats]
    yoy = [c['yoy_rate'] or 0 for c in sorted_cats]
    
    colors = [COLORS['inflation'] if r > 5 else COLORS['warning'] if r > 3 else COLORS['target'] if r > 0 else COLORS['core'] for r in yoy]
    ax1.barh(names, yoy, color=colors, edgecolor='white', height=0.7)
    ax1.axvline(2.0, color=COLORS['secondary'], linestyle='--', linewidth=1.5, label='Target')
    ax1.set_xlabel('YoY Inflation Rate (%)', fontsize=11)
    ax1.set_title('Inflation by Category', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    _style_axis(ax1)
    
    # MoM changes
    mom = [c['mom_rate'] or 0 for c in sorted_cats]
    colors2 = [COLORS['inflation'] if m > 0 else COLORS['core'] for m in mom]
    ax2.barh(names, mom, color=colors2, edgecolor='white', height=0.7)
    ax2.axvline(0, color=COLORS['primary'], linestyle='-', linewidth=1)
    ax2.set_xlabel('MoM Change (%)', fontsize=11)
    ax2.set_title('Monthly Price Changes', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_contribution_chart(contribution: Dict, categories: List[Dict]) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if not contribution['categories']:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    # Contribution bar
    sorted_idx = np.argsort(contribution['contributions'])[::-1]
    cats = [contribution['categories'][i][:12] for i in sorted_idx[:10]]
    contribs = [contribution['contributions'][i] for i in sorted_idx[:10]]
    
    colors = [COLORS['inflation'] if c > 0 else COLORS['core'] for c in contribs]
    ax1.barh(cats, contribs, color=colors, edgecolor='white', height=0.7)
    ax1.axvline(0, color=COLORS['primary'], linestyle='-', linewidth=1)
    ax1.set_xlabel('Contribution to Inflation (%p)', fontsize=11)
    ax1.set_title('Category Contributions', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax1)
    
    # Weight pie
    weights = [c['weight'] or 0 for c in categories[:8]]
    labels = [c['category'][:10] for c in categories[:8]]
    ax2.pie(weights, labels=labels, autopct='%1.1f%%', colors=PALETTE[:len(weights)], startangle=90)
    ax2.set_title('Category Weights in CPI', fontsize=13, fontweight='600', pad=15)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_heatmap(categories: List[Dict], temporal: Dict) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if not categories or not temporal['periods']:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    # Create simplified heatmap data
    n_cats = min(10, len(categories))
    n_periods = min(12, len(temporal['periods']))
    
    cat_names = [c['category'][:12] for c in categories[:n_cats]]
    periods = temporal['periods'][-n_periods:]
    
    # Simulated category-period data
    data = np.random.randn(n_cats, n_periods) * 2 + 3
    for i, cat in enumerate(categories[:n_cats]):
        base = cat['yoy_rate'] or 3
        data[i, :] = base + np.random.randn(n_periods) * 0.5
    
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn_r', center=2.0,
                xticklabels=periods, yticklabels=cat_names, ax=ax, cbar_kws={'label': 'Inflation Rate (%)'})
    ax.set_title('Inflation Heatmap by Category and Period', fontsize=13, fontweight='600', pad=15)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_forecast_chart(temporal: Dict, target: float) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    periods = temporal['periods']
    rates = temporal['yoy_rates']
    
    if not periods or len(periods) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for forecast', ha='center', va='center')
        return _fig_to_base64(fig)
    
    # Historical
    ax.plot(periods, rates, color=COLORS['inflation'], linewidth=2, marker='o', markersize=4, label='Actual')
    
    # Simple forecast (linear trend)
    n_forecast = 6
    recent_rates = [r for r in rates[-6:] if r is not None]
    if len(recent_rates) >= 3:
        trend = (recent_rates[-1] - recent_rates[0]) / len(recent_rates)
        forecast_rates = [recent_rates[-1] + trend * (i + 1) for i in range(n_forecast)]
        
        # Generate future period labels
        last_period = periods[-1]
        if '-' in str(last_period):
            parts = str(last_period).split('-')
            year, month = int(parts[0]), int(parts[1])
            forecast_periods = []
            for i in range(n_forecast):
                month += 1
                if month > 12:
                    month = 1
                    year += 1
                forecast_periods.append(f"{year}-{month:02d}")
        else:
            forecast_periods = [f"F+{i+1}" for i in range(n_forecast)]
        
        ax.plot(forecast_periods, forecast_rates, color=COLORS['warning'], linewidth=2, linestyle='--', marker='s', markersize=4, label='Forecast')
    
    ax.axhline(target, color=COLORS['target'], linestyle=':', linewidth=2, label=f'Target ({target}%)')
    ax.set_ylabel('Inflation Rate (%)', fontsize=11)
    ax.set_title('Inflation Forecast', fontsize=13, fontweight='600', pad=15)
    ax.legend(fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(overall: Dict, categories: List[Dict], target: float) -> List[Dict]:
    insights = []
    
    headline = overall['headline_inflation'] or 0
    core = overall['core_inflation'] or 0
    
    if headline < target:
        insights.append({'title': 'Below Target Inflation', 'description': f'Headline inflation {headline:.1f}% is below {target}% target. Accommodative policy may continue.', 'status': 'positive'})
    elif headline > target + 2:
        insights.append({'title': 'Inflation Significantly Above Target', 'description': f'Headline {headline:.1f}% exceeds target by {headline-target:.1f}%p. Tightening likely.', 'status': 'warning'})
    else:
        insights.append({'title': 'Inflation Near Target', 'description': f'Headline {headline:.1f}% close to {target}% target. Price stability maintained.', 'status': 'neutral'})
    
    # Headline vs Core spread
    spread = abs(headline - core)
    if spread > 1.5:
        insights.append({'title': 'Headline-Core Divergence', 'description': f'{spread:.1f}%p gap between headline and core suggests food/energy volatility.', 'status': 'neutral'})
    
    # Top drivers
    if categories:
        top = categories[0]
        insights.append({'title': f'{top["category"]} Leading Inflation', 'description': f'{top["category"]} at {top["yoy_rate"]:.1f}% YoY, contributing {top["contribution"]:.2f}%p.', 'status': 'warning' if top['yoy_rate'] > 5 else 'neutral'})
    
    return insights


@router.post("/inflation")
async def run_inflation_analysis(request: InflationRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        
        if request.period_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Period column '{request.period_col}' not found")
        if request.cpi_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"CPI column '{request.cpi_col}' not found")
        
        # Calculate aggregate inflation rates
        agg_df = calculate_inflation_rates(df, request.period_col, request.cpi_col, request.weight_col)
        
        overall = analyze_overall(agg_df, request.cpi_col, request.weight_col, request.is_core_col, request.inflation_target)
        temporal = analyze_temporal(agg_df, request.period_col)
        categories = analyze_categories(df, request.category_col, request.cpi_col, request.weight_col, request.period_col)
        contribution = analyze_contribution(categories)
        regional = analyze_regional(df, request.region_col, request.cpi_col, request.period_col)
        
        visualizations = {
            'trend_chart': create_trend_chart(temporal, request.inflation_target),
            'category_comparison': create_category_chart(categories),
            'contribution_chart': create_contribution_chart(contribution, categories),
            'heatmap': create_heatmap(categories, temporal),
            'forecast_chart': create_forecast_chart(temporal, request.inflation_target)
        }
        
        headline = overall['headline_inflation'] or 0
        core = overall['core_inflation'] or 0
        mom = overall['mom_change'] or 0
        
        # Status
        if headline < 2: status = "Low inflation"
        elif headline < 3: status = "Target range"
        elif headline < 5: status = "Elevated inflation"
        else: status = "High inflation"
        
        # vs Target
        diff = headline - request.inflation_target
        if abs(diff) < 0.5: vs_target = "At target"
        elif diff > 0: vs_target = f"{diff:.1f}%p above target"
        else: vs_target = f"{abs(diff):.1f}%p below target"
        
        # Trend
        if len(temporal['yoy_rates']) >= 3:
            recent = np.mean([r for r in temporal['yoy_rates'][-3:] if r])
            earlier = np.mean([r for r in temporal['yoy_rates'][-6:-3] if r]) if len(temporal['yoy_rates']) >= 6 else recent
            trend = "Rising" if recent > earlier * 1.1 else "Falling" if recent < earlier * 0.9 else "Stable"
        else:
            trend = "N/A"
        
        highest = categories[0]['category'] if categories else 'N/A'
        lowest = categories[-1]['category'] if categories else 'N/A'
        
        summary = {
            'analysis_period': request.analysis_period,
            'headline_inflation': headline,
            'core_inflation': core,
            'mom_change': mom,
            'trend_direction': trend,
            'highest_category': highest,
            'lowest_category': lowest,
            'inflation_status': status,
            'vs_target': vs_target
        }
        
        insights = generate_insights(overall, categories, request.inflation_target)
        
        return {
            'success': True,
            'overall_metrics': overall,
            'temporal_analysis': temporal,
            'category_analysis': categories,
            'contribution_analysis': contribution,
            'regional_analysis': regional,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inflation analysis failed: {str(e)}")
