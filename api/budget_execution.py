"""
Regional Budget Execution Comparison Router for FastAPI
Execution Rate Analysis, Regional Rankings, Disparity Analysis
Darkgrid style with clean, professional colors
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
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5

COLORS = {
    'primary': '#4a5568',
    'secondary': '#718096',
    'accent': '#2d3748',
    'positive': '#48bb78',
    'warning': '#ed8936',
    'danger': '#e53e3e',
    'excellent': '#38a169',
    'good': '#4299e1',
    'fair': '#ecc94b',
    'poor': '#e53e3e',
}

REGION_COLORS = ['#4a5568', '#718096', '#a0aec0', '#2d3748', '#4a5568', '#718096', '#a0aec0', '#2d3748',
                 '#4a5568', '#718096', '#a0aec0', '#2d3748', '#4a5568', '#718096', '#a0aec0', '#2d3748', '#4a5568']

router = APIRouter()


class BudgetExecutionRequest(BaseModel):
    data: List[Dict[str, Any]]
    region_col: str
    allocated_col: str
    executed_col: str
    period_col: Optional[str] = None
    category_col: Optional[str] = None
    population_col: Optional[str] = None
    analysis_type: str = "execution_rate"
    benchmark_type: str = "national_average"
    target_rate: Optional[float] = None
    fiscal_period: str = "Current Period"


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
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def _setup_style():
    sns.set_style("darkgrid", {
        'axes.facecolor': '#f8f9fa',
        'grid.color': '#dee2e6',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'axes.edgecolor': '#adb5bd',
    })
    sns.set_context("notebook", font_scale=1.0)


def _style_axis(ax):
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def analyze_regional_execution(df: pd.DataFrame, region_col: str, allocated_col: str, 
                                executed_col: str, population_col: Optional[str] = None) -> Dict[str, Any]:
    """Analyze budget execution by region"""
    regional_data = df.groupby(region_col).agg({
        allocated_col: 'sum',
        executed_col: 'sum'
    }).reset_index()
    
    if population_col and population_col in df.columns:
        pop_data = df.groupby(region_col)[population_col].first().reset_index()
        regional_data = regional_data.merge(pop_data, on=region_col)
    
    regional_data['execution_rate'] = regional_data[executed_col] / regional_data[allocated_col]
    regional_data = regional_data.sort_values('execution_rate', ascending=False)
    regional_data['rank'] = range(1, len(regional_data) + 1)
    
    regions = []
    for _, row in regional_data.iterrows():
        region_dict = {
            'region': row[region_col],
            'allocated_budget': _to_native_type(row[allocated_col]),
            'executed_amount': _to_native_type(row[executed_col]),
            'execution_rate': _to_native_type(row['execution_rate']),
            'population': _to_native_type(row[population_col]) if population_col and population_col in row else None,
            'per_capita_spending': _to_native_type(row[executed_col] / row[population_col]) if population_col and population_col in row and row[population_col] > 0 else None,
            'yoy_change': None,
            'rank': int(row['rank']),
            'category_breakdown': {}
        }
        regions.append(region_dict)
    
    total_allocated = regional_data[allocated_col].sum()
    total_executed = regional_data[executed_col].sum()
    overall_rate = total_executed / total_allocated if total_allocated > 0 else 0
    
    rates = regional_data['execution_rate'].values
    disparity_index = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0
    
    return {
        'regions': regions,
        'total_allocated': _to_native_type(total_allocated),
        'total_executed': _to_native_type(total_executed),
        'overall_execution_rate': _to_native_type(overall_rate),
        'disparity_index': _to_native_type(disparity_index)
    }


def analyze_by_category(df: pd.DataFrame, region_col: str, category_col: str,
                        allocated_col: str, executed_col: str) -> List[Dict[str, Any]]:
    """Analyze execution by budget category"""
    if category_col not in df.columns:
        return []
    
    category_data = df.groupby(category_col).agg({
        allocated_col: 'sum',
        executed_col: 'sum'
    }).reset_index()
    
    category_data['execution_rate'] = category_data[executed_col] / category_data[allocated_col]
    
    results = []
    for _, row in category_data.iterrows():
        cat_df = df[df[category_col] == row[category_col]]
        region_rates = cat_df.groupby(region_col).apply(
            lambda x: x[executed_col].sum() / x[allocated_col].sum() if x[allocated_col].sum() > 0 else 0
        )
        
        results.append({
            'category': row[category_col],
            'total_allocated': _to_native_type(row[allocated_col]),
            'total_executed': _to_native_type(row[executed_col]),
            'execution_rate': _to_native_type(row['execution_rate']),
            'top_region': region_rates.idxmax() if len(region_rates) > 0 else 'N/A',
            'bottom_region': region_rates.idxmin() if len(region_rates) > 0 else 'N/A'
        })
    
    return results


def analyze_temporal(df: pd.DataFrame, region_col: str, period_col: str,
                     allocated_col: str, executed_col: str) -> Dict[str, Any]:
    """Analyze execution trends over time"""
    if period_col not in df.columns:
        return {'periods': [], 'execution_rates': {}, 'overall_trend': []}
    
    periods = sorted(df[period_col].unique())
    
    execution_rates = {}
    overall_trend = []
    
    for period in periods:
        period_df = df[df[period_col] == period]
        overall = period_df[executed_col].sum() / period_df[allocated_col].sum() if period_df[allocated_col].sum() > 0 else 0
        overall_trend.append(_to_native_type(overall))
        
        for region in df[region_col].unique():
            if region not in execution_rates:
                execution_rates[region] = []
            region_df = period_df[period_df[region_col] == region]
            rate = region_df[executed_col].sum() / region_df[allocated_col].sum() if region_df[allocated_col].sum() > 0 else 0
            execution_rates[region].append(_to_native_type(rate))
    
    return {
        'periods': [str(p) for p in periods],
        'execution_rates': execution_rates,
        'overall_trend': overall_trend
    }


def calculate_benchmark(regions: List[Dict], benchmark_type: str, target_rate: Optional[float] = None) -> Dict[str, Any]:
    """Calculate benchmark statistics"""
    rates = [r['execution_rate'] for r in regions if r['execution_rate'] is not None]
    
    avg_rate = np.mean(rates) if rates else 0
    median_rate = np.median(rates) if rates else 0
    std_dev = np.std(rates) if rates else 0
    
    benchmark = avg_rate if benchmark_type == 'national_average' else (max(rates) if benchmark_type == 'top_performer' else (target_rate or 0.85))
    
    above_avg = [r['region'] for r in regions if r['execution_rate'] and r['execution_rate'] >= benchmark]
    below_avg = [r['region'] for r in regions if r['execution_rate'] and r['execution_rate'] < benchmark]
    
    return {
        'average_execution_rate': _to_native_type(avg_rate),
        'median_execution_rate': _to_native_type(median_rate),
        'std_deviation': _to_native_type(std_dev),
        'above_average_regions': above_avg,
        'below_average_regions': below_avg
    }
# ============================================================
# Visualization Functions (Part 2)
# ============================================================

def create_regional_comparison_chart(regions: List[Dict]) -> str:
    """Create regional execution rate comparison chart"""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sorted_regions = sorted(regions, key=lambda x: x['execution_rate'] or 0, reverse=True)[:15]
    names = [r['region'] for r in sorted_regions]
    rates = [(r['execution_rate'] or 0) * 100 for r in sorted_regions]
    
    colors = [COLORS['excellent'] if r >= 90 else COLORS['good'] if r >= 80 else COLORS['fair'] if r >= 70 else COLORS['poor'] for r in rates]
    
    bars = ax1.barh(names, rates, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax1.axvline(x=85, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Target (85%)')
    ax1.set_xlabel('Execution Rate (%)', fontsize=11, color='#333333', labelpad=10)
    ax1.set_title('Regional Execution Rates', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.set_xlim(0, 105)
    _style_axis(ax1)
    
    for bar, val in zip(bars, rates):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=8, color='#333333')
    
    executed = [r['executed_amount'] / 1e9 for r in sorted_regions]
    ax2.barh(names, executed, color=COLORS['primary'], edgecolor='white', linewidth=0.5, height=0.7)
    ax2.set_xlabel('Executed Amount (₩ Billion)', fontsize=11, color='#333333', labelpad=10)
    ax2.set_title('Budget Executed by Region', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_execution_heatmap(df: pd.DataFrame, region_col: str, period_col: str,
                             allocated_col: str, executed_col: str) -> str:
    """Create execution rate heatmap by region and period"""
    _setup_style()
    
    if period_col not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No period data available', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    pivot = df.pivot_table(
        values=executed_col, 
        index=region_col, 
        columns=period_col,
        aggfunc='sum'
    )
    allocated_pivot = df.pivot_table(
        values=allocated_col,
        index=region_col,
        columns=period_col,
        aggfunc='sum'
    )
    rate_pivot = (pivot / allocated_pivot * 100).fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(rate_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=80,
                ax=ax, cbar_kws={'label': 'Execution Rate (%)'}, linewidths=0.5)
    ax.set_title('Execution Rate Heatmap by Region and Period', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Region', fontsize=11)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_category_breakdown_chart(category_analysis: List[Dict]) -> str:
    """Create category breakdown chart"""
    _setup_style()
    
    if not category_analysis:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No category data available', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = [c['category'] for c in category_analysis]
    rates = [c['execution_rate'] * 100 for c in category_analysis]
    allocated = [c['total_allocated'] / 1e9 for c in category_analysis]
    
    colors = [COLORS['excellent'] if r >= 90 else COLORS['good'] if r >= 80 else COLORS['fair'] if r >= 70 else COLORS['poor'] for r in rates]
    
    bars = ax1.bar(categories, rates, color=colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=85, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Target')
    ax1.set_ylabel('Execution Rate (%)', fontsize=11, color='#333333')
    ax1.set_title('Execution Rate by Category', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    ax1.legend(fontsize=9)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    ax2.pie(allocated, labels=categories, autopct='%1.1f%%', colors=REGION_COLORS[:len(categories)],
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax2.set_title('Budget Allocation by Category', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_temporal_trend_chart(temporal_analysis: Dict) -> str:
    """Create temporal trend chart"""
    _setup_style()
    
    periods = temporal_analysis['periods']
    overall = temporal_analysis['overall_trend']
    
    if not periods:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No temporal data available', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    overall_pct = [r * 100 for r in overall]
    ax.plot(periods, overall_pct, color=COLORS['primary'], linewidth=3, marker='o', markersize=8,
            markerfacecolor='white', markeredgewidth=2, label='Overall', zorder=5)
    
    region_rates = temporal_analysis['execution_rates']
    for i, (region, rates) in enumerate(list(region_rates.items())[:5]):
        rates_pct = [r * 100 for r in rates]
        ax.plot(periods, rates_pct, color=REGION_COLORS[i], linewidth=1, alpha=0.5, label=region)
    
    ax.axhline(y=85, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Target (85%)')
    ax.set_xlabel('Period', fontsize=11, color='#333333', labelpad=10)
    ax.set_ylabel('Execution Rate (%)', fontsize=11, color='#333333', labelpad=10)
    ax.set_title('Execution Rate Trends Over Time', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_disparity_chart(regions: List[Dict], benchmark: Dict) -> str:
    """Create disparity analysis chart"""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rates = [(r['execution_rate'] or 0) * 100 for r in regions]
    avg = benchmark['average_execution_rate'] * 100
    
    ax1.hist(rates, bins=15, color=COLORS['primary'], edgecolor='white', linewidth=0.5, alpha=0.7)
    ax1.axvline(x=avg, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Average: {avg:.1f}%')
    ax1.set_xlabel('Execution Rate (%)', fontsize=11, color='#333333')
    ax1.set_ylabel('Number of Regions', fontsize=11, color='#333333')
    ax1.set_title('Distribution of Execution Rates', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    ax1.legend(fontsize=9)
    _style_axis(ax1)
    
    deviations = [r - avg for r in rates]
    names = [r['region'] for r in regions]
    sorted_data = sorted(zip(names, deviations), key=lambda x: x[1])
    names_sorted, devs_sorted = zip(*sorted_data) if sorted_data else ([], [])
    
    colors = [COLORS['positive'] if d > 0 else COLORS['danger'] for d in devs_sorted]
    ax2.barh(names_sorted[:15], devs_sorted[:15], color=colors[:15], edgecolor='white', linewidth=0.5, height=0.7)
    ax2.axvline(x=0, color=COLORS['accent'], linestyle='-', linewidth=1)
    ax2.set_xlabel('Deviation from Average (%p)', fontsize=11, color='#333333')
    ax2.set_title('Regional Deviation from Average', fontsize=13, fontweight='600', color='#1a202c', pad=15)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(regional: Dict, benchmark: Dict, summary: Dict) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    rate = summary['overall_execution_rate']
    if rate >= 0.9:
        insights.append({'title': 'Excellent Overall Execution', 'description': f'Overall execution rate of {rate*100:.1f}% exceeds targets.', 'status': 'positive'})
    elif rate >= 0.8:
        insights.append({'title': 'Good Execution Performance', 'description': f'Overall execution rate of {rate*100:.1f}% meets expectations.', 'status': 'neutral'})
    else:
        insights.append({'title': 'Execution Below Target', 'description': f'Overall rate of {rate*100:.1f}% is below 80% target. Review needed.', 'status': 'warning'})
    
    disparity = summary['disparity_index']
    if disparity > 0.2:
        insights.append({'title': 'High Regional Disparity', 'description': f'Disparity index of {disparity:.2f} indicates significant variation across regions.', 'status': 'warning'})
    elif disparity < 0.1:
        insights.append({'title': 'Balanced Execution', 'description': f'Low disparity index ({disparity:.2f}) shows consistent performance.', 'status': 'positive'})
    
    above = len(benchmark['above_average_regions'])
    below = len(benchmark['below_average_regions'])
    if below > above:
        insights.append({'title': 'Majority Below Average', 'description': f'{below} regions below average vs {above} above. Targeted support recommended.', 'status': 'warning'})
    else:
        insights.append({'title': 'Strong Regional Performance', 'description': f'{above} regions performing above average.', 'status': 'positive'})
    
    return insights


@router.post("/budget-execution")
async def run_budget_execution_analysis(request: BudgetExecutionRequest) -> Dict[str, Any]:
    """Perform Regional Budget Execution Comparison analysis."""
    try:
        df = pd.DataFrame(request.data)
        
        if request.region_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Region column '{request.region_col}' not found")
        if request.allocated_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Allocated column '{request.allocated_col}' not found")
        if request.executed_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Executed column '{request.executed_col}' not found")
        
        df[request.allocated_col] = pd.to_numeric(df[request.allocated_col], errors='coerce').fillna(0)
        df[request.executed_col] = pd.to_numeric(df[request.executed_col], errors='coerce').fillna(0)
        
        regional_analysis = analyze_regional_execution(df, request.region_col, request.allocated_col,
                                                        request.executed_col, request.population_col)
        
        category_analysis = []
        if request.category_col:
            category_analysis = analyze_by_category(df, request.region_col, request.category_col,
                                                     request.allocated_col, request.executed_col)
        
        temporal_analysis = {'periods': [], 'execution_rates': {}, 'overall_trend': []}
        if request.period_col:
            temporal_analysis = analyze_temporal(df, request.region_col, request.period_col,
                                                  request.allocated_col, request.executed_col)
        
        benchmark_analysis = calculate_benchmark(regional_analysis['regions'], request.benchmark_type, request.target_rate)
        
        visualizations = {}
        visualizations['regional_comparison'] = create_regional_comparison_chart(regional_analysis['regions'])
        visualizations['execution_heatmap'] = create_execution_heatmap(df, request.region_col, request.period_col or '',
                                                                        request.allocated_col, request.executed_col)
        visualizations['category_breakdown'] = create_category_breakdown_chart(category_analysis)
        visualizations['temporal_trend'] = create_temporal_trend_chart(temporal_analysis)
        visualizations['disparity_chart'] = create_disparity_chart(regional_analysis['regions'], benchmark_analysis)
        
        best_region = regional_analysis['regions'][0]['region'] if regional_analysis['regions'] else 'N/A'
        worst_region = regional_analysis['regions'][-1]['region'] if regional_analysis['regions'] else 'N/A'
        
        summary = {
            'n_regions': len(regional_analysis['regions']),
            'n_categories': len(category_analysis),
            'fiscal_period': request.fiscal_period,
            'total_allocated': regional_analysis['total_allocated'],
            'total_executed': regional_analysis['total_executed'],
            'overall_execution_rate': regional_analysis['overall_execution_rate'],
            'best_performing_region': best_region,
            'worst_performing_region': worst_region,
            'disparity_index': regional_analysis['disparity_index']
        }
        
        insights = generate_key_insights(regional_analysis, benchmark_analysis, summary)
        
        return {
            'success': True,
            'regional_analysis': regional_analysis,
            'category_analysis': category_analysis,
            'temporal_analysis': temporal_analysis,
            'benchmark_analysis': benchmark_analysis,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Budget execution analysis failed: {str(e)}")
