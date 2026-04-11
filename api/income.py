"""
Regional Income Level Analysis Router for FastAPI
Income Distribution, Gini, Regional Comparison, Inequality
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
    'income': '#48bb78', 'median': '#4299e1', 'poverty': '#e53e3e',
    'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936',
}
PALETTE = ['#48bb78', '#38a169', '#2f855a', '#276749', '#22543d', '#4a5568', '#718096', '#a0aec0']

router = APIRouter()


class IncomeRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: Optional[str] = None
    region_col: str
    gross_income_col: Optional[str] = None
    net_income_col: Optional[str] = None
    population_col: Optional[str] = None
    households_col: Optional[str] = None
    quintile_col: Optional[str] = None
    income_type_col: Optional[str] = None
    cost_of_living_col: Optional[str] = None
    analysis_focus: str = "regional"
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


def calculate_gini(incomes: np.ndarray, weights: np.ndarray = None) -> float:
    """Calculate Gini coefficient"""
    if len(incomes) == 0:
        return 0.0
    
    incomes = np.array(incomes)
    if weights is None:
        weights = np.ones(len(incomes))
    weights = np.array(weights)
    
    # Sort by income
    sorted_idx = np.argsort(incomes)
    incomes = incomes[sorted_idx]
    weights = weights[sorted_idx]
    
    # Weighted cumulative sums
    cum_weights = np.cumsum(weights)
    cum_income = np.cumsum(incomes * weights)
    
    total_weight = cum_weights[-1]
    total_income = cum_income[-1]
    
    if total_income == 0 or total_weight == 0:
        return 0.0
    
    # Gini calculation
    gini = 1 - 2 * np.sum(cum_income * weights) / (total_weight * total_income)
    return max(0, min(1, gini))


def analyze_overall(df: pd.DataFrame, income_col: str, pop_col: str) -> Dict:
    """Calculate overall income metrics"""
    incomes = df[income_col].dropna()
    weights = df[pop_col].dropna() if pop_col and pop_col in df.columns else None
    
    if weights is not None and len(weights) == len(incomes):
        total_income = (incomes * weights).sum()
        total_pop = weights.sum()
        mean_income = total_income / total_pop if total_pop > 0 else incomes.mean()
        median_income = incomes.median()
        per_capita = mean_income
    else:
        total_income = incomes.sum()
        mean_income = incomes.mean()
        median_income = incomes.median()
        per_capita = mean_income
        total_pop = len(incomes)
    
    gini = calculate_gini(incomes.values, weights.values if weights is not None else None)
    
    # Poverty rate (relative: below 50% of median)
    poverty_line = median_income * 0.5
    if weights is not None and len(weights) == len(incomes):
        below_poverty = weights[incomes < poverty_line].sum()
        poverty_rate = (below_poverty / total_pop * 100) if total_pop > 0 else 0
    else:
        poverty_rate = (incomes < poverty_line).mean() * 100
    
    # Income gap (Q5/Q1)
    q1 = incomes.quantile(0.2)
    q5 = incomes.quantile(0.8)
    income_gap = q5 / q1 if q1 > 0 else 0
    
    return {
        'total_income': _to_native(total_income),
        'mean_income': _to_native(mean_income),
        'median_income': _to_native(median_income),
        'per_capita_income': _to_native(per_capita),
        'gini_coefficient': _to_native(gini),
        'poverty_rate': _to_native(poverty_rate),
        'income_gap_ratio': _to_native(income_gap),
        'yoy_growth': _to_native(0)
    }


def analyze_temporal(df: pd.DataFrame, period_col: str, income_col: str, pop_col: str) -> Dict:
    """Analyze income trends over time"""
    if not period_col or period_col not in df.columns:
        return {'periods': [], 'mean_incomes': [], 'median_incomes': [], 'gini_trend': [], 'growth_rates': []}
    
    periods = sorted(df[period_col].unique())
    mean_incomes = []
    median_incomes = []
    gini_trend = []
    
    for period in periods:
        pdf = df[df[period_col] == period]
        incomes = pdf[income_col].dropna()
        weights = pdf[pop_col].dropna() if pop_col and pop_col in pdf.columns else None
        
        if weights is not None and len(weights) == len(incomes):
            mean_inc = (incomes * weights).sum() / weights.sum() if weights.sum() > 0 else incomes.mean()
        else:
            mean_inc = incomes.mean()
        
        mean_incomes.append(_to_native(mean_inc))
        median_incomes.append(_to_native(incomes.median()))
        gini_trend.append(_to_native(calculate_gini(incomes.values, weights.values if weights is not None else None)))
    
    # Growth rates
    growth_rates = [0]
    for i in range(1, len(mean_incomes)):
        if mean_incomes[i-1] and mean_incomes[i-1] > 0:
            growth = ((mean_incomes[i] or 0) - mean_incomes[i-1]) / mean_incomes[i-1] * 100
            growth_rates.append(_to_native(growth))
        else:
            growth_rates.append(0)
    
    return {
        'periods': [str(p) for p in periods],
        'mean_incomes': mean_incomes,
        'median_incomes': median_incomes,
        'gini_trend': gini_trend,
        'growth_rates': growth_rates
    }


def analyze_regional(df: pd.DataFrame, region_col: str, income_col: str, pop_col: str, col_col: str, period_col: str) -> List[Dict]:
    """Analyze income by region"""
    results = []
    periods = sorted(df[period_col].unique()) if period_col and period_col in df.columns else []
    
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region]
        incomes = rdf[income_col].dropna()
        weights = rdf[pop_col].dropna() if pop_col and pop_col in rdf.columns else None
        
        if weights is not None and len(weights) == len(incomes):
            total_pop = weights.sum()
            mean_inc = (incomes * weights).sum() / total_pop if total_pop > 0 else incomes.mean()
        else:
            total_pop = len(incomes)
            mean_inc = incomes.mean()
        
        median_inc = incomes.median()
        gini = calculate_gini(incomes.values, weights.values if weights is not None else None)
        
        poverty_line = median_inc * 0.5
        if weights is not None and len(weights) == len(incomes):
            poverty_rate = (weights[incomes < poverty_line].sum() / total_pop * 100) if total_pop > 0 else 0
        else:
            poverty_rate = (incomes < poverty_line).mean() * 100
        
        # YoY growth
        yoy = None
        if len(periods) >= 2:
            recent = rdf[rdf[period_col] == periods[-1]][income_col].mean() if period_col else None
            prev = rdf[rdf[period_col] == periods[0]][income_col].mean() if period_col else None
            if recent and prev and prev > 0:
                yoy = ((recent - prev) / prev) * 100 / max(1, len(periods) - 1)
        
        col_idx = rdf[col_col].mean() if col_col and col_col in rdf.columns else 100
        
        results.append({
            'region': str(region),
            'population': _to_native(total_pop),
            'mean_income': _to_native(mean_inc),
            'median_income': _to_native(median_inc),
            'per_capita_income': _to_native(mean_inc),
            'gini': _to_native(gini),
            'poverty_rate': _to_native(poverty_rate),
            'yoy_growth': _to_native(yoy),
            'cost_of_living_index': _to_native(col_idx)
        })
    
    return sorted(results, key=lambda x: x['mean_income'] or 0, reverse=True)


def analyze_distribution(df: pd.DataFrame, income_col: str, pop_col: str, quintile_col: str) -> Dict:
    """Analyze income distribution by quintiles"""
    quintiles = []
    deciles = []
    
    if quintile_col and quintile_col in df.columns:
        total_income = df[income_col].sum()
        total_pop = df[pop_col].sum() if pop_col and pop_col in df.columns else len(df)
        
        for q in sorted(df[quintile_col].unique()):
            qdf = df[df[quintile_col] == q]
            q_income = qdf[income_col].sum()
            q_pop = qdf[pop_col].sum() if pop_col and pop_col in qdf.columns else len(qdf)
            
            quintiles.append({
                'quintile': str(q),
                'income_share': _to_native(q_income / total_income if total_income > 0 else 0),
                'avg_income': _to_native(qdf[income_col].mean()),
                'population_share': _to_native(q_pop / total_pop if total_pop > 0 else 0)
            })
    else:
        # Calculate quintiles from data
        incomes = df[income_col].dropna()
        for i, (lo, hi) in enumerate([(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]):
            q_lo = incomes.quantile(lo)
            q_hi = incomes.quantile(hi)
            mask = (incomes >= q_lo) & (incomes <= q_hi)
            q_inc = incomes[mask]
            
            quintiles.append({
                'quintile': f'Q{i+1}',
                'income_share': _to_native(q_inc.sum() / incomes.sum() if incomes.sum() > 0 else 0),
                'avg_income': _to_native(q_inc.mean()),
                'population_share': _to_native(0.2)
            })
    
    return {'quintiles': quintiles, 'deciles': deciles}


def analyze_inequality(df: pd.DataFrame, income_col: str, pop_col: str) -> Dict:
    """Calculate inequality metrics"""
    incomes = df[income_col].dropna()
    weights = df[pop_col].dropna() if pop_col and pop_col in df.columns else None
    
    gini = calculate_gini(incomes.values, weights.values if weights is not None else None)
    
    p10 = incomes.quantile(0.1)
    p40 = incomes.quantile(0.4)
    p90 = incomes.quantile(0.9)
    
    # Palma ratio: top 10% / bottom 40%
    top_10 = incomes[incomes >= p90].sum()
    bottom_40 = incomes[incomes <= p40].sum()
    palma = top_10 / bottom_40 if bottom_40 > 0 else 0
    
    # P90/P10 ratio
    p90_p10 = p90 / p10 if p10 > 0 else 0
    
    # Shares
    total = incomes.sum()
    top_10_share = top_10 / total if total > 0 else 0
    bottom_40_share = bottom_40 / total if total > 0 else 0
    
    return {
        'gini': _to_native(gini),
        'palma_ratio': _to_native(palma),
        'p90_p10_ratio': _to_native(p90_p10),
        'top_10_share': _to_native(top_10_share),
        'bottom_40_share': _to_native(bottom_40_share)
    }
# ============================================================
# Visualization Functions (Part 2)
# ============================================================

def create_regional_chart(regional: List[Dict]) -> str:
    _setup_style()
    if not regional:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No regional data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sorted_reg = sorted(regional, key=lambda x: x['mean_income'] or 0, reverse=True)[:15]
    regions = [r['region'] for r in sorted_reg]
    mean_inc = [r['mean_income'] / 10000 for r in sorted_reg]  # 만원 단위
    
    colors = [COLORS['income'] if m > np.mean(mean_inc) else COLORS['secondary'] for m in mean_inc]
    ax1.barh(regions, mean_inc, color=colors, edgecolor='white', height=0.7)
    ax1.axvline(np.mean(mean_inc), color=COLORS['primary'], linestyle='--', linewidth=1.5, label='National Avg')
    ax1.set_xlabel('Mean Income (만원)', fontsize=11)
    ax1.set_title('Income by Region', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    _style_axis(ax1)
    
    # Gini by region
    ginis = [r['gini'] or 0 for r in sorted_reg]
    colors2 = [COLORS['poverty'] if g > 0.4 else COLORS['warning'] if g > 0.35 else COLORS['income'] for g in ginis]
    ax2.barh(regions, ginis, color=colors2, edgecolor='white', height=0.7)
    ax2.axvline(0.35, color=COLORS['primary'], linestyle='--', linewidth=1.5, label='Moderate threshold')
    ax2.set_xlabel('Gini Coefficient', fontsize=11)
    ax2.set_title('Inequality by Region', fontsize=13, fontweight='600', pad=15)
    ax2.legend(fontsize=9)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_distribution_chart(distribution: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    quintiles = distribution['quintiles']
    if not quintiles:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    labels = [q['quintile'] for q in quintiles]
    shares = [q['income_share'] * 100 for q in quintiles]
    avg_inc = [q['avg_income'] / 10000 for q in quintiles]
    
    ax1.bar(labels, shares, color=PALETTE[:len(labels)], edgecolor='white')
    ax1.axhline(20, color=COLORS['primary'], linestyle='--', linewidth=1, label='Equal share (20%)')
    ax1.set_ylabel('Income Share (%)', fontsize=11)
    ax1.set_title('Income Share by Quintile', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    _style_axis(ax1)
    
    ax2.bar(labels, avg_inc, color=PALETTE[:len(labels)], edgecolor='white')
    ax2.set_ylabel('Average Income (만원)', fontsize=11)
    ax2.set_title('Average Income by Quintile', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_trend_chart(temporal: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    periods = temporal['periods']
    if not periods:
        ax1.text(0.5, 0.5, 'No temporal data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    mean_inc = [m / 10000 if m else 0 for m in temporal['mean_incomes']]
    median_inc = [m / 10000 if m else 0 for m in temporal['median_incomes']]
    
    ax1.plot(periods, mean_inc, color=COLORS['income'], linewidth=2, marker='o', label='Mean Income')
    ax1.plot(periods, median_inc, color=COLORS['median'], linewidth=2, marker='s', label='Median Income')
    ax1.fill_between(periods, mean_inc, median_inc, alpha=0.2, color=COLORS['primary'])
    ax1.set_ylabel('Income (만원)', fontsize=11)
    ax1.set_title('Income Trend', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    gini = temporal['gini_trend']
    ax2.plot(periods, gini, color=COLORS['poverty'], linewidth=2, marker='o')
    ax2.fill_between(periods, gini, alpha=0.2, color=COLORS['poverty'])
    ax2.axhline(0.35, color=COLORS['secondary'], linestyle='--', linewidth=1, label='Moderate threshold')
    ax2.set_ylabel('Gini Coefficient', fontsize=11)
    ax2.set_title('Inequality Trend', fontsize=13, fontweight='600', pad=15)
    ax2.legend(fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_inequality_chart(inequality: Dict, distribution: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['Gini', 'Palma', 'P90/P10']
    values = [inequality['gini'], inequality['palma_ratio'], inequality['p90_p10_ratio'] / 10]
    colors = [COLORS['poverty'] if v > 0.4 else COLORS['warning'] if v > 0.3 else COLORS['income'] for v in values]
    
    ax1.bar(metrics, values, color=colors, edgecolor='white')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Inequality Metrics', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax1)
    
    # Top vs Bottom shares
    labels = ['Top 10%', 'Bottom 40%', 'Middle 50%']
    shares = [
        inequality['top_10_share'] * 100,
        inequality['bottom_40_share'] * 100,
        (1 - inequality['top_10_share'] - inequality['bottom_40_share']) * 100
    ]
    ax2.pie(shares, labels=labels, autopct='%1.1f%%', colors=[COLORS['poverty'], COLORS['income'], COLORS['median']], startangle=90)
    ax2.set_title('Income Share Distribution', fontsize=13, fontweight='600', pad=15)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_lorenz_chart(df: pd.DataFrame, income_col: str) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    incomes = df[income_col].dropna().sort_values()
    n = len(incomes)
    
    if n == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    cum_pop = np.arange(1, n + 1) / n
    cum_income = np.cumsum(incomes) / incomes.sum()
    
    # Lorenz curve
    ax.plot([0] + list(cum_pop), [0] + list(cum_income), color=COLORS['income'], linewidth=2, label='Lorenz Curve')
    ax.plot([0, 1], [0, 1], color=COLORS['primary'], linestyle='--', linewidth=1.5, label='Perfect Equality')
    ax.fill_between([0] + list(cum_pop), [0] + list(cum_income), [0] + list(cum_pop), alpha=0.3, color=COLORS['poverty'])
    
    ax.set_xlabel('Cumulative Population Share', fontsize=11)
    ax.set_ylabel('Cumulative Income Share', fontsize=11)
    ax.set_title('Lorenz Curve', fontsize=13, fontweight='600', pad=15)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    _style_axis(ax)
    
    return _fig_to_base64(fig)


def generate_insights(overall: Dict, regional: List[Dict], inequality: Dict) -> List[Dict]:
    insights = []
    
    gini = overall['gini_coefficient'] or 0
    if gini < 0.30:
        insights.append({'title': 'Low Inequality', 'description': f'Gini of {gini:.3f} indicates relatively equal income distribution.', 'status': 'positive'})
    elif gini > 0.40:
        insights.append({'title': 'High Inequality', 'description': f'Gini of {gini:.3f} suggests significant income disparity. Policy intervention may be needed.', 'status': 'warning'})
    else:
        insights.append({'title': 'Moderate Inequality', 'description': f'Gini of {gini:.3f} is within typical range for developed economies.', 'status': 'neutral'})
    
    poverty = overall['poverty_rate'] or 0
    if poverty > 15:
        insights.append({'title': 'High Poverty Rate', 'description': f'{poverty:.1f}% below poverty line. Strengthening safety net recommended.', 'status': 'warning'})
    
    if regional:
        highest = regional[0]
        lowest = regional[-1]
        gap = (highest['mean_income'] or 0) / (lowest['mean_income'] or 1)
        if gap > 1.5:
            insights.append({'title': 'Regional Income Gap', 'description': f'{highest["region"]} income {gap:.1f}x higher than {lowest["region"]}. Regional development policy needed.', 'status': 'warning'})
    
    top_10 = inequality['top_10_share'] or 0
    if top_10 > 0.30:
        insights.append({'title': 'Top Income Concentration', 'description': f'Top 10% hold {top_10*100:.1f}% of total income.', 'status': 'warning'})
    
    return insights


@router.post("/income")
async def run_income_analysis(request: IncomeRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        
        if request.region_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Region column '{request.region_col}' not found")
        
        income_col = request.gross_income_col or request.net_income_col
        if not income_col or income_col not in df.columns:
            raise HTTPException(status_code=400, detail="Income column not found")
        
        df[income_col] = pd.to_numeric(df[income_col], errors='coerce')
        
        pop_col = request.population_col or request.households_col
        
        overall = analyze_overall(df, income_col, pop_col)
        temporal = analyze_temporal(df, request.period_col, income_col, pop_col)
        regional = analyze_regional(df, request.region_col, income_col, pop_col, request.cost_of_living_col, request.period_col)
        distribution = analyze_distribution(df, income_col, pop_col, request.quintile_col)
        inequality = analyze_inequality(df, income_col, pop_col)
        
        # YoY growth
        if len(temporal['mean_incomes']) >= 2:
            first = temporal['mean_incomes'][0]
            last = temporal['mean_incomes'][-1]
            if first and last and first > 0:
                overall['yoy_growth'] = _to_native(((last - first) / first) * 100 / max(1, len(temporal['periods']) - 1))
        
        visualizations = {
            'regional_map': create_regional_chart(regional),
            'distribution_chart': create_distribution_chart(distribution),
            'trend_chart': create_trend_chart(temporal),
            'inequality_chart': create_inequality_chart(inequality, distribution),
            'lorenz_curve': create_lorenz_chart(df, income_col)
        }
        
        highest = regional[0] if regional else {'region': 'N/A'}
        lowest = regional[-1] if regional else {'region': 'N/A'}
        
        gini = overall['gini_coefficient'] or 0
        ineq_level = "Low" if gini < 0.30 else "Moderate" if gini < 0.35 else "High" if gini < 0.40 else "Very High"
        gap_status = f"{((highest['mean_income'] or 1) / (lowest['mean_income'] or 1)):.1f}x gap" if regional else "N/A"
        
        summary = {
            'analysis_period': request.analysis_period,
            'mean_income': overall['mean_income'] or 0,
            'median_income': overall['median_income'] or 0,
            'gini': gini,
            'poverty_rate': overall['poverty_rate'] or 0,
            'yoy_growth': overall['yoy_growth'] or 0,
            'highest_region': highest['region'],
            'lowest_region': lowest['region'],
            'income_gap_status': gap_status,
            'inequality_level': ineq_level
        }
        
        insights = generate_insights(overall, regional, inequality)
        
        return {
            'success': True,
            'overall_metrics': overall,
            'temporal_analysis': temporal,
            'regional_analysis': regional,
            'distribution_analysis': distribution,
            'inequality_analysis': inequality,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Income analysis failed: {str(e)}")
