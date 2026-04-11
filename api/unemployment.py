"""
Unemployment Trend Analysis Router for FastAPI
Labor Market Trends, Regional & Demographic Analysis
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
    'unemployment': '#e53e3e', 'employment': '#48bb78', 'participation': '#4299e1',
    'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936',
}
PALETTE = ['#4a5568', '#718096', '#a0aec0', '#4299e1', '#48bb78', '#ed8936', '#e53e3e', '#9f7aea']

router = APIRouter()


class UnemploymentRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str
    unemployment_rate_col: Optional[str] = None
    labor_force_col: Optional[str] = None
    unemployed_col: Optional[str] = None
    employed_col: Optional[str] = None
    region_col: Optional[str] = None
    age_group_col: Optional[str] = None
    gender_col: Optional[str] = None
    education_col: Optional[str] = None
    duration_col: Optional[str] = None
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


def calculate_metrics(df: pd.DataFrame, req: UnemploymentRequest) -> pd.DataFrame:
    """Calculate unemployment metrics"""
    df = df.copy()
    
    if req.unemployment_rate_col and req.unemployment_rate_col in df.columns:
        df['_unemp_rate'] = pd.to_numeric(df[req.unemployment_rate_col], errors='coerce')
    elif req.unemployed_col and req.labor_force_col:
        unemployed = pd.to_numeric(df[req.unemployed_col], errors='coerce')
        labor_force = pd.to_numeric(df[req.labor_force_col], errors='coerce')
        df['_unemp_rate'] = (unemployed / labor_force.replace(0, np.nan)) * 100
    else:
        df['_unemp_rate'] = np.nan
    
    if req.labor_force_col:
        df['_labor_force'] = pd.to_numeric(df[req.labor_force_col], errors='coerce')
    else:
        df['_labor_force'] = 0
    
    if req.unemployed_col:
        df['_unemployed'] = pd.to_numeric(df[req.unemployed_col], errors='coerce')
    elif req.labor_force_col and '_unemp_rate' in df.columns:
        df['_unemployed'] = df['_labor_force'] * df['_unemp_rate'] / 100
    else:
        df['_unemployed'] = 0
    
    if req.employed_col:
        df['_employed'] = pd.to_numeric(df[req.employed_col], errors='coerce')
    else:
        df['_employed'] = df['_labor_force'] - df['_unemployed']
    
    return df


def analyze_overall(df: pd.DataFrame) -> Dict:
    """Calculate overall metrics"""
    total_lf = df['_labor_force'].sum()
    total_unemployed = df['_unemployed'].sum()
    total_employed = df['_employed'].sum()
    
    unemp_rate = (total_unemployed / total_lf * 100) if total_lf > 0 else df['_unemp_rate'].mean()
    emp_rate = (total_employed / total_lf * 100) if total_lf > 0 else 100 - unemp_rate
    
    return {
        'unemployment_rate': _to_native(unemp_rate),
        'labor_force': _to_native(total_lf),
        'employed': _to_native(total_employed),
        'unemployed': _to_native(total_unemployed),
        'participation_rate': _to_native(63.0),  # Placeholder
        'employment_rate': _to_native(emp_rate),
        'yoy_change': _to_native(0),
        'youth_unemployment': None,
        'long_term_share': None
    }


def analyze_temporal(df: pd.DataFrame, period_col: str) -> Dict:
    """Analyze trends over time"""
    temporal = df.groupby(period_col).agg({
        '_unemp_rate': 'mean',
        '_labor_force': 'sum',
        '_employed': 'sum',
        '_unemployed': 'sum'
    }).reset_index().sort_values(period_col)
    
    temporal['_emp_rate'] = (temporal['_employed'] / temporal['_labor_force'].replace(0, np.nan)) * 100
    temporal['_part_rate'] = 63.0  # Placeholder
    
    return {
        'periods': [str(p) for p in temporal[period_col].tolist()],
        'unemployment_rates': [_to_native(x) for x in temporal['_unemp_rate'].tolist()],
        'employment_rates': [_to_native(x) for x in temporal['_emp_rate'].tolist()],
        'participation_rates': [_to_native(x) for x in temporal['_part_rate'].tolist()],
        'labor_force': [_to_native(x) for x in temporal['_labor_force'].tolist()]
    }


def analyze_regional(df: pd.DataFrame, region_col: str, period_col: str) -> List[Dict]:
    """Analyze by region"""
    if not region_col or region_col not in df.columns:
        return []
    
    results = []
    periods = sorted(df[period_col].unique())
    
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region]
        total_lf = rdf['_labor_force'].sum()
        total_unemployed = rdf['_unemployed'].sum()
        total_employed = rdf['_employed'].sum()
        
        unemp_rate = (total_unemployed / total_lf * 100) if total_lf > 0 else rdf['_unemp_rate'].mean()
        
        # YoY change
        yoy = None
        if len(periods) >= 2:
            recent = rdf[rdf[period_col] == periods[-1]]['_unemp_rate'].mean()
            prev = rdf[rdf[period_col] == periods[-5]]['_unemp_rate'].mean() if len(periods) >= 5 else rdf[rdf[period_col] == periods[0]]['_unemp_rate'].mean()
            if not np.isnan(recent) and not np.isnan(prev):
                yoy = recent - prev
        
        results.append({
            'region': str(region),
            'unemployment_rate': _to_native(unemp_rate),
            'labor_force': _to_native(total_lf),
            'employed': _to_native(total_employed),
            'unemployed': _to_native(total_unemployed),
            'participation_rate': _to_native(63.0),
            'employment_rate': _to_native((total_employed / total_lf * 100) if total_lf > 0 else 0),
            'yoy_change': _to_native(yoy)
        })
    
    return sorted(results, key=lambda x: x['unemployment_rate'] or 0, reverse=True)


def analyze_demographic(df: pd.DataFrame, age_col: str, gender_col: str, edu_col: str) -> Dict:
    """Analyze by demographics"""
    result = {'by_age': [], 'by_gender': [], 'by_education': []}
    total_unemployed = df['_unemployed'].sum()
    
    for col, key in [(age_col, 'by_age'), (gender_col, 'by_gender'), (edu_col, 'by_education')]:
        if col and col in df.columns:
            for segment in df[col].unique():
                sdf = df[df[col] == segment]
                lf = sdf['_labor_force'].sum()
                unemployed = sdf['_unemployed'].sum()
                rate = (unemployed / lf * 100) if lf > 0 else sdf['_unemp_rate'].mean()
                
                result[key].append({
                    'segment': str(segment),
                    'unemployment_rate': _to_native(rate),
                    'labor_force': _to_native(lf),
                    'unemployed': _to_native(unemployed),
                    'share_of_unemployed': _to_native(unemployed / total_unemployed if total_unemployed > 0 else 0)
                })
    
    return result


def analyze_duration(df: pd.DataFrame, duration_col: str) -> Dict:
    """Analyze unemployment duration"""
    if not duration_col or duration_col not in df.columns:
        return {'buckets': [], 'counts': [], 'shares': [], 'avg_duration_weeks': 0}
    
    duration_data = df.groupby(duration_col)['_unemployed'].sum().reset_index()
    total = duration_data['_unemployed'].sum()
    
    buckets = duration_data[duration_col].tolist()
    counts = [_to_native(x) for x in duration_data['_unemployed'].tolist()]
    shares = [_to_native(x / total if total > 0 else 0) for x in duration_data['_unemployed'].tolist()]
    
    return {
        'buckets': [str(b) for b in buckets],
        'counts': counts,
        'shares': shares,
        'avg_duration_weeks': _to_native(12)  # Placeholder
    }
# ============================================================
# Visualization Functions (Part 2)
# ============================================================

def create_trend_chart(temporal: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    periods = temporal['periods']
    if not periods:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    ax1.plot(periods, temporal['unemployment_rates'], color=COLORS['unemployment'], linewidth=2, marker='o', label='Unemployment Rate')
    ax1.fill_between(periods, temporal['unemployment_rates'], alpha=0.2, color=COLORS['unemployment'])
    ax1.axhline(5, color=COLORS['secondary'], linestyle='--', linewidth=1, label='Natural Rate (~5%)')
    ax1.set_ylabel('Unemployment Rate (%)', fontsize=11)
    ax1.set_title('Unemployment Rate Trend', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    ax2.plot(periods, temporal['employment_rates'], color=COLORS['employment'], linewidth=2, marker='s', label='Employment Rate')
    ax2.plot(periods, temporal['participation_rates'], color=COLORS['participation'], linewidth=2, marker='^', label='Participation Rate')
    ax2.set_ylabel('Rate (%)', fontsize=11)
    ax2.set_title('Employment & Participation Rates', fontsize=13, fontweight='600', pad=15)
    ax2.legend(fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_regional_chart(regional: List[Dict]) -> str:
    _setup_style()
    if not regional:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No regional data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sorted_reg = sorted(regional, key=lambda x: x['unemployment_rate'] or 0, reverse=True)[:15]
    regions = [r['region'] for r in sorted_reg]
    rates = [r['unemployment_rate'] or 0 for r in sorted_reg]
    
    colors = [COLORS['unemployment'] if r > 5 else COLORS['warning'] if r > 4 else COLORS['employment'] for r in rates]
    ax1.barh(regions, rates, color=colors, edgecolor='white', height=0.7)
    ax1.axvline(5, color=COLORS['secondary'], linestyle='--', linewidth=1.5, label='Natural Rate')
    ax1.set_xlabel('Unemployment Rate (%)', fontsize=11)
    ax1.set_title('Unemployment by Region', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    _style_axis(ax1)
    
    # YoY change
    yoy = [r['yoy_change'] or 0 for r in sorted_reg]
    colors2 = [COLORS['unemployment'] if y > 0 else COLORS['employment'] for y in yoy]
    ax2.barh(regions, yoy, color=colors2, edgecolor='white', height=0.7)
    ax2.axvline(0, color=COLORS['primary'], linestyle='-', linewidth=1)
    ax2.set_xlabel('YoY Change (%p)', fontsize=11)
    ax2.set_title('Year-over-Year Change', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_demographic_chart(demographics: Dict) -> str:
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (key, title) in zip(axes, [('by_age', 'By Age'), ('by_gender', 'By Gender'), ('by_education', 'By Education')]):
        data = demographics.get(key, [])
        if data:
            segments = [d['segment'] for d in data]
            rates = [d['unemployment_rate'] or 0 for d in data]
            colors = [COLORS['unemployment'] if r > 8 else COLORS['warning'] if r > 5 else COLORS['employment'] for r in rates]
            ax.bar(segments, rates, color=colors, edgecolor='white')
            ax.axhline(5, color=COLORS['secondary'], linestyle='--', linewidth=1)
            ax.set_ylabel('Unemployment Rate (%)', fontsize=10)
            ax.set_title(f'Unemployment {title}', fontsize=11, fontweight='600')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            _style_axis(ax)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Unemployment {title}', fontsize=11, fontweight='600')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_duration_chart(duration: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    buckets = duration['buckets']
    counts = duration['counts']
    shares = [s * 100 for s in duration['shares']]
    
    if not buckets:
        ax1.text(0.5, 0.5, 'No duration data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    colors = [COLORS['employment'], COLORS['participation'], COLORS['warning'], COLORS['unemployment'], COLORS['primary']][:len(buckets)]
    ax1.bar(buckets, counts, color=colors, edgecolor='white')
    ax1.set_ylabel('Number of Unemployed', fontsize=11)
    ax1.set_title('Unemployment by Duration', fontsize=13, fontweight='600', pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    ax2.pie(shares, labels=buckets, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Duration Distribution', fontsize=13, fontweight='600', pad=15)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_decomposition_chart(temporal: Dict, demographics: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    periods = temporal['periods']
    rates = temporal['unemployment_rates']
    
    if periods:
        ax1.fill_between(periods, rates, alpha=0.3, color=COLORS['unemployment'])
        ax1.plot(periods, rates, color=COLORS['unemployment'], linewidth=2)
        ax1.set_ylabel('Unemployment Rate (%)', fontsize=11)
        ax1.set_title('Unemployment Trend Decomposition', fontsize=13, fontweight='600', pad=15)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        _style_axis(ax1)
    
    by_age = demographics.get('by_age', [])
    if by_age:
        segments = [d['segment'] for d in by_age]
        shares = [d['share_of_unemployed'] * 100 for d in by_age]
        ax2.pie(shares, labels=segments, autopct='%1.1f%%', colors=PALETTE[:len(segments)], startangle=90)
        ax2.set_title('Unemployed by Age Group', fontsize=13, fontweight='600', pad=15)
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(overall: Dict, temporal: Dict, regional: List[Dict], demographics: Dict, duration: Dict) -> List[Dict]:
    insights = []
    
    rate = overall['unemployment_rate'] or 0
    if rate < 4:
        insights.append({'title': 'Low Unemployment', 'description': f'Unemployment rate of {rate:.1f}% indicates a tight labor market with near-full employment.', 'status': 'positive'})
    elif rate > 6:
        insights.append({'title': 'High Unemployment', 'description': f'Unemployment rate of {rate:.1f}% indicates labor market slack. Policy intervention may be needed.', 'status': 'warning'})
    else:
        insights.append({'title': 'Moderate Unemployment', 'description': f'Unemployment rate of {rate:.1f}% is near the natural rate.', 'status': 'neutral'})
    
    if demographics.get('by_age'):
        youth = next((d for d in demographics['by_age'] if '15-24' in d['segment'] or '청년' in d['segment']), None)
        if youth and youth['unemployment_rate'] and youth['unemployment_rate'] > rate * 2:
            insights.append({'title': 'Youth Unemployment Crisis', 'description': f'Youth unemployment ({youth["unemployment_rate"]:.1f}%) is significantly higher than overall rate.', 'status': 'warning'})
    
    if regional:
        highest = regional[0]
        lowest = regional[-1]
        gap = (highest['unemployment_rate'] or 0) - (lowest['unemployment_rate'] or 0)
        if gap > 3:
            insights.append({'title': 'Regional Disparity', 'description': f'{gap:.1f}%p gap between {highest["region"]} and {lowest["region"]}. Regional policy attention needed.', 'status': 'warning'})
    
    if duration.get('shares') and len(duration['shares']) >= 5:
        long_term = duration['shares'][-1]
        if long_term > 0.2:
            insights.append({'title': 'Long-term Unemployment', 'description': f'{long_term*100:.1f}% unemployed for 12+ months. Structural issues possible.', 'status': 'warning'})
    
    return insights


@router.post("/unemployment")
async def run_unemployment_analysis(request: UnemploymentRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        
        if request.period_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Period column '{request.period_col}' not found")
        
        df = calculate_metrics(df, request)
        overall = analyze_overall(df)
        temporal = analyze_temporal(df, request.period_col)
        regional = analyze_regional(df, request.region_col, request.period_col)
        demographics = analyze_demographic(df, request.age_group_col, request.gender_col, request.education_col)
        duration = analyze_duration(df, request.duration_col)
        
        # YoY change
        if len(temporal['unemployment_rates']) >= 5:
            overall['yoy_change'] = _to_native(temporal['unemployment_rates'][-1] - temporal['unemployment_rates'][-5])
        
        visualizations = {
            'trend_chart': create_trend_chart(temporal),
            'regional_comparison': create_regional_chart(regional),
            'demographic_chart': create_demographic_chart(demographics),
            'duration_chart': create_duration_chart(duration),
            'decomposition_chart': create_decomposition_chart(temporal, demographics)
        }
        
        # Trend direction
        if len(temporal['unemployment_rates']) >= 3:
            recent = np.mean(temporal['unemployment_rates'][-3:])
            earlier = np.mean(temporal['unemployment_rates'][:3])
            trend = "Improving" if recent < earlier * 0.95 else "Worsening" if recent > earlier * 1.05 else "Stable"
        else:
            trend = "Insufficient data"
        
        # Labor market status
        rate = overall['unemployment_rate'] or 0
        status = "Tight labor market" if rate < 4 else "Full employment" if rate < 5 else "Moderate slack" if rate < 7 else "Significant slack"
        
        highest = regional[0] if regional else {'region': 'N/A'}
        lowest = regional[-1] if regional else {'region': 'N/A'}
        
        # Highest demographic
        all_demos = demographics.get('by_age', []) + demographics.get('by_gender', []) + demographics.get('by_education', [])
        highest_demo = max(all_demos, key=lambda x: x['unemployment_rate'] or 0)['segment'] if all_demos else 'N/A'
        
        summary = {
            'analysis_period': request.analysis_period,
            'unemployment_rate': overall['unemployment_rate'] or 0,
            'labor_force': overall['labor_force'] or 0,
            'unemployed': overall['unemployed'] or 0,
            'yoy_change': overall['yoy_change'] or 0,
            'trend_direction': trend,
            'highest_region': highest['region'],
            'lowest_region': lowest['region'],
            'highest_demographic': highest_demo,
            'labor_market_status': status
        }
        
        insights = generate_insights(overall, temporal, regional, demographics, duration)
        
        return {
            'success': True,
            'overall_metrics': overall,
            'temporal_analysis': temporal,
            'regional_analysis': regional,
            'demographic_analysis': demographics,
            'duration_analysis': duration,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unemployment analysis failed: {str(e)}")
