"""
Birth & Mortality Rate Trends Analysis Router for FastAPI
Demographic Trends, Fertility, Mortality, Population Projections
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
plt.rcParams['axes.facecolor'] = 'white'

COLORS = {
    'birth': '#4299e1', 'death': '#e53e3e', 'natural': '#48bb78',
    'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936',
}
PALETTE = ['#4a5568', '#718096', '#a0aec0', '#4299e1', '#48bb78', '#ed8936', '#e53e3e', '#9f7aea']

router = APIRouter()


class BirthMortalityRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str
    population_col: Optional[str] = None
    births_col: Optional[str] = None
    deaths_col: Optional[str] = None
    birth_rate_col: Optional[str] = None
    death_rate_col: Optional[str] = None
    tfr_col: Optional[str] = None
    imr_col: Optional[str] = None
    region_col: Optional[str] = None
    age_group_col: Optional[str] = None
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


def calculate_rates(df: pd.DataFrame, req: BirthMortalityRequest) -> pd.DataFrame:
    """Calculate birth and death rates if not provided"""
    df = df.copy()
    
    if req.birth_rate_col and req.birth_rate_col in df.columns:
        df['_birth_rate'] = pd.to_numeric(df[req.birth_rate_col], errors='coerce')
    elif req.births_col and req.population_col:
        births = pd.to_numeric(df[req.births_col], errors='coerce')
        pop = pd.to_numeric(df[req.population_col], errors='coerce')
        df['_birth_rate'] = (births / pop) * 1000
    else:
        df['_birth_rate'] = np.nan
    
    if req.death_rate_col and req.death_rate_col in df.columns:
        df['_death_rate'] = pd.to_numeric(df[req.death_rate_col], errors='coerce')
    elif req.deaths_col and req.population_col:
        deaths = pd.to_numeric(df[req.deaths_col], errors='coerce')
        pop = pd.to_numeric(df[req.population_col], errors='coerce')
        df['_death_rate'] = (deaths / pop) * 1000
    else:
        df['_death_rate'] = np.nan
    
    df['_natural_increase'] = df['_birth_rate'] - df['_death_rate']
    
    if req.tfr_col and req.tfr_col in df.columns:
        df['_tfr'] = pd.to_numeric(df[req.tfr_col], errors='coerce')
    else:
        df['_tfr'] = df['_birth_rate'] / 8  # Rough approximation
    
    if req.imr_col and req.imr_col in df.columns:
        df['_imr'] = pd.to_numeric(df[req.imr_col], errors='coerce')
    else:
        df['_imr'] = 3.0  # Default
    
    return df


def analyze_overall(df: pd.DataFrame, req: BirthMortalityRequest) -> Dict:
    """Calculate overall demographic metrics"""
    total_pop = df[req.population_col].sum() if req.population_col else 0
    total_births = df[req.births_col].sum() if req.births_col and req.births_col in df.columns else 0
    total_deaths = df[req.deaths_col].sum() if req.deaths_col and req.deaths_col in df.columns else 0
    
    birth_rate = df['_birth_rate'].mean()
    death_rate = df['_death_rate'].mean()
    natural_inc = df['_natural_increase'].mean()
    tfr = df['_tfr'].mean()
    imr = df['_imr'].mean()
    
    return {
        'total_population': _to_native(total_pop),
        'total_births': _to_native(total_births),
        'total_deaths': _to_native(total_deaths),
        'crude_birth_rate': _to_native(birth_rate),
        'crude_death_rate': _to_native(death_rate),
        'natural_increase_rate': _to_native(natural_inc),
        'tfr': _to_native(tfr),
        'imr': _to_native(imr),
        'life_expectancy': None
    }


def analyze_temporal(df: pd.DataFrame, period_col: str) -> Dict:
    """Analyze trends over time"""
    temporal = df.groupby(period_col).agg({
        '_birth_rate': 'mean',
        '_death_rate': 'mean',
        '_natural_increase': 'mean',
        '_tfr': 'mean'
    }).reset_index().sort_values(period_col)
    
    return {
        'periods': [str(p) for p in temporal[period_col].tolist()],
        'birth_rates': [_to_native(x) for x in temporal['_birth_rate'].tolist()],
        'death_rates': [_to_native(x) for x in temporal['_death_rate'].tolist()],
        'natural_increase': [_to_native(x) for x in temporal['_natural_increase'].tolist()],
        'tfr_trend': [_to_native(x) for x in temporal['_tfr'].tolist()]
    }


def analyze_regional(df: pd.DataFrame, region_col: str, req: BirthMortalityRequest) -> List[Dict]:
    """Analyze by region"""
    if not region_col or region_col not in df.columns:
        return []
    
    results = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region]
        pop = rdf[req.population_col].sum() if req.population_col else 0
        births = rdf[req.births_col].sum() if req.births_col and req.births_col in rdf.columns else 0
        deaths = rdf[req.deaths_col].sum() if req.deaths_col and req.deaths_col in rdf.columns else 0
        
        results.append({
            'region': str(region),
            'population': _to_native(pop),
            'births': _to_native(births),
            'deaths': _to_native(deaths),
            'birth_rate': _to_native(rdf['_birth_rate'].mean()),
            'death_rate': _to_native(rdf['_death_rate'].mean()),
            'natural_increase': _to_native(rdf['_natural_increase'].mean()),
            'tfr': _to_native(rdf['_tfr'].mean()),
            'imr': _to_native(rdf['_imr'].mean())
        })
    
    return sorted(results, key=lambda x: x['birth_rate'] or 0, reverse=True)


def analyze_age(df: pd.DataFrame, age_col: str) -> Dict:
    """Analyze by age group"""
    if not age_col or age_col not in df.columns:
        return {'fertility_by_age': [], 'mortality_by_age': []}
    
    fertility = []
    mortality = []
    
    for age in df[age_col].unique():
        adf = df[df[age_col] == age]
        fertility.append({
            'age_group': str(age),
            'population': _to_native(adf['population'].sum() if 'population' in adf.columns else 0),
            'births': _to_native(adf['births'].sum() if 'births' in adf.columns else 0),
            'deaths': _to_native(adf['deaths'].sum() if 'deaths' in adf.columns else 0),
            'fertility_rate': _to_native(adf['fertility_rate'].mean() if 'fertility_rate' in adf.columns else None),
            'mortality_rate': _to_native(adf['mortality_rate'].mean() if 'mortality_rate' in adf.columns else adf['_death_rate'].mean())
        })
    
    return {'fertility_by_age': fertility, 'mortality_by_age': fertility}


def project_population(temporal: Dict, years_ahead: int = 10) -> Dict:
    """Simple population projection"""
    if len(temporal['periods']) < 3:
        return {'years': [], 'projected_population': [], 'projected_births': [], 'projected_deaths': []}
    
    last_year = int(temporal['periods'][-1]) if temporal['periods'][-1].isdigit() else 2024
    years = list(range(last_year + 1, last_year + years_ahead + 1))
    
    # Simple linear projection
    birth_trend = np.mean(np.diff(temporal['birth_rates'][-5:])) if len(temporal['birth_rates']) >= 5 else -0.2
    death_trend = np.mean(np.diff(temporal['death_rates'][-5:])) if len(temporal['death_rates']) >= 5 else 0.1
    
    proj_births = []
    proj_deaths = []
    proj_pop = []
    
    last_birth = temporal['birth_rates'][-1] if temporal['birth_rates'] else 7
    last_death = temporal['death_rates'][-1] if temporal['death_rates'] else 6
    base_pop = 50000000
    
    for i, year in enumerate(years):
        br = max(3, last_birth + birth_trend * (i + 1))
        dr = max(5, last_death + death_trend * (i + 1))
        pop = base_pop * (1 + (br - dr) / 1000) ** (i + 1)
        proj_births.append(_to_native(br))
        proj_deaths.append(_to_native(dr))
        proj_pop.append(_to_native(pop))
    
    return {
        'years': years,
        'projected_population': proj_pop,
        'projected_births': proj_births,
        'projected_deaths': proj_deaths
    }
# ============================================================
# Visualization Functions (Part 2)
# ============================================================

def create_rate_trends_chart(temporal: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    periods = temporal['periods']
    if not periods:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    ax1.plot(periods, temporal['birth_rates'], color=COLORS['birth'], linewidth=2, marker='o', label='Birth Rate')
    ax1.plot(periods, temporal['death_rates'], color=COLORS['death'], linewidth=2, marker='s', label='Death Rate')
    ax1.fill_between(periods, temporal['birth_rates'], temporal['death_rates'], alpha=0.2, 
                     color=COLORS['natural'] if temporal['birth_rates'][-1] > temporal['death_rates'][-1] else COLORS['death'])
    ax1.set_ylabel('Rate (per 1,000)', fontsize=11)
    ax1.set_title('Birth and Death Rate Trends', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    colors = [COLORS['natural'] if n >= 0 else COLORS['death'] for n in temporal['natural_increase']]
    ax2.bar(periods, temporal['natural_increase'], color=colors, edgecolor='white')
    ax2.axhline(0, color=COLORS['primary'], linestyle='-', linewidth=1)
    ax2.set_ylabel('Natural Increase (‰)', fontsize=11)
    ax2.set_title('Natural Population Change', fontsize=13, fontweight='600', pad=15)
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
    
    regions = [r['region'] for r in regional[:12]]
    birth_rates = [r['birth_rate'] or 0 for r in regional[:12]]
    death_rates = [r['death_rate'] or 0 for r in regional[:12]]
    natural_inc = [r['natural_increase'] or 0 for r in regional[:12]]
    
    x = np.arange(len(regions))
    width = 0.35
    ax1.barh(x - width/2, birth_rates, width, label='Birth Rate', color=COLORS['birth'])
    ax1.barh(x + width/2, death_rates, width, label='Death Rate', color=COLORS['death'])
    ax1.set_yticks(x)
    ax1.set_yticklabels(regions)
    ax1.set_xlabel('Rate (per 1,000)', fontsize=11)
    ax1.set_title('Birth vs Death Rate by Region', fontsize=13, fontweight='600', pad=15)
    ax1.legend(fontsize=9)
    _style_axis(ax1)
    
    colors = [COLORS['natural'] if n >= 0 else COLORS['death'] for n in natural_inc]
    ax2.barh(regions, natural_inc, color=colors, edgecolor='white', height=0.6)
    ax2.axvline(0, color=COLORS['primary'], linestyle='-', linewidth=1)
    ax2.set_xlabel('Natural Increase (‰)', fontsize=11)
    ax2.set_title('Natural Increase by Region', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_age_chart(age_data: Dict) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    fertility = age_data.get('fertility_by_age', [])
    if not fertility:
        ax.text(0.5, 0.5, 'No age data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    ages = [a['age_group'] for a in fertility]
    fert_rates = [a['fertility_rate'] or 0 for a in fertility]
    mort_rates = [a['mortality_rate'] or 0 for a in fertility]
    
    x = np.arange(len(ages))
    ax.bar(x - 0.2, fert_rates, 0.4, label='Fertility Rate', color=COLORS['birth'])
    ax.bar(x + 0.2, mort_rates, 0.4, label='Mortality Rate', color=COLORS['death'])
    ax.set_xticks(x)
    ax.set_xticklabels(ages, rotation=45, ha='right')
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title('Age-Specific Rates', fontsize=13, fontweight='600', pad=15)
    ax.legend(fontsize=9)
    _style_axis(ax)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_projection_chart(projection: Dict, temporal: Dict) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not projection['years']:
        ax.text(0.5, 0.5, 'No projection data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    # Historical
    hist_periods = temporal['periods']
    hist_births = temporal['birth_rates']
    hist_deaths = temporal['death_rates']
    
    # Projected
    proj_years = [str(y) for y in projection['years']]
    proj_births = projection['projected_births']
    proj_deaths = projection['projected_deaths']
    
    all_periods = hist_periods + proj_years
    all_births = hist_births + proj_births
    all_deaths = hist_deaths + proj_deaths
    
    ax.plot(hist_periods, hist_births, color=COLORS['birth'], linewidth=2, marker='o', label='Birth Rate (Historical)')
    ax.plot(hist_periods, hist_deaths, color=COLORS['death'], linewidth=2, marker='s', label='Death Rate (Historical)')
    ax.plot(proj_years, proj_births, color=COLORS['birth'], linewidth=2, linestyle='--', marker='o', alpha=0.6, label='Birth Rate (Projected)')
    ax.plot(proj_years, proj_deaths, color=COLORS['death'], linewidth=2, linestyle='--', marker='s', alpha=0.6, label='Death Rate (Projected)')
    
    ax.axvline(hist_periods[-1], color=COLORS['primary'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Rate (per 1,000)', fontsize=11)
    ax.set_title('Demographic Rate Projection', fontsize=13, fontweight='600', pad=15)
    ax.legend(fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_pyramid_chart(age_data: Dict) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fertility = age_data.get('fertility_by_age', [])
    if not fertility:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    ages = [a['age_group'] for a in fertility]
    pops = [a['population'] or 1000 for a in fertility]
    
    # Simulated male/female split
    male_pops = [-p * 0.49 for p in pops]
    female_pops = [p * 0.51 for p in pops]
    
    y = np.arange(len(ages))
    ax.barh(y, male_pops, height=0.8, color=COLORS['birth'], label='Male', alpha=0.8)
    ax.barh(y, female_pops, height=0.8, color=COLORS['death'], label='Female', alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(ages)
    ax.set_xlabel('Population', fontsize=11)
    ax.set_title('Population Pyramid', fontsize=13, fontweight='600', pad=15)
    ax.legend(fontsize=9)
    ax.axvline(0, color=COLORS['primary'], linewidth=1)
    _style_axis(ax)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(overall: Dict, temporal: Dict, regional: List[Dict], summary: Dict) -> List[Dict]:
    insights = []
    
    tfr = summary['tfr']
    if tfr < 1.0:
        insights.append({'title': 'Ultra-Low Fertility Crisis', 'description': f'TFR of {tfr:.2f} is critically low. Rapid population aging and decline expected.', 'status': 'warning'})
    elif tfr < 1.5:
        insights.append({'title': 'Very Low Fertility', 'description': f'TFR of {tfr:.2f} far below replacement level (2.1). Long-term demographic challenges ahead.', 'status': 'warning'})
    elif tfr < 2.1:
        insights.append({'title': 'Below Replacement Fertility', 'description': f'TFR of {tfr:.2f} below replacement. Population will decline without immigration.', 'status': 'neutral'})
    else:
        insights.append({'title': 'Healthy Fertility', 'description': f'TFR of {tfr:.2f} at or above replacement level.', 'status': 'positive'})
    
    natural_inc = summary['natural_increase']
    if natural_inc < 0:
        insights.append({'title': 'Population Natural Decline', 'description': f'Deaths exceed births by {abs(natural_inc):.1f}‰. Population shrinking naturally.', 'status': 'warning'})
    else:
        insights.append({'title': 'Natural Population Growth', 'description': f'Births exceed deaths by {natural_inc:.1f}‰.', 'status': 'positive'})
    
    if len(temporal['birth_rates']) >= 5:
        recent_trend = temporal['birth_rates'][-1] - temporal['birth_rates'][-5]
        if recent_trend < -2:
            insights.append({'title': 'Rapidly Declining Birth Rate', 'description': f'Birth rate dropped {abs(recent_trend):.1f}‰ over recent periods.', 'status': 'warning'})
    
    return insights


@router.post("/birth-mortality")
async def run_birth_mortality_analysis(request: BirthMortalityRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        
        if request.period_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Period column '{request.period_col}' not found")
        
        df = calculate_rates(df, request)
        overall = analyze_overall(df, request)
        temporal = analyze_temporal(df, request.period_col)
        regional = analyze_regional(df, request.region_col, request)
        age_data = analyze_age(df, request.age_group_col)
        projection = project_population(temporal)
        
        visualizations = {
            'rate_trends': create_rate_trends_chart(temporal),
            'regional_comparison': create_regional_chart(regional),
            'age_distribution': create_age_chart(age_data),
            'projection_chart': create_projection_chart(projection, temporal),
            'pyramid_chart': create_pyramid_chart(age_data)
        }
        
        # Determine fertility status
        tfr = overall['tfr'] or 1.0
        if tfr < 1.0: fertility_status = "Ultra-low fertility"
        elif tfr < 1.5: fertility_status = "Very low fertility"
        elif tfr < 2.1: fertility_status = "Below replacement"
        else: fertility_status = "At replacement"
        
        # Trend direction
        if len(temporal['birth_rates']) >= 3:
            recent = np.mean(temporal['birth_rates'][-3:])
            earlier = np.mean(temporal['birth_rates'][:3])
            trend = "Declining fertility" if recent < earlier * 0.95 else "Improving" if recent > earlier * 1.05 else "Stable"
        else:
            trend = "Insufficient data"
        
        highest = regional[0]['region'] if regional else 'N/A'
        lowest = regional[-1]['region'] if regional else 'N/A'
        
        summary = {
            'analysis_period': request.analysis_period,
            'total_population': overall['total_population'],
            'birth_rate': overall['crude_birth_rate'] or 0,
            'death_rate': overall['crude_death_rate'] or 0,
            'natural_increase': overall['natural_increase_rate'] or 0,
            'tfr': tfr,
            'fertility_status': fertility_status,
            'trend_direction': trend,
            'highest_birth_region': highest,
            'lowest_birth_region': lowest
        }
        
        insights = generate_insights(overall, temporal, regional, summary)
        
        return {
            'success': True,
            'overall_metrics': overall,
            'temporal_analysis': temporal,
            'regional_analysis': regional,
            'age_analysis': age_data,
            'projection': projection,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
