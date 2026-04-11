"""
Employment & Industry Structure Analysis Router for FastAPI
Sector Analysis, Structural Change, Regional Specialization
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
    'primary': '#48bb78', 'secondary': '#4299e1', 'tertiary': '#9f7aea', 'quaternary': '#ed8936',
    'growth': '#38a169', 'decline': '#e53e3e', 'neutral': '#718096',
}
SECTOR_COLORS = {'Primary': '#48bb78', 'Secondary': '#4299e1', 'Tertiary': '#9f7aea', 'Quaternary': '#ed8936'}
PALETTE = ['#4299e1', '#48bb78', '#9f7aea', '#ed8936', '#e53e3e', '#718096', '#38a169', '#4a5568']

router = APIRouter()


class IndustryRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: Optional[str] = None
    industry_col: str
    sector_col: Optional[str] = None
    employment_col: str
    region_col: Optional[str] = None
    value_added_col: Optional[str] = None
    productivity_col: Optional[str] = None
    wage_col: Optional[str] = None
    analysis_focus: str = "structure"
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


def classify_sector(industry_name: str) -> str:
    """Auto-classify industry into sector"""
    name = industry_name.lower()
    if any(k in name for k in ['agricult', 'farm', 'fish', 'mining', 'forestry', 'livestock']):
        return 'Primary'
    if any(k in name for k in ['manufactur', 'construct', 'mfg', 'factory', 'textile', 'chemical', 'auto', 'machine', 'electron']):
        return 'Secondary'
    if any(k in name for k in ['it', 'software', 'r&d', 'research', 'consult', 'tech']):
        return 'Quaternary'
    return 'Tertiary'


def analyze_overall(df: pd.DataFrame, industry_col: str, sector_col: str, employment_col: str, period_col: str) -> Dict:
    """Calculate overall industry metrics"""
    # Latest period data
    if period_col and period_col in df.columns:
        periods = sorted(df[period_col].unique())
        latest = df[df[period_col] == periods[-1]]
        prev = df[df[period_col] == periods[-2]] if len(periods) >= 2 else latest
    else:
        latest = df
        prev = df
    
    total_emp = latest[employment_col].sum()
    prev_emp = prev[employment_col].sum()
    yoy_change = total_emp - prev_emp
    yoy_rate = (yoy_change / prev_emp * 100) if prev_emp > 0 else 0
    
    n_industries = latest[industry_col].nunique()
    n_sectors = latest[sector_col].nunique() if sector_col and sector_col in latest.columns else 0
    
    # Sector shares
    sector_shares = latest.groupby(sector_col)[employment_col].sum() / total_emp * 100 if sector_col else pd.Series()
    primary = sector_shares.get('Primary', 0)
    secondary = sector_shares.get('Secondary', 0)
    tertiary = sector_shares.get('Tertiary', 0)
    
    # Largest and fastest growing
    industry_emp = latest.groupby(industry_col)[employment_col].sum().sort_values(ascending=False)
    largest = industry_emp.index[0] if len(industry_emp) > 0 else 'N/A'
    
    if period_col and period_col in df.columns and len(periods) >= 2:
        latest_by_ind = latest.groupby(industry_col)[employment_col].sum()
        prev_by_ind = prev.groupby(industry_col)[employment_col].sum()
        growth = ((latest_by_ind - prev_by_ind) / prev_by_ind.replace(0, np.nan) * 100).dropna()
        fastest = growth.idxmax() if len(growth) > 0 else 'N/A'
    else:
        fastest = 'N/A'
    
    return {
        'total_employment': _to_native(total_emp),
        'n_industries': _to_native(n_industries),
        'n_sectors': _to_native(n_sectors),
        'primary_share': _to_native(primary),
        'secondary_share': _to_native(secondary),
        'tertiary_share': _to_native(tertiary),
        'yoy_employment_change': _to_native(yoy_change),
        'yoy_growth_rate': _to_native(yoy_rate),
        'largest_industry': str(largest),
        'fastest_growing': str(fastest)
    }


def analyze_sectors(df: pd.DataFrame, sector_col: str, employment_col: str, industry_col: str, period_col: str) -> List[Dict]:
    """Analyze by sector"""
    if not sector_col or sector_col not in df.columns:
        return []
    
    results = []
    if period_col and period_col in df.columns:
        periods = sorted(df[period_col].unique())
        latest = df[df[period_col] == periods[-1]]
        prev = df[df[period_col] == periods[-2]] if len(periods) >= 2 else latest
    else:
        latest = df
        prev = df
    
    total_emp = latest[employment_col].sum()
    
    for sector in latest[sector_col].unique():
        sdf = latest[latest[sector_col] == sector]
        sdf_prev = prev[prev[sector_col] == sector]
        
        emp = sdf[employment_col].sum()
        emp_prev = sdf_prev[employment_col].sum()
        
        results.append({
            'sector': str(sector),
            'employment': _to_native(emp),
            'employment_share': _to_native(emp / total_emp * 100 if total_emp > 0 else 0),
            'n_industries': _to_native(sdf[industry_col].nunique()),
            'yoy_change': _to_native(emp - emp_prev),
            'growth_rate': _to_native((emp - emp_prev) / emp_prev * 100 if emp_prev > 0 else 0)
        })
    
    return sorted(results, key=lambda x: x['employment'] or 0, reverse=True)


def analyze_industries(df: pd.DataFrame, industry_col: str, sector_col: str, employment_col: str, 
                       value_col: str, prod_col: str, wage_col: str, period_col: str) -> List[Dict]:
    """Analyze by industry"""
    results = []
    
    if period_col and period_col in df.columns:
        periods = sorted(df[period_col].unique())
        latest = df[df[period_col] == periods[-1]]
        prev = df[df[period_col] == periods[-2]] if len(periods) >= 2 else latest
    else:
        latest = df
        prev = df
    
    total_emp = latest[employment_col].sum()
    
    for industry in latest[industry_col].unique():
        idf = latest[latest[industry_col] == industry]
        idf_prev = prev[prev[industry_col] == industry]
        
        emp = idf[employment_col].sum()
        emp_prev = idf_prev[employment_col].sum()
        yoy = emp - emp_prev
        rate = (yoy / emp_prev * 100) if emp_prev > 0 else 0
        
        sector = idf[sector_col].iloc[0] if sector_col and sector_col in idf.columns else classify_sector(str(industry))
        value_added = idf[value_col].sum() if value_col and value_col in idf.columns else None
        productivity = idf[prod_col].mean() if prod_col and prod_col in idf.columns else None
        avg_wage = idf[wage_col].mean() if wage_col and wage_col in idf.columns else None
        
        results.append({
            'industry': str(industry),
            'sector': str(sector),
            'employment': _to_native(emp),
            'employment_share': _to_native(emp / total_emp * 100 if total_emp > 0 else 0),
            'yoy_change': _to_native(yoy),
            'yoy_growth_rate': _to_native(rate),
            'value_added': _to_native(value_added),
            'productivity': _to_native(productivity),
            'avg_wage': _to_native(avg_wage)
        })
    
    return sorted(results, key=lambda x: x['employment'] or 0, reverse=True)


def analyze_temporal(df: pd.DataFrame, period_col: str, sector_col: str, employment_col: str) -> Dict:
    """Analyze trends over time"""
    if not period_col or period_col not in df.columns:
        return {'periods': [], 'total_employment': [], 'sector_trends': {}}
    
    periods = sorted(df[period_col].unique())
    total_emp = []
    sector_trends = {}
    
    for period in periods:
        pdf = df[df[period_col] == period]
        total_emp.append(_to_native(pdf[employment_col].sum()))
        
        if sector_col and sector_col in pdf.columns:
            for sector in pdf[sector_col].unique():
                if str(sector) not in sector_trends:
                    sector_trends[str(sector)] = []
                sector_trends[str(sector)].append(_to_native(pdf[pdf[sector_col] == sector][employment_col].sum()))
    
    return {
        'periods': [str(p) for p in periods],
        'total_employment': total_emp,
        'sector_trends': sector_trends
    }


def analyze_regional(df: pd.DataFrame, region_col: str, industry_col: str, employment_col: str, period_col: str) -> List[Dict]:
    """Analyze regional specialization"""
    if not region_col or region_col not in df.columns:
        return []
    
    if period_col and period_col in df.columns:
        periods = sorted(df[period_col].unique())
        latest = df[df[period_col] == periods[-1]]
    else:
        latest = df
    
    results = []
    total_by_industry = latest.groupby(industry_col)[employment_col].sum()
    national_total = latest[employment_col].sum()
    
    for region in latest[region_col].unique():
        rdf = latest[latest[region_col] == region]
        emp = rdf[employment_col].sum()
        
        # Dominant industry
        ind_emp = rdf.groupby(industry_col)[employment_col].sum()
        dominant = ind_emp.idxmax() if len(ind_emp) > 0 else 'N/A'
        
        # Specialization index (simplified LQ)
        if len(ind_emp) > 0 and national_total > 0:
            local_share = ind_emp.max() / emp if emp > 0 else 0
            national_share = total_by_industry[dominant] / national_total if dominant in total_by_industry else 0
            lq = local_share / national_share if national_share > 0 else 1
        else:
            lq = 1
        
        results.append({
            'region': str(region),
            'employment': _to_native(emp),
            'dominant_industry': str(dominant),
            'specialization_index': _to_native(lq)
        })
    
    return sorted(results, key=lambda x: x['employment'] or 0, reverse=True)


def analyze_structural_change(df: pd.DataFrame, period_col: str, sector_col: str, industry_col: str, employment_col: str) -> Dict:
    """Analyze structural changes between periods"""
    if not period_col or period_col not in df.columns:
        return {'from_period': '', 'to_period': '', 'sector_shifts': [], 'growing_industries': [], 'declining_industries': []}
    
    periods = sorted(df[period_col].unique())
    if len(periods) < 2:
        return {'from_period': str(periods[0]) if periods else '', 'to_period': '', 'sector_shifts': [], 'growing_industries': [], 'declining_industries': []}
    
    first = df[df[period_col] == periods[0]]
    last = df[df[period_col] == periods[-1]]
    
    # Sector shifts
    sector_shifts = []
    if sector_col and sector_col in df.columns:
        first_total = first[employment_col].sum()
        last_total = last[employment_col].sum()
        
        for sector in df[sector_col].unique():
            first_share = first[first[sector_col] == sector][employment_col].sum() / first_total * 100 if first_total > 0 else 0
            last_share = last[last[sector_col] == sector][employment_col].sum() / last_total * 100 if last_total > 0 else 0
            sector_shifts.append({'sector': str(sector), 'change': _to_native(last_share - first_share)})
    
    # Growing/declining industries
    first_by_ind = first.groupby(industry_col)[employment_col].sum()
    last_by_ind = last.groupby(industry_col)[employment_col].sum()
    growth = ((last_by_ind - first_by_ind) / first_by_ind.replace(0, np.nan) * 100).dropna().sort_values(ascending=False)
    
    growing = [str(i) for i in growth.head(5).index.tolist()]
    declining = [str(i) for i in growth.tail(5).index.tolist()]
    
    return {
        'from_period': str(periods[0]),
        'to_period': str(periods[-1]),
        'sector_shifts': sector_shifts,
        'growing_industries': growing,
        'declining_industries': declining
    }
# ============================================================
# Visualization Functions (Part 2)
# ============================================================

def create_sector_composition_chart(sectors: List[Dict]) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if not sectors:
        ax1.text(0.5, 0.5, 'No sector data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    names = [s['sector'] for s in sectors]
    shares = [s['employment_share'] or 0 for s in sectors]
    colors = [SECTOR_COLORS.get(s, COLORS['neutral']) for s in names]
    
    # Pie chart
    ax1.pie(shares, labels=names, autopct='%1.1f%%', colors=colors, startangle=90, explode=[0.02]*len(names))
    ax1.set_title('Employment by Sector', fontsize=13, fontweight='600', pad=15)
    
    # Bar chart with employment
    employment = [s['employment'] / 1000000 for s in sectors]
    ax2.barh(names, employment, color=colors, edgecolor='white', height=0.6)
    ax2.set_xlabel('Employment (Millions)', fontsize=11)
    ax2.set_title('Sector Employment', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_industry_ranking_chart(industries: List[Dict]) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    if not industries:
        ax1.text(0.5, 0.5, 'No industry data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    top = industries[:15]
    names = [i['industry'][:20] for i in top]
    emp = [i['employment'] / 1000000 for i in top]
    sectors = [i['sector'] for i in top]
    colors = [SECTOR_COLORS.get(s, COLORS['neutral']) for s in sectors]
    
    ax1.barh(names, emp, color=colors, edgecolor='white', height=0.7)
    ax1.set_xlabel('Employment (Millions)', fontsize=11)
    ax1.set_title('Top Industries by Employment', fontsize=13, fontweight='600', pad=15)
    ax1.invert_yaxis()
    _style_axis(ax1)
    
    # Growth rates
    growth = [i['yoy_growth_rate'] or 0 for i in top]
    colors2 = [COLORS['growth'] if g >= 0 else COLORS['decline'] for g in growth]
    ax2.barh(names, growth, color=colors2, edgecolor='white', height=0.7)
    ax2.axvline(0, color=COLORS['neutral'], linestyle='-', linewidth=1)
    ax2.set_xlabel('YoY Growth Rate (%)', fontsize=11)
    ax2.set_title('Employment Growth by Industry', fontsize=13, fontweight='600', pad=15)
    ax2.invert_yaxis()
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
    
    total_emp = [e / 1000000 for e in temporal['total_employment']]
    ax1.plot(periods, total_emp, color=COLORS['neutral'], linewidth=2, marker='o')
    ax1.fill_between(periods, total_emp, alpha=0.3, color=COLORS['neutral'])
    ax1.set_ylabel('Total Employment (Millions)', fontsize=11)
    ax1.set_title('Total Employment Trend', fontsize=13, fontweight='600', pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    # Sector trends (stacked area)
    sector_trends = temporal['sector_trends']
    if sector_trends:
        bottom = np.zeros(len(periods))
        for sector, values in sector_trends.items():
            values_m = [v / 1000000 if v else 0 for v in values]
            ax2.fill_between(periods, bottom, bottom + np.array(values_m), 
                           label=sector, color=SECTOR_COLORS.get(sector, COLORS['neutral']), alpha=0.7)
            bottom += np.array(values_m)
        ax2.set_ylabel('Employment (Millions)', fontsize=11)
        ax2.set_title('Employment by Sector Over Time', fontsize=13, fontweight='600', pad=15)
        ax2.legend(fontsize=9, loc='upper left')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_structural_change_chart(change: Dict, sectors: List[Dict]) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    shifts = change['sector_shifts']
    if shifts:
        names = [s['sector'] for s in shifts]
        changes = [s['change'] or 0 for s in shifts]
        colors = [COLORS['growth'] if c > 0 else COLORS['decline'] for c in changes]
        
        ax1.barh(names, changes, color=colors, edgecolor='white', height=0.6)
        ax1.axvline(0, color=COLORS['neutral'], linestyle='-', linewidth=1)
        ax1.set_xlabel('Change in Employment Share (%p)', fontsize=11)
        ax1.set_title(f'Structural Shift: {change["from_period"]} → {change["to_period"]}', fontsize=13, fontweight='600', pad=15)
        _style_axis(ax1)
    else:
        ax1.text(0.5, 0.5, 'No shift data', ha='center', va='center', transform=ax1.transAxes)
    
    # Growing vs Declining industries
    growing = change['growing_industries'][:5]
    declining = change['declining_industries'][:5]
    
    if growing or declining:
        all_ind = growing + declining
        vals = list(range(len(growing), 0, -1)) + list(range(-1, -len(declining)-1, -1))
        colors = [COLORS['growth']]*len(growing) + [COLORS['decline']]*len(declining)
        
        ax2.barh([i[:15] for i in all_ind], vals, color=colors, edgecolor='white', height=0.6)
        ax2.axvline(0, color=COLORS['neutral'], linestyle='-', linewidth=1)
        ax2.set_xlabel('Growth Ranking', fontsize=11)
        ax2.set_title('Growing vs Declining Industries', fontsize=13, fontweight='600', pad=15)
        _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_regional_chart(regional: List[Dict]) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if not regional:
        ax1.text(0.5, 0.5, 'No regional data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    top = regional[:12]
    regions = [r['region'] for r in top]
    emp = [r['employment'] / 1000000 for r in top]
    
    ax1.barh(regions, emp, color=COLORS['neutral'], edgecolor='white', height=0.7)
    ax1.set_xlabel('Employment (Millions)', fontsize=11)
    ax1.set_title('Employment by Region', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax1)
    
    # Specialization index
    lq = [r['specialization_index'] or 1 for r in top]
    colors = [COLORS['growth'] if l > 1.2 else COLORS['decline'] if l < 0.8 else COLORS['neutral'] for l in lq]
    ax2.barh(regions, lq, color=colors, edgecolor='white', height=0.7)
    ax2.axvline(1.0, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='National Avg')
    ax2.set_xlabel('Specialization Index (LQ)', fontsize=11)
    ax2.set_title('Regional Specialization', fontsize=13, fontweight='600', pad=15)
    ax2.legend(fontsize=9)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(overall: Dict, sectors: List[Dict], industries: List[Dict], change: Dict) -> List[Dict]:
    insights = []
    
    # Dominant sector
    if sectors:
        dominant = sectors[0]
        insights.append({
            'title': f'{dominant["sector"]} Sector Dominates',
            'description': f'{dominant["sector"]} accounts for {dominant["employment_share"]:.1f}% of employment ({dominant["employment"]:,} workers).',
            'status': 'neutral'
        })
    
    # Structural change
    if change['sector_shifts']:
        growing_sectors = [s for s in change['sector_shifts'] if (s['change'] or 0) > 1]
        if growing_sectors:
            top_grow = max(growing_sectors, key=lambda x: x['change'] or 0)
            insights.append({
                'title': 'Servicification Trend',
                'description': f'{top_grow["sector"]} share increased by {top_grow["change"]:.1f}%p. Economy shifting toward services.',
                'status': 'positive'
            })
    
    # Fast growing industries
    if overall['fastest_growing'] != 'N/A':
        insights.append({
            'title': f'{overall["fastest_growing"]} Leading Growth',
            'description': f'Fastest growing industry. Consider workforce development in this area.',
            'status': 'positive'
        })
    
    # Declining industries
    if change['declining_industries']:
        insights.append({
            'title': 'Industries in Decline',
            'description': f'{", ".join(change["declining_industries"][:3])} showing employment decline. Transition support may be needed.',
            'status': 'warning'
        })
    
    return insights


@router.post("/industry-structure")
async def run_industry_analysis(request: IndustryRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        
        if request.industry_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Industry column '{request.industry_col}' not found")
        if request.employment_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Employment column '{request.employment_col}' not found")
        
        df[request.employment_col] = pd.to_numeric(df[request.employment_col], errors='coerce').fillna(0)
        
        # Auto-classify sectors if not provided
        sector_col = request.sector_col
        if not sector_col or sector_col not in df.columns:
            df['_sector'] = df[request.industry_col].apply(classify_sector)
            sector_col = '_sector'
        
        overall = analyze_overall(df, request.industry_col, sector_col, request.employment_col, request.period_col)
        sectors = analyze_sectors(df, sector_col, request.employment_col, request.industry_col, request.period_col)
        industries = analyze_industries(df, request.industry_col, sector_col, request.employment_col,
                                       request.value_added_col, request.productivity_col, request.wage_col, request.period_col)
        temporal = analyze_temporal(df, request.period_col, sector_col, request.employment_col)
        regional = analyze_regional(df, request.region_col, request.industry_col, request.employment_col, request.period_col)
        change = analyze_structural_change(df, request.period_col, sector_col, request.industry_col, request.employment_col)
        
        visualizations = {
            'sector_composition': create_sector_composition_chart(sectors),
            'industry_ranking': create_industry_ranking_chart(industries),
            'trend_chart': create_trend_chart(temporal),
            'structural_change': create_structural_change_chart(change, sectors),
            'regional_map': create_regional_chart(regional)
        }
        
        # Dominant sector
        dominant = sectors[0] if sectors else {'sector': 'N/A', 'employment_share': 0}
        
        # Structural trend
        if change['sector_shifts']:
            tertiary_shift = next((s['change'] for s in change['sector_shifts'] if s['sector'] in ['Tertiary', 'Quaternary']), 0) or 0
            trend = "Servicification ongoing" if tertiary_shift > 0.5 else "Stable structure" if abs(tertiary_shift) < 0.5 else "Manufacturing recovery"
        else:
            trend = "N/A"
        
        # Employment trend
        emp_trend = "Growing" if overall['yoy_growth_rate'] > 1 else "Declining" if overall['yoy_growth_rate'] < -1 else "Stable"
        
        # Top declining
        top_declining = change['declining_industries'][0] if change['declining_industries'] else 'N/A'
        
        summary = {
            'analysis_period': request.analysis_period,
            'total_employment': overall['total_employment'],
            'dominant_sector': dominant['sector'],
            'dominant_sector_share': dominant['employment_share'] or 0,
            'structural_trend': trend,
            'employment_trend': emp_trend,
            'top_growing_industry': overall['fastest_growing'],
            'top_declining_industry': top_declining
        }
        
        insights = generate_insights(overall, sectors, industries, change)
        
        return {
            'success': True,
            'overall_metrics': overall,
            'sector_analysis': sectors,
            'industry_analysis': industries,
            'temporal_analysis': temporal,
            'regional_analysis': regional,
            'structural_change': change,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Industry structure analysis failed: {str(e)}")
