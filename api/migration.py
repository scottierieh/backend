"""
Regional Population Migration Analysis Router for FastAPI
Migration Flows, Net Balance, Corridors, Demographics
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
    'in': '#48bb78', 'out': '#e53e3e', 'net_pos': '#38a169', 'net_neg': '#c53030',
    'primary': '#4a5568', 'secondary': '#718096',
}
PALETTE = ['#4a5568', '#718096', '#a0aec0', '#4299e1', '#48bb78', '#ed8936', '#e53e3e', '#9f7aea']

router = APIRouter()


class MigrationRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: Optional[str] = None
    origin_col: str
    destination_col: str
    migrants_col: str
    population_col: Optional[str] = None
    age_group_col: Optional[str] = None
    reason_col: Optional[str] = None
    analysis_focus: str = "balance"
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


def analyze_regional(df: pd.DataFrame, origin_col: str, dest_col: str, migrants_col: str, pop_col: Optional[str]) -> List[Dict]:
    """Analyze migration balance by region"""
    # Calculate in-migration
    in_mig = df.groupby(dest_col)[migrants_col].sum().reset_index()
    in_mig.columns = ['region', 'in_migration']
    
    # Calculate out-migration
    out_mig = df.groupby(origin_col)[migrants_col].sum().reset_index()
    out_mig.columns = ['region', 'out_migration']
    
    # Merge
    regional = in_mig.merge(out_mig, on='region', how='outer').fillna(0)
    regional['net_migration'] = regional['in_migration'] - regional['out_migration']
    regional['gross_migration'] = regional['in_migration'] + regional['out_migration']
    
    # Population for rates
    if pop_col and pop_col in df.columns:
        pop_data = df.groupby(origin_col)[pop_col].first().reset_index()
        pop_data.columns = ['region', 'population']
        regional = regional.merge(pop_data, on='region', how='left')
    else:
        regional['population'] = 1000000  # Default
    
    regional['net_migration_rate'] = (regional['net_migration'] / regional['population']) * 1000
    regional['migration_effectiveness'] = np.abs(regional['net_migration']) / regional['gross_migration'].replace(0, 1) * 100
    
    results = []
    for _, row in regional.iterrows():
        results.append({
            'region': str(row['region']),
            'population': _to_native(row['population']),
            'in_migration': _to_native(row['in_migration']),
            'out_migration': _to_native(row['out_migration']),
            'net_migration': _to_native(row['net_migration']),
            'net_migration_rate': _to_native(row['net_migration_rate']),
            'gross_migration': _to_native(row['gross_migration']),
            'migration_effectiveness': _to_native(row['migration_effectiveness'])
        })
    
    return sorted(results, key=lambda x: x['net_migration'], reverse=True)


def analyze_flows(df: pd.DataFrame, origin_col: str, dest_col: str, migrants_col: str) -> Dict:
    """Analyze migration flows between regions"""
    flows = df.groupby([origin_col, dest_col])[migrants_col].sum().reset_index()
    flows.columns = ['origin', 'destination', 'flow_count']
    flows = flows.sort_values('flow_count', ascending=False)
    
    total = flows['flow_count'].sum()
    flows['flow_share'] = flows['flow_count'] / total
    
    top_flows = []
    for _, row in flows.head(20).iterrows():
        top_flows.append({
            'origin': str(row['origin']),
            'destination': str(row['destination']),
            'flow_count': _to_native(row['flow_count']),
            'flow_share': _to_native(row['flow_share'])
        })
    
    # Flow matrix
    matrix = df.pivot_table(values=migrants_col, index=origin_col, columns=dest_col, aggfunc='sum', fill_value=0)
    flow_matrix = {str(idx): {str(col): _to_native(matrix.loc[idx, col]) for col in matrix.columns} for idx in matrix.index}
    
    return {'top_flows': top_flows, 'flow_matrix': flow_matrix}


def analyze_temporal(df: pd.DataFrame, period_col: str, origin_col: str, dest_col: str, migrants_col: str) -> Dict:
    """Analyze migration trends over time"""
    if not period_col or period_col not in df.columns:
        return {'periods': [], 'total_migration': [], 'net_migration_trend': {}}
    
    temporal = df.groupby(period_col)[migrants_col].sum().reset_index()
    temporal = temporal.sort_values(period_col)
    
    periods = [str(p) for p in temporal[period_col].tolist()]
    total_mig = [_to_native(x) for x in temporal[migrants_col].tolist()]
    
    # Net migration trend by region
    regions = list(set(df[origin_col].unique()) | set(df[dest_col].unique()))
    net_trend = {}
    
    for region in regions[:10]:  # Top 10 regions
        region_trend = []
        for period in df[period_col].unique():
            period_df = df[df[period_col] == period]
            in_mig = period_df[period_df[dest_col] == region][migrants_col].sum()
            out_mig = period_df[period_df[origin_col] == region][migrants_col].sum()
            region_trend.append(_to_native(in_mig - out_mig))
        net_trend[str(region)] = region_trend
    
    return {'periods': periods, 'total_migration': total_mig, 'net_migration_trend': net_trend}


def analyze_demographics(df: pd.DataFrame, age_col: Optional[str], reason_col: Optional[str], migrants_col: str) -> Dict:
    """Analyze migration demographics"""
    by_age = []
    by_reason = []
    total = df[migrants_col].sum()
    
    if age_col and age_col in df.columns:
        age_data = df.groupby(age_col)[migrants_col].sum().reset_index()
        for _, row in age_data.iterrows():
            by_age.append({
                'age_group': str(row[age_col]),
                'migration_count': _to_native(row[migrants_col]),
                'share': _to_native(row[migrants_col] / total)
            })
    
    if reason_col and reason_col in df.columns:
        reason_data = df.groupby(reason_col)[migrants_col].sum().reset_index()
        for _, row in reason_data.iterrows():
            by_reason.append({
                'reason': str(row[reason_col]),
                'count': _to_native(row[migrants_col]),
                'share': _to_native(row[migrants_col] / total)
            })
    
    return {'by_age': by_age, 'by_reason': by_reason}
# ============================================================
# Visualization Functions (Part 2)
# ============================================================

def create_regional_balance_chart(regional: List[Dict]) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if not regional:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    sorted_reg = sorted(regional, key=lambda x: x['net_migration'])[:15]
    regions = [r['region'] for r in sorted_reg]
    net_mig = [r['net_migration'] for r in sorted_reg]
    colors = [COLORS['net_pos'] if n >= 0 else COLORS['net_neg'] for n in net_mig]
    
    ax1.barh(regions, net_mig, color=colors, edgecolor='white', height=0.7)
    ax1.axvline(0, color=COLORS['primary'], linestyle='-', linewidth=1)
    ax1.set_xlabel('Net Migration', fontsize=11)
    ax1.set_title('Net Migration by Region', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax1)
    
    # In vs Out comparison
    sorted_gross = sorted(regional, key=lambda x: x['gross_migration'], reverse=True)[:10]
    regions2 = [r['region'] for r in sorted_gross]
    in_mig = [r['in_migration'] for r in sorted_gross]
    out_mig = [-r['out_migration'] for r in sorted_gross]
    
    y = np.arange(len(regions2))
    ax2.barh(y, in_mig, height=0.4, label='In-Migration', color=COLORS['in'])
    ax2.barh(y, out_mig, height=0.4, label='Out-Migration', color=COLORS['out'])
    ax2.set_yticks(y)
    ax2.set_yticklabels(regions2)
    ax2.axvline(0, color=COLORS['primary'], linestyle='-', linewidth=1)
    ax2.set_xlabel('Migration Count', fontsize=11)
    ax2.set_title('In vs Out Migration', fontsize=13, fontweight='600', pad=15)
    ax2.legend(fontsize=9)
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_flow_chart(flows: Dict) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_flows = flows['top_flows'][:15]
    if not top_flows:
        ax.text(0.5, 0.5, 'No flow data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    labels = [f"{f['origin']} → {f['destination']}" for f in top_flows]
    values = [f['flow_count'] for f in top_flows]
    
    bars = ax.barh(labels, values, color=PALETTE[0], edgecolor='white', height=0.7)
    ax.set_xlabel('Number of Migrants', fontsize=11)
    ax.set_title('Top Migration Corridors', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:,.0f}', va='center', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_temporal_chart(temporal: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    periods = temporal['periods']
    if not periods:
        ax1.text(0.5, 0.5, 'No temporal data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    ax1.bar(periods, temporal['total_migration'], color=COLORS['primary'], edgecolor='white')
    ax1.set_ylabel('Total Migration', fontsize=11)
    ax1.set_title('Total Migration Over Time', fontsize=13, fontweight='600', pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax1)
    
    net_trend = temporal['net_migration_trend']
    for i, (region, trend) in enumerate(list(net_trend.items())[:5]):
        ax2.plot(periods[:len(trend)], trend, marker='o', linewidth=2, label=region, color=PALETTE[i])
    ax2.axhline(0, color=COLORS['primary'], linestyle='--', linewidth=1)
    ax2.set_ylabel('Net Migration', fontsize=11)
    ax2.set_title('Net Migration Trend by Region', fontsize=13, fontweight='600', pad=15)
    ax2.legend(fontsize=8, loc='best')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    _style_axis(ax2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_corridors_chart(flows: Dict) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    top = flows['top_flows'][:10]
    if not top:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return _fig_to_base64(fig)
    
    # Chord-like visualization (simplified)
    origins = list(set(f['origin'] for f in top))
    dests = list(set(f['destination'] for f in top))
    all_regions = list(set(origins) | set(dests))
    
    n = len(all_regions)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = {r: (np.cos(a), np.sin(a)) for r, a in zip(all_regions, angles)}
    
    for region in all_regions:
        x, y = pos[region]
        ax.scatter(x, y, s=200, c=COLORS['primary'], zorder=3)
        ax.annotate(region, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    for f in top:
        x1, y1 = pos[f['origin']]
        x2, y2 = pos[f['destination']]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1+f['flow_share']*10, alpha=0.6))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Migration Flow Network', fontsize=13, fontweight='600', pad=15)
    
    return _fig_to_base64(fig)


def create_demographic_chart(demographics: Dict) -> str:
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    by_age = demographics['by_age']
    by_reason = demographics['by_reason']
    
    if by_age:
        ages = [a['age_group'] for a in by_age]
        shares = [a['share'] * 100 for a in by_age]
        ax1.bar(ages, shares, color=PALETTE[:len(ages)], edgecolor='white')
        ax1.set_ylabel('Share (%)', fontsize=11)
        ax1.set_title('Migration by Age Group', fontsize=13, fontweight='600', pad=15)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        _style_axis(ax1)
    else:
        ax1.text(0.5, 0.5, 'No age data', ha='center', va='center', transform=ax1.transAxes)
    
    if by_reason:
        reasons = [r['reason'] for r in by_reason]
        shares = [r['share'] * 100 for r in by_reason]
        ax2.pie(shares, labels=reasons, autopct='%1.1f%%', colors=PALETTE[:len(reasons)], startangle=90)
        ax2.set_title('Migration by Reason', fontsize=13, fontweight='600', pad=15)
    else:
        ax2.text(0.5, 0.5, 'No reason data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(regional: List[Dict], flows: Dict, demographics: Dict, overall: Dict) -> List[Dict]:
    insights = []
    
    gainers = [r for r in regional if r['net_migration'] > 0]
    losers = [r for r in regional if r['net_migration'] < 0]
    
    if len(losers) > len(gainers):
        insights.append({'title': 'Population Concentration', 'description': f'{len(losers)} regions losing population vs {len(gainers)} gaining. Migration concentrated to fewer areas.', 'status': 'warning'})
    else:
        insights.append({'title': 'Balanced Migration', 'description': f'{len(gainers)} net gainers vs {len(losers)} net losers.', 'status': 'neutral'})
    
    if flows['top_flows']:
        top = flows['top_flows'][0]
        insights.append({'title': 'Dominant Corridor', 'description': f'{top["origin"]} → {top["destination"]} is the largest flow ({top["flow_share"]*100:.1f}% of total).', 'status': 'neutral'})
    
    if demographics['by_age']:
        top_age = max(demographics['by_age'], key=lambda x: x['share'])
        insights.append({'title': 'Age-Selective Migration', 'description': f'{top_age["age_group"]} age group dominates migration ({top_age["share"]*100:.1f}%).', 'status': 'neutral'})
    
    return insights


@router.post("/migration")
async def run_migration_analysis(request: MigrationRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        
        if request.origin_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Origin column '{request.origin_col}' not found")
        if request.destination_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Destination column '{request.destination_col}' not found")
        if request.migrants_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Migrants column '{request.migrants_col}' not found")
        
        df[request.migrants_col] = pd.to_numeric(df[request.migrants_col], errors='coerce').fillna(0)
        
        regional = analyze_regional(df, request.origin_col, request.destination_col, request.migrants_col, request.population_col)
        flows = analyze_flows(df, request.origin_col, request.destination_col, request.migrants_col)
        temporal = analyze_temporal(df, request.period_col, request.origin_col, request.destination_col, request.migrants_col)
        demographics = analyze_demographics(df, request.age_group_col, request.reason_col, request.migrants_col)
        
        total_migrants = df[request.migrants_col].sum()
        gainers = [r for r in regional if r['net_migration'] > 0]
        losers = [r for r in regional if r['net_migration'] < 0]
        
        overall = {
            'total_migrants': _to_native(total_migrants),
            'total_in_migration': _to_native(sum(r['in_migration'] for r in regional)),
            'total_out_migration': _to_native(sum(r['out_migration'] for r in regional)),
            'net_gainers': len(gainers),
            'net_losers': len(losers),
            'avg_migration_rate': _to_native(np.mean([r['net_migration_rate'] for r in regional]) if regional else 0)
        }
        
        visualizations = {
            'regional_balance': create_regional_balance_chart(regional),
            'flow_map': create_flow_chart(flows),
            'temporal_trend': create_temporal_chart(temporal),
            'top_corridors': create_corridors_chart(flows),
            'demographic_chart': create_demographic_chart(demographics)
        }
        
        top_gainer = regional[0] if regional else {'region': 'N/A', 'net_migration': 0}
        top_loser = regional[-1] if regional else {'region': 'N/A', 'net_migration': 0}
        top_flow = flows['top_flows'][0] if flows['top_flows'] else {'origin': 'N/A', 'destination': 'N/A'}
        
        # Urbanization trend
        metro_regions = ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Gwangju', 'Daejeon', 'Ulsan', 'Gyeonggi', 'Sejong']
        metro_net = sum(r['net_migration'] for r in regional if r['region'] in metro_regions)
        trend = "Metropolitan concentration" if metro_net > 0 else "Counter-urbanization" if metro_net < 0 else "Balanced"
        
        summary = {
            'analysis_period': request.analysis_period,
            'total_migrants': _to_native(total_migrants),
            'n_regions': len(regional),
            'top_gainer': top_gainer['region'],
            'top_gainer_net': _to_native(top_gainer['net_migration']),
            'top_loser': top_loser['region'],
            'top_loser_net': _to_native(top_loser['net_migration']),
            'largest_flow_origin': top_flow['origin'],
            'largest_flow_dest': top_flow['destination'],
            'urbanization_trend': trend
        }
        
        insights = generate_insights(regional, flows, demographics, overall)
        
        return {
            'success': True,
            'overall_metrics': overall,
            'regional_analysis': regional,
            'flow_analysis': flows,
            'temporal_analysis': temporal,
            'demographic_analysis': demographics,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration analysis failed: {str(e)}")
