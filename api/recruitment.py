"""
Recruitment Funnel Analysis Router for FastAPI
Hiring Pipeline, Conversion Rates, Time-to-Hire, Source Effectiveness
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class RecruitmentRequest(BaseModel):
    data: List[Dict[str, Any]]
    candidate_id_col: Optional[str] = None
    stage_col: Optional[str] = None  # Current stage
    source_col: Optional[str] = None  # Recruitment source
    position_col: Optional[str] = None  # Job position
    department_col: Optional[str] = None
    apply_date_col: Optional[str] = None
    hire_date_col: Optional[str] = None
    status_col: Optional[str] = None  # Hired/Rejected/Withdrawn
    stage_order: Optional[List[str]] = None  # Custom stage order


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
    if isinstance(obj, (pd.Timestamp, datetime)):
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


def analyze_funnel_stages(df: pd.DataFrame, stage_col: str,
                          stage_order: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyze recruitment funnel by stages"""
    
    # Get stage counts
    stage_counts = df[stage_col].value_counts()
    
    # Define default stage order if not provided
    if stage_order is None:
        default_order = ['Applied', 'Screening', 'Phone Interview', 'Technical Interview', 
                        'Onsite Interview', 'Final Interview', 'Offer', 'Hired']
        # Filter to existing stages
        stage_order = [s for s in default_order if s in stage_counts.index]
        # Add any remaining stages
        for s in stage_counts.index:
            if s not in stage_order:
                stage_order.append(s)
    
    # Calculate funnel metrics
    funnel_data = []
    prev_count = None
    
    for i, stage in enumerate(stage_order):
        count = stage_counts.get(stage, 0)
        
        if prev_count is not None and prev_count > 0:
            conversion_rate = count / prev_count * 100
            dropoff_rate = (prev_count - count) / prev_count * 100
        else:
            conversion_rate = 100.0
            dropoff_rate = 0.0
        
        total_count = stage_counts.get(stage_order[0], 1)
        cumulative_rate = count / total_count * 100 if total_count > 0 else 0
        
        funnel_data.append({
            'stage': stage,
            'stage_order': i + 1,
            'count': int(count),
            'conversion_rate': _to_native_type(conversion_rate),
            'dropoff_rate': _to_native_type(dropoff_rate),
            'cumulative_rate': _to_native_type(cumulative_rate)
        })
        
        prev_count = count
    
    # Overall metrics
    total_applicants = stage_counts.get(stage_order[0], 0) if stage_order else 0
    total_hired = stage_counts.get('Hired', stage_counts.get(stage_order[-1], 0)) if stage_order else 0
    overall_conversion = total_hired / total_applicants * 100 if total_applicants > 0 else 0
    
    # Find biggest dropoff
    biggest_dropoff = max(funnel_data, key=lambda x: x['dropoff_rate']) if funnel_data else None
    
    return {
        'stages': funnel_data,
        'total_applicants': int(total_applicants),
        'total_hired': int(total_hired),
        'overall_conversion_rate': _to_native_type(overall_conversion),
        'biggest_dropoff_stage': biggest_dropoff['stage'] if biggest_dropoff else None,
        'biggest_dropoff_rate': biggest_dropoff['dropoff_rate'] if biggest_dropoff else None
    }


def analyze_by_source(df: pd.DataFrame, source_col: str, 
                      status_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Analyze recruitment effectiveness by source"""
    
    source_stats = []
    
    for source in df[source_col].unique():
        source_df = df[df[source_col] == source]
        total = len(source_df)
        
        # Calculate hired if status column exists
        if status_col and status_col in df.columns:
            hired = len(source_df[source_df[status_col].str.lower().isin(['hired', 'accepted', 'yes', '1'])])
            conversion_rate = hired / total * 100 if total > 0 else 0
        else:
            hired = None
            conversion_rate = None
        
        source_stats.append({
            'source': str(source),
            'total_candidates': int(total),
            'hired': int(hired) if hired is not None else None,
            'conversion_rate': _to_native_type(conversion_rate),
            'pct_of_total': _to_native_type(total / len(df) * 100)
        })
    
    # Sort by total candidates
    source_stats.sort(key=lambda x: x['total_candidates'], reverse=True)
    
    return source_stats


def analyze_by_position(df: pd.DataFrame, position_col: str,
                        status_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Analyze recruitment by position"""
    
    position_stats = []
    
    for position in df[position_col].unique():
        pos_df = df[df[position_col] == position]
        total = len(pos_df)
        
        if status_col and status_col in df.columns:
            hired = len(pos_df[pos_df[status_col].str.lower().isin(['hired', 'accepted', 'yes', '1'])])
            conversion_rate = hired / total * 100 if total > 0 else 0
        else:
            hired = None
            conversion_rate = None
        
        position_stats.append({
            'position': str(position),
            'total_candidates': int(total),
            'hired': int(hired) if hired is not None else None,
            'conversion_rate': _to_native_type(conversion_rate)
        })
    
    position_stats.sort(key=lambda x: x['total_candidates'], reverse=True)
    
    return position_stats


def calculate_time_metrics(df: pd.DataFrame, apply_date_col: str,
                           hire_date_col: Optional[str] = None,
                           status_col: Optional[str] = None) -> Dict[str, Any]:
    """Calculate time-to-hire and other time metrics"""
    
    # Parse dates
    df['_apply_date'] = pd.to_datetime(df[apply_date_col], errors='coerce')
    
    if hire_date_col and hire_date_col in df.columns:
        df['_hire_date'] = pd.to_datetime(df[hire_date_col], errors='coerce')
        
        # Filter to hired candidates
        if status_col and status_col in df.columns:
            hired_df = df[df[status_col].str.lower().isin(['hired', 'accepted', 'yes', '1'])]
        else:
            hired_df = df[df['_hire_date'].notna()]
        
        # Calculate time to hire
        hired_df = hired_df[hired_df['_hire_date'].notna() & hired_df['_apply_date'].notna()]
        
        if len(hired_df) > 0:
            hired_df['_time_to_hire'] = (hired_df['_hire_date'] - hired_df['_apply_date']).dt.days
            
            time_to_hire = {
                'mean_days': _to_native_type(hired_df['_time_to_hire'].mean()),
                'median_days': _to_native_type(hired_df['_time_to_hire'].median()),
                'min_days': _to_native_type(hired_df['_time_to_hire'].min()),
                'max_days': _to_native_type(hired_df['_time_to_hire'].max()),
                'std_days': _to_native_type(hired_df['_time_to_hire'].std()),
                'p25_days': _to_native_type(hired_df['_time_to_hire'].quantile(0.25)),
                'p75_days': _to_native_type(hired_df['_time_to_hire'].quantile(0.75))
            }
        else:
            time_to_hire = None
    else:
        time_to_hire = None
    
    # Application volume by month
    df['_apply_month'] = df['_apply_date'].dt.to_period('M')
    monthly_volume = df.groupby('_apply_month').size()
    
    volume_trend = []
    for period, count in monthly_volume.items():
        volume_trend.append({
            'month': str(period),
            'applications': int(count)
        })
    
    return {
        'time_to_hire': time_to_hire,
        'volume_trend': volume_trend[-12:] if len(volume_trend) > 12 else volume_trend  # Last 12 months
    }


def analyze_dropoff_reasons(df: pd.DataFrame, status_col: str) -> Dict[str, Any]:
    """Analyze reasons for candidate dropoff"""
    
    status_counts = df[status_col].value_counts()
    
    dropoff_stats = []
    total = len(df)
    
    for status, count in status_counts.items():
        dropoff_stats.append({
            'status': str(status),
            'count': int(count),
            'pct': _to_native_type(count / total * 100)
        })
    
    # Calculate pass-through rate
    hired_statuses = ['hired', 'accepted', 'yes', '1', 'offer accepted']
    hired_count = sum(count for status, count in status_counts.items() 
                      if str(status).lower() in hired_statuses)
    
    rejected_statuses = ['rejected', 'declined', 'no', '0', 'not selected']
    rejected_count = sum(count for status, count in status_counts.items() 
                         if str(status).lower() in rejected_statuses)
    
    withdrawn_statuses = ['withdrawn', 'cancelled', 'dropped']
    withdrawn_count = sum(count for status, count in status_counts.items() 
                          if str(status).lower() in withdrawn_statuses)
    
    return {
        'status_breakdown': dropoff_stats,
        'hired_count': int(hired_count),
        'rejected_count': int(rejected_count),
        'withdrawn_count': int(withdrawn_count),
        'hired_pct': _to_native_type(hired_count / total * 100) if total > 0 else 0,
        'rejected_pct': _to_native_type(rejected_count / total * 100) if total > 0 else 0,
        'withdrawn_pct': _to_native_type(withdrawn_count / total * 100) if total > 0 else 0
    }


def create_funnel_chart(funnel_data: Dict) -> str:
    """Create recruitment funnel visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    stages = funnel_data['stages']
    stage_names = [s['stage'] for s in stages]
    counts = [s['count'] for s in stages]
    conversion_rates = [s['conversion_rate'] for s in stages]
    
    # Funnel chart (horizontal bars)
    max_count = max(counts) if counts else 1
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
    
    y_pos = range(len(stages))
    bars = ax1.barh(y_pos, counts, color=colors, edgecolor='white', linewidth=2)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stage_names)
    ax1.invert_yaxis()
    ax1.set_xlabel('Number of Candidates')
    ax1.set_title('Recruitment Funnel', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, count, rate in zip(bars, counts, conversion_rates):
        ax1.text(bar.get_width() + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                f'{count:,} ({rate:.0f}%)', va='center', fontsize=9)
    
    # Conversion rates chart
    ax2.plot(range(len(stages)), conversion_rates, 'bo-', linewidth=2, markersize=8)
    ax2.fill_between(range(len(stages)), conversion_rates, alpha=0.3)
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels([s[:10] + '...' if len(s) > 10 else s for s in stage_names], rotation=45, ha='right')
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.set_ylim(0, 105)
    ax2.set_title('Stage-to-Stage Conversion', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_source_chart(source_data: List[Dict]) -> str:
    """Create source effectiveness visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sources = [s['source'][:15] for s in source_data[:8]]
    totals = [s['total_candidates'] for s in source_data[:8]]
    conv_rates = [s['conversion_rate'] or 0 for s in source_data[:8]]
    
    # Volume by source
    colors = plt.cm.Set2(np.linspace(0, 1, len(sources)))
    bars = ax1.bar(sources, totals, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Number of Candidates')
    ax1.set_title('Candidates by Source', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, total in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals) * 0.02,
                f'{total}', ha='center', fontsize=9)
    
    # Conversion rate by source
    colors2 = ['#22c55e' if r >= 10 else '#f59e0b' if r >= 5 else '#ef4444' for r in conv_rates]
    bars2 = ax2.bar(sources, conv_rates, color=colors2, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.set_title('Hire Rate by Source', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% target')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, rate in zip(bars2, conv_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_time_chart(time_data: Dict) -> str:
    """Create time metrics visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time to hire distribution (if available)
    if time_data['time_to_hire']:
        tth = time_data['time_to_hire']
        metrics = ['Mean', 'Median', 'P25', 'P75']
        values = [tth['mean_days'], tth['median_days'], tth['p25_days'], tth['p75_days']]
        
        colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444']
        bars = ax1.bar(metrics, values, color=colors, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Days')
        ax1.set_title(f'Time to Hire (Avg: {tth["mean_days"]:.0f} days)', fontsize=12, fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No time data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Time to Hire', fontsize=12, fontweight='bold')
    
    # Volume trend
    if time_data['volume_trend']:
        months = [v['month'] for v in time_data['volume_trend']]
        apps = [v['applications'] for v in time_data['volume_trend']]
        
        ax2.plot(range(len(months)), apps, 'b-o', linewidth=2, markersize=6)
        ax2.fill_between(range(len(months)), apps, alpha=0.3)
        ax2.set_xticks(range(len(months)))
        ax2.set_xticklabels([m[-5:] for m in months], rotation=45, ha='right')
        ax2.set_ylabel('Applications')
        ax2.set_title('Application Volume Trend', fontsize=12, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.text(0.5, 0.5, 'No trend data available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_status_chart(status_data: Dict) -> str:
    """Create candidate status visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    breakdown = status_data['status_breakdown']
    statuses = [s['status'] for s in breakdown]
    counts = [s['count'] for s in breakdown]
    
    # Color mapping
    def get_color(status):
        s = status.lower()
        if any(x in s for x in ['hired', 'accepted', 'offer']):
            return '#22c55e'
        elif any(x in s for x in ['rejected', 'declined', 'not']):
            return '#ef4444'
        elif any(x in s for x in ['withdrawn', 'cancelled']):
            return '#f59e0b'
        else:
            return '#3b82f6'
    
    colors = [get_color(s) for s in statuses]
    
    wedges, texts, autotexts = ax.pie(counts, labels=statuses, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       explode=[0.02] * len(statuses))
    ax.set_title('Candidate Outcome Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(funnel_data: Dict, source_data: Optional[List],
                      time_data: Optional[Dict], status_data: Optional[Dict]) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    # Overall conversion
    conv_rate = funnel_data['overall_conversion_rate']
    if conv_rate >= 10:
        insights.append({
            'title': f'Strong Hiring Rate: {conv_rate:.1f}%',
            'description': f'{funnel_data["total_hired"]} hired from {funnel_data["total_applicants"]} applicants.',
            'status': 'positive'
        })
    elif conv_rate >= 5:
        insights.append({
            'title': f'Average Hiring Rate: {conv_rate:.1f}%',
            'description': 'Consider optimizing screening to improve conversion.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Hiring Rate: {conv_rate:.1f}%',
            'description': 'Review sourcing strategy and qualification criteria.',
            'status': 'warning'
        })
    
    # Biggest dropoff
    if funnel_data['biggest_dropoff_stage']:
        insights.append({
            'title': f'Biggest Dropoff: {funnel_data["biggest_dropoff_stage"]}',
            'description': f'{funnel_data["biggest_dropoff_rate"]:.1f}% candidates lost at this stage.',
            'status': 'warning'
        })
    
    # Best source
    if source_data and len(source_data) > 0:
        best_source = max([s for s in source_data if s['conversion_rate']], 
                         key=lambda x: x['conversion_rate'] or 0, default=None)
        if best_source and best_source['conversion_rate']:
            insights.append({
                'title': f'Top Source: {best_source["source"]}',
                'description': f'{best_source["conversion_rate"]:.1f}% hire rate with {best_source["total_candidates"]} candidates.',
                'status': 'positive'
            })
    
    # Time to hire
    if time_data and time_data.get('time_to_hire'):
        tth = time_data['time_to_hire']['mean_days']
        if tth <= 30:
            insights.append({
                'title': f'Fast Hiring: {tth:.0f} days average',
                'description': 'Efficient recruitment process.',
                'status': 'positive'
            })
        elif tth <= 60:
            insights.append({
                'title': f'Moderate Time-to-Hire: {tth:.0f} days',
                'description': 'Consider streamlining interview process.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': f'Slow Hiring: {tth:.0f} days average',
                'description': 'Long hiring cycle may lose top candidates.',
                'status': 'warning'
            })
    
    return insights


@router.post("/recruitment-funnel")
async def run_recruitment_analysis(request: RecruitmentRequest) -> Dict[str, Any]:
    """
    Perform Recruitment Funnel Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 records")
        
        results = {}
        visualizations = {}
        
        # Funnel analysis
        if request.stage_col and request.stage_col in df.columns:
            funnel_data = analyze_funnel_stages(df, request.stage_col, request.stage_order)
            results['funnel'] = funnel_data
            visualizations['funnel_chart'] = create_funnel_chart(funnel_data)
        else:
            # Create basic funnel from status
            funnel_data = {
                'total_applicants': len(df),
                'total_hired': 0,
                'overall_conversion_rate': 0,
                'stages': [],
                'biggest_dropoff_stage': None,
                'biggest_dropoff_rate': None
            }
            results['funnel'] = funnel_data
        
        # Source analysis
        source_data = None
        if request.source_col and request.source_col in df.columns:
            source_data = analyze_by_source(df, request.source_col, request.status_col)
            results['source_analysis'] = source_data
            visualizations['source_chart'] = create_source_chart(source_data)
        
        # Position analysis
        if request.position_col and request.position_col in df.columns:
            position_data = analyze_by_position(df, request.position_col, request.status_col)
            results['position_analysis'] = position_data
        
        # Time metrics
        time_data = None
        if request.apply_date_col and request.apply_date_col in df.columns:
            time_data = calculate_time_metrics(df, request.apply_date_col, 
                                               request.hire_date_col, request.status_col)
            results['time_metrics'] = time_data
            visualizations['time_chart'] = create_time_chart(time_data)
        
        # Status/dropoff analysis
        status_data = None
        if request.status_col and request.status_col in df.columns:
            status_data = analyze_dropoff_reasons(df, request.status_col)
            results['status_analysis'] = status_data
            visualizations['status_chart'] = create_status_chart(status_data)
        
        # Generate insights
        insights = generate_insights(funnel_data, source_data, time_data, status_data)
        
        # Summary
        summary = {
            'total_candidates': len(df),
            'total_hired': funnel_data['total_hired'],
            'overall_conversion_rate': funnel_data['overall_conversion_rate'],
            'time_to_hire_avg': time_data['time_to_hire']['mean_days'] if time_data and time_data.get('time_to_hire') else None,
            'top_source': source_data[0]['source'] if source_data else None,
            'biggest_dropoff': funnel_data['biggest_dropoff_stage']
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
        raise HTTPException(status_code=500, detail=f"Recruitment analysis failed: {str(e)}")
