"""
Funnel Analysis Router for FastAPI
Conversion funnel analysis with drop-off detection and segment comparison
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
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class FunnelAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    user_col: str
    stage_col: str
    stage_order_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    segment_col: Optional[str] = None
    analysis_type: Literal["standard", "time_based", "segmented"] = "standard"
    custom_stage_order: Optional[List[str]] = None


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
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def determine_stage_order(df: pd.DataFrame, stage_col: str, 
                          stage_order_col: Optional[str] = None,
                          custom_order: Optional[List[str]] = None) -> List[str]:
    if custom_order:
        return custom_order
    if stage_order_col and stage_order_col in df.columns:
        order_df = df.groupby(stage_col)[stage_order_col].min().reset_index()
        order_df = order_df.sort_values(stage_order_col)
        return order_df[stage_col].tolist()
    stages = []
    for stage in df[stage_col].values:
        if stage not in stages:
            stages.append(stage)
    return stages


def calculate_funnel_metrics(df: pd.DataFrame, user_col: str, stage_col: str,
                             stage_order: List[str]) -> List[Dict[str, Any]]:
    user_stages = df.groupby(user_col)[stage_col].apply(set).reset_index()
    user_stages.columns = [user_col, 'stages']
    
    stage_metrics = []
    total_users = len(user_stages)
    prev_users = total_users
    
    for idx, stage in enumerate(stage_order):
        users_at_stage = user_stages[user_stages['stages'].apply(lambda x: stage in x)]
        user_count = len(users_at_stage)
        
        if idx == 0:
            conversion_rate = 1.0
            drop_off_rate = 0.0
            drop_off_count = 0
        else:
            conversion_rate = user_count / prev_users if prev_users > 0 else 0
            drop_off_rate = (1 - conversion_rate) * 100
            drop_off_count = prev_users - user_count
        
        cumulative_conversion = (user_count / total_users * 100) if total_users > 0 else 0
        
        stage_metrics.append({
            'stage_name': stage,
            'users': user_count,
            'conversion_rate': conversion_rate,
            'drop_off_rate': drop_off_rate,
            'drop_off_count': drop_off_count,
            'cumulative_conversion': cumulative_conversion
        })
        prev_users = user_count
    
    return stage_metrics


def calculate_time_metrics(df: pd.DataFrame, user_col: str, stage_col: str,
                           timestamp_col: str, stage_order: List[str]) -> Dict[str, Any]:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    stage_times = defaultdict(list)
    
    for user_id in df[user_col].unique():
        user_data = df[df[user_col] == user_id].sort_values(timestamp_col)
        for i, stage in enumerate(stage_order):
            stage_data = user_data[user_data[stage_col] == stage]
            if len(stage_data) > 0 and i < len(stage_order) - 1:
                next_stage = stage_order[i + 1]
                next_data = user_data[user_data[stage_col] == next_stage]
                if len(next_data) > 0:
                    time_diff = (next_data[timestamp_col].min() - stage_data[timestamp_col].min()).total_seconds()
                    if time_diff > 0:
                        stage_times[stage].append(time_diff)
    
    time_metrics = {}
    for stage in stage_order[:-1]:
        times = stage_times.get(stage, [])
        if times:
            time_metrics[stage] = {
                'avg_time': np.mean(times),
                'median_time': np.median(times)
            }
    
    bottleneck_stage = None
    max_time = 0
    for stage, metrics in time_metrics.items():
        if metrics['avg_time'] > max_time:
            max_time = metrics['avg_time']
            bottleneck_stage = stage
    
    all_times = []
    for user_id in df[user_col].unique():
        user_data = df[df[user_col] == user_id].sort_values(timestamp_col)
        if len(user_data) > 1:
            total_time = (user_data[timestamp_col].max() - user_data[timestamp_col].min()).total_seconds()
            if total_time > 0:
                all_times.append(total_time)
    
    return {
        'stage_times': time_metrics,
        'avg_total_time': np.mean(all_times) if all_times else 0,
        'median_total_time': np.median(all_times) if all_times else 0,
        'bottleneck_stage': bottleneck_stage
    }


def calculate_segment_metrics(df: pd.DataFrame, user_col: str, stage_col: str,
                              segment_col: str, stage_order: List[str]) -> Dict[str, Any]:
    segment_results = {}
    for segment in df[segment_col].unique():
        segment_df = df[df[segment_col] == segment]
        stage_metrics = calculate_funnel_metrics(segment_df, user_col, stage_col, stage_order)
        if stage_metrics:
            first_stage_users = stage_metrics[0]['users']
            last_stage_users = stage_metrics[-1]['users']
            overall_conversion = (last_stage_users / first_stage_users * 100) if first_stage_users > 0 else 0
        else:
            overall_conversion = 0
        segment_results[str(segment)] = {
            'stages': stage_metrics,
            'overall_conversion': overall_conversion
        }
    return segment_results


def create_funnel_chart(stage_metrics: List[Dict]) -> str:
    if not stage_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    stages = [s['stage_name'] for s in stage_metrics]
    users = [s['users'] for s in stage_metrics]
    max_users = max(users) if users else 1
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(stages)))
    
    for i, (stage, user_count) in enumerate(zip(stages, users)):
        width = user_count / max_users
        left = (1 - width) / 2
        ax.barh(len(stages) - i - 1, width, left=left, height=0.7, 
                color=colors[i], edgecolor='white', linewidth=2)
        ax.text(0.5, len(stages) - i - 1, f'{stage}\n{user_count:,} ({stage_metrics[i]["cumulative_conversion"]:.1f}%)',
                ha='center', va='center', fontsize=10, fontweight='bold', 
                color='white' if width > 0.3 else 'black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(stages) - 0.5)
    ax.axis('off')
    ax.set_title('Conversion Funnel', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_conversion_bars(stage_metrics: List[Dict]) -> str:
    if not stage_metrics or len(stage_metrics) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Need at least 2 stages', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    stages = [s['stage_name'] for s in stage_metrics[1:]]
    rates = [s['conversion_rate'] * 100 for s in stage_metrics[1:]]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#22c55e' if r >= 70 else '#f59e0b' if r >= 50 else '#ef4444' for r in rates]
    bars = ax.bar(stages, rates, color=colors, edgecolor='white', linewidth=1)
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax.set_xlabel('Stage', fontsize=11)
    ax.set_title('Stage-to-Stage Conversion Rates', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% benchmark')
    ax.legend(loc='lower right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_dropoff_chart(stage_metrics: List[Dict]) -> str:
    if not stage_metrics or len(stage_metrics) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Need at least 2 stages', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    stages = [s['stage_name'] for s in stage_metrics[1:]]
    drop_rates = [s['drop_off_rate'] for s in stage_metrics[1:]]
    drop_counts = [s['drop_off_count'] for s in stage_metrics[1:]]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    colors = ['#ef4444' if r > 30 else '#f59e0b' if r > 20 else '#22c55e' for r in drop_rates]
    ax1.bar(stages, drop_counts, color=colors, alpha=0.7, label='Drop-off Count')
    ax1.set_ylabel('Users Lost', fontsize=11, color='#ef4444')
    ax1.tick_params(axis='y', labelcolor='#ef4444')
    
    ax2 = ax1.twinx()
    ax2.plot(stages, drop_rates, 'o-', color='#3b82f6', linewidth=2, markersize=8, label='Drop-off Rate')
    ax2.set_ylabel('Drop-off Rate (%)', fontsize=11, color='#3b82f6')
    ax2.tick_params(axis='y', labelcolor='#3b82f6')
    
    ax1.set_xlabel('Stage', fontsize=11)
    ax1.set_title('Drop-off Analysis by Stage', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_comparison(segment_metrics: Dict[str, Any]) -> str:
    if not segment_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No segment data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    segments = list(segment_metrics.keys())
    conversions = [segment_metrics[s]['overall_conversion'] for s in segments]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(segments)))
    bars = ax.bar(segments, conversions, color=colors, edgecolor='white', linewidth=1)
    
    for bar, conv in zip(bars, conversions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{conv:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Overall Conversion Rate (%)', fontsize=11)
    ax.set_xlabel('Segment', fontsize=11)
    ax.set_title('Conversion Rate by Segment', fontsize=14, fontweight='bold')
    avg_conv = np.mean(conversions)
    ax.axhline(y=avg_conv, color='gray', linestyle='--', alpha=0.7, label=f'Average: {avg_conv:.1f}%')
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_time_distribution(time_metrics: Dict[str, Any]) -> str:
    stage_times = time_metrics.get('stage_times', {})
    if not stage_times:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No time data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    stages = list(stage_times.keys())
    avg_times = [stage_times[s]['avg_time'] / 60 for s in stages]
    median_times = [stage_times[s]['median_time'] / 60 for s in stages]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(stages))
    width = 0.35
    ax.bar(x - width/2, avg_times, width, label='Average', color='#3b82f6', alpha=0.8)
    ax.bar(x + width/2, median_times, width, label='Median', color='#22c55e', alpha=0.8)
    ax.set_ylabel('Time (minutes)', fontsize=11)
    ax.set_xlabel('Stage', fontsize=11)
    ax.set_title('Time Spent at Each Stage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(stage_metrics: List[Dict], overall_conversion: float,
                          biggest_drop_off: Dict, segment_metrics: Optional[Dict] = None,
                          time_metrics: Optional[Dict] = None) -> List[Dict]:
    insights = []
    
    if overall_conversion >= 10:
        insights.append({
            'title': f'Strong Overall Conversion: {overall_conversion:.1f}%',
            'description': 'Your funnel is performing above average.',
            'status': 'positive'
        })
    elif overall_conversion >= 3:
        insights.append({
            'title': f'Moderate Overall Conversion: {overall_conversion:.1f}%',
            'description': 'Your funnel has room for improvement.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Overall Conversion: {overall_conversion:.1f}%',
            'description': 'Your funnel needs significant optimization.',
            'status': 'warning'
        })
    
    insights.append({
        'title': f'Critical Drop-off at "{biggest_drop_off["stage"]}"',
        'description': f'{biggest_drop_off["drop_off_rate"]:.1f}% of users abandon at this stage.',
        'status': 'warning'
    })
    
    if len(stage_metrics) > 1:
        best_stage = min(stage_metrics[1:], key=lambda x: x['drop_off_rate'])
        if best_stage['drop_off_rate'] < 20:
            insights.append({
                'title': f'Strong Stage: "{best_stage["stage_name"]}"',
                'description': f'Only {best_stage["drop_off_rate"]:.1f}% drop-off at this stage.',
                'status': 'positive'
            })
    
    if segment_metrics and len(segment_metrics) > 1:
        conversions = [(s, segment_metrics[s]['overall_conversion']) for s in segment_metrics]
        best = max(conversions, key=lambda x: x[1])
        worst = min(conversions, key=lambda x: x[1])
        if best[1] - worst[1] > 5:
            insights.append({
                'title': f'Segment Gap: {best[0]} vs {worst[0]}',
                'description': f'{best[0]} converts at {best[1]:.1f}% while {worst[0]} only converts at {worst[1]:.1f}%.',
                'status': 'neutral'
            })
    
    if time_metrics and time_metrics.get('bottleneck_stage'):
        insights.append({
            'title': f'Time Bottleneck: "{time_metrics["bottleneck_stage"]}"',
            'description': 'Users spend the most time at this stage.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/funnel")
async def run_funnel_analysis(request: FunnelAnalysisRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        
        if request.user_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"User column '{request.user_col}' not found")
        if request.stage_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Stage column '{request.stage_col}' not found")
        
        stage_order = determine_stage_order(df, request.stage_col, request.stage_order_col, request.custom_stage_order)
        
        if len(stage_order) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 stages for funnel analysis")
        
        stage_metrics = calculate_funnel_metrics(df, request.user_col, request.stage_col, stage_order)
        
        total_users = stage_metrics[0]['users'] if stage_metrics else 0
        total_converted = stage_metrics[-1]['users'] if stage_metrics else 0
        overall_conversion = (total_converted / total_users * 100) if total_users > 0 else 0
        
        biggest_drop_off = {'stage': '', 'drop_off_rate': 0, 'drop_off_count': 0}
        for s in stage_metrics[1:]:
            if s['drop_off_rate'] > biggest_drop_off['drop_off_rate']:
                biggest_drop_off = {
                    'stage': s['stage_name'],
                    'drop_off_rate': s['drop_off_rate'],
                    'drop_off_count': s['drop_off_count']
                }
        
        visualizations = {
            'funnel_chart': create_funnel_chart(stage_metrics),
            'conversion_bars': create_conversion_bars(stage_metrics),
            'drop_off_chart': create_dropoff_chart(stage_metrics)
        }
        
        segment_analysis = None
        time_analysis = None
        
        if request.analysis_type == "segmented" and request.segment_col:
            if request.segment_col in df.columns:
                segment_analysis = calculate_segment_metrics(df, request.user_col, request.stage_col, 
                                                            request.segment_col, stage_order)
                visualizations['segment_comparison'] = create_segment_comparison(segment_analysis)
        
        if request.analysis_type == "time_based" and request.timestamp_col:
            if request.timestamp_col in df.columns:
                time_analysis = calculate_time_metrics(df, request.user_col, request.stage_col,
                                                       request.timestamp_col, stage_order)
                visualizations['time_distribution'] = create_time_distribution(time_analysis)
        
        key_insights = generate_key_insights(stage_metrics, overall_conversion, biggest_drop_off,
                                             segment_analysis, time_analysis)
        
        results = {
            'stage_metrics': [{k: _to_native_type(v) for k, v in s.items()} for s in stage_metrics],
            'overall_conversion': _to_native_type(overall_conversion),
            'total_users': total_users,
            'total_converted': total_converted,
            'biggest_drop_off': {k: _to_native_type(v) for k, v in biggest_drop_off.items()}
        }
        
        if segment_analysis:
            results['segment_analysis'] = {
                seg: {
                    'stages': [{k: _to_native_type(v) for k, v in s.items()} for s in data['stages']],
                    'overall_conversion': _to_native_type(data['overall_conversion'])
                }
                for seg, data in segment_analysis.items()
            }
        
        if time_analysis:
            results['time_analysis'] = {
                'avg_total_time': _to_native_type(time_analysis['avg_total_time']),
                'median_total_time': _to_native_type(time_analysis['median_total_time']),
                'bottleneck_stage': time_analysis['bottleneck_stage']
            }
        
        summary = {
            'analysis_type': request.analysis_type,
            'total_stages': len(stage_order),
            'total_users': total_users,
            'overall_conversion': _to_native_type(overall_conversion)
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funnel analysis failed: {str(e)}")
