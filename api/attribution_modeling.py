"""
Marketing Attribution Modeling Router for FastAPI
Implements multiple attribution models: First Touch, Last Touch, Linear, Time Decay, Position Based
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
import time
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class AttributionRequest(BaseModel):
    data: List[Dict[str, Any]]
    journey_id_col: str
    channel_col: str
    order_col: str
    converted_col: str
    value_col: Optional[str] = None
    time_decay_half_life: int = 7


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
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


CHANNEL_COLORS = {
    'Paid Search': '#4285f4',
    'Organic Search': '#34a853',
    'Social': '#e91e63',
    'Email': '#ff9800',
    'Display': '#9c27b0',
    'Direct': '#607d8b',
    'Referral': '#00bcd4',
    'Affiliate': '#795548',
    'Mobile': '#3f51b5',
}


def first_touch_attribution(journeys: Dict[str, List[Dict]], total_conversions: int) -> Dict[str, float]:
    """100% credit to first touchpoint"""
    attribution = defaultdict(float)
    
    for journey_id, touchpoints in journeys.items():
        if touchpoints and touchpoints[-1].get('converted', 0):
            first_channel = touchpoints[0]['channel']
            attribution[first_channel] += 1
    
    # Normalize to percentage
    for channel in attribution:
        attribution[channel] = (attribution[channel] / total_conversions * 100) if total_conversions > 0 else 0
    
    return dict(attribution)


def last_touch_attribution(journeys: Dict[str, List[Dict]], total_conversions: int) -> Dict[str, float]:
    """100% credit to last touchpoint"""
    attribution = defaultdict(float)
    
    for journey_id, touchpoints in journeys.items():
        if touchpoints and touchpoints[-1].get('converted', 0):
            last_channel = touchpoints[-1]['channel']
            attribution[last_channel] += 1
    
    for channel in attribution:
        attribution[channel] = (attribution[channel] / total_conversions * 100) if total_conversions > 0 else 0
    
    return dict(attribution)


def linear_attribution(journeys: Dict[str, List[Dict]], total_conversions: int) -> Dict[str, float]:
    """Equal credit to all touchpoints"""
    attribution = defaultdict(float)
    
    for journey_id, touchpoints in journeys.items():
        if touchpoints and touchpoints[-1].get('converted', 0):
            n = len(touchpoints)
            credit = 1 / n
            for tp in touchpoints:
                attribution[tp['channel']] += credit
    
    for channel in attribution:
        attribution[channel] = (attribution[channel] / total_conversions * 100) if total_conversions > 0 else 0
    
    return dict(attribution)


def time_decay_attribution(journeys: Dict[str, List[Dict]], total_conversions: int, half_life: int = 7) -> Dict[str, float]:
    """More credit to recent touchpoints using exponential decay"""
    attribution = defaultdict(float)
    decay_rate = 0.5 ** (1 / half_life)
    
    for journey_id, touchpoints in journeys.items():
        if touchpoints and touchpoints[-1].get('converted', 0):
            n = len(touchpoints)
            weights = []
            
            for i in range(n):
                # Days from end (last touchpoint = 0)
                days_from_end = n - 1 - i
                weight = decay_rate ** days_from_end
                weights.append(weight)
            
            total_weight = sum(weights)
            
            for i, tp in enumerate(touchpoints):
                credit = weights[i] / total_weight if total_weight > 0 else 1 / n
                attribution[tp['channel']] += credit
    
    for channel in attribution:
        attribution[channel] = (attribution[channel] / total_conversions * 100) if total_conversions > 0 else 0
    
    return dict(attribution)


def position_based_attribution(journeys: Dict[str, List[Dict]], total_conversions: int) -> Dict[str, float]:
    """40% first, 40% last, 20% distributed among middle"""
    attribution = defaultdict(float)
    
    for journey_id, touchpoints in journeys.items():
        if touchpoints and touchpoints[-1].get('converted', 0):
            n = len(touchpoints)
            
            if n == 1:
                attribution[touchpoints[0]['channel']] += 1
            elif n == 2:
                attribution[touchpoints[0]['channel']] += 0.5
                attribution[touchpoints[1]['channel']] += 0.5
            else:
                # First gets 40%
                attribution[touchpoints[0]['channel']] += 0.4
                # Last gets 40%
                attribution[touchpoints[-1]['channel']] += 0.4
                # Middle shares 20%
                middle_credit = 0.2 / (n - 2)
                for tp in touchpoints[1:-1]:
                    attribution[tp['channel']] += middle_credit
    
    for channel in attribution:
        attribution[channel] = (attribution[channel] / total_conversions * 100) if total_conversions > 0 else 0
    
    return dict(attribution)


def analyze_paths(journeys: Dict[str, List[Dict]]) -> List[Dict]:
    """Analyze conversion paths"""
    path_stats = defaultdict(lambda: {'count': 0, 'conversions': 0, 'total_value': 0})
    
    for journey_id, touchpoints in journeys.items():
        if not touchpoints:
            continue
        
        path = ' > '.join([tp['channel'] for tp in touchpoints])
        path_stats[path]['count'] += 1
        
        if touchpoints[-1].get('converted', 0):
            path_stats[path]['conversions'] += 1
            path_stats[path]['total_value'] += touchpoints[-1].get('value', 0)
    
    results = []
    for path, stats in path_stats.items():
        if stats['conversions'] > 0:
            results.append({
                'path': path,
                'conversions': stats['conversions'],
                'conversion_rate': stats['conversions'] / stats['count'] if stats['count'] > 0 else 0,
                'avg_value': stats['total_value'] / stats['conversions'] if stats['conversions'] > 0 else 0,
                'total_value': stats['total_value'],
            })
    
    results.sort(key=lambda x: x['conversions'], reverse=True)
    return results[:20]


def analyze_channel_interactions(journeys: Dict[str, List[Dict]]) -> List[Dict]:
    """Analyze channel transitions"""
    transitions = defaultdict(lambda: {'count': 0, 'conversions': 0})
    
    for journey_id, touchpoints in journeys.items():
        converted = touchpoints[-1].get('converted', 0) if touchpoints else 0
        
        for i in range(len(touchpoints) - 1):
            from_channel = touchpoints[i]['channel']
            to_channel = touchpoints[i + 1]['channel']
            key = (from_channel, to_channel)
            transitions[key]['count'] += 1
            if converted:
                transitions[key]['conversions'] += 1
    
    results = []
    for (from_ch, to_ch), stats in transitions.items():
        results.append({
            'from': from_ch,
            'to': to_ch,
            'count': stats['count'],
            'conversion_rate': stats['conversions'] / stats['count'] if stats['count'] > 0 else 0,
        })
    
    results.sort(key=lambda x: x['count'], reverse=True)
    return results[:30]


def create_attribution_comparison_chart(channel_attribution: List[Dict]) -> str:
    """Create model comparison chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    channels = [c['channel'] for c in channel_attribution]
    models = ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']
    model_labels = ['First Touch', 'Last Touch', 'Linear', 'Time Decay', 'Position Based']
    
    x = np.arange(len(channels))
    width = 0.15
    
    colors = ['#4285f4', '#34a853', '#f59e0b', '#e91e63', '#9c27b0']
    
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        values = [c[model] for c in channel_attribution]
        ax.bar(x + i * width, values, width, label=label, color=color, edgecolor='white')
    
    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel('Attribution (%)', fontsize=11)
    ax.set_title('Attribution by Model and Channel', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(channels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_channel_contribution_chart(channel_attribution: List[Dict]) -> str:
    """Create channel contribution pie chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    channels = [c['channel'] for c in channel_attribution]
    colors = [CHANNEL_COLORS.get(ch, '#6b7280') for ch in channels]
    
    # First Touch
    first_values = [c['first_touch'] for c in channel_attribution]
    axes[0].pie(first_values, labels=channels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('First Touch Attribution', fontsize=12, fontweight='bold')
    
    # Last Touch
    last_values = [c['last_touch'] for c in channel_attribution]
    axes[1].pie(last_values, labels=channels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Last Touch Attribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_transition_heatmap(interactions: List[Dict], channels: List[str]) -> str:
    """Create channel transition heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Build transition matrix
    matrix = pd.DataFrame(0, index=channels, columns=channels, dtype=float)
    
    for interaction in interactions:
        if interaction['from'] in channels and interaction['to'] in channels:
            matrix.loc[interaction['from'], interaction['to']] = interaction['count']
    
    # Normalize by row
    row_sums = matrix.sum(axis=1)
    matrix_normalized = matrix.div(row_sums, axis=0).fillna(0) * 100
    
    sns.heatmap(matrix_normalized, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Transition %'})
    
    ax.set_xlabel('To Channel', fontsize=11)
    ax.set_ylabel('From Channel', fontsize=11)
    ax.set_title('Channel Transition Matrix (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_conversion_funnel_chart(channel_attribution: List[Dict]) -> str:
    """Create funnel-style chart showing first vs last touch"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    channels = [c['channel'] for c in channel_attribution]
    first_touch = [c['first_touch'] for c in channel_attribution]
    last_touch = [c['last_touch'] for c in channel_attribution]
    
    y = np.arange(len(channels))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, first_touch, height, label='First Touch', color='#4285f4', edgecolor='white')
    bars2 = ax.barh(y + height/2, last_touch, height, label='Last Touch', color='#34a853', edgecolor='white')
    
    ax.set_xlabel('Attribution (%)', fontsize=11)
    ax.set_ylabel('Channel', fontsize=11)
    ax.set_title('First Touch vs Last Touch Attribution', fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(channels)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(channel_attribution: List[Dict], summary: Dict, top_paths: List[Dict]) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Find best first/last touch channels
    best_first = max(channel_attribution, key=lambda x: x['first_touch'])
    best_last = max(channel_attribution, key=lambda x: x['last_touch'])
    
    # First vs Last comparison
    if best_first['channel'] != best_last['channel']:
        insights.append({
            'title': 'Different Channels for Awareness vs Conversion',
            'description': f"{best_first['channel']} is best at starting journeys ({best_first['first_touch']:.1f}%), while {best_last['channel']} closes conversions ({best_last['last_touch']:.1f}%).",
            'status': 'positive'
        })
    else:
        insights.append({
            'title': f'{best_first["channel"]} Dominates Full Funnel',
            'description': f'This channel performs well in both awareness ({best_first["first_touch"]:.1f}%) and conversion ({best_first["last_touch"]:.1f}%).',
            'status': 'neutral'
        })
    
    # Multi-touch importance
    avg_path = summary['avg_path_length']
    if avg_path > 2:
        insights.append({
            'title': f'Multi-Touch Journeys ({avg_path:.1f} avg)',
            'description': 'Customers need multiple touchpoints. Use multi-touch attribution for budget allocation.',
            'status': 'neutral'
        })
    
    # Top path
    if top_paths:
        top = top_paths[0]
        insights.append({
            'title': f'Top Conversion Path: {top["path"]}',
            'description': f'{top["conversions"]} conversions ({top["conversion_rate"]*100:.1f}% rate). Consider amplifying this combination.',
            'status': 'positive'
        })
    
    return insights


@router.post("/attribution")
async def run_attribution_analysis(request: AttributionRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        for col in [request.journey_id_col, request.channel_col, request.order_col, request.converted_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Sort by journey and order
        df = df.sort_values([request.journey_id_col, request.order_col])
        
        # Parse conversion
        df[request.converted_col] = pd.to_numeric(df[request.converted_col], errors='coerce').fillna(0).astype(int)
        
        # Parse value if provided
        if request.value_col and request.value_col in df.columns:
            df[request.value_col] = pd.to_numeric(df[request.value_col], errors='coerce').fillna(0)
        else:
            df['_value'] = 1
            request.value_col = '_value'
        
        # Build journeys
        journeys: Dict[str, List[Dict]] = {}
        
        for journey_id, group in df.groupby(request.journey_id_col):
            touchpoints = []
            for _, row in group.iterrows():
                touchpoints.append({
                    'channel': row[request.channel_col],
                    'order': row[request.order_col],
                    'converted': row[request.converted_col],
                    'value': row[request.value_col] if request.value_col else 0,
                })
            journeys[str(journey_id)] = touchpoints
        
        # Calculate conversions
        total_conversions = sum(1 for j in journeys.values() if j and j[-1].get('converted', 0))
        total_value = sum(j[-1].get('value', 0) for j in journeys.values() if j and j[-1].get('converted', 0))
        
        if total_conversions == 0:
            raise HTTPException(status_code=400, detail="No conversions found in data")
        
        # Run attribution models
        first_touch = first_touch_attribution(journeys, total_conversions)
        last_touch = last_touch_attribution(journeys, total_conversions)
        linear = linear_attribution(journeys, total_conversions)
        time_decay = time_decay_attribution(journeys, total_conversions, request.time_decay_half_life)
        position_based = position_based_attribution(journeys, total_conversions)
        
        # Get all channels
        all_channels = set()
        for model in [first_touch, last_touch, linear, time_decay, position_based]:
            all_channels.update(model.keys())
        all_channels = sorted(all_channels)
        
        # Build channel attribution list
        channel_attribution = []
        for channel in all_channels:
            # Count channel stats
            channel_conversions = sum(1 for j in journeys.values() 
                                     if j and j[-1].get('converted', 0) and 
                                     any(tp['channel'] == channel for tp in j))
            channel_value = sum(j[-1].get('value', 0) for j in journeys.values()
                               if j and j[-1].get('converted', 0) and
                               any(tp['channel'] == channel for tp in j))
            channel_touchpoints = sum(1 for j in journeys.values() for tp in j if tp['channel'] == channel)
            
            # Average position
            positions = []
            for j in journeys.values():
                for i, tp in enumerate(j):
                    if tp['channel'] == channel:
                        positions.append(i + 1)
            avg_position = np.mean(positions) if positions else 0
            
            channel_attribution.append({
                'channel': channel,
                'first_touch': first_touch.get(channel, 0),
                'last_touch': last_touch.get(channel, 0),
                'linear': linear.get(channel, 0),
                'time_decay': time_decay.get(channel, 0),
                'position_based': position_based.get(channel, 0),
                'conversions': channel_conversions,
                'total_value': channel_value,
                'touchpoints': channel_touchpoints,
                'avg_position': avg_position,
            })
        
        # Sort by linear attribution
        channel_attribution.sort(key=lambda x: x['linear'], reverse=True)
        
        # Analyze paths and interactions
        top_paths = analyze_paths(journeys)
        interactions = analyze_channel_interactions(journeys)
        
        # Summary
        total_touchpoints = sum(len(j) for j in journeys.values())
        path_lengths = [len(j) for j in journeys.values() if j and j[-1].get('converted', 0)]
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        
        summary_data = {
            'total_conversions': total_conversions,
            'total_value': total_value,
            'total_touchpoints': total_touchpoints,
            'unique_channels': len(all_channels),
            'avg_path_length': avg_path_length,
            'conversion_rate': total_conversions / len(journeys) if journeys else 0,
        }
        
        # Best channels
        best_first = max(channel_attribution, key=lambda x: x['first_touch'])['channel']
        best_last = max(channel_attribution, key=lambda x: x['last_touch'])['channel']
        
        # Most assisted (high linear, low first/last)
        for c in channel_attribution:
            c['assist_score'] = c['linear'] - (c['first_touch'] + c['last_touch']) / 2
        most_assisted = max(channel_attribution, key=lambda x: x['assist_score'])['channel']
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {
            'attribution_comparison': create_attribution_comparison_chart(channel_attribution),
            'channel_contribution': create_channel_contribution_chart(channel_attribution),
            'conversion_funnel': create_conversion_funnel_chart(channel_attribution),
            'transition_heatmap': create_transition_heatmap(interactions, all_channels) if len(all_channels) <= 10 else None,
        }
        
        # Key insights
        key_insights = generate_key_insights(channel_attribution, summary_data, top_paths)
        
        results = {
            'summary': {k: _to_native_type(v) for k, v in summary_data.items()},
            'channel_attribution': [{k: _to_native_type(v) for k, v in c.items()} for c in channel_attribution],
            'model_comparison': [],
            'top_paths': top_paths,
            'channel_interactions': interactions,
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'best_first_touch': best_first,
                'best_last_touch': best_last,
                'most_assisted': most_assisted,
                'avg_path_length': avg_path_length,
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attribution analysis failed: {str(e)}")
