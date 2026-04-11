"""
Funnel Analysis Router for FastAPI
Track user journey through conversion steps and identify drop-off points
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class FunnelRequest(BaseModel):
    data: List[Dict[str, Any]]
    user_id_col: str
    event_col: str
    funnel_steps: List[str]
    timestamp_col: Optional[str] = None
    segment_col: Optional[str] = None


class FunnelResult(BaseModel):
    step: str
    users: int
    conversion_rate_from_start: float
    conversion_rate_from_previous: float


class DropOff(BaseModel):
    from_step: str
    to_step: str
    dropped_users: int
    drop_rate: float


class TimeAnalysis(BaseModel):
    step: str
    avg_time_hours: float
    median_time_hours: float


class SegmentAnalysis(BaseModel):
    segment: str
    segment_type: str
    total_users: int
    converted_users: int
    conversion_rate: float


class FunnelResults(BaseModel):
    results: List[Dict[str, Any]]
    drop_offs: List[Dict[str, Any]]
    time_analysis: Optional[List[Dict[str, Any]]] = None
    segment_analysis: Optional[List[Dict[str, Any]]] = None
    summary: Dict[str, Any]


class FunnelResponse(BaseModel):
    results: FunnelResults
    funnel_plot: str
    dropoff_plot: str
    dashboard_plot: str


def _to_native_type(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _convert_dict(d: dict) -> dict:
    """Recursively convert numpy types in dictionary"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_dict(v)
        elif isinstance(v, list):
            result[k] = [_convert_dict(i) if isinstance(i, dict) else _to_native_type(i) for i in v]
        else:
            result[k] = _to_native_type(v)
    return result


def create_funnel_plot(funnel_results: List[Dict], funnel_steps: List[str]) -> str:
    """Create funnel visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    steps = [r['step'] for r in funnel_results]
    users = [r['users'] for r in funnel_results]
    conversion_rates = [r['conversion_rate_from_start'] * 100 for r in funnel_results]
    
    max_users = users[0] if users else 1
    
    # Create funnel bars
    colors = plt.cm.Blues(np.linspace(0.8, 0.3, len(steps)))
    
    y_positions = range(len(steps) - 1, -1, -1)
    
    for i, (step, user_count, conv_rate, y_pos) in enumerate(zip(steps, users, conversion_rates, y_positions)):
        width = (user_count / max_users) * 100
        bar = ax.barh(y_pos, width, height=0.6, color=colors[i], edgecolor='black', linewidth=1.5)
        
        # Add labels
        label_text = f"{step}\n{user_count:,} users ({conv_rate:.1f}%)"
        ax.text(width / 2, y_pos, label_text, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white' if width > 30 else 'black')
    
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.5, len(steps) - 0.5)
    ax.set_xlabel('Relative Volume (%)', fontsize=12, fontweight='bold')
    ax.set_title('Funnel Conversion Flow', fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_dropoff_plot(drop_offs: List[Dict]) -> str:
    """Create drop-off analysis bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not drop_offs:
        ax.text(0.5, 0.5, 'No drop-off data available', ha='center', va='center', fontsize=12)
        ax.set_title('Drop-off Analysis')
    else:
        transitions = [f"{d['from_step']} → {d['to_step']}" for d in drop_offs]
        drop_rates = [d['drop_rate'] for d in drop_offs]
        
        colors = ['#F44336' if rate > 50 else '#FF9800' if rate > 30 else '#4CAF50' for rate in drop_rates]
        
        bars = ax.barh(range(len(transitions)), drop_rates, color=colors, edgecolor='black', linewidth=1)
        
        for i, (bar, rate, dropped) in enumerate(zip(bars, drop_rates, [d['dropped_users'] for d in drop_offs])):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{rate:.1f}% ({dropped:,} users)', va='center', fontsize=9)
        
        ax.set_yticks(range(len(transitions)))
        ax.set_yticklabels(transitions, fontsize=9)
        ax.set_xlabel('Drop-off Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Drop-off Rate by Transition', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlim(0, max(drop_rates) * 1.3 if drop_rates else 100)
        ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='Critical (50%)')
        ax.axvline(30, color='orange', linestyle='--', alpha=0.5, label='Warning (30%)')
        ax.legend(loc='lower right', fontsize=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_dashboard_plot(funnel_results: List[Dict], drop_offs: List[Dict], summary: Dict) -> str:
    """Create comprehensive dashboard with multiple visualizations"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # 1. Funnel Flow (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    steps = [r['step'] for r in funnel_results]
    users = [r['users'] for r in funnel_results]
    max_users = users[0] if users else 1
    
    colors = plt.cm.Blues(np.linspace(0.8, 0.3, len(steps)))
    y_positions = range(len(steps) - 1, -1, -1)
    
    for i, (step, user_count, y_pos) in enumerate(zip(steps, users, y_positions)):
        width = (user_count / max_users) * 100
        ax1.barh(y_pos, width, height=0.6, color=colors[i], edgecolor='black')
        ax1.text(width + 2, y_pos, f'{user_count:,}', va='center', fontsize=9, fontweight='bold')
    
    ax1.set_yticks(list(y_positions))
    ax1.set_yticklabels(steps[::-1], fontsize=9)
    ax1.set_xlim(0, 120)
    ax1.set_title('Funnel Flow', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Conversion Rate per Step (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    step_conv = [r['conversion_rate_from_previous'] * 100 for r in funnel_results]
    bar_colors = ['#4CAF50' if c >= 70 else '#FF9800' if c >= 50 else '#F44336' for c in step_conv]
    bars = ax2.bar(range(len(steps)), step_conv, color=bar_colors, edgecolor='black')
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Conversion Rate (%)', fontsize=10)
    ax2.set_title('Step-to-Step Conversion', fontsize=12, fontweight='bold')
    ax2.axhline(70, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(50, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 110)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, step_conv):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
    
    # 3. Drop-off Analysis (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    if drop_offs:
        transitions = [f"{d['from_step'][:10]}→{d['to_step'][:10]}" for d in drop_offs]
        drop_rates = [d['drop_rate'] for d in drop_offs]
        colors_drop = ['#F44336' if r > 50 else '#FF9800' if r > 30 else '#4CAF50' for r in drop_rates]
        bars = ax3.barh(range(len(transitions)), drop_rates, color=colors_drop, edgecolor='black')
        ax3.set_yticks(range(len(transitions)))
        ax3.set_yticklabels(transitions, fontsize=8)
        ax3.set_xlabel('Drop Rate (%)', fontsize=10)
        ax3.axvline(50, color='red', linestyle='--', alpha=0.3)
        ax3.axvline(30, color='orange', linestyle='--', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No drop-off data', ha='center', va='center')
    ax3.set_title('Drop-off by Transition', fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Summary Stats (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = f"""
    FUNNEL SUMMARY
    {'='*40}
    
    Starting Users:     {summary.get('start_users', 0):,}
    Completed Users:    {summary.get('end_users', 0):,}
    Overall Conversion: {summary.get('overall_conversion', 0):.1f}%
    
    Total Dropped:      {summary.get('total_dropped', 0):,}
    Avg Step Conversion: {summary.get('avg_step_conversion', 0):.1f}%
    
    Worst Drop-off:     {summary.get('worst_dropoff_step', 'N/A')}
                        ({summary.get('worst_dropoff_rate', 0):.1f}%)
    
    Best Transition:    {summary.get('best_transition_step', 'N/A')}
                        ({summary.get('best_transition_rate', 0):.1f}% retained)
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle('Funnel Analysis Dashboard', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@router.post("/funnel")
async def run_funnel_analysis(request: FunnelRequest) -> Dict[str, Any]:
    """
    Perform Funnel Analysis to track user journey through conversion steps.
    
    This endpoint:
    1. Tracks users through sequential funnel steps
    2. Calculates conversion rates at each stage
    3. Identifies drop-off points and bottlenecks
    4. Generates visualizations for strategic decision-making
    """
    try:
        data = request.data
        user_id_col = request.user_id_col
        event_col = request.event_col
        funnel_steps = request.funnel_steps
        timestamp_col = request.timestamp_col
        segment_col = request.segment_col

        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        if not user_id_col:
            raise HTTPException(status_code=400, detail="User ID column not specified.")
        
        if not event_col:
            raise HTTPException(status_code=400, detail="Event column not specified.")
        
        if not funnel_steps or len(funnel_steps) < 2:
            raise HTTPException(status_code=400, detail="At least 2 funnel steps required.")

        df = pd.DataFrame(data)

        # Validate columns exist
        if user_id_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"User ID column '{user_id_col}' not found in data.")
        
        if event_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Event column '{event_col}' not found in data.")

        # Filter data to only include events in the funnel
        df_funnel = df[df[event_col].isin(funnel_steps)].copy()
        
        if df_funnel.empty:
            raise HTTPException(status_code=400, detail="No matching events found for the specified funnel steps.")

        # --- 1. Calculate Funnel Metrics ---
        funnel_results = []
        users_at_start = None
        users_in_previous_step = None
        
        for i, step in enumerate(funnel_steps):
            # Get users who completed this step and all previous steps
            if i == 0:
                users_in_step = set(df_funnel[df_funnel[event_col] == step][user_id_col].unique())
            else:
                # Users must have completed previous step to count
                step_users = set(df_funnel[df_funnel[event_col] == step][user_id_col].unique())
                users_in_step = users_in_previous_step.intersection(step_users)
            
            user_count = len(users_in_step)
            
            if i == 0:
                users_at_start = user_count
                conversion_from_start = 1.0
                conversion_from_previous = 1.0
            else:
                conversion_from_start = user_count / users_at_start if users_at_start > 0 else 0
                conversion_from_previous = user_count / len(users_in_previous_step) if users_in_previous_step and len(users_in_previous_step) > 0 else 0
            
            funnel_results.append({
                'step': step,
                'users': user_count,
                'conversion_rate_from_start': float(conversion_from_start),
                'conversion_rate_from_previous': float(conversion_from_previous)
            })
            
            users_in_previous_step = users_in_step

        # --- 2. Calculate Drop-offs ---
        drop_offs = []
        for i in range(len(funnel_results) - 1):
            current_step = funnel_results[i]
            next_step = funnel_results[i + 1]
            
            dropped_users = current_step['users'] - next_step['users']
            drop_rate = (dropped_users / current_step['users'] * 100) if current_step['users'] > 0 else 0
            
            drop_offs.append({
                'from_step': current_step['step'],
                'to_step': next_step['step'],
                'dropped_users': int(dropped_users),
                'drop_rate': float(drop_rate)
            })

        # --- 3. Time Analysis (if timestamp provided) ---
        time_analysis = None
        if timestamp_col and timestamp_col in df.columns:
            try:
                df_funnel['_timestamp'] = pd.to_datetime(df_funnel[timestamp_col], errors='coerce')
                df_funnel = df_funnel.dropna(subset=['_timestamp'])
                
                if not df_funnel.empty:
                    time_analysis = []
                    for i, step in enumerate(funnel_steps[1:], 1):
                        prev_step = funnel_steps[i - 1]
                        
                        prev_times = df_funnel[df_funnel[event_col] == prev_step].groupby(user_id_col)['_timestamp'].min()
                        curr_times = df_funnel[df_funnel[event_col] == step].groupby(user_id_col)['_timestamp'].min()
                        
                        common_users = prev_times.index.intersection(curr_times.index)
                        if len(common_users) > 0:
                            time_diffs = (curr_times[common_users] - prev_times[common_users]).dt.total_seconds() / 3600
                            time_diffs = time_diffs[time_diffs >= 0]
                            
                            if len(time_diffs) > 0:
                                time_analysis.append({
                                    'step': f'{prev_step} → {step}',
                                    'avg_time_hours': float(time_diffs.mean()),
                                    'median_time_hours': float(time_diffs.median())
                                })
            except Exception:
                time_analysis = None

        # --- 4. Segment Analysis (if segment column provided) ---
        segment_analysis = None
        if segment_col and segment_col in df.columns:
            try:
                first_step = funnel_steps[0]
                last_step = funnel_steps[-1]
                
                segments = df_funnel[segment_col].unique()
                segment_results = []
                
                for segment in segments:
                    if pd.isna(segment):
                        continue
                    
                    segment_df = df_funnel[df_funnel[segment_col] == segment]
                    
                    start_users = set(segment_df[segment_df[event_col] == first_step][user_id_col].unique())
                    end_users = set(segment_df[segment_df[event_col] == last_step][user_id_col].unique())
                    converted_users = start_users.intersection(end_users)
                    
                    total = len(start_users)
                    converted = len(converted_users)
                    
                    if total > 0:
                        segment_results.append({
                            'segment': str(segment),
                            'segment_type': segment_col,
                            'total_users': int(total),
                            'converted_users': int(converted),
                            'conversion_rate': float(converted / total * 100)
                        })
                
                if segment_results:
                    segment_analysis = sorted(segment_results, key=lambda x: x['conversion_rate'], reverse=True)
            except Exception:
                segment_analysis = None

        # --- 5. Summary Statistics ---
        start_users = funnel_results[0]['users'] if funnel_results else 0
        end_users = funnel_results[-1]['users'] if funnel_results else 0
        overall_conversion = (end_users / start_users * 100) if start_users > 0 else 0
        total_dropped = start_users - end_users
        
        avg_step_conversion = np.mean([r['conversion_rate_from_previous'] for r in funnel_results[1:]]) * 100 if len(funnel_results) > 1 else 100
        
        worst_dropoff = max(drop_offs, key=lambda x: x['drop_rate']) if drop_offs else None
        best_transition = min(drop_offs, key=lambda x: x['drop_rate']) if drop_offs else None
        
        summary = {
            'start_users': int(start_users),
            'end_users': int(end_users),
            'overall_conversion': float(overall_conversion),
            'total_dropped': int(total_dropped),
            'avg_step_conversion': float(avg_step_conversion),
            'num_steps': len(funnel_steps),
            'worst_dropoff_step': f"{worst_dropoff['from_step']} → {worst_dropoff['to_step']}" if worst_dropoff else 'N/A',
            'worst_dropoff_rate': float(worst_dropoff['drop_rate']) if worst_dropoff else 0,
            'best_transition_step': f"{best_transition['from_step']} → {best_transition['to_step']}" if best_transition else 'N/A',
            'best_transition_rate': float(100 - best_transition['drop_rate']) if best_transition else 100,
            'has_critical_dropoff': any(d['drop_rate'] > 50 for d in drop_offs),
            'has_good_conversion': overall_conversion >= 10
        }

        # --- 6. Generate Plots ---
        funnel_plot_img = create_funnel_plot(funnel_results, funnel_steps)
        dropoff_plot_img = create_dropoff_plot(drop_offs)
        dashboard_plot_img = create_dashboard_plot(funnel_results, drop_offs, summary)

        # --- 7. Prepare Response ---
        funnel_results = [_convert_dict(r) for r in funnel_results]
        drop_offs = [_convert_dict(d) for d in drop_offs]
        
        if time_analysis:
            time_analysis = [_convert_dict(t) for t in time_analysis]
        
        if segment_analysis:
            segment_analysis = [_convert_dict(s) for s in segment_analysis]

        response = {
            'results': {
                'results': funnel_results,
                'drop_offs': drop_offs,
                'time_analysis': time_analysis,
                'segment_analysis': segment_analysis,
                'summary': _convert_dict(summary)
            },
            'funnel_plot': funnel_plot_img,
            'dropoff_plot': dropoff_plot_img,
            'dashboard_plot': dashboard_plot_img
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Funnel analysis failed: {str(e)}")
