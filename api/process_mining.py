"""
Process Mining FastAPI Endpoint
Discover, analyze, and visualize business processes from event logs using pm4py
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
from datetime import datetime, timedelta

# pm4py imports
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.log.importer.xes import importer as xes_importer

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class ProcessMiningRequest(BaseModel):
    """Request model for Process Mining"""
    data: List[Dict[str, Any]]
    case_id_col: str
    activity_col: str
    timestamp_col: str
    algorithm: str = Field(default="heuristics", pattern="^(alpha|inductive|heuristics|dfg)$")
    additional_cols: Optional[List[str]] = None


def prepare_event_log(df: pd.DataFrame, case_id_col: str, activity_col: str, 
                      timestamp_col: str, additional_cols: Optional[List[str]] = None):
    """Prepare event log for pm4py"""
    
    # Rename columns to pm4py standard
    df_log = df.copy()
    df_log = df_log.rename(columns={
        case_id_col: 'case:concept:name',
        activity_col: 'concept:name',
        timestamp_col: 'time:timestamp'
    })
    
    # Convert timestamp
    df_log['time:timestamp'] = pd.to_datetime(df_log['time:timestamp'], errors='coerce')
    
    # Sort by case and timestamp
    df_log = df_log.sort_values(['case:concept:name', 'time:timestamp'])
    
    # Convert to pm4py event log
    event_log = log_converter.apply(df_log, variant=log_converter.Variants.TO_EVENT_LOG)
    
    return event_log, df_log


def discover_process(event_log, algorithm: str):
    """Discover process model using specified algorithm"""
    
    if algorithm == 'alpha':
        net, initial_marking, final_marking = alpha_miner.apply(event_log)
        return net, initial_marking, final_marking, 'petri_net'
    
    elif algorithm == 'inductive':
        tree = inductive_miner.apply(event_log)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
        return net, initial_marking, final_marking, 'process_tree'
    
    elif algorithm == 'heuristics':
        net, initial_marking, final_marking = heuristics_miner.apply(event_log, 
                                                                     parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})
        return net, initial_marking, final_marking, 'petri_net'
    
    elif algorithm == 'dfg':
        dfg = dfg_discovery.apply(event_log)
        return dfg, None, None, 'dfg'
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def calculate_process_metrics(event_log, df_log: pd.DataFrame):
    """Calculate process mining metrics"""
    
    metrics = {}
    
    # Basic statistics
    metrics['n_cases'] = len(event_log)
    metrics['n_events'] = len(df_log)
    metrics['n_activities'] = df_log['concept:name'].nunique()
    metrics['n_variants'] = len(pm4py.get_variants(event_log))
    
    # Time-based metrics
    case_durations = case_statistics.get_all_case_durations(event_log, parameters={
        case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"
    })
    
    if case_durations:
        metrics['avg_case_duration'] = float(np.mean(case_durations))
        metrics['median_case_duration'] = float(np.median(case_durations))
        metrics['min_case_duration'] = float(np.min(case_durations))
        metrics['max_case_duration'] = float(np.max(case_durations))
    else:
        metrics['avg_case_duration'] = 0
        metrics['median_case_duration'] = 0
        metrics['min_case_duration'] = 0
        metrics['max_case_duration'] = 0
    
    # Activity frequency
    activity_counts = df_log['concept:name'].value_counts()
    metrics['top_activities'] = activity_counts.head(10).to_dict()
    
    # Start and end activities
    start_activities = df_log.groupby('case:concept:name')['concept:name'].first().value_counts()
    end_activities = df_log.groupby('case:concept:name')['concept:name'].last().value_counts()
    
    metrics['start_activities'] = start_activities.head(5).to_dict()
    metrics['end_activities'] = end_activities.head(5).to_dict()
    
    return metrics


def analyze_variants(event_log):
    """Analyze process variants"""
    
    variants = pm4py.get_variants(event_log)
    
    variant_stats = []
    total_cases = len(event_log)
    
    for idx, (variant, cases) in enumerate(sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)[:10]):
        variant_stats.append({
            'rank': idx + 1,
            'variant': ' → '.join(variant),
            'frequency': len(cases),
            'percentage': round(len(cases) / total_cases * 100, 2)
        })
    
    return variant_stats


def calculate_bottlenecks(df_log: pd.DataFrame):
    """Identify bottlenecks in the process"""
    
    # Calculate time between activities
    df_log = df_log.sort_values(['case:concept:name', 'time:timestamp'])
    df_log['time_to_next'] = df_log.groupby('case:concept:name')['time:timestamp'].diff().shift(-1)
    df_log['time_to_next_seconds'] = df_log['time_to_next'].dt.total_seconds()
    
    # Get activity transitions
    df_log['next_activity'] = df_log.groupby('case:concept:name')['concept:name'].shift(-1)
    df_log['transition'] = df_log['concept:name'] + ' → ' + df_log['next_activity'].fillna('END')
    
    # Calculate average waiting time per transition
    bottlenecks = df_log.groupby('transition').agg({
        'time_to_next_seconds': ['mean', 'median', 'count']
    }).reset_index()
    
    bottlenecks.columns = ['transition', 'avg_duration', 'median_duration', 'frequency']
    bottlenecks = bottlenecks.dropna()
    bottlenecks = bottlenecks.sort_values('avg_duration', ascending=False).head(10)
    
    return bottlenecks.to_dict('records')


def generate_visualizations(event_log, df_log: pd.DataFrame, net, initial_marking, 
                            final_marking, model_type: str, algorithm: str):
    """Generate process mining visualizations"""
    
    visualizations = {}
    
    # 1. Process Model Visualization
    try:
        if model_type == 'dfg':
            gviz = dfg_visualizer.apply(net, parameters={dfg_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"})
            # DFG visualization returns a graph, convert to base64
            visualizations['process_model'] = "dfg_visualization"
        elif model_type == 'petri_net':
            gviz = pn_visualizer.apply(net, initial_marking, final_marking)
            visualizations['process_model'] = "petri_net_visualization"
        else:
            visualizations['process_model'] = "model_visualization"
    except Exception as e:
        print(f"Process model visualization error: {e}")
        visualizations['process_model'] = None
    
    # 2. Activity Frequency Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    activity_counts = df_log['concept:name'].value_counts().head(15)
    colors = sns.color_palette("husl", len(activity_counts))
    
    bars = ax.barh(range(len(activity_counts)), activity_counts.values, color=colors)
    ax.set_yticks(range(len(activity_counts)))
    ax.set_yticklabels(activity_counts.index)
    ax.set_xlabel('Frequency', fontsize=11)
    ax.set_ylabel('Activity', fontsize=11)
    ax.set_title('Top 15 Activities by Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(activity_counts.values):
        ax.text(v + max(activity_counts.values) * 0.01, i, str(v), 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    visualizations['activity_frequency'] = fig_to_base64(fig)
    
    # 3. Case Duration Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    case_durations = case_statistics.get_all_case_durations(event_log)
    case_durations_hours = [d / 3600 for d in case_durations]  # Convert to hours
    
    ax.hist(case_durations_hours, bins=30, color='#4A90E2', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(case_durations_hours), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(case_durations_hours):.1f}h')
    ax.axvline(np.median(case_durations_hours), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {np.median(case_durations_hours):.1f}h')
    
    ax.set_xlabel('Duration (hours)', fontsize=11)
    ax.set_ylabel('Number of Cases', fontsize=11)
    ax.set_title('Case Duration Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    visualizations['case_duration'] = fig_to_base64(fig)
    
    # 4. Process Flow (Start to End Activities)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    start_acts = df_log.groupby('case:concept:name')['concept:name'].first().value_counts().head(10)
    end_acts = df_log.groupby('case:concept:name')['concept:name'].last().value_counts().head(10)
    
    colors1 = sns.color_palette("Greens_r", len(start_acts))
    colors2 = sns.color_palette("Reds_r", len(end_acts))
    
    ax1.barh(range(len(start_acts)), start_acts.values, color=colors1)
    ax1.set_yticks(range(len(start_acts)))
    ax1.set_yticklabels(start_acts.index)
    ax1.set_xlabel('Frequency', fontsize=10)
    ax1.set_title('Start Activities', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    ax2.barh(range(len(end_acts)), end_acts.values, color=colors2)
    ax2.set_yticks(range(len(end_acts)))
    ax2.set_yticklabels(end_acts.index)
    ax2.set_xlabel('Frequency', fontsize=10)
    ax2.set_title('End Activities', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    visualizations['flow_activities'] = fig_to_base64(fig)
    
    # 5. Events Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_log['date'] = df_log['time:timestamp'].dt.date
    events_per_day = df_log.groupby('date').size()
    
    ax.plot(events_per_day.index, events_per_day.values, marker='o', 
           linewidth=2, markersize=4, color='#4A90E2')
    ax.fill_between(events_per_day.index, events_per_day.values, alpha=0.3, color='#4A90E2')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Number of Events', fontsize=11)
    ax.set_title('Events Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    visualizations['events_timeline'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(metrics: Dict, variants: List[Dict], bottlenecks: List[Dict]):
    """Generate key insights"""
    
    insights = []
    
    # Complexity insight
    if metrics['n_variants'] > metrics['n_cases'] * 0.5:
        insights.append({
            'title': 'High Process Variability',
            'description': f"{metrics['n_variants']} variants from {metrics['n_cases']} cases indicates high process complexity. Consider standardization.",
            'status': 'warning'
        })
    elif metrics['n_variants'] < metrics['n_cases'] * 0.1:
        insights.append({
            'title': 'Standardized Process',
            'description': f"Only {metrics['n_variants']} variants shows good process standardization and consistency.",
            'status': 'positive'
        })
    
    # Duration insight
    avg_duration_hours = metrics['avg_case_duration'] / 3600
    if avg_duration_hours > 24:
        insights.append({
            'title': 'Long Process Duration',
            'description': f"Average case duration of {avg_duration_hours:.1f} hours. Review for optimization opportunities.",
            'status': 'warning'
        })
    
    # Top variant dominance
    if variants and variants[0]['percentage'] > 50:
        insights.append({
            'title': 'Dominant Process Path',
            'description': f"Top variant represents {variants[0]['percentage']}% of cases, indicating strong process adherence.",
            'status': 'positive'
        })
    
    # Bottleneck insight
    if bottlenecks:
        top_bottleneck = bottlenecks[0]
        avg_wait_hours = top_bottleneck['avg_duration'] / 3600
        insights.append({
            'title': 'Bottleneck Identified',
            'description': f"Transition '{top_bottleneck['transition']}' has average wait time of {avg_wait_hours:.1f} hours.",
            'status': 'warning'
        })
    
    return insights


@router.post("/process-mining")
async def analyze_process(request: ProcessMiningRequest):
    """
    Process Mining Analysis Endpoint
    
    Discovers and analyzes business processes from event logs
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        
        df = pd.DataFrame(request.data)
        
        required_cols = [request.case_id_col, request.activity_col, request.timestamp_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns: {missing}")
        
        if len(df) < 10:
            raise HTTPException(400, "Insufficient events (need at least 10)")
        
        # Prepare event log
        try:
            event_log, df_log = prepare_event_log(
                df, request.case_id_col, request.activity_col, 
                request.timestamp_col, request.additional_cols
            )
        except Exception as e:
            raise HTTPException(400, f"Event log preparation failed: {str(e)}")
        
        if len(event_log) < 2:
            raise HTTPException(400, "Need at least 2 cases for process mining")
        
        # Discover process model
        try:
            net, initial_marking, final_marking, model_type = discover_process(
                event_log, request.algorithm
            )
        except Exception as e:
            raise HTTPException(400, f"Process discovery failed: {str(e)}")
        
        # Calculate metrics
        metrics = calculate_process_metrics(event_log, df_log)
        
        # Analyze variants
        variants = analyze_variants(event_log)
        
        # Calculate bottlenecks
        bottlenecks = calculate_bottlenecks(df_log)
        
        # Generate visualizations
        visualizations = generate_visualizations(
            event_log, df_log, net, initial_marking, final_marking,
            model_type, request.algorithm
        )
        
        # Generate insights
        insights = generate_insights(metrics, variants, bottlenecks)
        
        # Prepare response
        response_data = {
            'success': True,
            'results': {
                'algorithm': request.algorithm,
                'model_type': model_type,
                'metrics': metrics,
                'variants': variants,
                'bottlenecks': bottlenecks
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'process_mining',
                'n_cases': metrics['n_cases'],
                'n_events': metrics['n_events'],
                'n_activities': metrics['n_activities'],
                'n_variants': metrics['n_variants'],
                'avg_duration_hours': round(metrics['avg_case_duration'] / 3600, 2)
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
