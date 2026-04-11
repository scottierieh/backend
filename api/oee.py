"""
OEE (Overall Equipment Effectiveness) Router for FastAPI
Availability, Performance, Quality calculations with Six Big Losses analysis
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
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class OEERequest(BaseModel):
    data: List[Dict[str, Any]]
    # Time columns (in minutes or hours - specify unit)
    planned_production_time_col: Optional[str] = None
    run_time_col: Optional[str] = None
    downtime_col: Optional[str] = None
    # Production columns
    total_count_col: Optional[str] = None
    good_count_col: Optional[str] = None
    defect_count_col: Optional[str] = None
    # Performance columns
    ideal_cycle_time_col: Optional[str] = None
    actual_cycle_time_col: Optional[str] = None
    # For aggregation
    time_col: Optional[str] = None
    equipment_col: Optional[str] = None
    # Manual input values (if not in data)
    planned_production_time: Optional[float] = None
    run_time: Optional[float] = None
    ideal_cycle_time: Optional[float] = None
    # Targets
    target_oee: float = 85.0
    target_availability: float = 90.0
    target_performance: float = 95.0
    target_quality: float = 99.9


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
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_oee_metrics(planned_time: float, run_time: float, 
                          total_count: int, good_count: int,
                          ideal_cycle_time: float) -> Dict[str, Any]:
    """Calculate OEE and its components"""
    
    # Availability = Run Time / Planned Production Time
    availability = run_time / planned_time if planned_time > 0 else 0
    
    # Performance = (Ideal Cycle Time × Total Count) / Run Time
    if run_time > 0 and ideal_cycle_time > 0:
        theoretical_output = run_time / ideal_cycle_time
        performance = total_count / theoretical_output if theoretical_output > 0 else 0
    else:
        performance = 0
    
    # Cap performance at 100%
    performance = min(performance, 1.0)
    
    # Quality = Good Count / Total Count
    quality = good_count / total_count if total_count > 0 else 0
    
    # OEE = Availability × Performance × Quality
    oee = availability * performance * quality
    
    # Calculate losses
    downtime = planned_time - run_time
    speed_loss_time = run_time - (total_count * ideal_cycle_time) if ideal_cycle_time > 0 else 0
    speed_loss_time = max(0, speed_loss_time)
    defect_count = total_count - good_count
    quality_loss_time = defect_count * ideal_cycle_time if ideal_cycle_time > 0 else 0
    
    # Productive time
    productive_time = good_count * ideal_cycle_time if ideal_cycle_time > 0 else 0
    
    return {
        'oee': _to_native_type(oee * 100),
        'availability': _to_native_type(availability * 100),
        'performance': _to_native_type(performance * 100),
        'quality': _to_native_type(quality * 100),
        'planned_time': _to_native_type(planned_time),
        'run_time': _to_native_type(run_time),
        'downtime': _to_native_type(downtime),
        'total_count': _to_native_type(total_count),
        'good_count': _to_native_type(good_count),
        'defect_count': _to_native_type(defect_count),
        'ideal_cycle_time': _to_native_type(ideal_cycle_time),
        'speed_loss_time': _to_native_type(speed_loss_time),
        'quality_loss_time': _to_native_type(quality_loss_time),
        'productive_time': _to_native_type(productive_time)
    }


def calculate_six_big_losses(metrics: Dict[str, Any], 
                              equipment_failure: float = None,
                              setup_adjustment: float = None,
                              idling_minor_stops: float = None,
                              reduced_speed: float = None,
                              startup_rejects: int = None,
                              production_rejects: int = None) -> Dict[str, Any]:
    """Calculate Six Big Losses breakdown"""
    
    downtime = metrics['downtime']
    speed_loss_time = metrics['speed_loss_time']
    defect_count = metrics['defect_count']
    ideal_cycle_time = metrics['ideal_cycle_time']
    
    # Estimate breakdown if not provided
    if equipment_failure is None:
        equipment_failure = downtime * 0.6
    if setup_adjustment is None:
        setup_adjustment = downtime * 0.4
    if idling_minor_stops is None:
        idling_minor_stops = speed_loss_time * 0.4
    if reduced_speed is None:
        reduced_speed = speed_loss_time * 0.6
    if startup_rejects is None:
        startup_rejects = int(defect_count * 0.3)
    if production_rejects is None:
        production_rejects = defect_count - startup_rejects
    
    startup_reject_time = startup_rejects * ideal_cycle_time if ideal_cycle_time else 0
    production_reject_time = production_rejects * ideal_cycle_time if ideal_cycle_time else 0
    
    total_loss_time = (equipment_failure + setup_adjustment + 
                       idling_minor_stops + reduced_speed + 
                       startup_reject_time + production_reject_time)
    
    losses = [
        {
            'category': 'Availability',
            'loss_type': 'Equipment Failure',
            'time_loss': _to_native_type(equipment_failure),
            'percentage': _to_native_type(equipment_failure / total_loss_time * 100 if total_loss_time > 0 else 0),
            'description': 'Unplanned equipment breakdowns and repairs'
        },
        {
            'category': 'Availability',
            'loss_type': 'Setup & Adjustments',
            'time_loss': _to_native_type(setup_adjustment),
            'percentage': _to_native_type(setup_adjustment / total_loss_time * 100 if total_loss_time > 0 else 0),
            'description': 'Changeover time and machine setup'
        },
        {
            'category': 'Performance',
            'loss_type': 'Idling & Minor Stops',
            'time_loss': _to_native_type(idling_minor_stops),
            'percentage': _to_native_type(idling_minor_stops / total_loss_time * 100 if total_loss_time > 0 else 0),
            'description': 'Brief stoppages and material jams'
        },
        {
            'category': 'Performance',
            'loss_type': 'Reduced Speed',
            'time_loss': _to_native_type(reduced_speed),
            'percentage': _to_native_type(reduced_speed / total_loss_time * 100 if total_loss_time > 0 else 0),
            'description': 'Running below optimal speed'
        },
        {
            'category': 'Quality',
            'loss_type': 'Startup Rejects',
            'time_loss': _to_native_type(startup_reject_time),
            'percentage': _to_native_type(startup_reject_time / total_loss_time * 100 if total_loss_time > 0 else 0),
            'description': 'Defects during startup/warmup',
            'count': _to_native_type(startup_rejects)
        },
        {
            'category': 'Quality',
            'loss_type': 'Production Rejects',
            'time_loss': _to_native_type(production_reject_time),
            'percentage': _to_native_type(production_reject_time / total_loss_time * 100 if total_loss_time > 0 else 0),
            'description': 'Defects during stable production',
            'count': _to_native_type(production_rejects)
        }
    ]
    
    return {
        'losses': losses,
        'total_loss_time': _to_native_type(total_loss_time),
        'availability_loss': _to_native_type(equipment_failure + setup_adjustment),
        'performance_loss': _to_native_type(idling_minor_stops + reduced_speed),
        'quality_loss': _to_native_type(startup_reject_time + production_reject_time)
    }


def create_oee_gauge_chart(metrics: Dict[str, Any], targets: Dict[str, float]) -> str:
    """Create OEE gauge chart with components"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    components = [
        ('OEE', metrics['oee'], targets.get('oee', 85)),
        ('Availability', metrics['availability'], targets.get('availability', 90)),
        ('Performance', metrics['performance'], targets.get('performance', 95)),
        ('Quality', metrics['quality'], targets.get('quality', 99.9))
    ]
    
    colors = {
        'world_class': '#22c55e',
        'good': '#3b82f6',
        'average': '#f59e0b',
        'poor': '#ef4444'
    }
    
    for ax, (name, value, target) in zip(axes, components):
        # Determine color based on value
        if value >= 85:
            color = colors['world_class']
        elif value >= 65:
            color = colors['good']
        elif value >= 40:
            color = colors['average']
        else:
            color = colors['poor']
        
        # Create donut chart
        sizes = [value, 100 - value]
        ax.pie(sizes, colors=[color, '#e5e7eb'], startangle=90,
               wedgeprops=dict(width=0.3, edgecolor='white'))
        
        # Add center text
        ax.text(0, 0.1, f'{value:.1f}%', ha='center', va='center', 
                fontsize=20, fontweight='bold', color=color)
        ax.text(0, -0.2, name, ha='center', va='center', fontsize=12, color='#374151')
        
        # Add target line indicator
        if target:
            ax.text(0, -0.4, f'Target: {target}%', ha='center', va='center', 
                    fontsize=9, color='#6b7280')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_time_breakdown_chart(metrics: Dict[str, Any]) -> str:
    """Create time breakdown waterfall chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = ['Planned\nTime', 'Downtime\nLoss', 'Run\nTime', 'Speed\nLoss', 
                  'Net Run\nTime', 'Quality\nLoss', 'Productive\nTime']
    
    planned = metrics['planned_time']
    downtime = metrics['downtime']
    run_time = metrics['run_time']
    speed_loss = metrics['speed_loss_time']
    net_run = run_time - speed_loss
    quality_loss = metrics['quality_loss_time']
    productive = metrics['productive_time']
    
    values = [planned, -downtime, run_time, -speed_loss, net_run, -quality_loss, productive]
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#ef4444', '#22c55e', '#ef4444', '#22c55e']
    
    # Create waterfall effect
    cumulative = 0
    for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
        if i == 0:
            ax.bar(i, val, color=color, edgecolor='white', linewidth=2)
            cumulative = val
        elif val < 0:
            ax.bar(i, abs(val), bottom=cumulative + val, color=color, 
                   edgecolor='white', linewidth=2)
            cumulative += val
        else:
            ax.bar(i, val, color=color, edgecolor='white', linewidth=2)
            cumulative = val
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel('Time (minutes)')
    ax.set_title('OEE Time Breakdown', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_six_losses_chart(losses: Dict[str, Any]) -> str:
    """Create Six Big Losses visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    loss_data = losses['losses']
    names = [l['loss_type'] for l in loss_data]
    values = [l['time_loss'] for l in loss_data]
    categories = [l['category'] for l in loss_data]
    
    # Color by category
    category_colors = {
        'Availability': '#ef4444',
        'Performance': '#f59e0b',
        'Quality': '#3b82f6'
    }
    colors = [category_colors[c] for c in categories]
    
    # Bar chart
    bars = ax1.barh(names, values, color=colors, edgecolor='white', linewidth=2)
    ax1.set_xlabel('Time Loss (minutes)')
    ax1.set_title('Six Big Losses', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontsize=10)
    
    # Pie chart by category
    category_losses = {
        'Availability': losses['availability_loss'],
        'Performance': losses['performance_loss'],
        'Quality': losses['quality_loss']
    }
    
    pie_colors = ['#ef4444', '#f59e0b', '#3b82f6']
    wedges, texts, autotexts = ax2.pie(
        list(category_losses.values()),
        labels=list(category_losses.keys()),
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05, 0.05)
    )
    ax2.set_title('Loss Distribution by Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_trend_chart(trend_data: List[Dict], time_col: str) -> str:
    """Create OEE trend over time"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    df = pd.DataFrame(trend_data)
    
    ax.plot(df[time_col], df['oee'], marker='o', linewidth=2, 
            color='#3b82f6', label='OEE', markersize=8)
    ax.plot(df[time_col], df['availability'], marker='s', linewidth=1.5, 
            color='#22c55e', label='Availability', alpha=0.7)
    ax.plot(df[time_col], df['performance'], marker='^', linewidth=1.5, 
            color='#f59e0b', label='Performance', alpha=0.7)
    ax.plot(df[time_col], df['quality'], marker='d', linewidth=1.5, 
            color='#8b5cf6', label='Quality', alpha=0.7)
    
    # Add world-class reference line
    ax.axhline(y=85, color='#22c55e', linestyle='--', alpha=0.5, label='World-Class (85%)')
    
    ax.set_xlabel(time_col)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('OEE Trend Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 105)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_equipment_comparison_chart(equipment_data: List[Dict]) -> str:
    """Create equipment comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = pd.DataFrame(equipment_data)
    equipment = df['equipment']
    x = np.arange(len(equipment))
    width = 0.2
    
    ax.bar(x - 1.5*width, df['availability'], width, label='Availability', color='#22c55e')
    ax.bar(x - 0.5*width, df['performance'], width, label='Performance', color='#f59e0b')
    ax.bar(x + 0.5*width, df['quality'], width, label='Quality', color='#8b5cf6')
    ax.bar(x + 1.5*width, df['oee'], width, label='OEE', color='#3b82f6')
    
    ax.axhline(y=85, color='#22c55e', linestyle='--', alpha=0.5, label='World-Class')
    
    ax.set_xticks(x)
    ax.set_xticklabels(equipment, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Equipment OEE Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(metrics: Dict[str, Any], losses: Dict[str, Any],
                          targets: Dict[str, float]) -> Dict[str, Any]:
    """Generate key insights for OEE analysis"""
    
    key_insights = []
    oee = metrics['oee']
    availability = metrics['availability']
    performance = metrics['performance']
    quality = metrics['quality']
    
    # OEE level insight
    if oee >= 85:
        key_insights.append({
            'title': 'World-Class OEE',
            'description': f'OEE = {oee:.1f}% (≥85%). Equipment is performing at world-class level.',
            'status': 'positive'
        })
    elif oee >= 65:
        key_insights.append({
            'title': 'Good OEE',
            'description': f'OEE = {oee:.1f}% (65-85%). Room for improvement but acceptable.',
            'status': 'neutral'
        })
    elif oee >= 40:
        key_insights.append({
            'title': 'Average OEE',
            'description': f'OEE = {oee:.1f}% (40-65%). Significant improvement potential.',
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': 'Low OEE',
            'description': f'OEE = {oee:.1f}% (<40%). Urgent improvement needed.',
            'status': 'warning'
        })
    
    # Identify limiting factor
    components = [('Availability', availability), ('Performance', performance), ('Quality', quality)]
    limiting = min(components, key=lambda x: x[1])
    
    key_insights.append({
        'title': f'{limiting[0]} is Limiting Factor',
        'description': f'{limiting[0]} = {limiting[1]:.1f}% is the lowest component. Focus improvement efforts here.',
        'status': 'neutral'
    })
    
    # Target comparison
    if oee >= targets['oee']:
        key_insights.append({
            'title': 'Target Achieved',
            'description': f"OEE {oee:.1f}% meets target of {targets['oee']}%.",
            'status': 'positive'
        })
    else:
        gap = targets['oee'] - oee
        key_insights.append({
            'title': 'Below Target',
            'description': f"OEE {oee:.1f}% is {gap:.1f}% below target of {targets['oee']}%.",
            'status': 'warning'
        })
    
    # Six Big Losses insight
    if losses and losses.get('total_loss_time', 0) > 0:
        total_loss = losses['total_loss_time']
        biggest_loss = max(losses['losses'], key=lambda x: x['time_loss'] or 0)
        key_insights.append({
            'title': 'Biggest Loss Category',
            'description': f"{biggest_loss['loss_type']} accounts for {biggest_loss['percentage']:.1f}% of total losses ({biggest_loss['time_loss']:.0f} min).",
            'status': 'neutral'
        })
    
    return {'key_insights': key_insights}


@router.post("/oee")
async def run_oee_analysis(request: OEERequest) -> Dict[str, Any]:
    """
    Perform OEE (Overall Equipment Effectiveness) Analysis.
    """
    try:
        data = request.data
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Get or calculate values
        if request.planned_production_time_col and request.planned_production_time_col in df.columns:
            planned_time = pd.to_numeric(df[request.planned_production_time_col], errors='coerce').sum()
        elif request.planned_production_time:
            planned_time = request.planned_production_time
        else:
            planned_time = 480  # Default 8 hours in minutes
        
        if request.run_time_col and request.run_time_col in df.columns:
            run_time = pd.to_numeric(df[request.run_time_col], errors='coerce').sum()
        elif request.run_time:
            run_time = request.run_time
        elif request.downtime_col and request.downtime_col in df.columns:
            downtime = pd.to_numeric(df[request.downtime_col], errors='coerce').sum()
            run_time = planned_time - downtime
        else:
            run_time = planned_time * 0.9
        
        if request.total_count_col and request.total_count_col in df.columns:
            total_count = int(pd.to_numeric(df[request.total_count_col], errors='coerce').sum())
        else:
            total_count = len(df)
        
        if request.good_count_col and request.good_count_col in df.columns:
            good_count = int(pd.to_numeric(df[request.good_count_col], errors='coerce').sum())
        elif request.defect_count_col and request.defect_count_col in df.columns:
            defect_count = int(pd.to_numeric(df[request.defect_count_col], errors='coerce').sum())
            good_count = total_count - defect_count
        else:
            good_count = int(total_count * 0.99)
        
        if request.ideal_cycle_time_col and request.ideal_cycle_time_col in df.columns:
            ideal_cycle_time = pd.to_numeric(df[request.ideal_cycle_time_col], errors='coerce').mean()
        elif request.ideal_cycle_time:
            ideal_cycle_time = request.ideal_cycle_time
        else:
            ideal_cycle_time = run_time / total_count if total_count > 0 else 1
        
        # Validate inputs
        if planned_time <= 0:
            raise HTTPException(status_code=400, detail="Planned production time must be positive.")
        
        if run_time > planned_time:
            run_time = planned_time
        
        if good_count > total_count:
            good_count = total_count
        
        # Calculate OEE metrics
        metrics = calculate_oee_metrics(
            planned_time=planned_time,
            run_time=run_time,
            total_count=total_count,
            good_count=good_count,
            ideal_cycle_time=ideal_cycle_time
        )
        
        # Calculate Six Big Losses
        losses = calculate_six_big_losses(metrics)
        
        # Targets
        targets = {
            'oee': request.target_oee,
            'availability': request.target_availability,
            'performance': request.target_performance,
            'quality': request.target_quality
        }
        
        # Generate visualizations
        visualizations = {
            'gauge_chart': create_oee_gauge_chart(metrics, targets),
            'time_breakdown': create_time_breakdown_chart(metrics),
            'six_losses': create_six_losses_chart(losses)
        }
        
        # Generate trend data if time column provided
        trend_data = None
        if request.time_col and request.time_col in df.columns:
            grouped = df.groupby(request.time_col)
            trend_data = []
            
            for time_val, group in grouped:
                if request.total_count_col and request.total_count_col in group.columns:
                    tc = int(pd.to_numeric(group[request.total_count_col], errors='coerce').sum())
                else:
                    tc = len(group)
                
                if request.good_count_col and request.good_count_col in group.columns:
                    gc = int(pd.to_numeric(group[request.good_count_col], errors='coerce').sum())
                else:
                    gc = int(tc * 0.99)
                
                if request.run_time_col and request.run_time_col in group.columns:
                    rt = pd.to_numeric(group[request.run_time_col], errors='coerce').sum()
                else:
                    rt = planned_time * 0.9 / len(grouped)
                
                pt = planned_time / len(grouped)
                
                period_metrics = calculate_oee_metrics(pt, rt, tc, gc, ideal_cycle_time)
                trend_data.append({
                    request.time_col: str(time_val),
                    'oee': period_metrics['oee'],
                    'availability': period_metrics['availability'],
                    'performance': period_metrics['performance'],
                    'quality': period_metrics['quality']
                })
            
            if trend_data:
                visualizations['trend_chart'] = create_trend_chart(trend_data, request.time_col)
        
        # Generate equipment comparison if equipment column provided
        equipment_data = None
        if request.equipment_col and request.equipment_col in df.columns:
            grouped = df.groupby(request.equipment_col)
            equipment_data = []
            
            for equip, group in grouped:
                if request.total_count_col and request.total_count_col in group.columns:
                    tc = int(pd.to_numeric(group[request.total_count_col], errors='coerce').sum())
                else:
                    tc = len(group)
                
                if request.good_count_col and request.good_count_col in group.columns:
                    gc = int(pd.to_numeric(group[request.good_count_col], errors='coerce').sum())
                else:
                    gc = int(tc * 0.99)
                
                if request.run_time_col and request.run_time_col in group.columns:
                    rt = pd.to_numeric(group[request.run_time_col], errors='coerce').sum()
                else:
                    rt = planned_time * 0.9 / len(grouped)
                
                pt = planned_time / len(grouped)
                
                equip_metrics = calculate_oee_metrics(pt, rt, tc, gc, ideal_cycle_time)
                equipment_data.append({
                    'equipment': str(equip),
                    'oee': equip_metrics['oee'],
                    'availability': equip_metrics['availability'],
                    'performance': equip_metrics['performance'],
                    'quality': equip_metrics['quality']
                })
            
            if equipment_data:
                visualizations['equipment_chart'] = create_equipment_comparison_chart(equipment_data)
        
        # Generate insights
        insights = generate_key_insights(metrics, losses, targets)
        
        # World-class benchmarks
        benchmarks = {
            'world_class': {'oee': 85, 'availability': 90, 'performance': 95, 'quality': 99.9},
            'typical': {'oee': 60, 'availability': 80, 'performance': 80, 'quality': 95},
            'low': {'oee': 40, 'availability': 70, 'performance': 70, 'quality': 90}
        }
        
        return {
            'success': True,
            'metrics': metrics,
            'losses': losses,
            'targets': targets,
            'benchmarks': benchmarks,
            'visualizations': visualizations,
            'trend_data': trend_data,
            'equipment_data': equipment_data,
            'key_insights': insights['key_insights'],
            'summary': {
                'oee': metrics['oee'],
                'availability': metrics['availability'],
                'performance': metrics['performance'],
                'quality': metrics['quality'],
                'limiting_factor': min(
                    [('Availability', metrics['availability']), 
                     ('Performance', metrics['performance']), 
                     ('Quality', metrics['quality'])],
                    key=lambda x: x[1]
                )[0],
                'total_loss_time': losses['total_loss_time'],
                'productive_time': metrics['productive_time']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OEE analysis failed: {str(e)}")
