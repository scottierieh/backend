"""
Space & Traffic Analysis API
5-step framework for spatial flow optimization
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

router = APIRouter()

class TrafficRequest(BaseModel):
    data: List[Dict[str, Any]]
    zone_col: str
    time_col: str
    visitor_col: str
    capacity_col: Optional[str] = None
    dwell_col: Optional[str] = None
    flow_in_col: Optional[str] = None
    flow_out_col: Optional[str] = None

def _to_native(obj):
    if obj is None: return None
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

# Step 1: Time Congestion
def analyze_congestion(df, time_col, visitor_col):
    hourly = df.groupby(time_col)[visitor_col].sum().reset_index()
    hourly.columns = ['hour', 'visitors']
    hourly = hourly.sort_values('hour')
    peak_idx = hourly['visitors'].idxmax()
    peak_hour = hourly.loc[peak_idx, 'hour']
    peak_visitors = hourly.loc[peak_idx, 'visitors']
    off_peak_idx = hourly['visitors'].idxmin()
    off_peak_hour = hourly.loc[off_peak_idx, 'hour']
    off_peak_visitors = hourly.loc[off_peak_idx, 'visitors']
    peak_ratio = peak_visitors / off_peak_visitors if off_peak_visitors > 0 else 1
    hourly['pct_of_peak'] = hourly['visitors'] / peak_visitors * 100
    hourly['status'] = hourly['pct_of_peak'].apply(lambda x: 'high' if x >= 80 else 'medium' if x >= 50 else 'low')
    high_hours = hourly[hourly['status'] == 'high']['hour'].tolist()
    low_hours = hourly[hourly['status'] == 'low']['hour'].tolist()
    return {
        'peak_hour': _to_native(peak_hour), 'peak_visitors': _to_native(peak_visitors),
        'off_peak_hour': _to_native(off_peak_hour), 'off_peak_visitors': _to_native(off_peak_visitors),
        'peak_ratio': _to_native(peak_ratio), 'high_congestion_hours': [_to_native(h) for h in high_hours],
        'low_congestion_hours': [_to_native(h) for h in low_hours],
        'hourly_stats': [{'hour': _to_native(r['hour']), 'visitors': _to_native(r['visitors']), 
                         'pct_of_peak': _to_native(r['pct_of_peak']), 'status': r['status']} for _, r in hourly.iterrows()]
    }

def create_congestion_chart(cong):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hours = [h['hour'] for h in cong['hourly_stats']]
    visitors = [h['visitors'] for h in cong['hourly_stats']]
    colors = ['#ef4444' if h['status'] == 'high' else '#f59e0b' if h['status'] == 'medium' else '#10b981' for h in cong['hourly_stats']]
    axes[0].bar(hours, visitors, color=colors, edgecolor='black', alpha=0.8)
    axes[0].axhline(y=cong['peak_visitors'] * 0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[0].set_xlabel('Hour'); axes[0].set_ylabel('Visitors'); axes[0].set_title('Hourly Traffic', fontsize=11, fontweight='bold')
    axes[0].legend()
    pcts = [h['pct_of_peak'] for h in cong['hourly_stats']]
    axes[1].plot(hours, pcts, 'b-o', markersize=6)
    axes[1].axhline(y=80, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    axes[1].fill_between(hours, pcts, alpha=0.3)
    axes[1].set_xlabel('Hour'); axes[1].set_ylabel('% of Peak'); axes[1].set_title('Congestion Level', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)

# Step 2: Zone Density
def analyze_zone_density(df, zone_col, visitor_col, capacity_col):
    zone_stats = df.groupby(zone_col).agg({visitor_col: ['mean', 'max', 'sum']}).reset_index()
    zone_stats.columns = ['zone', 'avg_visitors', 'max_visitors', 'total_visitors']
    if capacity_col and capacity_col in df.columns:
        cap = df.groupby(zone_col)[capacity_col].first().reset_index()
        cap.columns = ['zone', 'capacity']
        zone_stats = zone_stats.merge(cap, on='zone')
    else:
        zone_stats['capacity'] = zone_stats['max_visitors'] * 1.2
    zone_stats['utilization'] = zone_stats['avg_visitors'] / zone_stats['capacity'] * 100
    zone_stats['status'] = zone_stats['utilization'].apply(lambda x: 'overcrowded' if x > 90 else 'underutilized' if x < 50 else 'optimal')
    zone_stats = zone_stats.sort_values('avg_visitors', ascending=False)
    busiest = zone_stats.iloc[0]
    least = zone_stats.iloc[-1]
    return {
        'busiest_zone': {'zone': busiest['zone'], 'avg_visitors': _to_native(busiest['avg_visitors']), 'utilization': _to_native(busiest['utilization'])},
        'least_busy_zone': {'zone': least['zone'], 'avg_visitors': _to_native(least['avg_visitors']), 'utilization': _to_native(least['utilization'])},
        'n_overcrowded': len(zone_stats[zone_stats['status'] == 'overcrowded']),
        'n_underutilized': len(zone_stats[zone_stats['status'] == 'underutilized']),
        'avg_utilization': _to_native(zone_stats['utilization'].mean()),
        'zone_stats': [{'zone': r['zone'], 'avg_visitors': _to_native(r['avg_visitors']), 'capacity': _to_native(r['capacity']),
                       'utilization': _to_native(r['utilization']), 'status': r['status']} for _, r in zone_stats.iterrows()]
    }

def create_zone_chart(zone):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    zones = [z['zone'] for z in zone['zone_stats']]
    utils = [z['utilization'] for z in zone['zone_stats']]
    colors = ['#ef4444' if z['status'] == 'overcrowded' else '#f59e0b' if z['status'] == 'underutilized' else '#10b981' for z in zone['zone_stats']]
    axes[0].barh(zones, utils, color=colors, edgecolor='black', alpha=0.8)
    axes[0].axvline(x=90, color='red', linestyle='--', alpha=0.5)
    axes[0].axvline(x=50, color='orange', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Utilization %'); axes[0].set_title('Zone Utilization', fontsize=11, fontweight='bold')
    visitors = [z['avg_visitors'] for z in zone['zone_stats']]
    capacities = [z['capacity'] for z in zone['zone_stats']]
    x = np.arange(len(zones)); width = 0.35
    axes[1].bar(x - width/2, visitors, width, label='Avg Visitors', color='#3b82f6', alpha=0.8)
    axes[1].bar(x + width/2, capacities, width, label='Capacity', color='#d1d5db', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(zones, rotation=45, ha='right')
    axes[1].set_ylabel('Count'); axes[1].set_title('Visitors vs Capacity', fontsize=11, fontweight='bold'); axes[1].legend()
    plt.tight_layout()
    return _fig_to_base64(fig)

# Step 3: Flow & Dwell
def analyze_flow(df, zone_col, dwell_col, flow_in_col, flow_out_col):
    if not dwell_col or dwell_col not in df.columns:
        return {'error': 'Dwell time column not configured'}
    zone_flow = df.groupby(zone_col).agg({dwell_col: 'mean'}).reset_index()
    zone_flow.columns = ['zone', 'dwell_time']
    if flow_in_col and flow_in_col in df.columns:
        fi = df.groupby(zone_col)[flow_in_col].sum().reset_index()
        fi.columns = ['zone', 'flow_in']
        zone_flow = zone_flow.merge(fi, on='zone')
    else:
        zone_flow['flow_in'] = 0
    if flow_out_col and flow_out_col in df.columns:
        fo = df.groupby(zone_col)[flow_out_col].sum().reset_index()
        fo.columns = ['zone', 'flow_out']
        zone_flow = zone_flow.merge(fo, on='zone')
    else:
        zone_flow['flow_out'] = 0
    zone_flow['retention'] = ((zone_flow['flow_in'] - zone_flow['flow_out']) / zone_flow['flow_in'] * 100).fillna(0)
    longest_idx = zone_flow['dwell_time'].idxmax()
    return {
        'avg_dwell_time': _to_native(zone_flow['dwell_time'].mean()),
        'max_dwell_time': _to_native(zone_flow['dwell_time'].max()),
        'total_flow_in': _to_native(zone_flow['flow_in'].sum()),
        'flow_efficiency': _to_native(zone_flow['flow_out'].sum() / zone_flow['flow_in'].sum() * 100) if zone_flow['flow_in'].sum() > 0 else 100,
        'longest_dwell_zone': {'zone': zone_flow.loc[longest_idx, 'zone'], 'dwell_time': _to_native(zone_flow.loc[longest_idx, 'dwell_time'])},
        'zone_flow': [{'zone': r['zone'], 'dwell_time': _to_native(r['dwell_time']), 'flow_in': _to_native(r['flow_in']),
                      'flow_out': _to_native(r['flow_out']), 'retention': _to_native(r['retention'])} for _, r in zone_flow.iterrows()]
    }

def create_flow_chart(flow):
    if flow.get('error'): return ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    zones = [z['zone'] for z in flow['zone_flow']]
    dwells = [z['dwell_time'] for z in flow['zone_flow']]
    axes[0].barh(zones, dwells, color='#8b5cf6', edgecolor='black', alpha=0.8)
    axes[0].axvline(x=flow['avg_dwell_time'], color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Dwell Time (min)'); axes[0].set_title('Dwell Time by Zone', fontsize=11, fontweight='bold')
    flow_in = [z['flow_in'] for z in flow['zone_flow']]
    flow_out = [z['flow_out'] for z in flow['zone_flow']]
    x = np.arange(len(zones)); width = 0.35
    axes[1].bar(x - width/2, flow_in, width, label='Flow In', color='#10b981', alpha=0.8)
    axes[1].bar(x + width/2, flow_out, width, label='Flow Out', color='#ef4444', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(zones, rotation=45, ha='right')
    axes[1].set_ylabel('Count'); axes[1].set_title('Flow In/Out', fontsize=11, fontweight='bold'); axes[1].legend()
    plt.tight_layout()
    return _fig_to_base64(fig)

# Step 4: Bottleneck Detection
def detect_bottlenecks(df, zone_col, time_col, visitor_col, capacity_col):
    if not capacity_col or capacity_col not in df.columns:
        df['_cap'] = df.groupby(zone_col)[visitor_col].transform('max') * 1.1
        capacity_col = '_cap'
    df['overflow'] = (df[visitor_col] - df[capacity_col]) / df[capacity_col] * 100
    bottlenecks = df[df['overflow'] > 0].copy()
    bottlenecks['severity'] = bottlenecks['overflow'].apply(lambda x: 'critical' if x > 20 else 'moderate')
    bottlenecks = bottlenecks.sort_values('overflow', ascending=False)
    bn_list = []
    for _, r in bottlenecks.head(10).iterrows():
        bn_list.append({'zone': r[zone_col], 'hour': _to_native(r[time_col]), 'visitors': _to_native(r[visitor_col]),
                       'capacity': _to_native(r[capacity_col]), 'overflow': _to_native(r['overflow']), 'severity': r['severity']})
    worst = bn_list[0] if bn_list else None
    return {
        'n_bottlenecks': len(bottlenecks),
        'n_critical': len(bottlenecks[bottlenecks['severity'] == 'critical']),
        'n_moderate': len(bottlenecks[bottlenecks['severity'] == 'moderate']),
        'worst_bottleneck': worst,
        'bottlenecks': bn_list
    }

def create_bottleneck_chart(bn):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if bn['bottlenecks']:
        labels = [f"{b['zone']} ({b['hour']}h)" for b in bn['bottlenecks'][:10]]
        overflows = [b['overflow'] for b in bn['bottlenecks'][:10]]
        colors = ['#ef4444' if b['severity'] == 'critical' else '#f59e0b' for b in bn['bottlenecks'][:10]]
        ax.barh(labels, overflows, color=colors, edgecolor='black', alpha=0.8)
        ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
        ax.set_xlabel('Overflow %'); ax.set_title('Bottleneck Severity', fontsize=11, fontweight='bold'); ax.legend()
    else:
        ax.text(0.5, 0.5, 'No bottlenecks detected', ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    return _fig_to_base64(fig)

# Step 5: Optimization
def simulate_optimization(zone_data, bn_data):
    current = {'n_zones': len(zone_data['zone_stats']), 'total_capacity': sum(z['capacity'] for z in zone_data['zone_stats']),
               'avg_utilization': zone_data['avg_utilization']}
    scenarios = []
    if bn_data['n_bottlenecks'] > 0:
        scenarios.append({'name': 'Expand Bottleneck Zones', 'description': f"Increase capacity of {bn_data['n_bottlenecks']} bottleneck zones by 30%", 'improvement': 15})
    overcrowded = [z for z in zone_data['zone_stats'] if z['status'] == 'overcrowded']
    underutilized = [z for z in zone_data['zone_stats'] if z['status'] == 'underutilized']
    if overcrowded and underutilized:
        scenarios.append({'name': 'Redistribute Capacity', 'description': f"Move capacity from {len(underutilized)} underutilized to {len(overcrowded)} overcrowded zones", 'improvement': 20})
    scenarios.append({'name': 'Stagger Peak Hours', 'description': 'Implement time-based pricing/scheduling to flatten peaks', 'improvement': 12})
    scenarios.append({'name': 'Add Signage/Wayfinding', 'description': 'Improve flow with better navigation', 'improvement': 8})
    recommendations = []
    if bn_data['n_critical'] > 0: recommendations.append(f"Address {bn_data['n_critical']} critical bottlenecks immediately")
    if len(underutilized) > 0: recommendations.append(f"Consider repurposing {len(underutilized)} underutilized zones")
    recommendations.append("Monitor peak hours for crowd management")
    return {'current_state': current, 'scenarios': scenarios, 'recommendations': recommendations}

def create_optimization_chart(opt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [s['name'][:20] for s in opt['scenarios']]
    improvements = [s['improvement'] for s in opt['scenarios']]
    axes[0].barh(names, improvements, color='#10b981', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Efficiency Gain %'); axes[0].set_title('Scenario Impact', fontsize=11, fontweight='bold')
    labels = ['Current', 'Optimized']
    utils = [opt['current_state']['avg_utilization'], min(85, opt['current_state']['avg_utilization'] + max(improvements) if improvements else 0)]
    axes[1].bar(labels, utils, color=['#3b82f6', '#10b981'], edgecolor='black', alpha=0.8)
    axes[1].axhline(y=75, color='green', linestyle='--', alpha=0.5, label='Target')
    axes[1].set_ylabel('Avg Utilization %'); axes[1].set_title('Utilization Improvement', fontsize=11, fontweight='bold'); axes[1].legend()
    plt.tight_layout()
    return _fig_to_base64(fig)

# Report Generation
def generate_report(cong, zone, flow, bn, opt):
    report = {}
    report['step1_congestion'] = {'title': '1. Time Congestion', 'question': 'When is traffic highest?',
        'finding': f"Peak hour is {cong['peak_hour']}:00 with {cong['peak_visitors']:,} visitors ({cong['peak_ratio']:.1f}x off-peak).",
        'detail': f"Traffic analysis shows {len(cong['high_congestion_hours'])} high-congestion hours (≥80% of peak). Peak occurs at {cong['peak_hour']}:00. Low-traffic hours: {', '.join(map(str, cong['low_congestion_hours'][:3]))}. Consider staffing and capacity adjustments for peak periods."}
    report['step2_zone'] = {'title': '2. Zone Density', 'question': 'Which zones are most crowded?',
        'finding': f"Busiest: {zone['busiest_zone']['zone']} ({zone['busiest_zone']['utilization']:.0f}% util). {zone['n_overcrowded']} zones overcrowded.",
        'detail': f"Zone analysis across {len(zone['zone_stats'])} zones shows avg utilization of {zone['avg_utilization']:.0f}%. {zone['n_overcrowded']} zones exceed 90% capacity (overcrowded), {zone['n_underutilized']} zones below 50% (underutilized). Focus on balancing load across zones."}
    if flow and not flow.get('error'):
        report['step3_flow'] = {'title': '3. Flow & Dwell', 'question': 'How do visitors move and stay?',
            'finding': f"Avg dwell time: {flow['avg_dwell_time']:.1f} min. Longest: {flow['longest_dwell_zone']['zone']} ({flow['longest_dwell_zone']['dwell_time']:.1f} min).",
            'detail': f"Flow analysis shows avg dwell of {flow['avg_dwell_time']:.1f} min across zones. {flow['longest_dwell_zone']['zone']} has longest engagement. Flow efficiency: {flow['flow_efficiency']:.0f}%. Consider optimizing high-dwell zones for better experience."}
    else:
        report['step3_flow'] = {'title': '3. Flow & Dwell', 'question': 'How do visitors move?', 'finding': 'Flow analysis not performed', 'detail': 'Dwell/flow columns not configured.'}
    report['step4_bottleneck'] = {'title': '4. Bottleneck Detection', 'question': 'Where are the congestion points?',
        'finding': f"{bn['n_bottlenecks']} bottlenecks detected, {bn['n_critical']} critical." + (f" Worst: {bn['worst_bottleneck']['zone']} at {bn['worst_bottleneck']['hour']}:00." if bn['worst_bottleneck'] else ""),
        'detail': f"Bottleneck analysis found {bn['n_bottlenecks']} overflow incidents. {bn['n_critical']} are critical (>20% overflow). " + (f"Primary bottleneck at {bn['worst_bottleneck']['zone']} requires immediate attention." if bn['worst_bottleneck'] else "All zones within acceptable capacity.")}
    best = max(opt['scenarios'], key=lambda x: x['improvement']) if opt['scenarios'] else None
    report['step5_optimization'] = {'title': '5. Space Optimization', 'question': 'How to improve efficiency?',
        'finding': f"Best scenario: {best['name']} (+{best['improvement']}% efficiency)." if best else "No optimization scenarios.",
        'detail': f"Current avg utilization: {opt['current_state']['avg_utilization']:.0f}%. Simulated {len(opt['scenarios'])} scenarios. " + (f"Top recommendation: {best['name']} with {best['improvement']}% improvement. " if best else "") + " ".join(opt['recommendations'][:2])}
    return report

def generate_insights(cong, zone, bn, opt):
    insights = []
    if cong['peak_ratio'] > 2: insights.append({'title': 'High Peak Variance', 'description': f"Peak is {cong['peak_ratio']:.1f}x off-peak. Consider load balancing.", 'status': 'warning'})
    if zone['n_overcrowded'] > 0: insights.append({'title': 'Overcrowded Zones', 'description': f"{zone['n_overcrowded']} zones exceed 90% capacity.", 'status': 'warning'})
    if bn['n_critical'] > 0: insights.append({'title': 'Critical Bottlenecks', 'description': f"{bn['n_critical']} critical congestion points need attention.", 'status': 'warning'})
    if zone['n_underutilized'] > 0: insights.append({'title': 'Underutilized Space', 'description': f"{zone['n_underutilized']} zones below 50% - potential for reallocation.", 'status': 'neutral'})
    if bn['n_bottlenecks'] == 0: insights.append({'title': 'Good Capacity Management', 'description': 'All zones within acceptable capacity limits.', 'status': 'positive'})
    return insights

@router.post("/traffic-analysis")
async def analyze_traffic(request: TrafficRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 5: raise HTTPException(status_code=400, detail="Need at least 5 records")
        results, visualizations = {}, {}
        
        cong = analyze_congestion(df, request.time_col, request.visitor_col)
        results['congestion'] = cong
        visualizations['congestion_chart'] = create_congestion_chart(cong)
        
        zone = analyze_zone_density(df, request.zone_col, request.visitor_col, request.capacity_col)
        results['zone'] = zone
        visualizations['zone_chart'] = create_zone_chart(zone)
        
        flow = analyze_flow(df, request.zone_col, request.dwell_col, request.flow_in_col, request.flow_out_col)
        results['flow'] = flow
        if not flow.get('error'): visualizations['flow_chart'] = create_flow_chart(flow)
        
        bn = detect_bottlenecks(df, request.zone_col, request.time_col, request.visitor_col, request.capacity_col)
        results['bottleneck'] = bn
        visualizations['bottleneck_chart'] = create_bottleneck_chart(bn)
        
        opt = simulate_optimization(zone, bn)
        results['optimization'] = opt
        visualizations['optimization_chart'] = create_optimization_chart(opt)
        
        report = generate_report(cong, zone, flow, bn, opt)
        insights = generate_insights(cong, zone, bn, opt)
        
        summary = {'n_records': len(df), 'n_zones': len(zone['zone_stats']), 'peak_hour': cong['peak_hour'],
                   'busiest_zone': zone['busiest_zone']['zone'], 'n_bottlenecks': bn['n_bottlenecks'],
                   'avg_dwell_time': flow.get('avg_dwell_time', 0)}
        
        return {'success': True, 'results': results, 'visualizations': visualizations, 'report': report, 'key_insights': insights, 'summary': summary}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
