"""
Internal Communication Analysis API
5-step framework for organizational communication effectiveness
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

router = APIRouter()

class CommunicationRequest(BaseModel):
    data: List[Dict[str, Any]]
    dept_col: str
    message_col: str
    channel_cols: Optional[List[str]] = None
    response_time_col: Optional[str] = None
    meeting_col: Optional[str] = None
    performance_col: Optional[str] = None
    collaboration_cols: Optional[List[str]] = None

def _to_native(obj):
    if obj is None: return None
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# Step 1: Current Status (Baseline)
def analyze_baseline(df: pd.DataFrame, dept_col: str, message_col: str, channel_cols: Optional[List[str]]) -> Dict:
    messages = pd.to_numeric(df[message_col], errors='coerce')
    n_records = len(df)
    total_messages = int(messages.sum())
    avg_messages = messages.mean()
    
    # Communication density assessment
    if avg_messages >= 50:
        density_level = 'High'
        density_status = 'very active communication culture'
    elif avg_messages >= 30:
        density_level = 'Moderate'
        density_status = 'healthy communication levels'
    elif avg_messages >= 15:
        density_level = 'Low'
        density_status = 'below typical engagement'
    else:
        density_level = 'Critical'
        density_status = 'significant communication gaps'
    
    # Department analysis
    dept_stats = []
    for dept in df[dept_col].unique():
        mask = df[dept_col] == dept
        dept_msgs = messages[mask]
        dept_stats.append({
            'department': str(dept),
            'count': int(mask.sum()),
            'total_messages': int(dept_msgs.sum()),
            'avg_messages': _to_native(dept_msgs.mean()),
            'std_messages': _to_native(dept_msgs.std()),
            'activity_level': 'High' if dept_msgs.mean() > avg_messages * 1.2 else 'Low' if dept_msgs.mean() < avg_messages * 0.8 else 'Average'
        })
    dept_stats = sorted(dept_stats, key=lambda x: x['avg_messages'] or 0, reverse=True)
    
    most_active = dept_stats[0] if dept_stats else None
    least_active = dept_stats[-1] if dept_stats else None
    activity_gap = most_active['avg_messages'] - least_active['avg_messages'] if most_active and least_active else 0
    
    result = {
        'n_records': n_records,
        'total_messages': total_messages,
        'avg_messages': _to_native(avg_messages),
        'messages_std': _to_native(messages.std()),
        'messages_median': _to_native(messages.median()),
        'density_level': density_level,
        'density_status': density_status,
        'dept_stats': dept_stats,
        'most_active_dept': most_active['department'] if most_active else None,
        'most_active_avg': most_active['avg_messages'] if most_active else None,
        'least_active_dept': least_active['department'] if least_active else None,
        'least_active_avg': least_active['avg_messages'] if least_active else None,
        'activity_gap': _to_native(activity_gap),
        'n_departments': len(dept_stats)
    }
    
    # Channel distribution if available
    if channel_cols:
        channel_stats = []
        for col in channel_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce')
                channel_stats.append({
                    'channel': col,
                    'total': int(vals.sum()),
                    'avg': _to_native(vals.mean()),
                    'share': _to_native(vals.sum() / total_messages * 100) if total_messages > 0 else 0
                })
        channel_stats = sorted(channel_stats, key=lambda x: x['total'], reverse=True)
        result['channel_stats'] = channel_stats
        result['primary_channel'] = channel_stats[0]['channel'] if channel_stats else None
    
    return result


def create_baseline_chart(baseline: Dict, df: pd.DataFrame, dept_col: str, message_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Department activity
    dept_stats = baseline['dept_stats'][:10]
    depts = [d['department'][:12] for d in dept_stats]
    avgs = [d['avg_messages'] for d in dept_stats]
    colors = ['#10b981' if d['activity_level'] == 'High' else '#ef4444' if d['activity_level'] == 'Low' else '#3b82f6' for d in dept_stats]
    
    axes[0].barh(depts, avgs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=baseline['avg_messages'], color='red', linestyle='--', label=f"Org Avg: {baseline['avg_messages']:.0f}")
    axes[0].set_xlabel('Avg Messages')
    axes[0].set_title('Communication Activity by Department', fontsize=11, fontweight='bold')
    axes[0].legend()
    
    # Chart 2: Channel distribution or message histogram
    if baseline.get('channel_stats'):
        channels = [c['channel'][:12] for c in baseline['channel_stats']]
        shares = [c['share'] for c in baseline['channel_stats']]
        axes[1].pie(shares, labels=channels, autopct='%1.1f%%', colors=['#3b82f6', '#10b981', '#f97316', '#8b5cf6', '#ec4899'])
        axes[1].set_title('Channel Distribution', fontsize=11, fontweight='bold')
    else:
        messages = pd.to_numeric(df[message_col], errors='coerce')
        axes[1].hist(messages.dropna(), bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=baseline['avg_messages'], color='red', linestyle='--', label=f"Avg: {baseline['avg_messages']:.0f}")
        axes[1].set_xlabel('Messages')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Message Distribution', fontsize=11, fontweight='bold')
        axes[1].legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 2: Comparison (Gap Analysis)
def analyze_comparison(df: pd.DataFrame, dept_col: str, message_col: str, 
                       response_time_col: Optional[str], performance_col: Optional[str]) -> Dict:
    messages = pd.to_numeric(df[message_col], errors='coerce')
    
    result = {
        'comparison_type': 'department',
        'n_groups': df[dept_col].nunique()
    }
    
    # Response time analysis
    if response_time_col and response_time_col in df.columns:
        response = pd.to_numeric(df[response_time_col], errors='coerce')
        dept_response = []
        for dept in df[dept_col].unique():
            mask = df[dept_col] == dept
            avg_resp = response[mask].mean()
            dept_response.append({
                'department': str(dept),
                'avg_response_time': _to_native(avg_resp),
                'response_rating': 'Fast' if avg_resp < response.mean() * 0.8 else 'Slow' if avg_resp > response.mean() * 1.2 else 'Average'
            })
        dept_response = sorted(dept_response, key=lambda x: x['avg_response_time'] or float('inf'))
        result['response_stats'] = dept_response
        result['fastest_dept'] = dept_response[0]['department'] if dept_response else None
        result['fastest_time'] = dept_response[0]['avg_response_time'] if dept_response else None
        result['slowest_dept'] = dept_response[-1]['department'] if dept_response else None
        result['slowest_time'] = dept_response[-1]['avg_response_time'] if dept_response else None
        result['response_gap'] = _to_native(dept_response[-1]['avg_response_time'] - dept_response[0]['avg_response_time']) if dept_response else 0
        result['avg_response_time'] = _to_native(response.mean())
    
    # Performance-based comparison
    if performance_col and performance_col in df.columns:
        perf = pd.to_numeric(df[performance_col], errors='coerce')
        median_perf = perf.median()
        
        high_perf_mask = perf >= median_perf
        low_perf_mask = perf < median_perf
        
        high_perf_msgs = messages[high_perf_mask].mean()
        low_perf_msgs = messages[low_perf_mask].mean()
        
        high_perf_resp = None
        low_perf_resp = None
        if response_time_col and response_time_col in df.columns:
            response = pd.to_numeric(df[response_time_col], errors='coerce')
            high_perf_resp = response[high_perf_mask].mean()
            low_perf_resp = response[low_perf_mask].mean()
        
        result['performance_comparison'] = {
            'high_performers': {
                'count': int(high_perf_mask.sum()),
                'avg_messages': _to_native(high_perf_msgs),
                'avg_response_time': _to_native(high_perf_resp)
            },
            'low_performers': {
                'count': int(low_perf_mask.sum()),
                'avg_messages': _to_native(low_perf_msgs),
                'avg_response_time': _to_native(low_perf_resp)
            },
            'message_difference': _to_native(high_perf_msgs - low_perf_msgs),
            'message_diff_pct': _to_native((high_perf_msgs - low_perf_msgs) / low_perf_msgs * 100) if low_perf_msgs > 0 else 0,
            'response_difference': _to_native(low_perf_resp - high_perf_resp) if high_perf_resp and low_perf_resp else None
        }
        
        # Statistical test
        high_msgs = messages[high_perf_mask].dropna()
        low_msgs = messages[low_perf_mask].dropna()
        if len(high_msgs) > 5 and len(low_msgs) > 5:
            t_stat, p_val = stats.ttest_ind(high_msgs, low_msgs)
            result['performance_comparison']['t_statistic'] = _to_native(t_stat)
            result['performance_comparison']['p_value'] = _to_native(p_val)
            result['performance_comparison']['significant'] = bool(p_val < 0.05)
    
    # Collaboration index (cross-department communication)
    dept_collaboration = []
    unique_depts = df[dept_col].unique()
    for dept in unique_depts:
        dept_mask = df[dept_col] == dept
        cross_dept_ratio = messages[dept_mask].std() / messages[dept_mask].mean() if messages[dept_mask].mean() > 0 else 0
        dept_collaboration.append({
            'department': str(dept),
            'collaboration_index': _to_native(1 / (1 + cross_dept_ratio)),  # Higher = more consistent
            'consistency': 'High' if cross_dept_ratio < 0.5 else 'Low' if cross_dept_ratio > 1 else 'Medium'
        })
    result['collaboration_index'] = dept_collaboration
    
    return result


def create_comparison_chart(comparison: Dict, df: pd.DataFrame, dept_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Response time by department
    if comparison.get('response_stats'):
        resp_stats = comparison['response_stats'][:8]
        depts = [r['department'][:12] for r in resp_stats]
        times = [r['avg_response_time'] or 0 for r in resp_stats]
        colors = ['#10b981' if r['response_rating'] == 'Fast' else '#ef4444' if r['response_rating'] == 'Slow' else '#3b82f6' for r in resp_stats]
        
        axes[0].barh(depts, times, color=colors, alpha=0.8, edgecolor='black')
        if comparison.get('avg_response_time'):
            axes[0].axvline(x=comparison['avg_response_time'], color='red', linestyle='--', label=f"Avg: {comparison['avg_response_time']:.1f}")
        axes[0].set_xlabel('Avg Response Time')
        axes[0].set_title('Response Time by Department', fontsize=11, fontweight='bold')
        axes[0].legend()
    
    # Chart 2: High vs Low performer comparison
    if comparison.get('performance_comparison'):
        perf = comparison['performance_comparison']
        categories = ['Messages', 'Response Time']
        high_vals = [perf['high_performers']['avg_messages'] or 0, perf['high_performers']['avg_response_time'] or 0]
        low_vals = [perf['low_performers']['avg_messages'] or 0, perf['low_performers']['avg_response_time'] or 0]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1].bar(x - width/2, high_vals, width, label='High Performers', color='#10b981', alpha=0.8)
        axes[1].bar(x + width/2, low_vals, width, label='Low Performers', color='#ef4444', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].set_title('High vs Low Performer Communication', fontsize=11, fontweight='bold')
        axes[1].legend()
    else:
        # Collaboration index
        collab = comparison.get('collaboration_index', [])[:8]
        depts = [c['department'][:12] for c in collab]
        indices = [c['collaboration_index'] or 0 for c in collab]
        axes[1].bar(depts, indices, color='#3b82f6', alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Collaboration Index')
        axes[1].set_title('Collaboration Consistency', fontsize=11, fontweight='bold')
        plt.sca(axes[1])
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 3: Causality (Drivers)
def analyze_causality(df: pd.DataFrame, message_col: str, meeting_col: Optional[str],
                      performance_col: Optional[str], collaboration_cols: Optional[List[str]]) -> Dict:
    messages = pd.to_numeric(df[message_col], errors='coerce')
    
    correlations = []
    
    # Meeting frequency vs productivity
    if meeting_col and meeting_col in df.columns:
        meetings = pd.to_numeric(df[meeting_col], errors='coerce')
        valid = messages.notna() & meetings.notna()
        if valid.sum() > 10:
            corr, p_val = stats.pearsonr(meetings[valid], messages[valid])
            correlations.append({
                'factor': meeting_col,
                'factor_type': 'Meeting Frequency',
                'correlation': _to_native(corr),
                'p_value': _to_native(p_val),
                'significant': bool(p_val < 0.05),
                'interpretation': 'More meetings correlate with more messages' if corr > 0 else 'More meetings correlate with fewer messages',
                'strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            })
    
    # Performance correlation
    if performance_col and performance_col in df.columns:
        perf = pd.to_numeric(df[performance_col], errors='coerce')
        valid = messages.notna() & perf.notna()
        if valid.sum() > 10:
            corr, p_val = stats.pearsonr(messages[valid], perf[valid])
            correlations.append({
                'factor': performance_col,
                'factor_type': 'Performance',
                'correlation': _to_native(corr),
                'p_value': _to_native(p_val),
                'significant': bool(p_val < 0.05),
                'interpretation': 'More communication associates with higher performance' if corr > 0 else 'More communication associates with lower performance',
                'strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            })
        
        # Meeting-performance relationship
        if meeting_col and meeting_col in df.columns:
            meetings = pd.to_numeric(df[meeting_col], errors='coerce')
            valid = meetings.notna() & perf.notna()
            if valid.sum() > 10:
                corr, p_val = stats.pearsonr(meetings[valid], perf[valid])
                correlations.append({
                    'factor': f"{meeting_col} vs {performance_col}",
                    'factor_type': 'Meeting-Performance',
                    'correlation': _to_native(corr),
                    'p_value': _to_native(p_val),
                    'significant': bool(p_val < 0.05),
                    'interpretation': 'More meetings associate with higher performance' if corr > 0 else 'More meetings may hinder performance',
                    'strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
                })
    
    # Collaboration factors
    if collaboration_cols:
        for col in collaboration_cols:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors='coerce')
            valid = messages.notna() & vals.notna()
            if valid.sum() < 10:
                continue
            corr, p_val = stats.pearsonr(messages[valid], vals[valid])
            correlations.append({
                'factor': col,
                'factor_type': 'Collaboration',
                'correlation': _to_native(corr),
                'p_value': _to_native(p_val),
                'significant': bool(p_val < 0.05),
                'interpretation': f"Higher {col} correlates with {'more' if corr > 0 else 'less'} communication",
                'strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            })
    
    correlations = sorted(correlations, key=lambda x: abs(x.get('correlation') or 0), reverse=True)
    
    # Identify communication barriers vs enablers
    barriers = [c for c in correlations if c['correlation'] < -0.2 and c['significant']]
    enablers = [c for c in correlations if c['correlation'] > 0.2 and c['significant']]
    
    # Key insight
    key_insight = None
    if correlations:
        top = correlations[0]
        if top['correlation'] < -0.3 and top['significant']:
            key_insight = f"Communication barrier identified: {top['factor']} negatively impacts collaboration"
        elif top['correlation'] > 0.3 and top['significant']:
            key_insight = f"Communication enabler: {top['factor']} strongly supports collaboration"
    
    return {
        'correlations': correlations,
        'n_significant': sum(1 for c in correlations if c['significant']),
        'top_driver': correlations[0] if correlations else None,
        'barriers': barriers,
        'enablers': enablers,
        'key_insight': key_insight,
        'communication_impact': 'Positive' if any(c['correlation'] > 0.3 for c in correlations if c['factor_type'] == 'Performance') else 'Negative' if any(c['correlation'] < -0.3 for c in correlations if c['factor_type'] == 'Performance') else 'Neutral'
    }


def create_causality_chart(causality: Dict, df: pd.DataFrame, message_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Correlation bars
    corrs = causality['correlations'][:8]
    if corrs:
        factors = [c['factor'][:15] for c in corrs]
        vals = [c['correlation'] for c in corrs]
        colors = ['#10b981' if c['significant'] and c['correlation'] > 0 else '#ef4444' if c['significant'] and c['correlation'] < 0 else '#d1d5db' for c in corrs]
        
        axes[0].barh(factors, vals, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axvline(x=0, color='black', linewidth=0.5)
        axes[0].set_xlabel('Correlation with Communication')
        axes[0].set_title('Communication Drivers', fontsize=11, fontweight='bold')
    
    # Chart 2: Top driver scatter
    top = causality.get('top_driver')
    if top and top['factor'] in df.columns:
        messages = pd.to_numeric(df[message_col], errors='coerce')
        driver = pd.to_numeric(df[top['factor']], errors='coerce')
        valid = messages.notna() & driver.notna()
        
        axes[1].scatter(driver[valid], messages[valid], alpha=0.5, color='#3b82f6')
        if valid.sum() > 5:
            z = np.polyfit(driver[valid], messages[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(driver[valid].min(), driver[valid].max(), 100)
            axes[1].plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
        axes[1].set_xlabel(top['factor'])
        axes[1].set_ylabel('Messages')
        axes[1].set_title(f"Top Driver (r={top['correlation']:.3f})", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 4: Deep Evaluation (Validity Check)
def analyze_validity(df: pd.DataFrame, dept_col: str, message_col: str, 
                     comparison: Dict, causality: Dict) -> Dict:
    messages = pd.to_numeric(df[message_col], errors='coerce')
    
    result = {
        'sample_size': len(df),
        'statistical_power': 'High' if len(df) > 100 else 'Moderate' if len(df) > 50 else 'Limited',
        'findings_validity': []
    }
    
    # ANOVA test for department differences
    dept_groups = [messages[df[dept_col] == dept].dropna() for dept in df[dept_col].unique()]
    valid_groups = [g for g in dept_groups if len(g) >= 3]
    
    if len(valid_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*valid_groups)
        result['anova'] = {
            'f_statistic': _to_native(f_stat),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05),
            'interpretation': 'Significant differences exist between departments' if p_val < 0.05 else 'No significant differences between departments'
        }
        result['findings_validity'].append({
            'finding': 'Department communication differences',
            'test': 'ANOVA',
            'p_value': _to_native(p_val),
            'valid': bool(p_val < 0.05),
            'confidence': '95%' if p_val < 0.05 else 'Not significant'
        })
    
    # Validate performance comparison
    if comparison.get('performance_comparison', {}).get('p_value'):
        p_val = comparison['performance_comparison']['p_value']
        result['findings_validity'].append({
            'finding': 'High vs Low performer communication gap',
            'test': 'Independent t-test',
            'p_value': _to_native(p_val),
            'valid': bool(p_val < 0.05),
            'confidence': '95%' if p_val < 0.05 else 'Not significant'
        })
    
    # Validate causality findings
    for corr in causality.get('correlations', [])[:3]:
        result['findings_validity'].append({
            'finding': f"{corr['factor']} correlation",
            'test': 'Pearson correlation',
            'p_value': corr['p_value'],
            'valid': corr['significant'],
            'confidence': '95%' if corr['significant'] else 'Not significant'
        })
    
    # Effect size calculations
    if comparison.get('performance_comparison'):
        perf = comparison['performance_comparison']
        high_avg = perf['high_performers']['avg_messages'] or 0
        low_avg = perf['low_performers']['avg_messages'] or 0
        pooled_std = messages.std()
        cohens_d = abs(high_avg - low_avg) / pooled_std if pooled_std > 0 else 0
        
        result['effect_size'] = {
            'cohens_d': _to_native(cohens_d),
            'interpretation': 'Large' if cohens_d > 0.8 else 'Medium' if cohens_d > 0.5 else 'Small' if cohens_d > 0.2 else 'Negligible'
        }
    
    # Overall validity assessment
    n_valid = sum(1 for f in result['findings_validity'] if f['valid'])
    n_total = len(result['findings_validity'])
    result['overall_validity'] = {
        'valid_findings': n_valid,
        'total_findings': n_total,
        'validity_rate': _to_native(n_valid / n_total * 100) if n_total > 0 else 0,
        'assessment': 'Strong' if n_valid / n_total >= 0.7 and n_total >= 3 else 'Moderate' if n_valid / n_total >= 0.5 else 'Weak',
        'is_structural': n_valid >= 2 and result['statistical_power'] in ['High', 'Moderate']
    }
    
    return result


def create_validity_chart(validity: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Findings validity
    findings = validity.get('findings_validity', [])
    if findings:
        names = [f['finding'][:20] for f in findings]
        p_vals = [f['p_value'] or 1 for f in findings]
        colors = ['#10b981' if f['valid'] else '#ef4444' for f in findings]
        
        axes[0].barh(names, [-np.log10(max(p, 0.0001)) for p in p_vals], color=colors, alpha=0.8, edgecolor='black')
        axes[0].axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
        axes[0].set_xlabel('-log10(p-value)')
        axes[0].set_title('Statistical Validity of Findings', fontsize=11, fontweight='bold')
        axes[0].legend()
    
    # Chart 2: Validity summary
    overall = validity.get('overall_validity', {})
    valid_count = overall.get('valid_findings', 0)
    invalid_count = overall.get('total_findings', 0) - valid_count
    
    axes[1].pie([valid_count, invalid_count], labels=['Validated', 'Not Validated'], 
                colors=['#10b981', '#ef4444'], autopct='%1.0f%%', explode=(0.05, 0))
    axes[1].set_title(f"Findings Validity ({overall.get('assessment', 'N/A')})", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 5: Optimization (Simulation)
def simulate_optimization(df: pd.DataFrame, baseline: Dict, comparison: Dict, 
                          causality: Dict, meeting_col: Optional[str]) -> Dict:
    current_messages = baseline['avg_messages']
    current_response = comparison.get('avg_response_time', 60)  # Default 60 min
    
    scenarios = []
    
    # Scenario 1: Reduce meetings by 20%
    if meeting_col:
        meeting_reduction = 20
        # Assume reducing meetings by 20% improves focus time by 10%
        productivity_gain = meeting_reduction * 0.5
        new_response = current_response * 0.9  # 10% faster response
        scenarios.append({
            'name': 'Reduce Meetings 20%',
            'description': 'Cut unnecessary meetings, use async communication',
            'intervention': 'Meeting time reduction',
            'change': f"-{meeting_reduction}% meetings",
            'productivity_gain': _to_native(productivity_gain),
            'new_response_time': _to_native(new_response),
            'response_improvement': _to_native((current_response - new_response) / current_response * 100),
            'roi_estimate': _to_native(productivity_gain * 2.5),  # Hours saved per week
            'recommended': False
        })
    
    # Scenario 2: Unified communication platform
    channel_consolidation = 15 if baseline.get('channel_stats') and len(baseline['channel_stats']) > 2 else 8
    scenarios.append({
        'name': 'Unify Communication Channels',
        'description': 'Consolidate tools, reduce context switching',
        'intervention': 'Platform consolidation',
        'change': f"-{len(baseline.get('channel_stats', [])) - 1} platforms",
        'productivity_gain': _to_native(channel_consolidation),
        'new_response_time': _to_native(current_response * 0.85),
        'response_improvement': _to_native(15),
        'roi_estimate': _to_native(channel_consolidation * 3),
        'recommended': True
    })
    
    # Scenario 3: Cross-department collaboration initiative
    dept_gap = baseline.get('activity_gap', 0)
    if dept_gap > 10:
        gap_reduction = min(dept_gap * 0.4, 20)
        scenarios.append({
            'name': 'Break Down Silos',
            'description': 'Cross-functional projects, shared channels',
            'intervention': 'Silo elimination',
            'change': f"+{gap_reduction:.0f}% cross-dept communication",
            'productivity_gain': _to_native(gap_reduction * 0.8),
            'new_response_time': _to_native(current_response * 0.88),
            'response_improvement': _to_native(12),
            'roi_estimate': _to_native(gap_reduction * 2),
            'recommended': False
        })
    
    # Scenario 4: Comprehensive communication overhaul
    scenarios.append({
        'name': 'Full Communication Redesign',
        'description': 'New tools, training, culture change',
        'intervention': 'Complete overhaul',
        'change': 'Comprehensive transformation',
        'productivity_gain': _to_native(25),
        'new_response_time': _to_native(current_response * 0.7),
        'response_improvement': _to_native(30),
        'roi_estimate': _to_native(40),
        'recommended': False
    })
    
    best = max(scenarios, key=lambda x: x['roi_estimate'] or 0)
    
    # Lead time impact calculation
    if meeting_col and comparison.get('avg_response_time'):
        lead_time_reduction = best['response_improvement'] * 0.6  # 60% of response improvement translates to lead time
    else:
        lead_time_reduction = best['productivity_gain'] * 0.5
    
    return {
        'current_state': {
            'avg_messages': current_messages,
            'avg_response_time': current_response,
            'n_channels': len(baseline.get('channel_stats', [])),
            'activity_gap': baseline.get('activity_gap', 0)
        },
        'scenarios': scenarios,
        'best_scenario': best,
        'lead_time_reduction': _to_native(lead_time_reduction),
        'recommendations': [
            f"Priority: {best['name']} - highest ROI of {best['roi_estimate']:.0f}%",
            "Start with quick wins: Reduce unnecessary meetings",
            "Implement communication guidelines and training",
            "Monitor response times and collaboration metrics monthly",
            "Conduct quarterly communication health assessments"
        ]
    }


def create_optimization_chart(optimization: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = optimization['scenarios']
    
    # Chart 1: Productivity gain
    names = [s['name'][:15] for s in scenarios]
    gains = [s['productivity_gain'] or 0 for s in scenarios]
    colors = ['#10b981' if s['recommended'] else '#3b82f6' for s in scenarios]
    
    axes[0].barh(names, gains, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Productivity Gain (%)')
    axes[0].set_title('Projected Productivity Impact', fontsize=11, fontweight='bold')
    
    # Chart 2: Response time improvement
    current = optimization['current_state']['avg_response_time']
    times = ['Current'] + [s['name'][:12] for s in scenarios]
    values = [current] + [s['new_response_time'] or current for s in scenarios]
    bar_colors = ['#ef4444'] + ['#10b981' if s['recommended'] else '#3b82f6' for s in scenarios]
    
    axes[1].bar(times, values, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Response Time')
    axes[1].set_title('Projected Response Time', fontsize=11, fontweight='bold')
    plt.sca(axes[1])
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Report Generation
def generate_report(baseline: Dict, comparison: Dict, causality: Dict, 
                    validity: Dict, optimization: Dict) -> Dict:
    report = {}
    
    # Step 1: Baseline
    most_active = baseline.get('most_active_dept', 'N/A')
    least_active = baseline.get('least_active_dept', 'N/A')
    activity_gap = baseline.get('activity_gap', 0)
    
    report['step1_baseline'] = {
        'title': '1. Communication Baseline Assessment',
        'question': 'What is the current state of internal communication across the organization?',
        'finding': f"Communication density: {baseline['density_level']} ({baseline['avg_messages']:.0f} avg messages), Most active: {most_active}, Gap: {activity_gap:.0f} messages",
        'detail': (f"Organizational communication analysis across {baseline['n_departments']} departments and {baseline['n_records']:,} data points reveals "
                  f"an average communication volume of {baseline['avg_messages']:.0f} messages per unit (median: {baseline['messages_median']:.0f}, std: {baseline['messages_std']:.1f}), "
                  f"classified as {baseline['density_level']} density level indicating {baseline['density_status']}. "
                  f"Department-level analysis identifies '{most_active}' as the most communicative unit ({baseline['most_active_avg']:.0f} avg messages), "
                  f"while '{least_active}' shows lowest activity ({baseline['least_active_avg']:.0f} avg messages)—a gap of {activity_gap:.0f} messages ({activity_gap/baseline['avg_messages']*100:.0f}% variance). "
                  + (f"Channel distribution: Primary channel is '{baseline.get('primary_channel', 'N/A')}' accounting for {baseline['channel_stats'][0]['share']:.0f}% of total volume. " if baseline.get('channel_stats') else "")
                  + "Organizational insight: " + ("Significant activity disparity suggests potential silo formation or uneven workload distribution. " if activity_gap > baseline['avg_messages'] * 0.5 else "Relatively balanced communication activity across departments. ")
                  + "Recommendation: Investigate low-activity departments for potential engagement issues or communication barriers.")
    }
    
    # Step 2: Comparison
    if comparison.get('performance_comparison'):
        perf = comparison['performance_comparison']
        diff_pct = perf.get('message_diff_pct', 0)
        sig_text = "statistically significant (p<0.05)" if perf.get('significant') else "not statistically significant"
        report['step2_comparison'] = {
            'title': '2. High vs Low Performer Communication Gap Analysis',
            'question': 'How do communication patterns differ between high and low performing teams?',
            'finding': f"High performers: {perf['high_performers']['avg_messages']:.0f} msgs vs Low performers: {perf['low_performers']['avg_messages']:.0f} msgs ({diff_pct:+.1f}% difference, {sig_text})",
            'detail': (f"Performance-based segmentation comparing {perf['high_performers']['count']} high performers against {perf['low_performers']['count']} low performers reveals distinct communication patterns. "
                      f"High-performing teams average {perf['high_performers']['avg_messages']:.0f} messages versus {perf['low_performers']['avg_messages']:.0f} for lower performers—"
                      f"a {abs(diff_pct):.1f}% {'higher' if diff_pct > 0 else 'lower'} communication rate. This difference is {sig_text}. "
                      + (f"Response time analysis: High performers respond in {perf['high_performers']['avg_response_time']:.0f} minutes vs {perf['low_performers']['avg_response_time']:.0f} for low performers "
                         f"({perf['response_difference']:.0f} minute gap). " if perf.get('response_difference') else "")
                      + "Key insight: " + ("High performers demonstrate significantly more active communication, suggesting communication volume correlates with team effectiveness. " if diff_pct > 10 else "Communication volume differences are modest, suggesting quality over quantity may be the differentiator. ")
                      + "Action item: Document and replicate high-performer communication practices across the organization.")
        }
    elif comparison.get('response_stats'):
        report['step2_comparison'] = {
            'title': '2. Department Response Time Comparison',
            'question': 'Which departments respond fastest and slowest?',
            'finding': f"Fastest: {comparison['fastest_dept']} ({comparison['fastest_time']:.0f} min), Slowest: {comparison['slowest_dept']} ({comparison['slowest_time']:.0f} min), Gap: {comparison['response_gap']:.0f} min",
            'detail': (f"Response time analysis across departments reveals significant variation in communication responsiveness. "
                      f"'{comparison['fastest_dept']}' leads with average {comparison['fastest_time']:.0f} minute response time, "
                      f"while '{comparison['slowest_dept']}' lags at {comparison['slowest_time']:.0f} minutes—a {comparison['response_gap']:.0f} minute gap. "
                      f"Organization-wide average: {comparison['avg_response_time']:.0f} minutes. "
                      "Slow response times often indicate workflow bottlenecks, unclear ownership, or communication overload. "
                      "Recommendation: Establish response time SLAs and investigate root causes in slow-responding departments.")
        }
    else:
        report['step2_comparison'] = {'title': '2. Gap Analysis', 'question': 'Performance comparison', 'finding': 'Configure performance or response time columns', 'detail': 'Add performance metrics or response time data for comparative analysis.'}
    
    # Step 3: Causality
    if causality.get('correlations'):
        top = causality.get('top_driver', {})
        n_barriers = len(causality.get('barriers', []))
        n_enablers = len(causality.get('enablers', []))
        report['step3_causality'] = {
            'title': '3. Communication Driver Analysis',
            'question': 'What factors drive or hinder effective communication?',
            'finding': f"Top driver: {top.get('factor', 'N/A')} (r={top.get('correlation', 0):.3f}, {top.get('strength', 'N/A')}), Impact: {causality['communication_impact']}",
            'detail': (f"Causal analysis of {len(causality['correlations'])} factors identifies {causality['n_significant']} with statistically significant relationships to communication effectiveness. "
                      f"'{top.get('factor', 'N/A')}' emerges as the primary driver with correlation r={top.get('correlation', 0):.3f} ({top.get('strength', 'N/A')} relationship)—"
                      f"{top.get('interpretation', 'N/A')}. "
                      f"Communication barriers identified: {n_barriers}; Communication enablers: {n_enablers}. "
                      + (f"Critical insight: {causality['key_insight']}. " if causality.get('key_insight') else "")
                      + "Strategic implication: " + ("Meeting load appears to negatively impact productive communication—consider async alternatives. " if any(c['factor_type'] == 'Meeting-Performance' and c['correlation'] < -0.2 for c in causality.get('correlations', [])) else "")
                      + "Focus resources on strengthening enablers while systematically reducing barriers. "
                      "Monitor these drivers as leading indicators of collaboration health.")
        }
    else:
        report['step3_causality'] = {'title': '3. Causality Analysis', 'question': 'Communication drivers', 'finding': 'Insufficient data for causality analysis', 'detail': 'Add meeting, performance, or collaboration factor columns for driver analysis.'}
    
    # Step 4: Validity
    overall = validity.get('overall_validity', {})
    n_valid = overall.get('valid_findings', 0)
    n_total = overall.get('total_findings', 0)
    is_structural = overall.get('is_structural', False)
    
    report['step4_validity'] = {
        'title': '4. Statistical Validity & Structural Assessment',
        'question': 'Are the identified communication patterns statistically valid and structural?',
        'finding': f"Validity: {overall.get('assessment', 'N/A')} ({n_valid}/{n_total} findings validated), Statistical power: {validity['statistical_power']}, Structural: {'Yes' if is_structural else 'No'}",
        'detail': (f"Statistical validation of {n_total} key findings confirms {n_valid} ({overall.get('validity_rate', 0):.0f}%) at 95% confidence level. "
                  f"Sample size of {validity['sample_size']} provides {validity['statistical_power'].lower()} statistical power for detecting meaningful effects. "
                  + (f"ANOVA test for department differences: F={validity['anova']['f_statistic']:.2f}, p={validity['anova']['p_value']:.4f}—{validity['anova']['interpretation']}. " if validity.get('anova') else "")
                  + (f"Effect size (Cohen's d): {validity['effect_size']['cohens_d']:.2f} ({validity['effect_size']['interpretation']} effect). " if validity.get('effect_size') else "")
                  + f"Assessment: The identified communication patterns are {'confirmed as structural organizational characteristics that will persist without intervention' if is_structural else 'potentially situational and may not represent permanent organizational patterns—continued monitoring recommended'}. "
                  + "Implication: " + ("These findings represent entrenched organizational dynamics requiring systematic intervention. " if is_structural else "Consider longitudinal analysis to confirm pattern persistence before major investments."))
    }
    
    # Step 5: Optimization
    best = optimization.get('best_scenario', {})
    lead_time = optimization.get('lead_time_reduction', 0)
    report['step5_optimization'] = {
        'title': '5. Communication Optimization & ROI Simulation',
        'question': 'What improvements are achievable through communication optimization?',
        'finding': f"Best scenario: {best.get('name', 'N/A')} → +{best.get('productivity_gain', 0):.0f}% productivity, -{best.get('response_improvement', 0):.0f}% response time, ROI: {best.get('roi_estimate', 0):.0f}%",
        'detail': (f"Optimization simulation models four intervention scenarios against current baseline "
                  f"(avg messages: {optimization['current_state']['avg_messages']:.0f}, response time: {optimization['current_state']['avg_response_time']:.0f} min). "
                  f"Scenario comparison: " + "; ".join([f"{s['name']}: +{s['productivity_gain']:.0f}% productivity" for s in optimization['scenarios']]) + ". "
                  f"'{best.get('name', 'N/A')}' ({best.get('description', 'N/A')}) offers optimal ROI with projected {best.get('productivity_gain', 0):.0f}% productivity gain "
                  f"and {best.get('response_improvement', 0):.0f}% response time improvement. "
                  f"Estimated lead time reduction: {lead_time:.0f}% faster project completion. "
                  "Implementation roadmap: Begin with low-hanging fruit (meeting reduction), then proceed to platform consolidation. "
                  "Establish baseline metrics before changes, measure at 30/60/90 days post-implementation. "
                  "Expected payback period: 3-6 months for technology investments, immediate for process changes.")
    }
    
    return report


def generate_insights(baseline: Dict, comparison: Dict, causality: Dict, 
                      validity: Dict, optimization: Dict) -> List[Dict]:
    insights = []
    
    # Silo detection
    if baseline.get('activity_gap', 0) > baseline['avg_messages'] * 0.5:
        insights.append({
            'title': 'Department Silo Detected',
            'description': f"Communication gap of {baseline['activity_gap']:.0f} messages between {baseline['most_active_dept']} and {baseline['least_active_dept']} suggests silo formation.",
            'status': 'warning'
        })
    
    # Performance-communication link
    if comparison.get('performance_comparison', {}).get('significant'):
        diff = comparison['performance_comparison']['message_diff_pct']
        insights.append({
            'title': 'Communication-Performance Link',
            'description': f"High performers communicate {abs(diff):.0f}% {'more' if diff > 0 else 'less'} than low performers (statistically significant).",
            'status': 'positive' if diff > 0 else 'warning'
        })
    
    # Meeting overload
    meeting_barrier = next((c for c in causality.get('correlations', []) if c['factor_type'] == 'Meeting-Performance' and c['correlation'] < -0.2), None)
    if meeting_barrier:
        insights.append({
            'title': 'Meeting Overload Risk',
            'description': f"Meetings negatively correlate with performance (r={meeting_barrier['correlation']:.3f}). Consider async alternatives.",
            'status': 'warning'
        })
    
    # Structural issues
    if validity.get('overall_validity', {}).get('is_structural'):
        insights.append({
            'title': 'Structural Communication Issues',
            'description': 'Identified patterns are statistically validated as structural—require systematic intervention.',
            'status': 'warning'
        })
    
    # High ROI opportunity
    best = optimization.get('best_scenario', {})
    if best.get('roi_estimate', 0) > 20:
        insights.append({
            'title': f"High-ROI Opportunity: {best['name']}",
            'description': f"Projected {best['roi_estimate']:.0f}% ROI with {best['productivity_gain']:.0f}% productivity gain. Recommend immediate implementation.",
            'status': 'positive'
        })
    
    return insights


@router.post("/internal-communication")
async def analyze_communication(request: CommunicationRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 records for reliable analysis")
        
        results, visualizations = {}, {}
        
        # Step 1: Baseline
        baseline = analyze_baseline(df, request.dept_col, request.message_col, request.channel_cols)
        results['baseline'] = baseline
        visualizations['baseline_chart'] = create_baseline_chart(baseline, df, request.dept_col, request.message_col)
        
        # Step 2: Comparison
        comparison = analyze_comparison(df, request.dept_col, request.message_col, 
                                        request.response_time_col, request.performance_col)
        results['comparison'] = comparison
        visualizations['comparison_chart'] = create_comparison_chart(comparison, df, request.dept_col)
        
        # Step 3: Causality
        causality = analyze_causality(df, request.message_col, request.meeting_col,
                                      request.performance_col, request.collaboration_cols)
        results['causality'] = causality
        if causality.get('correlations'):
            visualizations['causality_chart'] = create_causality_chart(causality, df, request.message_col)
        
        # Step 4: Validity
        validity = analyze_validity(df, request.dept_col, request.message_col, comparison, causality)
        results['validity'] = validity
        visualizations['validity_chart'] = create_validity_chart(validity)
        
        # Step 5: Optimization
        optimization = simulate_optimization(df, baseline, comparison, causality, request.meeting_col)
        results['optimization'] = optimization
        visualizations['optimization_chart'] = create_optimization_chart(optimization)
        
        # Generate report and insights
        report = generate_report(baseline, comparison, causality, validity, optimization)
        insights = generate_insights(baseline, comparison, causality, validity, optimization)
        
        summary = {
            'n_records': baseline['n_records'],
            'n_departments': baseline['n_departments'],
            'communication_density': baseline['density_level'],
            'avg_messages': baseline['avg_messages'],
            'activity_gap': baseline.get('activity_gap'),
            'top_driver': causality.get('top_driver', {}).get('factor') if causality.get('top_driver') else None,
            'is_structural': validity.get('overall_validity', {}).get('is_structural', False),
            'potential_productivity_gain': optimization.get('best_scenario', {}).get('productivity_gain', 0)
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'report': report,
            'key_insights': insights,
            'summary': summary
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
