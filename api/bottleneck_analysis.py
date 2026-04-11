"""
Process Bottleneck Diagnosis API
5-step framework for comprehensive process efficiency analysis
1. Stage Duration - 단계별 소요 시간
2. Failure Rate Comparison - 프로세스별 실패율 비교
3. Resource-Speed Relationship - 자원 투입과 속도 관계
4. Bottleneck Root Cause - 병목 구간 원인 규명
5. Revenue Impact Simulation - 프로세스 단축 시 수익
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class BottleneckRequest(BaseModel):
    data: List[Dict[str, Any]]
    stage_cols: List[str]  # Duration columns for each stage
    failure_col: Optional[str] = None  # Failure/success indicator
    resource_cols: Optional[List[str]] = None  # Resource input columns
    revenue_col: Optional[str] = None  # Revenue/output column
    process_col: Optional[str] = None  # Process type identifier


def _to_native(obj):
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# =============================================================================
# Step 1: Stage Duration Analysis
# =============================================================================
def analyze_stage_duration(df: pd.DataFrame, stage_cols: List[str]) -> Dict:
    stage_stats = []
    total_duration = None
    
    for col in stage_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) == 0:
            continue
        
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        p95 = values.quantile(0.95)
        
        stage_stats.append({
            'stage': col,
            'mean': _to_native(mean_val),
            'median': _to_native(median_val),
            'std': _to_native(std_val),
            'min': _to_native(values.min()),
            'max': _to_native(values.max()),
            'p95': _to_native(p95),
            'cv': _to_native(std_val / mean_val * 100) if mean_val > 0 else None
        })
    
    # Calculate total and percentages
    if stage_stats:
        total_mean = sum(s['mean'] for s in stage_stats)
        for s in stage_stats:
            s['pct_of_total'] = _to_native(s['mean'] / total_mean * 100) if total_mean > 0 else None
        
        # Identify bottleneck (longest stage)
        bottleneck = max(stage_stats, key=lambda x: x['mean'])
        most_variable = max(stage_stats, key=lambda x: x['cv'] or 0)
    else:
        total_mean = 0
        bottleneck = None
        most_variable = None
    
    return {
        'stages': stage_stats,
        'n_stages': len(stage_stats),
        'total_mean_duration': _to_native(total_mean),
        'bottleneck_stage': bottleneck,
        'most_variable_stage': most_variable
    }


def create_duration_chart(duration_data: Dict) -> str:
    stages = duration_data.get('stages', [])
    if not stages:
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Mean duration by stage
    ax1 = axes[0]
    names = [s['stage'][:12] for s in stages]
    means = [s['mean'] for s in stages]
    colors = ['#ef4444' if s['stage'] == duration_data['bottleneck_stage']['stage'] else '#3b82f6' for s in stages]
    ax1.barh(names, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Mean Duration')
    ax1.set_title('Duration by Stage', fontsize=11, fontweight='bold')
    
    # Chart 2: Percentage of total
    ax2 = axes[1]
    pcts = [s['pct_of_total'] for s in stages]
    ax2.pie(pcts, labels=names, autopct='%1.1f%%', colors=plt.cm.Blues(np.linspace(0.3, 0.9, len(stages))))
    ax2.set_title('Time Distribution', fontsize=11, fontweight='bold')
    
    # Chart 3: Variability (CV)
    ax3 = axes[2]
    cvs = [s['cv'] or 0 for s in stages]
    colors_cv = ['#ef4444' if cv > 50 else '#f59e0b' if cv > 30 else '#10b981' for cv in cvs]
    ax3.barh(names, cvs, color=colors_cv, alpha=0.7, edgecolor='black')
    ax3.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('CV (%)')
    ax3.set_title('Variability by Stage', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Failure Rate Comparison
# =============================================================================
def analyze_failure_rate(df: pd.DataFrame, failure_col: str, process_col: Optional[str] = None,
                        stage_cols: Optional[List[str]] = None) -> Dict:
    df_clean = df.copy()
    
    # Convert failure column (assume 1=failure, 0=success or similar)
    failure_values = df[failure_col].unique()
    if len(failure_values) == 2:
        # Binary - map to 0/1
        fail_map = {v: i for i, v in enumerate(sorted(failure_values, key=lambda x: str(x)))}
        df_clean['failure_binary'] = df[failure_col].map(fail_map)
    else:
        # Numeric threshold
        median_val = pd.to_numeric(df[failure_col], errors='coerce').median()
        df_clean['failure_binary'] = (pd.to_numeric(df[failure_col], errors='coerce') > median_val).astype(int)
    
    overall_rate = df_clean['failure_binary'].mean() * 100
    
    result = {
        'overall_failure_rate': _to_native(overall_rate),
        'n_total': len(df_clean),
        'n_failures': int(df_clean['failure_binary'].sum())
    }
    
    # By process type
    if process_col and process_col in df.columns:
        process_rates = []
        for proc in df[process_col].unique():
            proc_data = df_clean[df[process_col] == proc]
            rate = proc_data['failure_binary'].mean() * 100
            process_rates.append({
                'process': _to_native(proc),
                'failure_rate': _to_native(rate),
                'n': len(proc_data),
                'n_failures': int(proc_data['failure_binary'].sum())
            })
        process_rates = sorted(process_rates, key=lambda x: x['failure_rate'], reverse=True)
        result['by_process'] = process_rates
        result['worst_process'] = process_rates[0] if process_rates else None
        result['best_process'] = process_rates[-1] if process_rates else None
    
    # Failure correlation with stage durations
    if stage_cols:
        stage_correlations = []
        for col in stage_cols:
            duration = pd.to_numeric(df[col], errors='coerce')
            valid_idx = duration.notna() & df_clean['failure_binary'].notna()
            if valid_idx.sum() > 10:
                corr, p_val = stats.pointbiserialr(df_clean.loc[valid_idx, 'failure_binary'], duration[valid_idx])
                stage_correlations.append({
                    'stage': col,
                    'correlation': _to_native(corr),
                    'p_value': _to_native(p_val),
                    'significant': bool(p_val < 0.05)
                })
        result['stage_correlations'] = sorted(stage_correlations, key=lambda x: abs(x['correlation'] or 0), reverse=True)
    
    return result


def create_failure_chart(failure_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Failure rate by process
    ax1 = axes[0]
    if failure_data.get('by_process'):
        procs = [p['process'][:12] for p in failure_data['by_process']]
        rates = [p['failure_rate'] for p in failure_data['by_process']]
        colors = ['#ef4444' if r > failure_data['overall_failure_rate'] else '#10b981' for r in rates]
        ax1.barh(procs, rates, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=failure_data['overall_failure_rate'], color='gray', linestyle='--', label=f"Avg: {failure_data['overall_failure_rate']:.1f}%")
        ax1.set_xlabel('Failure Rate (%)')
        ax1.set_title('Failure Rate by Process', fontsize=11, fontweight='bold')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No process comparison', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Failure Rate by Process', fontsize=11, fontweight='bold')
    
    # Chart 2: Stage correlation with failure
    ax2 = axes[1]
    if failure_data.get('stage_correlations'):
        stages = [s['stage'][:12] for s in failure_data['stage_correlations']]
        corrs = [s['correlation'] or 0 for s in failure_data['stage_correlations']]
        colors = ['#ef4444' if c > 0 else '#10b981' for c in corrs]
        ax2.barh(stages, corrs, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='gray', linestyle='-')
        ax2.set_xlabel('Correlation with Failure')
        ax2.set_title('Stage Duration vs Failure', fontsize=11, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No stage correlation data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Resource-Speed Relationship
# =============================================================================
def analyze_resource_speed(df: pd.DataFrame, stage_cols: List[str], resource_cols: List[str]) -> Dict:
    # Calculate total duration
    df_clean = df.copy()
    for col in stage_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['total_duration'] = df_clean[stage_cols].sum(axis=1)
    
    # Speed = 1 / duration (or throughput proxy)
    df_clean['speed'] = 1 / df_clean['total_duration'].replace(0, np.nan)
    
    relationships = []
    for res_col in resource_cols:
        resource = pd.to_numeric(df[res_col], errors='coerce')
        valid_idx = resource.notna() & df_clean['speed'].notna()
        
        if valid_idx.sum() < 10:
            continue
        
        # Correlation
        corr, p_val = stats.pearsonr(resource[valid_idx], df_clean.loc[valid_idx, 'speed'])
        
        # Simple regression
        X = resource[valid_idx].values.reshape(-1, 1)
        y = df_clean.loc[valid_idx, 'speed'].values
        reg = LinearRegression().fit(X, y)
        r_squared = reg.score(X, y)
        
        relationships.append({
            'resource': res_col,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05),
            'r_squared': _to_native(r_squared),
            'coefficient': _to_native(reg.coef_[0]),
            'direction': 'positive' if corr > 0 else 'negative'
        })
    
    relationships = sorted(relationships, key=lambda x: abs(x['correlation'] or 0), reverse=True)
    
    # Most impactful resource
    most_impactful = relationships[0] if relationships else None
    
    return {
        'relationships': relationships,
        'n_resources': len(relationships),
        'most_impactful': most_impactful,
        'avg_speed': _to_native(df_clean['speed'].mean()),
        'speed_std': _to_native(df_clean['speed'].std())
    }


def create_resource_chart(resource_data: Dict, df: pd.DataFrame, stage_cols: List[str], resource_cols: List[str]) -> str:
    n_res = min(len(resource_cols), 3)
    if n_res == 0:
        return ""
    
    fig, axes = plt.subplots(1, n_res, figsize=(5 * n_res, 4))
    if n_res == 1:
        axes = [axes]
    
    # Calculate total duration
    df_clean = df.copy()
    for col in stage_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['total_duration'] = df_clean[stage_cols].sum(axis=1)
    
    for i, res_col in enumerate(resource_cols[:n_res]):
        ax = axes[i]
        resource = pd.to_numeric(df[res_col], errors='coerce')
        valid_idx = resource.notna() & df_clean['total_duration'].notna()
        
        ax.scatter(resource[valid_idx], df_clean.loc[valid_idx, 'total_duration'], alpha=0.5, color='#3b82f6')
        
        # Trend line
        z = np.polyfit(resource[valid_idx], df_clean.loc[valid_idx, 'total_duration'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(resource[valid_idx].min(), resource[valid_idx].max(), 100)
        ax.plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
        
        ax.set_xlabel(res_col)
        ax.set_ylabel('Total Duration')
        
        rel = next((r for r in resource_data['relationships'] if r['resource'] == res_col), None)
        sig = '***' if rel and rel['significant'] else ''
        ax.set_title(f'{res_col[:15]} vs Duration {sig}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Bottleneck Root Cause
# =============================================================================
def analyze_bottleneck_cause(df: pd.DataFrame, stage_cols: List[str], 
                            resource_cols: Optional[List[str]] = None,
                            process_col: Optional[str] = None) -> Dict:
    # Find bottleneck stage
    stage_means = {}
    for col in stage_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        stage_means[col] = values.mean()
    
    bottleneck_stage = max(stage_means, key=stage_means.get)
    bottleneck_duration = stage_means[bottleneck_stage]
    
    causes = []
    
    # Analyze bottleneck by process type
    if process_col and process_col in df.columns:
        process_bottleneck = df.groupby(process_col)[bottleneck_stage].mean().to_dict()
        worst_process = max(process_bottleneck, key=process_bottleneck.get)
        causes.append({
            'factor': 'Process Type',
            'finding': f"'{worst_process}' has longest {bottleneck_stage}",
            'value': _to_native(process_bottleneck[worst_process]),
            'comparison': {str(k): _to_native(v) for k, v in process_bottleneck.items()}
        })
    
    # Resource impact on bottleneck
    if resource_cols:
        for res_col in resource_cols:
            resource = pd.to_numeric(df[res_col], errors='coerce')
            bottleneck_vals = pd.to_numeric(df[bottleneck_stage], errors='coerce')
            valid_idx = resource.notna() & bottleneck_vals.notna()
            
            if valid_idx.sum() > 10:
                corr, p_val = stats.pearsonr(resource[valid_idx], bottleneck_vals[valid_idx])
                if p_val < 0.05:
                    direction = "increases" if corr > 0 else "decreases"
                    causes.append({
                        'factor': res_col,
                        'finding': f"Higher {res_col} {direction} bottleneck duration",
                        'correlation': _to_native(corr),
                        'p_value': _to_native(p_val),
                        'actionable': corr < 0  # Negative = more resource helps
                    })
    
    # Variability analysis
    bottleneck_std = pd.to_numeric(df[bottleneck_stage], errors='coerce').std()
    bottleneck_cv = bottleneck_std / bottleneck_duration * 100 if bottleneck_duration > 0 else 0
    
    if bottleneck_cv > 50:
        causes.append({
            'factor': 'High Variability',
            'finding': f"CV of {bottleneck_cv:.1f}% indicates inconsistent process",
            'value': _to_native(bottleneck_cv),
            'recommendation': 'Standardize process to reduce variation'
        })
    
    # Sequential dependency check
    correlations_with_next = []
    for i, col in enumerate(stage_cols[:-1]):
        next_col = stage_cols[i + 1]
        curr = pd.to_numeric(df[col], errors='coerce')
        next_vals = pd.to_numeric(df[next_col], errors='coerce')
        valid_idx = curr.notna() & next_vals.notna()
        if valid_idx.sum() > 10:
            corr, _ = stats.pearsonr(curr[valid_idx], next_vals[valid_idx])
            correlations_with_next.append({
                'from': col,
                'to': next_col,
                'correlation': _to_native(corr)
            })
    
    return {
        'bottleneck_stage': bottleneck_stage,
        'bottleneck_duration': _to_native(bottleneck_duration),
        'bottleneck_cv': _to_native(bottleneck_cv),
        'causes': causes,
        'n_causes_found': len(causes),
        'stage_dependencies': correlations_with_next
    }


def create_bottleneck_chart(bottleneck_data: Dict, df: pd.DataFrame, stage_cols: List[str]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Bottleneck stage distribution
    ax1 = axes[0]
    bottleneck_stage = bottleneck_data['bottleneck_stage']
    values = pd.to_numeric(df[bottleneck_stage], errors='coerce').dropna()
    ax1.hist(values, bins=20, color='#ef4444', alpha=0.7, edgecolor='black')
    ax1.axvline(values.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.1f}')
    ax1.axvline(values.median(), color='#3b82f6', linestyle='--', linewidth=2, label=f'Median: {values.median():.1f}')
    ax1.set_xlabel(bottleneck_stage)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Bottleneck: {bottleneck_stage[:20]}', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # Chart 2: Stage flow with bottleneck highlighted
    ax2 = axes[1]
    means = [pd.to_numeric(df[col], errors='coerce').mean() for col in stage_cols]
    names = [col[:10] for col in stage_cols]
    colors = ['#ef4444' if col == bottleneck_stage else '#3b82f6' for col in stage_cols]
    
    ax2.bar(names, means, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Mean Duration')
    ax2.set_title('Process Flow (Bottleneck in Red)', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Revenue Impact Simulation
# =============================================================================
def simulate_revenue_impact(df: pd.DataFrame, stage_cols: List[str], revenue_col: str,
                           bottleneck_data: Dict) -> Dict:
    df_clean = df.copy()
    for col in stage_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['total_duration'] = df_clean[stage_cols].sum(axis=1)
    df_clean[revenue_col] = pd.to_numeric(df_clean[revenue_col], errors='coerce')
    
    # Current state
    current_duration = df_clean['total_duration'].mean()
    current_revenue = df_clean[revenue_col].mean()
    total_revenue = df_clean[revenue_col].sum()
    
    # Regression: duration -> revenue
    valid_idx = df_clean['total_duration'].notna() & df_clean[revenue_col].notna()
    X = df_clean.loc[valid_idx, 'total_duration'].values.reshape(-1, 1)
    y = df_clean.loc[valid_idx, revenue_col].values
    
    reg = LinearRegression().fit(X, y)
    duration_revenue_coef = reg.coef_[0]
    
    # Scenarios
    bottleneck_stage = bottleneck_data['bottleneck_stage']
    bottleneck_mean = bottleneck_data['bottleneck_duration']
    
    scenarios = []
    
    # Scenario 1: Reduce bottleneck by 10%
    reduction_10 = bottleneck_mean * 0.1
    new_duration_10 = current_duration - reduction_10
    revenue_change_10 = -reduction_10 * duration_revenue_coef  # Negative coef means less time = more revenue
    scenarios.append({
        'scenario': 'Reduce bottleneck 10%',
        'stage': bottleneck_stage,
        'duration_reduction': _to_native(reduction_10),
        'new_total_duration': _to_native(new_duration_10),
        'revenue_change_per_unit': _to_native(revenue_change_10),
        'total_revenue_impact': _to_native(revenue_change_10 * len(df_clean)),
        'pct_improvement': _to_native(abs(revenue_change_10) / current_revenue * 100) if current_revenue > 0 else None
    })
    
    # Scenario 2: Reduce bottleneck by 25%
    reduction_25 = bottleneck_mean * 0.25
    new_duration_25 = current_duration - reduction_25
    revenue_change_25 = -reduction_25 * duration_revenue_coef
    scenarios.append({
        'scenario': 'Reduce bottleneck 25%',
        'stage': bottleneck_stage,
        'duration_reduction': _to_native(reduction_25),
        'new_total_duration': _to_native(new_duration_25),
        'revenue_change_per_unit': _to_native(revenue_change_25),
        'total_revenue_impact': _to_native(revenue_change_25 * len(df_clean)),
        'pct_improvement': _to_native(abs(revenue_change_25) / current_revenue * 100) if current_revenue > 0 else None
    })
    
    # Scenario 3: Eliminate variability (reduce to median)
    bottleneck_median = pd.to_numeric(df[bottleneck_stage], errors='coerce').median()
    variability_reduction = bottleneck_mean - bottleneck_median
    if variability_reduction > 0:
        new_duration_var = current_duration - variability_reduction
        revenue_change_var = -variability_reduction * duration_revenue_coef
        scenarios.append({
            'scenario': 'Standardize to median',
            'stage': bottleneck_stage,
            'duration_reduction': _to_native(variability_reduction),
            'new_total_duration': _to_native(new_duration_var),
            'revenue_change_per_unit': _to_native(revenue_change_var),
            'total_revenue_impact': _to_native(revenue_change_var * len(df_clean)),
            'pct_improvement': _to_native(abs(revenue_change_var) / current_revenue * 100) if current_revenue > 0 else None
        })
    
    # Best scenario
    best_scenario = max(scenarios, key=lambda x: x['total_revenue_impact'] or 0)
    
    return {
        'current_state': {
            'avg_duration': _to_native(current_duration),
            'avg_revenue': _to_native(current_revenue),
            'total_revenue': _to_native(total_revenue),
            'n_observations': len(df_clean)
        },
        'duration_revenue_relationship': {
            'coefficient': _to_native(duration_revenue_coef),
            'interpretation': 'Each unit decrease in duration increases revenue' if duration_revenue_coef < 0 else 'Duration positively affects revenue'
        },
        'scenarios': scenarios,
        'best_scenario': best_scenario,
        'bottleneck_stage': bottleneck_stage
    }


def create_simulation_chart(sim_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Scenario comparison
    ax1 = axes[0]
    scenarios = sim_data.get('scenarios', [])
    if scenarios:
        names = [s['scenario'][:20] for s in scenarios]
        impacts = [s['total_revenue_impact'] or 0 for s in scenarios]
        colors = ['#10b981' if i > 0 else '#ef4444' for i in impacts]
        ax1.barh(names, impacts, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-')
        ax1.set_xlabel('Total Revenue Impact')
        ax1.set_title('Scenario Revenue Impact', fontsize=11, fontweight='bold')
    
    # Chart 2: Duration reduction vs Revenue
    ax2 = axes[1]
    if scenarios:
        reductions = [s['duration_reduction'] or 0 for s in scenarios]
        revenues = [s['total_revenue_impact'] or 0 for s in scenarios]
        ax2.scatter(reductions, revenues, s=100, c='#3b82f6', alpha=0.7, edgecolor='black')
        for i, s in enumerate(scenarios):
            ax2.annotate(s['scenario'][:15], (reductions[i], revenues[i]), fontsize=8, ha='left')
        ax2.set_xlabel('Duration Reduction')
        ax2.set_ylabel('Revenue Impact')
        ax2.set_title('Duration vs Revenue Trade-off', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(duration: Dict, failure: Optional[Dict], resource: Optional[Dict],
                   bottleneck: Dict, simulation: Optional[Dict]) -> Dict:
    report = {}
    
    # Step 1: Stage Duration Analysis
    bn_stage = duration['bottleneck_stage']
    var_stage = duration['most_variable_stage']
    cv_interpretation = "highly inconsistent process execution" if var_stage['cv'] > 50 else "moderate process variability" if var_stage['cv'] > 30 else "relatively stable process timing"
    stage_breakdown = ", ".join([f"{s['stage']} ({s['pct_of_total']:.1f}%)" for s in duration['stages'][:4]])
    
    report['step1_duration'] = {
        'title': '1. Stage Duration Analysis',
        'question': 'How long does each process stage take?',
        'finding': f"Bottleneck: {bn_stage['stage']} ({bn_stage['pct_of_total']:.1f}% of total time)",
        'detail': (f"Comprehensive timing analysis across {duration['n_stages']} process stages reveals a total average duration of {duration['total_mean_duration']:.1f} units. "
                 f"The critical bottleneck is '{bn_stage['stage']}', consuming {bn_stage['pct_of_total']:.1f}% of total process time "
                 f"with an average duration of {bn_stage['mean']:.1f} units (median: {bn_stage['median']:.1f}, P95: {bn_stage['p95']:.1f}). "
                 + ("This stage alone accounts for more than a third" if bn_stage['pct_of_total'] > 33 else "This stage alone accounts for more than a quarter" if bn_stage['pct_of_total'] > 25 else "This stage alone accounts for a significant portion")
                 + " of end-to-end process time, making it the primary target for optimization efforts. "
                 f"The most variable stage is '{var_stage['stage']}' with a coefficient of variation (CV) of {var_stage['cv']:.1f}%, indicating {cv_interpretation}. "
                 "This variability represents unpredictability in process execution that impacts planning, resource allocation, and customer expectations. "
                 f"Stage duration breakdown: {stage_breakdown}. "
                 "Reducing the bottleneck stage duration by even 10-20% could yield significant improvements in overall throughput and cycle time.")
    }
    
    # Step 2: Failure Rate Analysis
    if failure and not failure.get('error'):
        rate_interpretation = "critically high failure rate requiring immediate intervention" if failure['overall_failure_rate'] > 15 else "elevated failure rate warranting attention" if failure['overall_failure_rate'] > 8 else "acceptable failure rate within industry norms" if failure['overall_failure_rate'] > 3 else "excellent quality performance"
        
        process_text = ""
        if failure.get('worst_process') and failure.get('best_process'):
            gap = failure['worst_process']['failure_rate'] - failure['best_process']['failure_rate']
            process_text = (f"Process type comparison reveals significant variation: '{failure['worst_process']['process']}' has the highest failure rate at {failure['worst_process']['failure_rate']:.1f}% ({failure['worst_process']['n_failures']} failures), "
                           f"while '{failure['best_process']['process']}' performs best at {failure['best_process']['failure_rate']:.1f}%. "
                           f"This {gap:.1f} percentage point gap suggests process-specific quality issues that should be investigated. ")
        
        corr_text = ""
        if failure.get('stage_correlations'):
            corr_parts = [f"{c['stage']} (r={c['correlation']:.3f}{'***' if c['significant'] else ''})" for c in failure.get('stage_correlations', [])[:3]]
            corr_text = "Stage-failure correlation analysis shows: " + ", ".join(corr_parts) + ". Positive correlations indicate stages where longer duration is associated with higher failure probability, suggesting quality degrades with extended processing time. "
        
        report['step2_failure'] = {
            'title': '2. Failure Rate Analysis',
            'question': 'Where do processes fail?',
            'finding': f"Overall failure rate: {failure['overall_failure_rate']:.1f}%",
            'detail': (f"Quality analysis across {failure['n_total']} process instances identifies {failure['n_failures']} failures, "
                     f"yielding an overall failure rate of {failure['overall_failure_rate']:.1f}%. This represents {rate_interpretation}. "
                     + process_text + corr_text
                     + "Reducing failure rate not only improves quality metrics but also eliminates rework costs and reduces effective cycle time. "
                     f"Each percentage point reduction in failure rate translates to approximately {failure['n_total'] // 100} fewer failures in this dataset.")
        }
    else:
        report['step2_failure'] = {
            'title': '2. Failure Rate Analysis',
            'question': 'Where do processes fail?',
            'finding': 'Failure analysis not performed',
            'detail': 'Failure rate analysis was not conducted as no failure/success indicator column was specified. To understand quality performance and identify failure patterns, configure a binary outcome column (e.g., pass/fail, success/failure, defect/no defect) and re-run the analysis.'
        }
    
    # Step 3: Resource-Speed Relationship
    if resource and not resource.get('error'):
        most_impact = resource.get('most_impactful')
        
        impact_text = ""
        if most_impact:
            direction_text = "increasing this resource reduces process duration" if most_impact['correlation'] < 0 else "higher resource levels are associated with longer duration (investigate potential inefficiency)"
            sig_text = "This statistically significant relationship (p<0.05) provides a data-driven basis for resource allocation decisions. " if most_impact['significant'] else "While directionally indicative, this relationship did not reach statistical significance. "
            impact_text = (f"The most impactful resource is '{most_impact['resource']}' with correlation r={most_impact['correlation']:.3f} "
                          f"(R²={most_impact['r_squared']:.3f}, explaining {most_impact['r_squared']*100:.1f}% of speed variance). "
                          f"The {most_impact['direction']} correlation indicates that {direction_text}. {sig_text}")
        
        rel_parts = [f"{r['resource']} (r={r['correlation']:.3f}, {r['direction']})" for r in resource['relationships'][:4]]
        rel_text = "Resource-speed relationships: " + ", ".join(rel_parts) + ". "
        
        opp_text = "reveals significant optimization opportunities. " if most_impact and abs(most_impact['correlation']) > 0.3 else "reveals limited direct resource-speed relationships. "
        
        report['step3_resource'] = {
            'title': '3. Resource-Speed Relationship',
            'question': 'How do resources affect speed?',
            'finding': f"Most impactful: {most_impact['resource']} (r={most_impact['correlation']:.3f})" if most_impact else "No significant relationships",
            'detail': (f"Resource efficiency analysis examining {resource['n_resources']} resource variables against process speed (inverse of duration) "
                     + opp_text + impact_text + rel_text
                     + "Negative correlations represent efficiency-positive resources where more input yields faster output. "
                     "Positive correlations may indicate diminishing returns, coordination overhead, or misallocated resources requiring investigation. "
                     "These insights enable evidence-based resource planning to optimize process velocity.")
        }
    else:
        report['step3_resource'] = {
            'title': '3. Resource-Speed Relationship',
            'question': 'How do resources affect speed?',
            'finding': 'Resource analysis not performed',
            'detail': 'Resource-speed relationship analysis was not conducted as no resource columns were specified. To understand how staffing, equipment, or other inputs affect process velocity, configure resource columns (e.g., worker count, machine hours, budget allocated) and re-run the analysis.'
        }
    
    # Step 4: Bottleneck Root Cause
    causes = bottleneck.get('causes', [])
    cv_issue = bottleneck['bottleneck_cv'] > 40
    
    causes_text = ""
    if causes:
        cause_parts = []
        for c in causes[:3]:
            part = f"[{c['factor']}] {c['finding']}"
            if c.get('recommendation'):
                part += f" (Action: {c.get('recommendation', 'Investigate further')})"
            cause_parts.append(part)
        causes_text = "Identified causes: " + "; ".join(cause_parts) + ". "
    else:
        causes_text = "No specific causal factors were identified through the analysis. "
    
    dep_text = ""
    if bottleneck.get('stage_dependencies'):
        dep_parts = [f"{d['from']}→{d['to']} (r={d['correlation']:.2f})" for d in bottleneck.get('stage_dependencies', [])[:3]]
        dep_text = "Stage dependency analysis reveals sequential correlations: " + ", ".join(dep_parts) + ". Strong positive correlations between consecutive stages indicate cascading effects where delays propagate downstream. "
    
    cv_text = "highly inconsistent execution that suggests process standardization opportunities" if cv_issue else "relatively consistent execution suggesting systemic rather than variability-driven constraints"
    priority_text = "Prioritize variability reduction through standardized procedures and training before addressing capacity constraints." if cv_issue else "Focus on capacity enhancement or parallel processing to address the systemic bottleneck."
    
    report['step4_bottleneck'] = {
        'title': '4. Bottleneck Root Cause',
        'question': 'Why is the bottleneck slow?',
        'finding': f"{bottleneck['n_causes_found']} potential causes identified for {bottleneck['bottleneck_stage']}",
        'detail': (f"Root cause analysis of the bottleneck stage '{bottleneck['bottleneck_stage']}' (averaging {bottleneck['bottleneck_duration']:.1f} units) "
                 f"identifies {bottleneck['n_causes_found']} potential contributing factors. "
                 f"The bottleneck exhibits a coefficient of variation of {bottleneck['bottleneck_cv']:.1f}%, indicating {cv_text}. "
                 + causes_text + dep_text
                 + "Addressing these root causes through targeted interventions—whether process redesign, resource reallocation, or standardization—"
                 f"represents the highest-leverage opportunity for overall process improvement. {priority_text}")
    }
    
    # Step 5: Revenue Impact Simulation
    if simulation and not simulation.get('error'):
        best_scenario = simulation['best_scenario']
        scenarios = simulation.get('scenarios', [])
        current = simulation['current_state']
        
        scenario_parts = [f"{s['scenario']} → {s['pct_improvement']:.1f}% improvement (+{s['total_revenue_impact']:.0f} revenue)" for s in scenarios[:3]]
        scenario_text = "Scenario analysis results: " + "; ".join(scenario_parts) + ". "
        
        report['step5_simulation'] = {
            'title': '5. Revenue Impact Simulation',
            'question': 'What is the financial impact?',
            'finding': f"Best scenario: {best_scenario['scenario']} (+{best_scenario['total_revenue_impact']:.0f} revenue)",
            'detail': (f"Financial impact simulation translating process improvements into revenue outcomes provides a business case for optimization investments. "
                     f"Current state: average process duration of {current['avg_duration']:.1f} units yields average revenue of {current['avg_revenue']:.1f} per instance "
                     f"(total: {current['total_revenue']:.0f} across {current['n_observations']} observations). "
                     f"The duration-revenue relationship (coefficient: {simulation['duration_revenue_relationship']['coefficient']:.2f}) indicates that "
                     f"{simulation['duration_revenue_relationship']['interpretation'].lower()}. "
                     + scenario_text
                     + f"The optimal scenario is '{best_scenario['scenario']}', projecting a total revenue impact of +{best_scenario['total_revenue_impact']:.0f} "
                     f"(+{best_scenario['pct_improvement']:.1f}% improvement) through a duration reduction of {best_scenario['duration_reduction']:.1f} units. "
                     "This represents a clear ROI opportunity: investments in bottleneck reduction that achieve the projected time savings "
                     "would generate measurable revenue gains. The analysis provides quantitative justification for process improvement initiatives "
                     "and helps prioritize investments based on expected financial returns.")
        }
    else:
        report['step5_simulation'] = {
            'title': '5. Revenue Impact Simulation',
            'question': 'What is the financial impact?',
            'finding': 'Simulation not performed',
            'detail': 'Revenue impact simulation was not conducted as no revenue/output column was specified. To translate process improvements into financial outcomes, configure a revenue or output metric column and re-run the analysis. This enables ROI calculation and business case development for optimization investments.'
        }
    
    return report


def generate_insights(duration: Dict, failure: Optional[Dict], resource: Optional[Dict],
                     bottleneck: Dict, simulation: Optional[Dict]) -> List[Dict]:
    insights = []
    
    # Bottleneck insight
    if duration['bottleneck_stage']['pct_of_total'] > 40:
        insights.append({
            'title': 'Critical Bottleneck',
            'description': f"{duration['bottleneck_stage']['stage']} consumes {duration['bottleneck_stage']['pct_of_total']:.1f}% of total time. Priority for optimization.",
            'status': 'warning'
        })
    
    # Variability insight
    if duration['most_variable_stage']['cv'] > 50:
        insights.append({
            'title': 'High Process Variability',
            'description': f"{duration['most_variable_stage']['stage']} has CV of {duration['most_variable_stage']['cv']:.1f}%. Standardization needed.",
            'status': 'warning'
        })
    
    # Failure insight
    if failure and failure.get('overall_failure_rate', 0) > 10:
        insights.append({
            'title': 'High Failure Rate',
            'description': f"{failure['overall_failure_rate']:.1f}% failure rate impacts efficiency. Focus on quality control.",
            'status': 'warning'
        })
    
    # Resource insight
    if resource and resource.get('most_impactful') and resource['most_impactful']['correlation'] < -0.3:
        insights.append({
            'title': 'Resource Opportunity',
            'description': f"Increasing {resource['most_impactful']['resource']} significantly reduces duration.",
            'status': 'positive'
        })
    
    # Revenue insight
    if simulation and simulation.get('best_scenario'):
        insights.append({
            'title': 'Revenue Opportunity',
            'description': f"{simulation['best_scenario']['scenario']} could generate +{simulation['best_scenario']['total_revenue_impact']:.0f} additional revenue.",
            'status': 'positive'
        })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/bottleneck-analysis")
async def analyze_bottleneck(request: BottleneckRequest):
    try:
        df = pd.DataFrame(request.data)
        stage_cols = request.stage_cols
        failure_col = request.failure_col
        resource_cols = request.resource_cols or []
        revenue_col = request.revenue_col
        process_col = request.process_col
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 observations")
        
        if len(stage_cols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 stage columns")
        
        results = {}
        visualizations = {}
        
        # Step 1: Stage Duration
        duration = analyze_stage_duration(df, stage_cols)
        results['duration'] = duration
        visualizations['duration_chart'] = create_duration_chart(duration)
        
        # Step 2: Failure Rate
        failure = None
        if failure_col and failure_col in df.columns:
            failure = analyze_failure_rate(df, failure_col, process_col, stage_cols)
            results['failure'] = failure
            visualizations['failure_chart'] = create_failure_chart(failure)
        
        # Step 3: Resource-Speed
        resource = None
        if resource_cols:
            valid_res = [c for c in resource_cols if c in df.columns]
            if valid_res:
                resource = analyze_resource_speed(df, stage_cols, valid_res)
                results['resource'] = resource
                visualizations['resource_chart'] = create_resource_chart(resource, df, stage_cols, valid_res)
        
        # Step 4: Bottleneck Cause
        bottleneck = analyze_bottleneck_cause(df, stage_cols, resource_cols, process_col)
        results['bottleneck'] = bottleneck
        visualizations['bottleneck_chart'] = create_bottleneck_chart(bottleneck, df, stage_cols)
        
        # Step 5: Revenue Simulation
        simulation = None
        if revenue_col and revenue_col in df.columns:
            simulation = simulate_revenue_impact(df, stage_cols, revenue_col, bottleneck)
            results['simulation'] = simulation
            visualizations['simulation_chart'] = create_simulation_chart(simulation)
        
        report = generate_report(duration, failure, resource, bottleneck, simulation)
        insights = generate_insights(duration, failure, resource, bottleneck, simulation)
        
        summary = {
            'n_observations': len(df),
            'n_stages': duration['n_stages'],
            'bottleneck_stage': duration['bottleneck_stage']['stage'],
            'total_duration': duration['total_mean_duration'],
            'failure_rate': failure['overall_failure_rate'] if failure else None,
            'best_scenario_impact': simulation['best_scenario']['total_revenue_impact'] if simulation else None
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
