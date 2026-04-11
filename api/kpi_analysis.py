"""
KPI Performance Analysis API
Analyzes goal achievement, team comparisons, resource-performance relationships,
key drivers, and resource reallocation simulations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()


class KPIAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    kpi_col: str                          # Performance/achievement column
    target_col: Optional[str] = None      # Target value column
    group_col: Optional[str] = None       # Team/department column
    resource_cols: Optional[List[str]] = None  # Resource columns (budget, headcount, etc.)
    kpi_name_col: Optional[str] = None    # KPI name/label column
    weight_col: Optional[str] = None      # KPI weight column


def _to_native_type(obj):
    """Convert numpy types to Python native types"""
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_native_type(x) for x in obj.tolist()]
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return img


# =============================================================================
# Step 1: Current Status Analysis
# =============================================================================
def analyze_current_status(df: pd.DataFrame, kpi_col: str, target_col: Optional[str],
                           kpi_name_col: Optional[str], weight_col: Optional[str]) -> Dict[str, Any]:
    """Analyze overall KPI achievement status"""
    df = df.copy()
    
    total_records = len(df)
    avg_performance = df[kpi_col].mean()
    std_performance = df[kpi_col].std()
    min_performance = df[kpi_col].min()
    max_performance = df[kpi_col].max()
    median_performance = df[kpi_col].median()
    
    result = {
        'total_kpis': total_records,
        'avg_performance': _to_native_type(avg_performance),
        'std_performance': _to_native_type(std_performance),
        'min_performance': _to_native_type(min_performance),
        'max_performance': _to_native_type(max_performance),
        'median_performance': _to_native_type(median_performance),
        'performance_distribution': {
            'q1': _to_native_type(df[kpi_col].quantile(0.25)),
            'q2': _to_native_type(df[kpi_col].quantile(0.50)),
            'q3': _to_native_type(df[kpi_col].quantile(0.75))
        }
    }
    
    # If target column exists, calculate achievement rates
    if target_col and target_col in df.columns:
        df['_achievement_rate'] = (df[kpi_col] / df[target_col] * 100).replace([np.inf, -np.inf], np.nan)
        df['_achieved'] = df[kpi_col] >= df[target_col]
        df['_gap'] = df[kpi_col] - df[target_col]
        
        achieved_count = df['_achieved'].sum()
        achievement_rate = achieved_count / total_records * 100
        avg_achievement_pct = df['_achievement_rate'].mean()
        total_gap = df['_gap'].sum()
        
        result.update({
            'has_targets': True,
            'achieved_count': int(achieved_count),
            'not_achieved_count': int(total_records - achieved_count),
            'achievement_rate': _to_native_type(achievement_rate),
            'avg_achievement_pct': _to_native_type(avg_achievement_pct),
            'total_gap': _to_native_type(total_gap),
            'avg_gap': _to_native_type(df['_gap'].mean()),
            'total_target': _to_native_type(df[target_col].sum()),
            'total_actual': _to_native_type(df[kpi_col].sum()),
            'overall_achievement_pct': _to_native_type(df[kpi_col].sum() / df[target_col].sum() * 100) if df[target_col].sum() > 0 else None
        })
        
        # Performance tier distribution
        df['_tier'] = pd.cut(df['_achievement_rate'], 
                            bins=[0, 50, 80, 100, 120, float('inf')],
                            labels=['Critical (<50%)', 'Below (50-80%)', 'Near (80-100%)', 'Met (100-120%)', 'Exceeded (>120%)'])
        tier_counts = df['_tier'].value_counts().to_dict()
        result['tier_distribution'] = {str(k): int(v) for k, v in tier_counts.items()}
        
        # Worst performing KPIs
        if kpi_name_col and kpi_name_col in df.columns:
            worst = df.nsmallest(5, '_achievement_rate')[[kpi_name_col, kpi_col, target_col, '_achievement_rate', '_gap']]
            worst.columns = ['kpi_name', 'actual', 'target', 'achievement_pct', 'gap']
            result['worst_performers'] = worst.to_dict('records')
            
            best = df.nlargest(5, '_achievement_rate')[[kpi_name_col, kpi_col, target_col, '_achievement_rate', '_gap']]
            best.columns = ['kpi_name', 'actual', 'target', 'achievement_pct', 'gap']
            result['best_performers'] = best.to_dict('records')
    else:
        result['has_targets'] = False
    
    return result


# =============================================================================
# Step 2: Group Comparison Analysis
# =============================================================================
def analyze_group_comparison(df: pd.DataFrame, kpi_col: str, target_col: Optional[str],
                             group_col: str) -> Dict[str, Any]:
    """Compare performance across groups (teams/departments)"""
    df = df.copy()
    
    # Basic group stats
    group_stats = df.groupby(group_col).agg({
        kpi_col: ['sum', 'mean', 'std', 'count', 'min', 'max']
    }).reset_index()
    group_stats.columns = ['group', 'total', 'avg', 'std', 'count', 'min', 'max']
    
    if target_col and target_col in df.columns:
        target_stats = df.groupby(group_col)[target_col].sum().reset_index()
        target_stats.columns = ['group', 'total_target']
        group_stats = group_stats.merge(target_stats, on='group')
        group_stats['achievement_pct'] = group_stats['total'] / group_stats['total_target'] * 100
        group_stats['gap'] = group_stats['total'] - group_stats['total_target']
        group_stats['met_target'] = group_stats['achievement_pct'] >= 100
    
    # Rank groups
    group_stats['rank'] = group_stats['avg'].rank(ascending=False).astype(int)
    group_stats = group_stats.sort_values('rank')
    
    # Statistical comparison (ANOVA if multiple groups)
    groups = [g[kpi_col].values for _, g in df.groupby(group_col)]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
        significant_diff = p_value < 0.05
    else:
        f_stat, p_value, significant_diff = None, None, None
    
    best_group = group_stats.iloc[0]
    worst_group = group_stats.iloc[-1]
    
    return {
        'group_data': group_stats.to_dict('records'),
        'n_groups': len(group_stats),
        'best_performer': {
            'group': best_group['group'],
            'avg': _to_native_type(best_group['avg']),
            'total': _to_native_type(best_group['total'])
        },
        'worst_performer': {
            'group': worst_group['group'],
            'avg': _to_native_type(worst_group['avg']),
            'total': _to_native_type(worst_group['total'])
        },
        'performance_gap': _to_native_type(best_group['avg'] - worst_group['avg']),
        'gap_pct': _to_native_type((best_group['avg'] - worst_group['avg']) / worst_group['avg'] * 100) if worst_group['avg'] != 0 else None,
        'statistical_test': {
            'method': 'ANOVA',
            'f_statistic': _to_native_type(f_stat),
            'p_value': _to_native_type(p_value),
            'significant_difference': significant_diff
        }
    }


# =============================================================================
# Step 3: Resource-Performance Correlation Analysis
# =============================================================================
def analyze_correlation(df: pd.DataFrame, kpi_col: str, resource_cols: List[str]) -> Dict[str, Any]:
    """Analyze correlation between resources and performance"""
    df = df.copy()
    
    correlations = []
    for res_col in resource_cols:
        if res_col not in df.columns:
            continue
            
        # Clean data
        valid_data = df[[res_col, kpi_col]].dropna()
        if len(valid_data) < 3:
            continue
        
        # Pearson correlation
        corr, p_value = stats.pearsonr(valid_data[res_col], valid_data[kpi_col])
        
        # Spearman correlation (rank-based, more robust)
        spearman_corr, spearman_p = stats.spearmanr(valid_data[res_col], valid_data[kpi_col])
        
        # Interpret strength
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = 'strong'
        elif abs_corr >= 0.4:
            strength = 'moderate'
        elif abs_corr >= 0.2:
            strength = 'weak'
        else:
            strength = 'negligible'
        
        direction = 'positive' if corr > 0 else 'negative'
        
        correlations.append({
            'resource': res_col,
            'correlation': _to_native_type(corr),
            'p_value': _to_native_type(p_value),
            'significant': p_value < 0.05,
            'spearman_correlation': _to_native_type(spearman_corr),
            'spearman_p_value': _to_native_type(spearman_p),
            'strength': strength,
            'direction': direction,
            'interpretation': f"{strength.capitalize()} {direction} correlation"
        })
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['correlation'] or 0), reverse=True)
    
    strongest = correlations[0] if correlations else None
    significant_correlations = [c for c in correlations if c['significant']]
    
    return {
        'correlations': correlations,
        'n_resources': len(correlations),
        'strongest_correlation': strongest,
        'significant_count': len(significant_correlations),
        'significant_correlations': significant_correlations,
        'summary': f"Found {len(significant_correlations)} significant correlations out of {len(correlations)} resources analyzed"
    }


# =============================================================================
# Step 4: Key Driver Analysis (Regression)
# =============================================================================
def analyze_drivers(df: pd.DataFrame, kpi_col: str, resource_cols: List[str],
                    group_col: Optional[str]) -> Dict[str, Any]:
    """Identify key performance drivers using regression analysis"""
    df = df.copy()
    
    # Prepare features
    available_cols = [c for c in resource_cols if c in df.columns]
    if not available_cols:
        return {'error': 'No valid resource columns found'}
    
    # Clean data
    analysis_cols = available_cols + [kpi_col]
    clean_df = df[analysis_cols].dropna()
    
    if len(clean_df) < len(available_cols) + 2:
        return {'error': 'Insufficient data for regression analysis'}
    
    X = clean_df[available_cols].values
    y = clean_df[kpi_col].values
    
    # Standardize for comparable coefficients
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X_standardized = (X - X_mean) / X_std
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X_standardized)), X_standardized])
    
    # OLS regression
    try:
        coefficients, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        
        # Calculate R-squared
        y_pred = X_with_intercept @ coefficients
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Adjusted R-squared
        n = len(y)
        p = len(available_cols)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        
        # Calculate importance (absolute standardized coefficients)
        std_coefficients = coefficients[1:]  # Exclude intercept
        total_abs_coef = np.sum(np.abs(std_coefficients))
        
        drivers = []
        for i, col in enumerate(available_cols):
            importance = abs(std_coefficients[i]) / total_abs_coef * 100 if total_abs_coef > 0 else 0
            drivers.append({
                'factor': col,
                'coefficient': _to_native_type(std_coefficients[i]),
                'importance': _to_native_type(importance),
                'direction': 'positive' if std_coefficients[i] > 0 else 'negative',
                'interpretation': f"{'Increases' if std_coefficients[i] > 0 else 'Decreases'} performance by {abs(std_coefficients[i]):.2f} std units per 1 std increase"
            })
        
        # Sort by importance
        drivers.sort(key=lambda x: x['importance'], reverse=True)
        
        result = {
            'drivers': drivers,
            'r_squared': _to_native_type(r_squared),
            'adj_r_squared': _to_native_type(adj_r_squared),
            'model_quality': 'good' if r_squared >= 0.6 else 'moderate' if r_squared >= 0.3 else 'weak',
            'key_driver': drivers[0] if drivers else None,
            'n_observations': int(n),
            'n_features': int(p)
        }
        
        # Identify underperforming groups if group_col provided
        if group_col and group_col in df.columns:
            group_residuals = []
            for group_name, group_df in df.groupby(group_col):
                group_clean = group_df[analysis_cols].dropna()
                if len(group_clean) == 0:
                    continue
                X_g = group_clean[available_cols].values
                y_g = group_clean[kpi_col].values
                X_g_std = (X_g - X_mean) / X_std
                X_g_int = np.column_stack([np.ones(len(X_g_std)), X_g_std])
                y_g_pred = X_g_int @ coefficients
                avg_residual = (y_g - y_g_pred).mean()
                group_residuals.append({
                    'group': group_name,
                    'avg_residual': _to_native_type(avg_residual),
                    'status': 'underperforming' if avg_residual < 0 else 'overperforming'
                })
            
            group_residuals.sort(key=lambda x: x['avg_residual'])
            result['group_performance'] = group_residuals
            result['underperforming_groups'] = [g['group'] for g in group_residuals if g['avg_residual'] < -y.std() * 0.5]
            result['overperforming_groups'] = [g['group'] for g in group_residuals if g['avg_residual'] > y.std() * 0.5]
        
        return result
        
    except Exception as e:
        return {'error': f'Regression analysis failed: {str(e)}'}


# =============================================================================
# Step 5: Resource Reallocation Simulation
# =============================================================================
def simulate_reallocation(df: pd.DataFrame, kpi_col: str, resource_cols: List[str],
                          group_col: Optional[str], drivers: Dict) -> Dict[str, Any]:
    """Simulate impact of resource reallocation"""
    
    if 'error' in drivers or not drivers.get('drivers'):
        return {'error': 'Driver analysis required for simulation'}
    
    df = df.copy()
    scenarios = []
    
    # Get the key driver
    key_driver = drivers['drivers'][0]
    driver_col = key_driver['factor']
    coefficient = key_driver['coefficient']
    
    if driver_col not in df.columns:
        return {'error': f'Driver column {driver_col} not found'}
    
    # Current state
    current_total_kpi = df[kpi_col].sum()
    current_avg_kpi = df[kpi_col].mean()
    current_resource = df[driver_col].sum()
    
    # Scenario 1: 10% increase in key resource
    increase_pct = 10
    expected_change = coefficient * (df[driver_col].std() * (increase_pct / 100))
    scenarios.append({
        'name': f'{increase_pct}% Increase in {driver_col}',
        'resource_change': f'+{increase_pct}%',
        'expected_performance_change': _to_native_type(expected_change),
        'expected_performance_change_pct': _to_native_type(expected_change / current_avg_kpi * 100) if current_avg_kpi != 0 else None,
        'new_expected_avg': _to_native_type(current_avg_kpi + expected_change)
    })
    
    # Scenario 2: 10% decrease
    decrease_pct = 10
    expected_change_neg = -coefficient * (df[driver_col].std() * (decrease_pct / 100))
    scenarios.append({
        'name': f'{decrease_pct}% Decrease in {driver_col}',
        'resource_change': f'-{decrease_pct}%',
        'expected_performance_change': _to_native_type(expected_change_neg),
        'expected_performance_change_pct': _to_native_type(expected_change_neg / current_avg_kpi * 100) if current_avg_kpi != 0 else None,
        'new_expected_avg': _to_native_type(current_avg_kpi + expected_change_neg)
    })
    
    # Group-level reallocation if group_col exists
    group_scenarios = []
    if group_col and group_col in df.columns and 'group_performance' in drivers:
        underperforming = [g for g in drivers['group_performance'] if g['status'] == 'underperforming']
        overperforming = [g for g in drivers['group_performance'] if g['status'] == 'overperforming']
        
        if underperforming and overperforming:
            worst_group = underperforming[0]['group']
            best_group = overperforming[-1]['group']
            
            # Simulate moving 10% of resource from best to worst
            best_resource = df[df[group_col] == best_group][driver_col].sum()
            transfer_amount = best_resource * 0.1
            
            group_scenarios.append({
                'name': f'Transfer {driver_col} from {best_group} to {worst_group}',
                'from_group': best_group,
                'to_group': worst_group,
                'transfer_amount': _to_native_type(transfer_amount),
                'transfer_pct': 10,
                'rationale': f'{worst_group} is underperforming relative to resources; {best_group} is overperforming'
            })
    
    # Optimal allocation suggestion
    optimal = {
        'focus_resource': driver_col,
        'importance': key_driver['importance'],
        'recommendation': f"Focus on optimizing {driver_col} - it explains {key_driver['importance']:.1f}% of performance variation",
        'secondary_drivers': [d['factor'] for d in drivers['drivers'][1:3]] if len(drivers['drivers']) > 1 else []
    }
    
    return {
        'scenarios': scenarios,
        'group_scenarios': group_scenarios,
        'optimal_allocation': optimal,
        'current_state': {
            'total_kpi': _to_native_type(current_total_kpi),
            'avg_kpi': _to_native_type(current_avg_kpi),
            'total_key_resource': _to_native_type(current_resource)
        },
        'model_reliability': drivers['model_quality'],
        'r_squared': drivers['r_squared']
    }


# =============================================================================
# Visualization Functions
# =============================================================================
def create_status_chart(status: Dict) -> str:
    """Create current status overview chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Achievement Distribution (if targets exist)
    if status.get('has_targets') and status.get('tier_distribution'):
        ax1 = axes[0]
        tiers = status['tier_distribution']
        labels = list(tiers.keys())
        values = list(tiers.values())
        colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6']
        
        bars = ax1.barh(labels, values, color=colors[:len(labels)])
        ax1.set_xlabel('Count')
        ax1.set_title('Achievement Distribution', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    str(val), va='center', fontsize=10)
    else:
        ax1 = axes[0]
        ax1.text(0.5, 0.5, 'No target data\navailable', ha='center', va='center', fontsize=14)
        ax1.axis('off')
    
    # Chart 2: Performance Summary
    ax2 = axes[1]
    if status.get('has_targets'):
        metrics = ['Achievement\nRate', 'Avg Achievement\n%', 'Achieved\nCount']
        values = [
            status.get('achievement_rate', 0),
            status.get('avg_achievement_pct', 0),
            status.get('achieved_count', 0) / status.get('total_kpis', 1) * 100
        ]
        colors = ['#22c55e' if v >= 80 else '#eab308' if v >= 60 else '#ef4444' for v in values]
    else:
        metrics = ['Avg\nPerformance', 'Median', 'Std Dev']
        values = [
            status.get('avg_performance', 0),
            status.get('median_performance', 0),
            status.get('std_performance', 0)
        ]
        colors = ['#3b82f6', '#22c55e', '#f97316']
    
    bars = ax2.bar(metrics, values, color=colors)
    ax2.set_ylabel('Value')
    ax2.set_title('Key Metrics', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_comparison_chart(comparison: Dict) -> str:
    """Create group comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    group_data = comparison['group_data']
    groups = [str(g['group'])[:15] for g in group_data]
    avgs = [g['avg'] for g in group_data]
    
    # Color by rank
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(groups)))[::-1]
    
    bars = ax.barh(groups[::-1], avgs[::-1], color=colors)
    ax.set_xlabel('Average Performance')
    ax.set_title('Performance by Group', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, avgs[::-1]):
        ax.text(bar.get_width() + max(avgs) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=10)
    
    ax.axvline(x=np.mean(avgs), color='red', linestyle='--', label=f'Average: {np.mean(avgs):.1f}')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_correlation_chart(correlation: Dict) -> str:
    """Create correlation analysis chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    corrs = correlation['correlations']
    if not corrs:
        ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
        ax.axis('off')
        return _fig_to_base64(fig)
    
    resources = [c['resource'][:20] for c in corrs]
    values = [c['correlation'] for c in corrs]
    colors = ['#22c55e' if v > 0 else '#ef4444' for v in values]
    
    bars = ax.barh(resources[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Resource-Performance Correlations', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Mark significance
    for i, (bar, corr) in enumerate(zip(bars, corrs[::-1])):
        marker = '***' if corr['significant'] else ''
        ax.text(bar.get_width() + 0.02 if bar.get_width() >= 0 else bar.get_width() - 0.02,
                bar.get_y() + bar.get_height()/2,
                f'{corr["correlation"]:.2f}{marker}',
                va='center', ha='left' if bar.get_width() >= 0 else 'right', fontsize=9)
    
    ax.set_xlim(-1.1, 1.1)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_driver_chart(drivers: Dict) -> str:
    """Create key drivers chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'error' in drivers:
        axes[0].text(0.5, 0.5, drivers['error'], ha='center', va='center')
        axes[0].axis('off')
        axes[1].axis('off')
        return _fig_to_base64(fig)
    
    driver_data = drivers['drivers']
    
    # Chart 1: Importance
    ax1 = axes[0]
    factors = [d['factor'][:15] for d in driver_data]
    importance = [d['importance'] for d in driver_data]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(factors)))[::-1]
    
    bars = ax1.barh(factors[::-1], importance[::-1], color=colors)
    ax1.set_xlabel('Importance (%)')
    ax1.set_title('Driver Importance', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, importance[::-1]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    
    # Chart 2: Model Quality
    ax2 = axes[1]
    r2 = drivers['r_squared'] * 100
    ax2.pie([r2, 100-r2], labels=['Explained', 'Unexplained'],
            colors=['#22c55e', '#e5e7eb'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Model Fit (R² = {drivers["r_squared"]:.2%})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_simulation_chart(simulation: Dict) -> str:
    """Create simulation results chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'error' in simulation:
        ax.text(0.5, 0.5, simulation['error'], ha='center', va='center')
        ax.axis('off')
        return _fig_to_base64(fig)
    
    scenarios = simulation['scenarios']
    current_avg = simulation['current_state']['avg_kpi']
    
    names = ['Current'] + [s['name'][:25] for s in scenarios]
    values = [current_avg] + [s['new_expected_avg'] for s in scenarios]
    colors = ['#3b82f6'] + ['#22c55e' if s['expected_performance_change'] > 0 else '#ef4444' for s in scenarios]
    
    bars = ax.bar(range(len(names)), values, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Expected Performance')
    ax.set_title('Scenario Analysis', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                f'{val:.1f}', ha='center', fontsize=10)
    
    ax.axhline(y=current_avg, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Insight Generation
# =============================================================================
def generate_insights(status: Dict, comparison: Optional[Dict], correlation: Optional[Dict],
                      drivers: Optional[Dict], simulation: Optional[Dict]) -> List[Dict]:
    """Generate key insights from all analyses"""
    insights = []
    
    # Status insights
    if status.get('has_targets'):
        rate = status.get('achievement_rate', 0)
        if rate >= 80:
            insights.append({
                'title': 'Strong Overall Achievement',
                'description': f"{rate:.1f}% of KPIs met their targets. Overall performance is on track.",
                'status': 'positive'
            })
        elif rate >= 50:
            insights.append({
                'title': 'Moderate Achievement',
                'description': f"Only {rate:.1f}% of KPIs met targets. Focus on underperforming areas.",
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Achievement Gap',
                'description': f"Only {rate:.1f}% of KPIs met targets. Urgent attention needed.",
                'status': 'warning'
            })
    
    # Comparison insights
    if comparison and comparison.get('statistical_test', {}).get('significant_difference'):
        gap = comparison.get('gap_pct', 0)
        insights.append({
            'title': 'Significant Group Differences',
            'description': f"Performance varies significantly across groups. Gap between best and worst: {gap:.1f}%.",
            'status': 'warning' if gap > 30 else 'neutral'
        })
    
    # Correlation insights
    if correlation and correlation.get('strongest_correlation'):
        strongest = correlation['strongest_correlation']
        if strongest.get('significant'):
            insights.append({
                'title': f"Key Resource: {strongest['resource']}",
                'description': f"{strongest['strength'].capitalize()} {strongest['direction']} correlation (r={strongest['correlation']:.2f}) with performance.",
                'status': 'positive' if strongest['direction'] == 'positive' else 'neutral'
            })
    
    # Driver insights
    if drivers and not drivers.get('error') and drivers.get('key_driver'):
        key = drivers['key_driver']
        insights.append({
            'title': f"Primary Driver: {key['factor']}",
            'description': f"Explains {key['importance']:.1f}% of performance variation. Model R² = {drivers['r_squared']:.1%}.",
            'status': 'positive'
        })
        
        if drivers.get('underperforming_groups'):
            insights.append({
                'title': 'Underperforming Groups Identified',
                'description': f"Groups performing below expected: {', '.join(drivers['underperforming_groups'][:3])}.",
                'status': 'warning'
            })
    
    # Simulation insights
    if simulation and not simulation.get('error') and simulation.get('optimal_allocation'):
        opt = simulation['optimal_allocation']
        insights.append({
            'title': 'Optimization Opportunity',
            'description': opt['recommendation'],
            'status': 'positive'
        })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/kpi-analysis")
async def analyze_kpi(request: KPIAnalysisRequest):
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 data points")
        
        if request.kpi_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"KPI column '{request.kpi_col}' not found")
        
        df[request.kpi_col] = pd.to_numeric(df[request.kpi_col], errors='coerce')
        
        if request.target_col and request.target_col in df.columns:
            df[request.target_col] = pd.to_numeric(df[request.target_col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Current Status
        status = analyze_current_status(df, request.kpi_col, request.target_col,
                                        request.kpi_name_col, request.weight_col)
        results['current_status'] = status
        visualizations['status_chart'] = create_status_chart(status)
        
        # Step 2: Group Comparison
        comparison = None
        if request.group_col and request.group_col in df.columns:
            comparison = analyze_group_comparison(df, request.kpi_col, request.target_col, request.group_col)
            results['group_comparison'] = comparison
            visualizations['comparison_chart'] = create_comparison_chart(comparison)
        
        # Step 3: Correlation Analysis
        correlation = None
        if request.resource_cols:
            valid_cols = [c for c in request.resource_cols if c in df.columns]
            if valid_cols:
                correlation = analyze_correlation(df, request.kpi_col, valid_cols)
                results['correlation'] = correlation
                visualizations['correlation_chart'] = create_correlation_chart(correlation)
        
        # Step 4: Driver Analysis
        drivers = None
        if request.resource_cols:
            valid_cols = [c for c in request.resource_cols if c in df.columns]
            if valid_cols:
                drivers = analyze_drivers(df, request.kpi_col, valid_cols, request.group_col)
                results['drivers'] = drivers
                visualizations['driver_chart'] = create_driver_chart(drivers)
        
        # Step 5: Simulation
        simulation = None
        if drivers and not drivers.get('error') and request.resource_cols:
            valid_cols = [c for c in request.resource_cols if c in df.columns]
            simulation = simulate_reallocation(df, request.kpi_col, valid_cols, request.group_col, drivers)
            results['simulation'] = simulation
            visualizations['simulation_chart'] = create_simulation_chart(simulation)
        
        # Generate Insights
        insights = generate_insights(status, comparison, correlation, drivers, simulation)
        
        # Summary
        summary = {
            'total_kpis': status['total_kpis'],
            'avg_performance': status['avg_performance'],
            'has_targets': status['has_targets'],
            'achievement_rate': status.get('achievement_rate'),
            'n_groups': comparison['n_groups'] if comparison else None,
            'key_driver': drivers['key_driver']['factor'] if drivers and drivers.get('key_driver') else None,
            'model_r_squared': drivers['r_squared'] if drivers and not drivers.get('error') else None
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
        raise HTTPException(status_code=500, detail=f"KPI analysis failed: {str(e)}")
