"""
Satisfaction Diagnosis API
5-step framework for comprehensive satisfaction analysis
1. Overall Satisfaction Distribution
2. Group Satisfaction Comparison
3. Key Drivers Analysis
4. IPA (Importance-Performance Analysis)
5. Improvement Simulation
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


class SatisfactionRequest(BaseModel):
    data: List[Dict[str, Any]]
    satisfaction_col: str
    attribute_cols: List[str]
    group_col: Optional[str] = None


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
# Step 1: Overall Satisfaction Distribution
# =============================================================================
def analyze_distribution(df: pd.DataFrame, sat_col: str) -> Dict:
    values = df[sat_col].dropna()
    mean_val = values.mean()
    median_val = values.median()
    std_val = values.std()
    min_val = values.min()
    max_val = values.max()
    
    if values.nunique() <= 10:
        distribution = values.value_counts().sort_index().to_dict()
    else:
        distribution = {}
    
    scale_max = values.max()
    if scale_max <= 5:
        thresholds = {'low': 2, 'high': 4}
    elif scale_max <= 7:
        thresholds = {'low': 3, 'high': 6}
    else:
        thresholds = {'low': 4, 'high': 8}
    
    low_count = len(values[values <= thresholds['low']])
    mid_count = len(values[(values > thresholds['low']) & (values <= thresholds['high'])])
    high_count = len(values[values > thresholds['high']])
    total = len(values)
    
    if total >= 8:
        try:
            _, normality_p = stats.shapiro(values.sample(min(5000, total)))
        except:
            normality_p = None
    else:
        normality_p = None
    
    return {
        'n_responses': total,
        'mean': _to_native(mean_val),
        'median': _to_native(median_val),
        'std': _to_native(std_val),
        'min': _to_native(min_val),
        'max': _to_native(max_val),
        'distribution': {str(k): _to_native(v) for k, v in distribution.items()},
        'satisfaction_levels': {
            'low': _to_native(low_count),
            'mid': _to_native(mid_count),
            'high': _to_native(high_count),
            'low_pct': _to_native(low_count / total * 100) if total > 0 else 0,
            'mid_pct': _to_native(mid_count / total * 100) if total > 0 else 0,
            'high_pct': _to_native(high_count / total * 100) if total > 0 else 0
        },
        'normality_p': _to_native(normality_p),
        'is_normal': bool(normality_p > 0.05) if normality_p else None
    }


def create_distribution_chart(df: pd.DataFrame, sat_col: str, dist_data: Dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    values = df[sat_col].dropna()
    
    ax1 = axes[0]
    if values.nunique() <= 10:
        counts = values.value_counts().sort_index()
        ax1.bar(counts.index.astype(str), counts.values, color='#3b82f6', alpha=0.7, edgecolor='black')
    else:
        ax1.hist(values, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
    ax1.axvline(dist_data['mean'], color='#ef4444', linestyle='--', linewidth=2, label=f"Mean: {dist_data['mean']:.2f}")
    ax1.set_xlabel('Satisfaction Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Satisfaction Distribution', fontsize=11, fontweight='bold')
    ax1.legend()
    
    ax2 = axes[1]
    bp = ax2.boxplot(values, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3b82f6')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Satisfaction Score')
    ax2.set_title('Score Distribution', fontsize=11, fontweight='bold')
    
    ax3 = axes[2]
    levels = dist_data['satisfaction_levels']
    sizes = [levels['low'], levels['mid'], levels['high']]
    labels = ['Low', 'Medium', 'High']
    colors = ['#ef4444', '#f59e0b', '#10b981']
    if sum(sizes) > 0:
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Satisfaction Levels', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Group Comparison
# =============================================================================
def analyze_group_comparison(df: pd.DataFrame, sat_col: str, group_col: str) -> Dict:
    group_stats = df.groupby(group_col)[sat_col].agg(['mean', 'std', 'count', 'median']).reset_index()
    group_stats.columns = ['group', 'mean', 'std', 'count', 'median']
    group_stats = group_stats.sort_values('mean', ascending=False)
    group_stats['rank'] = range(1, len(group_stats) + 1)
    
    best = group_stats.iloc[0]
    worst = group_stats.iloc[-1]
    gap = best['mean'] - worst['mean']
    
    groups = [group[sat_col].dropna().values for name, group in df.groupby(group_col)]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            significant = bool(p_value < 0.05)
        except:
            f_stat, p_value, significant = None, None, False
    else:
        f_stat, p_value, significant = None, None, False
    
    eta_squared = None
    if f_stat and len(groups) >= 2:
        try:
            ss_between = sum(len(g) * (np.mean(g) - df[sat_col].mean())**2 for g in groups)
            ss_total = sum((df[sat_col] - df[sat_col].mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
        except:
            pass
    
    group_data = []
    for _, row in group_stats.iterrows():
        group_data.append({
            'group': _to_native(row['group']),
            'rank': _to_native(row['rank']),
            'mean': _to_native(row['mean']),
            'std': _to_native(row['std']),
            'count': _to_native(row['count']),
            'median': _to_native(row['median'])
        })
    
    return {
        'group_data': group_data,
        'n_groups': len(group_stats),
        'best_group': {'group': _to_native(best['group']), 'mean': _to_native(best['mean'])},
        'worst_group': {'group': _to_native(worst['group']), 'mean': _to_native(worst['mean'])},
        'gap': _to_native(gap),
        'statistical_test': {
            'method': 'ANOVA',
            'f_statistic': _to_native(f_stat),
            'p_value': _to_native(p_value),
            'significant': significant,
            'eta_squared': _to_native(eta_squared)
        }
    }


def create_group_chart(group_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    groups = [g['group'] for g in group_data['group_data']]
    means = [g['mean'] for g in group_data['group_data']]
    stds = [g['std'] or 0 for g in group_data['group_data']]
    
    ax1 = axes[0]
    colors = ['#10b981' if i == 0 else '#ef4444' if i == len(groups)-1 else '#3b82f6' for i in range(len(groups))]
    ax1.bar(groups, means, yerr=stds, color=colors, alpha=0.7, edgecolor='black', capsize=5)
    ax1.axhline(y=np.mean(means), color='gray', linestyle='--', alpha=0.5, label='Overall Mean')
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Mean Satisfaction')
    ax1.set_title('Satisfaction by Group', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    ax2 = axes[1]
    overall_mean = np.mean(means)
    gaps = [m - overall_mean for m in means]
    colors_gap = ['#10b981' if g > 0 else '#ef4444' for g in gaps]
    ax2.barh(groups, gaps, color=colors_gap, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Gap from Overall Mean')
    ax2.set_title('Group Performance Gap', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Key Drivers Analysis
# =============================================================================
def analyze_key_drivers(df: pd.DataFrame, sat_col: str, attr_cols: List[str]) -> Dict:
    df_clean = df[[sat_col] + attr_cols].dropna()
    
    if len(df_clean) < len(attr_cols) + 2:
        return {'error': 'Insufficient data'}
    
    X = df_clean[attr_cols]
    y = df_clean[sat_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    r2 = model.score(X_scaled, y)
    n, p = len(y), len(attr_cols)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2
    
    drivers = []
    total_abs = sum(abs(c) for c in model.coef_)
    for attr, coef in zip(attr_cols, model.coef_):
        rel_imp = (abs(coef) / total_abs * 100) if total_abs > 0 else 0
        drivers.append({
            'attribute': attr,
            'beta': _to_native(coef),
            'relative_importance': _to_native(rel_imp),
            'direction': 'positive' if coef > 0 else 'negative'
        })
    
    drivers = sorted(drivers, key=lambda x: abs(x['beta']), reverse=True)
    
    correlations = []
    for attr in attr_cols:
        corr, p_val = stats.pearsonr(df_clean[attr], df_clean[sat_col])
        correlations.append({
            'attribute': attr,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05)
        })
    
    return {
        'drivers': drivers,
        'top_driver': drivers[0] if drivers else None,
        'r_squared': _to_native(r2),
        'adj_r_squared': _to_native(adj_r2),
        'model_quality': 'good' if r2 > 0.5 else 'moderate' if r2 > 0.3 else 'low',
        'correlations': correlations,
        'n_observations': len(df_clean)
    }


def create_drivers_chart(drivers_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    drivers = drivers_data.get('drivers', [])
    if not drivers:
        return ""
    
    ax1 = axes[0]
    attrs = [d['attribute'][:15] for d in drivers]
    imps = [d['relative_importance'] for d in drivers]
    colors = ['#10b981' if d['direction'] == 'positive' else '#ef4444' for d in drivers]
    ax1.barh(attrs, imps, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Relative Importance (%)')
    ax1.set_title('Key Drivers Ranking', fontsize=11, fontweight='bold')
    
    ax2 = axes[1]
    betas = [d['beta'] for d in drivers]
    ax2.barh(attrs, betas, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Standardized Beta Coefficient')
    ax2.set_title('Impact Direction & Magnitude', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: IPA (Importance-Performance Analysis)
# =============================================================================
def analyze_ipa(df: pd.DataFrame, sat_col: str, attr_cols: List[str]) -> Dict:
    df_clean = df[[sat_col] + attr_cols].dropna()
    
    if len(df_clean) < len(attr_cols) + 2:
        return {'error': 'Insufficient data'}
    
    performance = df_clean[attr_cols].mean()
    
    X = df_clean[attr_cols]
    y = df_clean[sat_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    ipa_items = []
    perf_mean = performance.mean()
    imp_mean = 0
    
    for attr, beta in zip(attr_cols, model.coef_):
        perf = performance[attr]
        
        if beta >= imp_mean and perf >= perf_mean:
            quadrant = 'Q1: Keep Up'
        elif beta >= imp_mean and perf < perf_mean:
            quadrant = 'Q2: Focus Here'
        elif beta < imp_mean and perf < perf_mean:
            quadrant = 'Q3: Low Priority'
        else:
            quadrant = 'Q4: Possible Overkill'
        
        ipa_items.append({
            'attribute': attr,
            'performance': _to_native(perf),
            'importance': _to_native(beta),
            'quadrant': quadrant
        })
    
    quadrant_counts = {}
    for item in ipa_items:
        q = item['quadrant']
        quadrant_counts[q] = quadrant_counts.get(q, 0) + 1
    
    priority_items = [item for item in ipa_items if 'Q2' in item['quadrant']]
    
    return {
        'ipa_matrix': ipa_items,
        'performance_mean': _to_native(perf_mean),
        'importance_mean': _to_native(imp_mean),
        'quadrant_counts': quadrant_counts,
        'priority_items': priority_items,
        'n_priority': len(priority_items)
    }


def create_ipa_chart(ipa_data: Dict) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    items = ipa_data.get('ipa_matrix', [])
    if not items:
        return ""
    
    quadrant_colors = {
        'Q1: Keep Up': '#10b981',
        'Q2: Focus Here': '#ef4444',
        'Q3: Low Priority': '#9ca3af',
        'Q4: Possible Overkill': '#f59e0b'
    }
    
    for item in items:
        color = quadrant_colors.get(item['quadrant'], '#3b82f6')
        ax.scatter(item['performance'], item['importance'], c=color, s=300, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(item['attribute'][:12], (item['performance'], item['importance']), 
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.axhline(y=ipa_data['importance_mean'], color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=ipa_data['performance_mean'], color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Performance (Mean Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance (Beta Coefficient)', fontsize=12, fontweight='bold')
    ax.set_title('IPA Matrix: Importance vs Performance', fontsize=14, fontweight='bold')
    
    for q, c in quadrant_colors.items():
        ax.scatter([], [], c=c, s=100, label=q, alpha=0.7, edgecolors='black')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Improvement Simulation
# =============================================================================
def simulate_improvement(df: pd.DataFrame, sat_col: str, attr_cols: List[str], drivers_data: Dict) -> Dict:
    df_clean = df[[sat_col] + attr_cols].dropna()
    
    current_satisfaction = df_clean[sat_col].mean()
    current_performance = df_clean[attr_cols].mean().to_dict()
    
    drivers = drivers_data.get('drivers', [])
    if not drivers:
        return {'error': 'No drivers data'}
    
    scenarios = []
    
    # Scenario 1: Top driver +1 point
    top_driver = drivers[0]
    improvement = 1.0
    expected_change = top_driver['beta'] * improvement
    scenarios.append({
        'name': f"Improve {top_driver['attribute']} +1pt",
        'target_attribute': top_driver['attribute'],
        'current_score': _to_native(current_performance.get(top_driver['attribute'], 0)),
        'improvement': improvement,
        'expected_satisfaction_change': _to_native(expected_change),
        'new_satisfaction': _to_native(current_satisfaction + expected_change)
    })
    
    # Scenario 2: Bottom 3 attributes +0.5
    bottom_performers = sorted(current_performance.items(), key=lambda x: x[1])[:3]
    total_change = 0
    for attr, score in bottom_performers:
        driver = next((d for d in drivers if d['attribute'] == attr), None)
        if driver:
            total_change += driver['beta'] * 0.5
    scenarios.append({
        'name': "Bottom 3 attrs +0.5pt each",
        'target_attribute': ', '.join([b[0][:10] for b in bottom_performers]),
        'improvement': 0.5,
        'expected_satisfaction_change': _to_native(total_change),
        'new_satisfaction': _to_native(current_satisfaction + total_change)
    })
    
    # Scenario 3: All attributes +0.3
    total_change_all = sum(d['beta'] * 0.3 for d in drivers)
    scenarios.append({
        'name': "All attributes +0.3pt",
        'target_attribute': 'All',
        'improvement': 0.3,
        'expected_satisfaction_change': _to_native(total_change_all),
        'new_satisfaction': _to_native(current_satisfaction + total_change_all)
    })
    
    best_scenario = max(scenarios, key=lambda x: x['expected_satisfaction_change'])
    
    recommendations = []
    priority_attrs = [d for d in drivers if d['beta'] > 0][:3]
    for d in priority_attrs:
        current = current_performance.get(d['attribute'], 0)
        recommendations.append({
            'attribute': d['attribute'],
            'current_score': _to_native(current),
            'impact_per_point': _to_native(d['beta']),
            'priority': 'high' if d['relative_importance'] > 20 else 'medium'
        })
    
    return {
        'current_satisfaction': _to_native(current_satisfaction),
        'scenarios': scenarios,
        'best_scenario': best_scenario,
        'recommendations': recommendations,
        'model_r_squared': drivers_data.get('r_squared', 0)
    }


def create_simulation_chart(sim_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = sim_data.get('scenarios', [])
    current = sim_data.get('current_satisfaction', 0)
    
    ax1 = axes[0]
    names = ['Current'] + [s['name'][:20] for s in scenarios]
    values = [current] + [s['new_satisfaction'] for s in scenarios]
    colors = ['#6b7280'] + ['#10b981' if s['expected_satisfaction_change'] > 0 else '#ef4444' for s in scenarios]
    
    bars = ax1.bar(range(len(names)), values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Satisfaction Score')
    ax1.set_title('Scenario Impact on Satisfaction', fontsize=11, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}',
                ha='center', fontsize=9)
    
    ax2 = axes[1]
    if scenarios:
        names = [s['name'][:20] for s in scenarios]
        changes = [s['expected_satisfaction_change'] for s in scenarios]
        colors = ['#10b981' if c > 0 else '#ef4444' for c in changes]
        ax2.barh(names, changes, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Expected Change in Satisfaction')
        ax2.set_title('Improvement Impact', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(dist: Dict, group: Optional[Dict], drivers: Dict, ipa: Dict, sim: Dict) -> Dict:
    report = {}
    
    report['step1_distribution'] = {
        'title': '1. Overall Satisfaction Distribution',
        'question': 'What is the current satisfaction level?',
        'finding': f"Mean satisfaction {dist['mean']:.2f}, {dist['n_responses']} responses",
        'detail': f"Analysis of {dist['n_responses']} responses shows mean satisfaction of {dist['mean']:.2f} "
                 f"(std: {dist['std']:.2f}). Distribution: High {dist['satisfaction_levels']['high_pct']:.1f}%, "
                 f"Medium {dist['satisfaction_levels']['mid_pct']:.1f}%, Low {dist['satisfaction_levels']['low_pct']:.1f}%."
    }
    
    if group and not group.get('error'):
        report['step2_group'] = {
            'title': '2. Group Satisfaction Comparison',
            'question': 'Are there differences between groups?',
            'finding': f"Best: {group['best_group']['group']} ({group['best_group']['mean']:.2f}), Worst: {group['worst_group']['group']} ({group['worst_group']['mean']:.2f})",
            'detail': f"Among {group['n_groups']} groups, {group['best_group']['group']} scored highest ({group['best_group']['mean']:.2f}) "
                     f"while {group['worst_group']['group']} scored lowest ({group['worst_group']['mean']:.2f}). "
                     f"Gap: {group['gap']:.2f}pts. {'Statistically significant' if group['statistical_test']['significant'] else 'Not significant'} "
                     f"(p={group['statistical_test']['p_value']:.4f})."
        }
    else:
        report['step2_group'] = {
            'title': '2. Group Satisfaction Comparison',
            'question': 'Are there differences between groups?',
            'finding': 'Group analysis not performed',
            'detail': 'No group column specified for comparison analysis.'
        }
    
    if drivers and not drivers.get('error'):
        top = drivers.get('top_driver', {})
        report['step3_drivers'] = {
            'title': '3. Key Drivers Analysis',
            'question': 'What factors drive satisfaction?',
            'finding': f"Top driver: {top.get('attribute', 'N/A')} ({top.get('relative_importance', 0):.1f}% importance)",
            'detail': f"Regression analysis (R²={drivers['r_squared']:.3f}) shows {top.get('attribute', 'N/A')} "
                     f"has highest impact (β={top.get('beta', 0):.3f}). Model quality: {drivers['model_quality']}."
        }
    else:
        report['step3_drivers'] = {
            'title': '3. Key Drivers Analysis',
            'question': 'What factors drive satisfaction?',
            'finding': 'Analysis unavailable',
            'detail': 'Insufficient data for driver analysis.'
        }
    
    if ipa and not ipa.get('error'):
        report['step4_ipa'] = {
            'title': '4. IPA Analysis',
            'question': 'Which attributes need priority improvement?',
            'finding': f"{ipa['n_priority']} items need focus (Q2: Focus Here)",
            'detail': f"IPA analysis identified {ipa['n_priority']} attributes in Q2 (high importance, low performance) "
                     f"requiring immediate attention. Q1 items should be maintained at current levels."
        }
    else:
        report['step4_ipa'] = {
            'title': '4. IPA Analysis',
            'question': 'Which attributes need priority improvement?',
            'finding': 'Analysis unavailable',
            'detail': 'Could not perform IPA analysis.'
        }
    
    if sim and not sim.get('error'):
        best = sim.get('best_scenario', {})
        report['step5_simulation'] = {
            'title': '5. Improvement Simulation',
            'question': 'How much can satisfaction improve?',
            'finding': f"Best scenario: {best.get('name', 'N/A')[:30]} → {best.get('new_satisfaction', 0):.2f}",
            'detail': f"From current satisfaction of {sim['current_satisfaction']:.2f}, "
                     f"'{best.get('name', 'N/A')}' scenario yields +{best.get('expected_satisfaction_change', 0):.2f} improvement "
                     f"to reach {best.get('new_satisfaction', 0):.2f}."
        }
    else:
        report['step5_simulation'] = {
            'title': '5. Improvement Simulation',
            'question': 'How much can satisfaction improve?',
            'finding': 'Simulation unavailable',
            'detail': 'Could not perform improvement simulation.'
        }
    
    return report


def generate_insights(dist: Dict, group: Optional[Dict], drivers: Dict, ipa: Dict, sim: Dict) -> List[Dict]:
    insights = []
    
    if dist['satisfaction_levels']['low_pct'] > 20:
        insights.append({
            'title': 'High Dissatisfaction Rate',
            'description': f"{dist['satisfaction_levels']['low_pct']:.1f}% of respondents show low satisfaction.",
            'status': 'warning'
        })
    elif dist['satisfaction_levels']['high_pct'] > 50:
        insights.append({
            'title': 'Strong Satisfaction',
            'description': f"{dist['satisfaction_levels']['high_pct']:.1f}% of respondents show high satisfaction.",
            'status': 'positive'
        })
    
    if group and group.get('gap', 0) > 1:
        insights.append({
            'title': 'Group Gap Detected',
            'description': f"{group['gap']:.2f}pt gap between {group['best_group']['group']} and {group['worst_group']['group']}.",
            'status': 'warning'
        })
    
    if ipa and ipa.get('n_priority', 0) > 0:
        priority_names = ', '.join([p['attribute'][:15] for p in ipa['priority_items'][:3]])
        insights.append({
            'title': 'Priority Improvements Identified',
            'description': f"{ipa['n_priority']} items need focus: {priority_names}.",
            'status': 'warning'
        })
    
    if sim and sim.get('best_scenario'):
        best = sim['best_scenario']
        insights.append({
            'title': 'Improvement Opportunity',
            'description': f"'{best['name'][:25]}' can boost satisfaction by {best['expected_satisfaction_change']:.2f}pts.",
            'status': 'positive'
        })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/satisfaction-analysis")
async def analyze_satisfaction(request: SatisfactionRequest):
    try:
        df = pd.DataFrame(request.data)
        sat_col = request.satisfaction_col
        attr_cols = request.attribute_cols
        group_col = request.group_col
        
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 data points")
        
        if sat_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Satisfaction column '{sat_col}' not found")
        
        df[sat_col] = pd.to_numeric(df[sat_col], errors='coerce')
        for col in attr_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1
        dist = analyze_distribution(df, sat_col)
        results['distribution'] = dist
        visualizations['distribution_chart'] = create_distribution_chart(df, sat_col, dist)
        
        # Step 2
        group = None
        if group_col and group_col in df.columns:
            group = analyze_group_comparison(df, sat_col, group_col)
            results['group_comparison'] = group
            visualizations['group_chart'] = create_group_chart(group)
        
        # Step 3
        drivers = analyze_key_drivers(df, sat_col, attr_cols)
        results['drivers'] = drivers
        if not drivers.get('error'):
            visualizations['drivers_chart'] = create_drivers_chart(drivers)
        
        # Step 4
        ipa = analyze_ipa(df, sat_col, attr_cols)
        results['ipa'] = ipa
        if not ipa.get('error'):
            visualizations['ipa_chart'] = create_ipa_chart(ipa)
        
        # Step 5
        sim = simulate_improvement(df, sat_col, attr_cols, drivers)
        results['simulation'] = sim
        if not sim.get('error'):
            visualizations['simulation_chart'] = create_simulation_chart(sim)
        
        report = generate_report(dist, group, drivers, ipa, sim)
        insights = generate_insights(dist, group, drivers, ipa, sim)
        
        summary = {
            'n_responses': dist['n_responses'],
            'mean_satisfaction': dist['mean'],
            'top_driver': drivers.get('top_driver', {}).get('attribute') if drivers else None,
            'n_priority_items': ipa.get('n_priority', 0) if ipa else 0,
            'best_scenario': sim.get('best_scenario', {}).get('name') if sim else None
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
