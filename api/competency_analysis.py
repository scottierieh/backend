"""
Competency & Skill Diagnosis API
5-step framework for comprehensive competency analysis
1. Competency Distribution
2. Role Competency Comparison
3. Competency-Performance Relationship
4. Skill Gap Analysis
5. Training Impact
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


class CompetencyRequest(BaseModel):
    data: List[Dict[str, Any]]
    competency_cols: List[str]  # Competency scores
    role_col: Optional[str] = None  # Job role/position column
    performance_col: Optional[str] = None  # Performance metric
    required_cols: Optional[List[str]] = None  # Required competency levels
    pre_col: Optional[str] = None  # Pre-training score
    post_col: Optional[str] = None  # Post-training score


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
# Step 1: Competency Distribution
# =============================================================================
def analyze_competency_distribution(df: pd.DataFrame, competency_cols: List[str]) -> Dict:
    results = []
    
    for col in competency_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) == 0:
            continue
        
        # Determine scale
        max_val = values.max()
        if max_val <= 5:
            scale = 5
            low_threshold = 2.5
            high_threshold = 4.0
        elif max_val <= 10:
            scale = 10
            low_threshold = 5.0
            high_threshold = 8.0
        else:
            scale = 100
            low_threshold = 50
            high_threshold = 80
        
        low_count = len(values[values < low_threshold])
        mid_count = len(values[(values >= low_threshold) & (values < high_threshold)])
        high_count = len(values[values >= high_threshold])
        
        results.append({
            'competency': col,
            'n': len(values),
            'mean': _to_native(values.mean()),
            'std': _to_native(values.std()),
            'min': _to_native(values.min()),
            'max': _to_native(values.max()),
            'median': _to_native(values.median()),
            'q1': _to_native(values.quantile(0.25)),
            'q3': _to_native(values.quantile(0.75)),
            'scale': scale,
            'proficiency': {
                'low': _to_native(low_count),
                'mid': _to_native(mid_count),
                'high': _to_native(high_count),
                'low_pct': _to_native(low_count / len(values) * 100),
                'mid_pct': _to_native(mid_count / len(values) * 100),
                'high_pct': _to_native(high_count / len(values) * 100)
            }
        })
    
    # Sort by mean (descending)
    results = sorted(results, key=lambda x: x['mean'], reverse=True)
    
    # Overall stats
    overall_means = [r['mean'] for r in results]
    
    return {
        'competencies': results,
        'n_competencies': len(results),
        'n_employees': len(df),
        'overall_mean': _to_native(np.mean(overall_means)) if overall_means else None,
        'strongest': results[0] if results else None,
        'weakest': results[-1] if results else None
    }


def create_distribution_chart(dist_data: Dict, df: pd.DataFrame, competency_cols: List[str]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    comps = dist_data.get('competencies', [])
    
    # Chart 1: Mean scores by competency
    ax1 = axes[0]
    if comps:
        names = [c['competency'][:15] for c in comps]
        means = [c['mean'] for c in comps]
        colors = ['#10b981' if m >= comps[0]['mean'] * 0.9 else '#ef4444' if m < comps[-1]['mean'] * 1.1 else '#3b82f6' for m in means]
        ax1.barh(names, means, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Mean Score')
        ax1.set_title('Competency Scores Overview', fontsize=11, fontweight='bold')
    
    # Chart 2: Proficiency distribution (stacked)
    ax2 = axes[1]
    if comps:
        names = [c['competency'][:12] for c in comps]
        lows = [c['proficiency']['low_pct'] for c in comps]
        mids = [c['proficiency']['mid_pct'] for c in comps]
        highs = [c['proficiency']['high_pct'] for c in comps]
        
        y_pos = range(len(comps))
        ax2.barh(y_pos, lows, color='#ef4444', alpha=0.7, label='Low')
        ax2.barh(y_pos, mids, left=lows, color='#f59e0b', alpha=0.7, label='Mid')
        ax2.barh(y_pos, highs, left=[l+m for l,m in zip(lows, mids)], color='#10b981', alpha=0.7, label='High')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Percentage (%)')
        ax2.set_title('Proficiency Levels', fontsize=11, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=8)
    
    # Chart 3: Box plot
    ax3 = axes[2]
    plot_data = [pd.to_numeric(df[col], errors='coerce').dropna() for col in competency_cols[:6]]
    if plot_data:
        bp = ax3.boxplot(plot_data, patch_artist=True, labels=[c[:10] for c in competency_cols[:6]])
        colors_box = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_ylabel('Score')
        ax3.set_title('Score Distribution', fontsize=11, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Role Competency Comparison
# =============================================================================
def analyze_role_comparison(df: pd.DataFrame, competency_cols: List[str], role_col: str) -> Dict:
    role_stats = []
    
    for role in df[role_col].unique():
        role_data = df[df[role_col] == role]
        comp_means = role_data[competency_cols].apply(pd.to_numeric, errors='coerce').mean()
        overall_mean = comp_means.mean()
        
        role_stats.append({
            'role': _to_native(role),
            'n_employees': len(role_data),
            'overall_mean': _to_native(overall_mean),
            'competency_means': {col: _to_native(comp_means[col]) for col in competency_cols}
        })
    
    role_stats = sorted(role_stats, key=lambda x: x['overall_mean'], reverse=True)
    
    # ANOVA for each competency
    comp_differences = []
    for col in competency_cols:
        groups = [df[df[role_col] == role][col].dropna().values for role in df[role_col].unique()]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                significant = bool(p_value < 0.05)
            except:
                f_stat, p_value, significant = None, None, False
        else:
            f_stat, p_value, significant = None, None, False
        
        comp_differences.append({
            'competency': col,
            'f_statistic': _to_native(f_stat),
            'p_value': _to_native(p_value),
            'significant': significant
        })
    
    return {
        'role_stats': role_stats,
        'n_roles': len(role_stats),
        'competency_differences': comp_differences,
        'n_significant': sum(1 for c in comp_differences if c['significant']),
        'top_role': role_stats[0] if role_stats else None,
        'bottom_role': role_stats[-1] if role_stats else None
    }


def create_role_chart(role_data: Dict, df: pd.DataFrame, competency_cols: List[str], role_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Role overall means
    ax1 = axes[0]
    roles = [r['role'] for r in role_data['role_stats']]
    means = [r['overall_mean'] for r in role_data['role_stats']]
    colors = ['#10b981' if i == 0 else '#ef4444' if i == len(roles)-1 else '#3b82f6' for i in range(len(roles))]
    ax1.bar(roles, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Role')
    ax1.set_ylabel('Overall Competency Mean')
    ax1.set_title('Competency by Role', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Chart 2: Heatmap-like comparison
    ax2 = axes[1]
    role_comp_matrix = df.groupby(role_col)[competency_cols].mean()
    im = ax2.imshow(role_comp_matrix.values, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(len(competency_cols)))
    ax2.set_xticklabels([c[:10] for c in competency_cols], rotation=45, ha='right')
    ax2.set_yticks(range(len(role_comp_matrix.index)))
    ax2.set_yticklabels(role_comp_matrix.index)
    ax2.set_title('Role-Competency Matrix', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Score')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Competency-Performance Relationship
# =============================================================================
def analyze_competency_performance(df: pd.DataFrame, competency_cols: List[str], performance_col: str) -> Dict:
    df_clean = df[competency_cols + [performance_col]].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Insufficient data'}
    
    performance = df_clean[performance_col]
    
    # Correlations
    correlations = []
    for comp in competency_cols:
        corr, p_value = stats.pearsonr(df_clean[comp], performance)
        correlations.append({
            'competency': comp,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_value),
            'significant': bool(p_value < 0.05),
            'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
        })
    
    correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    # Regression analysis
    X = df_clean[competency_cols]
    y = performance
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    r2 = model.score(X_scaled, y)
    
    # Beta coefficients
    drivers = []
    total_abs = sum(abs(c) for c in model.coef_)
    for comp, coef in zip(competency_cols, model.coef_):
        rel_imp = (abs(coef) / total_abs * 100) if total_abs > 0 else 0
        drivers.append({
            'competency': comp,
            'beta': _to_native(coef),
            'relative_importance': _to_native(rel_imp)
        })
    
    drivers = sorted(drivers, key=lambda x: abs(x['beta']), reverse=True)
    
    return {
        'correlations': correlations,
        'top_predictor': correlations[0] if correlations else None,
        'drivers': drivers,
        'r_squared': _to_native(r2),
        'model_quality': 'good' if r2 > 0.5 else 'moderate' if r2 > 0.3 else 'low',
        'n_observations': len(df_clean),
        'n_significant': sum(1 for c in correlations if c['significant'])
    }


def create_performance_chart(perf_data: Dict, df: pd.DataFrame, competency_cols: List[str], performance_col: str) -> str:
    if perf_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Correlation bars
    ax1 = axes[0]
    corrs = perf_data.get('correlations', [])
    if corrs:
        comps = [c['competency'][:12] for c in corrs]
        vals = [c['correlation'] for c in corrs]
        colors = ['#10b981' if v > 0 else '#ef4444' for v in vals]
        ax1.barh(comps, vals, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Correlation with Performance')
        ax1.set_title('Competency-Performance Correlation', fontsize=11, fontweight='bold')
    
    # Chart 2: Relative importance
    ax2 = axes[1]
    drivers = perf_data.get('drivers', [])
    if drivers:
        comps = [d['competency'][:12] for d in drivers]
        imps = [d['relative_importance'] for d in drivers]
        ax2.barh(comps, imps, color='#3b82f6', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Relative Importance (%)')
        ax2.set_title('Performance Drivers', fontsize=11, fontweight='bold')
    
    # Chart 3: Scatter of top predictor
    ax3 = axes[2]
    top = perf_data.get('top_predictor')
    if top:
        df_clean = df[[top['competency'], performance_col]].apply(pd.to_numeric, errors='coerce').dropna()
        ax3.scatter(df_clean[top['competency']], df_clean[performance_col], alpha=0.5, color='#3b82f6')
        z = np.polyfit(df_clean[top['competency']], df_clean[performance_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_clean[top['competency']].min(), df_clean[top['competency']].max(), 100)
        ax3.plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
        ax3.set_xlabel(top['competency'])
        ax3.set_ylabel(performance_col)
        ax3.set_title(f"Top Predictor (r={top['correlation']:.3f})", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Skill Gap Analysis
# =============================================================================
def analyze_skill_gap(df: pd.DataFrame, competency_cols: List[str], required_cols: Optional[List[str]] = None) -> Dict:
    # If required columns provided, use them; otherwise use threshold
    if required_cols and len(required_cols) == len(competency_cols):
        # Match required to actual
        gaps = []
        for comp, req in zip(competency_cols, required_cols):
            actual = pd.to_numeric(df[comp], errors='coerce').dropna()
            required = pd.to_numeric(df[req], errors='coerce').dropna() if req in df.columns else None
            
            if required is not None and len(required) == len(actual):
                gap_values = required.values - actual.values
                avg_gap = gap_values.mean()
                gap_positive = (gap_values > 0).sum()
            else:
                # Use scale midpoint as required
                max_val = actual.max()
                if max_val <= 5:
                    req_level = 4.0
                elif max_val <= 10:
                    req_level = 7.0
                else:
                    req_level = 70
                gap_values = req_level - actual.values
                avg_gap = gap_values.mean()
                gap_positive = (gap_values > 0).sum()
            
            gaps.append({
                'competency': comp,
                'actual_mean': _to_native(actual.mean()),
                'required_level': _to_native(required.mean() if required is not None else req_level),
                'gap': _to_native(avg_gap),
                'gap_pct': _to_native(gap_positive / len(actual) * 100),
                'n_below': _to_native(gap_positive),
                'priority': 'high' if avg_gap > 1 else 'medium' if avg_gap > 0.5 else 'low'
            })
    else:
        # Use standard thresholds
        gaps = []
        for comp in competency_cols:
            actual = pd.to_numeric(df[comp], errors='coerce').dropna()
            max_val = actual.max()
            
            if max_val <= 5:
                req_level = 4.0
            elif max_val <= 10:
                req_level = 7.0
            else:
                req_level = 70
            
            gap = req_level - actual.mean()
            n_below = (actual < req_level).sum()
            
            gaps.append({
                'competency': comp,
                'actual_mean': _to_native(actual.mean()),
                'required_level': _to_native(req_level),
                'gap': _to_native(gap),
                'gap_pct': _to_native(n_below / len(actual) * 100),
                'n_below': _to_native(n_below),
                'priority': 'high' if gap > 1 else 'medium' if gap > 0.5 else 'low'
            })
    
    # Sort by gap (largest first)
    gaps = sorted(gaps, key=lambda x: x['gap'], reverse=True)
    
    # Summary
    high_priority = [g for g in gaps if g['priority'] == 'high']
    total_gap = sum(g['gap'] for g in gaps if g['gap'] > 0)
    
    return {
        'gaps': gaps,
        'n_competencies': len(gaps),
        'n_high_priority': len(high_priority),
        'high_priority_skills': high_priority,
        'total_gap': _to_native(total_gap),
        'largest_gap': gaps[0] if gaps else None,
        'smallest_gap': gaps[-1] if gaps else None
    }


def create_gap_chart(gap_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gaps = gap_data.get('gaps', [])
    
    # Chart 1: Actual vs Required
    ax1 = axes[0]
    if gaps:
        comps = [g['competency'][:12] for g in gaps]
        actuals = [g['actual_mean'] for g in gaps]
        requireds = [g['required_level'] for g in gaps]
        
        x = np.arange(len(comps))
        width = 0.35
        ax1.bar(x - width/2, actuals, width, label='Actual', color='#3b82f6', alpha=0.7, edgecolor='black')
        ax1.bar(x + width/2, requireds, width, label='Required', color='#ef4444', alpha=0.7, edgecolor='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comps, rotation=45, ha='right')
        ax1.set_ylabel('Score')
        ax1.set_title('Actual vs Required Competency', fontsize=11, fontweight='bold')
        ax1.legend()
    
    # Chart 2: Gap magnitude
    ax2 = axes[1]
    if gaps:
        comps = [g['competency'][:12] for g in gaps]
        gap_vals = [g['gap'] for g in gaps]
        colors = ['#ef4444' if g > 1 else '#f59e0b' if g > 0.5 else '#10b981' for g in gap_vals]
        ax2.barh(comps, gap_vals, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium threshold')
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='High threshold')
        ax2.set_xlabel('Skill Gap (Required - Actual)')
        ax2.set_title('Skill Gap Analysis', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Training Impact
# =============================================================================
def analyze_training_impact(df: pd.DataFrame, pre_col: str, post_col: str, role_col: Optional[str] = None) -> Dict:
    df_clean = df[[pre_col, post_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if role_col and role_col in df.columns:
        df_clean[role_col] = df.loc[df_clean.index, role_col]
    
    if len(df_clean) < 5:
        return {'error': 'Insufficient paired data'}
    
    pre = df_clean[pre_col]
    post = df_clean[post_col]
    
    # Basic stats
    pre_mean = pre.mean()
    post_mean = post.mean()
    change = post_mean - pre_mean
    change_pct = (change / pre_mean) * 100 if pre_mean != 0 else 0
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(pre, post)
    significant = bool(p_value < 0.05)
    
    # Effect size (Cohen's d)
    diff = post - pre
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    effect_size = 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    
    # Individual changes
    improved = (diff > 0).sum()
    declined = (diff < 0).sum()
    unchanged = (diff == 0).sum()
    
    result = {
        'n_participants': len(df_clean),
        'pre_mean': _to_native(pre_mean),
        'post_mean': _to_native(post_mean),
        'change': _to_native(change),
        'change_pct': _to_native(change_pct),
        't_statistic': _to_native(t_stat),
        'p_value': _to_native(p_value),
        'significant': significant,
        'cohens_d': _to_native(cohens_d),
        'effect_size': effect_size,
        'improved': _to_native(improved),
        'declined': _to_native(declined),
        'unchanged': _to_native(unchanged),
        'improved_pct': _to_native(improved / len(df_clean) * 100),
        'effectiveness': 'effective' if significant and change > 0 else 'ineffective'
    }
    
    # By role if available
    if role_col and role_col in df_clean.columns:
        role_impact = []
        for role in df_clean[role_col].unique():
            role_data = df_clean[df_clean[role_col] == role]
            r_pre = role_data[pre_col].mean()
            r_post = role_data[post_col].mean()
            r_change = r_post - r_pre
            
            role_impact.append({
                'role': _to_native(role),
                'pre_mean': _to_native(r_pre),
                'post_mean': _to_native(r_post),
                'change': _to_native(r_change),
                'n': len(role_data)
            })
        
        role_impact = sorted(role_impact, key=lambda x: x['change'], reverse=True)
        result['role_impact'] = role_impact
        result['best_improvement_role'] = role_impact[0] if role_impact else None
    
    return result


def create_training_chart(training_data: Dict, df: pd.DataFrame, pre_col: str, post_col: str) -> str:
    if training_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Pre vs Post
    ax1 = axes[0]
    means = [training_data['pre_mean'], training_data['post_mean']]
    labels = ['Pre-Training', 'Post-Training']
    colors = ['#6b7280', '#10b981' if training_data['change'] > 0 else '#ef4444']
    bars = ax1.bar(labels, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Training Impact', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', ha='center')
    
    # Chart 2: Individual changes
    ax2 = axes[1]
    df_clean = df[[pre_col, post_col]].apply(pd.to_numeric, errors='coerce').dropna()
    for _, row in df_clean.sample(min(50, len(df_clean))).iterrows():
        color = '#10b981' if row[post_col] > row[pre_col] else '#ef4444' if row[post_col] < row[pre_col] else '#9ca3af'
        ax2.plot([0, 1], [row[pre_col], row[post_col]], color=color, alpha=0.3)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Pre', 'Post'])
    ax2.set_ylabel('Score')
    ax2.set_title('Individual Progress', fontsize=11, fontweight='bold')
    
    # Chart 3: Outcome distribution
    ax3 = axes[2]
    outcomes = ['Improved', 'Unchanged', 'Declined']
    counts = [training_data['improved'], training_data['unchanged'], training_data['declined']]
    colors = ['#10b981', '#9ca3af', '#ef4444']
    ax3.pie(counts, labels=outcomes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Training Outcomes', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(dist: Dict, role: Optional[Dict], perf: Optional[Dict], gap: Dict, training: Optional[Dict]) -> Dict:
    report = {}
    
    report['step1_distribution'] = {
        'title': '1. Competency Distribution',
        'question': 'What is the current competency level?',
        'finding': f"{dist['n_employees']} employees, {dist['n_competencies']} competencies assessed",
        'detail': f"Analysis of {dist['n_competencies']} competencies shows overall mean of {dist['overall_mean']:.2f}. "
                 f"Strongest: {dist['strongest']['competency']} ({dist['strongest']['mean']:.2f}). "
                 f"Weakest: {dist['weakest']['competency']} ({dist['weakest']['mean']:.2f})."
    }
    
    if role and not role.get('error'):
        report['step2_role'] = {
            'title': '2. Role Competency Comparison',
            'question': 'How do roles differ in competency?',
            'finding': f"{role['n_roles']} roles compared, {role['n_significant']} significant differences",
            'detail': f"Comparison across {role['n_roles']} roles shows {role['n_significant']} competencies with significant differences. "
                     f"Top role: {role['top_role']['role']} ({role['top_role']['overall_mean']:.2f}). "
                     f"Needs development: {role['bottom_role']['role']} ({role['bottom_role']['overall_mean']:.2f})."
        }
    else:
        report['step2_role'] = {
            'title': '2. Role Competency Comparison',
            'question': 'How do roles differ in competency?',
            'finding': 'Role analysis not performed',
            'detail': 'No role column specified.'
        }
    
    if perf and not perf.get('error'):
        report['step3_performance'] = {
            'title': '3. Competency-Performance Relationship',
            'question': 'Which competencies drive performance?',
            'finding': f"Top predictor: {perf['top_predictor']['competency']} (r={perf['top_predictor']['correlation']:.3f})",
            'detail': f"Regression analysis (R²={perf['r_squared']:.3f}) shows {perf['n_significant']} competencies significantly correlated with performance. "
                     f"Model quality: {perf['model_quality']}."
        }
    else:
        report['step3_performance'] = {
            'title': '3. Competency-Performance Relationship',
            'question': 'Which competencies drive performance?',
            'finding': 'Performance analysis not performed',
            'detail': perf.get('error', 'No performance column specified.')
        }
    
    report['step4_gap'] = {
        'title': '4. Skill Gap Analysis',
        'question': 'Where are the competency gaps?',
        'finding': f"{gap['n_high_priority']} high-priority gaps identified",
        'detail': f"Gap analysis reveals {gap['n_high_priority']} competencies requiring urgent attention. "
                 f"Largest gap: {gap['largest_gap']['competency']} (gap={gap['largest_gap']['gap']:.2f}). "
                 f"Total development gap: {gap['total_gap']:.2f} points."
    }
    
    if training and not training.get('error'):
        report['step5_training'] = {
            'title': '5. Training Impact',
            'question': 'Did training improve competency?',
            'finding': f"Change: {training['change']:+.2f} ({training['change_pct']:+.1f}%), {training['effect_size']} effect",
            'detail': f"Pre-post analysis (n={training['n_participants']}) shows {training['effectiveness']} training. "
                     f"{'Significant' if training['significant'] else 'Not significant'} improvement (p={training['p_value']:.4f}). "
                     f"{training['improved_pct']:.1f}% of participants improved."
        }
    else:
        report['step5_training'] = {
            'title': '5. Training Impact',
            'question': 'Did training improve competency?',
            'finding': 'Training analysis not performed',
            'detail': training.get('error', 'No pre/post columns specified.')
        }
    
    return report


def generate_insights(dist: Dict, role: Optional[Dict], perf: Optional[Dict], gap: Dict, training: Optional[Dict]) -> List[Dict]:
    insights = []
    
    # Gap insights
    if gap['n_high_priority'] > 0:
        skills = ', '.join([g['competency'][:15] for g in gap['high_priority_skills'][:3]])
        insights.append({
            'title': 'Critical Skill Gaps',
            'description': f"{gap['n_high_priority']} high-priority gaps: {skills}",
            'status': 'warning'
        })
    
    # Role insights
    if role and role.get('bottom_role'):
        insights.append({
            'title': 'Role Development Needed',
            'description': f"{role['bottom_role']['role']} shows lowest competency ({role['bottom_role']['overall_mean']:.2f}).",
            'status': 'warning'
        })
    
    # Training insights
    if training and not training.get('error'):
        if training['significant'] and training['change'] > 0:
            insights.append({
                'title': 'Training Effective',
                'description': f"Significant improvement of {training['change']:+.2f} pts ({training['effect_size']} effect).",
                'status': 'positive'
            })
        elif training['change'] <= 0:
            insights.append({
                'title': 'Training Review Needed',
                'description': f"No improvement observed (change: {training['change']:+.2f}).",
                'status': 'warning'
            })
    
    # Performance insights
    if perf and not perf.get('error') and perf['top_predictor']:
        insights.append({
            'title': 'Key Performance Driver',
            'description': f"{perf['top_predictor']['competency']} most strongly predicts performance (r={perf['top_predictor']['correlation']:.3f}).",
            'status': 'positive'
        })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/competency-analysis")
async def analyze_competency(request: CompetencyRequest):
    try:
        df = pd.DataFrame(request.data)
        competency_cols = request.competency_cols
        role_col = request.role_col
        performance_col = request.performance_col
        required_cols = request.required_cols
        pre_col = request.pre_col
        post_col = request.post_col
        
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 data points")
        
        # Convert competency columns to numeric
        for col in competency_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Distribution
        dist = analyze_competency_distribution(df, competency_cols)
        results['distribution'] = dist
        visualizations['distribution_chart'] = create_distribution_chart(dist, df, competency_cols)
        
        # Step 2: Role Comparison
        role = None
        if role_col and role_col in df.columns:
            role = analyze_role_comparison(df, competency_cols, role_col)
            results['role'] = role
            visualizations['role_chart'] = create_role_chart(role, df, competency_cols, role_col)
        
        # Step 3: Performance Relationship
        perf = None
        if performance_col and performance_col in df.columns:
            df[performance_col] = pd.to_numeric(df[performance_col], errors='coerce')
            perf = analyze_competency_performance(df, competency_cols, performance_col)
            results['performance'] = perf
            if not perf.get('error'):
                visualizations['performance_chart'] = create_performance_chart(perf, df, competency_cols, performance_col)
        
        # Step 4: Skill Gap
        gap = analyze_skill_gap(df, competency_cols, required_cols)
        results['gap'] = gap
        visualizations['gap_chart'] = create_gap_chart(gap)
        
        # Step 5: Training Impact
        training = None
        if pre_col and post_col and pre_col in df.columns and post_col in df.columns:
            df[pre_col] = pd.to_numeric(df[pre_col], errors='coerce')
            df[post_col] = pd.to_numeric(df[post_col], errors='coerce')
            training = analyze_training_impact(df, pre_col, post_col, role_col)
            results['training'] = training
            if not training.get('error'):
                visualizations['training_chart'] = create_training_chart(training, df, pre_col, post_col)
        
        report = generate_report(dist, role, perf, gap, training)
        insights = generate_insights(dist, role, perf, gap, training)
        
        summary = {
            'n_employees': dist['n_employees'],
            'n_competencies': dist['n_competencies'],
            'overall_mean': dist['overall_mean'],
            'n_high_priority_gaps': gap['n_high_priority'],
            'training_effective': training.get('effectiveness') == 'effective' if training else None
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
