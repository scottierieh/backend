"""
Perception & Attitude Analysis API
5-step framework for comprehensive perception/attitude survey analysis
1. Agreement Distribution
2. Group Perception Gap
3. Background-Perception Relationship
4. Perception Structure (Factor Analysis)
5. Pre-Post Campaign Change
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis, PCA
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


class PerceptionRequest(BaseModel):
    data: List[Dict[str, Any]]
    perception_cols: List[str]  # Likert scale perception items
    group_col: Optional[str] = None  # Group column for comparison
    background_cols: Optional[List[str]] = None  # Demographics/background
    pre_col: Optional[str] = None  # Pre-campaign score
    post_col: Optional[str] = None  # Post-campaign score


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
# Step 1: Agreement Distribution
# =============================================================================
def analyze_agreement_distribution(df: pd.DataFrame, perception_cols: List[str]) -> Dict:
    results = []
    
    for col in perception_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
            
        mean_val = values.mean()
        std_val = values.std()
        
        # Detect scale (5 or 7 point)
        max_val = values.max()
        if max_val <= 5:
            scale = 5
            agree_threshold = 4  # 4-5 = agree
            disagree_threshold = 2  # 1-2 = disagree
        else:
            scale = 7
            agree_threshold = 5  # 5-7 = agree
            disagree_threshold = 3  # 1-3 = disagree
        
        # Agreement rates
        agree_count = len(values[values >= agree_threshold])
        disagree_count = len(values[values <= disagree_threshold])
        neutral_count = len(values) - agree_count - disagree_count
        
        agree_pct = agree_count / len(values) * 100
        disagree_pct = disagree_count / len(values) * 100
        neutral_pct = neutral_count / len(values) * 100
        
        # Distribution
        distribution = values.value_counts().sort_index().to_dict()
        
        # Net agreement (agree - disagree)
        net_agreement = agree_pct - disagree_pct
        
        results.append({
            'item': col,
            'n': len(values),
            'mean': _to_native(mean_val),
            'std': _to_native(std_val),
            'scale': scale,
            'agree_pct': _to_native(agree_pct),
            'neutral_pct': _to_native(neutral_pct),
            'disagree_pct': _to_native(disagree_pct),
            'net_agreement': _to_native(net_agreement),
            'distribution': {str(k): _to_native(v) for k, v in distribution.items()}
        })
    
    # Sort by agreement rate
    results = sorted(results, key=lambda x: x['agree_pct'], reverse=True)
    
    # Overall stats
    overall_agree = np.mean([r['agree_pct'] for r in results]) if results else 0
    overall_disagree = np.mean([r['disagree_pct'] for r in results]) if results else 0
    
    return {
        'items': results,
        'n_items': len(results),
        'overall_agree_pct': _to_native(overall_agree),
        'overall_disagree_pct': _to_native(overall_disagree),
        'overall_net_agreement': _to_native(overall_agree - overall_disagree),
        'highest_agreement': results[0] if results else None,
        'lowest_agreement': results[-1] if results else None
    }


def create_agreement_chart(agree_data: Dict) -> str:
    items = agree_data.get('items', [])
    if not items:
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(items) * 0.4)))
    
    # Chart 1: Stacked bar (Disagree, Neutral, Agree)
    ax1 = axes[0]
    item_names = [i['item'][:20] for i in items]
    agrees = [i['agree_pct'] for i in items]
    neutrals = [i['neutral_pct'] for i in items]
    disagrees = [i['disagree_pct'] for i in items]
    
    y_pos = range(len(items))
    ax1.barh(y_pos, disagrees, color='#ef4444', alpha=0.7, label='Disagree')
    ax1.barh(y_pos, neutrals, left=disagrees, color='#9ca3af', alpha=0.7, label='Neutral')
    ax1.barh(y_pos, agrees, left=[d+n for d,n in zip(disagrees, neutrals)], color='#10b981', alpha=0.7, label='Agree')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(item_names)
    ax1.set_xlabel('Percentage (%)')
    ax1.set_title('Agreement Distribution by Item', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 100)
    
    # Chart 2: Net Agreement
    ax2 = axes[1]
    net_agrees = [i['net_agreement'] for i in items]
    colors = ['#10b981' if n > 0 else '#ef4444' for n in net_agrees]
    ax2.barh(y_pos, net_agrees, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(item_names)
    ax2.set_xlabel('Net Agreement (Agree% - Disagree%)')
    ax2.set_title('Net Agreement Score', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Group Perception Gap
# =============================================================================
def analyze_group_gap(df: pd.DataFrame, perception_cols: List[str], group_col: str) -> Dict:
    group_means = df.groupby(group_col)[perception_cols].mean()
    group_stds = df.groupby(group_col)[perception_cols].std()
    group_counts = df.groupby(group_col).size()
    
    # Overall mean per item
    item_results = []
    for col in perception_cols:
        col_data = group_means[col].to_dict()
        best_group = max(col_data, key=col_data.get)
        worst_group = min(col_data, key=col_data.get)
        gap = col_data[best_group] - col_data[worst_group]
        
        # ANOVA
        groups = [group[col].dropna().values for name, group in df.groupby(group_col)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                significant = bool(p_value < 0.05)
            except:
                f_stat, p_value, significant = None, None, False
        else:
            f_stat, p_value, significant = None, None, False
        
        item_results.append({
            'item': col,
            'best_group': best_group,
            'best_mean': _to_native(col_data[best_group]),
            'worst_group': worst_group,
            'worst_mean': _to_native(col_data[worst_group]),
            'gap': _to_native(gap),
            'f_statistic': _to_native(f_stat),
            'p_value': _to_native(p_value),
            'significant': significant
        })
    
    # Group summary
    group_summary = []
    overall_means = group_means.mean(axis=1)
    for grp in group_means.index:
        group_summary.append({
            'group': _to_native(grp),
            'overall_mean': _to_native(overall_means[grp]),
            'n': _to_native(group_counts[grp])
        })
    group_summary = sorted(group_summary, key=lambda x: x['overall_mean'], reverse=True)
    
    # Largest gaps
    item_results_sorted = sorted(item_results, key=lambda x: x['gap'], reverse=True)
    
    return {
        'item_gaps': item_results,
        'group_summary': group_summary,
        'n_groups': len(group_means),
        'largest_gap_item': item_results_sorted[0] if item_results_sorted else None,
        'n_significant': sum(1 for i in item_results if i['significant'])
    }


def create_group_gap_chart(gap_data: Dict, df: pd.DataFrame, perception_cols: List[str], group_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Group means comparison
    ax1 = axes[0]
    group_means = df.groupby(group_col)[perception_cols].mean()
    group_means.plot(kind='bar', ax=ax1, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Perception Scores by Group', fontsize=11, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.tick_params(axis='x', rotation=45)
    
    # Chart 2: Gap analysis
    ax2 = axes[1]
    items = [i['item'][:15] for i in gap_data['item_gaps']]
    gaps = [i['gap'] for i in gap_data['item_gaps']]
    colors = ['#ef4444' if g > 1 else '#f59e0b' if g > 0.5 else '#10b981' for g in gaps]
    ax2.barh(items, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Gap (Best - Worst Group)')
    ax2.set_title('Perception Gap by Item', fontsize=11, fontweight='bold')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate gap')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Background-Perception Relationship
# =============================================================================
def analyze_background_relationship(df: pd.DataFrame, perception_cols: List[str], background_cols: List[str]) -> Dict:
    results = []
    
    # Create composite perception score
    df_clean = df[perception_cols + background_cols].dropna()
    perception_score = df_clean[perception_cols].mean(axis=1)
    
    for bg_col in background_cols:
        bg_values = df_clean[bg_col]
        
        # Check if categorical or numeric
        if bg_values.dtype == 'object' or bg_values.nunique() <= 10:
            # Categorical - use ANOVA
            groups = [perception_score[bg_values == val].values for val in bg_values.unique()]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    effect_size = 'significant' if p_value < 0.05 else 'not significant'
                except:
                    f_stat, p_value, effect_size = None, None, 'error'
            else:
                f_stat, p_value, effect_size = None, None, 'insufficient groups'
            
            # Group means
            group_means = df_clean.groupby(bg_col)[perception_cols].mean().mean(axis=1).to_dict()
            
            results.append({
                'variable': bg_col,
                'type': 'categorical',
                'test': 'ANOVA',
                'statistic': _to_native(f_stat),
                'p_value': _to_native(p_value),
                'significant': bool(p_value < 0.05) if p_value else False,
                'group_means': {str(k): _to_native(v) for k, v in group_means.items()}
            })
        else:
            # Numeric - use correlation
            try:
                corr, p_value = stats.pearsonr(bg_values, perception_score)
                strength = 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
            except:
                corr, p_value, strength = None, None, 'error'
            
            results.append({
                'variable': bg_col,
                'type': 'numeric',
                'test': 'Pearson correlation',
                'statistic': _to_native(corr),
                'p_value': _to_native(p_value),
                'significant': bool(p_value < 0.05) if p_value else False,
                'strength': strength
            })
    
    # Sort by significance
    results = sorted(results, key=lambda x: x['p_value'] or 1)
    significant_vars = [r for r in results if r['significant']]
    
    return {
        'relationships': results,
        'n_variables': len(results),
        'n_significant': len(significant_vars),
        'significant_variables': significant_vars
    }


def create_background_chart(bg_data: Dict, df: pd.DataFrame, perception_cols: List[str], background_cols: List[str]) -> str:
    n_vars = len(background_cols)
    if n_vars == 0:
        return ""
    
    fig, axes = plt.subplots(1, min(n_vars, 3), figsize=(min(n_vars, 3) * 5, 4))
    if n_vars == 1:
        axes = [axes]
    
    perception_score = df[perception_cols].mean(axis=1)
    
    for i, bg_col in enumerate(background_cols[:3]):
        ax = axes[i]
        rel = next((r for r in bg_data['relationships'] if r['variable'] == bg_col), None)
        
        if rel and rel['type'] == 'categorical':
            means = df.groupby(bg_col)[perception_cols].mean().mean(axis=1)
            means.plot(kind='bar', ax=ax, color='#3b82f6', alpha=0.7, edgecolor='black')
            ax.set_ylabel('Mean Perception Score')
        else:
            ax.scatter(df[bg_col], perception_score, alpha=0.5, color='#3b82f6')
            z = np.polyfit(df[bg_col].dropna(), perception_score[df[bg_col].notna()], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[bg_col].min(), df[bg_col].max(), 100)
            ax.plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
            ax.set_ylabel('Perception Score')
        
        ax.set_xlabel(bg_col)
        sig_mark = '***' if rel and rel['significant'] else ''
        ax.set_title(f'{bg_col} {sig_mark}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Perception Structure (Factor Analysis)
# =============================================================================
def analyze_factor_structure(df: pd.DataFrame, perception_cols: List[str]) -> Dict:
    df_clean = df[perception_cols].dropna()
    
    if len(df_clean) < len(perception_cols) + 5:
        return {'error': 'Insufficient data for factor analysis'}
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Determine number of factors using Kaiser criterion
    pca = PCA()
    pca.fit(X_scaled)
    eigenvalues = pca.explained_variance_
    n_factors = sum(1 for ev in eigenvalues if ev > 1)
    n_factors = max(1, min(n_factors, len(perception_cols) // 2))
    
    # Factor Analysis
    try:
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(X_scaled)
        loadings = fa.components_.T  # Items x Factors
        
        # Variance explained
        variance_explained = np.var(fa.transform(X_scaled), axis=0)
        total_var = np.sum(variance_explained)
        variance_pct = variance_explained / X_scaled.var(axis=0).sum() * 100
        
        # Build factor structure
        factors = []
        for f_idx in range(n_factors):
            factor_loadings = loadings[:, f_idx]
            items_in_factor = []
            
            for i, col in enumerate(perception_cols):
                loading = factor_loadings[i]
                if abs(loading) > 0.3:  # Meaningful loading threshold
                    items_in_factor.append({
                        'item': col,
                        'loading': _to_native(loading)
                    })
            
            items_in_factor = sorted(items_in_factor, key=lambda x: abs(x['loading']), reverse=True)
            
            factors.append({
                'factor': f'Factor {f_idx + 1}',
                'variance_explained': _to_native(variance_pct[f_idx]),
                'eigenvalue': _to_native(eigenvalues[f_idx]),
                'items': items_in_factor,
                'n_items': len(items_in_factor)
            })
        
        # Communalities
        communalities = 1 - fa.noise_variance_
        item_communalities = [{'item': col, 'communality': _to_native(c)} for col, c in zip(perception_cols, communalities)]
        
        return {
            'factors': factors,
            'n_factors': n_factors,
            'total_variance_explained': _to_native(sum(variance_pct)),
            'eigenvalues': [_to_native(e) for e in eigenvalues[:n_factors]],
            'communalities': item_communalities,
            'kmo_adequate': len(df_clean) / len(perception_cols) > 5  # Rough KMO proxy
        }
    except Exception as e:
        return {'error': str(e)}


def create_factor_chart(factor_data: Dict) -> str:
    if factor_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Scree plot
    ax1 = axes[0]
    eigenvalues = factor_data.get('eigenvalues', [])
    n_factors = factor_data.get('n_factors', 0)
    
    if eigenvalues:
        x = range(1, len(eigenvalues) + 1)
        ax1.plot(x, eigenvalues, 'bo-', markersize=8)
        ax1.axhline(y=1, color='red', linestyle='--', label='Kaiser criterion')
        ax1.set_xlabel('Factor Number')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Scree Plot', fontsize=11, fontweight='bold')
        ax1.legend()
    
    # Chart 2: Variance explained
    ax2 = axes[1]
    factors = factor_data.get('factors', [])
    if factors:
        factor_names = [f['factor'] for f in factors]
        variances = [f['variance_explained'] for f in factors]
        ax2.bar(factor_names, variances, color='#3b82f6', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Variance Explained (%)')
        ax2.set_title('Variance by Factor', fontsize=11, fontweight='bold')
        
        # Add cumulative line
        cumulative = np.cumsum(variances)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(factor_names, cumulative, 'r-o', label='Cumulative')
        ax2_twin.set_ylabel('Cumulative %', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Pre-Post Campaign Change
# =============================================================================
def analyze_prepost_change(df: pd.DataFrame, pre_col: str, post_col: str, group_col: Optional[str] = None) -> Dict:
    df_clean = df[[pre_col, post_col]].dropna()
    if group_col:
        df_clean = df[[pre_col, post_col, group_col]].dropna()
    
    if len(df_clean) < 5:
        return {'error': 'Insufficient paired data'}
    
    pre_values = df_clean[pre_col]
    post_values = df_clean[post_col]
    
    # Basic stats
    pre_mean = pre_values.mean()
    post_mean = post_values.mean()
    change = post_mean - pre_mean
    change_pct = (change / pre_mean) * 100 if pre_mean != 0 else 0
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(pre_values, post_values)
    significant = bool(p_value < 0.05)
    
    # Effect size (Cohen's d for paired samples)
    diff = post_values - pre_values
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    effect_size = 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    
    # Individual changes
    improved = len(diff[diff > 0])
    declined = len(diff[diff < 0])
    unchanged = len(diff[diff == 0])
    
    result = {
        'n_paired': len(df_clean),
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
        'direction': 'improved' if change > 0 else 'declined' if change < 0 else 'no change'
    }
    
    # Group-level analysis if available
    if group_col:
        group_changes = []
        for grp in df_clean[group_col].unique():
            grp_data = df_clean[df_clean[group_col] == grp]
            grp_pre = grp_data[pre_col].mean()
            grp_post = grp_data[post_col].mean()
            grp_change = grp_post - grp_pre
            
            group_changes.append({
                'group': _to_native(grp),
                'pre_mean': _to_native(grp_pre),
                'post_mean': _to_native(grp_post),
                'change': _to_native(grp_change),
                'n': len(grp_data)
            })
        
        group_changes = sorted(group_changes, key=lambda x: x['change'], reverse=True)
        result['group_changes'] = group_changes
        result['best_improvement_group'] = group_changes[0] if group_changes else None
    
    return result


def create_prepost_chart(prepost_data: Dict, df: pd.DataFrame, pre_col: str, post_col: str) -> str:
    if prepost_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Pre vs Post comparison
    ax1 = axes[0]
    means = [prepost_data['pre_mean'], prepost_data['post_mean']]
    labels = ['Pre', 'Post']
    colors = ['#6b7280', '#10b981' if prepost_data['change'] > 0 else '#ef4444']
    bars = ax1.bar(labels, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Pre vs Post Comparison', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', ha='center')
    
    # Chart 2: Individual changes (spaghetti plot)
    ax2 = axes[1]
    df_clean = df[[pre_col, post_col]].dropna()
    for _, row in df_clean.sample(min(50, len(df_clean))).iterrows():
        color = '#10b981' if row[post_col] > row[pre_col] else '#ef4444' if row[post_col] < row[pre_col] else '#9ca3af'
        ax2.plot([0, 1], [row[pre_col], row[post_col]], color=color, alpha=0.3)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Pre', 'Post'])
    ax2.set_ylabel('Score')
    ax2.set_title('Individual Changes', fontsize=11, fontweight='bold')
    
    # Chart 3: Change distribution
    ax3 = axes[2]
    changes = df_clean[post_col] - df_clean[pre_col]
    colors_hist = ['#10b981' if c > 0 else '#ef4444' if c < 0 else '#9ca3af' for c in changes]
    ax3.hist(changes, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax3.axvline(x=changes.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {changes.mean():.2f}')
    ax3.set_xlabel('Change (Post - Pre)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Change Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(agree: Dict, gap: Optional[Dict], bg: Optional[Dict], factor: Dict, prepost: Optional[Dict]) -> Dict:
    report = {}
    
    report['step1_agreement'] = {
        'title': '1. Agreement Distribution',
        'question': 'What is the overall agreement level?',
        'finding': f"Overall agreement: {agree['overall_agree_pct']:.1f}%, Net agreement: {agree['overall_net_agreement']:.1f}%",
        'detail': f"Analysis of {agree['n_items']} perception items shows overall agreement rate of {agree['overall_agree_pct']:.1f}%. "
                 f"Highest agreement: {agree['highest_agreement']['item']} ({agree['highest_agreement']['agree_pct']:.1f}%). "
                 f"Lowest: {agree['lowest_agreement']['item']} ({agree['lowest_agreement']['agree_pct']:.1f}%)."
    }
    
    if gap and not gap.get('error'):
        report['step2_gap'] = {
            'title': '2. Group Perception Gap',
            'question': 'Are there perception differences between groups?',
            'finding': f"{gap['n_significant']}/{len(gap['item_gaps'])} items show significant group differences",
            'detail': f"Among {gap['n_groups']} groups, {gap['n_significant']} items show statistically significant differences. "
                     f"Largest gap: {gap['largest_gap_item']['item']} ({gap['largest_gap_item']['gap']:.2f}pt between {gap['largest_gap_item']['best_group']} and {gap['largest_gap_item']['worst_group']})."
        }
    else:
        report['step2_gap'] = {
            'title': '2. Group Perception Gap',
            'question': 'Are there perception differences between groups?',
            'finding': 'Group analysis not performed',
            'detail': 'No group column specified for comparison analysis.'
        }
    
    if bg and not bg.get('error'):
        report['step3_background'] = {
            'title': '3. Background-Perception Relationship',
            'question': 'Which background factors influence perception?',
            'finding': f"{bg['n_significant']}/{bg['n_variables']} background variables significantly related",
            'detail': f"Analysis of {bg['n_variables']} background variables found {bg['n_significant']} with significant relationship to perception. "
                     + (f"Key factors: {', '.join([v['variable'] for v in bg['significant_variables'][:3]])}." if bg['significant_variables'] else "No significant factors found.")
        }
    else:
        report['step3_background'] = {
            'title': '3. Background-Perception Relationship',
            'question': 'Which background factors influence perception?',
            'finding': 'Background analysis not performed',
            'detail': 'No background columns specified for analysis.'
        }
    
    if factor and not factor.get('error'):
        report['step4_factor'] = {
            'title': '4. Perception Structure',
            'question': 'What are the underlying perception dimensions?',
            'finding': f"{factor['n_factors']} factors identified, explaining {factor['total_variance_explained']:.1f}% variance",
            'detail': f"Factor analysis identified {factor['n_factors']} underlying dimensions. "
                     + ' '.join([f"{f['factor']}: {f['n_items']} items ({f['variance_explained']:.1f}% var)." for f in factor['factors'][:3]])
        }
    else:
        report['step4_factor'] = {
            'title': '4. Perception Structure',
            'question': 'What are the underlying perception dimensions?',
            'finding': 'Factor analysis unavailable',
            'detail': factor.get('error', 'Could not perform factor analysis.')
        }
    
    if prepost and not prepost.get('error'):
        report['step5_prepost'] = {
            'title': '5. Pre-Post Campaign Change',
            'question': 'Did the campaign change perceptions?',
            'finding': f"Change: {prepost['change']:+.2f} ({prepost['change_pct']:+.1f}%), {prepost['effect_size']} effect",
            'detail': f"Pre-post comparison (n={prepost['n_paired']}) shows {prepost['direction']} from {prepost['pre_mean']:.2f} to {prepost['post_mean']:.2f}. "
                     f"{'Statistically significant' if prepost['significant'] else 'Not significant'} (p={prepost['p_value']:.4f}). "
                     f"{prepost['improved_pct']:.1f}% of respondents improved."
        }
    else:
        report['step5_prepost'] = {
            'title': '5. Pre-Post Campaign Change',
            'question': 'Did the campaign change perceptions?',
            'finding': 'Pre-post analysis not performed',
            'detail': prepost.get('error', 'No pre/post columns specified.')
        }
    
    return report


def generate_insights(agree: Dict, gap: Optional[Dict], bg: Optional[Dict], factor: Dict, prepost: Optional[Dict]) -> List[Dict]:
    insights = []
    
    # Agreement insights
    if agree['overall_net_agreement'] < 0:
        insights.append({
            'title': 'Negative Net Agreement',
            'description': f"More disagreement than agreement overall ({agree['overall_net_agreement']:.1f}%).",
            'status': 'warning'
        })
    elif agree['overall_agree_pct'] > 70:
        insights.append({
            'title': 'Strong Overall Agreement',
            'description': f"{agree['overall_agree_pct']:.1f}% agreement rate across items.",
            'status': 'positive'
        })
    
    # Gap insights
    if gap and gap.get('n_significant', 0) > 0:
        insights.append({
            'title': 'Group Perception Gaps Found',
            'description': f"{gap['n_significant']} items show significant differences between groups.",
            'status': 'warning'
        })
    
    # Pre-post insights
    if prepost and not prepost.get('error'):
        if prepost['significant'] and prepost['change'] > 0:
            insights.append({
                'title': 'Campaign Effective',
                'description': f"Significant improvement of {prepost['change']:+.2f}pts ({prepost['effect_size']} effect).",
                'status': 'positive'
            })
        elif prepost['significant'] and prepost['change'] < 0:
            insights.append({
                'title': 'Perception Declined',
                'description': f"Significant decline of {prepost['change']:.2f}pts post-campaign.",
                'status': 'warning'
            })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/perception-analysis")
async def analyze_perception(request: PerceptionRequest):
    try:
        df = pd.DataFrame(request.data)
        perception_cols = request.perception_cols
        group_col = request.group_col
        background_cols = request.background_cols or []
        pre_col = request.pre_col
        post_col = request.post_col
        
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 data points")
        
        # Convert perception columns to numeric
        for col in perception_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Agreement Distribution
        agree = analyze_agreement_distribution(df, perception_cols)
        results['agreement'] = agree
        visualizations['agreement_chart'] = create_agreement_chart(agree)
        
        # Step 2: Group Gap
        gap = None
        if group_col and group_col in df.columns:
            gap = analyze_group_gap(df, perception_cols, group_col)
            results['group_gap'] = gap
            visualizations['gap_chart'] = create_group_gap_chart(gap, df, perception_cols, group_col)
        
        # Step 3: Background Relationship
        bg = None
        if background_cols:
            valid_bg = [c for c in background_cols if c in df.columns]
            if valid_bg:
                bg = analyze_background_relationship(df, perception_cols, valid_bg)
                results['background'] = bg
                visualizations['background_chart'] = create_background_chart(bg, df, perception_cols, valid_bg)
        
        # Step 4: Factor Analysis
        factor = analyze_factor_structure(df, perception_cols)
        results['factor'] = factor
        if not factor.get('error'):
            visualizations['factor_chart'] = create_factor_chart(factor)
        
        # Step 5: Pre-Post Change
        prepost = None
        if pre_col and post_col and pre_col in df.columns and post_col in df.columns:
            df[pre_col] = pd.to_numeric(df[pre_col], errors='coerce')
            df[post_col] = pd.to_numeric(df[post_col], errors='coerce')
            prepost = analyze_prepost_change(df, pre_col, post_col, group_col)
            results['prepost'] = prepost
            if not prepost.get('error'):
                visualizations['prepost_chart'] = create_prepost_chart(prepost, df, pre_col, post_col)
        
        report = generate_report(agree, gap, bg, factor, prepost)
        insights = generate_insights(agree, gap, bg, factor, prepost)
        
        summary = {
            'n_responses': len(df),
            'n_items': len(perception_cols),
            'overall_agree_pct': agree['overall_agree_pct'],
            'n_factors': factor.get('n_factors', 0) if not factor.get('error') else 0,
            'campaign_change': prepost.get('change') if prepost and not prepost.get('error') else None
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
