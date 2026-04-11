"""
Propensity Score Matching (PSM) Router for FastAPI
Match treatment and control units based on propensity scores for causal inference
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class PSMRequest(BaseModel):
    data: List[Dict[str, Any]]
    treatment_col: str  # Treatment indicator (0/1)
    outcome_col: str  # Outcome variable
    covariates: List[str]  # Variables to match on
    matching_method: str = "nearest"  # nearest, caliper, radius
    caliper: Optional[float] = 0.2  # Caliper width (in SD of propensity score)
    n_neighbors: int = 1  # Number of matches per treated unit
    with_replacement: bool = False  # Allow control units to be matched multiple times


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


def estimate_propensity_scores(df: pd.DataFrame, treatment_col: str, 
                                covariates: List[str]) -> Dict[str, Any]:
    """Estimate propensity scores using logistic regression"""
    X = df[covariates].copy()
    y = df[treatment_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # Get propensity scores
    propensity_scores = model.predict_proba(X_scaled)[:, 1]
    
    # Model diagnostics
    coef_dict = {}
    for i, cov in enumerate(covariates):
        coef_dict[cov] = {
            'coefficient': _to_native_type(model.coef_[0][i]),
            'odds_ratio': _to_native_type(np.exp(model.coef_[0][i]))
        }
    
    return {
        'propensity_scores': propensity_scores,
        'model_coefficients': coef_dict,
        'intercept': _to_native_type(model.intercept_[0]),
        'model_accuracy': _to_native_type(model.score(X_scaled, y))
    }


def perform_nearest_neighbor_matching(df: pd.DataFrame, propensity_scores: np.ndarray,
                                       treatment_col: str, n_neighbors: int = 1,
                                       caliper: float = None, with_replacement: bool = False) -> Dict[str, Any]:
    """Perform nearest neighbor matching on propensity scores"""
    
    df = df.copy()
    df['propensity_score'] = propensity_scores
    
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    if len(treated) == 0 or len(control) == 0:
        return {'error': 'Need both treated and control units'}
    
    # Calculate caliper if specified (in SD of propensity score)
    if caliper:
        caliper_width = caliper * propensity_scores.std()
    else:
        caliper_width = np.inf
    
    # Nearest neighbor matching
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(control)), metric='euclidean')
    nn.fit(control[['propensity_score']].values)
    
    matched_pairs = []
    matched_control_indices = set()
    
    for idx, row in treated.iterrows():
        distances, indices = nn.kneighbors([[row['propensity_score']]])
        
        for i, (dist, ctrl_idx) in enumerate(zip(distances[0], indices[0])):
            ctrl_original_idx = control.index[ctrl_idx]
            
            # Check caliper
            if dist > caliper_width:
                continue
            
            # Check replacement
            if not with_replacement and ctrl_original_idx in matched_control_indices:
                continue
            
            matched_pairs.append({
                'treated_idx': idx,
                'control_idx': ctrl_original_idx,
                'treated_ps': _to_native_type(row['propensity_score']),
                'control_ps': _to_native_type(control.loc[ctrl_original_idx, 'propensity_score']),
                'distance': _to_native_type(dist)
            })
            
            matched_control_indices.add(ctrl_original_idx)
            break  # One match per treated
    
    # Get matched sample
    matched_treated_idx = [p['treated_idx'] for p in matched_pairs]
    matched_control_idx = [p['control_idx'] for p in matched_pairs]
    
    matched_treated = df.loc[matched_treated_idx].copy()
    matched_control = df.loc[matched_control_idx].copy()
    
    return {
        'matched_pairs': matched_pairs,
        'n_treated': len(treated),
        'n_control': len(control),
        'n_matched': len(matched_pairs),
        'match_rate': _to_native_type(len(matched_pairs) / len(treated) * 100),
        'matched_treated_idx': matched_treated_idx,
        'matched_control_idx': matched_control_idx,
        'caliper_used': _to_native_type(caliper_width) if caliper else None
    }


def calculate_balance_statistics(df: pd.DataFrame, treatment_col: str, 
                                  covariates: List[str], matched_idx: Dict = None) -> Dict[str, Any]:
    """Calculate covariate balance before and after matching"""
    
    balance_stats = []
    
    for cov in covariates:
        # Before matching
        treated_before = df[df[treatment_col] == 1][cov]
        control_before = df[df[treatment_col] == 0][cov]
        
        mean_treated_before = treated_before.mean()
        mean_control_before = control_before.mean()
        
        # Standardized difference before
        pooled_std = np.sqrt((treated_before.var() + control_before.var()) / 2)
        std_diff_before = (mean_treated_before - mean_control_before) / pooled_std if pooled_std > 0 else 0
        
        stat = {
            'covariate': cov,
            'mean_treated_before': _to_native_type(mean_treated_before),
            'mean_control_before': _to_native_type(mean_control_before),
            'std_diff_before': _to_native_type(std_diff_before),
            'balance_before': bool(abs(std_diff_before) < 0.1)
        }
        
        # After matching (if matched indices provided)
        if matched_idx:
            treated_after = df.loc[matched_idx['treated']][cov]
            control_after = df.loc[matched_idx['control']][cov]
            
            mean_treated_after = treated_after.mean()
            mean_control_after = control_after.mean()
            
            pooled_std_after = np.sqrt((treated_after.var() + control_after.var()) / 2)
            std_diff_after = (mean_treated_after - mean_control_after) / pooled_std_after if pooled_std_after > 0 else 0
            
            stat.update({
                'mean_treated_after': _to_native_type(mean_treated_after),
                'mean_control_after': _to_native_type(mean_control_after),
                'std_diff_after': _to_native_type(std_diff_after),
                'balance_after': bool(abs(std_diff_after) < 0.1),
                'improvement': _to_native_type(abs(std_diff_before) - abs(std_diff_after))
            })
        
        balance_stats.append(stat)
    
    return {
        'balance_statistics': balance_stats,
        'overall_balance_before': sum(1 for s in balance_stats if s.get('balance_before', False)) / len(balance_stats) * 100,
        'overall_balance_after': sum(1 for s in balance_stats if s.get('balance_after', False)) / len(balance_stats) * 100 if matched_idx else None
    }


def estimate_treatment_effect(df: pd.DataFrame, treatment_col: str, outcome_col: str,
                               matched_idx: Dict = None) -> Dict[str, Any]:
    """Estimate Average Treatment Effect on the Treated (ATT)"""
    
    results = {}
    
    # Before matching (naive estimate)
    treated_outcome = df[df[treatment_col] == 1][outcome_col]
    control_outcome = df[df[treatment_col] == 0][outcome_col]
    
    naive_ate = treated_outcome.mean() - control_outcome.mean()
    t_stat, p_value = stats.ttest_ind(treated_outcome, control_outcome)
    
    results['naive_estimate'] = {
        'ate': _to_native_type(naive_ate),
        'treated_mean': _to_native_type(treated_outcome.mean()),
        'control_mean': _to_native_type(control_outcome.mean()),
        't_statistic': _to_native_type(t_stat),
        'p_value': _to_native_type(p_value),
        'significant': bool(p_value < 0.05)
    }
    
    # After matching (ATT)
    if matched_idx:
        matched_treated = df.loc[matched_idx['treated']][outcome_col]
        matched_control = df.loc[matched_idx['control']][outcome_col]
        
        att = matched_treated.mean() - matched_control.mean()
        
        # Paired t-test for matched samples
        t_stat_matched, p_value_matched = stats.ttest_rel(
            matched_treated.values, matched_control.values
        )
        
        # Standard error
        diff = matched_treated.values - matched_control.values
        se = diff.std() / np.sqrt(len(diff))
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se
        
        results['matched_estimate'] = {
            'att': _to_native_type(att),
            'treated_mean': _to_native_type(matched_treated.mean()),
            'control_mean': _to_native_type(matched_control.mean()),
            'std_error': _to_native_type(se),
            'ci_lower': _to_native_type(ci_lower),
            'ci_upper': _to_native_type(ci_upper),
            't_statistic': _to_native_type(t_stat_matched),
            'p_value': _to_native_type(p_value_matched),
            'significant': bool(p_value_matched < 0.05),
            'n_pairs': len(matched_treated)
        }
    
    return results


def generate_propensity_distribution_plot(df: pd.DataFrame, propensity_scores: np.ndarray,
                                           treatment_col: str) -> str:
    """Generate propensity score distribution plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_temp = df.copy()
    df_temp['propensity_score'] = propensity_scores
    
    treated = df_temp[df_temp[treatment_col] == 1]['propensity_score']
    control = df_temp[df_temp[treatment_col] == 0]['propensity_score']
    
    # Histogram
    axes[0].hist(control, bins=30, alpha=0.6, label='Control', color='#3b82f6', density=True)
    axes[0].hist(treated, bins=30, alpha=0.6, label='Treatment', color='#ef4444', density=True)
    axes[0].set_xlabel('Propensity Score', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Propensity Score Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Box plot
    data_for_box = [control, treated]
    bp = axes[1].boxplot(data_for_box, labels=['Control', 'Treatment'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3b82f6')
    bp['boxes'][1].set_facecolor('#ef4444')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    axes[1].set_ylabel('Propensity Score', fontsize=11)
    axes[1].set_title('Propensity Score by Group', fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_balance_plot(balance_stats: List[Dict]) -> str:
    """Generate covariate balance plot (Love plot)"""
    fig, ax = plt.subplots(figsize=(12, max(6, len(balance_stats) * 0.5)))
    
    covariates = [s['covariate'] for s in balance_stats]
    std_diff_before = [s['std_diff_before'] for s in balance_stats]
    std_diff_after = [s.get('std_diff_after', s['std_diff_before']) for s in balance_stats]
    
    y_pos = np.arange(len(covariates))
    
    ax.scatter(std_diff_before, y_pos, s=100, marker='o', color='#ef4444', 
               label='Before Matching', zorder=3)
    ax.scatter(std_diff_after, y_pos, s=100, marker='s', color='#22c55e', 
               label='After Matching', zorder=3)
    
    # Connect points
    for i in range(len(covariates)):
        ax.plot([std_diff_before[i], std_diff_after[i]], [y_pos[i], y_pos[i]], 
                'gray', alpha=0.5, linewidth=1)
    
    # Reference lines
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.7, label='|d| = 0.1')
    ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.7)
    ax.axvspan(-0.1, 0.1, alpha=0.1, color='green')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel('Standardized Difference', fontsize=11)
    ax.set_title('Covariate Balance (Love Plot)', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_matching_plot(df: pd.DataFrame, propensity_scores: np.ndarray,
                           treatment_col: str, matched_pairs: List[Dict]) -> str:
    """Generate matching visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_temp = df.copy()
    df_temp['propensity_score'] = propensity_scores
    
    treated = df_temp[df_temp[treatment_col] == 1]
    control = df_temp[df_temp[treatment_col] == 0]
    
    # Jitter for visualization
    np.random.seed(42)
    treated_x = np.ones(len(treated)) + np.random.normal(0, 0.05, len(treated))
    control_x = np.zeros(len(control)) + np.random.normal(0, 0.05, len(control))
    
    # Plot all points
    ax.scatter(control_x, control['propensity_score'], alpha=0.4, 
               color='#3b82f6', s=50, label='Control (unmatched)')
    ax.scatter(treated_x, treated['propensity_score'], alpha=0.4, 
               color='#ef4444', s=50, label='Treatment (unmatched)')
    
    # Highlight matched pairs
    matched_treated_idx = [p['treated_idx'] for p in matched_pairs]
    matched_control_idx = [p['control_idx'] for p in matched_pairs]
    
    matched_treated = df_temp.loc[matched_treated_idx]
    matched_control = df_temp.loc[matched_control_idx]
    
    ax.scatter(np.ones(len(matched_treated)) * 1.15, matched_treated['propensity_score'],
               color='#ef4444', s=80, edgecolors='black', linewidth=1.5, 
               label='Treatment (matched)', zorder=5)
    ax.scatter(np.ones(len(matched_control)) * -0.15, matched_control['propensity_score'],
               color='#3b82f6', s=80, edgecolors='black', linewidth=1.5,
               label='Control (matched)', zorder=5)
    
    # Draw matching lines
    for pair in matched_pairs[:50]:  # Limit lines for clarity
        ax.plot([-0.15, 1.15], 
                [pair['control_ps'], pair['treated_ps']],
                'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Treatment'])
    ax.set_ylabel('Propensity Score', fontsize=11)
    ax.set_title('Propensity Score Matching Visualization', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_outcome_comparison_plot(df: pd.DataFrame, treatment_col: str, 
                                      outcome_col: str, matched_idx: Dict) -> str:
    """Generate outcome comparison before and after matching"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before matching
    treated_before = df[df[treatment_col] == 1][outcome_col]
    control_before = df[df[treatment_col] == 0][outcome_col]
    
    data_before = [control_before, treated_before]
    bp1 = axes[0].boxplot(data_before, labels=['Control', 'Treatment'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#3b82f6')
    bp1['boxes'][1].set_facecolor('#ef4444')
    for box in bp1['boxes']:
        box.set_alpha(0.6)
    axes[0].set_ylabel(outcome_col, fontsize=11)
    axes[0].set_title('Before Matching', fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Add mean annotations
    axes[0].axhline(y=control_before.mean(), color='#3b82f6', linestyle='--', alpha=0.7)
    axes[0].axhline(y=treated_before.mean(), color='#ef4444', linestyle='--', alpha=0.7)
    
    # After matching
    treated_after = df.loc[matched_idx['treated']][outcome_col]
    control_after = df.loc[matched_idx['control']][outcome_col]
    
    data_after = [control_after, treated_after]
    bp2 = axes[1].boxplot(data_after, labels=['Control', 'Treatment'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#3b82f6')
    bp2['boxes'][1].set_facecolor('#ef4444')
    for box in bp2['boxes']:
        box.set_alpha(0.6)
    axes[1].set_ylabel(outcome_col, fontsize=11)
    axes[1].set_title('After Matching', fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    axes[1].axhline(y=control_after.mean(), color='#3b82f6', linestyle='--', alpha=0.7)
    axes[1].axhline(y=treated_after.mean(), color='#ef4444', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(ps_results: Dict, matching_results: Dict, 
                            balance_results: Dict, effect_results: Dict) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    key_insights = []
    
    # Matching success
    match_rate = matching_results['match_rate']
    key_insights.append({
        'title': 'Matching Success',
        'description': f"{matching_results['n_matched']} of {matching_results['n_treated']} treated units matched ({match_rate:.1f}%).",
        'status': 'positive' if match_rate >= 80 else 'warning' if match_rate >= 50 else 'negative'
    })
    
    # Balance improvement
    balance_before = balance_results['overall_balance_before'] or 0.0
    balance_after_raw = balance_results.get('overall_balance_after')
    balance_after = balance_after_raw if balance_after_raw is not None else balance_before
    key_insights.append({
        'title': 'Covariate Balance',
        'description': f"Balance improved from {balance_before:.0f}% to {balance_after:.0f}% of covariates meeting threshold (|d| < 0.1).",
        'status': 'positive' if balance_after >= 80 else 'warning'
    })
    
    # Treatment effect
    if 'matched_estimate' in effect_results:
        att = effect_results['matched_estimate']['att']
        p_val = effect_results['matched_estimate']['p_value']
        sig = effect_results['matched_estimate']['significant']
        
        key_insights.append({
            'title': 'Treatment Effect (ATT)',
            'description': f"Average Treatment Effect on Treated = {att:.3f} (p = {p_val:.4f}). {'Statistically significant.' if sig else 'Not statistically significant.'}",
            'status': 'positive' if sig else 'neutral'
        })
        
        # Compare with naive
        naive_ate = effect_results['naive_estimate']['ate']
        bias_reduction = abs(naive_ate - att)
        key_insights.append({
            'title': 'Bias Reduction',
            'description': f"Naive estimate: {naive_ate:.3f}, Matched estimate: {att:.3f}. Difference: {bias_reduction:.3f}",
            'status': 'neutral'
        })
    
    return {
        'key_insights': key_insights,
        'recommendation': 'PSM successfully reduced selection bias.' if (balance_after is not None and balance_after >= 70) else 'Consider additional matching methods or covariates.'
    }


@router.post("/propensity-score-matching")
async def run_psm_analysis(request: PSMRequest) -> Dict[str, Any]:
    """
    Perform Propensity Score Matching (PSM) Analysis.
    
    Estimates causal treatment effects by matching treated and control
    units based on their propensity scores.
    """
    try:
        data = request.data
        treatment_col = request.treatment_col
        outcome_col = request.outcome_col
        covariates = request.covariates
        matching_method = request.matching_method
        caliper = request.caliper
        n_neighbors = request.n_neighbors
        with_replacement = request.with_replacement
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        all_cols = [treatment_col, outcome_col] + covariates
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")
        
        # Convert to numeric
        df[outcome_col] = pd.to_numeric(df[outcome_col], errors='coerce')
        df[treatment_col] = pd.to_numeric(df[treatment_col], errors='coerce')
        for cov in covariates:
            df[cov] = pd.to_numeric(df[cov], errors='coerce')
        
        df = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        
        # Validate treatment is binary
        unique_treatment = df[treatment_col].unique()
        if len(unique_treatment) != 2 or not all(v in [0, 1] for v in unique_treatment):
            raise HTTPException(status_code=400, detail="Treatment must be binary (0/1).")
        
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="At least 20 observations required.")
        
        # Estimate propensity scores
        ps_results = estimate_propensity_scores(df, treatment_col, covariates)
        propensity_scores = ps_results['propensity_scores']
        
        # Perform matching
        matching_results = perform_nearest_neighbor_matching(
            df, propensity_scores, treatment_col, 
            n_neighbors=n_neighbors,
            caliper=caliper if matching_method in ['caliper', 'nearest'] else None,
            with_replacement=with_replacement
        )
        
        # Prepare matched indices
        matched_idx = {
            'treated': matching_results['matched_treated_idx'],
            'control': matching_results['matched_control_idx']
        }
        
        # Calculate balance
        balance_results = calculate_balance_statistics(df, treatment_col, covariates, matched_idx)
        
        # Estimate treatment effect
        effect_results = estimate_treatment_effect(df, treatment_col, outcome_col, matched_idx)
        
        # Generate visualizations
        ps_distribution = generate_propensity_distribution_plot(df, propensity_scores, treatment_col)
        balance_plot = generate_balance_plot(balance_results['balance_statistics'])
        matching_plot = generate_matching_plot(df, propensity_scores, treatment_col, matching_results['matched_pairs'])
        outcome_plot = generate_outcome_comparison_plot(df, treatment_col, outcome_col, matched_idx)
        
        # Generate interpretation
        interpretation = generate_interpretation(ps_results, matching_results, balance_results, effect_results)
        
        # Descriptive stats
        descriptive_stats = {
            'n_total': len(df),
            'n_treated': int((df[treatment_col] == 1).sum()),
            'n_control': int((df[treatment_col] == 0).sum()),
            'ps_mean': _to_native_type(propensity_scores.mean()),
            'ps_std': _to_native_type(propensity_scores.std()),
            'ps_min': _to_native_type(propensity_scores.min()),
            'ps_max': _to_native_type(propensity_scores.max())
        }
        
        return {
            'propensity_score_model': {
                'coefficients': ps_results['model_coefficients'],
                'intercept': ps_results['intercept'],
                'accuracy': ps_results['model_accuracy']
            },
            'matching_results': {
                'n_matched': matching_results['n_matched'],
                'match_rate': matching_results['match_rate'],
                'method': matching_method,
                'caliper': matching_results['caliper_used'],
                'with_replacement': with_replacement
            },
            'balance_statistics': balance_results['balance_statistics'],
            'balance_summary': {
                'before': balance_results['overall_balance_before'],
                'after': balance_results['overall_balance_after']
            },
            'treatment_effects': effect_results,
            'descriptive_stats': descriptive_stats,
            'ps_distribution_plot': ps_distribution,
            'balance_plot': balance_plot,
            'matching_plot': matching_plot,
            'outcome_plot': outcome_plot,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PSM analysis failed: {str(e)}")
