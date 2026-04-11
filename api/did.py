"""
Difference-in-Differences (DiD) Analysis Router for FastAPI
Estimate causal effects by comparing treatment and control groups before and after intervention
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class DiDRequest(BaseModel):
    data: List[Dict[str, Any]]
    outcome_col: str  # Outcome/dependent variable
    treatment_col: str  # Treatment group indicator (0/1 or group names)
    time_col: str  # Time period indicator (0/1 or pre/post)
    covariates: Optional[List[str]] = None  # Control variables
    cluster_col: Optional[str] = None  # Cluster variable for robust SE


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


def calculate_did_manual(df: pd.DataFrame, outcome_col: str, 
                         treatment_col: str, time_col: str) -> Dict[str, Any]:
    """Calculate DiD estimate manually using 2x2 means"""
    
    # Get means for each group-time combination
    means = df.groupby([treatment_col, time_col])[outcome_col].mean()
    counts = df.groupby([treatment_col, time_col])[outcome_col].count()
    stds = df.groupby([treatment_col, time_col])[outcome_col].std()
    
    # Get unique values
    treatment_vals = sorted(df[treatment_col].unique())
    time_vals = sorted(df[time_col].unique())
    
    if len(treatment_vals) != 2 or len(time_vals) != 2:
        return None
    
    control, treated = treatment_vals[0], treatment_vals[1]
    pre, post = time_vals[0], time_vals[1]
    
    # Calculate means
    y_control_pre = means.get((control, pre), 0)
    y_control_post = means.get((control, post), 0)
    y_treated_pre = means.get((treated, pre), 0)
    y_treated_post = means.get((treated, post), 0)
    
    # DiD calculation
    diff_treated = y_treated_post - y_treated_pre
    diff_control = y_control_post - y_control_pre
    did_estimate = diff_treated - diff_control
    
    # Sample sizes
    n_control_pre = counts.get((control, pre), 1)
    n_control_post = counts.get((control, post), 1)
    n_treated_pre = counts.get((treated, pre), 1)
    n_treated_post = counts.get((treated, post), 1)
    
    # Standard deviations for each cell
    std_control_pre = stds.get((control, pre), 0)
    std_control_post = stds.get((control, post), 0)
    std_treated_pre = stds.get((treated, pre), 0)
    std_treated_post = stds.get((treated, post), 0)
    
    # Variance for each cell mean
    var_control_pre = (std_control_pre ** 2) / max(n_control_pre, 1)
    var_control_post = (std_control_post ** 2) / max(n_control_post, 1)
    var_treated_pre = (std_treated_pre ** 2) / max(n_treated_pre, 1)
    var_treated_post = (std_treated_post ** 2) / max(n_treated_post, 1)
    
    # Standard error of DiD estimate
    se_did = np.sqrt(var_control_pre + var_control_post + var_treated_pre + var_treated_post)
    
    # Degrees of freedom (Welch-Satterthwaite approximation simplified)
    total_n = n_control_pre + n_control_post + n_treated_pre + n_treated_post
    df_approx = max(total_n - 4, 1)
    
    # T-statistic and p-value
    t_stat = None
    p_value = None
    ci_lower = None
    ci_upper = None
    significant = None
    
    if se_did > 0:
        t_stat = did_estimate / se_did
        # Two-tailed p-value
        p_value = 2 * stats.t.sf(abs(t_stat), df_approx)
        # 95% CI
        t_critical = stats.t.ppf(0.975, df_approx)
        ci_lower = did_estimate - t_critical * se_did
        ci_upper = did_estimate + t_critical * se_did
        significant = bool(p_value < 0.05)
    
    return {
        'did_estimate': _to_native_type(did_estimate),
        'std_error': _to_native_type(se_did),
        't_statistic': _to_native_type(t_stat),
        'p_value': _to_native_type(p_value),
        'ci_lower': _to_native_type(ci_lower),
        'ci_upper': _to_native_type(ci_upper),
        'significant': significant,
        'means': {
            'control_pre': _to_native_type(y_control_pre),
            'control_post': _to_native_type(y_control_post),
            'treated_pre': _to_native_type(y_treated_pre),
            'treated_post': _to_native_type(y_treated_post)
        },
        'stds': {
            'control_pre': _to_native_type(std_control_pre),
            'control_post': _to_native_type(std_control_post),
            'treated_pre': _to_native_type(std_treated_pre),
            'treated_post': _to_native_type(std_treated_post)
        },
        'differences': {
            'treated_diff': _to_native_type(diff_treated),
            'control_diff': _to_native_type(diff_control)
        },
        'sample_sizes': {
            'control_pre': _to_native_type(n_control_pre),
            'control_post': _to_native_type(n_control_post),
            'treated_pre': _to_native_type(n_treated_pre),
            'treated_post': _to_native_type(n_treated_post)
        },
        'df': _to_native_type(df_approx)
    }


def run_did_regression(df: pd.DataFrame, outcome_col: str, treatment_col: str,
                       time_col: str, covariates: List[str] = None,
                       cluster_col: str = None) -> Dict[str, Any]:
    """Run DiD regression: Y = β0 + β1*Treatment + β2*Post + β3*Treatment*Post + ε"""
    
    # Create interaction term
    df = df.copy()
    df['treatment_numeric'] = pd.to_numeric(df[treatment_col].astype('category').cat.codes, errors='coerce')
    df['time_numeric'] = pd.to_numeric(df[time_col].astype('category').cat.codes, errors='coerce')
    df['interaction'] = df['treatment_numeric'] * df['time_numeric']
    
    # Build formula
    formula = f"{outcome_col} ~ treatment_numeric + time_numeric + interaction"
    if covariates:
        valid_covariates = [c for c in covariates if c in df.columns and c not in [outcome_col, treatment_col, time_col]]
        if valid_covariates:
            formula += " + " + " + ".join(valid_covariates)
    
    try:
        # Fit OLS model
        model = smf.ols(formula, data=df).fit()
        
        # Get robust standard errors if clustering
        if cluster_col and cluster_col in df.columns:
            model = smf.ols(formula, data=df).fit(cov_type='cluster', 
                                                   cov_kwds={'groups': df[cluster_col]})
        
        # Extract coefficients
        coefficients = []
        param_names = {'Intercept': 'Intercept (β0)', 
                       'treatment_numeric': 'Treatment (β1)',
                       'time_numeric': 'Post Period (β2)', 
                       'interaction': 'DiD Effect (β3)'}
        
        for param in model.params.index:
            display_name = param_names.get(param, param)
            coefficients.append({
                'term': display_name,
                'estimate': _to_native_type(model.params[param]),
                'std_error': _to_native_type(model.bse[param]),
                't_value': _to_native_type(model.tvalues[param]),
                'p_value': _to_native_type(model.pvalues[param]),
                'ci_lower': _to_native_type(model.conf_int().loc[param, 0]),
                'ci_upper': _to_native_type(model.conf_int().loc[param, 1]),
                'significant': bool(model.pvalues[param] < 0.05)
            })
        
        # DiD estimate is the interaction coefficient
        did_coef = model.params.get('interaction', 0)
        did_se = model.bse.get('interaction', 0)
        did_pval = model.pvalues.get('interaction', 1)
        did_ci = model.conf_int().loc['interaction'] if 'interaction' in model.conf_int().index else [None, None]
        
        return {
            'did_estimate': _to_native_type(did_coef),
            'std_error': _to_native_type(did_se),
            'p_value': _to_native_type(did_pval),
            'ci_lower': _to_native_type(did_ci[0]),
            'ci_upper': _to_native_type(did_ci[1]),
            'significant': bool(did_pval < 0.05),
            'coefficients': coefficients,
            'r_squared': _to_native_type(model.rsquared),
            'adj_r_squared': _to_native_type(model.rsquared_adj),
            'f_statistic': _to_native_type(model.fvalue),
            'f_pvalue': _to_native_type(model.f_pvalue),
            'n_obs': _to_native_type(model.nobs),
            'aic': _to_native_type(model.aic),
            'bic': _to_native_type(model.bic)
        }
    except Exception as e:
        return {'error': str(e)}


def generate_parallel_trends_plot(df: pd.DataFrame, outcome_col: str,
                                   treatment_col: str, time_col: str) -> str:
    """Generate parallel trends visualization"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get group means
    means = df.groupby([treatment_col, time_col])[outcome_col].agg(['mean', 'std', 'count'])
    means['se'] = means['std'] / np.sqrt(means['count'])
    means = means.reset_index()
    
    treatment_vals = sorted(df[treatment_col].unique())
    time_vals = sorted(df[time_col].unique())
    
    colors = {'control': '#3b82f6', 'treated': '#ef4444'}
    markers = {'control': 'o', 'treated': 's'}
    
    for i, treatment in enumerate(treatment_vals):
        group_data = means[means[treatment_col] == treatment]
        label = 'Treatment' if i == 1 else 'Control'
        color = colors['treated'] if i == 1 else colors['control']
        marker = markers['treated'] if i == 1 else markers['control']
        
        ax.errorbar(range(len(time_vals)), group_data['mean'], 
                   yerr=1.96 * group_data['se'],
                   marker=marker, markersize=12, linewidth=3,
                   capsize=5, capthick=2, label=label, color=color)
    
    # Add vertical line at treatment time
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Treatment')
    
    ax.set_xticks(range(len(time_vals)))
    ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'], fontsize=12)
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel(f'Mean {outcome_col}', fontsize=12)
    ax.set_title('Difference-in-Differences: Parallel Trends', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_did_diagram(manual_results: Dict) -> str:
    """Generate DiD visual diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    means = manual_results['means']
    
    # Plot actual lines
    x = [0, 1]
    control_y = [means['control_pre'], means['control_post']]
    treated_y = [means['treated_pre'], means['treated_post']]
    
    # Control group
    ax.plot(x, control_y, 'b-o', linewidth=3, markersize=15, label='Control Group', color='#3b82f6')
    
    # Treatment group
    ax.plot(x, treated_y, 'r-s', linewidth=3, markersize=15, label='Treatment Group', color='#ef4444')
    
    # Counterfactual (what treatment would be without effect)
    counterfactual_post = means['treated_pre'] + (means['control_post'] - means['control_pre'])
    ax.plot([0, 1], [means['treated_pre'], counterfactual_post], 
            'r--', linewidth=2, alpha=0.5, label='Counterfactual')
    ax.scatter([1], [counterfactual_post], s=150, color='#ef4444', alpha=0.5, marker='s')
    
    # DiD arrow
    did = manual_results['did_estimate']
    ax.annotate('', xy=(1.05, means['treated_post']), 
                xytext=(1.05, counterfactual_post),
                arrowprops=dict(arrowstyle='<->', color='green', lw=3))
    ax.text(1.1, (means['treated_post'] + counterfactual_post) / 2, 
            f'DiD = {did:.2f}', fontsize=12, fontweight='bold', color='green', va='center')
    
    # Annotations
    ax.annotate(f"{means['control_pre']:.2f}", (0, means['control_pre']), 
                textcoords="offset points", xytext=(-30, 10), fontsize=10)
    ax.annotate(f"{means['control_post']:.2f}", (1, means['control_post']), 
                textcoords="offset points", xytext=(10, 10), fontsize=10)
    ax.annotate(f"{means['treated_pre']:.2f}", (0, means['treated_pre']), 
                textcoords="offset points", xytext=(-30, -15), fontsize=10)
    ax.annotate(f"{means['treated_post']:.2f}", (1, means['treated_post']), 
                textcoords="offset points", xytext=(10, -15), fontsize=10)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'], fontsize=12)
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Outcome', fontsize=12)
    ax.set_title('Difference-in-Differences Diagram', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Expand x-axis for annotations
    ax.set_xlim(-0.2, 1.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_group_comparison_plot(df: pd.DataFrame, outcome_col: str,
                                    treatment_col: str, time_col: str) -> str:
    """Generate box plot comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    treatment_vals = sorted(df[treatment_col].unique())
    time_vals = sorted(df[time_col].unique())
    
    # Pre-treatment comparison
    pre_data = df[df[time_col] == time_vals[0]]
    sns.boxplot(data=pre_data, x=treatment_col, y=outcome_col, ax=axes[0],
                palette=['#3b82f6', '#ef4444'])
    axes[0].set_title('Pre-Treatment Period', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Group', fontsize=11)
    axes[0].set_ylabel(outcome_col, fontsize=11)
    
    # Post-treatment comparison
    post_data = df[df[time_col] == time_vals[1]]
    sns.boxplot(data=post_data, x=treatment_col, y=outcome_col, ax=axes[1],
                palette=['#3b82f6', '#ef4444'])
    axes[1].set_title('Post-Treatment Period', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Group', fontsize=11)
    axes[1].set_ylabel(outcome_col, fontsize=11)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_distribution_plot(df: pd.DataFrame, outcome_col: str,
                                treatment_col: str, time_col: str) -> str:
    """Generate distribution comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    treatment_vals = sorted(df[treatment_col].unique())
    time_vals = sorted(df[time_col].unique())
    
    combinations = [
        (treatment_vals[0], time_vals[0], 'Control - Pre', axes[0, 0], '#3b82f6'),
        (treatment_vals[0], time_vals[1], 'Control - Post', axes[0, 1], '#60a5fa'),
        (treatment_vals[1], time_vals[0], 'Treatment - Pre', axes[1, 0], '#ef4444'),
        (treatment_vals[1], time_vals[1], 'Treatment - Post', axes[1, 1], '#f87171')
    ]
    
    for treat_val, time_val, title, ax, color in combinations:
        subset = df[(df[treatment_col] == treat_val) & (df[time_col] == time_val)]
        if len(subset) > 0:
            sns.histplot(subset[outcome_col], kde=True, ax=ax, color=color, alpha=0.7)
            ax.axvline(subset[outcome_col].mean(), color='black', linestyle='--', 
                      linewidth=2, label=f'Mean: {subset[outcome_col].mean():.2f}')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(outcome_col, fontsize=10)
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(manual_results: Dict, regression_results: Dict) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    key_insights = []
    
    did = manual_results['did_estimate']
    p_value = regression_results.get('p_value', manual_results.get('p_value'))
    significant = regression_results.get('significant', manual_results.get('significant'))
    
    # Main effect
    if significant:
        direction = 'increased' if did > 0 else 'decreased'
        key_insights.append({
            'title': 'Significant Treatment Effect',
            'description': f'The treatment {direction} the outcome by {abs(did):.3f} units (p = {p_value:.4f}). This effect is statistically significant.',
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'No Significant Effect',
            'description': f'The DiD estimate is {did:.3f} but not statistically significant (p = {p_value:.4f}).',
            'status': 'negative'
        })
    
    # Group differences
    means = manual_results['means']
    key_insights.append({
        'title': 'Group Changes',
        'description': f"Treatment group changed from {means['treated_pre']:.2f} to {means['treated_post']:.2f} (Δ = {manual_results['differences']['treated_diff']:.2f}). Control group changed from {means['control_pre']:.2f} to {means['control_post']:.2f} (Δ = {manual_results['differences']['control_diff']:.2f}).",
        'status': 'neutral'
    })
    
    # Model fit (if regression)
    if 'r_squared' in regression_results:
        r2 = regression_results['r_squared']
        key_insights.append({
            'title': 'Model Fit',
            'description': f'The regression model explains {r2*100:.1f}% of the variance in the outcome.',
            'status': 'neutral'
        })
    
    # Assumptions reminder
    key_insights.append({
        'title': 'Key Assumption',
        'description': 'DiD validity depends on the parallel trends assumption: without treatment, both groups would have followed similar trajectories.',
        'status': 'warning'
    })
    
    return {
        'key_insights': key_insights,
        'effect_size': _to_native_type(did),
        'is_significant': significant,
        'recommendation': 'The treatment had a causal effect on the outcome.' if significant else 'Cannot conclude the treatment had a causal effect.'
    }


@router.post("/difference-in-differences")
async def run_did_analysis(request: DiDRequest) -> Dict[str, Any]:
    """
    Perform Difference-in-Differences (DiD) Analysis.
    
    Estimates causal treatment effects by comparing treatment and control
    groups before and after an intervention.
    """
    try:
        data = request.data
        outcome_col = request.outcome_col
        treatment_col = request.treatment_col
        time_col = request.time_col
        covariates = request.covariates
        cluster_col = request.cluster_col
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        for col in [outcome_col, treatment_col, time_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
        
        # Convert outcome to numeric
        df[outcome_col] = pd.to_numeric(df[outcome_col], errors='coerce')
        df = df.dropna(subset=[outcome_col, treatment_col, time_col])
        
        # Validate 2x2 structure
        n_treatment_groups = df[treatment_col].nunique()
        n_time_periods = df[time_col].nunique()
        
        if n_treatment_groups != 2:
            raise HTTPException(status_code=400, detail=f"Treatment column must have exactly 2 groups. Found {n_treatment_groups}.")
        if n_time_periods != 2:
            raise HTTPException(status_code=400, detail=f"Time column must have exactly 2 periods. Found {n_time_periods}.")
        
        if len(df) < 8:
            raise HTTPException(status_code=400, detail="At least 8 observations required (2 per cell).")
        
        # Manual DiD calculation
        manual_results = calculate_did_manual(df, outcome_col, treatment_col, time_col)
        
        # Regression DiD
        regression_results = run_did_regression(df, outcome_col, treatment_col, time_col, 
                                                 covariates, cluster_col)
        
        # Generate visualizations
        parallel_trends_plot = generate_parallel_trends_plot(df, outcome_col, treatment_col, time_col)
        did_diagram = generate_did_diagram(manual_results)
        group_comparison = generate_group_comparison_plot(df, outcome_col, treatment_col, time_col)
        distribution_plot = generate_distribution_plot(df, outcome_col, treatment_col, time_col)
        
        # Generate interpretation
        interpretation = generate_interpretation(manual_results, regression_results)
        
        # Get treatment and time values for grouping
        treatment_vals = sorted(df[treatment_col].unique())
        time_vals = sorted(df[time_col].unique())
        control, treated = treatment_vals[0], treatment_vals[1]
        pre, post = time_vals[0], time_vals[1]
        
        # Descriptive statistics with std for each cell
        descriptive_stats = {
            'n_total': len(df),
            'n_treatment': len(df[df[treatment_col] == treated]),
            'n_control': len(df[df[treatment_col] == control]),
            'outcome_mean': _to_native_type(df[outcome_col].mean()),
            'outcome_std': _to_native_type(df[outcome_col].std()),
            'treatment_groups': [str(v) for v in treatment_vals],
            'time_periods': [str(v) for v in time_vals],
            # Add std for each cell
            'treated_pre_std': _to_native_type(manual_results['stds']['treated_pre']) if manual_results else None,
            'treated_post_std': _to_native_type(manual_results['stds']['treated_post']) if manual_results else None,
            'control_pre_std': _to_native_type(manual_results['stds']['control_pre']) if manual_results else None,
            'control_post_std': _to_native_type(manual_results['stds']['control_post']) if manual_results else None
        }
        
        return {
            'manual_did': manual_results,
            'regression_did': regression_results,
            'descriptive_stats': descriptive_stats,
            'parallel_trends_plot': parallel_trends_plot,
            'did_diagram': did_diagram,
            'group_comparison': group_comparison,
            'distribution_plot': distribution_plot,
            'interpretation': interpretation,
            'covariates_used': covariates if covariates else [],
            'clustered_se': cluster_col is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DiD analysis failed: {str(e)}")
