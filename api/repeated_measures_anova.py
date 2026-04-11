"""
Repeated Measures ANOVA Router for FastAPI
Analyze within-subject effects across multiple time points or conditions
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


class RepeatedMeasuresRequest(BaseModel):
    data: List[Dict[str, Any]]
    subject_col: str
    measure_cols: List[str]  # Columns representing repeated measures (e.g., Time1, Time2, Time3)
    alpha: float = 0.05


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
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_sphericity(data_wide: pd.DataFrame, measure_cols: List[str]) -> Dict[str, Any]:
    """Calculate Mauchly's test for sphericity"""
    try:
        k = len(measure_cols)
        n = len(data_wide)
        
        if k < 3:
            return {
                'w_statistic': None,
                'p_value': None,
                'epsilon_greenhouse_geisser': 1.0,
                'epsilon_huynh_feldt': 1.0,
                'sphericity_assumed': True,
                'message': 'Sphericity test requires 3+ levels'
            }
        
        # Calculate covariance matrix
        cov_matrix = data_wide[measure_cols].cov()
        
        # Calculate epsilon corrections
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Greenhouse-Geisser epsilon
        sum_eigenvalues = np.sum(eigenvalues)
        sum_squared_eigenvalues = np.sum(eigenvalues ** 2)
        epsilon_gg = (sum_eigenvalues ** 2) / ((k - 1) * sum_squared_eigenvalues) if sum_squared_eigenvalues > 0 else 1.0
        epsilon_gg = min(1.0, max(1.0 / (k - 1), epsilon_gg))
        
        # Huynh-Feldt epsilon
        epsilon_hf = (n * (k - 1) * epsilon_gg - 2) / ((k - 1) * (n - 1 - (k - 1) * epsilon_gg))
        epsilon_hf = min(1.0, max(epsilon_gg, epsilon_hf))
        
        # Simplified Mauchly's W calculation
        det_cov = np.linalg.det(cov_matrix)
        trace_cov = np.trace(cov_matrix)
        w_statistic = (det_cov / ((trace_cov / k) ** k)) if trace_cov > 0 else 0
        
        # Chi-square approximation for p-value
        df = (k * (k - 1) / 2) - 1
        if df > 0 and w_statistic > 0:
            chi_sq = -((n - 1) - (2 * k ** 2 - 3 * k + 3) / (6 * (k - 1))) * np.log(w_statistic)
            p_value = 1 - stats.chi2.cdf(chi_sq, df)
        else:
            chi_sq = None
            p_value = None
        
        sphericity_assumed = bool(p_value is None or p_value > 0.05)
        
        return {
            'w_statistic': _to_native_type(w_statistic),
            'chi_square': _to_native_type(chi_sq),
            'df': _to_native_type(df),
            'p_value': _to_native_type(p_value),
            'epsilon_greenhouse_geisser': _to_native_type(epsilon_gg),
            'epsilon_huynh_feldt': _to_native_type(epsilon_hf),
            'sphericity_assumed': sphericity_assumed,
            'message': 'Sphericity assumed' if sphericity_assumed else 'Sphericity violated - use corrected values'
        }
    except Exception as e:
        return {
            'w_statistic': None,
            'p_value': None,
            'epsilon_greenhouse_geisser': 1.0,
            'epsilon_huynh_feldt': 1.0,
            'sphericity_assumed': True,
            'message': f'Sphericity calculation error: {str(e)}'
        }


def perform_repeated_measures_anova(data_wide: pd.DataFrame, subject_col: str, measure_cols: List[str], alpha: float) -> Dict[str, Any]:
    """Perform repeated measures ANOVA calculation"""
    
    n = len(data_wide)  # Number of subjects
    k = len(measure_cols)  # Number of conditions/time points
    
    # Calculate means
    grand_mean = data_wide[measure_cols].values.mean()
    condition_means = data_wide[measure_cols].mean()
    subject_means = data_wide[measure_cols].mean(axis=1)
    
    # Total Sum of Squares
    ss_total = np.sum((data_wide[measure_cols].values - grand_mean) ** 2)
    
    # Between-subjects SS
    ss_between_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    
    # Within-subjects SS (total)
    ss_within_subjects = ss_total - ss_between_subjects
    
    # Treatment (condition) SS
    ss_treatment = n * np.sum((condition_means - grand_mean) ** 2)
    
    # Error SS (residual)
    ss_error = ss_within_subjects - ss_treatment
    
    # Degrees of freedom
    df_treatment = k - 1
    df_error = (n - 1) * (k - 1)
    df_total = n * k - 1
    
    # Mean squares
    ms_treatment = ss_treatment / df_treatment if df_treatment > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # F-statistic
    f_statistic = ms_treatment / ms_error if ms_error > 0 else 0
    
    # P-value
    p_value = 1 - stats.f.cdf(f_statistic, df_treatment, df_error) if df_treatment > 0 and df_error > 0 else 1
    
    # Effect sizes
    # Partial eta squared
    partial_eta_squared = ss_treatment / (ss_treatment + ss_error) if (ss_treatment + ss_error) > 0 else 0
    
    # Generalized eta squared
    gen_eta_squared = ss_treatment / ss_total if ss_total > 0 else 0
    
    # Omega squared
    omega_squared = (ss_treatment - df_treatment * ms_error) / (ss_total + ms_error) if (ss_total + ms_error) > 0 else 0
    omega_squared = max(0, omega_squared)
    
    return {
        'ss_treatment': _to_native_type(ss_treatment),
        'ss_error': _to_native_type(ss_error),
        'ss_between_subjects': _to_native_type(ss_between_subjects),
        'ss_total': _to_native_type(ss_total),
        'df_treatment': _to_native_type(df_treatment),
        'df_error': _to_native_type(df_error),
        'df_total': _to_native_type(df_total),
        'ms_treatment': _to_native_type(ms_treatment),
        'ms_error': _to_native_type(ms_error),
        'f_statistic': _to_native_type(f_statistic),
        'p_value': _to_native_type(p_value),
        'partial_eta_squared': _to_native_type(partial_eta_squared),
        'generalized_eta_squared': _to_native_type(gen_eta_squared),
        'omega_squared': _to_native_type(omega_squared),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'n_subjects': n,
        'n_conditions': k,
        'grand_mean': _to_native_type(grand_mean)
    }


def pairwise_comparisons(data_wide: pd.DataFrame, measure_cols: List[str], alpha: float) -> List[Dict[str, Any]]:
    """Perform pairwise t-tests with Bonferroni correction"""
    comparisons = []
    n_comparisons = len(measure_cols) * (len(measure_cols) - 1) // 2
    adjusted_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
    
    for i, col1 in enumerate(measure_cols):
        for j, col2 in enumerate(measure_cols):
            if i < j:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(data_wide[col1], data_wide[col2])
                
                # Effect size (Cohen's d for paired samples)
                diff = data_wide[col1] - data_wide[col2]
                cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
                
                # Mean difference
                mean_diff = data_wide[col1].mean() - data_wide[col2].mean()
                
                comparisons.append({
                    'comparison': f'{col1} vs {col2}',
                    'condition_1': col1,
                    'condition_2': col2,
                    'mean_1': _to_native_type(data_wide[col1].mean()),
                    'mean_2': _to_native_type(data_wide[col2].mean()),
                    'mean_difference': _to_native_type(mean_diff),
                    't_statistic': _to_native_type(t_stat),
                    'p_value': _to_native_type(p_value),
                    'p_adjusted': _to_native_type(min(p_value * n_comparisons, 1.0)),
                    'cohens_d': _to_native_type(cohens_d),
                    'significant_raw': bool(p_value < alpha),
                    'significant_adjusted': bool(p_value < adjusted_alpha)
                })
    
    return comparisons


def generate_profile_plot(data_wide: pd.DataFrame, measure_cols: List[str]) -> str:
    """Generate profile plot (line plot with individual subjects)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual subject lines
    for idx, row in data_wide.iterrows():
        ax.plot(range(len(measure_cols)), row[measure_cols].values, 
                color='gray', alpha=0.3, linewidth=1)
    
    # Plot mean line
    means = data_wide[measure_cols].mean()
    stds = data_wide[measure_cols].std()
    
    ax.errorbar(range(len(measure_cols)), means, yerr=stds, 
                color='blue', linewidth=3, marker='o', markersize=10,
                capsize=5, capthick=2, label='Mean ± SD')
    
    ax.set_xticks(range(len(measure_cols)))
    ax.set_xticklabels(measure_cols, rotation=45, ha='right')
    ax.set_xlabel('Condition / Time Point', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Profile Plot: Individual Trajectories with Mean', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_boxplot(data_wide: pd.DataFrame, measure_cols: List[str]) -> str:
    """Generate box plot for each condition"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_long = data_wide[measure_cols].melt(var_name='Condition', value_name='Value')
    
    box = ax.boxplot([data_wide[col].dropna() for col in measure_cols],
                     labels=measure_cols, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(measure_cols)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, col in enumerate(measure_cols):
        y = data_wide[col].dropna()
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=30, color='darkblue', zorder=3)
    
    ax.set_xlabel('Condition / Time Point', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Distribution by Condition', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_mean_comparison_plot(data_wide: pd.DataFrame, measure_cols: List[str], anova_result: Dict) -> str:
    """Generate bar plot of means with error bars"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = data_wide[measure_cols].mean()
    sems = data_wide[measure_cols].sem()
    
    x = np.arange(len(measure_cols))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(measure_cols)))
    
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, edgecolor='darkblue', linewidth=1.5)
    
    # Add value labels
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.02 * max(means),
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(measure_cols, rotation=45, ha='right')
    ax.set_xlabel('Condition / Time Point', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    
    sig_text = '(p < .05)' if anova_result['significant'] else '(n.s.)'
    ax.set_title(f'Mean Comparison Across Conditions {sig_text}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_pairwise_heatmap(pairwise: List[Dict], measure_cols: List[str]) -> str:
    """Generate heatmap of pairwise p-values"""
    if not pairwise:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n = len(measure_cols)
    p_matrix = np.ones((n, n))
    
    for comp in pairwise:
        i = measure_cols.index(comp['condition_1'])
        j = measure_cols.index(comp['condition_2'])
        p_matrix[i, j] = comp['p_adjusted']
        p_matrix[j, i] = comp['p_adjusted']
    
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))
    
    sns.heatmap(p_matrix, mask=mask, annot=True, fmt='.3f',
                xticklabels=measure_cols, yticklabels=measure_cols,
                cmap='RdYlGn_r', vmin=0, vmax=0.1, ax=ax,
                cbar_kws={'label': 'Adjusted p-value'})
    
    ax.set_title('Pairwise Comparison p-values (Bonferroni)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(anova_result: Dict, sphericity: Dict, pairwise: List[Dict], descriptives: Dict) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    
    # Effect size interpretation
    eta_sq = anova_result['partial_eta_squared']
    if eta_sq >= 0.14:
        effect_size_interp = 'large'
    elif eta_sq >= 0.06:
        effect_size_interp = 'medium'
    elif eta_sq >= 0.01:
        effect_size_interp = 'small'
    else:
        effect_size_interp = 'negligible'
    
    # Significant pairwise comparisons
    sig_pairs = [p for p in pairwise if p['significant_adjusted']]
    
    # Key insights
    key_insights = []
    
    if anova_result['significant']:
        key_insights.append({
            'title': 'Significant Effect Found',
            'description': f"There is a statistically significant difference across conditions (F({anova_result['df_treatment']}, {anova_result['df_error']}) = {anova_result['f_statistic']:.2f}, p = {anova_result['p_value']:.4f})."
        })
    else:
        key_insights.append({
            'title': 'No Significant Effect',
            'description': f"No statistically significant difference found (F({anova_result['df_treatment']}, {anova_result['df_error']}) = {anova_result['f_statistic']:.2f}, p = {anova_result['p_value']:.4f})."
        })
    
    key_insights.append({
        'title': 'Effect Size',
        'description': f"Partial η² = {eta_sq:.3f}, indicating a {effect_size_interp} effect."
    })
    
    if not sphericity['sphericity_assumed']:
        key_insights.append({
            'title': 'Sphericity Violated',
            'description': f"Mauchly's test indicates sphericity violation. Use Greenhouse-Geisser (ε = {sphericity['epsilon_greenhouse_geisser']:.3f}) correction."
        })
    
    if sig_pairs:
        pairs_str = ', '.join([p['comparison'] for p in sig_pairs[:3]])
        key_insights.append({
            'title': 'Significant Differences',
            'description': f"{len(sig_pairs)} pairwise comparisons are significant after Bonferroni correction: {pairs_str}{'...' if len(sig_pairs) > 3 else ''}"
        })
    
    return {
        'effect_size_interpretation': effect_size_interp,
        'significant_pairs_count': len(sig_pairs),
        'key_insights': key_insights,
        'recommendation': 'Report corrected p-values if sphericity is violated.' if not sphericity['sphericity_assumed'] else 'Sphericity assumption met, uncorrected values are valid.'
    }


@router.post("/repeated-measures-anova")
async def run_repeated_measures_anova(request: RepeatedMeasuresRequest) -> Dict[str, Any]:
    """
    Perform Repeated Measures ANOVA analysis.
    
    Analyzes within-subject effects across multiple time points or conditions.
    Includes sphericity test, pairwise comparisons, and visualizations.
    """
    try:
        data = request.data
        subject_col = request.subject_col
        measure_cols = request.measure_cols
        alpha = request.alpha
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        if not subject_col:
            raise HTTPException(status_code=400, detail="Subject column not specified.")
        
        if len(measure_cols) < 2:
            raise HTTPException(status_code=400, detail="At least 2 measure columns required.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        missing_cols = [col for col in [subject_col] + measure_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_cols)}")
        
        # Convert to numeric and drop missing
        for col in measure_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = df[[subject_col] + measure_cols].dropna()
        
        if len(df_clean) < 3:
            raise HTTPException(status_code=400, detail="At least 3 complete observations required.")
        
        # Descriptive statistics
        descriptives = {}
        for col in measure_cols:
            descriptives[col] = {
                'mean': _to_native_type(df_clean[col].mean()),
                'std': _to_native_type(df_clean[col].std()),
                'sem': _to_native_type(df_clean[col].sem()),
                'min': _to_native_type(df_clean[col].min()),
                'max': _to_native_type(df_clean[col].max()),
                'n': _to_native_type(len(df_clean[col]))
            }
        
        # Sphericity test
        sphericity = calculate_sphericity(df_clean, measure_cols)
        
        # Main ANOVA
        anova_result = perform_repeated_measures_anova(df_clean, subject_col, measure_cols, alpha)
        
        # Corrected p-values if sphericity violated
        if not sphericity['sphericity_assumed'] and sphericity['epsilon_greenhouse_geisser']:
            eps_gg = sphericity['epsilon_greenhouse_geisser']
            df_treatment_gg = anova_result['df_treatment'] * eps_gg
            df_error_gg = anova_result['df_error'] * eps_gg
            p_value_gg = 1 - stats.f.cdf(anova_result['f_statistic'], df_treatment_gg, df_error_gg)
            anova_result['p_value_greenhouse_geisser'] = _to_native_type(p_value_gg)
            anova_result['df_treatment_greenhouse_geisser'] = _to_native_type(df_treatment_gg)
            anova_result['df_error_greenhouse_geisser'] = _to_native_type(df_error_gg)
            
            eps_hf = sphericity['epsilon_huynh_feldt']
            df_treatment_hf = anova_result['df_treatment'] * eps_hf
            df_error_hf = anova_result['df_error'] * eps_hf
            p_value_hf = 1 - stats.f.cdf(anova_result['f_statistic'], df_treatment_hf, df_error_hf)
            anova_result['p_value_huynh_feldt'] = _to_native_type(p_value_hf)
            anova_result['df_treatment_huynh_feldt'] = _to_native_type(df_treatment_hf)
            anova_result['df_error_huynh_feldt'] = _to_native_type(df_error_hf)
        
        # Pairwise comparisons
        pairwise = pairwise_comparisons(df_clean, measure_cols, alpha)
        
        # Generate visualizations
        profile_plot = generate_profile_plot(df_clean, measure_cols)
        boxplot = generate_boxplot(df_clean, measure_cols)
        mean_plot = generate_mean_comparison_plot(df_clean, measure_cols, anova_result)
        pairwise_heatmap = generate_pairwise_heatmap(pairwise, measure_cols) if len(measure_cols) >= 3 else None
        
        # Interpretation
        interpretation = generate_interpretation(anova_result, sphericity, pairwise, descriptives)
        
        return {
            'anova_result': anova_result,
            'sphericity': sphericity,
            'descriptives': descriptives,
            'pairwise_comparisons': pairwise,
            'profile_plot': profile_plot,
            'boxplot': boxplot,
            'mean_plot': mean_plot,
            'pairwise_heatmap': pairwise_heatmap,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Repeated measures ANOVA failed: {str(e)}")
