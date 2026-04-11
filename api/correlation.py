from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import stats
import io
import base64
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()

# Maximum variables for pairs plot to prevent memory issues
MAX_PAIRS_PLOT_VARS = 6


class CorrelationRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    variables: List[str] = Field(..., description="Variables to correlate")
    controlVariables: Optional[List[str]] = Field(default=None, description="Variables to control for in partial correlation")
    groupVar: Optional[str] = Field(default=None)
    method: str = Field(default="pearson")
    alpha: float = Field(default=0.05)


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def interpret_magnitude(r: float) -> str:
    """Interpret correlation coefficient magnitude using Cohen's guidelines"""
    abs_r = abs(r)
    if abs_r >= 0.7:
        return 'very strong'
    if abs_r >= 0.5:
        return 'strong'
    if abs_r >= 0.3:
        return 'moderate'
    if abs_r >= 0.1:
        return 'weak'
    return 'negligible'


def compute_confidence_interval(r: float, n: int, alpha: float = 0.05) -> dict:
    """
    Compute confidence interval for correlation using Fisher's z-transformation.
    Works for Pearson and reasonably for Spearman.
    """
    if n < 4 or abs(r) >= 1.0:
        return {'lower': None, 'upper': None}
    
    try:
        # Fisher z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        # Critical value for confidence interval
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # CI in z-space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back to r-space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        # Clamp to [-1, 1]
        r_lower = max(-1.0, min(1.0, r_lower))
        r_upper = max(-1.0, min(1.0, r_upper))
        
        return {'lower': float(r_lower), 'upper': float(r_upper)}
    except:
        return {'lower': None, 'upper': None}


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction for multiple comparisons.
    Returns adjusted alpha and which p-values remain significant.
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return {'adjusted_alpha': alpha, 'significant_after_correction': 0}
    
    adjusted_alpha = alpha / n_tests
    significant_count = sum(1 for p in p_values if p < adjusted_alpha)
    
    return {
        'adjusted_alpha': adjusted_alpha,
        'significant_after_correction': significant_count,
        'n_tests': n_tests
    }


def compute_partial_correlation(df: pd.DataFrame, var1: str, var2: str, control_vars: List[str], method: str = 'pearson') -> dict:
    """
    Compute partial correlation between var1 and var2, controlling for control_vars.
    Uses regression-based approach.
    """
    if len(control_vars) == 0:
        # No control variables, return regular correlation
        if method == 'pearson':
            r, p = pearsonr(df[var1], df[var2])
        elif method == 'spearman':
            r, p = spearmanr(df[var1], df[var2])
        else:
            r, p = kendalltau(df[var1], df[var2])
        return {'r': float(r), 'p': float(p)}
    
    try:
        # Residualize var1 and var2 on control variables
        from sklearn.linear_model import LinearRegression
        
        X_control = df[control_vars].values
        
        # Residuals of var1
        model1 = LinearRegression()
        model1.fit(X_control, df[var1].values)
        resid1 = df[var1].values - model1.predict(X_control)
        
        # Residuals of var2
        model2 = LinearRegression()
        model2.fit(X_control, df[var2].values)
        resid2 = df[var2].values - model2.predict(X_control)
        
        # Correlation of residuals
        if method == 'pearson':
            r, p = pearsonr(resid1, resid2)
        elif method == 'spearman':
            r, p = spearmanr(resid1, resid2)
        else:
            r, p = kendalltau(resid1, resid2)
        
        return {'r': float(r), 'p': float(p)}
    except Exception as e:
        print(f"Partial correlation error: {e}")
        return {'r': None, 'p': None}


def compute_partial_correlation_matrix(df: pd.DataFrame, variables: List[str], control_vars: List[str], method: str = 'pearson') -> dict:
    """
    Compute partial correlation matrix where each pair is controlled for specified control variables.
    
    Args:
        df: DataFrame with the data
        variables: Variables to compute correlations between
        control_vars: Variables to control for (held constant)
        method: Correlation method ('pearson', 'spearman', 'kendall')
    """
    n_vars = len(variables)
    partial_corr_matrix = pd.DataFrame(np.eye(n_vars), index=variables, columns=variables)
    partial_p_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)), index=variables, columns=variables)
    
    partial_correlations = []
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            var1, var2 = variables[i], variables[j]
            
            result = compute_partial_correlation(df, var1, var2, control_vars, method)
            
            if result['r'] is not None:
                partial_corr_matrix.loc[var1, var2] = result['r']
                partial_corr_matrix.loc[var2, var1] = result['r']
                partial_p_matrix.loc[var1, var2] = result['p']
                partial_p_matrix.loc[var2, var1] = result['p']
                
                partial_correlations.append({
                    'variable_1': var1,
                    'variable_2': var2,
                    'partial_r': result['r'],
                    'p_value': result['p'],
                    'controlled_for': control_vars,
                    'significant': result['p'] < 0.05 if result['p'] is not None else False,
                    'magnitude': interpret_magnitude(result['r'])
                })
    
    return {
        'matrix': partial_corr_matrix.to_dict(),
        'p_matrix': partial_p_matrix.to_dict(),
        'pairs': sorted(partial_correlations, key=lambda x: abs(x['partial_r']) if x['partial_r'] else 0, reverse=True),
        'control_variables': control_vars
    }


def generate_heatmap(corr_matrix, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='vlag', 
        center=0, 
        vmin=-1, 
        vmax=1, 
        square=True, 
        linewidths=1, 
        ax=ax,
        annot_kws={'size': 9}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def generate_partial_heatmap(partial_matrix_dict, variables, title):
    """Generate heatmap for partial correlation matrix"""
    partial_df = pd.DataFrame(partial_matrix_dict)
    # Reorder to match variables order
    partial_df = partial_df.reindex(index=variables, columns=variables)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        partial_df, 
        annot=True, 
        fmt='.2f', 
        cmap='vlag', 
        center=0, 
        vmin=-1, 
        vmax=1, 
        square=True, 
        linewidths=1, 
        ax=ax,
        annot_kws={'size': 9}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def generate_pairs_plot(df, variables, group_var):
    """Generate pairs plot with variable limit to prevent memory issues"""
    plot_vars = [v for v in variables if v != group_var]
    
    # Limit variables to prevent memory issues
    if len(plot_vars) > MAX_PAIRS_PLOT_VARS:
        plot_vars = plot_vars[:MAX_PAIRS_PLOT_VARS]
    
    if len(plot_vars) < 2:
        return None
    
    try:
        if group_var and group_var in df.columns:
            g = sns.pairplot(
                df, 
                vars=plot_vars, 
                hue=group_var, 
                diag_kind='kde', 
                plot_kws={'alpha': 0.6, 's': 30}
            )
        else:
            g = sns.pairplot(
                df[plot_vars], 
                diag_kind='kde', 
                plot_kws={'alpha': 0.6, 's': 30}
            )
        
        title = 'Pairs Plot'
        if len(variables) > MAX_PAIRS_PLOT_VARS:
            title += f' (showing first {MAX_PAIRS_PLOT_VARS} of {len(variables)} variables)'
        g.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(g.fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        print(f"Pairs plot generation failed: {e}")
        return None


@router.post("/correlation")
def correlation_analysis(req: CorrelationRequest):
    try:
        df = pd.DataFrame(req.data)
        variables = req.variables
        group_var = req.groupVar
        method = req.method
        alpha = req.alpha
        
        original_len = len(df)
        
        # Clean data - convert to numeric
        for col in variables:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        clean_df = df[variables].dropna()
        dropped_rows = list(set(range(original_len)) - set(clean_df.index.tolist()))
        n_dropped = len(dropped_rows)
        n = len(clean_df)
        
        if n < 4:
            raise ValueError("Not enough valid data (minimum 4 observations required)")
        
        n_vars = len(variables)
        corr_matrix = pd.DataFrame(np.eye(n_vars), index=variables, columns=variables)
        p_value_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)), index=variables, columns=variables)
        
        all_correlations = []
        all_p_values = []
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                var1, var2 = variables[i], variables[j]
                col1, col2 = clean_df[var1], clean_df[var2]
                
                try:
                    if method == 'pearson':
                        corr, p_value = pearsonr(col1, col2)
                    elif method == 'spearman':
                        corr, p_value = spearmanr(col1, col2)
                    elif method == 'kendall':
                        corr, p_value = kendalltau(col1, col2)
                    else:
                        corr, p_value = pearsonr(col1, col2)
                    
                    corr_matrix.loc[var1, var2] = corr_matrix.loc[var2, var1] = corr
                    p_value_matrix.loc[var1, var2] = p_value_matrix.loc[var2, var1] = p_value
                    
                    if not np.isnan(corr):
                        # Compute confidence interval
                        ci = compute_confidence_interval(corr, n, alpha)
                        
                        # Compute r-squared (coefficient of determination)
                        r_squared = corr ** 2
                        
                        # Get magnitude interpretation
                        magnitude = interpret_magnitude(corr)
                        
                        all_correlations.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlation': float(corr),
                            'r_squared': float(r_squared),
                            'p_value': float(p_value),
                            'significant': bool(p_value < alpha),
                            'ci_lower': ci['lower'],
                            'ci_upper': ci['upper'],
                            'magnitude': magnitude
                        })
                        all_p_values.append(p_value)
                except Exception as e:
                    print(f"Error computing correlation for {var1} vs {var2}: {e}")
                    continue
        
        # Apply Bonferroni correction
        bonferroni = bonferroni_correction(all_p_values, alpha)
        
        # Update significant_bonferroni flag for each correlation
        for corr_item in all_correlations:
            corr_item['significant_bonferroni'] = bool(corr_item['p_value'] < bonferroni['adjusted_alpha'])
        
        # Compute Partial Correlations (if control variables are specified)
        partial_correlation_data = None
        partial_heatmap = None
        control_vars = req.controlVariables or []
        
        # Filter control variables to only include valid numeric columns
        control_vars = [v for v in control_vars if v in df.columns and v not in variables]
        
        if control_vars:
            try:
                # Convert control variables to numeric and add to clean_df
                for col in control_vars:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Create combined df with both analysis and control variables
                all_cols = variables + control_vars
                partial_clean_df = df[all_cols].dropna()
                
                if len(partial_clean_df) >= 4:
                    partial_correlation_data = compute_partial_correlation_matrix(
                        partial_clean_df, 
                        variables, 
                        control_vars, 
                        method
                    )
                    partial_heatmap = generate_partial_heatmap(
                        partial_correlation_data['matrix'], 
                        variables, 
                        f'Partial {method.capitalize()} Correlation (controlling for {", ".join(control_vars)})'
                    )
            except Exception as e:
                print(f"Partial correlation computation failed: {e}")
        
        # Summary statistics
        if all_correlations:
            corrs = [c['correlation'] for c in all_correlations]
            abs_corrs = [abs(c) for c in corrs]
            summary_stats = {
                'mean_correlation': float(np.mean(abs_corrs)),
                'median_correlation': float(np.median(abs_corrs)),
                'std_dev': float(np.std(abs_corrs)),
                'range': [float(np.min(corrs)), float(np.max(corrs))],
                'significant_correlations': sum(1 for c in all_correlations if c['significant']),
                'significant_after_bonferroni': bonferroni['significant_after_correction'],
                'total_pairs': len(all_correlations),
                'bonferroni_alpha': bonferroni['adjusted_alpha']
            }
        else:
            summary_stats = {
                'mean_correlation': 0,
                'median_correlation': 0,
                'std_dev': 0,
                'range': [0, 0],
                'significant_correlations': 0,
                'significant_after_bonferroni': 0,
                'total_pairs': 0,
                'bonferroni_alpha': alpha
            }
        
        # Sort by absolute correlation strength
        strongest = sorted(all_correlations, key=lambda x: abs(x['correlation']), reverse=True)
        
        # Generate interpretation text
        n_sig = summary_stats['significant_correlations']
        n_sig_bonf = summary_stats['significant_after_bonferroni']
        total = summary_stats['total_pairs']
        
        if strongest:
            top = strongest[0]
            top_mag = top['magnitude']
            top_dir = 'positive' if top['correlation'] > 0 else 'negative'
            
            interpretation_body = (
                f"{method.capitalize()} correlation analysis was performed on {n} observations across {n_vars} variables. "
                f"Of {total} correlations computed, {n_sig} ({(n_sig/total*100):.1f}%) were significant at α = {alpha}. "
            )
            
            if n_sig_bonf < n_sig:
                interpretation_body += (
                    f"After Bonferroni correction (adjusted α = {bonferroni['adjusted_alpha']:.4f}), "
                    f"{n_sig_bonf} correlations remained significant. "
                )
            
            interpretation_body += (
                f"The strongest relationship was between {top['variable_1']} and {top['variable_2']} "
                f"(r = {top['correlation']:.3f}, {top_mag} {top_dir}), explaining {top['r_squared']*100:.1f}% of variance."
            )
            
            if top['ci_lower'] is not None:
                interpretation_body += f" 95% CI: [{top['ci_lower']:.3f}, {top['ci_upper']:.3f}]."
            
            # Add partial correlation insight if available
            if partial_correlation_data and partial_correlation_data['pairs']:
                top_partial = partial_correlation_data['pairs'][0]
                interpretation_body += (
                    f" When controlling for other variables, the strongest partial correlation was between "
                    f"{top_partial['variable_1']} and {top_partial['variable_2']} (partial r = {top_partial['partial_r']:.3f})."
                )
        else:
            interpretation_body = f"No valid correlations could be computed from the data."
        
        interpretation = {
            'title': f'Correlation Analysis: {n_vars} Variables',
            'body': interpretation_body
        }
        
        # Generate plots
        heatmap = generate_heatmap(corr_matrix, f'{method.capitalize()} Correlation Matrix')
        
        plot_df = clean_df.copy()
        if group_var and group_var in df.columns:
            plot_df[group_var] = df.loc[clean_df.index, group_var]
        
        pairs_plot = generate_pairs_plot(plot_df, variables, group_var)
        
        # Note if pairs plot was limited
        pairs_plot_note = None
        if len(variables) > MAX_PAIRS_PLOT_VARS:
            pairs_plot_note = f"Pairs plot limited to first {MAX_PAIRS_PLOT_VARS} variables for performance."
        
        return _to_native({
            'correlation_matrix': corr_matrix.to_dict(),
            'p_value_matrix': p_value_matrix.to_dict(),
            'summary_statistics': summary_stats,
            'strongest_correlations': strongest,
            'interpretation': interpretation,
            'heatmap_plot': heatmap,
            'pairs_plot': pairs_plot,
            'pairs_plot_note': pairs_plot_note,
            'partial_correlations': partial_correlation_data,
            'partial_heatmap': partial_heatmap,
            'n_dropped': n_dropped,
            'dropped_rows': dropped_rows,
            'sample_size': n,
            'method': method,
            'alpha': alpha
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
