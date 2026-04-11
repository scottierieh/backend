"""
IMPROVED ANCOVA ANALYSIS WITH PROPER HOMOGENEITY OF SLOPES CHECKING
Critical: Interaction terms are NOT included in final ANCOVA table
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.nonparametric.smoothers_lowess import lowess
import io
import base64
import re
import math
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

router = APIRouter()


class AncovaRequest(BaseModel):
    """Request model for ANCOVA analysis"""
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    dependentVar: str = Field(..., description="Numeric dependent variable")
    factorVar: str = Field(..., description="Categorical factor variable")
    covariateVars: List[str] = Field(..., description="List of covariate variables")
    alpha: float = Field(0.05, description="Significance level")
    performPosthoc: bool = Field(False, description="Perform post-hoc tests if significant")


def _to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def validate_data(df, dependent_var, factor_var, covariate_vars):
    """Validate data before ANCOVA analysis"""
    errors = []
    
    # Check minimum sample size
    if len(df) < 20:
        errors.append("Sample size too small (minimum 20 required)")
    
    # Check if variables exist
    all_vars = [dependent_var, factor_var] + covariate_vars
    missing_vars = [var for var in all_vars if var not in df.columns]
    if missing_vars:
        errors.append(f"Missing variables: {', '.join(missing_vars)}")
    
    # Check group sizes
    if factor_var in df.columns:
        group_counts = df[factor_var].value_counts()
        small_groups = group_counts[group_counts < 3]
        if not small_groups.empty:
            errors.append(f"Groups with < 3 observations: {small_groups.to_dict()}")
        
        n_groups = df[factor_var].nunique()
        if n_groups < 2:
            errors.append("At least 2 groups required for ANCOVA")
    
    # Check covariate variation
    for cov in covariate_vars:
        if cov in df.columns:
            numeric_cov = pd.to_numeric(df[cov], errors='coerce')
            if numeric_cov.nunique() < 2:
                errors.append(f"Covariate '{cov}' has no variation")
    
    if errors:
        raise ValueError("; ".join(errors))


def check_homogeneity_of_slopes(clean_data, dv_clean, fv_clean, cv_clean, alpha):
    """
    CRITICAL: Check homogeneity of regression slopes
    Interaction terms should NOT be significant for valid ANCOVA
    """
    covariates_formula = ' + '.join(cv_clean)
    interaction_formula = f'{dv_clean} ~ C({fv_clean}) * ({covariates_formula})'
    
    try:
        interaction_model = ols(interaction_formula, data=clean_data).fit()
        interaction_anova = anova_lm(interaction_model, typ=2)
    except Exception as e:
        return {'testable': False, 'assumption_met': None, 'error': str(e)}
    
    # Check interaction terms - they should NOT be significant
    interaction_terms = [term for term in interaction_anova.index 
                        if ':' in term and fv_clean in term]
    
    test_results = {}
    interaction_significant = False
    
    for term in interaction_terms:
        display_name = term.replace(f'C({fv_clean}):', '').replace(':', ' × ')
        f_stat = interaction_anova.loc[term, 'F']
        p_value = interaction_anova.loc[term, 'PR(>F)']
        
        test_results[display_name] = {
            'F_statistic': float(f_stat) if not pd.isna(f_stat) else None,
            'p_value': float(p_value) if not pd.isna(p_value) else None,
            'significant': p_value < alpha if not pd.isna(p_value) else None
        }
        
        if not pd.isna(p_value) and p_value < alpha:
            interaction_significant = True
    
    return {
        'testable': True,
        'test_results': test_results,
        'assumption_met': not interaction_significant,
        'interpretation': (
            "✓ Homogeneity of slopes assumption MET - ANCOVA is appropriate" 
            if not interaction_significant else 
            "⚠️ CRITICAL: Homogeneity of slopes VIOLATED - ANCOVA may be invalid!"
        )
    }


def check_multicollinearity(clean_data, cv_clean, covariate_vars):
    """Check multicollinearity among covariates using VIF"""
    if len(cv_clean) <= 1:
        return None
    
    try:
        X = clean_data[cv_clean].values
        vif_data = []
        
        for i, (var_clean, var_orig) in enumerate(zip(cv_clean, covariate_vars)):
            vif_value = variance_inflation_factor(X, i)
            vif_data.append({
                'variable': var_orig,
                'VIF': float(vif_value) if not math.isinf(vif_value) else None,
                'interpretation': (
                    'No multicollinearity' if vif_value < 5 else
                    'Moderate multicollinearity' if vif_value < 10 else
                    'High multicollinearity - consider removing'
                )
            })
        return vif_data
    except:
        return None


def check_all_assumptions(clean_data, model, dv_clean, fv_clean, cv_clean, alpha):
    """Check all ANCOVA assumptions"""
    assumptions = {}
    residuals = model.resid
    
    # 1. Normality
    if len(residuals) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        assumptions['normality'] = {
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'met': shapiro_p > alpha,
            'interpretation': (
                '✓ Residuals normally distributed' if shapiro_p > alpha 
                else '⚠️ Normality assumption violated'
            )
        }
    
    # 2. Homoscedasticity
    groups = [group[dv_clean].values for _, group in clean_data.groupby(fv_clean)]
    if len(groups) > 1:
        levene_stat, levene_p = stats.levene(*groups)
        assumptions['homoscedasticity'] = {
            'statistic': float(levene_stat),
            'p_value': float(levene_p),
            'met': levene_p > alpha,
            'interpretation': (
                '✓ Equal variances' if levene_p > alpha 
                else '⚠️ Unequal variances'
            )
        }
    
    # 3. Independence
    dw_stat = durbin_watson(residuals)
    assumptions['independence'] = {
        'durbin_watson': float(dw_stat),
        'met': 1.5 < dw_stat < 2.5,
        'interpretation': (
            '✓ Residuals independent' if 1.5 < dw_stat < 2.5 
            else '⚠️ Autocorrelation detected'
        )
    }
    
    # 4. Outliers
    z_scores = np.abs(stats.zscore(residuals))
    n_outliers = np.sum(z_scores > 3)
    assumptions['outliers'] = {
        'n_outliers': int(n_outliers),
        'percent': float(100 * n_outliers / len(residuals)),
        'met': n_outliers / len(residuals) < 0.05
    }
    
    return assumptions


def perform_posthoc_tests(clean_data, model, dv_clean, fv_clean, alpha):
    """Perform Tukey HSD post-hoc tests"""
    try:
        adjusted_dv = model.fittedvalues
        mc = pairwise_tukeyhsd(adjusted_dv, clean_data[fv_clean], alpha=alpha)
        
        results = []
        for i in range(1, len(mc.summary().data)):
            row = mc.summary().data[i]
            results.append({
                'group1': str(row[0]),
                'group2': str(row[1]),
                'mean_diff': float(row[2]),
                'p_adj': float(row[5]),
                'ci_lower': float(row[6]),
                'ci_upper': float(row[7]),
                'significant': bool(row[8])
            })
        return results
    except:
        return None


def calculate_effect_sizes(anova_table):
    """Calculate effect sizes"""
    if 'Residual' in anova_table.index:
        ss_residual = anova_table.loc['Residual', 'sum_sq']
        ss_total = anova_table['sum_sq'].sum()
        
        for idx in anova_table.index:
            if idx != 'Residual':
                ss_effect = anova_table.loc[idx, 'sum_sq']
                # Partial eta squared
                eta_p = ss_effect / (ss_effect + ss_residual)
                anova_table.loc[idx, 'eta_squared_partial'] = eta_p
                # Cohen's f
                cohen_f = np.sqrt(eta_p / (1 - eta_p))
                anova_table.loc[idx, 'cohen_f'] = cohen_f
                # Interpretation
                anova_table.loc[idx, 'effect_size'] = (
                    'small' if cohen_f < 0.25 else
                    'medium' if cohen_f < 0.40 else 'large'
                )
    return anova_table


def generate_comprehensive_plot(clean_data, model, dv_clean, fv_clean, cv_clean, 
                                dependent_var, factor_var, covariate_vars):
    """Generate comprehensive diagnostic plots"""
    fig = plt.figure(figsize=(16, 10))
    palette = sns.color_palette("husl", n_colors=len(clean_data[fv_clean].unique()))
    
    # 1. Homogeneity of slopes check
    ax1 = plt.subplot(2, 3, 1)
    if cv_clean:
        for i, (group_name, group_data) in enumerate(clean_data.groupby(fv_clean)):
            ax1.scatter(group_data[cv_clean[0]], group_data[dv_clean], 
                       alpha=0.6, s=50, label=str(group_name), color=palette[i])
            if len(group_data) > 1:
                z = np.polyfit(group_data[cv_clean[0]], group_data[dv_clean], 1)
                p = np.poly1d(z)
                x_line = np.linspace(group_data[cv_clean[0]].min(), 
                                   group_data[cv_clean[0]].max(), 100)
                ax1.plot(x_line, p(x_line), color=palette[i], linewidth=2, alpha=0.8)
        ax1.set_xlabel(covariate_vars[0])
        ax1.set_ylabel(dependent_var)
        ax1.set_title('Homogeneity of Slopes Check', fontweight='bold')
        ax1.legend(title=factor_var)
        ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    ax2 = plt.subplot(2, 3, 2)
    residuals = model.resid
    sm.qqplot(residuals, line='s', ax=ax2)
    ax2.set_title('Q-Q Plot (Normality)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals vs Fitted
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(model.fittedvalues, residuals, alpha=0.6)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs Fitted', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Histogram of residuals
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Scale-Location plot
    ax5 = plt.subplot(2, 3, 5)
    sqrt_abs_resid = np.sqrt(np.abs(residuals / residuals.std()))
    ax5.scatter(model.fittedvalues, sqrt_abs_resid, alpha=0.6)
    ax5.set_xlabel('Fitted Values')
    ax5.set_ylabel('√|Standardized Residuals|')
    ax5.set_title('Scale-Location', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Adjusted means
    ax6 = plt.subplot(2, 3, 6)
    covariate_mean = {cv: clean_data[cv].mean() for cv in cv_clean}
    adjusted_means = []
    group_names = []
    
    for group in clean_data[fv_clean].unique():
        pred_data = pd.DataFrame({
            fv_clean: [group],
            **{cv: [covariate_mean[cv]] for cv in cv_clean}
        })
        pred = model.predict(pred_data)[0]
        adjusted_means.append(pred)
        group_names.append(str(group))
    
    x_pos = np.arange(len(group_names))
    ax6.bar(x_pos, adjusted_means, color=palette, alpha=0.7)
    ax6.set_xlabel(factor_var)
    ax6.set_ylabel(f'Adjusted {dependent_var}')
    ax6.set_title('Adjusted Group Means', fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(group_names)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('ANCOVA Diagnostic Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/ancova")
def ancova_analysis(req: AncovaRequest):
    """
    Perform ANCOVA with proper assumptions checking
    CRITICAL: No interaction terms in final ANOVA table
    """
    try:
        df = pd.DataFrame(req.data)
        dependent_var = req.dependentVar
        factor_var = req.factorVar
        covariate_vars = req.covariateVars if isinstance(req.covariateVars, list) else [req.covariateVars]
        alpha = req.alpha if hasattr(req, 'alpha') else 0.05
        
        # Validate data
        validate_data(df, dependent_var, factor_var, covariate_vars)
        
        # Clean data
        all_vars = [dependent_var, factor_var] + covariate_vars
        original_len = len(df)
        clean_data = df[all_vars].copy()
        
        for var in [dependent_var] + covariate_vars:
            clean_data[var] = pd.to_numeric(clean_data[var], errors='coerce')
        
        clean_data[factor_var] = clean_data[factor_var].astype('category')
        clean_data = clean_data.dropna()
        
        dropped_rows = list(set(range(original_len)) - set(clean_data.index.tolist()))
        n_dropped = len(dropped_rows)
        
        if len(clean_data) < 20:
            raise ValueError(f"Insufficient data: {len(clean_data)} observations")
        
        # Sanitize names
        dv_clean = re.sub(r'[^A-Za-z0-9_]', '_', dependent_var)
        fv_clean = re.sub(r'[^A-Za-z0-9_]', '_', factor_var)
        cv_clean = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in covariate_vars]
        
        rename_dict = {dependent_var: dv_clean, factor_var: fv_clean}
        for orig, clean in zip(covariate_vars, cv_clean):
            rename_dict[orig] = clean
        clean_data = clean_data.rename(columns=rename_dict)
        
        # CRITICAL: Check homogeneity of slopes FIRST
        homogeneity_test = check_homogeneity_of_slopes(
            clean_data, dv_clean, fv_clean, cv_clean, alpha
        )
        
        # Check multicollinearity
        multicollinearity = check_multicollinearity(clean_data, cv_clean, covariate_vars)
        
        # Fit ANCOVA model WITHOUT interaction terms
        covariates_formula = ' + '.join(cv_clean)
        main_formula = f'{dv_clean} ~ C({fv_clean}) + {covariates_formula}'
        model = ols(main_formula, data=clean_data).fit()
        anova_table = anova_lm(model, typ=2)
        
        # Calculate effect sizes
        anova_table = calculate_effect_sizes(anova_table)
        
        # Format ANOVA table (NO interaction terms!)
        cleaned_index = {
            f'C({fv_clean})': f'Between Groups ({factor_var})',
            'Residual': 'Within Groups (Error)'
        }
        for i, cv in enumerate(cv_clean):
            cleaned_index[cv] = f'Covariate ({covariate_vars[i]})'
        
        anova_table_renamed = anova_table.rename(index=cleaned_index)
        
        # Convert to dict
        anova_results = []
        for idx, row in anova_table_renamed.iterrows():
            result = {
                'Source': idx,
                'sum_sq': float(row['sum_sq']),
                'df': float(row['df']),
                'F': float(row['F']) if 'F' in row and not pd.isna(row['F']) else None,
                'p-value': float(row['PR(>F)']) if 'PR(>F)' in row else None,
            }
            if 'eta_squared_partial' in row and idx != 'Within Groups (Error)':
                result['η²p'] = float(row['eta_squared_partial'])
                result['effect_size'] = row.get('effect_size', '')
            anova_results.append(result)
        
        # Adjusted means
        adjusted_means = {}
        covariate_means = {cv: float(clean_data[cv].mean()) for cv in cv_clean}
        
        for group_name in clean_data[fv_clean].unique():
            pred_data = pd.DataFrame({
                fv_clean: [group_name],
                **{cv: [covariate_means[cv]] for cv in cv_clean}
            })
            pred_value = model.predict(pred_data)[0]
            n_in_group = len(clean_data[clean_data[fv_clean] == group_name])
            mse = np.sum(model.resid**2) / model.df_resid
            se = np.sqrt(mse / n_in_group)
            adjusted_means[str(group_name)] = {
                'adjusted_mean': float(pred_value),
                'se': float(se),
                'n': int(n_in_group),
                'ci_lower': float(pred_value - 1.96*se),
                'ci_upper': float(pred_value + 1.96*se)
            }
        
        # Covariate info
        covariate_info = {}
        for i, cv in enumerate(cv_clean):
            if cv in model.params.index:
                covariate_info[covariate_vars[i]] = {
                    'coefficient': float(model.params[cv]),
                    'std_err': float(model.bse[cv]),
                    't_value': float(model.tvalues[cv]),
                    'p_value': float(model.pvalues[cv])
                }
        
        # Check assumptions
        assumptions = check_all_assumptions(
            clean_data, model, dv_clean, fv_clean, cv_clean, alpha
        )
        
        # Post-hoc tests
        posthoc_results = None
        main_effect_sig = any(
            'Between Groups' in r['Source'] and r.get('p-value', 1) < alpha 
            for r in anova_results
        )
        if hasattr(req, 'performPosthoc') and req.performPosthoc and main_effect_sig and len(adjusted_means) > 2:
            posthoc_results = perform_posthoc_tests(
                clean_data, model, dv_clean, fv_clean, alpha
            )
        
        # Generate interpretation
        interpretation = []
        interpretation.append(f"ANCOVA: '{factor_var}' effect on '{dependent_var}' controlling for {', '.join(covariate_vars)}")
        
        if not homogeneity_test['assumption_met']:
            interpretation.append("⚠️ WARNING: Homogeneity of slopes violated - results may be invalid!")
        else:
            interpretation.append("✓ Homogeneity of slopes assumption met")
        
        if main_effect_sig:
            interpretation.append(f"✓ Significant main effect found (p<{alpha})")
        else:
            interpretation.append(f"No significant main effect (p≥{alpha})")
        
        # Plot
        plot = generate_comprehensive_plot(
            clean_data, model, dv_clean, fv_clean, cv_clean,
            dependent_var, factor_var, covariate_vars
        )
        
        residuals = model.resid.tolist()
        
        return _to_native({
            "results": {
                "homogeneity_of_slopes": homogeneity_test,
                "multicollinearity": multicollinearity,
                "anova_table": anova_results,
                "adjusted_means": adjusted_means,
                "covariate_info": covariate_info,
                "covariate_means": {covariate_vars[i]: covariate_means[cv] 
                                   for i, cv in enumerate(cv_clean)},
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "assumptions": assumptions,
                "posthoc_tests": posthoc_results,
                "interpretation": "\n".join(interpretation),
                "residuals": residuals,
                "dropped_rows": dropped_rows,
                "n_dropped": n_dropped
            },
            "plot": plot,
            "warnings": [] if homogeneity_test['assumption_met'] else 
                       ["Homogeneity of slopes violated - ANCOVA may be inappropriate"]
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
