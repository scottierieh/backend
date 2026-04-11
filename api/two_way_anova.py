from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io
import base64
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class TwoWayAnovaRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    dependentVar: str = Field(..., description="Numeric dependent variable")
    factorA: str = Field(..., description="First categorical factor")
    factorB: str = Field(..., description="Second categorical factor")
    ssType: int = Field(default=3, description="Sum of Squares type (1, 2, or 3)")


def games_howell_test(data, group_col, value_col, alpha=0.05):
    """Games-Howell post-hoc test for unequal variances"""
    groups = data[group_col].unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        return []
    
    # Get group statistics
    group_stats = {}
    for g in groups:
        group_data = data[data[group_col] == g][value_col]
        if len(group_data) > 1:
            group_stats[g] = {
                'mean': group_data.mean(),
                'var': group_data.var(ddof=1),
                'n': len(group_data)
            }
    
    results = []
    comparisons = []
    
    group_list = list(group_stats.keys())
    for i, g1 in enumerate(group_list):
        for g2 in group_list[i+1:]:
            comparisons.append((g1, g2))
    
    for g1, g2 in comparisons:
        n1, n2 = group_stats[g1]['n'], group_stats[g2]['n']
        v1, v2 = group_stats[g1]['var'], group_stats[g2]['var']
        m1, m2 = group_stats[g1]['mean'], group_stats[g2]['mean']
        
        # Standard error
        se = np.sqrt(v1/n1 + v2/n2)
        
        # Mean difference
        mean_diff = m1 - m2
        
        # Welch-Satterthwaite degrees of freedom
        df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        # t-statistic
        t_stat = mean_diff / se
        
        # p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Bonferroni-style adjustment
        n_comparisons = len(comparisons)
        p_adj = min(p_value * n_comparisons, 1.0)
        
        # 95% CI
        t_crit = stats.t.ppf(1 - alpha/2, df)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        
        results.append({
            'group1': str(g1),
            'group2': str(g2),
            'meandiff': float(mean_diff),
            'se': float(se),
            'df': float(df),
            't_stat': float(t_stat),
            'p_adj': float(p_adj),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'reject': bool(p_adj < alpha)
        })
    
    return results


def generate_plot(clean_data, fa_clean, fb_clean, dv_clean, factor_a, factor_b, dependent_var, model):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Interaction plot
    sns.pointplot(data=clean_data, x=fa_clean, y=dv_clean, hue=fb_clean, ax=axes[0, 0], dodge=True, errorbar='ci', capsize=.1, palette='crest')
    axes[0, 0].set_title('Interaction Plot', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel(factor_a)
    axes[0, 0].set_ylabel(f'Mean of {dependent_var}')
    axes[0, 0].legend(title=factor_b)
    
    # Boxplot Factor A
    sns.boxplot(x=fa_clean, y=dv_clean, data=clean_data, ax=axes[0, 1], palette='crest')
    axes[0, 1].set_title(f'Distribution by {factor_a}')
    
    # Boxplot Factor B
    sns.boxplot(x=fb_clean, y=dv_clean, data=clean_data, ax=axes[1, 0], palette='crest')
    axes[1, 0].set_title(f'Distribution by {factor_b}')
    
    # Q-Q plot
    sm.qqplot(model.resid, line='s', ax=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/two-way-anova")
def two_way_anova(req: TwoWayAnovaRequest):
    try:
        df = pd.DataFrame(req.data)
        dependent_var = req.dependentVar
        factor_a = req.factorA
        factor_b = req.factorB
        ss_type = req.ssType if req.ssType in [1, 2, 3] else 3
        alpha = 0.05
        
        # Clean data
        original_len = len(df)
        all_vars = [dependent_var, factor_a, factor_b]
        clean_data = df[all_vars].dropna().copy()
        clean_data[dependent_var] = pd.to_numeric(clean_data[dependent_var], errors='coerce')
        clean_data = clean_data.dropna()
        
        dropped_rows = list(set(range(original_len)) - set(clean_data.index.tolist()))
        n_dropped = len(dropped_rows)
        n_used = len(clean_data)
        
        # Sanitize column names
        dv_clean = re.sub(r'[^A-Za-z0-9_]', '_', dependent_var)
        fa_clean = re.sub(r'[^A-Za-z0-9_]', '_', factor_a)
        fb_clean = re.sub(r'[^A-Za-z0-9_]', '_', factor_b)
        
        clean_data_renamed = clean_data.rename(columns={
            dependent_var: dv_clean,
            factor_a: fa_clean,
            factor_b: fb_clean
        })
        
        # Fit model
        model = ols(f'Q("{dv_clean}") ~ C(Q("{fa_clean}")) * C(Q("{fb_clean}"))', data=clean_data_renamed).fit()
        
        # ANOVA table with specified SS type
        anova_table = anova_lm(model, typ=ss_type)
        anova_table['MS'] = anova_table['sum_sq'] / anova_table['df']
        
        # Calculate effect sizes
        ss_total = anova_table['sum_sq'].sum()
        ss_error = anova_table.loc['Residual', 'sum_sq'] if 'Residual' in anova_table.index else 0
        ms_error = anova_table.loc['Residual', 'MS'] if 'Residual' in anova_table.index else 0
        df_error = anova_table.loc['Residual', 'df'] if 'Residual' in anova_table.index else 0
        
        # Partial eta-squared and omega-squared
        anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_error)
        anova_table['partial_omega_sq'] = (anova_table['sum_sq'] - anova_table['df'] * ms_error) / (ss_total + ms_error)
        anova_table['partial_omega_sq'] = anova_table['partial_omega_sq'].clip(lower=0)  # Can't be negative
        
        # Rename index
        interaction_key = f'C(Q("{fa_clean}")):C(Q("{fb_clean}"))'
        cleaned_index = {
            f'C(Q("{fa_clean}"))': factor_a,
            f'C(Q("{fb_clean}"))': factor_b,
            interaction_key: f'{factor_a} * {factor_b}'
        }
        anova_table_renamed = anova_table.rename(index=cleaned_index)
        anova_df = anova_table_renamed.reset_index().rename(columns={'index': 'Source', 'PR(>F)': 'p_value'})
        anova_results = anova_df.replace({np.nan: None}).to_dict('records')
        
        # Convert numpy types
        for row in anova_results:
            for k, v in row.items():
                if isinstance(v, (np.integer, np.floating)):
                    row[k] = float(v) if not np.isnan(v) else None
                elif isinstance(v, np.bool_):
                    row[k] = bool(v)
        
        # Get p-values for each effect
        factor_a_p = None
        factor_b_p = None
        interaction_p = None
        factor_a_significant = False
        factor_b_significant = False
        interaction_significant = False
        
        for row in anova_results:
            source = row.get('Source')
            p = row.get('p_value')
            if source == factor_a:
                factor_a_p = p
                factor_a_significant = p is not None and p < alpha
            elif source == factor_b:
                factor_b_p = p
                factor_b_significant = p is not None and p < alpha
            elif '*' in str(source):
                interaction_p = p
                interaction_significant = p is not None and p < alpha
        
        # Cell descriptive stats with 95% CI
        grouped = clean_data_renamed.groupby([fa_clean, fb_clean])
        cell_stats = {}
        means_dict = {}
        stds_dict = {}
        
        for (fa_val, fb_val), group in grouped:
            cell_key = f"{fa_val} * {fb_val}"
            data_values = group[dv_clean]
            n = len(data_values)
            mean = float(data_values.mean())
            std = float(data_values.std(ddof=1)) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 0 else 0.0
            
            # 95% CI
            if n > 1:
                t_crit = t_dist.ppf(0.975, n - 1)
                ci_lower = mean - t_crit * se
                ci_upper = mean + t_crit * se
            else:
                ci_lower = mean
                ci_upper = mean
            
            cell_stats[cell_key] = {
                'n': int(n),
                'mean': mean,
                'std': std,
                'se': float(se),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'min': float(data_values.min()),
                'max': float(data_values.max())
            }
            
            # For backward compatibility
            if str(fa_val) not in means_dict:
                means_dict[str(fa_val)] = {}
                stds_dict[str(fa_val)] = {}
            means_dict[str(fa_val)][str(fb_val)] = mean
            stds_dict[str(fa_val)][str(fb_val)] = std
        
        # Row means
        for fa_val in means_dict.keys():
            row_values = list(means_dict[fa_val].values())
            means_dict[fa_val]['Row Mean'] = float(np.mean(row_values))
        
        # Column means
        fb_levels = list(means_dict[list(means_dict.keys())[0]].keys())
        fb_levels = [k for k in fb_levels if k != 'Row Mean']
        means_dict['Column Mean'] = {}
        stds_dict['Column Mean'] = {}
        for fb_val in fb_levels:
            col_values = [means_dict[fa_val][fb_val] for fa_val in means_dict.keys() if fa_val != 'Column Mean']
            means_dict['Column Mean'][fb_val] = float(np.mean(col_values))
            stds_dict['Column Mean'][fb_val] = 0.0
        
        all_means = [means_dict[fa][fb] for fa in means_dict.keys() if fa != 'Column Mean' for fb in means_dict[fa].keys() if fb != 'Row Mean']
        means_dict['Column Mean']['Row Mean'] = float(np.mean(all_means))
        stds_dict['Column Mean']['Row Mean'] = 0.0
        
        # Marginal means with CI
        def calc_marginal_means(data, group_col, value_col):
            result = []
            for level in data[group_col].unique():
                subset = data[data[group_col] == level][value_col]
                n = len(subset)
                mean = subset.mean()
                std = subset.std(ddof=1) if n > 1 else 0.0
                se = std / np.sqrt(n) if n > 0 else 0.0
                
                if n > 1:
                    t_crit = t_dist.ppf(0.975, n - 1)
                    ci_lower = mean - t_crit * se
                    ci_upper = mean + t_crit * se
                else:
                    ci_lower = mean
                    ci_upper = mean
                
                result.append({
                    'group': str(level),
                    'n': int(n),
                    'mean': float(mean),
                    'std': float(std),
                    'se': float(se),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper)
                })
            return result
        
        means_a = calc_marginal_means(clean_data_renamed, fa_clean, dv_clean)
        means_b = calc_marginal_means(clean_data_renamed, fb_clean, dv_clean)
        
        # Normality tests
        normality = {}
        for name, group in grouped:
            group_name = " * ".join(map(str, name))
            data = group[dv_clean]
            if len(data) >= 3:
                stat, p = stats.shapiro(data)
                normality[group_name] = {'statistic': float(stat), 'p_value': float(p), 'normal': bool(p > alpha)}
            else:
                normality[group_name] = {'statistic': None, 'p_value': None, 'normal': None}
        
        # Homogeneity tests
        samples = [group[dv_clean].values for name, group in grouped]
        homogeneity = {}
        
        if len(samples) >= 2:
            # Levene's test (mean-based)
            lev_stat, lev_p = stats.levene(*samples, center='mean')
            k = len(samples)
            n = sum(len(s) for s in samples)
            homogeneity['levene'] = {
                'statistic': float(lev_stat),
                'df1': int(k - 1),
                'df2': int(n - k),
                'p_value': float(lev_p),
                'equal_variances': bool(lev_p > alpha)
            }
            
            # Brown-Forsythe test (median-based, more robust)
            bf_stat, bf_p = stats.levene(*samples, center='median')
            homogeneity['brown_forsythe'] = {
                'statistic': float(bf_stat),
                'df1': int(k - 1),
                'df2': int(n - k),
                'p_value': float(bf_p),
                'equal_variances': bool(bf_p > alpha)
            }
        else:
            homogeneity['levene'] = {'statistic': None, 'p_value': None, 'equal_variances': None}
            homogeneity['brown_forsythe'] = {'statistic': None, 'p_value': None, 'equal_variances': None}
        
        # Determine if variances are equal (use Brown-Forsythe as primary)
        variances_equal = homogeneity['brown_forsythe'].get('equal_variances', True)
        
        # Simple main effects (if interaction significant)
        simple_effects = None
        
        if interaction_significant:
            simple_effects = []
            
            # Simple effects of A at each level of B
            for fb_level in clean_data_renamed[fb_clean].unique():
                subset = clean_data_renamed[clean_data_renamed[fb_clean] == fb_level]
                groups = [subset[subset[fa_clean] == level][dv_clean].values for level in subset[fa_clean].unique()]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    grand_mean = subset[dv_clean].mean()
                    ss_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
                    ss_within = sum([np.sum((g - np.mean(g))**2) for g in groups])
                    ss_total_simple = ss_between + ss_within
                    eta_sq = ss_between / ss_total_simple if ss_total_simple > 0 else 0
                    
                    # Omega-squared
                    df_between = len(groups) - 1
                    n_total = sum(len(g) for g in groups)
                    ms_within = ss_within / (n_total - len(groups)) if n_total > len(groups) else 0
                    omega_sq = (ss_between - df_between * ms_within) / (ss_total_simple + ms_within)
                    omega_sq = max(0, omega_sq)
                    
                    simple_effects.append({
                        'effect': f'{factor_a} at {factor_b}={fb_level}',
                        'factor_varied': factor_a,
                        'factor_fixed': factor_b,
                        'fixed_level': str(fb_level),
                        'f_statistic': float(f_stat),
                        'p_value': float(p_val),
                        'eta_squared': float(eta_sq),
                        'omega_squared': float(omega_sq),
                        'significant': bool(p_val < alpha)
                    })
            
            # Simple effects of B at each level of A
            for fa_level in clean_data_renamed[fa_clean].unique():
                subset = clean_data_renamed[clean_data_renamed[fa_clean] == fa_level]
                groups = [subset[subset[fb_clean] == level][dv_clean].values for level in subset[fb_clean].unique()]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    grand_mean = subset[dv_clean].mean()
                    ss_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
                    ss_within = sum([np.sum((g - np.mean(g))**2) for g in groups])
                    ss_total_simple = ss_between + ss_within
                    eta_sq = ss_between / ss_total_simple if ss_total_simple > 0 else 0
                    
                    # Omega-squared
                    df_between = len(groups) - 1
                    n_total = sum(len(g) for g in groups)
                    ms_within = ss_within / (n_total - len(groups)) if n_total > len(groups) else 0
                    omega_sq = (ss_between - df_between * ms_within) / (ss_total_simple + ms_within)
                    omega_sq = max(0, omega_sq)
                    
                    simple_effects.append({
                        'effect': f'{factor_b} at {factor_a}={fa_level}',
                        'factor_varied': factor_b,
                        'factor_fixed': factor_a,
                        'fixed_level': str(fa_level),
                        'f_statistic': float(f_stat),
                        'p_value': float(p_val),
                        'eta_squared': float(eta_sq),
                        'omega_squared': float(omega_sq),
                        'significant': bool(p_val < alpha)
                    })
        
        # Post-hoc tests
        posthoc = {
            'interaction': None,
            'factor_a': None,
            'factor_b': None,
            'recommended': 'tukey' if variances_equal else 'games_howell'
        }
        
        # Interaction post-hoc (if interaction significant)
        if interaction_significant:
            clean_data_renamed['combined_group'] = clean_data_renamed[fa_clean].astype(str) + " * " + clean_data_renamed[fb_clean].astype(str)
            
            # Tukey HSD
            tukey = pairwise_tukeyhsd(clean_data_renamed[dv_clean], clean_data_renamed['combined_group'], alpha=alpha)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            tukey_results = []
            for _, row in tukey_df.iterrows():
                tukey_results.append({
                    'group1': str(row['group1']),
                    'group2': str(row['group2']),
                    'meandiff': float(row['meandiff']),
                    'p_adj': float(row['p-adj']),
                    'lower': float(row['lower']),
                    'upper': float(row['upper']),
                    'reject': bool(row['reject'])
                })
            
            # Games-Howell
            games_howell_results = games_howell_test(clean_data_renamed, 'combined_group', dv_clean, alpha)
            
            posthoc['interaction'] = {
                'tukey': tukey_results,
                'games_howell': games_howell_results
            }
        
        # Main effect post-hoc for Factor A (if significant and > 2 levels)
        fa_levels = clean_data_renamed[fa_clean].nunique()
        if factor_a_significant and fa_levels > 2:
            # Tukey HSD
            tukey_a = pairwise_tukeyhsd(clean_data_renamed[dv_clean], clean_data_renamed[fa_clean], alpha=alpha)
            tukey_a_df = pd.DataFrame(data=tukey_a._results_table.data[1:], columns=tukey_a._results_table.data[0])
            tukey_a_results = []
            for _, row in tukey_a_df.iterrows():
                tukey_a_results.append({
                    'group1': str(row['group1']),
                    'group2': str(row['group2']),
                    'meandiff': float(row['meandiff']),
                    'p_adj': float(row['p-adj']),
                    'lower': float(row['lower']),
                    'upper': float(row['upper']),
                    'reject': bool(row['reject'])
                })
            
            # Games-Howell
            games_howell_a = games_howell_test(clean_data_renamed, fa_clean, dv_clean, alpha)
            
            posthoc['factor_a'] = {
                'tukey': tukey_a_results,
                'games_howell': games_howell_a
            }
        
        # Main effect post-hoc for Factor B (if significant and > 2 levels)
        fb_levels = clean_data_renamed[fb_clean].nunique()
        if factor_b_significant and fb_levels > 2:
            # Tukey HSD
            tukey_b = pairwise_tukeyhsd(clean_data_renamed[dv_clean], clean_data_renamed[fb_clean], alpha=alpha)
            tukey_b_df = pd.DataFrame(data=tukey_b._results_table.data[1:], columns=tukey_b._results_table.data[0])
            tukey_b_results = []
            for _, row in tukey_b_df.iterrows():
                tukey_b_results.append({
                    'group1': str(row['group1']),
                    'group2': str(row['group2']),
                    'meandiff': float(row['meandiff']),
                    'p_adj': float(row['p-adj']),
                    'lower': float(row['lower']),
                    'upper': float(row['upper']),
                    'reject': bool(row['reject'])
                })
            
            # Games-Howell
            games_howell_b = games_howell_test(clean_data_renamed, fb_clean, dv_clean, alpha)
            
            posthoc['factor_b'] = {
                'tukey': tukey_b_results,
                'games_howell': games_howell_b
            }
        
        # Interpretation
        residual_df = None
        fa_row, fb_row, ix_row = None, None, None
        for row in anova_results:
            src = row.get('Source', '')
            if src == 'Residual':
                residual_df = row.get('df')
            elif src == factor_a:
                fa_row = row
            elif src == factor_b:
                fb_row = row
            elif '*' in str(src):
                ix_row = row

        def effect_label(eta):
            if eta is None: return 'negligible'
            if eta >= 0.14: return 'large'
            if eta >= 0.06: return 'medium'
            if eta >= 0.01: return 'small'
            return 'negligible'

        def fmt_p(p):
            if p is None: return 'N/A'
            return 'p < .001' if p < 0.001 else f'p = {p:.3f}'

        # Marginal means summary
        ma_str = ', '.join(f"{lvl} (M = {v['mean']:.2f})" for lvl, v in means_a.items()) if means_a else 'N/A'
        mb_str = ', '.join(f"{lvl} (M = {v['mean']:.2f})" for lvl, v in means_b.items()) if means_b else 'N/A'

        # P1: Design & overview
        dropped_note = f" {n_dropped} row(s) were excluded due to missing data ({n_used} observations used)." if n_dropped > 0 else f" All {n_used} observations were included."
        interp_p1 = (
            f"A two-way analysis of variance (Type {ss_type} SS) was conducted to examine the independent and joint effects of "
            f"{factor_a} and {factor_b} on {dependent_var}.{dropped_note} "
            f"Marginal means for {factor_a}: {ma_str}. Marginal means for {factor_b}: {mb_str}."
        )

        # P2: Interaction effect
        if ix_row and ix_row.get('F') is not None:
            ix_sig = 'statistically significant' if interaction_significant else 'not statistically significant'
            ix_eta = ix_row.get('partial_eta_sq', 0) or 0
            ix_omega = ix_row.get('partial_omega_sq', 0) or 0
            interp_p2 = (
                f"The {factor_a} × {factor_b} interaction effect was {ix_sig}, "
                f"F({int(ix_row['df'])}, {int(residual_df)}) = {ix_row['F']:.2f}, {fmt_p(ix_row.get('p_value'))}, "
                f"η²p = {ix_eta:.3f}, ω²p = {ix_omega:.3f} ({effect_label(ix_eta)} effect). "
            )
            if interaction_significant:
                interp_p2 += (
                    f"A significant interaction indicates that the effect of {factor_a} on {dependent_var} "
                    f"varies depending on the level of {factor_b}. Main effects should be interpreted with caution — "
                    "simple effects analyses are recommended to decompose the interaction."
                )
            else:
                interp_p2 += (
                    f"A non-significant interaction indicates that the effects of {factor_a} and {factor_b} are additive; "
                    "each factor's main effect can be interpreted independently."
                )
        else:
            interp_p2 = ""

        # P3: Main effects
        parts = []
        if fa_row and fa_row.get('F') is not None:
            fa_eta = fa_row.get('partial_eta_sq', 0) or 0
            fa_omega = fa_row.get('partial_omega_sq', 0) or 0
            fa_sig = 'statistically significant' if factor_a_significant else 'not statistically significant'
            parts.append(
                f"The main effect of {factor_a} was {fa_sig}, "
                f"F({int(fa_row['df'])}, {int(residual_df)}) = {fa_row['F']:.2f}, {fmt_p(fa_row.get('p_value'))}, "
                f"η²p = {fa_eta:.3f}, ω²p = {fa_omega:.3f} ({effect_label(fa_eta)} effect). "
                f"This indicates that {factor_a} {'accounts for' if factor_a_significant else 'does not account for a meaningful proportion of'} "
                f"variance in {dependent_var} after controlling for other factors."
            )
        if fb_row and fb_row.get('F') is not None:
            fb_eta = fb_row.get('partial_eta_sq', 0) or 0
            fb_omega = fb_row.get('partial_omega_sq', 0) or 0
            fb_sig = 'statistically significant' if factor_b_significant else 'not statistically significant'
            parts.append(
                f"The main effect of {factor_b} was {fb_sig}, "
                f"F({int(fb_row['df'])}, {int(residual_df)}) = {fb_row['F']:.2f}, {fmt_p(fb_row.get('p_value'))}, "
                f"η²p = {fb_eta:.3f}, ω²p = {fb_omega:.3f} ({effect_label(fb_eta)} effect). "
                f"This indicates that {factor_b} {'accounts for' if factor_b_significant else 'does not account for a meaningful proportion of'} "
                f"variance in {dependent_var} after controlling for other factors."
            )
        interp_p3 = ' '.join(parts) if parts else ""

        # P4: Assumptions
        norm_issues = [g for g, v in normality.items() if v.get('normal') == False]
        if norm_issues:
            norm_text = (
                f"Shapiro-Wilk tests indicated non-normality in {len(norm_issues)} group(s) "
                f"({', '.join(norm_issues)}). ANOVA is generally robust to normality violations when sample sizes are adequate."
            )
        else:
            norm_text = "Shapiro-Wilk tests indicated no significant departure from normality across groups."

        if not variances_equal:
            hom_text = (
                f"The homogeneity of variances assumption was violated "
                f"(Brown-Forsythe p = {homogeneity['brown_forsythe']['p_value']:.3f} < .05). "
                "Games-Howell post-hoc tests are recommended over Tukey HSD."
            )
        else:
            hom_text = (
                f"Brown-Forsythe test confirmed homogeneity of variances "
                f"(p = {homogeneity['brown_forsythe']['p_value']:.3f} > .05), "
                "supporting the use of Tukey HSD post-hoc comparisons."
            )
        interp_p4 = norm_text + " " + hom_text

        # P5: Post-hoc
        interp_p5 = ""
        ph_name = "Games-Howell" if not variances_equal else "Tukey HSD"
        ph_parts = []
        for factor_key, factor_name in [('factor_a', factor_a), ('factor_b', factor_b)]:
            ph_data = posthoc.get(factor_key)
            if ph_data:
                rows_ph = ph_data.get('tukey') or ph_data.get('games_howell') or []
                sig_pairs = [r for r in rows_ph if r.get('reject')]
                if sig_pairs:
                    def _fmt_p(p):
                        return '< .001' if p < 0.001 else f'= {p:.3f}'
                    pairs_str = '; '.join(
                        f"{r['group1']} vs {r['group2']} (Δ = {r['meandiff']:+.2f}, p-adj {_fmt_p(r['p_adj'])})"
                        for r in sig_pairs[:5]
                    )
                    ph_parts.append(f"{factor_name} ({ph_name}): {pairs_str}.")
        if ph_parts:
            interp_p5 = "Post-hoc pairwise comparisons revealed the following significant differences: " + " ".join(ph_parts)
        elif factor_a_significant or factor_b_significant:
            interp_p5 = f"Post-hoc comparisons ({ph_name}) did not identify significant individual pairs after correction, despite a significant omnibus F-test."

        interpretation = "\n\n".join(filter(None, [interp_p1, interp_p2, interp_p3, interp_p4, interp_p5]))
        
        # Plot
        plot = generate_plot(clean_data_renamed, fa_clean, fb_clean, dv_clean, factor_a, factor_b, dependent_var, model)
        
        return {
            "results": {
                "anova_table": anova_results,
                "ss_type": ss_type,
                "cell_descriptives": cell_stats,
                "descriptive_stats_table": {"mean": means_dict, "std": stds_dict},
                "marginal_means": {
                    "factor_a": means_a,
                    "factor_b": means_b
                },
                "assumptions": {
                    "normality": normality,
                    "homogeneity": homogeneity
                },
                "simple_main_effects": simple_effects,
                "posthoc": posthoc,
                "significance": {
                    "factor_a": factor_a_significant,
                    "factor_b": factor_b_significant,
                    "interaction": interaction_significant
                },
                "interpretation": interpretation,
                "dropped_rows": dropped_rows,
                "n_dropped": n_dropped,
                "n_used": n_used,
                "n_original": original_len
            },
            "plot": plot
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
