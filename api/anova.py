from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class AnovaRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    independentVar: str = Field(..., description="Grouping variable")
    dependentVar: str = Field(..., description="Numeric variable")


def games_howell_test(data, group_col, value_col):
    """Games-Howell post-hoc test for unequal variances"""
    groups = data[group_col].unique()
    n_groups = len(groups)
    
    # Get group statistics
    group_stats = {}
    for g in groups:
        group_data = data[data[group_col] == g][value_col]
        group_stats[g] = {
            'mean': group_data.mean(),
            'var': group_data.var(ddof=1),
            'n': len(group_data)
        }
    
    results = []
    comparisons = []
    
    for i, g1 in enumerate(groups):
        for g2 in list(groups)[i+1:]:
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
        
        # p-value (two-tailed, adjusted for multiple comparisons using studentized range)
        # Approximation using t-distribution
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Bonferroni-style adjustment (simplified)
        n_comparisons = len(comparisons)
        p_adj = min(p_value * n_comparisons, 1.0)
        
        # 95% CI
        t_crit = stats.t.ppf(0.975, df)
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
            'reject': bool(p_adj < 0.05)
        })
    
    return results


def welch_anova(group_data, groups):
    """Welch's ANOVA for unequal variances"""
    k = len(groups)
    
    # Group statistics
    ns = np.array([len(group_data[g]) for g in groups])
    means = np.array([np.mean(group_data[g]) for g in groups])
    variances = np.array([np.var(group_data[g], ddof=1) for g in groups])
    
    # Weights
    weights = ns / variances
    sum_weights = np.sum(weights)
    
    # Weighted grand mean
    weighted_mean = np.sum(weights * means) / sum_weights
    
    # Welch's F statistic
    numerator = np.sum(weights * (means - weighted_mean)**2) / (k - 1)
    
    # Lambda values for denominator
    lambdas = (1 - weights / sum_weights)**2 / (ns - 1)
    denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * np.sum(lambdas)
    
    f_stat = numerator / denominator
    
    # Degrees of freedom
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * np.sum(lambdas))
    
    # p-value
    p_value = float(1 - stats.f.cdf(f_stat, df1, df2))
    
    return {
        'f_statistic': float(f_stat),
        'df1': float(df1),
        'df2': float(df2),
        'p_value': p_value,
        'significant': bool(p_value < 0.05)
    }


def generate_plot(clean_data, group_col, value_col, groups, descriptives, group_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('One-Way ANOVA Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Means with error bars (95% CI)
    means = [descriptives[g]['mean'] for g in groups]
    ci_lowers = [descriptives[g]['ci_lower'] for g in groups]
    ci_uppers = [descriptives[g]['ci_upper'] for g in groups]
    
    # Calculate error bar lengths
    yerr_lower = [means[i] - ci_lowers[i] for i in range(len(groups))]
    yerr_upper = [ci_uppers[i] - means[i] for i in range(len(groups))]
    
    axes[0, 0].errorbar(range(len(groups)), means, yerr=[yerr_lower, yerr_upper],
                        marker='o', markersize=8, linestyle='-', linewidth=2, capsize=5, capthick=2)
    axes[0, 0].set_title('Group Means with 95% CI')
    axes[0, 0].set_xticks(range(len(groups)))
    axes[0, 0].set_xticklabels(groups)
    axes[0, 0].set_xlabel(group_col)
    axes[0, 0].set_ylabel(f'Mean of {value_col}')
    
    # Plot 2: Boxplot
    sns.boxplot(x=group_col, y=value_col, data=clean_data, ax=axes[0, 1], palette='crest')
    axes[0, 1].set_title(f'Distribution by {group_col}')
    
    # Plot 3: Violin plot
    sns.violinplot(x=group_col, y=value_col, data=clean_data, ax=axes[1, 0], palette='crest')
    axes[1, 0].set_title(f'Violin Plot by {group_col}')
    
    # Plot 4: Q-Q plot
    all_residuals = []
    for g in groups:
        gdata = group_data[g]
        gmean = descriptives[str(g)]['mean']
        all_residuals.extend(gdata - gmean)
    sm.qqplot(np.array(all_residuals), line='s', ax=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/anova")
def one_way_anova(req: AnovaRequest):
    try:
        df = pd.DataFrame(req.data)
        group_col = req.independentVar
        value_col = req.dependentVar
        
        if group_col not in df.columns or value_col not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names")
        
        # Clean data
        original_len = len(df)
        clean_data = df[[group_col, value_col]].dropna()
        clean_data[value_col] = pd.to_numeric(clean_data[value_col], errors='coerce')
        clean_data = clean_data.dropna()
        
        dropped_rows = list(set(range(original_len)) - set(clean_data.index.tolist()))
        n_dropped = len(dropped_rows)
        
        groups = sorted(clean_data[group_col].unique())
        k = len(groups)
        
        if k < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 groups")
        
        group_data = {g: clean_data[clean_data[group_col] == g][value_col].values for g in groups}
        values = clean_data[value_col].values
        n_total = len(values)
        
        # Descriptives with 95% CI
        descriptives = {}
        for g in groups:
            gd = group_data[g]
            n = len(gd)
            mean = float(np.mean(gd))
            std = float(np.std(gd, ddof=1)) if n > 1 else 0.0
            se = float(stats.sem(gd)) if n > 0 else 0.0
            
            # 95% Confidence Interval
            if n > 1:
                t_crit = t_dist.ppf(0.975, n - 1)
                ci_lower = mean - t_crit * se
                ci_upper = mean + t_crit * se
            else:
                ci_lower = mean
                ci_upper = mean
            
            descriptives[str(g)] = {
                'n': int(n),
                'mean': mean,
                'std': std,
                'se': se,
                'min': float(np.min(gd)),
                'max': float(np.max(gd)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            }
        
        # ANOVA calculation
        grand_mean = np.mean(values)
        ssb = sum(len(group_data[g]) * (np.mean(group_data[g]) - grand_mean)**2 for g in groups)
        ssw = sum(np.sum((group_data[g] - np.mean(group_data[g]))**2) for g in groups)
        sst = ssb + ssw
        
        df_between = k - 1
        df_within = n_total - k
        
        msb = ssb / df_between if df_between > 0 else 0
        msw = ssw / df_within if df_within > 0 else 0
        
        f_stat = msb / msw if msw > 0 else float('inf')
        p_value = float(1 - stats.f.cdf(f_stat, df_between, df_within))
        
        # Effect sizes
        eta_squared = float(ssb / sst) if sst > 0 else 0
        
        # Omega-squared (less biased than eta-squared)
        omega_squared = float((ssb - (df_between * msw)) / (sst + msw)) if (sst + msw) > 0 else 0
        omega_squared = max(0, omega_squared)  # Can't be negative

        # 95% CI for eta_squared via non-central F distribution (Smithson 2003)
        def eta_sq_ci(f, df1, df2, conf=0.95):
            from scipy.optimize import brentq
            alpha = 1 - conf
            n_total = df1 + df2 + 1
            p_central = float(stats.f.sf(f, df1, df2))
            def ncf_sf(ncp):
                if ncp <= 1e-9:
                    return p_central
                return float(1 - stats.ncf.cdf(f, df1, df2, ncp))
            upper_bound = max(f * df1 * 200, 2000)
            # eta_ci_lower: ncp where ncf_sf = alpha/2  (small ncp → small eta)
            try:
                if p_central >= alpha / 2:
                    ncp_for_lower = 0.0
                else:
                    ncp_for_lower = brentq(lambda ncp: ncf_sf(ncp) - alpha / 2,
                                           1e-9, upper_bound, maxiter=500)
            except Exception:
                ncp_for_lower = 0.0
            # eta_ci_upper: ncp where ncf_sf = 1 - alpha/2  (large ncp → large eta)
            try:
                if ncf_sf(upper_bound) < (1 - alpha / 2):
                    ncp_for_upper = upper_bound
                else:
                    ncp_for_upper = brentq(lambda ncp: ncf_sf(ncp) - (1 - alpha / 2),
                                           1e-9, upper_bound, maxiter=500)
            except Exception:
                ncp_for_upper = upper_bound
            eta_ci_lower = ncp_for_lower / (ncp_for_lower + n_total) if ncp_for_lower > 0 else 0.0
            eta_ci_upper = ncp_for_upper / (ncp_for_upper + n_total)
            return max(0.0, eta_ci_lower), min(1.0, eta_ci_upper)

        if f_stat > 0 and not np.isinf(f_stat):
            eta_sq_ci_lower, eta_sq_ci_upper = eta_sq_ci(f_stat, df_between, df_within)
        else:
            eta_sq_ci_lower, eta_sq_ci_upper = 0.0, 0.0

        # 95% CI for omega_squared — transform from eta² CI bounds
        n_total_ci = df_between + df_within + 1
        def eta_to_omega_ci(eta, df1):
            denom = n_total_ci - 1 + (1 - eta)
            return max(0.0, (eta * (n_total_ci - 1) - df1) / denom) if denom > 0 else 0.0

        omega_sq_ci_lower = eta_to_omega_ci(eta_sq_ci_lower, df_between)
        omega_sq_ci_upper = eta_to_omega_ci(eta_sq_ci_upper, df_between)

        significant = bool(p_value < 0.05)
        
        # Normality tests (Shapiro-Wilk)
        normality = {}
        for g in groups:
            gd = group_data[g]
            if len(gd) >= 3:
                stat, p = stats.shapiro(gd)
                normality[str(g)] = {'statistic': float(stat), 'p_value': float(p), 'normal': bool(p > 0.05)}
            else:
                normality[str(g)] = {'statistic': None, 'p_value': None, 'normal': None}
        
        # Homogeneity tests
        group_arrays = [group_data[g] for g in groups if len(group_data[g]) > 0]
        
        # Levene's test (mean-based)
        if len(group_arrays) >= 2:
            lev_stat, lev_p = stats.levene(*group_arrays, center='mean')
            levene_test = {
                'statistic': float(lev_stat), 
                'p_value': float(lev_p), 
                'equal_variances': bool(lev_p > 0.05)
            }
        else:
            levene_test = {'statistic': None, 'p_value': None, 'equal_variances': None}
        
        # Brown-Forsythe test (median-based, more robust)
        if len(group_arrays) >= 2:
            bf_stat, bf_p = stats.levene(*group_arrays, center='median')
            brown_forsythe = {
                'statistic': float(bf_stat), 
                'p_value': float(bf_p), 
                'equal_variances': bool(bf_p > 0.05)
            }
        else:
            brown_forsythe = {'statistic': None, 'p_value': None, 'equal_variances': None}
        
        homogeneity = {
            'levene': levene_test,
            'brown_forsythe': brown_forsythe
        }
        
        # Determine if variances are equal (use Brown-Forsythe as primary)
        variances_equal = brown_forsythe.get('equal_variances', True)
        
        # Welch's ANOVA (robust to unequal variances)
        welch_results = welch_anova(group_data, groups)
        
        # Post-hoc tests
        post_hoc_tukey = None
        post_hoc_games_howell = None
        
        if significant and k > 2:
            # Tukey HSD (assumes equal variances)
            tukey = pairwise_tukeyhsd(clean_data[value_col], clean_data[group_col], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            post_hoc_tukey = []
            for _, row in tukey_df.iterrows():
                post_hoc_tukey.append({
                    'group1': str(row['group1']),
                    'group2': str(row['group2']),
                    'meandiff': float(row['meandiff']),
                    'p_adj': float(row['p-adj']),
                    'lower': float(row['lower']),
                    'upper': float(row['upper']),
                    'reject': bool(row['reject'])
                })
            
            # Games-Howell (does not assume equal variances)
            post_hoc_games_howell = games_howell_test(clean_data, group_col, value_col)
        
        # Interpretation
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        sig_text = "statistically significant" if significant else "not statistically significant"

        if eta_squared < 0.01:
            effect_text = "negligible"
        elif eta_squared < 0.06:
            effect_text = "small"
        elif eta_squared < 0.14:
            effect_text = "medium"
        else:
            effect_text = "large"

        # Group means summary for interpretation
        group_means_str = ", ".join(
            f"{g} (M = {descriptives[str(g)]['mean']:.2f}, SD = {descriptives[str(g)]['std']:.2f}, n = {descriptives[str(g)]['n']})"
            for g in groups
        )
        highest_group = max(groups, key=lambda g: descriptives[str(g)]['mean'])
        lowest_group  = min(groups, key=lambda g: descriptives[str(g)]['mean'])
        mean_range = descriptives[str(highest_group)]['mean'] - descriptives[str(lowest_group)]['mean']

        # Paragraph 1: design & main result
        interp_p1 = (
            f"A one-way ANOVA was conducted to examine the effect of {group_col} on {value_col}. "
            f"The analysis included {n_total} observations across {k} groups ({group_means_str}). "
            f"The omnibus F-test was {sig_text}, "
            f"F({df_between}, {df_within}) = {f_stat:.2f}, {p_text}, "
            f"η² = {eta_squared:.3f}, ω² = {omega_squared:.3f}."
        )

        # Paragraph 2: effect size
        interp_p2 = (
            f"The effect size was {effect_text} (η² = {eta_squared:.3f}), indicating that group membership "
            f"accounts for {eta_squared * 100:.1f}% of the total variance in {value_col}. "
            f"The bias-corrected estimate ω² = {omega_squared:.3f} is preferred for smaller samples. "
            f"The highest mean was observed in {highest_group} (M = {descriptives[str(highest_group)]['mean']:.2f}) "
            f"and the lowest in {lowest_group} (M = {descriptives[str(lowest_group)]['mean']:.2f}), "
            f"a difference of {mean_range:.2f} units."
        )

        # Paragraph 3: assumptions
        normality_issues = [g for g in groups if normality.get(str(g), {}).get('normal') == False]
        if normality_issues:
            norm_text = (
                f"Normality assumption was violated for {len(normality_issues)} group(s) "
                f"({', '.join(str(g) for g in normality_issues)}) based on Shapiro-Wilk tests. "
                "ANOVA is robust to normality violations when group sizes are sufficiently large (n > 30 per group)."
            )
        else:
            norm_text = "Shapiro-Wilk tests indicated no significant departure from normality in any group."

        if not variances_equal:
            welch_p_str = '< .001' if welch_results['p_value'] < 0.001 else f"{welch_results['p_value']:.3f}"
            hom_text = (
                f"Levene's test indicated unequal variances (Brown-Forsythe p = {brown_forsythe['p_value']:.3f}). "
                f"Welch's ANOVA (F = {welch_results['f_statistic']:.2f}, p = {welch_p_str}) "
                "and Games-Howell post-hoc tests are recommended."
            )
        else:
            hom_text = (
                f"Brown-Forsythe test confirmed homogeneity of variances (p = {brown_forsythe['p_value']:.3f}), "
                "supporting the use of standard ANOVA and Tukey HSD post-hoc comparisons."
            )
        interp_p3 = norm_text + " " + hom_text

        # Paragraph 4: post-hoc (only if significant)
        interp_p4 = ""
        if significant and post_hoc_tukey:
            sig_pairs = [r for r in post_hoc_tukey if r['reject']]
            ns_pairs  = [r for r in post_hoc_tukey if not r['reject']]
            recommended_ph = post_hoc_games_howell if not variances_equal else post_hoc_tukey
            ph_name = "Games-Howell" if not variances_equal else "Tukey HSD"
            if sig_pairs:
                def _p_fmt(p):
                    return '< .001' if p < 0.001 else f'= {p:.3f}'
                pairs_str = "; ".join(
                    f"{r['group1']} vs {r['group2']} (Δ = {r['meandiff']:+.2f}, p-adj {_p_fmt(r['p_adj'])})"
                    for r in sig_pairs[:5]
                )
                interp_p4 = (
                    f"Post-hoc comparisons ({ph_name}) revealed {len(sig_pairs)} significant pair(s): {pairs_str}. "
                    f"{len(ns_pairs)} comparison(s) did not reach significance after correction."
                )
            else:
                interp_p4 = (
                    f"Despite the significant omnibus F-test, {ph_name} post-hoc comparisons "
                    "did not identify any individual pairs as significantly different after correction for multiple comparisons."
                )

        interpretation = "\n\n".join(filter(None, [interp_p1, interp_p2, interp_p3, interp_p4]))
        
        # Plot
        plot = generate_plot(clean_data, group_col, value_col, groups, {str(g): descriptives[str(g)] for g in groups}, group_data)
        
        return {
            "results": {
                "anova": {
                    "f_statistic": float(f_stat),
                    "p_value": p_value,
                    "eta_squared": eta_squared,
                    "eta_squared_ci": [round(eta_sq_ci_lower, 4), round(eta_sq_ci_upper, 4)],
                    "omega_squared": omega_squared,
                    "omega_squared_ci": [round(omega_sq_ci_lower, 4), round(omega_sq_ci_upper, 4)],
                    "df_between": int(df_between),
                    "df_within": int(df_within),
                    "df_total": int(df_between + df_within),
                    "ssb": float(ssb),
                    "ssw": float(ssw),
                    "sst": float(sst),
                    "msb": float(msb),
                    "msw": float(msw),
                    "significant": significant
                },
                "welch_anova": welch_results,
                "descriptives": descriptives,
                "assumptions": {
                    "normality": normality,
                    "homogeneity": homogeneity
                },
                "post_hoc": {
                    "tukey": post_hoc_tukey,
                    "games_howell": post_hoc_games_howell,
                    "recommended": "games_howell" if not variances_equal else "tukey"
                },
                "interpretation": interpretation,
                "dropped_rows": dropped_rows,
                "n_dropped": n_dropped
            },
            "plot": plot
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
