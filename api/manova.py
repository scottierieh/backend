from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as f_dist, chi2
from scipy.linalg import inv, det
from statsmodels.stats.multitest import multipletests
import itertools
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class ManovaRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    dependentVars: List[str] = Field(..., description="List of dependent variables")
    factorVars: List[str] = Field(..., description="List of factor variables")


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def calculate_boxs_m_test(clean_data, factor, groups, dependent_vars):
    """
    Box's M Test for equality of covariance matrices.
    Tests the assumption that covariance matrices are equal across groups.
    """
    try:
        k = len(groups)
        p = len(dependent_vars)
        n_total = len(clean_data)
        
        # Calculate pooled covariance matrix and group covariance matrices
        group_covs = []
        group_ns = []
        
        for g in groups:
            group_data = clean_data[clean_data[factor] == g][dependent_vars].values
            n_g = len(group_data)
            if n_g > p:  # Need more observations than variables
                cov_g = np.cov(group_data.T, ddof=1)
                group_covs.append(cov_g)
                group_ns.append(n_g)
            else:
                return None  # Not enough data for Box's M
        
        if len(group_covs) < k:
            return None
        
        # Pooled covariance matrix
        S_pooled = np.zeros((p, p))
        df_total = 0
        for cov, n in zip(group_covs, group_ns):
            S_pooled += (n - 1) * cov
            df_total += (n - 1)
        S_pooled /= df_total
        
        # Box's M statistic
        M = 0
        for cov, n in zip(group_covs, group_ns):
            det_cov = det(cov)
            det_pooled = det(S_pooled)
            if det_cov > 0 and det_pooled > 0:
                M += (n - 1) * np.log(det_pooled / det_cov)
        
        # Correction factor
        sum_inv = sum(1 / (n - 1) for n in group_ns)
        c = (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (k - 1)) * (sum_inv - 1 / df_total)
        
        # Chi-square approximation
        chi2_stat = M * (1 - c)
        df = p * (p + 1) * (k - 1) / 2
        p_value = 1 - chi2.cdf(chi2_stat, df)
        
        return {
            "M": float(M),
            "chi2": float(chi2_stat),
            "df": float(df),
            "p_value": float(p_value),
            "equal_covariances": bool(p_value > 0.05)
        }
    except Exception:
        return None


def calculate_descriptives(clean_data, factor, groups, dependent_vars):
    """
    Calculate descriptive statistics for each group and DV.
    Returns: n, mean, std, se, min, max, 95% CI
    """
    descriptives = {}
    
    for dv in dependent_vars:
        dv_stats = {}
        for g in groups:
            group_data = clean_data[clean_data[factor] == g][dv].values
            n = len(group_data)
            mean = float(np.mean(group_data))
            std = float(np.std(group_data, ddof=1))
            se = std / np.sqrt(n) if n > 0 else 0
            
            # 95% CI
            if n > 1:
                t_crit = stats.t.ppf(0.975, n - 1)
                ci_lower = mean - t_crit * se
                ci_upper = mean + t_crit * se
            else:
                ci_lower = ci_upper = mean
            
            dv_stats[str(g)] = {
                "n": int(n),
                "mean": mean,
                "std": std,
                "se": float(se),
                "min": float(np.min(group_data)),
                "max": float(np.max(group_data)),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper)
            }
        descriptives[dv] = dv_stats
    
    return descriptives


def generate_plot(clean_data, factor, groups, dependent_vars, test_stats, alpha):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = sns.color_palette('crest', n_colors=len(groups))
    
    # 1. Group means
    group_means = np.array([clean_data[clean_data[factor] == g][dependent_vars].mean() for g in groups])
    x = np.arange(len(dependent_vars))
    width = 0.8 / len(groups)
    
    for i, group in enumerate(groups):
        offset = (i - len(groups)/2 + 0.5) * width
        axes[0, 0].bar(x + offset, group_means[i], width, label=str(group), color=colors[i], alpha=0.8)
    
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dependent_vars, rotation=45, ha='right')
    axes[0, 0].set_title('Group Means', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    
    # 2-3. Box plots
    for i, dv in enumerate(dependent_vars[:2]):
        ax = axes[0, 1] if i == 0 else axes[1, 0]
        sns.boxplot(x=factor, y=dv, data=clean_data, ax=ax, palette=colors)
        ax.set_title(f'Distribution of {dv}', fontsize=12, fontweight='bold')
    
    # 4. P-values
    test_names = list(test_stats.keys())
    p_values = [test_stats[t]['p_value'] for t in test_names]
    axes[1, 1].bar(test_names, p_values, color='#5B9BD5')
    axes[1, 1].axhline(y=alpha, color='r', linestyle='--', label=f'α={alpha}')
    axes[1, 1].set_title('Multivariate Test p-values', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/manova")
def manova_analysis(req: ManovaRequest):
    try:
        df = pd.DataFrame(req.data)
        dependent_vars = req.dependentVars
        factor_vars = req.factorVars
        alpha = 0.05
        
        factor = factor_vars[0]
        all_vars = dependent_vars + factor_vars
        
        # Clean data
        original_len = len(df)
        clean_data = df[all_vars].dropna().copy()
        dropped_rows = list(set(range(original_len)) - set(clean_data.index.tolist()))
        
        groups = sorted(clean_data[factor].unique())
        k = len(groups)
        if k < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 groups")
        
        Y = clean_data[dependent_vars].values
        n = len(Y)
        p = len(dependent_vars)
        
        # Calculate matrices
        group_data = [Y[clean_data[factor] == g] for g in groups]
        group_means = np.array([d.mean(axis=0) for d in group_data])
        overall_mean = Y.mean(axis=0)
        
        T = np.zeros((p, p))
        for i in range(n):
            dev = (Y[i] - overall_mean).reshape(-1, 1)
            T += dev @ dev.T
        
        H = np.zeros((p, p))
        for i, (g, gd) in enumerate(zip(groups, group_data)):
            dev = (group_means[i] - overall_mean).reshape(-1, 1)
            H += len(gd) * (dev @ dev.T)
        
        E = T - H
        df_between = k - 1
        df_within = n - k
        
        # Test statistics
        try:
            E_inv_H = inv(E) @ H
            eigenvals = np.real(np.linalg.eigvals(E_inv_H))
            eigenvals = eigenvals[eigenvals > 1e-10]
        except:
            eigenvals = np.array([])
        
        s = min(p, df_between)
        m = (abs(p - df_between) - 1) / 2
        n_prime = (df_within - p - 1) / 2
        
        # Pillai
        V = np.sum(eigenvals / (1 + eigenvals)) if len(eigenvals) > 0 else 0
        F_pillai = (2*n_prime + s + 1) / (2*m + s + 1) * V / (s - V) if (s-V) > 0 else 0
        df1_pillai = s * (2*m + s + 1)
        df2_pillai = s * (2*n_prime + s + 1)
        p_pillai = float(1 - f_dist.cdf(F_pillai, df1_pillai, df2_pillai)) if df1_pillai > 0 and df2_pillai > 0 else 1.0
        
        # Wilks
        L = np.prod(1 / (1 + eigenvals)) if len(eigenvals) > 0 else 1.0
        w = df_within - 0.5 * (p - df_between + 1)
        t = np.sqrt((p**2 * df_between**2 - 4) / (p**2 + df_between**2 - 5)) if (p**2 + df_between**2 - 5) > 0 else 1
        df1_wilks = p * df_between
        df2_wilks = w * t - 0.5 * (p * df_between - 2)
        F_wilks = ((1 - L**(1/t)) / (L**(1/t))) * (df2_wilks / df1_wilks) if L > 0 else 0
        p_wilks = float(1 - f_dist.cdf(F_wilks, df1_wilks, df2_wilks)) if df1_wilks > 0 and df2_wilks > 0 else 1.0
        
        # Hotelling
        T_hl = np.sum(eigenvals)
        F_hotelling = T_hl * (2*n_prime + s + 1) / (s * (2*m + s + 1)) if s > 0 else 0
        p_hotelling = float(1 - f_dist.cdf(F_hotelling, df1_pillai, df2_pillai)) if df1_pillai > 0 and df2_pillai > 0 else 1.0
        
        # Roy
        theta = float(np.max(eigenvals)) if len(eigenvals) > 0 else 0
        s_roy = max(p, df_between)
        F_roy = theta * (df_within - s_roy + df_between) / s_roy if s_roy > 0 else 0
        df1_roy = s_roy
        df2_roy = df_within - s_roy + df_between
        p_roy = float(1 - f_dist.cdf(F_roy, df1_roy, df2_roy)) if df1_roy > 0 and df2_roy > 0 else 1.0
        
        test_stats = {
            'pillai': {'statistic': float(V), 'F': float(F_pillai), 'df1': float(df1_pillai), 'df2': float(df2_pillai), 'p_value': p_pillai},
            'wilks': {'statistic': float(L), 'F': float(F_wilks), 'df1': float(df1_wilks), 'df2': float(df2_wilks), 'p_value': p_wilks},
            'hotelling': {'statistic': float(T_hl), 'F': float(F_hotelling), 'df1': float(df1_pillai), 'df2': float(df2_pillai), 'p_value': p_hotelling},
            'roy': {'statistic': theta, 'F': float(F_roy), 'df1': float(df1_roy), 'df2': float(df2_roy), 'p_value': p_roy}
        }
        
        # Univariate follow-up with omega-squared
        univariate = {}
        for dv in dependent_vars:
            gdata = [clean_data[clean_data[factor] == g][dv].values for g in groups]
            f_stat, p_val = stats.f_oneway(*gdata)
            
            # Calculate SS
            ss_between = sum(len(g) * (np.mean(g) - clean_data[dv].mean())**2 for g in gdata)
            ss_within = sum(np.sum((g - np.mean(g))**2) for g in gdata)
            ss_total = np.sum((clean_data[dv] - clean_data[dv].mean())**2)
            
            # Effect sizes
            eta_sq = ss_between / ss_total if ss_total > 0 else 0
            
            # Omega-squared (unbiased)
            ms_within = ss_within / df_within if df_within > 0 else 0
            omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0
            omega_sq = max(0, omega_sq)  # Can't be negative
            
            # Partial eta-squared
            partial_eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0
            
            univariate[dv] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_val),
                'eta_squared': float(eta_sq),
                'omega_squared': float(omega_sq),
                'partial_eta_squared': float(partial_eta_sq),
                'ss_between': float(ss_between),
                'ss_within': float(ss_within),
                'df_between': int(df_between),
                'df_within': int(df_within),
                'significant': bool(p_val < alpha)
            }
        
        # Post-hoc with enhanced statistics
        posthoc = {}
        if p_pillai < alpha:
            for dv in dependent_vars:
                pairwise = []
                for g1, g2 in itertools.combinations(groups, 2):
                    d1 = clean_data[clean_data[factor] == g1][dv].values
                    d2 = clean_data[clean_data[factor] == g2][dv].values
                    
                    n1, n2 = len(d1), len(d2)
                    mean1, mean2 = np.mean(d1), np.mean(d2)
                    var1, var2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
                    
                    t_stat, p_val = stats.ttest_ind(d1, d2)
                    
                    # Pooled std and Cohen's d
                    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    # Standard error of difference
                    se_diff = np.sqrt(var1/n1 + var2/n2)
                    
                    # 95% CI for mean difference
                    df_t = n1 + n2 - 2
                    t_crit = stats.t.ppf(0.975, df_t)
                    mean_diff = mean1 - mean2
                    ci_lower = mean_diff - t_crit * se_diff
                    ci_upper = mean_diff + t_crit * se_diff
                    
                    pairwise.append({
                        'group1': str(g1),
                        'group2': str(g2),
                        'mean1': float(mean1),
                        'mean2': float(mean2),
                        'mean_diff': float(mean_diff),
                        'se_diff': float(se_diff),
                        'ci_lower': float(ci_lower),
                        'ci_upper': float(ci_upper),
                        't_statistic': float(t_stat),
                        'df': int(df_t),
                        'p_value': float(p_val),
                        'cohens_d': float(cohens_d)
                    })
                
                if pairwise:
                    p_vals = [t['p_value'] for t in pairwise]
                    corrected = multipletests(p_vals, method='bonferroni')[1]
                    for t, pc in zip(pairwise, corrected):
                        t['p_corrected'] = float(pc)
                        t['significant_corrected'] = bool(pc < alpha)
                posthoc[dv] = pairwise
        
        # Descriptive statistics
        descriptives = calculate_descriptives(clean_data, factor, groups, dependent_vars)
        
        # Box's M test for homogeneity of covariance matrices
        boxs_m = calculate_boxs_m_test(clean_data, factor, groups, dependent_vars)
        
        # Assumptions summary
        assumptions = {
            "boxs_m": boxs_m,
            "sample_size_adequate": bool(n >= 20 and all(len(gd) > p for gd in group_data)),
            "min_group_size": int(min(len(gd) for gd in group_data)),
            "n_per_dv_ratio": float(n / p)
        }
        
        significant = bool(p_pillai < alpha)
        interpretation = f"MANOVA {'reveals significant' if significant else 'does not detect significant'} multivariate effect of '{factor}' (Pillai's Trace = {V:.3f}, p = {p_pillai:.4f})."
        
        plot = generate_plot(clean_data, factor, groups, dependent_vars, test_stats, alpha)
        
        return _to_native({
            "results": {
                "method": "one_way_manova",
                "factor": factor,
                "groups": [str(g) for g in groups],
                "n_groups": k,
                "test_statistics": test_stats,
                "univariate_results": univariate,
                "posthoc_results": posthoc if posthoc else None,
                "descriptives": descriptives,
                "assumptions": assumptions,
                "significant": significant,
                "interpretation": interpretation,
                "dropped_rows": dropped_rows,
                "n_dropped": len(dropped_rows),
                "n_used": len(clean_data),
                "n_original": original_len
            },
            "plot": plot
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
