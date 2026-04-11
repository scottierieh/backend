from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np
from scipy import stats
from scipy.stats import t, shapiro
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

router = APIRouter()


class WelchsTTestRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    params: dict = Field(..., description="Test parameters")


def get_effect_size_interpretation(d: float) -> str:
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "large"
    elif abs_d >= 0.5:
        return "medium"
    elif abs_d >= 0.2:
        return "small"
    else:
        return "negligible"


def generate_plot(data1, data2, groups, t_stat, df, mean1, mean2, std1, std2, n1, n2, variable, alpha=0.05):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distributions
    sns.histplot(data1, ax=axes[0, 0], color='#5B9BD5', label=f'{groups[0]} (n={n1})', kde=True, alpha=0.6, stat='density')
    sns.histplot(data2, ax=axes[0, 0], color='#F4A582', label=f'{groups[1]} (n={n2})', kde=True, alpha=0.6, stat='density')
    axes[0, 0].axvline(mean1, color='#2171B5', linestyle='--', linewidth=2)
    axes[0, 0].axvline(mean2, color='#CB181D', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Group Distributions', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    
    # Plot 2: Boxplots
    import pandas as pd
    plot_data = pd.DataFrame({
        'Group': [groups[0]] * len(data1) + [groups[1]] * len(data2),
        'Value': np.concatenate([data1, data2])
    })
    sns.boxplot(data=plot_data, x='Group', y='Value', ax=axes[0, 1], palette=['#5B9BD5', '#F4A582'])
    sns.stripplot(data=plot_data, x='Group', y='Value', ax=axes[0, 1], color='black', alpha=0.4, size=4)
    axes[0, 1].set_title('Group Comparison', fontsize=12, fontweight='bold')
    
    # Plot 3: T-distribution
    if df > 0 and np.isfinite(df):
        x = np.linspace(-5, 5, 500)
        y = t.pdf(x, df)
        axes[1, 0].plot(x, y, label=f't-distribution (df={df:.1f})', color='#5B9BD5')
        axes[1, 0].axvline(t_stat, color='red', linestyle='--', label=f't = {t_stat:.3f}')
        critical_t = t.ppf(1 - alpha/2, df)
        axes[1, 0].fill_between(x, 0, y, where=(x >= critical_t) | (x <= -critical_t), color='red', alpha=0.3)
        axes[1, 0].set_title("Welch's t-Distribution", fontsize=12, fontweight='bold')
        axes[1, 0].legend()
    
    # Plot 4: Means with CI
    means = [mean1, mean2]
    sems = [std1/np.sqrt(n1), std2/np.sqrt(n2)]
    axes[1, 1].bar(groups, means, color=['#5B9BD5', '#F4A582'], alpha=0.8)
    axes[1, 1].errorbar(groups, means, yerr=[s*1.96 for s in sems], fmt='none', color='black', capsize=8)
    axes[1, 1].set_title('Group Means with 95% CI', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/welchs-ttest")
def welchs_t_test(req: WelchsTTestRequest):
    try:
        data = req.data
        params = req.params
        
        variable = params.get('variable')
        group_variable = params.get('group_variable')
        alternative = params.get('alternative', 'two-sided')
        alpha = float(params.get('alpha', 0.05))
        
        if not variable or not group_variable:
            raise HTTPException(status_code=400, detail="Variable and group_variable are required")
        
        # Extract and clean data
        dropped_rows = []
        clean_data = []
        
        for idx, row in enumerate(data):
            val = row.get(variable)
            grp = row.get(group_variable)
            if val is None or val == '' or grp is None or grp == '':
                dropped_rows.append(idx)
            else:
                try:
                    clean_data.append({'value': float(val), 'group': str(grp)})
                except (ValueError, TypeError):
                    dropped_rows.append(idx)
        
        groups = list(set([d['group'] for d in clean_data]))
        
        if len(groups) != 2:
            raise HTTPException(status_code=400, detail=f"Grouping variable must have exactly 2 groups, found {len(groups)}")
        
        group1_data = np.array([d['value'] for d in clean_data if d['group'] == groups[0]])
        group2_data = np.array([d['value'] for d in clean_data if d['group'] == groups[1]])
        
        if len(group1_data) < 2 or len(group2_data) < 2:
            raise HTTPException(status_code=400, detail="Each group must have at least 2 observations")
        
        # Descriptive statistics
        n1, n2 = int(len(group1_data)), int(len(group2_data))
        mean1, mean2 = float(np.mean(group1_data)), float(np.mean(group2_data))
        std1, std2 = float(np.std(group1_data, ddof=1)), float(np.std(group2_data, ddof=1))
        var1, var2 = std1**2, std2**2
        mean_diff = float(mean1 - mean2)
        
        # Normality tests
        normality_test = {}
        for group_name, group_data in [(groups[0], group1_data), (groups[1], group2_data)]:
            if len(group_data) >= 3:
                stat, p = shapiro(group_data)
                normality_test[str(group_name)] = {
                    'statistic': float(stat),
                    'p_value': float(p),
                    'assumption_met': bool(p > alpha)
                }
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False, alternative=alternative)
        
        # Welch-Satterthwaite degrees of freedom
        s1_sq_n1 = var1 / n1
        s2_sq_n2 = var2 / n2
        df_num = (s1_sq_n1 + s2_sq_n2)**2
        df_den = (s1_sq_n1**2 / (n1 - 1)) + (s2_sq_n2**2 / (n2 - 1))
        df = float(df_num / df_den) if df_den > 0 else float('inf')
        
        # Standard error
        se_diff = float(np.sqrt(s1_sq_n1 + s2_sq_n2))
        
        # Confidence interval
        if np.isfinite(df) and df > 0:
            ci_margin = t.ppf(1 - alpha/2, df) * se_diff
            ci_lower = float(mean_diff - ci_margin)
            ci_upper = float(mean_diff + ci_margin)
        else:
            ci_lower, ci_upper = None, None
        
        # Effect sizes
        pooled_std = float(np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)))
        cohens_d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0
        glass_delta = float(mean_diff / std2) if std2 > 0 else 0.0
        correction = 1 - (3 / (4*(n1+n2) - 9)) if (n1+n2) > 2 else 1
        hedges_g = float(cohens_d * correction)
        
        # Variance ratio
        variance_ratio = float(max(var1, var2) / min(var1, var2)) if min(var1, var2) > 0 else float('inf')
        
        significant = bool(p_value < alpha)
        
        # Interpretation
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        sig_text = "statistically significant" if significant else "not statistically significant"
        effect_interp = get_effect_size_interpretation(cohens_d)
        
        interpretation = f"A Welch's t-test was conducted to compare '{variable}' between '{groups[0]}' and '{groups[1]}'.\n\n"
        interpretation += f"The variance ratio was {variance_ratio:.2f}. "
        interpretation += f"There was a {sig_text} difference between '{groups[0]}' (M={mean1:.2f}, SD={std1:.2f}) and '{groups[1]}' (M={mean2:.2f}, SD={std2:.2f}), t({df:.2f}) = {float(t_stat):.2f}, {p_text}.\n\n"
        interpretation += f"Cohen's d = {cohens_d:.3f} ({effect_interp}), Hedges' g = {hedges_g:.3f}, Glass's Δ = {glass_delta:.3f}."
        
        # Generate plot
        plot = generate_plot(group1_data, group2_data, groups, float(t_stat), df, mean1, mean2, std1, std2, n1, n2, variable, alpha)
        
        return {
            "results": {
                "test_type": "welchs_t_test",
                "variable": variable,
                "group_variable": group_variable,
                "groups": groups,
                "n1": n1,
                "n2": n2,
                "mean1": mean1,
                "mean2": mean2,
                "std1": std1,
                "std2": std2,
                "var1": var1,
                "var2": var2,
                "variance_ratio": variance_ratio,
                "mean_diff": mean_diff,
                "se_diff": se_diff,
                "t_statistic": float(t_stat),
                "degrees_of_freedom": df,
                "p_value": float(p_value),
                "significant": significant,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "cohens_d": cohens_d,
                "hedges_g": hedges_g,
                "glass_delta": glass_delta,
                "interpretation": interpretation,
                "dropped_rows": dropped_rows,
                "n_dropped": int(len(dropped_rows)),
                "normality_test": normality_test if normality_test else None,
                "descriptives": {
                    groups[0]: {"n": n1, "mean": mean1, "std_dev": std1, "variance": var1, "se_mean": float(std1/np.sqrt(n1))},
                    groups[1]: {"n": n2, "mean": mean2, "std_dev": std2, "variance": var2, "se_mean": float(std2/np.sqrt(n2))}
                }
            },
            "plot": plot
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
