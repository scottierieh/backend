from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np
from scipy import stats
from scipy.stats import t, levene, shapiro
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

router = APIRouter()


class IndependentTTestRequest(BaseModel):
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


def generate_plot(data1, data2, groups, t_stat, df, mean1, mean2, alternative='two-sided'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.histplot(data1, ax=axes[0, 0], color='#5B9BD5', label=str(groups[0]), kde=True, alpha=0.6)
    sns.histplot(data2, ax=axes[0, 0], color='#F4A582', label=str(groups[1]), kde=True, alpha=0.6)
    axes[0, 0].set_title('Group Distributions', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    
    sns.boxplot(data=[data1, data2], ax=axes[0, 1], palette=['#5B9BD5', '#F4A582'])
    axes[0, 1].set_xticklabels(groups)
    axes[0, 1].set_title('Group Boxplots', fontsize=12, fontweight='bold')
    
    if df > 0 and np.isfinite(df):
        x = np.linspace(-4, 4, 500)
        y = t.pdf(x, df)
        axes[1, 0].plot(x, y, label=f't-distribution (df={df:.1f})', color='#5B9BD5')
        axes[1, 0].axvline(t_stat, color='red', linestyle='--', label=f"t-stat = {t_stat:.2f}")

        if alternative == 'two-sided':
            shade = (x >= abs(t_stat)) | (x <= -abs(t_stat))
            title_suffix = '(two-sided)'
        elif alternative == 'greater':
            shade = x >= t_stat
            title_suffix = '(right-tailed)'
        else:  # less
            shade = x <= t_stat
            title_suffix = '(left-tailed)'

        axes[1, 0].fill_between(x, 0, y, where=shade, color='red', alpha=0.3)
        axes[1, 0].set_title(f'Test Statistic on t-Distribution {title_suffix}', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
    
    residuals1 = data1 - mean1
    residuals2 = data2 - mean2
    all_residuals = np.concatenate([residuals1, residuals2])
    stats.probplot(all_residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/independent-t-test")
def independent_t_test(req: IndependentTTestRequest):
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
        
        # Levene's test
        levene_stat, levene_p = levene(group1_data, group2_data)
        equal_var = bool(levene_p > alpha)
        
        # Student's t-test
        df_student = n1 + n2 - 2
        pooled_std = float(np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / df_student)) if df_student > 0 else 0
        se_diff_student = float(pooled_std * np.sqrt(1/n1 + 1/n2)) if n1 > 0 and n2 > 0 else 0
        t_stat_student, p_value_student = stats.ttest_ind(group1_data, group2_data, equal_var=True, alternative=alternative)
        ci_margin_student = float(t.ppf(1 - alpha/2, df_student) * se_diff_student) if df_student > 0 else 0
        ci_student = (float(mean_diff - ci_margin_student), float(mean_diff + ci_margin_student))
        
        # Welch's t-test
        t_stat_welch, p_value_welch = stats.ttest_ind(group1_data, group2_data, equal_var=False, alternative=alternative)
        s1_sq_n1 = std1**2 / n1 if n1 > 0 else 0
        s2_sq_n2 = std2**2 / n2 if n2 > 0 else 0
        df_num = (s1_sq_n1 + s2_sq_n2)**2
        df_den = ((s1_sq_n1**2/(n1-1)) + (s2_sq_n2**2/(n2-1))) if n1 > 1 and n2 > 1 else 1
        df_welch = float(df_num / df_den) if df_den > 0 else float('inf')
        se_diff_welch = float(np.sqrt(s1_sq_n1 + s2_sq_n2))
        ci_margin_welch = float(t.ppf(1 - alpha/2, df_welch) * se_diff_welch) if np.isfinite(df_welch) else 0
        ci_welch = (float(mean_diff - ci_margin_welch), float(mean_diff + ci_margin_welch))
        
        # Effect size
        cohens_d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0
        
        # Use appropriate test
        if equal_var:
            t_stat_main = float(t_stat_student)
            p_value_main = float(p_value_student)
            df_main = float(df_student)
            se_diff_main = se_diff_student
        else:
            t_stat_main = float(t_stat_welch)
            p_value_main = float(p_value_welch)
            df_main = df_welch
            se_diff_main = se_diff_welch
        
        significant = bool(p_value_main < alpha)
        
        # Interpretation
        test_name = "Student's" if equal_var else "Welch's"
        p_text = "p < .001" if p_value_main < 0.001 else f"p = {p_value_main:.3f}"
        sig_text = "statistically significant" if significant else "not statistically significant"
        effect_interp = get_effect_size_interpretation(cohens_d)
        
        interpretation = f"A {test_name} independent-samples t-test was conducted to compare '{variable}' between '{groups[0]}' and '{groups[1]}'.\n\n"
        if equal_var:
            interpretation += f"Levene's test was not significant (p = {levene_p:.3f}), indicating equal variances.\n\n"
        else:
            interpretation += f"Levene's test was significant (p = {levene_p:.3f}), indicating unequal variances (Welch's correction applied).\n\n"
        interpretation += f"There was a {sig_text} difference between '{groups[0]}' (M={mean1:.2f}, SD={std1:.2f}) and '{groups[1]}' (M={mean2:.2f}, SD={std2:.2f}), t({df_main:.2f}) = {t_stat_main:.2f}, {p_text}.\n\n"
        interpretation += f"Cohen's d = {cohens_d:.3f} indicates a {effect_interp} effect size."
        
        # Generate plot
        plot = generate_plot(group1_data, group2_data, groups, t_stat_main, df_main, mean1, mean2, alternative)
        
        return {
            "results": {
                "test_type": "independent_samples",
                "variable": variable,
                "group_variable": group_variable,
                "groups": groups,
                "equal_var": equal_var,
                "n1": n1,
                "n2": n2,
                "mean1": mean1,
                "mean2": mean2,
                "std1": std1,
                "std2": std2,
                "mean_diff": mean_diff,
                "se_diff": se_diff_main,
                "t_statistic": t_stat_main,
                "degrees_of_freedom": df_main,
                "p_value": p_value_main,
                "significant": significant,
                "cohens_d": cohens_d,
                "interpretation": interpretation,
                "dropped_rows": dropped_rows,
                "n_dropped": int(len(dropped_rows)),
                "normality_test": normality_test if normality_test else None,
                "levene_test": {
                    "statistic": float(levene_stat),
                    "p_value": float(levene_p),
                    "assumption_met": equal_var
                },
                "student_t": {
                    "t_statistic": float(t_stat_student),
                    "df": df_student,
                    "p_value": float(p_value_student),
                    "mean_diff": mean_diff,
		    "se_diff": se_diff_student,
                    "ci": list(ci_student)
                },
                "welch_t": {
                    "t_statistic": float(t_stat_welch),
                    "df": df_welch,
                    "p_value": float(p_value_welch),
		    "mean_diff": mean_diff,
                    "se_diff": se_diff_welch,
                    "ci": list(ci_welch)
                },
                "descriptives": {
                    groups[0]: {"n": n1, "mean": mean1, "std_dev": std1},
                    groups[1]: {"n": n2, "mean": mean2, "std_dev": std2}
                }
            },
            "plot": plot
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
