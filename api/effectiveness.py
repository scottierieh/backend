
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
import statsmodels.api as sm
from statsmodels.formula.api import ols
import io
import base64
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

router = APIRouter()


# ==============================================
# EFFECTIVENESS ANALYSIS ROUTER
# Endpoint: POST /api/analysis/effectiveness
# ==============================================


class EffectivenessRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    outcome: str = Field(..., description="Outcome variable (numeric)")
    time: Optional[str] = Field(None, description="Time variable")
    group: Optional[str] = Field(None, description="Group variable")
    covariates: list[str] = Field(default_factory=list, description="Covariate variables")


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def get_effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "large"
    elif abs_d >= 0.5:
        return "medium"
    elif abs_d >= 0.2:
        return "small"
    else:
        return "negligible"


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def sanitize_column_name(name: str) -> str:
    """Sanitize column name for statsmodels formula."""
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))


@router.post("/effectiveness")
def effectiveness_analysis(req: EffectivenessRequest):
    """
    Comprehensive effectiveness analysis including:
    1. Descriptive Statistics
    2. Pre-Post Comparison
    3. Effect Size Analysis
    4. Trend Analysis
    5. Difference-in-Differences (DID)
    6. Sensitivity Analysis
    """
    try:
        df = pd.DataFrame(req.data)
        outcome_var = req.outcome
        time_var = req.time
        group_var = req.group
        covariates = req.covariates
        alpha = 0.05
        
        # Validate outcome variable
        if outcome_var not in df.columns:
            raise HTTPException(status_code=400, detail=f"Outcome variable '{outcome_var}' not found")
        
        # Clean data
        df[outcome_var] = pd.to_numeric(df[outcome_var], errors='coerce')
        n_total = len(df)
        df_clean = df.dropna(subset=[outcome_var])
        n_valid = len(df_clean)
        
        if n_valid < 2:
            raise HTTPException(status_code=400, detail="Not enough valid data points")
        
        # ========================================
        # 1. DESCRIPTIVE STATISTICS
        # ========================================
        outcome_values = df_clean[outcome_var].values
        
        descriptive_overall = {
            "n": int(n_valid),
            "mean": float(np.mean(outcome_values)),
            "std": float(np.std(outcome_values, ddof=1)),
            "median": float(np.median(outcome_values)),
            "min": float(np.min(outcome_values)),
            "max": float(np.max(outcome_values)),
            "q1": float(np.percentile(outcome_values, 25)),
            "q3": float(np.percentile(outcome_values, 75)),
            "se": float(np.std(outcome_values, ddof=1) / np.sqrt(n_valid))
        }
        
        # By group
        descriptive_by_group = {}
        if group_var and group_var in df_clean.columns:
            for grp in df_clean[group_var].dropna().unique():
                grp_data = df_clean[df_clean[group_var] == grp][outcome_var].values
                if len(grp_data) > 0:
                    descriptive_by_group[str(grp)] = {
                        "n": int(len(grp_data)),
                        "mean": float(np.mean(grp_data)),
                        "std": float(np.std(grp_data, ddof=1)) if len(grp_data) > 1 else 0.0,
                        "median": float(np.median(grp_data)),
                        "min": float(np.min(grp_data)),
                        "max": float(np.max(grp_data)),
                        "se": float(np.std(grp_data, ddof=1) / np.sqrt(len(grp_data))) if len(grp_data) > 1 else 0.0
                    }
        
        # By time
        descriptive_by_time = {}
        if time_var and time_var in df_clean.columns:
            for t in df_clean[time_var].dropna().unique():
                t_data = df_clean[df_clean[time_var] == t][outcome_var].values
                if len(t_data) > 0:
                    descriptive_by_time[str(t)] = {
                        "n": int(len(t_data)),
                        "mean": float(np.mean(t_data)),
                        "std": float(np.std(t_data, ddof=1)) if len(t_data) > 1 else 0.0,
                        "median": float(np.median(t_data)),
                        "se": float(np.std(t_data, ddof=1) / np.sqrt(len(t_data))) if len(t_data) > 1 else 0.0
                    }
        
        # By group x time
        descriptive_by_group_time = {}
        if group_var and time_var and group_var in df_clean.columns and time_var in df_clean.columns:
            for grp in df_clean[group_var].dropna().unique():
                descriptive_by_group_time[str(grp)] = {}
                for t in df_clean[time_var].dropna().unique():
                    gt_data = df_clean[(df_clean[group_var] == grp) & (df_clean[time_var] == t)][outcome_var].values
                    if len(gt_data) > 0:
                        descriptive_by_group_time[str(grp)][str(t)] = {
                            "n": int(len(gt_data)),
                            "mean": float(np.mean(gt_data)),
                            "std": float(np.std(gt_data, ddof=1)) if len(gt_data) > 1 else 0.0,
                            "se": float(np.std(gt_data, ddof=1) / np.sqrt(len(gt_data))) if len(gt_data) > 1 else 0.0
                        }
        
        # Descriptive plot
        desc_plot = None
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            sns.histplot(outcome_values, ax=axes[0], kde=True, color='#5B9BD5')
            axes[0].axvline(descriptive_overall["mean"], color='red', linestyle='--', label=f'Mean: {descriptive_overall["mean"]:.2f}')
            axes[0].axvline(descriptive_overall["median"], color='orange', linestyle='--', label=f'Median: {descriptive_overall["median"]:.2f}')
            axes[0].set_title(f'Distribution of {outcome_var}', fontweight='bold')
            axes[0].set_xlabel(outcome_var)
            axes[0].legend()
            
            # Box plot by group or time
            if group_var and group_var in df_clean.columns:
                sns.boxplot(data=df_clean, x=group_var, y=outcome_var, ax=axes[1], palette='crest')
                axes[1].set_title(f'{outcome_var} by {group_var}', fontweight='bold')
            elif time_var and time_var in df_clean.columns:
                sns.boxplot(data=df_clean, x=time_var, y=outcome_var, ax=axes[1], palette='crest')
                axes[1].set_title(f'{outcome_var} by {time_var}', fontweight='bold')
            else:
                sns.boxplot(x=outcome_values, ax=axes[1], color='#5B9BD5')
                axes[1].set_title(f'Box Plot of {outcome_var}', fontweight='bold')
            
            plt.tight_layout()
            desc_plot = fig_to_base64(fig)
        except Exception:
            pass
        
        descriptive_stats = {
            "overall": descriptive_overall,
            "by_group": descriptive_by_group,
            "by_time": descriptive_by_time,
            "by_group_time": descriptive_by_group_time,
            "plot": desc_plot
        }
        
        # ========================================
        # 2. PRE-POST COMPARISON
        # ========================================
        pre_post_comparison = None
        pre_post_plot = None
        
        if time_var and time_var in df_clean.columns:
            time_values = df_clean[time_var].dropna().unique()
            
            if len(time_values) >= 2:
                # Sort time values
                try:
                    time_values_sorted = sorted(time_values, key=lambda x: float(x) if str(x).replace('.','').replace('-','').isdigit() else x)
                except:
                    time_values_sorted = sorted(time_values, key=str)
                
                pre_label = str(time_values_sorted[0])
                post_label = str(time_values_sorted[-1])
                
                pre_data = df_clean[df_clean[time_var] == time_values_sorted[0]][outcome_var].values
                post_data = df_clean[df_clean[time_var] == time_values_sorted[-1]][outcome_var].values
                
                if len(pre_data) >= 2 and len(post_data) >= 2:
                    pre_mean = float(np.mean(pre_data))
                    post_mean = float(np.mean(post_data))
                    pre_std = float(np.std(pre_data, ddof=1))
                    post_std = float(np.std(post_data, ddof=1))
                    difference = post_mean - pre_mean
                    percent_change = (difference / abs(pre_mean) * 100) if pre_mean != 0 else None
                    
                    # Choose test
                    if len(pre_data) == len(post_data):
                        # Paired t-test
                        t_stat, p_value = ttest_rel(post_data, pre_data)
                        test_used = "paired t-test"
                    else:
                        # Independent t-test
                        t_stat, p_value = ttest_ind(post_data, pre_data)
                        test_used = "independent t-test"
                    
                    # Effect size
                    effect_size = calculate_cohens_d(post_data, pre_data)
                    effect_interpretation = get_effect_size_interpretation(effect_size)
                    
                    pre_post_overall = {
                        "pre_mean": pre_mean,
                        "post_mean": post_mean,
                        "pre_std": pre_std,
                        "post_std": post_std,
                        "difference": difference,
                        "percent_change": percent_change,
                        "test_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "test_used": test_used,
                        "effect_size": float(effect_size),
                        "effect_interpretation": effect_interpretation,
                        "significant": bool(p_value < alpha),
                        "pre_n": int(len(pre_data)),
                        "post_n": int(len(post_data)),
                        "pre_label": pre_label,
                        "post_label": post_label
                    }
                    
                    # By group comparison
                    pre_post_by_group = {}
                    if group_var and group_var in df_clean.columns:
                        for grp in df_clean[group_var].dropna().unique():
                            grp_pre = df_clean[(df_clean[group_var] == grp) & (df_clean[time_var] == time_values_sorted[0])][outcome_var].values
                            grp_post = df_clean[(df_clean[group_var] == grp) & (df_clean[time_var] == time_values_sorted[-1])][outcome_var].values
                            
                            if len(grp_pre) >= 2 and len(grp_post) >= 2:
                                grp_diff = float(np.mean(grp_post)) - float(np.mean(grp_pre))
                                if len(grp_pre) == len(grp_post):
                                    grp_t, grp_p = ttest_rel(grp_post, grp_pre)
                                else:
                                    grp_t, grp_p = ttest_ind(grp_post, grp_pre)
                                grp_d = calculate_cohens_d(grp_post, grp_pre)
                                
                                pre_post_by_group[str(grp)] = {
                                    "pre_mean": float(np.mean(grp_pre)),
                                    "post_mean": float(np.mean(grp_post)),
                                    "difference": grp_diff,
                                    "p_value": float(grp_p),
                                    "effect_size": float(grp_d),
                                    "significant": bool(grp_p < alpha)
                                }
                    
                    pre_post_comparison = {
                        "overall": pre_post_overall,
                        "by_group": pre_post_by_group,
                        "plot": None
                    }
                    
                    # Pre-post plot
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        if group_var and group_var in df_clean.columns:
                            # Grouped bar chart
                            groups = list(df_clean[group_var].dropna().unique())
                            x = np.arange(len(groups))
                            width = 0.35
                            
                            pre_means = [df_clean[(df_clean[group_var] == g) & (df_clean[time_var] == time_values_sorted[0])][outcome_var].mean() for g in groups]
                            post_means = [df_clean[(df_clean[group_var] == g) & (df_clean[time_var] == time_values_sorted[-1])][outcome_var].mean() for g in groups]
                            
                            ax.bar(x - width/2, pre_means, width, label=f'Pre ({pre_label})', color='#5B9BD5')
                            ax.bar(x + width/2, post_means, width, label=f'Post ({post_label})', color='#ED7D31')
                            ax.set_xticks(x)
                            ax.set_xticklabels([str(g) for g in groups])
                            ax.set_xlabel(group_var)
                        else:
                            # Simple bar chart
                            ax.bar(['Pre', 'Post'], [pre_mean, post_mean], color=['#5B9BD5', '#ED7D31'])
                            ax.errorbar(['Pre', 'Post'], [pre_mean, post_mean], 
                                       yerr=[pre_std/np.sqrt(len(pre_data)), post_std/np.sqrt(len(post_data))],
                                       fmt='none', color='black', capsize=5)
                        
                        ax.set_ylabel(outcome_var)
                        ax.set_title(f'Pre-Post Comparison: {outcome_var}', fontweight='bold')
                        ax.legend()
                        
                        # Add significance annotation
                        if pre_post_overall["significant"]:
                            max_y = max(pre_mean, post_mean) + max(pre_std, post_std)
                            ax.annotate('*', xy=(0.5, max_y), fontsize=20, ha='center')
                        
                        plt.tight_layout()
                        pre_post_plot = fig_to_base64(fig)
                        pre_post_comparison["plot"] = pre_post_plot
                    except Exception:
                        pass
        
        # ========================================
        # 3. EFFECT SIZE ANALYSIS
        # ========================================
        effect_size_plot = None
        if pre_post_comparison and pre_post_comparison["overall"]:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                effect_d = pre_post_comparison["overall"]["effect_size"]
                
                # Effect size scale
                categories = ['Negligible\n(< 0.2)', 'Small\n(0.2-0.5)', 'Medium\n(0.5-0.8)', 'Large\n(≥ 0.8)']
                thresholds = [0.2, 0.5, 0.8, 1.5]
                colors = ['#E8E8E8', '#B8D4E8', '#7AB8D8', '#3A8FC2']
                
                # Draw scale
                left = 0
                for i, (cat, thresh) in enumerate(zip(categories, thresholds)):
                    width = thresh - (thresholds[i-1] if i > 0 else 0)
                    ax.barh(0, width, left=left, height=0.5, color=colors[i], edgecolor='white')
                    left = thresh
                
                # Mark current effect size
                ax.axvline(abs(effect_d), color='red', linewidth=3, linestyle='--')
                ax.plot(abs(effect_d), 0, 'rv', markersize=15)
                ax.annotate(f'd = {effect_d:.3f}', xy=(abs(effect_d), 0.3), fontsize=12, ha='center', fontweight='bold')
                
                ax.set_xlim(0, 1.5)
                ax.set_ylim(-0.5, 0.8)
                ax.set_xlabel("Cohen's d")
                ax.set_title('Effect Size Magnitude', fontweight='bold')
                ax.set_yticks([])
                
                # Add category labels
                ax.text(0.1, -0.35, 'Negligible', ha='center', fontsize=9)
                ax.text(0.35, -0.35, 'Small', ha='center', fontsize=9)
                ax.text(0.65, -0.35, 'Medium', ha='center', fontsize=9)
                ax.text(1.15, -0.35, 'Large', ha='center', fontsize=9)
                
                plt.tight_layout()
                effect_size_plot = fig_to_base64(fig)
            except Exception:
                pass
        
        effect_size_analysis = {"plot": effect_size_plot}
        
        # ========================================
        # 4. TREND ANALYSIS
        # ========================================
        trend_analysis = None
        
        if time_var and time_var in df_clean.columns:
            time_values = df_clean[time_var].dropna().unique()
            
            if len(time_values) >= 2:
                try:
                    # Convert time to numeric for regression
                    time_mapping = {t: i for i, t in enumerate(sorted(time_values, key=lambda x: str(x)))}
                    df_clean['_time_numeric'] = df_clean[time_var].map(time_mapping)
                    
                    # Group by time and calculate means
                    time_means = df_clean.groupby('_time_numeric')[outcome_var].mean()
                    time_points = [str(t) for t in sorted(time_values, key=lambda x: str(x))]
                    means = time_means.values.tolist()
                    
                    # Linear regression
                    X = sm.add_constant(time_means.index.values)
                    y = time_means.values
                    model = sm.OLS(y, X).fit()
                    
                    slope = float(model.params[1])
                    intercept = float(model.params[0])
                    r_squared = float(model.rsquared)
                    p_value = float(model.pvalues[1])
                    std_err = float(model.bse[1])
                    
                    if p_value < alpha:
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                    else:
                        trend_direction = "flat"
                    
                    overall_trend = {
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_squared,
                        "p_value": p_value,
                        "std_err": std_err,
                        "trend_direction": trend_direction,
                        "significant": bool(p_value < alpha),
                        "time_points": time_points,
                        "means": [float(m) for m in means]
                    }
                    
                    # By group trends
                    by_group_trends = {}
                    if group_var and group_var in df_clean.columns:
                        for grp in df_clean[group_var].dropna().unique():
                            grp_data = df_clean[df_clean[group_var] == grp]
                            grp_time_means = grp_data.groupby('_time_numeric')[outcome_var].mean()
                            
                            if len(grp_time_means) >= 2:
                                X_grp = sm.add_constant(grp_time_means.index.values)
                                y_grp = grp_time_means.values
                                model_grp = sm.OLS(y_grp, X_grp).fit()
                                
                                by_group_trends[str(grp)] = {
                                    "slope": float(model_grp.params[1]),
                                    "r_squared": float(model_grp.rsquared),
                                    "p_value": float(model_grp.pvalues[1]),
                                    "significant": bool(model_grp.pvalues[1] < alpha)
                                }
                    
                    # Trend plot
                    trend_plot = None
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if group_var and group_var in df_clean.columns:
                            for grp in df_clean[group_var].dropna().unique():
                                grp_data = df_clean[df_clean[group_var] == grp]
                                grp_means = grp_data.groupby('_time_numeric')[outcome_var].mean()
                                ax.plot(grp_means.index, grp_means.values, 'o-', label=str(grp), linewidth=2, markersize=8)
                        else:
                            ax.plot(time_means.index, time_means.values, 'o-', color='#5B9BD5', linewidth=2, markersize=8)
                            # Add trend line
                            x_line = np.array([time_means.index.min(), time_means.index.max()])
                            y_line = intercept + slope * x_line
                            ax.plot(x_line, y_line, '--', color='red', label=f'Trend (slope={slope:.3f})')
                        
                        ax.set_xticks(range(len(time_points)))
                        ax.set_xticklabels(time_points)
                        ax.set_xlabel(time_var)
                        ax.set_ylabel(outcome_var)
                        ax.set_title(f'Trend Analysis: {outcome_var} over {time_var}', fontweight='bold')
                        ax.legend()
                        
                        plt.tight_layout()
                        trend_plot = fig_to_base64(fig)
                    except Exception:
                        pass
                    
                    trend_analysis = {
                        "overall_trend": overall_trend,
                        "by_group": by_group_trends,
                        "plot": trend_plot
                    }
                    
                except Exception as e:
                    trend_analysis = {"error": str(e), "overall_trend": None, "by_group": {}, "plot": None}
        
        # ========================================
        # 5. DIFFERENCE-IN-DIFFERENCES (DID)
        # ========================================
        did_analysis = None
        
        if time_var and group_var and time_var in df_clean.columns and group_var in df_clean.columns:
            time_values = df_clean[time_var].dropna().unique()
            group_values = df_clean[group_var].dropna().unique()
            
            if len(time_values) >= 2 and len(group_values) == 2:
                try:
                    # Sort time values
                    try:
                        time_values_sorted = sorted(time_values, key=lambda x: float(x) if str(x).replace('.','').replace('-','').isdigit() else x)
                    except:
                        time_values_sorted = sorted(time_values, key=str)
                    
                    pre_time = time_values_sorted[0]
                    post_time = time_values_sorted[-1]
                    
                    # Assume first group is control, second is treatment
                    group_values_sorted = sorted(group_values, key=str)
                    control_label = str(group_values_sorted[0])
                    treatment_label = str(group_values_sorted[1])
                    
                    # Calculate cell means
                    control_pre = df_clean[(df_clean[group_var] == group_values_sorted[0]) & (df_clean[time_var] == pre_time)][outcome_var].values
                    control_post = df_clean[(df_clean[group_var] == group_values_sorted[0]) & (df_clean[time_var] == post_time)][outcome_var].values
                    treatment_pre = df_clean[(df_clean[group_var] == group_values_sorted[1]) & (df_clean[time_var] == pre_time)][outcome_var].values
                    treatment_post = df_clean[(df_clean[group_var] == group_values_sorted[1]) & (df_clean[time_var] == post_time)][outcome_var].values
                    
                    if len(control_pre) > 0 and len(control_post) > 0 and len(treatment_pre) > 0 and len(treatment_post) > 0:
                        cell_means = {
                            "control_pre": float(np.mean(control_pre)),
                            "control_post": float(np.mean(control_post)),
                            "treatment_pre": float(np.mean(treatment_pre)),
                            "treatment_post": float(np.mean(treatment_post))
                        }
                        
                        control_change = cell_means["control_post"] - cell_means["control_pre"]
                        treatment_change = cell_means["treatment_post"] - cell_means["treatment_pre"]
                        did_estimate = treatment_change - control_change
                        
                        # DID regression
                        df_did = df_clean[df_clean[time_var].isin([pre_time, post_time])].copy()
                        df_did['_post'] = (df_did[time_var] == post_time).astype(int)
                        df_did['_treatment'] = (df_did[group_var] == group_values_sorted[1]).astype(int)
                        df_did['_interaction'] = df_did['_post'] * df_did['_treatment']
                        
                        outcome_clean = sanitize_column_name(outcome_var)
                        df_did = df_did.rename(columns={outcome_var: outcome_clean})
                        
                        formula = f'{outcome_clean} ~ _post + _treatment + _interaction'
                        model = ols(formula, data=df_did).fit()
                        
                        did_coef = float(model.params['_interaction'])
                        did_se = float(model.bse['_interaction'])
                        did_pvalue = float(model.pvalues['_interaction'])
                        did_ci = model.conf_int().loc['_interaction']
                        
                        did_analysis = {
                            "did_estimate": did_coef,
                            "did_se": did_se,
                            "did_pvalue": did_pvalue,
                            "did_ci_lower": float(did_ci[0]),
                            "did_ci_upper": float(did_ci[1]),
                            "significant": bool(did_pvalue < alpha),
                            "r_squared": float(model.rsquared),
                            "cell_means": cell_means,
                            "control_change": float(control_change),
                            "treatment_change": float(treatment_change),
                            "control_label": control_label,
                            "treatment_label": treatment_label,
                            "plot": None
                        }
                        
                        # DID plot
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            x = [0, 1]
                            ax.plot(x, [cell_means["control_pre"], cell_means["control_post"]], 'o-', 
                                   label=f'Control ({control_label})', color='#5B9BD5', linewidth=2, markersize=10)
                            ax.plot(x, [cell_means["treatment_pre"], cell_means["treatment_post"]], 's-', 
                                   label=f'Treatment ({treatment_label})', color='#ED7D31', linewidth=2, markersize=10)
                            
                            # Counterfactual line
                            counterfactual_post = cell_means["treatment_pre"] + control_change
                            ax.plot([0, 1], [cell_means["treatment_pre"], counterfactual_post], '--', 
                                   color='#ED7D31', alpha=0.5, label='Counterfactual')
                            
                            # DID arrow
                            ax.annotate('', xy=(1.05, cell_means["treatment_post"]), 
                                       xytext=(1.05, counterfactual_post),
                                       arrowprops=dict(arrowstyle='<->', color='green', lw=2))
                            ax.text(1.1, (cell_means["treatment_post"] + counterfactual_post) / 2, 
                                   f'DID = {did_coef:.2f}', fontsize=10, color='green', fontweight='bold')
                            
                            ax.set_xticks([0, 1])
                            ax.set_xticklabels([f'Pre\n({pre_time})', f'Post\n({post_time})'])
                            ax.set_ylabel(outcome_var)
                            ax.set_title('Difference-in-Differences Analysis', fontweight='bold')
                            ax.legend(loc='best')
                            ax.set_xlim(-0.2, 1.4)
                            
                            plt.tight_layout()
                            did_analysis["plot"] = fig_to_base64(fig)
                        except Exception:
                            pass
                    else:
                        did_analysis = {"error": "Insufficient data in one or more cells", "did_estimate": None}
                        
                except Exception as e:
                    did_analysis = {"error": str(e), "did_estimate": None}
            else:
                did_analysis = {"error": "DID requires exactly 2 time points and 2 groups"}
        else:
            did_analysis = {"error": "Time and group variables required for DID analysis"}
        
        # ========================================
        # 6. SENSITIVITY ANALYSIS
        # ========================================
        sensitivity_analysis = None
        
        if did_analysis and did_analysis.get("did_estimate") is not None:
            try:
                time_values_sorted = sorted(df_clean[time_var].dropna().unique(), key=str)
                pre_time = time_values_sorted[0]
                post_time = time_values_sorted[-1]
                group_values_sorted = sorted(df_clean[group_var].dropna().unique(), key=str)
                
                df_did = df_clean[df_clean[time_var].isin([pre_time, post_time])].copy()
                df_did['_post'] = (df_did[time_var] == post_time).astype(int)
                df_did['_treatment'] = (df_did[group_var] == group_values_sorted[1]).astype(int)
                df_did['_interaction'] = df_did['_post'] * df_did['_treatment']
                
                outcome_clean = sanitize_column_name(outcome_var)
                df_did = df_did.rename(columns={outcome_var: outcome_clean})
                
                # Base model (already computed)
                base_model = {
                    "did_estimate": did_analysis["did_estimate"],
                    "did_pvalue": did_analysis["did_pvalue"],
                    "significant": did_analysis["significant"],
                    "r_squared": did_analysis["r_squared"]
                }
                
                # Model with covariates
                with_covariates = None
                if covariates:
                    try:
                        cov_terms = []
                        df_did_cov = df_did.copy()
                        for cov in covariates:
                            if cov in df_did_cov.columns:
                                cov_clean = sanitize_column_name(cov)
                                df_did_cov = df_did_cov.rename(columns={cov: cov_clean})
                                df_did_cov[cov_clean] = pd.to_numeric(df_did_cov[cov_clean], errors='coerce')
                                cov_terms.append(cov_clean)
                        
                        if cov_terms:
                            df_did_cov = df_did_cov.dropna(subset=cov_terms)
                            formula_cov = f'{outcome_clean} ~ _post + _treatment + _interaction + {" + ".join(cov_terms)}'
                            model_cov = ols(formula_cov, data=df_did_cov).fit()
                            
                            with_covariates = {
                                "did_estimate": float(model_cov.params['_interaction']),
                                "did_pvalue": float(model_cov.pvalues['_interaction']),
                                "significant": bool(model_cov.pvalues['_interaction'] < alpha),
                                "r_squared": float(model_cov.rsquared),
                                "covariates_used": cov_terms
                            }
                    except Exception:
                        pass
                
                # Robustness check (trimmed data - remove outliers)
                robustness_check = None
                try:
                    q1 = df_did[outcome_clean].quantile(0.05)
                    q3 = df_did[outcome_clean].quantile(0.95)
                    df_trimmed = df_did[(df_did[outcome_clean] >= q1) & (df_did[outcome_clean] <= q3)]
                    n_excluded = len(df_did) - len(df_trimmed)
                    
                    if len(df_trimmed) > 10:
                        formula_trim = f'{outcome_clean} ~ _post + _treatment + _interaction'
                        model_trim = ols(formula_trim, data=df_trimmed).fit()
                        
                        robustness_check = {
                            "did_estimate": float(model_trim.params['_interaction']),
                            "did_pvalue": float(model_trim.pvalues['_interaction']),
                            "significant": bool(model_trim.pvalues['_interaction'] < alpha),
                            "n_excluded": n_excluded,
                            "method": "5% trimmed"
                        }
                except Exception:
                    pass
                
                # Sensitivity plot
                sens_plot = None
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    models = ['Base Model']
                    estimates = [base_model["did_estimate"]]
                    errors = [did_analysis.get("did_se", 0)]
                    colors_list = ['#5B9BD5']
                    
                    if with_covariates:
                        models.append('With Covariates')
                        estimates.append(with_covariates["did_estimate"])
                        errors.append(0)  # SE not easily available
                        colors_list.append('#ED7D31')
                    
                    if robustness_check:
                        models.append('Trimmed (5%)')
                        estimates.append(robustness_check["did_estimate"])
                        errors.append(0)
                        colors_list.append('#70AD47')
                    
                    x = np.arange(len(models))
                    bars = ax.bar(x, estimates, color=colors_list, edgecolor='black')
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(models)
                    ax.set_ylabel('DID Estimate')
                    ax.set_title('Sensitivity Analysis: DID Estimates Across Models', fontweight='bold')
                    
                    # Add value labels
                    for bar, est in zip(bars, estimates):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{est:.3f}', ha='center', va='bottom', fontsize=10)
                    
                    plt.tight_layout()
                    sens_plot = fig_to_base64(fig)
                except Exception:
                    pass
                
                sensitivity_analysis = {
                    "base_model": base_model,
                    "with_covariates": with_covariates,
                    "robustness_check": robustness_check,
                    "plot": sens_plot
                }
                
            except Exception as e:
                sensitivity_analysis = {"error": str(e)}
        
        # ========================================
        # 7. OVERALL CONCLUSION
        # ========================================
        evidence_points = []
        conclusion_score = 0
        
        # Pre-post significance
        if pre_post_comparison and pre_post_comparison.get("overall"):
            pp = pre_post_comparison["overall"]
            if pp["significant"]:
                conclusion_score += 1
                evidence_points.append({
                    "interpretation": f"Pre-post change is statistically significant",
                    "statistic": f"p = {pp['p_value']:.4f}",
                    "significant": True
                })
            else:
                evidence_points.append({
                    "interpretation": f"Pre-post change is not statistically significant",
                    "statistic": f"p = {pp['p_value']:.4f}",
                    "significant": False
                })
            
            # Effect size
            if abs(pp["effect_size"]) >= 0.5:
                conclusion_score += 1
                evidence_points.append({
                    "interpretation": f"Effect size is {pp['effect_interpretation']}",
                    "statistic": f"d = {pp['effect_size']:.3f}",
                    "significant": True
                })
            else:
                evidence_points.append({
                    "interpretation": f"Effect size is {pp['effect_interpretation']}",
                    "statistic": f"d = {pp['effect_size']:.3f}",
                    "significant": False
                })
        
        # DID significance (most important)
        if did_analysis and did_analysis.get("did_estimate") is not None:
            if did_analysis["significant"]:
                conclusion_score += 2  # DID counts more
                evidence_points.append({
                    "interpretation": "DID estimate is statistically significant (causal evidence)",
                    "statistic": f"β = {did_analysis['did_estimate']:.3f}, p = {did_analysis['did_pvalue']:.4f}",
                    "significant": True
                })
            else:
                evidence_points.append({
                    "interpretation": "DID estimate is not statistically significant",
                    "statistic": f"β = {did_analysis['did_estimate']:.3f}, p = {did_analysis['did_pvalue']:.4f}",
                    "significant": False
                })
        
        # Sensitivity robustness
        if sensitivity_analysis and sensitivity_analysis.get("base_model") and sensitivity_analysis.get("robustness_check"):
            if sensitivity_analysis["base_model"]["significant"] == sensitivity_analysis["robustness_check"]["significant"]:
                conclusion_score += 1
                evidence_points.append({
                    "interpretation": "Results are robust across model specifications",
                    "statistic": "Consistent significance",
                    "significant": True
                })
        
        # Determine conclusion
        if conclusion_score >= 4:
            conclusion = "EFFECTIVE"
            conclusion_text = f"Strong evidence supports that the intervention had a significant effect on {outcome_var}."
            confidence_level = "high"
            recommendation = "The intervention appears effective. Consider scaling or continuing the program."
        elif conclusion_score >= 2:
            conclusion = "LIKELY EFFECTIVE"
            conclusion_text = f"Moderate evidence suggests the intervention may have affected {outcome_var}, but findings should be interpreted with caution."
            confidence_level = "medium"
            recommendation = "Results are promising but warrant further investigation or replication."
        else:
            conclusion = "NO CLEAR EFFECT"
            conclusion_text = f"Insufficient evidence to conclude that the intervention affected {outcome_var}."
            confidence_level = "low"
            recommendation = "Consider collecting more data, refining the intervention, or exploring alternative approaches."
        
        overall_conclusion = {
            "conclusion": conclusion,
            "conclusion_text": conclusion_text,
            "confidence_level": confidence_level,
            "evidence_points": evidence_points,
            "recommendation": recommendation
        }
        
        # ========================================
        # FINAL RESPONSE
        # ========================================
        return {
            "descriptive_stats": descriptive_stats,
            "pre_post_comparison": pre_post_comparison,
            "did_analysis": did_analysis,
            "trend_analysis": trend_analysis,
            "sensitivity_analysis": sensitivity_analysis,
            "effect_size_analysis": effect_size_analysis,
            "overall_conclusion": overall_conclusion,
            "summary_statistics": {
                "n_total": n_total,
                "n_valid": n_valid,
                "outcome_var": outcome_var,
                "time_var": time_var,
                "group_var": group_var,
                "covariates": covariates
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
