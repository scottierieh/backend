from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter, LogNormalAFTFitter, WeibullAFTFitter, NelsonAalenFitter
    from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
    from lifelines.utils import restricted_mean_survival_time
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

router = APIRouter()

class SurvivalRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    durationCol: str = Field(...)
    eventCol: str = Field(...)
    groupCol: Optional[str] = None
    covariates: Optional[List[str]] = []
    modelType: Optional[str] = 'all'

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    elif pd.isna(obj): return None
    return obj

def safe_float(val, default=None):
    try:
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except: return default

def generate_interpretation(results, duration_col, event_col, group_col, covariates, n_obs):
    parts = []
    data_summary = results.get('data_summary', {})
    km_results = results.get('kaplan_meier', {})
    log_rank = results.get('log_rank_test', {})
    cox_results = results.get('cox_ph', {})
    aft_weibull = results.get('aft_weibull', {})
    aft_lognormal = results.get('aft_lognormal', {})
    
    total_subjects = data_summary.get('total_subjects', n_obs)
    total_events = data_summary.get('total_events', 0)
    censored = data_summary.get('censored', 0)
    event_rate = data_summary.get('event_rate', 0)
    median_survival = km_results.get('median_survival_time')
    rmst = km_results.get('rmst')
    
    parts.append("**Overall Analysis**")
    parts.append(f"→ A survival analysis was conducted to examine time-to-event patterns for **{event_col}** using **{duration_col}** as the duration variable (N = {total_subjects}).")
    parts.append(f"→ Of {total_subjects} subjects, {total_events} experienced the event ({event_rate*100:.1f}% event rate) and {censored} were censored ({(1-event_rate)*100:.1f}% censoring rate).")
    
    if median_survival is not None and not np.isnan(median_survival):
        parts.append(f"→ The median survival time was **{median_survival:.2f}** units, indicating that 50% of subjects experienced the event by this time point.")
    else:
        parts.append("→ Median survival time could not be estimated (survival probability did not reach 50%).")
    
    if rmst is not None:
        parts.append(f"→ The restricted mean survival time (RMST) was **{rmst:.2f}**, representing the average event-free time over the observation period.")
    
    if log_rank:
        p_value = log_rank.get('p_value', 1)
        chi2 = log_rank.get('test_statistic', 0)
        df = log_rank.get('degrees_of_freedom', 1)
        is_sig = log_rank.get('is_significant', False)
        p_str = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        
        if is_sig:
            parts.append(f"→ The log-rank test revealed a **statistically significant difference** in survival curves between {group_col} groups, χ²({df}) = {chi2:.2f}, {p_str}.")
        else:
            parts.append(f"→ The log-rank test showed **no significant difference** in survival curves between {group_col} groups, χ²({df}) = {chi2:.2f}, {p_str}.")
    
    parts.append("")
    parts.append("**Key Insights**")
    
    if event_rate >= 0.5:
        parts.append(f"→ High event rate ({event_rate*100:.1f}%) provides good statistical power for survival estimates.")
    elif event_rate >= 0.2:
        parts.append(f"→ Moderate event rate ({event_rate*100:.1f}%) provides adequate statistical power.")
    else:
        parts.append(f"→ Low event rate ({event_rate*100:.1f}%) may limit precision of survival estimates; consider longer follow-up.")
    
    if cox_results and cox_results.get('summary'):
        concordance = cox_results.get('concordance', 0)
        if concordance >= 0.7: c_desc = "good"
        elif concordance >= 0.6: c_desc = "moderate"
        else: c_desc = "limited"
        parts.append(f"→ Cox proportional hazards model achieved C-index = **{concordance:.3f}**, indicating {c_desc} predictive discrimination.")
        
        sig_predictors = []
        for row in cox_results.get('summary', []):
            covariate = row.get('covariate', row.get('index', ''))
            p_val = row.get('p', 1)
            hr = row.get('exp(coef)', 1)
            if p_val < 0.05:
                direction = "increased" if hr > 1 else "decreased"
                hr_pct = abs(hr - 1) * 100
                sig_predictors.append(f"{covariate} (HR = {hr:.2f}, {direction} risk by {hr_pct:.1f}%)")
        
        if sig_predictors:
            parts.append("→ Significant predictors of survival:")
            for pred in sig_predictors[:5]:
                parts.append(f"  • {pred}")
        else:
            parts.append("→ No covariates reached statistical significance at α = .05.")
        
        ph_test = cox_results.get('proportional_hazard_assumption', {})
        if ph_test.get('passed', True):
            parts.append("→ Proportional hazards assumption was satisfied, validating model assumptions.")
        else:
            parts.append("→ **Warning**: Proportional hazards assumption may be violated; interpret with caution.")
    
    if 'kaplan_meier_grouped' in results:
        grouped = results['kaplan_meier_grouped']
        group_medians = []
        for group_name, group_data in grouped.items():
            med = group_data.get('median_survival')
            if med is not None and np.isfinite(med):
                group_medians.append((group_name, med))
        
        if len(group_medians) >= 2:
            group_medians.sort(key=lambda x: x[1], reverse=True)
            best_group = group_medians[0]
            worst_group = group_medians[-1]
            parts.append(f"→ Group **{best_group[0]}** showed longest median survival ({best_group[1]:.2f}), while **{worst_group[0]}** had shortest ({worst_group[1]:.2f}).")
    
    parts.append("")
    parts.append("**Recommendations**")
    
    events_per_covariate = total_events / max(len(covariates), 1) if covariates else total_events
    if events_per_covariate < 10:
        parts.append(f"→ **Warning**: Only {events_per_covariate:.1f} events per covariate; recommend 10-20+ for stable Cox estimates.")
    else:
        parts.append(f"→ Adequate events per covariate ratio ({events_per_covariate:.1f}) supports reliable Cox regression.")
    
    if (1 - event_rate) > 0.7:
        parts.append("→ High censoring rate (>70%) may bias survival estimates; consider informative censoring analysis.")
    
    if aft_weibull and aft_lognormal:
        aic_weibull = aft_weibull.get('aic', float('inf'))
        aic_lognormal = aft_lognormal.get('aic', float('inf'))
        if aic_weibull < aic_lognormal:
            parts.append(f"→ Weibull AFT model (AIC = {aic_weibull:.1f}) fits better than Log-Normal (AIC = {aic_lognormal:.1f}); suggests monotonic hazard pattern.")
        else:
            parts.append(f"→ Log-Normal AFT model (AIC = {aic_lognormal:.1f}) fits better than Weibull (AIC = {aic_weibull:.1f}); suggests non-monotonic hazard pattern.")
    
    if cox_results:
        parts.append("→ Report hazard ratios with 95% CIs for clinical interpretation of risk factors.")
    if log_rank and log_rank.get('is_significant'):
        parts.append("→ Significant group differences warrant targeted intervention strategies for high-risk groups.")
    parts.append("→ Consider external validation and time-dependent covariate analysis for robust conclusions.")
    
    return "\n".join(parts)

def smooth_hazard(cumulative_hazard, bandwidth=None):
    from scipy.ndimage import gaussian_filter1d
    time = cumulative_hazard.index.values
    cum_haz = cumulative_hazard.values.flatten()
    if bandwidth is None:
        bandwidth = max(1, len(time) // 20)
    smoothed_cum_haz = gaussian_filter1d(cum_haz, sigma=bandwidth)
    hazard = np.gradient(smoothed_cum_haz, time)
    hazard = np.maximum(hazard, 0)
    return pd.Series(hazard, index=time)

@router.post("/survival")
def survival_analysis(req: SurvivalRequest):
    if not LIFELINES_AVAILABLE:
        raise HTTPException(status_code=500, detail="lifelines library not installed")
    
    try:
        df = pd.DataFrame(req.data)
        duration_col = req.durationCol
        event_col = req.eventCol
        group_col = req.groupCol
        covariates = req.covariates or []
        model_type = req.modelType or 'all'

        # Data validation
        df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
        df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
        df = df.dropna(subset=[duration_col, event_col])
        df = df[df[duration_col] > 0]
        df = df[df[event_col].isin([0, 1])]

        if len(df) < 10:
            raise ValueError("Need at least 10 valid observations")

        results = {}

        # Data Summary
        total_subjects = len(df)
        total_events = int(df[event_col].sum())
        censored = total_subjects - total_events
        
        summary = {
            'total_subjects': total_subjects,
            'total_events': total_events,
            'censored': censored,
            'event_rate': total_events / total_subjects if total_subjects > 0 else 0,
            'censoring_rate': censored / total_subjects if total_subjects > 0 else 0,
            'mean_duration': safe_float(df[duration_col].mean()),
            'median_duration': safe_float(df[duration_col].median()),
            'min_duration': safe_float(df[duration_col].min()),
            'max_duration': safe_float(df[duration_col].max())
        }
        
        if group_col:
            group_summaries = {}
            for group in df[group_col].unique():
                gd = df[df[group_col] == group]
                ge = int(gd[event_col].sum())
                gt = len(gd)
                group_summaries[str(group)] = {
                    'n_subjects': gt, 'n_events': ge, 'n_censored': gt - ge,
                    'event_rate': ge / gt if gt > 0 else 0,
                    'mean_duration': safe_float(gd[duration_col].mean()),
                    'median_duration': safe_float(gd[duration_col].median())
                }
            summary['group_summaries'] = group_summaries
        results['data_summary'] = summary

        # Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(df[duration_col], df[event_col])
        
        max_time = float(df[duration_col].max())
        try:
            rmst_val = float(restricted_mean_survival_time(kmf, t=max_time))
        except:
            rmst_val = None
        
        median_surv = kmf.median_survival_time_
        if np.isinf(median_surv) or np.isnan(median_surv):
            median_surv = None
        
        results['kaplan_meier'] = {
            'survival_table': kmf.survival_function_.reset_index().rename(columns={'timeline': 'Time', 'KM_estimate': 'Survival Probability'}).to_dict('records'),
            'median_survival_time': safe_float(median_surv),
            'timeline': kmf.timeline.tolist(),
            'rmst': rmst_val
        }

        # Grouped KM
        if group_col:
            groups = df[group_col].unique()
            group_results = {}
            for group in groups:
                gd = df[df[group_col] == group]
                kmf_g = KaplanMeierFitter()
                kmf_g.fit(gd[duration_col], gd[event_col], label=str(group))
                med = kmf_g.median_survival_time_
                if np.isinf(med) or np.isnan(med): med = None
                try:
                    rmst_g = float(restricted_mean_survival_time(kmf_g, t=float(gd[duration_col].max())))
                except:
                    rmst_g = None
                group_results[str(group)] = {
                    'survival_function': kmf_g.survival_function_.reset_index().to_dict('records'),
                    'median_survival': safe_float(med),
                    'n_events': int(gd[event_col].sum()),
                    'n_subjects': len(gd),
                    'rmst': rmst_g
                }
            results['kaplan_meier_grouped'] = group_results
            
            # Log-rank test
            if len(groups) >= 2:
                lr = multivariate_logrank_test(df[duration_col], df[group_col], df[event_col])
                results['log_rank_test'] = {
                    'test_statistic': safe_float(lr.test_statistic),
                    'p_value': safe_float(lr.p_value),
                    'degrees_of_freedom': int(lr.degrees_of_freedom),
                    'is_significant': bool(lr.p_value < 0.05),
                    'test_name': lr.test_name
                }
                
                if len(groups) > 2:
                    try:
                        pw = pairwise_logrank_test(df[duration_col], df[group_col], df[event_col])
                        results['pairwise_log_rank_test'] = pw.summary.reset_index().to_dict('records')
                    except:
                        pass

        # Cox PH
        if covariates and model_type in ['cox', 'all']:
            cox_data = df[[duration_col, event_col] + covariates].dropna()
            cat_covs = [c for c in covariates if cox_data[c].dtype == 'object' or cox_data[c].dtype.name == 'category']
            if cat_covs:
                cox_data = pd.get_dummies(cox_data, columns=cat_covs, drop_first=True)
            
            used_covs = [c for c in cox_data.columns if c not in [duration_col, event_col]]
            
            try:
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
                
                summary_df = cph.summary.reset_index()
                results['cox_ph'] = {
                    'summary': summary_df.to_dict('records'),
                    'concordance': safe_float(cph.concordance_index_),
                    'log_likelihood': safe_float(cph.log_likelihood_),
                    'aic': safe_float(cph.AIC_partial_)
                }
                
                try:
                    plt.close('all')
                    cph.check_assumptions(cox_data, p_value_threshold=0.05, show_plots=False)
                    plt.close('all')
                    results['cox_ph']['proportional_hazard_assumption'] = {'passed': True}
                except:
                    results['cox_ph']['proportional_hazard_assumption'] = {'passed': False}
                
                # Risk stratification
                try:
                    risk_scores = cph.predict_partial_hazard(cox_data[used_covs])
                    results['risk_stratification'] = {
                        'risk_scores_mean': safe_float(risk_scores.mean()),
                        'risk_scores_std': safe_float(risk_scores.std()),
                        'risk_scores_min': safe_float(risk_scores.min()),
                        'risk_scores_max': safe_float(risk_scores.max())
                    }
                except:
                    pass
            except Exception as e:
                results['cox_ph'] = {'error': str(e)}

        # AFT Models
        if covariates and model_type in ['all', 'aft_weibull', 'aft_lognormal']:
            aft_data = df[[duration_col, event_col] + covariates].dropna()
            cat_covs = [c for c in covariates if aft_data[c].dtype == 'object']
            if cat_covs:
                aft_data = pd.get_dummies(aft_data, columns=cat_covs, drop_first=True)
            
            if model_type in ['all', 'aft_weibull']:
                try:
                    aft_w = WeibullAFTFitter()
                    aft_w.fit(aft_data, duration_col=duration_col, event_col=event_col)
                    results['aft_weibull'] = {
                        'summary': aft_w.summary.reset_index().to_dict('records'),
                        'log_likelihood': safe_float(aft_w.log_likelihood_),
                        'aic': safe_float(aft_w.AIC_)
                    }
                except:
                    pass
            
            if model_type in ['all', 'aft_lognormal']:
                try:
                    aft_ln = LogNormalAFTFitter()
                    aft_ln.fit(aft_data, duration_col=duration_col, event_col=event_col)
                    results['aft_lognormal'] = {
                        'summary': aft_ln.summary.reset_index().to_dict('records'),
                        'log_likelihood': safe_float(aft_ln.log_likelihood_),
                        'aic': safe_float(aft_ln.AIC_)
                    }
                except:
                    pass

        # Interpretation
        interpretation = generate_interpretation(results, duration_col, event_col, group_col, covariates, len(df))
        results['interpretation'] = interpretation

        # Plots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        line_color = '#C44E52'
        
        # 1. KM Curve
        kmf.plot_survival_function(ax=axes[0, 0], ci_show=True)
        axes[0, 0].set_title('Kaplan-Meier Survival Curve', fontweight='bold')
        if median_surv:
            axes[0, 0].axvline(median_surv, color=line_color, linestyle='--', lw=2, label=f'Median: {median_surv:.2f}')
        axes[0, 0].legend()
        
        # 2. Cumulative Hazard
        naf = NelsonAalenFitter()
        naf.fit(df[duration_col], event_observed=df[event_col])
        naf.plot_cumulative_hazard(ax=axes[0, 1], ci_show=True)
        axes[0, 1].set_title('Cumulative Hazard Function', fontweight='bold')
        
        # 3. Smoothed Hazard
        try:
            hazard = smooth_hazard(naf.cumulative_hazard_)
            axes[0, 2].plot(hazard.index, hazard.values, color=line_color, lw=2)
            axes[0, 2].fill_between(hazard.index, 0, hazard.values, alpha=0.3, color=line_color)
            axes[0, 2].set_title('Smoothed Hazard Function', fontweight='bold')
        except:
            axes[0, 2].text(0.5, 0.5, 'Hazard estimation unavailable', ha='center', va='center')
        
        # 4. Group Survival
        if group_col:
            colors = sns.color_palette('husl', len(df[group_col].unique()))
            for idx, group in enumerate(df[group_col].unique()):
                gd = df[df[group_col] == group]
                kmf_g = KaplanMeierFitter()
                kmf_g.fit(gd[duration_col], gd[event_col], label=str(group))
                kmf_g.plot_survival_function(ax=axes[1, 0], color=colors[idx])
            axes[1, 0].set_title(f'Survival by {group_col}', fontweight='bold')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No group variable', ha='center', va='center')
            axes[1, 0].set_title('Grouped Survival', fontweight='bold')
        
        # 5. Risk Groups (if Cox available)
        if 'cox_ph' in results and 'error' not in results['cox_ph']:
            try:
                cox_data = df[[duration_col, event_col] + covariates].dropna()
                cat_covs = [c for c in covariates if df[c].dtype == 'object']
                if cat_covs:
                    cox_data = pd.get_dummies(cox_data, columns=cat_covs, drop_first=True)
                used_covs = [c for c in cox_data.columns if c not in [duration_col, event_col]]
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
                risk_scores = cph.predict_partial_hazard(cox_data[used_covs])
                risk_groups = pd.qcut(risk_scores, q=[0, .33, .66, 1], labels=['Low', 'Medium', 'High'], duplicates='drop')
                cox_data['risk_group'] = risk_groups
                
                risk_colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
                for grp in ['Low', 'Medium', 'High']:
                    gdf = cox_data[cox_data['risk_group'] == grp]
                    if not gdf.empty:
                        kmf_r = KaplanMeierFitter()
                        kmf_r.fit(gdf[duration_col], gdf[event_col], label=grp)
                        kmf_r.plot_survival_function(ax=axes[1, 1], color=risk_colors[grp])
                axes[1, 1].set_title('Survival by Risk Group', fontweight='bold')
                axes[1, 1].legend()
            except:
                axes[1, 1].text(0.5, 0.5, 'Risk stratification unavailable', ha='center', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Cox model', ha='center', va='center')
            axes[1, 1].set_title('Survival by Risk Group', fontweight='bold')
        
        # 6. Hazard by Group
        if group_col:
            try:
                colors = sns.color_palette('husl', len(df[group_col].unique()))
                for idx, group in enumerate(df[group_col].unique()):
                    gd = df[df[group_col] == group]
                    naf_g = NelsonAalenFitter()
                    naf_g.fit(gd[duration_col], gd[event_col])
                    hazard_g = smooth_hazard(naf_g.cumulative_hazard_)
                    axes[1, 2].plot(hazard_g.index, hazard_g.values, color=colors[idx], lw=2, label=str(group))
                axes[1, 2].set_title(f'Hazard by {group_col}', fontweight='bold')
                axes[1, 2].legend()
            except:
                axes[1, 2].text(0.5, 0.5, 'Hazard by group unavailable', ha='center', va='center')
        else:
            axes[1, 2].text(0.5, 0.5, 'No group variable', ha='center', va='center')
            axes[1, 2].set_title('Hazard by Group', fontweight='bold')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

        return _to_native({'results': results, 'plot': plot})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
