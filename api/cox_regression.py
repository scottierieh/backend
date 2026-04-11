"""
Cox Proportional Hazards Regression Router for FastAPI
Survival analysis for time-to-event data
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CoxRegressionRequest(BaseModel):
    data: List[Dict[str, Any]]
    duration_col: str  # Time to event
    event_col: str  # Event indicator (1 = event occurred, 0 = censored)
    covariate_cols: List[str]  # Predictor variables
    # Cox model parameters
    penalizer: float = 0.0  # L2 regularization
    l1_ratio: float = 0.0  # Elastic net mixing (0 = L2, 1 = L1)
    alpha: float = 0.05  # Significance level for confidence intervals
    robust: bool = False  # Use robust standard errors
    # Analysis options
    check_assumptions: bool = True  # Test proportional hazards assumption


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
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def fit_cox_model(df: pd.DataFrame, duration_col: str, event_col: str, 
                  covariate_cols: List[str], params: dict) -> Dict[str, Any]:
    """Fit Cox Proportional Hazards model"""
    
    # Prepare data
    analysis_cols = [duration_col, event_col] + covariate_cols
    analysis_df = df[analysis_cols].copy()
    
    # Handle covariates: try numeric, fallback to categorical encoding
    for col in covariate_cols:
        # Try numeric conversion
        numeric_col = pd.to_numeric(analysis_df[col], errors='coerce')
        
        # If conversion fails (all NaN) or mostly fails, treat as categorical
        if numeric_col.isna().all() or (numeric_col.isna().sum() / len(numeric_col) > 0.5):
            # Categorical variable
            analysis_df[col] = analysis_df[col].astype(str)
            dummies = pd.get_dummies(analysis_df[col], prefix=col, drop_first=True)
            analysis_df = pd.concat([analysis_df.drop(col, axis=1), dummies], axis=1)
        else:
            # Numeric variable
            analysis_df[col] = numeric_col
    
    # Remove rows with missing values
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) < 50:
        raise ValueError("At least 50 valid samples required after removing missing values.")
    
    # Fit Cox model
    cph = CoxPHFitter(
        penalizer=params['penalizer'],
        l1_ratio=params['l1_ratio']
    )
    
    cph.fit(
        analysis_df, 
        duration_col=duration_col, 
        event_col=event_col,
        robust=params['robust']
    )
    
    # Extract results
    summary = cph.summary

    # CI column names depend on alpha — lifelines uses e.g. "coef lower 95%" for alpha=0.05
    ci_pct = int(round((1 - params['alpha']) * 100))
    coef_lower_col = f'coef lower {ci_pct}%'
    coef_upper_col = f'coef upper {ci_pct}%'
    exp_lower_col  = f'exp(coef) lower {ci_pct}%'
    exp_upper_col  = f'exp(coef) upper {ci_pct}%'

    # Fallback: if exact column not found (e.g. alpha=0.055 → 94%), find closest match
    def _find_col(summary, target):
        if target in summary.columns:
            return target
        for col in summary.columns:
            if col.startswith(target.split(' ')[0] + ' ' + target.split(' ')[1]):
                return col
        return None

    coef_lower_col = _find_col(summary, coef_lower_col) or 'coef lower 95%'
    coef_upper_col = _find_col(summary, coef_upper_col) or 'coef upper 95%'
    exp_lower_col  = _find_col(summary, exp_lower_col)  or 'exp(coef) lower 95%'
    exp_upper_col  = _find_col(summary, exp_upper_col)  or 'exp(coef) upper 95%'

    # Coefficients and hazard ratios
    coefficients = []
    for idx in summary.index:
        coef_data = {
            'covariate': str(idx),
            'coef': _to_native_type(summary.loc[idx, 'coef']),
            'exp_coef': _to_native_type(summary.loc[idx, 'exp(coef)']),
            'se_coef': _to_native_type(summary.loc[idx, 'se(coef)']),
            'coef_lower': _to_native_type(summary.loc[idx, coef_lower_col]),
            'coef_upper': _to_native_type(summary.loc[idx, coef_upper_col]),
            'exp_coef_lower': _to_native_type(summary.loc[idx, exp_lower_col]),
            'exp_coef_upper': _to_native_type(summary.loc[idx, exp_upper_col]),
            'z': _to_native_type(summary.loc[idx, 'z']),
            'p': _to_native_type(summary.loc[idx, 'p']),
            'significant': bool(summary.loc[idx, 'p'] < params['alpha']),
            'ci_label': f'{ci_pct}% CI'
        }
        coefficients.append(coef_data)
    
    # Sort: significant first, then by abs(coef); also keep secondary sort by HR distance from 1
    coefficients.sort(key=lambda x: (
        0 if x['significant'] else 1,
        -abs(x['coef']) if x['coef'] is not None else 0
    ))

    # Model fit statistics
    concordance = cph.concordance_index_
    log_likelihood = cph.log_likelihood_

    # Calculate AIC and BIC
    n_params = len(summary)
    n_samples = len(analysis_df)
    n_events = int(analysis_df[event_col].sum())
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + np.log(n_samples) * n_params

    # Events-per-variable warning
    epv = n_events / n_params if n_params > 0 else float('inf')
    epv_warning = None
    if epv < 10:
        epv_warning = (
            f"Low events-per-variable (EPV = {epv:.1f}; {n_events} events / {n_params} covariates). "
            f"EPV < 10 may lead to unstable estimates. Consider reducing covariates or collecting more data."
        )

    # Convergence / separation warnings
    model_warnings = []
    if epv_warning:
        model_warnings.append(epv_warning)
    if params.get('robust'):
        model_warnings.append("Robust standard errors used — p-values and CIs are based on sandwich variance estimator.")
    
    metrics = {
        'concordance_index': _to_native_type(concordance),
        'log_likelihood': _to_native_type(log_likelihood),
        'log_likelihood_ratio_test': _to_native_type(cph.log_likelihood_ratio_test().test_statistic),
        'log_likelihood_ratio_p': _to_native_type(cph.log_likelihood_ratio_test().p_value),
        'aic': _to_native_type(aic),
        'bic': _to_native_type(bic),
        'n_events': n_events,
        'n_censored': int(len(analysis_df) - n_events),
        'event_rate': _to_native_type(analysis_df[event_col].mean()),
        'epv': _to_native_type(epv),
        'ci_level': ci_pct
    }

    return {
        'model': cph,
        'coefficients': coefficients,
        'metrics': metrics,
        'model_warnings': model_warnings,
        'analysis_df': analysis_df,
        'duration_col': duration_col,
        'event_col': event_col
    }


def test_proportional_hazards(cph, analysis_df: pd.DataFrame, duration_col: str, event_col: str) -> Dict[str, Any]:
    """Test proportional hazards assumption"""
    try:
        results = proportional_hazard_test(cph, analysis_df, time_transform='rank')
        
        ph_tests = []
        for idx in results.summary.index:
            ph_tests.append({
                'covariate': str(idx),
                'test_statistic': _to_native_type(results.summary.loc[idx, 'test_statistic']),
                'p': _to_native_type(results.summary.loc[idx, 'p']),
                'assumption_met': bool(results.summary.loc[idx, 'p'] > 0.05)
            })
        
        # Overall test
        overall_p = results.summary['p'].min()
        
        return {
            'tests': ph_tests,
            'overall_p': _to_native_type(overall_p),
            'assumption_met': bool(overall_p > 0.05),
            'interpretation': 'Proportional hazards assumption is met.' if overall_p > 0.05 
                            else 'Proportional hazards assumption may be violated. Consider stratification or time-varying covariates.'
        }
    except Exception as e:
        return {
            'tests': [],
            'overall_p': None,
            'assumption_met': None,
            'interpretation': f'Could not test assumption: {str(e)}'
        }


def generate_forest_plot(coefficients: List[Dict]) -> str:
    """Generate forest plot for hazard ratios"""
    fig, ax = plt.subplots(figsize=(10, max(6, len(coefficients) * 0.5)))
    
    # Filter to top 15 covariates
    plot_data = coefficients[:15]
    
    y_pos = list(range(len(plot_data)))
    y_pos.reverse()
    
    names = [d['covariate'] for d in plot_data]
    hrs = [d['exp_coef'] for d in plot_data]
    hr_lower = [d['exp_coef_lower'] for d in plot_data]
    hr_upper = [d['exp_coef_upper'] for d in plot_data]
    significant = [d['significant'] for d in plot_data]
    
    # Calculate error bars
    xerr_lower = [hr - low for hr, low in zip(hrs, hr_lower)]
    xerr_upper = [up - hr for hr, up in zip(hrs, hr_upper)]
    
    # Plot
    colors = ['#22c55e' if sig else '#94a3b8' for sig in significant]
    
    for i, (y, hr, low, up, color) in enumerate(zip(y_pos, hrs, xerr_lower, xerr_upper, colors)):
        ax.errorbar(hr, y, xerr=[[low], [up]], fmt='o', color=color, 
                   capsize=4, capthick=2, markersize=8, elinewidth=2)
    
    # Reference line at HR = 1
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(f'Hazard Ratio ({coefficients[0].get("ci_label", "95% CI")})', fontsize=11)
    ax.set_title('Forest Plot - Hazard Ratios', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Set x-axis to log scale if range is large (HR plots are more natural on log scale)
    positive_hrs = [hr for hr in hrs if hr is not None and hr > 0]
    if positive_hrs and max(positive_hrs) / min(positive_hrs) > 10:
        ax.set_xscale('log')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#22c55e', markersize=10, label='Significant (p < 0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#94a3b8', markersize=10, label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_baseline_survival_plot(cph, analysis_df: pd.DataFrame, duration_col: str, event_col: str) -> str:
    """Generate baseline survival curve from Cox model"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get baseline survival
    baseline_survival = cph.baseline_survival_
    
    # Plot baseline survival
    ax.step(baseline_survival.index, baseline_survival.values, where='post', 
            color='#22c55e', linewidth=2, label='Baseline Survival')
    
    # Calculate median survival time
    try:
        median_idx = np.searchsorted(baseline_survival.values[::-1], 0.5)
        if median_idx < len(baseline_survival):
            median_time = baseline_survival.index[::-1][median_idx]
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=median_time, color='gray', linestyle=':', alpha=0.5)
            ax.annotate(f'Median: {median_time:.1f}', xy=(median_time, 0.5), 
                       xytext=(median_time + 5, 0.55), fontsize=9)
    except:
        pass
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title('Baseline Survival Function', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_hazard_ratios_plot(coefficients: List[Dict]) -> str:
    """Generate horizontal bar plot of hazard ratios"""
    fig, ax = plt.subplots(figsize=(10, max(6, len(coefficients[:15]) * 0.4)))
    
    plot_data = coefficients[:15]
    
    names = [d['covariate'] for d in plot_data][::-1]
    hrs = [d['exp_coef'] for d in plot_data][::-1]
    significant = [d['significant'] for d in plot_data][::-1]
    
    colors = ['#22c55e' if sig else '#94a3b8' for sig in significant]
    
    bars = ax.barh(names, hrs, color=colors, edgecolor='black', alpha=0.8)
    
    # Reference line at HR = 1
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='HR = 1 (no effect)')
    
    ax.set_xlabel('Hazard Ratio', fontsize=11)
    ax.set_title('Hazard Ratios by Covariate', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Add value labels
    for bar, hr in zip(bars, hrs):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{hr:.2f}', va='center', fontsize=9)
    
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_partial_effects_plot(cph, analysis_df: pd.DataFrame, duration_col: str, event_col: str) -> str:
    """Generate partial effects on outcome plot (per-covariate survival curves at mean ± 1SD)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    try:
        # Get top 4 covariates
        covariates = list(cph.summary.index)[:4]
        
        for i, covariate in enumerate(covariates):
            if i >= 4:
                break
            ax = axes[i]
            cph.plot_partial_effects_on_outcome(covariate, 
                                                 values=[cph.summary.loc[covariate, 'coef'] - 1,
                                                        cph.summary.loc[covariate, 'coef'],
                                                        cph.summary.loc[covariate, 'coef'] + 1],
                                                 ax=ax)
            ax.set_title(f'Partial Effect: {covariate}', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(covariates), 4):
            axes[i].set_visible(False)
            
    except Exception as e:
        # If partial effects fail, create placeholder
        for ax in axes:
            ax.text(0.5, 0.5, 'Partial effects plot not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(True)
    
    plt.suptitle('Partial Effects on Survival', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_log_log_plot(cph, analysis_df: pd.DataFrame) -> str:
    """Generate log-log plot to check PH assumption"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # Get baseline cumulative hazard
        baseline_cumhaz = cph.baseline_cumulative_hazard_
        
        # Log-log transformation
        times = baseline_cumhaz.index[baseline_cumhaz.values.flatten() > 0]
        cumhaz = baseline_cumhaz.values.flatten()[baseline_cumhaz.values.flatten() > 0]
        
        log_time = np.log(times)
        log_cumhaz = np.log(cumhaz)
        
        ax.plot(log_time, log_cumhaz, 'o-', color='#22c55e', markersize=4, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(log_time, log_cumhaz, 1)
        p = np.poly1d(z)
        ax.plot(log_time, p(log_time), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
        
        ax.set_xlabel('log(Time)', fontsize=11)
        ax.set_ylabel('log(Cumulative Hazard)', fontsize=11)
        ax.set_title('Log-Log Plot (PH Assumption Check)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Log-log plot not available', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, ph_test: Dict, params: dict) -> Dict[str, Any]:
    """Generate interpretation of Cox regression results"""
    key_insights = []
    
    metrics = result['metrics']
    coefficients = result['coefficients']
    
    # Concordance index interpretation
    c_index = metrics['concordance_index']
    if c_index >= 0.8:
        status = 'positive'
        c_desc = 'Excellent discrimination'
    elif c_index >= 0.7:
        status = 'neutral'
        c_desc = 'Good discrimination'
    elif c_index >= 0.6:
        status = 'neutral'
        c_desc = 'Moderate discrimination'
    else:
        status = 'warning'
        c_desc = 'Poor discrimination'
    
    key_insights.append({
        'title': 'Model Discrimination',
        'description': f'{c_desc}. Concordance Index (C-index) = {c_index:.3f}. Values > 0.7 indicate good predictive ability.',
        'status': status
    })
    
    # Event rate
    event_rate = metrics['event_rate']
    key_insights.append({
        'title': 'Event Rate',
        'description': f'{metrics["n_events"]} events observed out of {metrics["n_events"] + metrics["n_censored"]} subjects ({event_rate:.1%} event rate).',
        'status': 'neutral'
    })
    
    # Significant predictors
    sig_covariates = [c for c in coefficients if c['significant']]
    if len(sig_covariates) > 0:
        top_sig = sig_covariates[:3]
        sig_desc = ', '.join([f"{c['covariate']} (HR={c['exp_coef']:.2f}, p={c['p']:.3f})" for c in top_sig])
        key_insights.append({
            'title': 'Significant Predictors',
            'description': f'{len(sig_covariates)} significant covariate(s) found. Top predictors: {sig_desc}',
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'No Significant Predictors',
            'description': 'No covariates reached statistical significance at α = 0.05. Consider different predictors or larger sample size.',
            'status': 'warning'
        })
    
    # Hazard ratio interpretation for top predictor
    if len(coefficients) > 0:
        top_coef = coefficients[0]
        hr = top_coef['exp_coef']
        if hr > 1:
            hr_interp = f'increases hazard by {(hr-1)*100:.1f}%'
        else:
            hr_interp = f'decreases hazard by {(1-hr)*100:.1f}%'
        
        key_insights.append({
            'title': 'Strongest Predictor',
            'description': f'{top_coef["covariate"]} has HR = {hr:.2f}, meaning a one-unit increase {hr_interp}.',
            'status': 'neutral'
        })
    
    # EPV warning
    epv = metrics.get('epv')
    if epv is not None and epv < 10:
        key_insights.append({
            'title': 'Low Events-Per-Variable (EPV)',
            'description': (
                f'EPV = {epv:.1f} ({metrics["n_events"]} events / {metrics.get("n_covariates", "?")} covariates). '
                f'EPV < 10 may lead to unstable or biased estimates. Consider reducing covariates or using penalization.'
            ),
            'status': 'warning'
        })

    # Robust SE note
    if params.get('robust'):
        key_insights.append({
            'title': 'Robust Standard Errors',
            'description': 'Robust (sandwich) variance estimator was used. SE, z-statistics, and p-values reflect robust estimates.',
            'status': 'neutral'
        })
        key_insights.append({
            'title': 'PH Assumption',
            'description': ph_test['interpretation'],
            'status': 'positive' if ph_test['assumption_met'] else 'warning'
        })
    
    # Model fit
    llr_p = metrics['log_likelihood_ratio_p']
    if llr_p < 0.001:
        fit_desc = 'Model is highly significant'
    elif llr_p < 0.05:
        fit_desc = 'Model is significant'
    else:
        fit_desc = 'Model is not significant'
    
    key_insights.append({
        'title': 'Overall Model Fit',
        'description': f'{fit_desc}. Log-likelihood ratio test p-value = {llr_p:.4f}. AIC = {metrics["aic"]:.1f}, BIC = {metrics["bic"]:.1f}.',
        'status': 'positive' if llr_p < 0.05 else 'warning'
    })
    
    # Recommendation
    if c_index >= 0.7 and llr_p < 0.05:
        recommendation = (
            'Cox model shows good predictive ability and significant associations. '
            'Results are ready for interpretation and reporting. '
            'Consider testing for non-linear effects (splines) and interactions if clinically relevant.'
        )
    elif c_index >= 0.6:
        recommendation = (
            'Model shows moderate predictive ability. '
            'Consider adding more predictors, interaction terms, or categorized covariates. '
            'Restricted cubic splines can capture non-linear log-hazard relationships.'
        )
    else:
        recommendation = (
            'Model has limited predictive ability. '
            'Consider alternative modeling approaches, additional covariates, or non-linear transformations. '
            'Check for competing risks, informative censoring, or model misspecification.'
        )
    
    return {
        'key_insights': key_insights,
        'recommendation': recommendation
    }


@router.post("/cox-regression")
async def run_cox_regression_analysis(request: CoxRegressionRequest) -> Dict[str, Any]:
    """
    Perform Cox Proportional Hazards regression for survival analysis.
    
    Supports:
    - Time-to-event analysis
    - Hazard ratios with confidence intervals
    - Proportional hazards assumption testing
    - Survival curves
    - Forest plots
    """
    try:
        data = request.data
        duration_col = request.duration_col
        event_col = request.event_col
        covariate_cols = request.covariate_cols
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        all_cols = [duration_col, event_col] + covariate_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")
        
        # Validate duration and event columns
        df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')
        df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
        
        if df[duration_col].isna().all():
            raise HTTPException(status_code=400, detail="Duration column must be numeric.")

        # Duration must be positive
        if (df[duration_col].dropna() <= 0).any():
            raise HTTPException(status_code=400, detail="Duration column must contain only positive values (> 0). Cox regression requires positive survival times.")
        
        if not set(df[event_col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            raise HTTPException(status_code=400, detail="Event column must contain only 0 (censored) and 1 (event).")
        
        # Parameters
        params = {
            'penalizer': request.penalizer,
            'l1_ratio': request.l1_ratio,
            'alpha': request.alpha,
            'robust': request.robust,
            'check_assumptions': request.check_assumptions
        }
        
        # Fit model
        result = fit_cox_model(df, duration_col, event_col, covariate_cols, params)
        
        # Test PH assumption
        ph_test = {'tests': [], 'overall_p': None, 'assumption_met': None, 'interpretation': 'Not tested'}
        if request.check_assumptions:
            ph_test = test_proportional_hazards(
                result['model'], 
                result['analysis_df'], 
                duration_col, 
                event_col
            )
        
        # Generate visualizations
        forest_plot = generate_forest_plot(result['coefficients'])
        survival_plot = generate_baseline_survival_plot(
            result['model'], result['analysis_df'], duration_col, event_col
        )
        hazard_plot = generate_hazard_ratios_plot(result['coefficients'])
        log_log_plot = generate_log_log_plot(result['model'], result['analysis_df'])
        partial_effects_plot = generate_partial_effects_plot(
            result['model'], result['analysis_df'], duration_col, event_col
        )

        # Generate interpretation
        result['metrics']['n_covariates'] = len(covariate_cols)
        interpretation = generate_interpretation(result, ph_test, params)

        # Prepare response
        response = {
            'n_samples': len(result['analysis_df']),
            'n_covariates': len(covariate_cols),
            'n_events': result['metrics']['n_events'],
            'n_censored': result['metrics']['n_censored'],
            'ci_level': result['metrics']['ci_level'],
            'model_warnings': result.get('model_warnings', []),
            'parameters': params,
            'metrics': result['metrics'],
            'coefficients': result['coefficients'],
            'ph_test': ph_test,
            'forest_plot': forest_plot,
            'survival_plot': survival_plot,
            'hazard_plot': hazard_plot,
            'log_log_plot': log_log_plot,
            'partial_effects_plot': partial_effects_plot,
            'interpretation': interpretation
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cox regression analysis failed: {str(e)}")
