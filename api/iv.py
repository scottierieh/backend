"""
Instrumental Variable (IV) Analysis Router for FastAPI
Estimate causal effects using instrumental variables (2SLS)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.iv import IV2SLS as LM_IV2SLS
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class IVRequest(BaseModel):
    data: List[Dict[str, Any]]
    outcome_col: str  # Y: Dependent variable
    endogenous_col: str  # X: Endogenous regressor
    instrument_cols: List[str]  # Z: Instrumental variables
    exogenous_cols: Optional[List[str]] = None  # W: Exogenous controls
    robust_se: bool = True  # Use heteroskedasticity-robust SE


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
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def run_ols_regression(df: pd.DataFrame, outcome_col: str, endogenous_col: str,
                        exogenous_cols: List[str] = None) -> Dict[str, Any]:
    """Run OLS regression (potentially biased)"""
    
    regressors = [endogenous_col]
    if exogenous_cols:
        regressors.extend(exogenous_cols)
    
    X = df[regressors].copy()
    X = sm.add_constant(X)
    y = df[outcome_col]
    
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    coefficients = []
    for param in model.params.index:
        coefficients.append({
            'term': param,
            'estimate': _to_native_type(model.params[param]),
            'std_error': _to_native_type(model.bse[param]),
            't_value': _to_native_type(model.tvalues[param]),
            'p_value': _to_native_type(model.pvalues[param]),
            'ci_lower': _to_native_type(model.conf_int().loc[param, 0]),
            'ci_upper': _to_native_type(model.conf_int().loc[param, 1])
        })
    
    return {
        'coefficients': coefficients,
        'endogenous_effect': _to_native_type(model.params[endogenous_col]),
        'endogenous_se': _to_native_type(model.bse[endogenous_col]),
        'endogenous_pvalue': _to_native_type(model.pvalues[endogenous_col]),
        'r_squared': _to_native_type(model.rsquared),
        'adj_r_squared': _to_native_type(model.rsquared_adj),
        'n_obs': int(model.nobs)
    }


def run_first_stage(df: pd.DataFrame, endogenous_col: str, instrument_cols: List[str],
                     exogenous_cols: List[str] = None) -> Dict[str, Any]:
    """Run first stage regression: X = π*Z + γ*W + v"""
    
    regressors = instrument_cols.copy()
    if exogenous_cols:
        regressors.extend(exogenous_cols)
    
    X = df[regressors].copy()
    X = sm.add_constant(X)
    y = df[endogenous_col]
    
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    # Extract instrument coefficients
    instrument_coefs = []
    for inst in instrument_cols:
        instrument_coefs.append({
            'instrument': inst,
            'coefficient': _to_native_type(model.params[inst]),
            'std_error': _to_native_type(model.bse[inst]),
            't_value': _to_native_type(model.tvalues[inst]),
            'p_value': _to_native_type(model.pvalues[inst]),
            'significant': bool(model.pvalues[inst] < 0.05)
        })
    
    # F-statistic for instruments (weak instrument test)
    # Test joint significance of instruments
    r_matrix = np.zeros((len(instrument_cols), len(model.params)))
    for i, inst in enumerate(instrument_cols):
        r_matrix[i, list(model.params.index).index(inst)] = 1
    
    try:
        f_test = model.f_test(r_matrix)
        f_statistic = float(f_test.fvalue)
        f_pvalue = float(f_test.pvalue)
    except:
        f_statistic = model.fvalue
        f_pvalue = model.f_pvalue
    
    # Weak instrument rule of thumb: F > 10
    weak_instrument = f_statistic < 10
    
    return {
        'instrument_coefficients': instrument_coefs,
        'f_statistic': _to_native_type(f_statistic),
        'f_pvalue': _to_native_type(f_pvalue),
        'weak_instrument': weak_instrument,
        'r_squared': _to_native_type(model.rsquared),
        'partial_r_squared': _to_native_type(model.rsquared),  # Simplified
        'fitted_values': model.fittedvalues.values,
        'residuals': model.resid.values
    }


def run_2sls(df: pd.DataFrame, outcome_col: str, endogenous_col: str,
              instrument_cols: List[str], exogenous_cols: List[str] = None,
              robust_se: bool = True) -> Dict[str, Any]:
    """Run Two-Stage Least Squares (2SLS) estimation"""
    
    try:
        # Prepare data for linearmodels
        y = df[outcome_col]
        endog = df[[endogenous_col]]
        instruments = df[instrument_cols]
        
        if exogenous_cols:
            exog = sm.add_constant(df[exogenous_cols])
        else:
            exog = pd.DataFrame({'const': np.ones(len(df))})
        
        # Run IV2SLS
        model = LM_IV2SLS(y, exog, endog, instruments).fit(cov_type='robust' if robust_se else 'unadjusted')
        
        coefficients = []
        for param in model.params.index:
            coefficients.append({
                'term': param,
                'estimate': _to_native_type(model.params[param]),
                'std_error': _to_native_type(model.std_errors[param]),
                't_value': _to_native_type(model.tstats[param]),
                'p_value': _to_native_type(model.pvalues[param]),
                'ci_lower': _to_native_type(model.conf_int().loc[param, 'lower']),
                'ci_upper': _to_native_type(model.conf_int().loc[param, 'upper'])
            })
        
        return {
            'coefficients': coefficients,
            'endogenous_effect': _to_native_type(model.params[endogenous_col]),
            'endogenous_se': _to_native_type(model.std_errors[endogenous_col]),
            'endogenous_pvalue': _to_native_type(model.pvalues[endogenous_col]),
            'endogenous_ci': [
                _to_native_type(model.conf_int().loc[endogenous_col, 'lower']),
                _to_native_type(model.conf_int().loc[endogenous_col, 'upper'])
            ],
            'significant': bool(model.pvalues[endogenous_col] < 0.05),
            'r_squared': _to_native_type(model.rsquared),
            'n_obs': int(model.nobs),
            'residual_ss': _to_native_type(model.resid_ss),
            'fitted_values': model.fitted_values.values.flatten().tolist()  # 리스트로 변환하여 길이 보장
        }
    except Exception as e:
        return {'error': str(e)}


def run_diagnostic_tests(df: pd.DataFrame, outcome_col: str, endogenous_col: str,
                          instrument_cols: List[str], exogenous_cols: List[str],
                          first_stage: Dict, second_stage: Dict) -> Dict[str, Any]:
    """Run IV diagnostic tests"""
    
    diagnostics = {}
    
    # 1. Weak Instrument Test (from first stage)
    diagnostics['weak_instrument_test'] = {
        'f_statistic': first_stage['f_statistic'],
        'critical_value': 10,
        'weak_instrument': first_stage['weak_instrument'],
        'message': 'Weak instruments detected (F < 10)' if first_stage['weak_instrument'] else 'Instruments are strong (F ≥ 10)'
    }
    
    # 2. Overidentification Test (if more instruments than endogenous)
    if len(instrument_cols) > 1:
        # Sargan-Hansen J test (simplified version)
        try:
            # Get 2SLS residuals
            y = df[outcome_col].values
            X = df[[endogenous_col]].values
            if exogenous_cols:
                W = df[exogenous_cols].values
                X = np.column_stack([np.ones(len(df)), X, W])
            else:
                X = np.column_stack([np.ones(len(df)), X])
            
            Z = df[instrument_cols].values
            
            # Residuals from 2SLS
            if 'fitted_values' in second_stage and second_stage['fitted_values'] is not None:
                fitted = np.array(second_stage['fitted_values']).flatten()
                # 길이가 다른 경우 더 짧은 쪽에 맞춤
                min_len = min(len(y), len(fitted))
                y_trimmed = y[:min_len]
                fitted_trimmed = fitted[:min_len]
                residuals = y_trimmed - fitted_trimmed
                
                # Regress residuals on instruments
                Z_trimmed = Z[:min_len]
                Z_const = sm.add_constant(Z_trimmed)
                aux_model = sm.OLS(residuals, Z_const).fit()
                
                # J statistic = n * R²
                n = min_len
                j_stat = n * aux_model.rsquared
                df_j = len(instrument_cols) - 1
                j_pvalue = 1 - stats.chi2.cdf(j_stat, df_j) if df_j > 0 else 1
                
                diagnostics['overidentification_test'] = {
                    'test': 'Sargan-Hansen J',
                    'j_statistic': _to_native_type(j_stat),
                    'df': df_j,
                    'p_value': _to_native_type(j_pvalue),
                    'valid_instruments': bool(j_pvalue > 0.05),
                    'message': 'Instruments appear valid (p > 0.05)' if j_pvalue > 0.05 else 'Instruments may be invalid (p ≤ 0.05)'
                }
        except:
            diagnostics['overidentification_test'] = {'error': 'Could not compute'}
    else:
        diagnostics['overidentification_test'] = {
            'message': 'Exactly identified (# instruments = # endogenous). Cannot test overidentification.'
        }
    
    # 3. Endogeneity Test (Hausman test - comparing OLS vs 2SLS)
    try:
        ols_effect = run_ols_regression(df, outcome_col, endogenous_col, exogenous_cols)['endogenous_effect']
        iv_effect = second_stage['endogenous_effect']
        iv_se = second_stage['endogenous_se']
        ols_se = run_ols_regression(df, outcome_col, endogenous_col, exogenous_cols)['endogenous_se']
        
        # Hausman test statistic
        diff = iv_effect - ols_effect
        se_diff = np.sqrt(max(0, iv_se**2 - ols_se**2))
        
        if se_diff > 0:
            hausman_stat = (diff / se_diff) ** 2
            hausman_pvalue = 1 - stats.chi2.cdf(hausman_stat, 1)
        else:
            hausman_stat = None
            hausman_pvalue = None
        
        diagnostics['endogeneity_test'] = {
            'test': 'Hausman',
            'ols_estimate': _to_native_type(ols_effect),
            'iv_estimate': _to_native_type(iv_effect),
            'difference': _to_native_type(diff),
            'hausman_statistic': _to_native_type(hausman_stat),
            'p_value': _to_native_type(hausman_pvalue),
            'endogeneity_present': bool(hausman_pvalue < 0.05) if hausman_pvalue else None,
            'message': 'Endogeneity detected (use IV)' if hausman_pvalue and hausman_pvalue < 0.05 else 'No strong evidence of endogeneity'
        }
    except:
        diagnostics['endogeneity_test'] = {'error': 'Could not compute'}
    
    return diagnostics


def generate_first_stage_plot(df: pd.DataFrame, endogenous_col: str, 
                               instrument_cols: List[str], first_stage: Dict) -> str:
    """Generate first stage relationship plot"""
    n_instruments = len(instrument_cols)
    fig, axes = plt.subplots(1, n_instruments, figsize=(6 * n_instruments, 5))
    
    if n_instruments == 1:
        axes = [axes]
    
    for i, inst in enumerate(instrument_cols):
        ax = axes[i]
        
        # Drop NaN for this specific pair
        plot_df = df[[inst, endogenous_col]].dropna()
        x_data = plot_df[inst].values
        y_data = plot_df[endogenous_col].values
        
        # Scatter plot
        ax.scatter(x_data, y_data, alpha=0.5, color='#3b82f6', s=40)
        
        # Regression line
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, p(x_line), color='#ef4444', linewidth=2, label='First Stage Fit')
        
        # Get coefficient for this instrument
        coef_info = next((c for c in first_stage['instrument_coefficients'] if c['instrument'] == inst), None)
        if coef_info:
            ax.text(0.05, 0.95, f"π = {coef_info['coefficient']:.3f}\nt = {coef_info['t_value']:.2f}",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(f'{inst} (Instrument)', fontsize=11)
        ax.set_ylabel(f'{endogenous_col} (Endogenous)', fontsize=11)
        ax.set_title(f'First Stage: {inst}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_comparison_plot(ols_result: Dict, iv_result: Dict, endogenous_col: str) -> str:
    """Generate OLS vs IV comparison plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['OLS\n(Potentially Biased)', '2SLS\n(IV Estimate)']
    estimates = [ols_result['endogenous_effect'], iv_result['endogenous_effect']]
    errors = [1.96 * ols_result['endogenous_se'], 1.96 * iv_result['endogenous_se']]
    colors = ['#ef4444', '#22c55e']
    
    bars = ax.bar(methods, estimates, yerr=errors, capsize=10, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, est, err in zip(bars, estimates, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
                f'{est:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel(f'Effect of {endogenous_col}', fontsize=12)
    ax.set_title('OLS vs Instrumental Variable Estimates', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_residual_plot(df: pd.DataFrame, outcome_col: str, 
                            iv_result: Dict) -> str:
    """Generate residual diagnostic plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'fitted_values' in iv_result and iv_result['fitted_values'] is not None:
        fitted = np.array(iv_result['fitted_values'])
        y_values = df[outcome_col].dropna().values
        
        # Ensure same size
        min_len = min(len(fitted), len(y_values))
        fitted = fitted[:min_len]
        y_values = y_values[:min_len]
        residuals = y_values - fitted
        
        # Residuals vs Fitted
        axes[0].scatter(fitted, residuals, alpha=0.5, color='#3b82f6', s=40)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Fitted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.3)
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Residuals not available', ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'Residuals not available', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_instrument_strength_plot(first_stage: Dict) -> str:
    """Generate instrument strength visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    instruments = [c['instrument'] for c in first_stage['instrument_coefficients']]
    t_values = [abs(c['t_value']) for c in first_stage['instrument_coefficients']]
    significant = [c['significant'] for c in first_stage['instrument_coefficients']]
    
    colors = ['#22c55e' if sig else '#ef4444' for sig in significant]
    bars = ax.barh(instruments, t_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Critical value line
    ax.axvline(x=1.96, color='orange', linestyle='--', linewidth=2, label='t = 1.96 (5% sig)')
    ax.axvline(x=3.29, color='red', linestyle='--', linewidth=2, label='t = 3.29 (0.1% sig)')
    
    # F-statistic annotation
    ax.text(0.95, 0.95, f"First Stage F = {first_stage['f_statistic']:.2f}\n{'Weak' if first_stage['weak_instrument'] else 'Strong'} Instruments",
            transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('|t-statistic|', fontsize=12)
    ax.set_ylabel('Instrument', fontsize=12)
    ax.set_title('Instrument Strength (First Stage t-statistics)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(ols_result: Dict, iv_result: Dict, first_stage: Dict,
                            diagnostics: Dict, endogenous_col: str) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    key_insights = []
    
    # IV Effect
    if iv_result.get('significant'):
        key_insights.append({
            'title': 'Significant Causal Effect',
            'description': f"The IV estimate shows {endogenous_col} has a significant effect of {iv_result['endogenous_effect']:.3f} (p = {iv_result['endogenous_pvalue']:.4f}).",
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'No Significant Effect',
            'description': f"The IV estimate of {iv_result['endogenous_effect']:.3f} is not statistically significant (p = {iv_result['endogenous_pvalue']:.4f}).",
            'status': 'negative'
        })
    
    # Comparison with OLS
    ols_effect = ols_result['endogenous_effect']
    iv_effect = iv_result['endogenous_effect']
    bias = abs(ols_effect - iv_effect)
    key_insights.append({
        'title': 'OLS vs IV Comparison',
        'description': f"OLS estimate: {ols_effect:.3f}, IV estimate: {iv_effect:.3f}. Difference: {bias:.3f}. {'OLS appears upward biased.' if ols_effect > iv_effect else 'OLS appears downward biased.' if ols_effect < iv_effect else 'Similar estimates.'}",
        'status': 'neutral'
    })
    
    # Instrument strength
    if first_stage['weak_instrument']:
        key_insights.append({
            'title': 'Warning: Weak Instruments',
            'description': f"First stage F = {first_stage['f_statistic']:.2f} < 10. IV estimates may be biased and have incorrect standard errors.",
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': 'Strong Instruments',
            'description': f"First stage F = {first_stage['f_statistic']:.2f} ≥ 10. Instruments are sufficiently strong.",
            'status': 'positive'
        })
    
    # Overidentification (if applicable)
    if 'j_statistic' in diagnostics.get('overidentification_test', {}):
        overid = diagnostics['overidentification_test']
        key_insights.append({
            'title': 'Overidentification Test',
            'description': overid['message'],
            'status': 'positive' if overid.get('valid_instruments') else 'warning'
        })
    
    return {
        'key_insights': key_insights,
        'recommendation': 'IV estimates are reliable.' if not first_stage['weak_instrument'] else 'Consider finding stronger instruments or using weak-instrument robust methods.'
    }


@router.post("/instrumental-variable")
async def run_iv_analysis(request: IVRequest) -> Dict[str, Any]:
    """
    Perform Instrumental Variable (IV) Analysis using 2SLS.
    
    Estimates causal effects when there is endogeneity using
    instrumental variables.
    """
    try:
        data = request.data
        outcome_col = request.outcome_col
        endogenous_col = request.endogenous_col
        instrument_cols = request.instrument_cols
        exogenous_cols = request.exogenous_cols or []
        robust_se = request.robust_se
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        all_cols = [outcome_col, endogenous_col] + instrument_cols + exogenous_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")
        
        # Convert to numeric
        for col in [outcome_col, endogenous_col] + instrument_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in exogenous_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=[outcome_col, endogenous_col] + instrument_cols + exogenous_cols)
        df = df.reset_index(drop=True)  # 인덱스 리셋 추가
        
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="At least 30 observations required for IV analysis.")
        
        if len(instrument_cols) < 1:
            raise HTTPException(status_code=400, detail="At least one instrument required.")
        
        # Run OLS (for comparison)
        ols_result = run_ols_regression(df, outcome_col, endogenous_col, exogenous_cols)
        
        # Run First Stage
        first_stage = run_first_stage(df, endogenous_col, instrument_cols, exogenous_cols)
        
        # Run 2SLS
        iv_result = run_2sls(df, outcome_col, endogenous_col, instrument_cols, exogenous_cols, robust_se)
        
        if 'error' in iv_result:
            raise HTTPException(status_code=400, detail=iv_result['error'])
        
        # Diagnostic tests
        diagnostics = run_diagnostic_tests(df, outcome_col, endogenous_col, instrument_cols,
                                            exogenous_cols, first_stage, iv_result)
        
        # Generate visualizations
        first_stage_plot = generate_first_stage_plot(df, endogenous_col, instrument_cols, first_stage)
        comparison_plot = generate_comparison_plot(ols_result, iv_result, endogenous_col)
        residual_plot = generate_residual_plot(df, outcome_col, iv_result)
        strength_plot = generate_instrument_strength_plot(first_stage)
        
        # Interpretation
        interpretation = generate_interpretation(ols_result, iv_result, first_stage, 
                                                  diagnostics, endogenous_col)
        
        # Descriptive stats
        descriptive_stats = {
            'n_obs': len(df),
            'n_instruments': len(instrument_cols),
            'n_exogenous': len(exogenous_cols),
            'outcome_mean': _to_native_type(df[outcome_col].mean()),
            'outcome_std': _to_native_type(df[outcome_col].std()),
            'endogenous_mean': _to_native_type(df[endogenous_col].mean()),
            'endogenous_std': _to_native_type(df[endogenous_col].std())
        }
        
        return {
            'ols_result': ols_result,
            'first_stage': {
                'instrument_coefficients': first_stage['instrument_coefficients'],
                'f_statistic': first_stage['f_statistic'],
                'f_pvalue': first_stage['f_pvalue'],
                'weak_instrument': first_stage['weak_instrument'],
                'r_squared': first_stage['r_squared']
            },
            'iv_result': iv_result,
            'diagnostics': diagnostics,
            'descriptive_stats': descriptive_stats,
            'first_stage_plot': first_stage_plot,
            'comparison_plot': comparison_plot,
            'residual_plot': residual_plot,
            'strength_plot': strength_plot,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IV analysis failed: {str(e)}")
