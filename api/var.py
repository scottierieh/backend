"""
Vector Autoregression (VAR) Decomposition Router for FastAPI
Analyze dynamic relationships between multiple time series
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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class VARRequest(BaseModel):
    data: List[Dict[str, Any]]
    variables: List[str]  # Variables to include in VAR
    date_col: Optional[str] = None  # Date column (optional)
    max_lags: int = 8  # Maximum lags to consider
    selected_lag: Optional[int] = None  # Force specific lag order
    irf_periods: int = 20  # IRF horizon
    fevd_periods: int = 20  # FEVD horizon


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


def test_stationarity(series: pd.Series, name: str) -> Dict[str, Any]:
    """Perform Augmented Dickey-Fuller test"""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'variable': name,
            'adf_statistic': _to_native_type(result[0]),
            'p_value': _to_native_type(result[1]),
            'lags_used': int(result[2]),
            'n_obs': int(result[3]),
            'critical_values': {k: _to_native_type(v) for k, v in result[4].items()},
            'stationary': bool(result[1] < 0.05)
        }
    except Exception as e:
        return {'variable': name, 'error': str(e)}


def select_lag_order(data: pd.DataFrame, max_lags: int) -> Dict[str, Any]:
    """Select optimal lag order using information criteria"""
    try:
        model = VAR(data)
        lag_order = model.select_order(maxlags=max_lags)
        
        return {
            'aic': int(lag_order.aic),
            'bic': int(lag_order.bic),
            'hqic': int(lag_order.hqic),
            'fpe': _to_native_type(lag_order.fpe),
            'recommended': int(lag_order.aic),
            'criteria_values': {
                'AIC': {i: _to_native_type(v) for i, v in enumerate(lag_order.ics['aic'])},
                'BIC': {i: _to_native_type(v) for i, v in enumerate(lag_order.ics['bic'])}
            }
        }
    except Exception as e:
        return {'error': str(e), 'recommended': 1}


def estimate_var(data: pd.DataFrame, lags: int) -> Dict[str, Any]:
    """Estimate VAR model"""
    model = VAR(data)
    results = model.fit(lags)
    
    # Extract coefficients
    coefficients = {}
    for i, var in enumerate(data.columns):
        coef_data = []
        params = results.params[var]
        stderr = results.stderr[var]
        tvalues = results.tvalues[var]
        pvalues = results.pvalues[var]
        
        for j, param_name in enumerate(params.index):
            coef_data.append({
                'term': param_name,
                'coefficient': _to_native_type(params.iloc[j]),
                'std_error': _to_native_type(stderr.iloc[j]),
                't_value': _to_native_type(tvalues.iloc[j]),
                'p_value': _to_native_type(pvalues.iloc[j]),
                'significant': bool(pvalues.iloc[j] < 0.05)
            })
        coefficients[var] = coef_data
    
    # Model diagnostics
    diagnostics = {
        'aic': _to_native_type(results.aic),
        'bic': _to_native_type(results.bic),
        'hqic': _to_native_type(results.hqic),
        'fpe': _to_native_type(results.fpe),
        'log_likelihood': _to_native_type(results.llf),
        'n_obs': int(results.nobs),
        'k_ar': int(results.k_ar)
    }
    
    # Durbin-Watson stats
    dw_stats = {}
    for i, var in enumerate(data.columns):
        dw = durbin_watson(results.resid[var])
        dw_stats[var] = _to_native_type(dw)
    diagnostics['durbin_watson'] = dw_stats
    
    return {
        'coefficients': coefficients,
        'diagnostics': diagnostics,
        'fitted_model': results
    }


def compute_granger_causality(data: pd.DataFrame, max_lags: int) -> List[Dict[str, Any]]:
    """Compute Granger causality tests for all variable pairs"""
    results = []
    variables = data.columns.tolist()
    
    for caused in variables:
        for causing in variables:
            if caused != causing:
                try:
                    test_data = data[[caused, causing]].dropna()
                    gc_result = grangercausalitytests(test_data, maxlag=min(max_lags, 4), verbose=False)
                    
                    # Get results for optimal lag (using F-test)
                    best_pvalue = 1.0
                    best_lag = 1
                    for lag, test_result in gc_result.items():
                        f_pvalue = test_result[0]['ssr_ftest'][1]
                        if f_pvalue < best_pvalue:
                            best_pvalue = f_pvalue
                            best_lag = lag
                    
                    results.append({
                        'causing': causing,
                        'caused': caused,
                        'lag': best_lag,
                        'f_statistic': _to_native_type(gc_result[best_lag][0]['ssr_ftest'][0]),
                        'p_value': _to_native_type(best_pvalue),
                        'granger_causes': bool(best_pvalue < 0.05)
                    })
                except Exception as e:
                    results.append({
                        'causing': causing,
                        'caused': caused,
                        'error': str(e)
                    })
    
    return results


def compute_irf(var_result, periods: int, variables: List[str]) -> Dict[str, Any]:
    """Compute Impulse Response Functions"""
    irf = var_result.irf(periods)
    try:
        irf_ci = irf.ci(signif=0.05)  # shape: (periods+1, n_vars, n_vars, 2)
        has_ci = True
    except Exception:
        has_ci = False

    irf_data = {}
    for i, shock_var in enumerate(variables):
        irf_data[shock_var] = {}
        for j, response_var in enumerate(variables):
            irf_values = irf.irfs[:, j, i]

            if has_ci:
                try:
                    irf_lower = irf_ci[:, j, i, 0]
                    irf_upper = irf_ci[:, j, i, 1]
                except Exception:
                    has_ci = False

            if not has_ci:
                std = float(np.std(irf_values)) if np.std(irf_values) > 0 else 0.0
                irf_lower = irf_values - 1.96 * std
                irf_upper = irf_values + 1.96 * std

            irf_data[shock_var][response_var] = {
                'values': [_to_native_type(v) for v in irf_values],
                'lower': [_to_native_type(v) for v in irf_lower],
                'upper': [_to_native_type(v) for v in irf_upper],
            }

    return irf_data


def compute_fevd(var_result, periods: int, variables: List[str]) -> Dict[str, Any]:
    """Compute Forecast Error Variance Decomposition"""
    fevd = var_result.fevd(periods)
    
    fevd_data = {}
    for i, var in enumerate(variables):
        fevd_data[var] = {}
        decomp = fevd.decomp[i]  # Shape: (periods, n_vars)
        
        for j, shock_var in enumerate(variables):
            fevd_data[var][shock_var] = [_to_native_type(v * 100) for v in decomp[:, j]]
    
    return fevd_data


def generate_irf_plot(irf_data: Dict, variables: List[str], periods: int) -> str:
    """Generate IRF plot matrix"""
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(4 * n_vars, 3 * n_vars))
    
    if n_vars == 1:
        axes = np.array([[axes]])
    
    time = np.arange(periods + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_vars))
    
    for i, shock_var in enumerate(variables):
        for j, response_var in enumerate(variables):
            ax = axes[j, i]
            
            values = irf_data[shock_var][response_var]['values']
            ax.plot(time, values, color=colors[i], linewidth=2)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.fill_between(time, 
                           irf_data[shock_var][response_var].get('lower', values),
                           irf_data[shock_var][response_var].get('upper', values),
                           alpha=0.2, color=colors[i])
            
            ax.set_title(f'{shock_var} → {response_var}', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            if j == n_vars - 1:
                ax.set_xlabel('Periods', fontsize=9)
            if i == 0:
                ax.set_ylabel('Response', fontsize=9)
    
    fig.suptitle('Impulse Response Functions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_fevd_plot(fevd_data: Dict, variables: List[str], periods: int) -> str:
    """Generate FEVD stacked area plot"""
    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))
    
    if n_vars == 1:
        axes = [axes]
    
    time = np.arange(1, periods + 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_vars))
    
    for i, var in enumerate(variables):
        ax = axes[i]
        
        # Stack the contributions
        bottom = np.zeros(periods)
        for j, shock_var in enumerate(variables):
            values = fevd_data[var][shock_var]
            ax.fill_between(time, bottom, bottom + values, 
                           label=shock_var, color=colors[j], alpha=0.8)
            bottom = bottom + np.array(values)
        
        ax.set_xlim(1, periods)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Periods', fontsize=10)
        ax.set_ylabel('Variance %', fontsize=10)
        ax.set_title(f'FEVD: {var}', fontsize=11, fontweight='bold')
        ax.legend(loc='center right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    fig.suptitle('Forecast Error Variance Decomposition', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_granger_heatmap(granger_results: List[Dict], variables: List[str]) -> str:
    """Generate Granger causality heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_vars = len(variables)
    matrix = np.ones((n_vars, n_vars))  # Default p-value = 1 (diagonal)
    
    for result in granger_results:
        if 'error' not in result:
            i = variables.index(result['caused'])
            j = variables.index(result['causing'])
            matrix[i, j] = result['p_value']
    
    # Create heatmap
    mask = np.eye(n_vars, dtype=bool)  # Mask diagonal
    
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=variables, yticklabels=variables,
                mask=mask, vmin=0, vmax=0.2, ax=ax,
                cbar_kws={'label': 'p-value'})
    
    ax.set_xlabel('Causing Variable', fontsize=11)
    ax.set_ylabel('Caused Variable', fontsize=11)
    ax.set_title('Granger Causality Test (p-values)\nGreen = Significant (<0.05)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_time_series_plot(data: pd.DataFrame, variables: List[str]) -> str:
    """Generate time series plot of all variables"""
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3 * n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_vars))
    
    for i, var in enumerate(variables):
        ax = axes[i]
        ax.plot(data[var].values, color=colors[i], linewidth=1.5)
        ax.set_ylabel(var, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(y=data[var].mean(), color='gray', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Observation', fontsize=11)
    fig.suptitle('Time Series Data', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_lag_selection_plot(lag_order_result: Dict) -> str:
    """Generate lag selection criteria plot"""
    if 'criteria_values' not in lag_order_result:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    aic_values = lag_order_result['criteria_values']['AIC']
    bic_values = lag_order_result['criteria_values']['BIC']
    
    lags = list(aic_values.keys())
    aic_vals = list(aic_values.values())
    bic_vals = list(bic_values.values())
    
    ax.plot(lags, aic_vals, 'o-', color='#3b82f6', linewidth=2, markersize=8, label='AIC')
    ax.plot(lags, bic_vals, 's-', color='#ef4444', linewidth=2, markersize=8, label='BIC')
    
    # Mark optimal
    ax.axvline(x=lag_order_result['aic'], color='#3b82f6', linestyle='--', alpha=0.5)
    ax.axvline(x=lag_order_result['bic'], color='#ef4444', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Lag Order', fontsize=11)
    ax.set_ylabel('Information Criterion', fontsize=11)
    ax.set_title('Lag Order Selection', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(stationarity: List[Dict], lag_order: Dict, 
                            granger: List[Dict], fevd_data: Dict,
                            variables: List[str]) -> Dict[str, Any]:
    """Generate interpretation of VAR results"""
    key_insights = []
    
    # Stationarity
    non_stationary = [s['variable'] for s in stationarity if not s.get('stationary', True)]
    if non_stationary:
        key_insights.append({
            'title': 'Stationarity Warning',
            'description': f"Variables may be non-stationary: {', '.join(non_stationary)}. Consider differencing.",
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': 'Stationarity Check',
            'description': 'All variables appear stationary (ADF test p < 0.05).',
            'status': 'positive'
        })
    
    # Lag order
    key_insights.append({
        'title': 'Optimal Lag Order',
        'description': f"AIC suggests {lag_order.get('aic', 1)} lags, BIC suggests {lag_order.get('bic', 1)} lags.",
        'status': 'neutral'
    })
    
    # Granger causality
    significant_gc = [g for g in granger if g.get('granger_causes', False)]
    if significant_gc:
        gc_summary = '; '.join([f"{g['causing']} → {g['caused']}" for g in significant_gc[:5]])
        key_insights.append({
            'title': 'Granger Causality',
            'description': f"Significant causal relationships: {gc_summary}",
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'Granger Causality',
            'description': 'No significant Granger causality detected between variables.',
            'status': 'neutral'
        })
    
    # FEVD summary - which variable explains most variance
    for var in variables:
        if var in fevd_data:
            final_decomp = {k: v[-1] for k, v in fevd_data[var].items()}
            own_share = final_decomp.get(var, 0)
            key_insights.append({
                'title': f'{var} Variance',
                'description': f"At horizon {len(fevd_data[var][var])}, {own_share:.1f}% explained by own shocks.",
                'status': 'neutral'
            })
    
    return {
        'key_insights': key_insights,
        'recommendation': 'VAR model captures dynamic relationships between variables.' if not non_stationary else 'Consider differencing non-stationary variables before estimation.'
    }


@router.post("/var-decomposition")
async def run_var_analysis(request: VARRequest) -> Dict[str, Any]:
    """
    Perform Vector Autoregression (VAR) Analysis with:
    - Lag order selection
    - Granger causality tests
    - Impulse Response Functions (IRF)
    - Forecast Error Variance Decomposition (FEVD)
    """
    try:
        data = request.data
        variables = request.variables
        date_col = request.date_col
        max_lags = request.max_lags
        selected_lag = request.selected_lag
        irf_periods = request.irf_periods
        fevd_periods = request.fevd_periods
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        if len(variables) < 2:
            raise HTTPException(status_code=400, detail="At least 2 variables required for VAR.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        missing = [v for v in variables if v not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Variables not found: {', '.join(missing)}")
        
        # Convert to numeric
        for var in variables:
            df[var] = pd.to_numeric(df[var], errors='coerce')
        
        # Prepare data
        var_data = df[variables].dropna()
        
        if len(var_data) < 30:
            raise HTTPException(status_code=400, detail="At least 30 observations required.")
        
        # Stationarity tests
        stationarity_tests = []
        for var in variables:
            stationarity_tests.append(test_stationarity(var_data[var], var))
        
        # Lag order selection
        lag_order = select_lag_order(var_data, min(max_lags, len(var_data) // 5))
        optimal_lag = selected_lag if selected_lag else lag_order.get('recommended', 1)
        optimal_lag = max(1, min(optimal_lag, max_lags))
        
        # Estimate VAR
        var_result = estimate_var(var_data, optimal_lag)
        fitted_model = var_result['fitted_model']
        
        # Granger causality
        granger_results = compute_granger_causality(var_data, min(optimal_lag + 2, 4))
        
        # IRF
        irf_data = compute_irf(fitted_model, irf_periods, variables)
        
        # FEVD
        fevd_data = compute_fevd(fitted_model, fevd_periods, variables)
        
        # Generate visualizations
        ts_plot = generate_time_series_plot(var_data, variables)
        irf_plot = generate_irf_plot(irf_data, variables, irf_periods)
        fevd_plot = generate_fevd_plot(fevd_data, variables, fevd_periods)
        granger_plot = generate_granger_heatmap(granger_results, variables)
        lag_plot = generate_lag_selection_plot(lag_order)
        
        # Interpretation
        interpretation = generate_interpretation(stationarity_tests, lag_order, 
                                                  granger_results, fevd_data, variables)
        
        # Clean up result
        var_result_clean = {
            'coefficients': var_result['coefficients'],
            'diagnostics': var_result['diagnostics']
        }
        
        return {
            'variables': variables,
            'n_obs': len(var_data),
            'lag_order': optimal_lag,
            'stationarity_tests': stationarity_tests,
            'lag_selection': lag_order,
            'var_result': var_result_clean,
            'granger_causality': granger_results,
            'irf': irf_data,
            'fevd': fevd_data,
            'ts_plot': ts_plot,
            'irf_plot': irf_plot,
            'fevd_plot': fevd_plot,
            'granger_plot': granger_plot,
            'lag_plot': lag_plot,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VAR analysis failed: {str(e)}")
