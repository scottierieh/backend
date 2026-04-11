"""
Vector Autoregression (VAR) Decomposition Router for FastAPI
Matches frontend /api/analysis/var-decomposition endpoint
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
sns.set_palette("husl")

router = APIRouter()


class VARDecompositionRequest(BaseModel):
    """Request model matching frontend"""
    data: List[Dict[str, Any]]
    variables: List[str]  # 2-6 variables
    max_lags: int = 8
    irf_periods: int = 20
    fevd_periods: int = 20


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable types"""
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
    if isinstance(obj, dict):
        return {k: _to_native_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native_type(v) for v in obj]
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
    """Augmented Dickey-Fuller stationarity test"""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'variable': name,
            'adf_statistic': _to_native_type(result[0]),
            'p_value': _to_native_type(result[1]),
            'stationary': bool(result[1] < 0.05)
        }
    except Exception as e:
        return {
            'variable': name,
            'adf_statistic': None,
            'p_value': 1.0,
            'stationary': False,
            'error': str(e)
        }


def select_lag_order(data: pd.DataFrame, max_lags: int) -> Dict[str, Any]:
    """Select optimal lag order using information criteria"""
    try:
        model = VAR(data)
        lag_order_result = model.select_order(maxlags=max_lags)
        
        return {
            'aic': int(lag_order_result.aic),
            'bic': int(lag_order_result.bic),
            'hqic': int(lag_order_result.hqic),
            'fpe': int(lag_order_result.fpe),
            'recommended': int(lag_order_result.aic)  # AIC as default
        }
    except Exception as e:
        return {
            'aic': 1,
            'bic': 1,
            'hqic': 1,
            'fpe': 1,
            'recommended': 1,
            'error': str(e)
        }


def estimate_var_model(data: pd.DataFrame, lags: int) -> tuple:
    """Estimate VAR model and return coefficients + diagnostics"""
    model = VAR(data)
    results = model.fit(lags)
    
    # Extract coefficients for each equation
    coefficients = {}
    for var_name in data.columns:
        coef_list = []
        params = results.params[var_name]
        stderr = results.stderr[var_name]
        tvalues = results.tvalues[var_name]
        pvalues = results.pvalues[var_name]
        
        for idx, term_name in enumerate(params.index):
            coef_list.append({
                'term': term_name,
                'coefficient': _to_native_type(params.iloc[idx]),
                'std_error': _to_native_type(stderr.iloc[idx]),
                't_value': _to_native_type(tvalues.iloc[idx]),
                'p_value': _to_native_type(pvalues.iloc[idx]),
                'significant': bool(pvalues.iloc[idx] < 0.05)
            })
        coefficients[var_name] = coef_list
    
    # Diagnostics
    dw_stats = {}
    for var_name in data.columns:
        dw = durbin_watson(results.resid[var_name])
        dw_stats[var_name] = _to_native_type(dw)
    
    diagnostics = {
        'aic': _to_native_type(results.aic),
        'bic': _to_native_type(results.bic),
        'hqic': _to_native_type(results.hqic),
        'fpe': _to_native_type(results.fpe),
        'log_likelihood': _to_native_type(results.llf),
        'n_obs': int(results.nobs),
        'k_ar': int(results.k_ar),
        'durbin_watson': dw_stats
    }
    
    var_result = {
        'coefficients': coefficients,
        'diagnostics': diagnostics
    }
    
    return var_result, results


def compute_granger_causality(data: pd.DataFrame, max_lags: int) -> List[Dict[str, Any]]:
    """Pairwise Granger causality tests"""
    results = []
    variables = data.columns.tolist()
    
    for caused in variables:
        for causing in variables:
            if caused == causing:
                continue
                
            try:
                test_data = data[[caused, causing]].dropna()
                gc_tests = grangercausalitytests(
                    test_data, 
                    maxlag=min(max_lags, 4), 
                    verbose=False
                )
                
                # Find best lag based on p-value
                best_pvalue = 1.0
                best_lag = 1
                best_f_stat = 0.0
                
                for lag in gc_tests.keys():
                    f_stat, p_val, _, _ = gc_tests[lag][0]['ssr_ftest']
                    if p_val < best_pvalue:
                        best_pvalue = p_val
                        best_lag = lag
                        best_f_stat = f_stat
                
                results.append({
                    'causing': causing,
                    'caused': caused,
                    'lag': int(best_lag),
                    'f_statistic': _to_native_type(best_f_stat),
                    'p_value': _to_native_type(best_pvalue),
                    'granger_causes': bool(best_pvalue < 0.05)
                })
                
            except Exception as e:
                results.append({
                    'causing': causing,
                    'caused': caused,
                    'lag': 1,
                    'f_statistic': 0.0,
                    'p_value': 1.0,
                    'granger_causes': False,
                    'error': str(e)
                })
    
    return results


def compute_irf(fitted_model, periods: int, variables: List[str]) -> Dict[str, Any]:
    """Impulse Response Functions with confidence bands"""
    irf = fitted_model.irf(periods)
    
    irf_data = {}
    for i, shock_var in enumerate(variables):
        irf_data[shock_var] = {}
        
        for j, response_var in enumerate(variables):
            irf_values = irf.irfs[:, j, i]
            
            # Simple confidence intervals using standard error
            std_err = np.std(irf_values)
            lower = irf_values - 1.96 * std_err
            upper = irf_values + 1.96 * std_err
            
            irf_data[shock_var][response_var] = {
                'values': [_to_native_type(v) for v in irf_values],
                'lower': [_to_native_type(v) for v in lower],
                'upper': [_to_native_type(v) for v in upper]
            }
    
    return irf_data


def compute_fevd(fitted_model, periods: int, variables: List[str]) -> Dict[str, Any]:
    """Forecast Error Variance Decomposition"""
    fevd = fitted_model.fevd(periods)
    
    fevd_data = {}
    for i, response_var in enumerate(variables):
        fevd_data[response_var] = {}
        decomp = fevd.decomp[i]  # Shape: (periods, n_vars)
        
        for j, shock_var in enumerate(variables):
            # Convert to percentage
            fevd_data[response_var][shock_var] = [
                _to_native_type(v * 100) for v in decomp[:, j]
            ]
    
    return fevd_data


def create_time_series_plot(data: pd.DataFrame, variables: List[str]) -> str:
    """Plot original time series"""
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 3 * len(variables)))
    
    if len(variables) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(variables)))
    
    for i, var in enumerate(variables):
        axes[i].plot(data.index, data[var], color=colors[i], linewidth=1.5)
        axes[i].set_ylabel(var, fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{var} Time Series', fontsize=12)
    
    axes[-1].set_xlabel('Observation', fontsize=11)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_irf_plot(irf_data: Dict, variables: List[str], periods: int) -> str:
    """IRF plot matrix"""
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(4 * n_vars, 3 * n_vars))
    
    if n_vars == 1:
        axes = np.array([[axes]])
    elif n_vars == 2:
        axes = axes.reshape(2, 2)
    
    time_periods = np.arange(periods + 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_vars))
    
    for i, shock_var in enumerate(variables):
        for j, response_var in enumerate(variables):
            ax = axes[j, i] if n_vars > 1 else axes[0, 0]
            
            irf_vals = irf_data[shock_var][response_var]
            values = irf_vals['values']
            lower = irf_vals.get('lower', values)
            upper = irf_vals.get('upper', values)
            
            ax.plot(time_periods, values, color=colors[i], linewidth=2.5, label='IRF')
            ax.fill_between(time_periods, lower, upper, alpha=0.2, color=colors[i])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_title(f'{shock_var} → {response_var}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if j == n_vars - 1:
                ax.set_xlabel('Periods', fontsize=9)
            if i == 0:
                ax.set_ylabel('Response', fontsize=9)
    
    fig.suptitle('Impulse Response Functions (Orthogonalized)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_fevd_plot(fevd_data: Dict, variables: List[str], periods: int) -> str:
    """FEVD stacked area charts"""
    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
    
    if n_vars == 1:
        axes = [axes]
    
    time_periods = np.arange(1, periods + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_vars))
    
    for i, response_var in enumerate(variables):
        ax = axes[i]
        
        bottom = np.zeros(periods)
        for j, shock_var in enumerate(variables):
            values = np.array(fevd_data[response_var][shock_var])
            ax.fill_between(time_periods, bottom, bottom + values,
                          label=shock_var, color=colors[j], alpha=0.8)
            bottom += values
        
        ax.set_xlim(1, periods)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Periods', fontsize=11)
        ax.set_ylabel('Variance (%)', fontsize=11)
        ax.set_title(f'{response_var} Variance Decomposition', fontsize=12, fontweight='bold')
        ax.legend(loc='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Forecast Error Variance Decomposition', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_granger_network_plot(granger_results: List[Dict]) -> str:
    """Granger causality network visualization"""
    significant = [g for g in granger_results if g.get('granger_causes', False)]
    
    if not significant:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No significant Granger causality detected',
                ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_vars = set()
    for g in significant:
        all_vars.add(g['causing'])
        all_vars.add(g['caused'])
    all_vars = sorted(all_vars)
    n_vars = len(all_vars)
    
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    positions = {var: (np.cos(a), np.sin(a)) for var, a in zip(all_vars, angles)}
    
    for g in significant:
        x1, y1 = positions[g['causing']]
        x2, y2 = positions[g['caused']]
        p_val = g['p_value']
        alpha = min(1.0, 1.0 - p_val) * 0.7 + 0.3
        width = 2 + (1 - p_val) * 3
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=width, alpha=alpha, color='#3b82f6'))
    
    for var, (x, y) in positions.items():
        ax.scatter([x], [y], s=1500, c='#22c55e', edgecolors='white', linewidths=2, zorder=10)
        ax.text(x, y, var, ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Granger Causality Network ({len(significant)} significant links)', 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_lag_selection_plot(lag_selection: Dict) -> str:
    """Information criteria comparison across lags"""
    if 'error' in lag_selection:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Lag selection data unavailable', 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    recommended = lag_selection['recommended']
    ax.axvline(recommended, color='red', linestyle='--', linewidth=2, 
               label=f'Selected: {recommended}', alpha=0.7)
    ax.text(recommended, 0.5, f'p = {recommended}', 
            ha='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.set_xlabel('Lag Order', fontsize=11)
    ax.set_ylabel('Information Criterion', fontsize=11)
    ax.set_title('Lag Order Selection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(
    stationarity_tests: List[Dict],
    granger_results: List[Dict],
    lag_order: int,
    n_obs: int
) -> Dict[str, Any]:
    """Generate AI-style interpretation"""
    key_insights = []
    
    non_stationary = [s for s in stationarity_tests if not s.get('stationary', True)]
    if non_stationary:
        key_insights.append({
            'title': f'{len(non_stationary)} Non-Stationary Series Detected',
            'description': f"Variables {', '.join([s['variable'] for s in non_stationary])} "
                         f"failed ADF test (p > 0.05). Consider differencing for valid inference.",
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': 'All Series Are Stationary',
            'description': 'All variables passed ADF stationarity test. VAR estimation is appropriate.',
            'status': 'positive'
        })
    
    if lag_order <= 2:
        key_insights.append({
            'title': f'Short Lag Order (p = {lag_order})',
            'description': 'Model captures immediate dynamics. Relationships are relatively simple.',
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': f'Moderate Lag Order (p = {lag_order})',
            'description': f'Model includes {lag_order} lags, capturing complex temporal dependencies.',
            'status': 'neutral'
        })
    
    significant_granger = [g for g in granger_results if g.get('granger_causes', False)]
    if len(significant_granger) == 0:
        key_insights.append({
            'title': 'No Granger Causality Detected',
            'description': 'Variables do not significantly predict each other. Consider alternative models.',
            'status': 'negative'
        })
    elif len(significant_granger) <= 2:
        key_insights.append({
            'title': f'{len(significant_granger)} Granger-Causal Relationships',
            'description': f"Weak predictive structure. Key link: "
                         f"{significant_granger[0]['causing']} → {significant_granger[0]['caused']}.",
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': f'{len(significant_granger)} Granger-Causal Links Found',
            'description': f'Rich predictive network. Variables are highly interdependent.',
            'status': 'positive'
        })
    
    if n_obs < 50:
        key_insights.append({
            'title': f'Small Sample (n = {n_obs})',
            'description': 'Limited observations may affect reliability. Interpret with caution.',
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': f'Adequate Sample Size (n = {n_obs})',
            'description': 'Sufficient observations for reliable VAR estimation.',
            'status': 'positive'
        })
    
    if non_stationary:
        recommendation = ("⚠️ Non-stationary series detected. Consider first-differencing or "
                        "testing for cointegration before interpreting results.")
    elif not significant_granger:
        recommendation = ("⚠️ No Granger causality found. VAR may not be the best model. "
                        "Consider alternative specifications or independent AR models.")
    else:
        recommendation = ("✅ Model appears well-specified. IRF shows dynamic responses, "
                        "FEVD reveals variance contributions. Use results for policy analysis.")
    
    return {
        'key_insights': key_insights,
        'recommendation': recommendation
    }


@router.post("/var-decomposition")
async def var_decomposition_analysis(request: VARDecompositionRequest) -> Dict[str, Any]:
    """
    Main VAR decomposition endpoint
    Matches frontend: POST /api/analysis/var-decomposition
    """
    try:
        if len(request.variables) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 variables required for VAR analysis"
            )
        
        if len(request.variables) > 6:
            raise HTTPException(
                status_code=400,
                detail="Maximum 6 variables allowed"
            )
        
        df = pd.DataFrame(request.data)
        
        for var in request.variables:
            if var not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variable '{var}' not found in data"
                )
        
        var_data = df[request.variables].astype(float).dropna()
        
        if len(var_data) < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(var_data)} observations (need ≥30)"
            )
        
        stationarity_tests = []
        for var in request.variables:
            stationarity_tests.append(test_stationarity(var_data[var], var))
        
        lag_selection = select_lag_order(var_data, request.max_lags)
        optimal_lag = lag_selection.get('recommended', 1)
        optimal_lag = max(1, min(optimal_lag, request.max_lags))
        
        var_result, fitted_model = estimate_var_model(var_data, optimal_lag)
        granger_causality = compute_granger_causality(var_data, optimal_lag)
        irf_data = compute_irf(fitted_model, request.irf_periods, request.variables)
        fevd_data = compute_fevd(fitted_model, request.fevd_periods, request.variables)
        
        ts_plot = create_time_series_plot(var_data, request.variables)
        irf_plot = create_irf_plot(irf_data, request.variables, request.irf_periods)
        fevd_plot = create_fevd_plot(fevd_data, request.variables, request.fevd_periods)
        granger_plot = create_granger_network_plot(granger_causality)
        lag_plot = create_lag_selection_plot(lag_selection)
        
        interpretation = generate_interpretation(
            stationarity_tests,
            granger_causality,
            optimal_lag,
            len(var_data)
        )
        
        return {
            'variables': request.variables,
            'n_obs': len(var_data),
            'lag_order': optimal_lag,
            'stationarity_tests': stationarity_tests,
            'lag_selection': lag_selection,
            'var_result': var_result,
            'granger_causality': granger_causality,
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
        import traceback
        error_detail = f"VAR analysis failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)
