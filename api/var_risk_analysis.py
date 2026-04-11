"""
Value at Risk (VaR) Analysis Router for FastAPI
Comprehensive VaR calculation with multiple methodologies
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t as t_dist
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class VarMethod(str, Enum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


class VarRequest(BaseModel):
    """VaR calculation request parameters"""
    
    # Portfolio data
    returns: List[float] = Field(
        ...,
        description="Historical returns (daily or periodic)"
    )
    portfolio_value: float = Field(
        default=1000000.0,
        ge=1000,
        description="Current portfolio value"
    )
    
    # VaR parameters
    confidence_levels: List[float] = Field(
        default=[0.95, 0.99],
        description="Confidence levels for VaR calculation"
    )
    holding_period: int = Field(
        default=1,
        ge=1,
        le=252,
        description="Holding period in days"
    )
    method: VarMethod = Field(
        default=VarMethod.HISTORICAL,
        description="VaR calculation method"
    )
    
    # Monte Carlo parameters
    num_simulations: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo simulations"
    )
    
    # Additional options
    calculate_es: bool = Field(
        default=True,
        description="Calculate Expected Shortfall (CVaR)"
    )
    rolling_window: int = Field(
        default=252,
        ge=20,
        le=1260,
        description="Rolling window for time series analysis"
    )


def _to_native_type(obj):
    """Convert numpy types to JSON-serializable Python types"""
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


# =============================================================================
# VaR Calculation Methods
# =============================================================================

def calculate_historical_var(returns: np.ndarray, confidence_level: float, 
                             holding_period: int = 1) -> float:
    """
    Historical Simulation VaR
    Uses actual historical returns to estimate VaR
    """
    # Scale returns for holding period (square root of time rule)
    scaled_returns = returns * np.sqrt(holding_period)
    
    # VaR is the negative of the percentile
    var = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
    return var


def calculate_parametric_var(returns: np.ndarray, confidence_level: float,
                             holding_period: int = 1) -> float:
    """
    Parametric (Variance-Covariance) VaR
    Assumes returns are normally distributed
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    # Scale for holding period
    mu_scaled = mu * holding_period
    sigma_scaled = sigma * np.sqrt(holding_period)
    
    # Z-score for confidence level
    z = norm.ppf(1 - confidence_level)
    
    var = -(mu_scaled + z * sigma_scaled)
    return var


def calculate_monte_carlo_var(returns: np.ndarray, confidence_level: float,
                              holding_period: int = 1, num_simulations: int = 10000) -> tuple:
    """
    Monte Carlo Simulation VaR
    Generates random scenarios based on historical distribution
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    # Generate random returns
    np.random.seed(42)
    simulated_returns = np.random.normal(
        mu * holding_period, 
        sigma * np.sqrt(holding_period), 
        num_simulations
    )
    
    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return var, simulated_returns


def calculate_cornish_fisher_var(returns: np.ndarray, confidence_level: float,
                                  holding_period: int = 1) -> float:
    """
    Cornish-Fisher VaR (Modified VaR)
    Adjusts for skewness and kurtosis
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Excess kurtosis
    
    # Z-score for confidence level
    z = norm.ppf(1 - confidence_level)
    
    # Cornish-Fisher expansion
    z_cf = (z + (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * skew**2 / 36)
    
    # Scale for holding period
    mu_scaled = mu * holding_period
    sigma_scaled = sigma * np.sqrt(holding_period)
    
    var = -(mu_scaled + z_cf * sigma_scaled)
    return var


def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float,
                                  holding_period: int = 1) -> float:
    """
    Expected Shortfall (ES) / Conditional VaR (CVaR)
    Average loss beyond VaR threshold
    """
    scaled_returns = returns * np.sqrt(holding_period)
    var_threshold = np.percentile(scaled_returns, (1 - confidence_level) * 100)
    
    # Average of returns below VaR threshold
    tail_returns = scaled_returns[scaled_returns <= var_threshold]
    
    if len(tail_returns) > 0:
        es = -np.mean(tail_returns)
    else:
        es = -var_threshold
    
    return es


# =============================================================================
# Backtesting
# =============================================================================

def backtest_var(returns: np.ndarray, var_estimates: np.ndarray, 
                 confidence_level: float) -> Dict[str, Any]:
    """
    VaR Backtesting using Kupiec and Christoffersen tests
    """
    # Count exceptions (actual loss > VaR)
    exceptions = np.sum(returns < -var_estimates)
    n = len(returns)
    expected_exceptions = int(n * (1 - confidence_level))
    exception_rate = exceptions / n
    expected_rate = 1 - confidence_level
    
    # Kupiec POF Test (Proportion of Failures)
    if exceptions > 0 and exceptions < n:
        lr_pof = -2 * (np.log((1 - expected_rate)**(n - exceptions) * expected_rate**exceptions) -
                       np.log((1 - exception_rate)**(n - exceptions) * exception_rate**exceptions))
        p_value_kupiec = 1 - stats.chi2.cdf(lr_pof, 1)
    else:
        lr_pof = 0
        p_value_kupiec = 1.0
    
    # Traffic light zones (Basel)
    if exceptions <= expected_exceptions * 1.5:
        zone = "green"
    elif exceptions <= expected_exceptions * 2:
        zone = "yellow"
    else:
        zone = "red"
    
    return {
        'total_observations': _to_native_type(n),
        'exceptions': _to_native_type(exceptions),
        'expected_exceptions': _to_native_type(expected_exceptions),
        'exception_rate': _to_native_type(exception_rate),
        'expected_rate': _to_native_type(expected_rate),
        'kupiec_statistic': _to_native_type(lr_pof),
        'kupiec_p_value': _to_native_type(p_value_kupiec),
        'model_valid': p_value_kupiec > 0.05,
        'basel_zone': zone
    }


# =============================================================================
# Risk Metrics
# =============================================================================

def calculate_risk_metrics(returns: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive risk metrics"""
    
    # Basic statistics
    mean_return = np.mean(returns)
    volatility = np.std(returns, ddof=1)
    annualized_vol = volatility * np.sqrt(252)
    
    # Distribution characteristics
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis
    
    # Drawdown analysis
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Sharpe Ratio (assuming 0 risk-free rate)
    sharpe = (mean_return * 252) / annualized_vol if annualized_vol > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else volatility
    sortino = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    # Calmar Ratio
    calmar = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'mean_return': _to_native_type(mean_return),
        'volatility': _to_native_type(volatility),
        'annualized_volatility': _to_native_type(annualized_vol),
        'skewness': _to_native_type(skewness),
        'kurtosis': _to_native_type(kurtosis),
        'max_drawdown': _to_native_type(max_drawdown),
        'sharpe_ratio': _to_native_type(sharpe),
        'sortino_ratio': _to_native_type(sortino),
        'calmar_ratio': _to_native_type(calmar),
        'best_day': _to_native_type(np.max(returns)),
        'worst_day': _to_native_type(np.min(returns)),
        'positive_days': _to_native_type(np.sum(returns > 0)),
        'negative_days': _to_native_type(np.sum(returns < 0)),
        'win_rate': _to_native_type(np.mean(returns > 0))
    }


# =============================================================================
# Plot Generation
# =============================================================================

def generate_var_distribution_plot(returns: np.ndarray, var_results: Dict, 
                                   portfolio_value: float) -> str:
    """Generate return distribution with VaR levels"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(returns * 100, bins=50, density=True, 
                                alpha=0.7, color='#64748b', edgecolor='white')
    
    # Normal distribution overlay
    mu, sigma = np.mean(returns) * 100, np.std(returns) * 100
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    ax.plot(x, norm.pdf(x, mu, sigma), 'b-', linewidth=2, label='Normal Dist.')
    
    # VaR lines
    colors = ['#ef4444', '#f97316']
    for i, (conf, var_pct) in enumerate(zip(var_results['confidence_levels'], 
                                            var_results['var_percentages'])):
        ax.axvline(x=-var_pct, color=colors[i % 2], linestyle='--', linewidth=2,
                   label=f'VaR {conf*100:.0f}%: {var_pct:.2f}%')
    
    ax.set_xlabel('Returns (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Return Distribution with VaR Levels', fontsize=13, fontweight='600')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_var_time_series_plot(returns: np.ndarray, rolling_var: np.ndarray,
                                  confidence_level: float) -> str:
    """Generate time series of returns vs rolling VaR"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    days = np.arange(len(returns))
    
    # Returns
    ax.plot(days, returns * 100, color='#64748b', linewidth=0.8, alpha=0.7, label='Daily Returns')
    
    # Rolling VaR (negative because VaR is a loss)
    valid_idx = ~np.isnan(rolling_var)
    ax.plot(days[valid_idx], -rolling_var[valid_idx] * 100, color='#ef4444', 
            linewidth=1.5, label=f'Rolling VaR ({confidence_level*100:.0f}%)')
    
    # Highlight breaches
    breaches = returns < -rolling_var
    breach_days = days[breaches & valid_idx]
    breach_returns = returns[breaches & valid_idx]
    ax.scatter(breach_days, breach_returns * 100, color='#ef4444', s=30, 
               zorder=5, label=f'VaR Breaches ({np.sum(breaches)})')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel('Returns (%)', fontsize=11)
    ax.set_title('Returns vs Rolling VaR', fontsize=13, fontweight='600')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_var_comparison_plot(var_results: Dict) -> str:
    """Generate comparison of VaR methods"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = list(var_results['method_comparison'].keys())
    conf_levels = var_results['confidence_levels']
    
    x = np.arange(len(methods))
    width = 0.35
    
    colors = ['#3b82f6', '#ef4444']
    
    for i, conf in enumerate(conf_levels[:2]):
        values = [var_results['method_comparison'][m][f'var_{int(conf*100)}'] 
                  for m in methods]
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, values, width, label=f'{conf*100:.0f}% VaR', 
                      color=colors[i], edgecolor='white')
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('VaR (%)', fontsize=11)
    ax.set_title('VaR Comparison by Method', fontsize=13, fontweight='600')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_monte_carlo_plot(simulated_returns: np.ndarray, var_value: float,
                              confidence_level: float) -> str:
    """Generate Monte Carlo simulation histogram"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histogram
    n, bins, patches = ax.hist(simulated_returns * 100, bins=100, density=True,
                                alpha=0.7, color='#6366f1', edgecolor='white')
    
    # Color the tail
    var_pct = var_value * 100
    for i, (patch, b) in enumerate(zip(patches, bins[:-1])):
        if b < -var_pct:
            patch.set_facecolor('#ef4444')
    
    ax.axvline(x=-var_pct, color='#ef4444', linestyle='--', linewidth=2,
               label=f'VaR {confidence_level*100:.0f}%: {var_pct:.2f}%')
    
    ax.set_xlabel('Simulated Returns (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Monte Carlo Simulation Distribution', fontsize=13, fontweight='600')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_drawdown_plot(returns: np.ndarray) -> str:
    """Generate drawdown chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # Cumulative returns
    cumulative = np.cumprod(1 + returns)
    days = np.arange(len(cumulative))
    
    ax1.plot(days, cumulative, color='#3b82f6', linewidth=1.5)
    ax1.fill_between(days, 1, cumulative, alpha=0.2, color='#3b82f6')
    ax1.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Cumulative Return', fontsize=11)
    ax1.set_title('Portfolio Performance & Drawdown', fontsize=13, fontweight='600')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max * 100
    
    ax2.fill_between(days, 0, drawdowns, color='#ef4444', alpha=0.5)
    ax2.plot(days, drawdowns, color='#ef4444', linewidth=1)
    ax2.set_xlabel('Days', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Interpretation Generator
# =============================================================================

def generate_interpretation(var_results: Dict, risk_metrics: Dict, 
                            backtest: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of VaR results"""
    key_insights = []
    
    # VaR level assessment
    var_95 = var_results['var_percentages'][0] if var_results['var_percentages'] else 0
    
    if var_95 > 5:
        key_insights.append({
            'title': 'High Risk Portfolio',
            'description': f'95% VaR of {var_95:.2f}% indicates significant daily risk exposure.',
            'status': 'warning'
        })
    elif var_95 > 2:
        key_insights.append({
            'title': 'Moderate Risk',
            'description': f'95% VaR of {var_95:.2f}% represents moderate risk level.',
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'Conservative Risk Profile',
            'description': f'95% VaR of {var_95:.2f}% indicates relatively low risk.',
            'status': 'positive'
        })
    
    # Distribution characteristics
    skew = risk_metrics.get('skewness', 0)
    kurt = risk_metrics.get('kurtosis', 0)
    
    if abs(skew) > 0.5 or kurt > 1:
        key_insights.append({
            'title': 'Non-Normal Distribution',
            'description': f'Skewness: {skew:.2f}, Excess Kurtosis: {kurt:.2f}. Consider using Cornish-Fisher or Historical VaR.',
            'status': 'neutral'
        })
    
    # Backtest results
    if backtest:
        if backtest['basel_zone'] == 'green':
            key_insights.append({
                'title': 'Model Validation Passed',
                'description': f"VaR model is in Basel green zone with {backtest['exceptions']} exceptions.",
                'status': 'positive'
            })
        elif backtest['basel_zone'] == 'yellow':
            key_insights.append({
                'title': 'Model Needs Review',
                'description': f"VaR model is in Basel yellow zone. Consider model recalibration.",
                'status': 'neutral'
            })
        else:
            key_insights.append({
                'title': 'Model Validation Failed',
                'description': f"VaR model is in Basel red zone with {backtest['exceptions']} exceptions. Urgent review needed.",
                'status': 'warning'
            })
    
    # Max drawdown
    max_dd = risk_metrics.get('max_drawdown', 0)
    if max_dd < -0.2:
        key_insights.append({
            'title': 'Significant Historical Drawdown',
            'description': f'Maximum drawdown of {max_dd*100:.1f}% observed in history.',
            'status': 'warning'
        })
    
    # Recommendations
    recommendations = []
    
    if var_95 > 3:
        recommendations.append("Consider portfolio diversification to reduce concentration risk.")
    
    if kurt > 3:
        recommendations.append("Fat tails detected. Use Expected Shortfall (ES) for more conservative risk measure.")
    
    if backtest and not backtest['model_valid']:
        recommendations.append("VaR model underestimates risk. Consider increasing confidence level or using stress testing.")
    
    sharpe = risk_metrics.get('sharpe_ratio', 0)
    if sharpe < 0.5:
        recommendations.append("Risk-adjusted returns are low. Review asset allocation strategy.")
    
    if not recommendations:
        recommendations.append("Risk metrics are within acceptable ranges. Continue monitoring.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/var-risk")
async def calculate_var(request: VarRequest) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) using multiple methodologies.
    
    Methodologies:
    1. Historical Simulation - Non-parametric, uses actual return distribution
    2. Parametric (Variance-Covariance) - Assumes normal distribution
    3. Monte Carlo Simulation - Generates random scenarios
    4. Cornish-Fisher - Adjusts for skewness and kurtosis
    
    Also calculates:
    - Expected Shortfall (CVaR)
    - Risk metrics (Sharpe, Sortino, Max Drawdown)
    - Backtesting statistics
    """
    try:
        returns = np.array(request.returns)
        
        if len(returns) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data. Need at least 30 returns.")
        
        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]
        
        # Calculate VaR for each confidence level
        var_values = []
        var_amounts = []
        var_percentages = []
        es_values = []
        es_amounts = []
        
        for conf in request.confidence_levels:
            if request.method == VarMethod.HISTORICAL:
                var = calculate_historical_var(returns, conf, request.holding_period)
            elif request.method == VarMethod.PARAMETRIC:
                var = calculate_parametric_var(returns, conf, request.holding_period)
            elif request.method == VarMethod.MONTE_CARLO:
                var, simulated = calculate_monte_carlo_var(returns, conf, 
                                                           request.holding_period,
                                                           request.num_simulations)
            else:  # Cornish-Fisher
                var = calculate_cornish_fisher_var(returns, conf, request.holding_period)
            
            var_values.append(_to_native_type(var))
            var_amounts.append(_to_native_type(var * request.portfolio_value))
            var_percentages.append(_to_native_type(var * 100))
            
            if request.calculate_es:
                es = calculate_expected_shortfall(returns, conf, request.holding_period)
                es_values.append(_to_native_type(es))
                es_amounts.append(_to_native_type(es * request.portfolio_value))
        
        # Compare all methods
        method_comparison = {}
        for method in VarMethod:
            method_vars = {}
            for conf in request.confidence_levels[:2]:  # Only first two confidence levels
                if method == VarMethod.HISTORICAL:
                    v = calculate_historical_var(returns, conf, request.holding_period)
                elif method == VarMethod.PARAMETRIC:
                    v = calculate_parametric_var(returns, conf, request.holding_period)
                elif method == VarMethod.MONTE_CARLO:
                    v, _ = calculate_monte_carlo_var(returns, conf, request.holding_period, 5000)
                else:
                    v = calculate_cornish_fisher_var(returns, conf, request.holding_period)
                method_vars[f'var_{int(conf*100)}'] = _to_native_type(v * 100)
            method_comparison[method.value] = method_vars
        
        # Calculate rolling VaR for time series
        rolling_var = np.full(len(returns), np.nan)
        window = min(request.rolling_window, len(returns) - 1)
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            rolling_var[i] = calculate_historical_var(window_returns, 
                                                       request.confidence_levels[0], 1)
        
        # Backtest
        backtest = backtest_var(returns[window:], rolling_var[window:], 
                               request.confidence_levels[0])
        
        # Risk metrics
        risk_metrics = calculate_risk_metrics(returns)
        
        # Build results
        var_results = {
            'confidence_levels': request.confidence_levels,
            'var_values': var_values,
            'var_amounts': var_amounts,
            'var_percentages': var_percentages,
            'es_values': es_values if request.calculate_es else None,
            'es_amounts': es_amounts if request.calculate_es else None,
            'method': request.method.value,
            'holding_period': request.holding_period,
            'method_comparison': method_comparison
        }
        
        # Generate plots
        plots = {
            'distribution': generate_var_distribution_plot(returns, var_results, 
                                                           request.portfolio_value),
            'time_series': generate_var_time_series_plot(returns, rolling_var,
                                                          request.confidence_levels[0]),
            'method_comparison': generate_var_comparison_plot(var_results),
            'drawdown': generate_drawdown_plot(returns)
        }
        
        # Monte Carlo plot if applicable
        if request.method == VarMethod.MONTE_CARLO:
            _, simulated = calculate_monte_carlo_var(returns, request.confidence_levels[0],
                                                     request.holding_period, request.num_simulations)
            plots['monte_carlo'] = generate_monte_carlo_plot(simulated, var_values[0],
                                                              request.confidence_levels[0])
        
        # Generate interpretation
        interpretation = generate_interpretation(var_results, risk_metrics, 
                                                  backtest, {
                                                      'method': request.method.value,
                                                      'holding_period': request.holding_period
                                                  })
        
        return {
            'var_results': var_results,
            'risk_metrics': risk_metrics,
            'backtest': backtest,
            'rolling_var': [_to_native_type(x) for x in rolling_var.tolist()],
            'plots': plots,
            'interpretation': interpretation,
            'parameters': {
                'portfolio_value': request.portfolio_value,
                'method': request.method.value,
                'holding_period': request.holding_period,
                'confidence_levels': request.confidence_levels,
                'num_observations': len(returns),
                'rolling_window': request.rolling_window
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VaR calculation failed: {str(e)}")
