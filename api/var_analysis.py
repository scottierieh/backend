"""
Value at Risk (VaR) Analysis Router for FastAPI
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
import time
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class VaRRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    asset_cols: List[str]
    method: Literal["parametric", "historical", "monte_carlo", "all"] = "parametric"
    confidence_level: float = 0.95
    time_horizon: int = 1
    portfolio_value: float = 1000000
    num_simulations: int = 10000


def _to_native_type(obj):
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns"""
    return np.log(prices_df / prices_df.shift(1)).dropna()


def parametric_var(returns: pd.DataFrame, weights: np.ndarray, 
                   confidence: float, portfolio_value: float, 
                   time_horizon: int) -> Dict:
    """Calculate Parametric (Variance-Covariance) VaR"""
    portfolio_returns = returns.dot(weights)
    
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    
    z_score = stats.norm.ppf(1 - confidence)
    
    var_1day = -(mu + z_score * sigma)
    var_nday = var_1day * np.sqrt(time_horizon)
    
    var_amount = var_nday * portfolio_value
    
    # CVaR (Expected Shortfall)
    cvar_z = stats.norm.pdf(z_score) / (1 - confidence)
    cvar_1day = -(mu - sigma * cvar_z)
    cvar_amount = cvar_1day * np.sqrt(time_horizon) * portfolio_value
    
    return {
        'var_amount': var_amount,
        'var_percent': var_nday,
        'cvar_amount': cvar_amount,
        'mu': mu,
        'sigma': sigma
    }


def historical_var(returns: pd.DataFrame, weights: np.ndarray,
                   confidence: float, portfolio_value: float,
                   time_horizon: int) -> Dict:
    """Calculate Historical Simulation VaR"""
    portfolio_returns = returns.dot(weights)
    
    var_percentile = np.percentile(portfolio_returns, (1 - confidence) * 100)
    var_1day = -var_percentile
    var_nday = var_1day * np.sqrt(time_horizon)
    var_amount = var_nday * portfolio_value
    
    # CVaR
    tail_returns = portfolio_returns[portfolio_returns <= var_percentile]
    cvar_1day = -tail_returns.mean() if len(tail_returns) > 0 else var_1day
    cvar_amount = cvar_1day * np.sqrt(time_horizon) * portfolio_value
    
    return {
        'var_amount': var_amount,
        'var_percent': var_nday,
        'cvar_amount': cvar_amount,
        'historical_returns': portfolio_returns.tolist()
    }


def monte_carlo_var(returns: pd.DataFrame, weights: np.ndarray,
                    confidence: float, portfolio_value: float,
                    time_horizon: int, num_simulations: int) -> Dict:
    """Calculate Monte Carlo VaR"""
    mu = returns.mean().values
    cov = returns.cov().values
    
    # Simulate returns
    simulated_returns = np.random.multivariate_normal(mu, cov, num_simulations)
    portfolio_returns = simulated_returns.dot(weights)
    
    # Scale for time horizon
    portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
    
    var_percentile = np.percentile(portfolio_returns, (1 - confidence) * 100)
    var_amount = -var_percentile * portfolio_value
    
    # CVaR
    tail_returns = portfolio_returns[portfolio_returns <= var_percentile]
    cvar_amount = -tail_returns.mean() * portfolio_value if len(tail_returns) > 0 else var_amount
    
    return {
        'var_amount': var_amount,
        'var_percent': -var_percentile,
        'cvar_amount': cvar_amount,
        'simulated_returns': portfolio_returns.tolist()
    }


def create_var_comparison_chart(var_p: float, var_h: float, var_mc: float, 
                                 cvar_p: float, cvar_h: float) -> str:
    """Create VaR method comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Parametric', 'Historical', 'Monte Carlo']
    var_values = [var_p, var_h, var_mc]
    cvar_values = [cvar_p, cvar_h, cvar_h]  # Use historical CVaR for MC
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, var_values, width, label='VaR', color='#ef4444', edgecolor='white')
    bars2 = ax.bar(x + width/2, cvar_values, width, label='CVaR', color='#f59e0b', edgecolor='white')
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(var_values) * 0.02,
                f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cvar_values) * 0.02,
                f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Amount ($)', fontsize=11)
    ax.set_title('VaR Comparison by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_distribution_chart(returns: List[float], var_amount: float, 
                               cvar_amount: float, portfolio_value: float) -> str:
    """Create returns distribution with VaR threshold"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    returns_pct = [r * 100 for r in returns]
    var_pct = -(var_amount / portfolio_value) * 100
    cvar_pct = -(cvar_amount / portfolio_value) * 100
    
    ax.hist(returns_pct, bins=50, density=True, alpha=0.7, color='#3b82f6', edgecolor='white')
    
    # VaR line
    ax.axvline(x=var_pct, color='#ef4444', linestyle='--', linewidth=2, label=f'VaR: {var_pct:.2f}%')
    ax.axvline(x=cvar_pct, color='#f59e0b', linestyle='--', linewidth=2, label=f'CVaR: {cvar_pct:.2f}%')
    
    # Shade tail
    ax.axvspan(min(returns_pct), var_pct, alpha=0.3, color='#ef4444')
    
    ax.set_xlabel('Daily Return (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Return Distribution with VaR Threshold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_historical_var_chart(returns: pd.Series, var_threshold: float, 
                                 confidence: float) -> str:
    """Create historical returns with VaR breaches"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    returns_pct = returns * 100
    var_pct = -var_threshold * 100
    
    colors = ['#ef4444' if r < -var_threshold else '#3b82f6' for r in returns]
    
    ax.bar(range(len(returns_pct)), returns_pct, color=colors, alpha=0.7, width=1.0)
    ax.axhline(y=var_pct, color='#ef4444', linestyle='--', linewidth=2, 
               label=f'{confidence*100:.0f}% VaR: {var_pct:.2f}%')
    
    breach_count = sum(1 for r in returns if r < -var_threshold)
    expected_breaches = len(returns) * (1 - confidence)
    
    ax.set_xlabel('Trading Day', fontsize=11)
    ax.set_ylabel('Daily Return (%)', fontsize=11)
    ax.set_title(f'Historical Returns with VaR Threshold (Breaches: {breach_count}, Expected: {expected_breaches:.0f})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_contribution_chart(assets: List[Dict]) -> str:
    """Create VaR contribution by asset chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [a['asset'] for a in assets]
    contributions = [a['var_amount'] for a in assets]
    colors = [COLORS[i % len(COLORS)] for i in range(len(assets))]
    
    bars = ax.barh(names, contributions, color=colors, edgecolor='white', linewidth=1)
    
    for bar, val in zip(bars, contributions):
        ax.text(bar.get_width() + max(contributions) * 0.02, 
                bar.get_y() + bar.get_height()/2,
                f'${val:,.0f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('VaR Contribution ($)', fontsize=11)
    ax.set_title('VaR Contribution by Asset', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(var_amount: float, var_percent: float, cvar_amount: float,
                          portfolio_value: float, confidence: float,
                          portfolio_vol: float) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Risk level
    if var_percent > 0.05:
        insights.append({
            'title': f'High VaR: {var_percent*100:.2f}% of Portfolio',
            'description': 'VaR exceeds 5% of portfolio value. Consider risk reduction.',
            'status': 'warning'
        })
    elif var_percent > 0.02:
        insights.append({
            'title': f'Moderate VaR: {var_percent*100:.2f}% of Portfolio',
            'description': 'VaR between 2-5%. Monitor closely.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low VaR: {var_percent*100:.2f}% of Portfolio',
            'description': 'VaR below 2%. Risk level is conservative.',
            'status': 'positive'
        })
    
    # CVaR vs VaR
    cvar_ratio = cvar_amount / var_amount if var_amount > 0 else 1
    if cvar_ratio > 1.5:
        insights.append({
            'title': f'High Tail Risk: CVaR {cvar_ratio:.1f}x VaR',
            'description': 'Significant fat-tail risk. Extreme losses could be severe.',
            'status': 'warning'
        })
    
    # Volatility
    if portfolio_vol > 0.25:
        insights.append({
            'title': f'High Volatility: {portfolio_vol*100:.1f}% Annual',
            'description': 'Portfolio volatility is elevated.',
            'status': 'warning'
        })
    
    # Confidence interpretation
    insights.append({
        'title': f'{confidence*100:.0f}% Confidence Interpretation',
        'description': f"Expect to exceed ${var_amount:,.0f} loss on {(1-confidence)*100:.0f}% of days (~{int((1-confidence)*252)} days/year).",
        'status': 'neutral'
    })
    
    return insights


@router.post("/var")
async def run_var_analysis(request: VaRRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        if request.date_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{request.date_col}' not found")
        
        for col in request.asset_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Asset column '{col}' not found")
        
        # Prepare price data
        df[request.date_col] = pd.to_datetime(df[request.date_col])
        df = df.sort_values(request.date_col)
        
        prices_df = df[request.asset_cols].astype(float)
        prices_df = prices_df.dropna()
        
        if len(prices_df) < 30:
            raise HTTPException(status_code=400, detail="Need at least 30 data points")
        
        # Calculate returns
        returns_df = calculate_returns(prices_df)
        
        # Equal weights if not specified
        n_assets = len(request.asset_cols)
        weights = np.ones(n_assets) / n_assets
        
        # Portfolio metrics
        portfolio_returns = returns_df.dot(weights)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        portfolio_return = portfolio_returns.mean() * 252
        
        # Calculate VaR by all methods
        param_result = parametric_var(returns_df, weights, request.confidence_level,
                                       request.portfolio_value, request.time_horizon)
        hist_result = historical_var(returns_df, weights, request.confidence_level,
                                      request.portfolio_value, request.time_horizon)
        mc_result = monte_carlo_var(returns_df, weights, request.confidence_level,
                                     request.portfolio_value, request.time_horizon,
                                     request.num_simulations)
        
        # Select primary result based on method
        if request.method == 'parametric':
            primary_var = param_result['var_amount']
            primary_cvar = param_result['cvar_amount']
        elif request.method == 'historical':
            primary_var = hist_result['var_amount']
            primary_cvar = hist_result['cvar_amount']
        elif request.method == 'monte_carlo':
            primary_var = mc_result['var_amount']
            primary_cvar = mc_result['cvar_amount']
        else:  # all - use parametric as primary
            primary_var = param_result['var_amount']
            primary_cvar = param_result['cvar_amount']
        
        # Asset-level VaR
        assets = []
        for i, col in enumerate(request.asset_cols):
            asset_returns = returns_df[col]
            asset_vol = asset_returns.std() * np.sqrt(252)
            asset_var = stats.norm.ppf(1 - request.confidence_level) * asset_returns.std() * np.sqrt(request.time_horizon)
            asset_var_amount = -asset_var * request.portfolio_value * weights[i]
            
            # Asset CVaR
            z = stats.norm.ppf(1 - request.confidence_level)
            cvar_z = stats.norm.pdf(z) / (1 - request.confidence_level)
            asset_cvar = (asset_returns.mean() - asset_returns.std() * cvar_z) * np.sqrt(request.time_horizon)
            asset_cvar_amount = -asset_cvar * request.portfolio_value * weights[i]
            
            assets.append({
                'asset': col,
                'weight': weights[i],
                'var_amount': max(asset_var_amount, 0),
                'var_percent': -asset_var,
                'cvar_amount': max(asset_cvar_amount, 0),
                'volatility': asset_vol,
                'contribution_to_var': weights[i] * asset_var_amount / primary_var if primary_var > 0 else 0
            })
        
        # VaR at different confidence levels
        var_by_confidence = []
        for conf in [0.90, 0.95, 0.99]:
            z = stats.norm.ppf(1 - conf)
            var_temp = -(portfolio_returns.mean() + z * portfolio_returns.std()) * np.sqrt(request.time_horizon)
            var_by_confidence.append({
                'confidence': conf,
                'var_amount': var_temp * request.portfolio_value
            })
        
        # Metrics
        max_drawdown = (prices_df / prices_df.cummax() - 1).min().min()
        worst_day = portfolio_returns.min()
        var_threshold = param_result['var_percent'] / np.sqrt(request.time_horizon)
        breach_count = sum(1 for r in portfolio_returns if r < -var_threshold)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        metrics = {
            'max_drawdown': abs(max_drawdown),
            'worst_day_loss': abs(worst_day) * request.portfolio_value,
            'breach_count': breach_count,
            'sharpe_ratio': sharpe
        }
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {
            'var_comparison': create_var_comparison_chart(
                param_result['var_amount'], hist_result['var_amount'], mc_result['var_amount'],
                param_result['cvar_amount'], hist_result['cvar_amount']
            ),
            'distribution_chart': create_distribution_chart(
                portfolio_returns.tolist(), primary_var, primary_cvar, request.portfolio_value
            ),
            'historical_var': create_historical_var_chart(
                portfolio_returns, var_threshold, request.confidence_level
            ),
            'contribution_chart': create_contribution_chart(assets)
        }
        
        # Key insights
        key_insights = generate_key_insights(
            primary_var, primary_var / request.portfolio_value, primary_cvar,
            request.portfolio_value, request.confidence_level, portfolio_vol
        )
        
        results = {
            'portfolio_value': request.portfolio_value,
            'confidence_level': request.confidence_level,
            'time_horizon': request.time_horizon,
            'var_parametric': param_result['var_amount'],
            'var_historical': hist_result['var_amount'],
            'var_monte_carlo': mc_result['var_amount'],
            'cvar_parametric': param_result['cvar_amount'],
            'cvar_historical': hist_result['cvar_amount'],
            'portfolio_volatility': portfolio_vol,
            'portfolio_return': portfolio_return,
            'assets': [{k: _to_native_type(v) for k, v in a.items()} for a in assets],
            'var_by_confidence': [{k: _to_native_type(v) for k, v in v.items()} for v in var_by_confidence],
            'historical_losses': portfolio_returns.tolist(),
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'method': request.method,
            'var_amount': primary_var,
            'var_percent': primary_var / request.portfolio_value,
            'solve_time_ms': solve_time_ms
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VaR analysis failed: {str(e)}")
