"""
Portfolio Optimization Router for FastAPI
Mean-Variance Optimization, Efficient Frontier, Risk Analysis
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
import io
import base64
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class PortfolioRequest(BaseModel):
    data: List[Dict[str, Any]]
    asset_cols: List[str]  # Columns containing asset prices or returns
    date_col: Optional[str] = None
    data_type: Literal["prices", "returns"] = "prices"  # Whether data is prices or returns
    risk_free_rate: float = 0.02  # Annual risk-free rate
    target_return: Optional[float] = None  # For minimum variance at target return
    target_volatility: Optional[float] = None  # For maximum return at target volatility
    constraints: Optional[Dict[str, Any]] = None  # Min/max weights per asset
    short_selling: bool = False  # Allow negative weights
    num_portfolios: int = 5000  # For Monte Carlo simulation
    optimization_method: Literal["max_sharpe", "min_variance", "max_return", "risk_parity", "equal_weight"] = "max_sharpe"


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


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from prices"""
    return prices.pct_change().dropna()


def calculate_portfolio_metrics(weights: np.ndarray, mean_returns: np.ndarray,
                                cov_matrix: np.ndarray, risk_free_rate: float,
                                trading_days: int = 252) -> Dict[str, float]:
    """Calculate portfolio return, volatility, and Sharpe ratio"""
    # Annualized return
    portfolio_return = np.sum(mean_returns * weights) * trading_days
    
    # Annualized volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    
    # Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        'return': _to_native_type(portfolio_return),
        'volatility': _to_native_type(portfolio_volatility),
        'sharpe_ratio': _to_native_type(sharpe_ratio)
    }


def negative_sharpe_ratio(weights: np.ndarray, mean_returns: np.ndarray,
                          cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """Negative Sharpe ratio for minimization"""
    metrics = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
    return -metrics['sharpe_ratio']


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Portfolio volatility for minimization"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)


def negative_portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Negative portfolio return for minimization"""
    return -np.sum(mean_returns * weights) * 252


def risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Risk parity objective: minimize difference in risk contributions"""
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / portfolio_vol
    target_risk = portfolio_vol / len(weights)
    return np.sum((risk_contrib - target_risk) ** 2)


def optimize_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                       risk_free_rate: float, method: str,
                       short_selling: bool = False,
                       target_return: Optional[float] = None,
                       target_volatility: Optional[float] = None,
                       constraints_dict: Optional[Dict] = None) -> Dict[str, Any]:
    """Optimize portfolio based on specified method"""
    n_assets = len(mean_returns)
    
    # Initial weights (equal weight)
    init_weights = np.array([1/n_assets] * n_assets)
    
    # Bounds
    if short_selling:
        bounds = tuple((-1, 1) for _ in range(n_assets))
    else:
        bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Apply custom constraints if provided
    if constraints_dict:
        bounds = tuple(
            (constraints_dict.get(f'min_{i}', bounds[i][0]),
             constraints_dict.get(f'max_{i}', bounds[i][1]))
            for i in range(n_assets)
        )
    
    # Constraint: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Add target return constraint if specified
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(mean_returns * x) * 252 - target_return
        })
    
    # Add target volatility constraint if specified
    if target_volatility is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: portfolio_volatility(x, cov_matrix) - target_volatility
        })
    
    # Optimize based on method
    if method == 'max_sharpe':
        result = minimize(negative_sharpe_ratio, init_weights,
                         args=(mean_returns, cov_matrix, risk_free_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'min_variance':
        result = minimize(portfolio_volatility, init_weights,
                         args=(cov_matrix,),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'max_return':
        result = minimize(negative_portfolio_return, init_weights,
                         args=(mean_returns,),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'risk_parity':
        # Risk parity doesn't require weights to sum to 1 initially
        rp_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        result = minimize(risk_parity_objective, init_weights,
                         args=(cov_matrix,),
                         method='SLSQP', bounds=bounds, constraints=rp_constraints)
    elif method == 'equal_weight':
        result = type('Result', (), {'x': init_weights, 'success': True})()
    else:
        result = minimize(negative_sharpe_ratio, init_weights,
                         args=(mean_returns, cov_matrix, risk_free_rate),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return {
        'weights': result.x,
        'success': result.success if hasattr(result, 'success') else True
    }


def generate_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                risk_free_rate: float, short_selling: bool = False,
                                n_points: int = 100) -> Dict[str, Any]:
    """Generate efficient frontier points"""
    n_assets = len(mean_returns)
    
    # Find min and max return portfolios
    bounds = tuple((-1, 1) if short_selling else (0, 1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Minimum variance portfolio
    min_var_result = minimize(portfolio_volatility, np.array([1/n_assets] * n_assets),
                              args=(cov_matrix,), method='SLSQP',
                              bounds=bounds, constraints=constraints)
    min_var_return = np.sum(mean_returns * min_var_result.x) * 252
    
    # Maximum return portfolio
    max_ret_result = minimize(negative_portfolio_return, np.array([1/n_assets] * n_assets),
                              args=(mean_returns,), method='SLSQP',
                              bounds=bounds, constraints=constraints)
    max_return = np.sum(mean_returns * max_ret_result.x) * 252
    
    # Generate target returns
    target_returns = np.linspace(min_var_return, max_return, n_points)
    
    frontier_volatilities = []
    frontier_returns = []
    frontier_weights = []
    
    for target in target_returns:
        target_constraint = {
            'type': 'eq',
            'fun': lambda x, t=target: np.sum(mean_returns * x) * 252 - t
        }
        
        result = minimize(portfolio_volatility, np.array([1/n_assets] * n_assets),
                         args=(cov_matrix,), method='SLSQP',
                         bounds=bounds, constraints=constraints + [target_constraint])
        
        if result.success:
            vol = portfolio_volatility(result.x, cov_matrix)
            frontier_volatilities.append(vol)
            frontier_returns.append(target)
            frontier_weights.append(result.x.tolist())
    
    return {
        'returns': [_to_native_type(r) for r in frontier_returns],
        'volatilities': [_to_native_type(v) for v in frontier_volatilities],
        'weights': frontier_weights
    }


def monte_carlo_simulation(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                           risk_free_rate: float, n_portfolios: int = 5000,
                           short_selling: bool = False) -> Dict[str, Any]:
    """Run Monte Carlo simulation to generate random portfolios"""
    n_assets = len(mean_returns)
    
    results = {
        'returns': [],
        'volatilities': [],
        'sharpe_ratios': [],
        'weights': []
    }
    
    for _ in range(n_portfolios):
        if short_selling:
            weights = np.random.randn(n_assets)
            weights = weights / np.sum(np.abs(weights))  # Normalize
        else:
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
        
        metrics = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
        
        results['returns'].append(metrics['return'])
        results['volatilities'].append(metrics['volatility'])
        results['sharpe_ratios'].append(metrics['sharpe_ratio'])
        results['weights'].append(weights.tolist())
    
    return results


def calculate_risk_metrics(returns: pd.DataFrame, weights: np.ndarray,
                           confidence_level: float = 0.95) -> Dict[str, Any]:
    """Calculate additional risk metrics"""
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Value at Risk (VaR)
    var_parametric = stats.norm.ppf(1 - confidence_level) * portfolio_returns.std() * np.sqrt(252)
    var_historical = np.percentile(portfolio_returns, (1 - confidence_level) * 100) * np.sqrt(252)
    
    # Conditional VaR (Expected Shortfall)
    cvar = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, (1 - confidence_level) * 100)].mean() * np.sqrt(252)
    
    # Maximum Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Sortino Ratio (downside deviation)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (portfolio_returns.mean() * 252) / downside_std if downside_std > 0 else 0
    
    # Calmar Ratio
    annual_return = portfolio_returns.mean() * 252
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Beta and Alpha (vs equal-weight benchmark)
    benchmark_returns = returns.mean(axis=1)
    if len(portfolio_returns) > 1:
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_var = benchmark_returns.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 1
        alpha = annual_return - beta * (benchmark_returns.mean() * 252)
    else:
        beta = 1
        alpha = 0
    
    return {
        'var_parametric': _to_native_type(abs(var_parametric)),
        'var_historical': _to_native_type(abs(var_historical)),
        'cvar': _to_native_type(abs(cvar)),
        'max_drawdown': _to_native_type(max_drawdown),
        'sortino_ratio': _to_native_type(sortino_ratio),
        'calmar_ratio': _to_native_type(calmar_ratio),
        'beta': _to_native_type(beta),
        'alpha': _to_native_type(alpha),
        'skewness': _to_native_type(stats.skew(portfolio_returns)),
        'kurtosis': _to_native_type(stats.kurtosis(portfolio_returns))
    }


def create_efficient_frontier_chart(frontier: Dict, mc_results: Dict,
                                    optimal_portfolio: Dict, assets_metrics: List[Dict]) -> str:
    """Create efficient frontier visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Monte Carlo portfolios
    scatter = ax.scatter(mc_results['volatilities'], mc_results['returns'],
                        c=mc_results['sharpe_ratios'], cmap='viridis',
                        alpha=0.5, s=10, label='Random Portfolios')
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    # Efficient frontier
    ax.plot(frontier['volatilities'], frontier['returns'], 'r-', linewidth=3,
            label='Efficient Frontier')
    
    # Optimal portfolio
    ax.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'],
               c='red', marker='*', s=500, zorder=5, label='Optimal Portfolio',
               edgecolors='white', linewidth=2)
    
    # Individual assets
    for asset in assets_metrics:
        ax.scatter(asset['volatility'], asset['return'], marker='D', s=100,
                   zorder=4, edgecolors='black', linewidth=1)
        ax.annotate(asset['name'], (asset['volatility'], asset['return']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier & Portfolio Optimization', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_weights_chart(weights: np.ndarray, asset_names: List[str]) -> str:
    """Create portfolio weights visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
    
    # Pie chart
    non_zero_weights = [(w, n, c) for w, n, c in zip(weights, asset_names, colors) if abs(w) > 0.001]
    if non_zero_weights:
        w_vals, w_names, w_colors = zip(*non_zero_weights)
        wedges, texts, autotexts = ax1.pie(np.abs(w_vals), labels=w_names,
                                           autopct='%1.1f%%', colors=w_colors,
                                           startangle=90)
        ax1.set_title('Portfolio Allocation', fontsize=12, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(asset_names, weights * 100, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Weight (%)')
    ax2.set_title('Asset Weights', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(asset_names, rotation=45, ha='right')
    
    for bar, w in zip(bars, weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{w*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_correlation_heatmap(returns: pd.DataFrame) -> str:
    """Create correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = returns.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Correlation'})
    
    ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_risk_return_chart(assets_metrics: List[Dict]) -> str:
    """Create individual asset risk-return chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    returns = [a['return'] for a in assets_metrics]
    vols = [a['volatility'] for a in assets_metrics]
    sharpes = [a['sharpe_ratio'] for a in assets_metrics]
    names = [a['name'] for a in assets_metrics]
    
    scatter = ax.scatter(vols, returns, c=sharpes, cmap='viridis', s=200,
                        edgecolors='white', linewidth=2)
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    for i, name in enumerate(names):
        ax.annotate(name, (vols[i], returns[i]), xytext=(8, 0),
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Individual Asset Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cumulative_returns_chart(returns: pd.DataFrame, weights: np.ndarray,
                                    asset_names: List[str]) -> str:
    """Create cumulative returns chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Portfolio cumulative returns
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    ax.plot(portfolio_cumulative.index, portfolio_cumulative.values, 'b-',
            linewidth=3, label='Optimized Portfolio')
    
    # Equal weight benchmark
    equal_weights = np.array([1/len(asset_names)] * len(asset_names))
    benchmark_returns = (returns * equal_weights).sum(axis=1)
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 'gray',
            linewidth=2, linestyle='--', label='Equal Weight', alpha=0.7)
    
    # Individual assets (lighter)
    for i, name in enumerate(asset_names):
        asset_cumulative = (1 + returns.iloc[:, i]).cumprod()
        ax.plot(asset_cumulative.index, asset_cumulative.values,
                alpha=0.3, linewidth=1, label=name)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(optimal: Dict, risk_metrics: Dict, assets_metrics: List[Dict],
                          weights: np.ndarray, asset_names: List[str]) -> List[Dict[str, Any]]:
    """Generate key insights for portfolio optimization"""
    insights = []
    
    # Sharpe ratio assessment
    sharpe = optimal.get('sharpe_ratio', 0)
    if sharpe >= 2:
        insights.append({
            'title': 'Excellent Risk-Adjusted Return',
            'description': f'Sharpe ratio = {sharpe:.2f} (≥2.0). Outstanding risk-adjusted performance.',
            'status': 'positive'
        })
    elif sharpe >= 1:
        insights.append({
            'title': 'Good Risk-Adjusted Return',
            'description': f'Sharpe ratio = {sharpe:.2f} (≥1.0). Solid risk-adjusted performance.',
            'status': 'positive'
        })
    elif sharpe >= 0.5:
        insights.append({
            'title': 'Moderate Risk-Adjusted Return',
            'description': f'Sharpe ratio = {sharpe:.2f} (0.5-1.0). Acceptable but room for improvement.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': 'Low Risk-Adjusted Return',
            'description': f'Sharpe ratio = {sharpe:.2f} (<0.5). Consider alternative strategies.',
            'status': 'warning'
        })
    
    # Diversification
    hhi = np.sum(weights ** 2)  # Herfindahl-Hirschman Index
    effective_n = 1 / hhi if hhi > 0 else 0
    if effective_n >= len(weights) * 0.7:
        insights.append({
            'title': 'Well Diversified',
            'description': f'Effective number of assets: {effective_n:.1f}. Good diversification.',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': 'Concentrated Portfolio',
            'description': f'Effective number of assets: {effective_n:.1f}. Portfolio is relatively concentrated.',
            'status': 'neutral'
        })
    
    # Top holdings
    sorted_idx = np.argsort(weights)[::-1]
    top_asset = asset_names[sorted_idx[0]]
    top_weight = weights[sorted_idx[0]]
    insights.append({
        'title': f'Largest Position: {top_asset}',
        'description': f'Weight: {top_weight*100:.1f}%. This asset has the highest allocation.',
        'status': 'neutral'
    })
    
    # Risk metrics
    max_dd = risk_metrics.get('max_drawdown', 0)
    if max_dd is not None and abs(max_dd) > 0.2:
        insights.append({
            'title': 'Significant Drawdown Risk',
            'description': f'Maximum drawdown: {abs(max_dd)*100:.1f}%. Consider risk management strategies.',
            'status': 'warning'
        })
    elif max_dd is not None:
        insights.append({
            'title': 'Moderate Drawdown Risk',
            'description': f'Maximum drawdown: {abs(max_dd)*100:.1f}%. Acceptable historical drawdown.',
            'status': 'positive'
        })
    
    # VaR insight
    var = risk_metrics.get('var_parametric', 0)
    if var is not None:
        insights.append({
            'title': f'Value at Risk (95%)',
            'description': f'VaR: {abs(var)*100:.1f}%. Maximum expected loss in 95% of scenarios.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/portfolio")
async def run_portfolio_optimization(request: PortfolioRequest) -> Dict[str, Any]:
    """
    Perform Portfolio Optimization analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate columns
        for col in request.asset_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Extract price/return data
        asset_data = df[request.asset_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(asset_data) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 data points for reliable analysis")
        
        # Convert to returns if prices
        if request.data_type == "prices":
            returns = calculate_returns(asset_data)
        else:
            returns = asset_data
        
        if len(returns) < 10:
            raise HTTPException(status_code=400, detail="Not enough return data after processing")
        
        # Calculate statistics
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Individual asset metrics
        assets_metrics = []
        for i, col in enumerate(request.asset_cols):
            asset_return = mean_returns[i] * 252
            asset_vol = returns.iloc[:, i].std() * np.sqrt(252)
            asset_sharpe = (asset_return - request.risk_free_rate) / asset_vol if asset_vol > 0 else 0
            assets_metrics.append({
                'name': col,
                'return': _to_native_type(asset_return),
                'volatility': _to_native_type(asset_vol),
                'sharpe_ratio': _to_native_type(asset_sharpe)
            })
        
        # Optimize portfolio
        opt_result = optimize_portfolio(
            mean_returns, cov_matrix, request.risk_free_rate,
            request.optimization_method, request.short_selling,
            request.target_return, request.target_volatility
        )
        
        optimal_weights = opt_result['weights']
        optimal_metrics = calculate_portfolio_metrics(
            optimal_weights, mean_returns, cov_matrix, request.risk_free_rate
        )
        
        # Generate efficient frontier
        frontier = generate_efficient_frontier(
            mean_returns, cov_matrix, request.risk_free_rate, request.short_selling
        )
        
        # Monte Carlo simulation
        mc_results = monte_carlo_simulation(
            mean_returns, cov_matrix, request.risk_free_rate,
            request.num_portfolios, request.short_selling
        )
        
        # Risk metrics
        risk_metrics = calculate_risk_metrics(returns, optimal_weights)
        
        # Create visualizations
        visualizations = {}
        visualizations['efficient_frontier'] = create_efficient_frontier_chart(
            frontier, mc_results, optimal_metrics, assets_metrics
        )
        visualizations['weights_chart'] = create_weights_chart(
            optimal_weights, request.asset_cols
        )
        visualizations['correlation_heatmap'] = create_correlation_heatmap(returns)
        visualizations['risk_return'] = create_risk_return_chart(assets_metrics)
        visualizations['cumulative_returns'] = create_cumulative_returns_chart(
            returns, optimal_weights, request.asset_cols
        )
        
        # Generate insights
        insights = generate_key_insights(
            optimal_metrics, risk_metrics, assets_metrics,
            optimal_weights, request.asset_cols
        )
        
        # Portfolio weights with names
        portfolio_weights = [
            {'asset': name, 'weight': _to_native_type(w), 'weight_pct': _to_native_type(w * 100)}
            for name, w in zip(request.asset_cols, optimal_weights)
        ]
        
        # Summary
        summary = {
            'expected_return': optimal_metrics['return'],
            'volatility': optimal_metrics['volatility'],
            'sharpe_ratio': optimal_metrics['sharpe_ratio'],
            'method': request.optimization_method,
            'n_assets': len(request.asset_cols),
            'n_periods': len(returns)
        }
        
        return {
            'success': True,
            'optimal_portfolio': optimal_metrics,
            'portfolio_weights': portfolio_weights,
            'assets_metrics': assets_metrics,
            'risk_metrics': risk_metrics,
            'efficient_frontier': frontier,
            'monte_carlo_summary': {
                'n_portfolios': len(mc_results['returns']),
                'max_sharpe': _to_native_type(max(mc_results['sharpe_ratios'])),
                'min_volatility': _to_native_type(min(mc_results['volatilities'])),
                'max_return': _to_native_type(max(mc_results['returns']))
            },
            'correlation_matrix': returns.corr().values.tolist(),
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")
