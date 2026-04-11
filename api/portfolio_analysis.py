"""
Portfolio Optimization FastAPI Endpoint
Modern Portfolio Theory (MPT) using PyPortfolioOpt library
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings

# PyPortfolioOpt imports
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import objective_functions

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class PortfolioRequest(BaseModel):
    """Request model for Portfolio Optimization"""
    data: List[Dict[str, Any]]
    date_col: str
    asset_cols: List[str]
    optimization_method: str = Field(default="max_sharpe", pattern="^(max_sharpe|min_variance|max_return)$")
    risk_free_rate: float = Field(default=0.02, ge=0, le=0.2)
    target_return: Optional[float] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)


def prepare_price_data(df: pd.DataFrame, date_col: str, asset_cols: List[str]) -> pd.DataFrame:
    """Prepare price data for PyPortfolioOpt"""
    df = df.set_index(date_col)
    prices = df[asset_cols].copy()
    for col in asset_cols:
        prices[col] = pd.to_numeric(prices[col], errors='coerce')
    prices = prices.dropna()
    return prices


def optimize_portfolio(prices: pd.DataFrame, method: str, risk_free_rate: float, constraints: Dict):
    """Optimize portfolio using PyPortfolioOpt"""
    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S = risk_models.sample_cov(prices, frequency=252)
    ef = EfficientFrontier(mu, S)
    
    # Add weight constraints
    if 'min_weight' in constraints or 'max_weight' in constraints:
        min_w = constraints.get('min_weight', 0)
        max_w = constraints.get('max_weight', 1)
        ef.add_constraint(lambda w: w >= min_w)
        ef.add_constraint(lambda w: w <= max_w)
    
    # Optimize
    if method == 'max_sharpe':
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif method == 'min_variance':
        weights = ef.min_volatility()
    elif method == 'max_return':
        ef.add_objective(objective_functions.ex_ante_returns)
        weights = ef.max_quadratic_utility(risk_aversion=0)
    else:
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    
    cleaned_weights = ef.clean_weights()
    return cleaned_weights, ef


def calculate_portfolio_performance(weights: Dict[str, float], mu: pd.Series, S: pd.DataFrame, risk_free_rate: float):
    """Calculate portfolio performance metrics"""
    weight_array = np.array([weights[asset] for asset in mu.index])
    expected_return = np.dot(weight_array, mu)
    volatility = np.sqrt(np.dot(weight_array, np.dot(S, weight_array)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    return {
        'expected_return': float(expected_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio)
    }


def generate_efficient_frontier(prices: pd.DataFrame, risk_free_rate: float, n_points: int = 100):
    """Generate efficient frontier points"""
    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S = risk_models.sample_cov(prices, frequency=252)
    
    min_ret = mu.min()
    max_ret = mu.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    efficient_portfolios = []
    for target_ret in target_returns:
        try:
            ef = EfficientFrontier(mu, S)
            ef.efficient_return(target_return=target_ret)
            ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            
            if not np.isnan(vol) and not np.isinf(vol) and vol > 0:
                efficient_portfolios.append({'return': float(ret), 'volatility': float(vol)})
        except:
            continue
    
    return pd.DataFrame(efficient_portfolios)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_visualizations(prices: pd.DataFrame, optimal_weights: Dict[str, float], 
                           risk_free_rate: float, efficient_frontier: pd.DataFrame,
                           portfolio_metrics: Dict[str, float]):
    """Generate portfolio optimization visualizations"""
    visualizations = {}
    display_weights = {k: v for k, v in optimal_weights.items() if v > 0.001}
    
    # 1. Allocation Pie
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(display_weights))
    wedges, texts, autotexts = ax.pie(display_weights.values(), labels=display_weights.keys(),
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Optimal Portfolio Allocation', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    visualizations['allocation_pie'] = fig_to_base64(fig)
    
    # 2. Efficient Frontier
    fig, ax = plt.subplots(figsize=(12, 8))
    if len(efficient_frontier) > 0:
        ax.plot(efficient_frontier['volatility'] * 100, efficient_frontier['return'] * 100,
               'b-', linewidth=2, label='Efficient Frontier')
    
    ax.scatter(portfolio_metrics['volatility'] * 100, portfolio_metrics['expected_return'] * 100,
              marker='*', color='red', s=500, 
              label=f"Optimal (Sharpe: {portfolio_metrics['sharpe_ratio']:.2f})",
              zorder=5, edgecolors='black', linewidths=2)
    
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    std_returns = returns.std() * np.sqrt(252)
    ax.scatter(std_returns * 100, mean_returns * 100, marker='o', s=100, alpha=0.6, label='Individual Assets')
    
    for asset in prices.columns:
        ax.annotate(asset, (std_returns[asset] * 100, mean_returns[asset] * 100),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Volatility %', fontsize=11)
    ax.set_ylabel('Expected Return %', fontsize=11)
    ax.set_title('Efficient Frontier & Optimal Portfolio', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['efficient_frontier'] = fig_to_base64(fig)
    
    # 3. Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = returns.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    visualizations['correlation_matrix'] = fig_to_base64(fig)
    
    # 4. Risk-Return Bars
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(prices.columns))
    width = 0.35
    ax.bar(x - width/2, mean_returns * 100, width, label='Return %', color='#4A90E2')
    ax.bar(x + width/2, std_returns * 100, width, label='Risk %', color='#E74C3C')
    ax.set_xlabel('Assets', fontsize=11)
    ax.set_ylabel('Percentage %', fontsize=11)
    ax.set_title('Individual Asset Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prices.columns, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    visualizations['risk_return_bars'] = fig_to_base64(fig)
    
    # 5. Weights Bar
    fig, ax = plt.subplots(figsize=(12, 6))
    assets = list(optimal_weights.keys())
    weights_pct = [optimal_weights[a] * 100 for a in assets]
    colors = sns.color_palette("husl", len(assets))
    bars = ax.bar(assets, weights_pct, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Assets', fontsize=11)
    ax.set_ylabel('Allocation %', fontsize=11)
    ax.set_title('Optimal Portfolio Weights', fontsize=14, fontweight='bold')
    ax.set_xticklabels(assets, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, weight in zip(bars, weights_pct):
        if weight > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{weight:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    visualizations['weights_bar'] = fig_to_base64(fig)
    
    return visualizations


def generate_insights(optimal_weights: Dict[str, float], portfolio_metrics: Dict[str, float], prices: pd.DataFrame):
    """Generate key insights"""
    insights = []
    active_assets = {k: v for k, v in optimal_weights.items() if v > 0.01}
    n_assets_used = len(active_assets)
    n_total_assets = len(optimal_weights)
    concentration = max(optimal_weights.values())
    sharpe = portfolio_metrics['sharpe_ratio']
    vol = portfolio_metrics['volatility']
    
    # Diversification
    if n_assets_used >= n_total_assets * 0.8:
        insights.append({
            'title': 'Well-Diversified Portfolio',
            'description': f'Uses {n_assets_used} out of {n_total_assets} assets.',
            'status': 'positive'
        })
    elif concentration > 0.5:
        top_asset = max(optimal_weights.items(), key=lambda x: x[1])
        insights.append({
            'title': 'Concentrated Position',
            'description': f'{concentration*100:.1f}% in {top_asset[0]}.',
            'status': 'warning'
        })
    
    # Sharpe ratio
    if sharpe > 1.5:
        insights.append({
            'title': 'Excellent Risk-Adjusted Returns',
            'description': f'Sharpe ratio of {sharpe:.2f} indicates strong performance.',
            'status': 'positive'
        })
    elif sharpe > 1.0:
        insights.append({
            'title': 'Good Risk-Adjusted Returns',
            'description': f'Sharpe ratio of {sharpe:.2f} shows solid performance.',
            'status': 'positive'
        })
    
    # Volatility
    returns = prices.pct_change().dropna()
    mean_vol = returns.std().mean() * np.sqrt(252)
    if vol < mean_vol * 0.8:
        insights.append({
            'title': 'Reduced Portfolio Risk',
            'description': f'Portfolio volatility ({vol*100:.1f}%) lower than average asset volatility.',
            'status': 'positive'
        })
    
    return insights


@router.post("/portfolio-optimization")
async def optimize_portfolio_endpoint(request: PortfolioRequest):
    """Portfolio Optimization using PyPortfolioOpt"""
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 30:
            raise HTTPException(400, "Need at least 30 periods")
        
        df = pd.DataFrame(request.data)
        required_cols = [request.date_col] + request.asset_cols
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns: {missing}")
        
        df[request.date_col] = pd.to_datetime(df[request.date_col], errors='coerce')
        df = df.dropna(subset=[request.date_col])
        df = df.sort_values(request.date_col)
        
        if len(df) < 30:
            raise HTTPException(400, f"Only {len(df)} valid rows after cleaning")
        
        prices = prepare_price_data(df, request.date_col, request.asset_cols)
        
        if len(prices) < 20:
            raise HTTPException(400, f"Only {len(prices)} price periods (need ≥20)")
        
        optimal_weights, ef = optimize_portfolio(prices, request.optimization_method,
                                                 request.risk_free_rate, request.constraints)
        
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        S = risk_models.sample_cov(prices, frequency=252)
        portfolio_metrics = calculate_portfolio_performance(optimal_weights, mu, S, request.risk_free_rate)
        
        efficient_frontier = generate_efficient_frontier(prices, request.risk_free_rate)
        visualizations = generate_visualizations(prices, optimal_weights, request.risk_free_rate,
                                                efficient_frontier, portfolio_metrics)
        insights = generate_insights(optimal_weights, portfolio_metrics, prices)
        
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        std_returns = returns.std() * np.sqrt(252)
        
        asset_metrics = []
        for asset in request.asset_cols:
            sharpe = (mean_returns[asset] - request.risk_free_rate) / std_returns[asset] if std_returns[asset] > 0 else 0
            asset_metrics.append({
                'asset': asset,
                'weight': float(optimal_weights.get(asset, 0)),
                'expected_return': float(mean_returns[asset]),
                'volatility': float(std_returns[asset]),
                'sharpe': float(sharpe)
            })
        
        response_data = {
            'success': True,
            'results': {
                'optimization_method': request.optimization_method,
                'n_assets': len(request.asset_cols),
                'n_periods': len(prices),
                'optimal_weights': {k: float(v) for k, v in optimal_weights.items()},
                'portfolio_metrics': portfolio_metrics,
                'asset_metrics': asset_metrics,
                'efficient_frontier': efficient_frontier.to_dict('records')[:50] if len(efficient_frontier) > 0 else []
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'portfolio_optimization',
                'n_assets': len(request.asset_cols),
                'expected_return': round(portfolio_metrics['expected_return'] * 100, 2),
                'volatility': round(portfolio_metrics['volatility'] * 100, 2),
                'sharpe_ratio': round(portfolio_metrics['sharpe_ratio'], 2)
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
