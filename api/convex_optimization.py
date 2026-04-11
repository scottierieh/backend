"""
Convex Optimization: Portfolio Theory Router for FastAPI
Modern Portfolio Theory implementation with convex optimization for asset allocation
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
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
import io
import base64
import warnings

# Try to import cvxpy, fallback to scipy if not available
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not available, using scipy optimization instead")

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

router = APIRouter()


class OptimizationObjective(str, Enum):
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    TARGET_RETURN = "target_return"
    RISK_PARITY = "risk_parity"


class Asset(BaseModel):
    """Individual asset definition"""
    name: str = Field(description="Asset name")
    expected_return: float = Field(ge=0.0, le=1.0, description="Expected return (as decimal)")
    volatility: float = Field(gt=0.0, le=2.0, description="Volatility (as decimal)")


class ConvexOptimizationRequest(BaseModel):
    """Portfolio optimization request parameters"""
    
    # Assets
    assets: List[Asset] = Field(
        ...,
        min_items=2,
        max_items=20,
        description="List of assets with returns and volatilities"
    )
    correlation_matrix: List[List[float]] = Field(
        ...,
        description="Correlation matrix between assets"
    )
    
    # Optimization parameters
    target_return: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Target return for optimization (as decimal)"
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.TARGET_RETURN,
        description="Optimization objective"
    )
    risk_free_rate: float = Field(
        default=0.02,
        ge=0.0,
        le=0.2,
        description="Risk-free rate for Sharpe ratio calculation"
    )
    
    # Constraints
    max_weight: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Maximum weight for any single asset"
    )
    min_weight: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Minimum weight for any single asset"
    )
    allow_short_selling: bool = Field(
        default=False,
        description="Allow negative weights (short selling)"
    )
    
    # Analysis options
    generate_efficient_frontier: bool = Field(
        default=True,
        description="Generate efficient frontier plot"
    )
    num_frontier_points: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of points on efficient frontier"
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
# Portfolio Optimization Functions
# =============================================================================

def validate_correlation_matrix(corr_matrix: np.ndarray, n_assets: int) -> bool:
    """Validate correlation matrix properties"""
    if corr_matrix.shape != (n_assets, n_assets):
        return False
    
    # Check symmetry
    if not np.allclose(corr_matrix, corr_matrix.T):
        return False
    
    # Check diagonal elements are 1
    if not np.allclose(np.diag(corr_matrix), 1.0):
        return False
    
    # Check values are in [-1, 1]
    if not np.all((-1 <= corr_matrix) & (corr_matrix <= 1)):
        return False
    
    # Check positive semi-definite
    eigenvalues = np.linalg.eigvals(corr_matrix)
    if not np.all(eigenvalues >= -1e-8):  # Allow small numerical errors
        return False
    
    return True


def calculate_covariance_matrix(returns: np.ndarray, volatilities: np.ndarray, 
                                correlation_matrix: np.ndarray) -> np.ndarray:
    """Calculate covariance matrix from volatilities and correlations"""
    # Cov = D * Corr * D, where D is diagonal matrix of volatilities
    vol_matrix = np.outer(volatilities, volatilities)
    return vol_matrix * correlation_matrix


def optimize_portfolio_scipy(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                            objective: str, target_return: float = None,
                            risk_free_rate: float = 0.02, max_weight: float = 1.0,
                            min_weight: float = 0.0, allow_short: bool = False) -> Dict[str, Any]:
    """
    Portfolio optimization using scipy.optimize as fallback when cvxpy is not available
    """
    n_assets = len(expected_returns)
    
    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1
    
    # Add target return constraint if needed
    if objective == OptimizationObjective.TARGET_RETURN:
        constraints.append({
            'type': 'ineq', 
            'fun': lambda x: np.dot(x, expected_returns) - target_return
        })
    
    # Bounds
    if allow_short:
        bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
    else:
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    
    try:
        if objective == OptimizationObjective.MAXIMIZE_SHARPE:
            # Maximize Sharpe ratio by minimizing negative Sharpe
            def neg_sharpe(weights):
                port_return = np.dot(weights, expected_returns)
                port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                if port_var <= 0:
                    return 1e6
                sharpe = (port_return - risk_free_rate) / np.sqrt(port_var)
                return -sharpe
            
            result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            # Minimize variance (default for all other objectives)
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            optimal_weights = np.maximum(optimal_weights, 0) if not allow_short else optimal_weights
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            port_return = np.dot(expected_returns, optimal_weights)
            port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
            port_volatility = np.sqrt(port_variance)
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
            
            return {
                'success': True,
                'optimal_weights': optimal_weights,
                'portfolio_return': _to_native_type(port_return),
                'portfolio_volatility': _to_native_type(port_volatility),
                'sharpe_ratio': _to_native_type(sharpe_ratio),
                'portfolio_variance': _to_native_type(port_variance)
            }
        else:
            return {'success': False, 'message': f'Scipy optimization failed: {result.message}'}
            
    except Exception as e:
        return {'success': False, 'message': f'Optimization error: {str(e)}'}


def optimize_portfolio_cvxpy(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                             objective: str, target_return: float = None,
                             risk_free_rate: float = 0.02, max_weight: float = 1.0,
                             min_weight: float = 0.0, allow_short: bool = False) -> Dict[str, Any]:
    """
    Portfolio optimization using CVXPY for convex optimization
    """
    if not CVXPY_AVAILABLE:
        return optimize_portfolio_scipy(expected_returns, cov_matrix, objective, 
                                       target_return, risk_free_rate, max_weight, 
                                       min_weight, allow_short)
    
    n_assets = len(expected_returns)
    
    # Decision variables
    w = cp.Variable(n_assets)
    
    # Portfolio return and variance
    portfolio_return = cp.sum(cp.multiply(expected_returns, w))
    portfolio_variance = cp.quad_form(w, cov_matrix)
    portfolio_risk = cp.sqrt(portfolio_variance)
    
    # Constraints
    constraints = [cp.sum(w) == 1]  # Weights sum to 1
    
    # Weight bounds
    if allow_short:
        constraints.append(w >= -max_weight)
    else:
        constraints.append(w >= min_weight)
    constraints.append(w <= max_weight)
    
    # Objective-specific formulation
    try:
        if objective == OptimizationObjective.TARGET_RETURN:
            # Minimize variance subject to target return
            constraints.append(portfolio_return >= target_return)
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
            
        elif objective == OptimizationObjective.MINIMIZE_VARIANCE:
            # Global minimum variance portfolio
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
            
        elif objective == OptimizationObjective.MAXIMIZE_SHARPE:
            # Maximize Sharpe ratio (use auxiliary variable method)
            # max (r - rf) / σ = max (r - rf) subject to σ = 1
            constraints.append(portfolio_risk == 1)
            problem = cp.Problem(cp.Maximize(portfolio_return - risk_free_rate), constraints)
            
        else:  # RISK_PARITY
            # Risk parity: each asset contributes equally to portfolio risk
            # Approximate using penalty method
            weights_init = np.ones(n_assets) / n_assets
            problem = cp.Problem(cp.Minimize(portfolio_variance + 
                                           0.1 * cp.norm(w - weights_init)**2), constraints)
        
        # Solve optimization problem
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = w.value
            
            # Handle numerical errors
            optimal_weights = np.maximum(optimal_weights, 0) if not allow_short else optimal_weights
            optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
            
            # Calculate portfolio metrics
            port_return = np.dot(expected_returns, optimal_weights)
            port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
            port_volatility = np.sqrt(port_variance)
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
            
            # Special handling for Sharpe maximization (need to rescale)
            if objective == OptimizationObjective.MAXIMIZE_SHARPE and optimal_weights is not None:
                optimal_weights = optimal_weights / np.sum(optimal_weights)
                port_return = np.dot(expected_returns, optimal_weights)
                port_variance = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
                port_volatility = np.sqrt(port_variance)
                sharpe_ratio = (port_return - risk_free_rate) / port_volatility
            
            return {
                'success': True,
                'optimal_weights': optimal_weights,
                'portfolio_return': _to_native_type(port_return),
                'portfolio_volatility': _to_native_type(port_volatility),
                'sharpe_ratio': _to_native_type(sharpe_ratio),
                'portfolio_variance': _to_native_type(port_variance)
            }
        else:
            # Fallback to scipy if cvxpy fails
            return optimize_portfolio_scipy(expected_returns, cov_matrix, objective, 
                                           target_return, risk_free_rate, max_weight, 
                                           min_weight, allow_short)
            
    except Exception as e:
        # Fallback to scipy if cvxpy fails
        return optimize_portfolio_scipy(expected_returns, cov_matrix, objective, 
                                       target_return, risk_free_rate, max_weight, 
                                       min_weight, allow_short)


def calculate_efficient_frontier(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                 risk_free_rate: float, num_points: int = 50,
                                 max_weight: float = 1.0, min_weight: float = 0.0,
                                 allow_short: bool = False) -> Dict[str, Any]:
    """Calculate efficient frontier"""
    
    # Calculate range of returns
    min_return = np.min(expected_returns) if not allow_short else np.min(expected_returns) - 0.05
    max_return = np.max(expected_returns) + 0.02
    target_returns = np.linspace(min_return, max_return, num_points)
    
    frontier_returns = []
    frontier_volatilities = []
    frontier_sharpes = []
    
    for target_ret in target_returns:
        result = optimize_portfolio_cvxpy(
            expected_returns, cov_matrix, OptimizationObjective.TARGET_RETURN,
            target_return=target_ret, risk_free_rate=risk_free_rate,
            max_weight=max_weight, min_weight=min_weight, allow_short=allow_short
        )
        
        if result['success']:
            frontier_returns.append(result['portfolio_return'])
            frontier_volatilities.append(result['portfolio_volatility'])
            frontier_sharpes.append(result['sharpe_ratio'])
        else:
            # If optimization fails, try to continue
            continue
    
    return {
        'returns': frontier_returns,
        'volatilities': frontier_volatilities,
        'sharpe_ratios': frontier_sharpes,
        'target_returns': target_returns[:len(frontier_returns)]
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def generate_allocation_plot(weights: np.ndarray, asset_names: List[str]) -> str:
    """Generate portfolio allocation pie chart"""
    
    # Filter out near-zero weights
    threshold = 0.005
    mask = weights > threshold
    filtered_weights = weights[mask]
    filtered_names = [asset_names[i] for i in range(len(asset_names)) if mask[i]]
    
    # Group small allocations
    small_weights_sum = np.sum(weights[~mask])
    if small_weights_sum > 0:
        filtered_weights = np.append(filtered_weights, small_weights_sum)
        filtered_names.append('Other')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_weights)))
    wedges, texts, autotexts = ax.pie(
        filtered_weights, 
        labels=filtered_names, 
        autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
        startangle=90,
        colors=colors,
        explode=[0.05 if w == max(filtered_weights) else 0 for w in filtered_weights]
    )
    
    # Beautify text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('600')
    
    ax.set_title('Optimal Portfolio Allocation', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_efficient_frontier_plot(frontier: Dict[str, Any], optimal_point: Dict[str, Any],
                                     individual_assets: Dict[str, Any], risk_free_rate: float) -> str:
    """Generate efficient frontier plot with individual assets"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot efficient frontier
    frontier_vols = frontier['volatilities']
    frontier_rets = frontier['returns']
    frontier_sharpes = frontier['sharpe_ratios']
    
    # Color by Sharpe ratio
    scatter = ax.scatter(
        [v * 100 for v in frontier_vols], 
        [r * 100 for r in frontier_rets],
        c=frontier_sharpes, 
        cmap='viridis', 
        s=30, 
        alpha=0.8,
        label='Efficient Frontier'
    )
    
    # Plot individual assets
    asset_vols = [v * 100 for v in individual_assets['volatilities']]
    asset_rets = [r * 100 for r in individual_assets['returns']]
    ax.scatter(asset_vols, asset_rets, c='red', s=100, alpha=0.7, 
               marker='o', edgecolor='darkred', linewidth=2, label='Individual Assets')
    
    # Annotate assets
    for i, name in enumerate(individual_assets['names']):
        ax.annotate(name, (asset_vols[i], asset_rets[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot optimal portfolio
    optimal_vol = optimal_point['portfolio_volatility'] * 100
    optimal_ret = optimal_point['portfolio_return'] * 100
    ax.scatter(optimal_vol, optimal_ret, c='gold', s=200, alpha=1.0, 
               marker='*', edgecolor='orange', linewidth=3, label='Optimal Portfolio')
    
    # Plot risk-free asset and capital allocation line
    if len(frontier_vols) > 0:
        max_sharpe_idx = np.argmax(frontier_sharpes)
        tangent_vol = frontier_vols[max_sharpe_idx] * 100
        tangent_ret = frontier_rets[max_sharpe_idx] * 100
        
        # Capital Allocation Line
        max_vol = max(max(asset_vols), max([v * 100 for v in frontier_vols])) * 1.1
        cal_vols = np.linspace(0, max_vol, 100)
        cal_rets = risk_free_rate * 100 + (tangent_ret - risk_free_rate * 100) * cal_vols / tangent_vol
        ax.plot(cal_vols, cal_rets, 'b--', alpha=0.6, linewidth=2, label='Capital Allocation Line')
        
        # Risk-free rate point
        ax.scatter(0, risk_free_rate * 100, c='blue', s=100, marker='s', 
                   label=f'Risk-free ({risk_free_rate*100:.1f}%)')
    
    # Formatting
    ax.set_xlabel('Portfolio Volatility (%)', fontsize=12, fontweight='600')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='600')
    ax.set_title('Efficient Frontier Analysis', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add colorbar for Sharpe ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=11, fontweight='600')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_asset_comparison_plot(assets: List[Asset], weights: np.ndarray) -> str:
    """Generate asset comparison plot showing return vs risk"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Return vs Risk scatter
    asset_names = [asset.name for asset in assets]
    returns = [asset.expected_return * 100 for asset in assets]
    volatilities = [asset.volatility * 100 for asset in assets]
    
    # Color by weight
    colors = plt.cm.Blues(weights / np.max(weights))
    
    scatter = ax1.scatter(volatilities, returns, c=weights, s=weights*1000+50, 
                         alpha=0.7, cmap='Blues', edgecolor='navy', linewidth=1.5)
    
    for i, name in enumerate(asset_names):
        ax1.annotate(name, (volatilities[i], returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
    
    ax1.set_xlabel('Volatility (%)', fontsize=11, fontweight='600')
    ax1.set_ylabel('Expected Return (%)', fontsize=11, fontweight='600')
    ax1.set_title('Assets: Return vs Risk', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Portfolio Weight', fontsize=10)
    
    # Right: Risk-Return efficiency
    efficiency = np.array(returns) / np.array(volatilities)
    bars = ax2.bar(range(len(asset_names)), efficiency, 
                   color=plt.cm.RdYlGn(efficiency / np.max(efficiency)),
                   alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add weight labels on bars
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{weight*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='600')
    
    ax2.set_xlabel('Assets', fontsize=11, fontweight='600')
    ax2.set_ylabel('Return/Risk Ratio', fontsize=11, fontweight='600')
    ax2.set_title('Asset Efficiency & Allocation', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(asset_names)))
    ax2.set_xticklabels(asset_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_risk_breakdown_plot(weights: np.ndarray, cov_matrix: np.ndarray, 
                                 asset_names: List[str]) -> str:
    """Generate risk contribution breakdown"""
    
    # Calculate marginal risk contributions
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    marginal_contrib = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)
    risk_contrib = weights * marginal_contrib
    risk_contrib_pct = risk_contrib / np.sum(risk_contrib) * 100
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Risk contribution by asset
    colors = plt.cm.Set3(np.linspace(0, 1, len(asset_names)))
    bars1 = ax1.bar(asset_names, risk_contrib_pct, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for bar, pct in zip(bars1, risk_contrib_pct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='600')
    
    ax1.set_xlabel('Assets', fontsize=11, fontweight='600')
    ax1.set_ylabel('Risk Contribution (%)', fontsize=11, fontweight='600')
    ax1.set_title('Portfolio Risk Breakdown', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Weight vs Risk contribution comparison
    x = np.arange(len(asset_names))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, weights * 100, width, label='Weight (%)', 
                    color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
    bars3 = ax2.bar(x + width/2, risk_contrib_pct, width, label='Risk Contrib (%)',
                    color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
    
    ax2.set_xlabel('Assets', fontsize=11, fontweight='600')
    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='600')
    ax2.set_title('Weight vs Risk Contribution', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(asset_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Analysis and Interpretation
# =============================================================================

def calculate_portfolio_metrics(weights: np.ndarray, expected_returns: np.ndarray,
                                cov_matrix: np.ndarray, risk_free_rate: float) -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics"""
    
    # Basic metrics
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Diversification metrics
    effective_num_assets = 1 / np.sum(weights**2)  # Inverse Herfindahl index
    max_weight = np.max(weights)
    weight_concentration = np.sum(weights**2)  # Herfindahl index
    
    # Risk metrics
    downside_deviation = portfolio_volatility * 0.8  # Approximation
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation
    
    # Utilization (how much of the available assets are actually used)
    significant_weights = np.sum(weights > 0.01)  # Assets with >1% allocation
    utilization = significant_weights / len(weights) * 100
    
    return {
        'portfolio_return': _to_native_type(portfolio_return),
        'portfolio_volatility': _to_native_type(portfolio_volatility),
        'portfolio_variance': _to_native_type(portfolio_variance),
        'sharpe_ratio': _to_native_type(sharpe_ratio),
        'sortino_ratio': _to_native_type(sortino_ratio),
        'effective_num_assets': _to_native_type(effective_num_assets),
        'max_weight': _to_native_type(max_weight),
        'weight_concentration': _to_native_type(weight_concentration),
        'utilization': _to_native_type(utilization),
        'num_assets': len(weights),
        'significant_assets': int(significant_weights)
    }


def generate_interpretation(optimization_result: Dict, metrics: Dict, 
                           assets: List[Asset], params: Dict) -> Dict[str, Any]:
    """Generate interpretation of optimization results"""
    
    key_insights = []
    
    # Portfolio performance insights
    portfolio_return = metrics['portfolio_return'] * 100
    portfolio_vol = metrics['portfolio_volatility'] * 100
    sharpe_ratio = metrics['sharpe_ratio']
    
    if sharpe_ratio > 1.0:
        key_insights.append({
            'title': 'Excellent Risk-Adjusted Returns',
            'description': f'Sharpe ratio of {sharpe_ratio:.2f} indicates strong risk-adjusted performance.',
            'status': 'positive'
        })
    elif sharpe_ratio > 0.5:
        key_insights.append({
            'title': 'Good Risk-Adjusted Returns',
            'description': f'Sharpe ratio of {sharpe_ratio:.2f} shows decent compensation for risk taken.',
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'Low Risk-Adjusted Returns',
            'description': f'Sharpe ratio of {sharpe_ratio:.2f} suggests limited compensation for portfolio risk.',
            'status': 'warning'
        })
    
    # Diversification insights
    effective_assets = metrics['effective_num_assets']
    utilization = metrics['utilization']
    
    if effective_assets < 2:
        key_insights.append({
            'title': 'Concentrated Portfolio',
            'description': f'Effective number of assets ({effective_assets:.1f}) indicates high concentration risk.',
            'status': 'warning'
        })
    elif effective_assets > len(assets) * 0.7:
        key_insights.append({
            'title': 'Well Diversified',
            'description': f'Effective number of assets ({effective_assets:.1f}) shows good diversification.',
            'status': 'positive'
        })
    
    # Asset selection insights
    selected_assets = [asset.name for i, asset in enumerate(assets) 
                      if optimization_result['optimal_weights'][i] > 0.01]
    
    if len(selected_assets) < len(assets) * 0.5:
        key_insights.append({
            'title': 'Selective Asset Allocation',
            'description': f'Optimizer selected {len(selected_assets)} of {len(assets)} assets, suggesting clear efficiency differences.',
            'status': 'neutral'
        })
    
    # Risk level assessment
    if portfolio_vol < 5:
        key_insights.append({
            'title': 'Conservative Risk Profile',
            'description': f'Portfolio volatility of {portfolio_vol:.1f}% indicates low-risk allocation.',
            'status': 'positive'
        })
    elif portfolio_vol > 15:
        key_insights.append({
            'title': 'Aggressive Risk Profile',
            'description': f'Portfolio volatility of {portfolio_vol:.1f}% indicates higher risk for higher returns.',
            'status': 'neutral'
        })
    
    # Generate recommendations
    recommendations = []
    
    if metrics['max_weight'] > 0.4:
        recommendations.append(f"Consider reducing maximum weight constraint as one asset ({metrics['max_weight']*100:.1f}%) dominates the portfolio.")
    
    if utilization < 50:
        recommendations.append("Many assets were excluded from optimal portfolio. Consider reviewing asset universe or relaxing constraints.")
    
    if sharpe_ratio < 0.3:
        recommendations.append("Low Sharpe ratio suggests reviewing expected returns or considering additional assets.")
    
    if effective_assets < 3:
        recommendations.append("Consider diversifying further to reduce concentration risk.")
    
    if not recommendations:
        recommendations.append("Portfolio optimization results appear balanced and well-diversified.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/convex-optimization")
async def optimize_portfolio(request: ConvexOptimizationRequest) -> Dict[str, Any]:
    """
    Portfolio optimization using Modern Portfolio Theory.
    
    Implements convex optimization to find optimal asset allocation:
    1. Mean-Variance Optimization (Markowitz)
    2. Risk Parity
    3. Maximum Sharpe Ratio
    4. Target Return with Minimum Risk
    
    Returns optimal weights, portfolio metrics, efficient frontier, and analysis.
    """
    try:
        # Extract asset data
        asset_names = [asset.name for asset in request.assets]
        expected_returns = np.array([asset.expected_return for asset in request.assets])
        volatilities = np.array([asset.volatility for asset in request.assets])
        correlation_matrix = np.array(request.correlation_matrix)
        
        # Validate inputs
        n_assets = len(request.assets)
        
        if len(correlation_matrix) != n_assets or any(len(row) != n_assets for row in correlation_matrix):
            raise ValueError("Correlation matrix dimensions don't match number of assets")
        
        if not validate_correlation_matrix(correlation_matrix, n_assets):
            raise ValueError("Invalid correlation matrix. Must be symmetric, positive semi-definite with 1s on diagonal")
        
        # Calculate covariance matrix
        cov_matrix = calculate_covariance_matrix(expected_returns, volatilities, correlation_matrix)
        
        # Optimize portfolio
        optimization_result = optimize_portfolio_cvxpy(
            expected_returns, cov_matrix, request.objective,
            target_return=request.target_return, risk_free_rate=request.risk_free_rate,
            max_weight=request.max_weight, min_weight=request.min_weight,
            allow_short=request.allow_short_selling
        )
        
        if not optimization_result['success']:
            raise ValueError(f"Optimization failed: {optimization_result.get('message', 'Unknown error')}")
        
        optimal_weights = optimization_result['optimal_weights']
        
        # Calculate comprehensive metrics
        metrics = calculate_portfolio_metrics(
            optimal_weights, expected_returns, cov_matrix, request.risk_free_rate
        )
        
        # Determine selected assets and create detailed breakdown
        threshold = 0.001
        asset_details = []
        asset_details_by_efficiency = []
        selected_assets = []
        
        for i, (asset, weight) in enumerate(zip(request.assets, optimal_weights)):
            selected = weight > threshold
            efficiency = asset.expected_return / asset.volatility
            
            detail = {
                'name': asset.name,
                'weight': _to_native_type(weight),
                'expected_return': asset.expected_return,
                'volatility': asset.volatility,
                'efficiency': _to_native_type(efficiency),
                'selected': selected
            }
            asset_details.append(detail)
            asset_details_by_efficiency.append(detail)
            
            if selected:
                selected_assets.append(asset.name)
        
        # Sort by efficiency
        asset_details_by_efficiency.sort(key=lambda x: x['efficiency'], reverse=True)
        
        # Calculate efficient frontier if requested
        frontier_data = None
        if request.generate_efficient_frontier:
            frontier_data = calculate_efficient_frontier(
                expected_returns, cov_matrix, request.risk_free_rate,
                request.num_frontier_points, request.max_weight, 
                request.min_weight, request.allow_short_selling
            )
        
        # Generate visualizations
        plots = {}
        
        # Portfolio allocation
        plots['allocation'] = generate_allocation_plot(optimal_weights, asset_names)
        
        # Efficient frontier
        if frontier_data:
            individual_assets = {
                'names': asset_names,
                'returns': expected_returns.tolist(),
                'volatilities': volatilities.tolist()
            }
            plots['efficient_frontier'] = generate_efficient_frontier_plot(
                frontier_data, optimization_result, individual_assets, request.risk_free_rate
            )
        
        # Asset comparison
        plots['asset_comparison'] = generate_asset_comparison_plot(request.assets, optimal_weights)
        
        # Risk breakdown
        plots['risk_breakdown'] = generate_risk_breakdown_plot(
            optimal_weights, cov_matrix, asset_names
        )
        
        # Generate interpretation
        interpretation = generate_interpretation(
            optimization_result, metrics, request.assets, {
                'objective': request.objective.value,
                'target_return': request.target_return
            }
        )
        
        return {
            'success': True,
            'portfolio_return': optimization_result['portfolio_return'],
            'portfolio_volatility': optimization_result['portfolio_volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio'],
            'optimal_weights': {name: _to_native_type(weight) 
                               for name, weight in zip(asset_names, optimal_weights)},
            'selected_assets': selected_assets,
            'asset_details': asset_details,
            'asset_details_by_efficiency': asset_details_by_efficiency,
            'utilization': metrics['utilization'],
            'portfolio': {
                'n_assets': n_assets,
                'target_return': request.target_return,
                'n_selected': len(selected_assets)
            },
            'plots': plots,
            'interpretation': interpretation,
            'frontier_data': frontier_data,
            'parameters': {
                'objective': request.objective.value,
                'target_return': request.target_return,
                'risk_free_rate': request.risk_free_rate,
                'max_weight': request.max_weight,
                'min_weight': request.min_weight,
                'allow_short_selling': request.allow_short_selling
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")
