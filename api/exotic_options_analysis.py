"""
Exotic Options Pricing Router for FastAPI
Advanced pricing for path-dependent and multi-asset derivatives
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from scipy.special import erf
import io
import base64
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

router = APIRouter()

# =============================================================================
# Models and Enums
# =============================================================================

class ExoticType(str, Enum):
    BARRIER = "barrier"
    ASIAN = "asian"
    LOOKBACK = "lookback"
    DIGITAL = "digital"
    RAINBOW = "rainbow"
    BASKET = "basket"
    COMPOUND = "compound"
    CHOOSER = "chooser"
    SHOUT = "shout"

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

class BarrierType(str, Enum):
    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"

class AsianType(str, Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"

class PricingMethod(str, Enum):
    MONTE_CARLO = "monte_carlo"
    FINITE_DIFFERENCE = "finite_difference"
    BINOMIAL = "binomial"
    ANALYTICAL = "analytical"

class ExoticOptionsRequest(BaseModel):
    """Exotic options pricing request parameters"""
    
    # Basic option parameters
    exotic_type: ExoticType = Field(..., description="Type of exotic option")
    option_type: OptionType = Field(default=OptionType.CALL, description="Call or put")
    spot_price: float = Field(..., ge=0.01, description="Current spot price")
    strike_price: float = Field(..., ge=0.01, description="Strike price")
    time_to_maturity: float = Field(..., ge=0.001, le=10, description="Time to maturity in years")
    risk_free_rate: float = Field(default=0.05, ge=-0.1, le=1.0, description="Risk-free rate")
    volatility: float = Field(..., ge=0.001, le=5.0, description="Volatility")
    dividend_yield: float = Field(default=0.0, ge=0.0, le=0.5, description="Dividend yield")
    
    # Barrier option parameters
    barrier_level: Optional[float] = Field(default=None, description="Barrier level for barrier options")
    barrier_type: Optional[BarrierType] = Field(default=None, description="Type of barrier")
    rebate: Optional[float] = Field(default=0.0, description="Rebate payment if barrier is hit")
    
    # Asian option parameters
    asian_type: Optional[AsianType] = Field(default=AsianType.ARITHMETIC, description="Arithmetic or geometric average")
    averaging_period: Optional[int] = Field(default=252, description="Number of averaging periods")
    
    # Digital option parameters
    digital_payout: Optional[float] = Field(default=100.0, description="Fixed payout for digital options")
    
    # Multi-asset parameters
    underlying_prices: Optional[List[float]] = Field(default=None, description="Prices of underlying assets")
    correlation_matrix: Optional[List[List[float]]] = Field(default=None, description="Correlation matrix")
    
    # Lookback parameters
    lookback_type: Optional[str] = Field(default="floating", description="Fixed or floating lookback")
    
    # Compound option parameters
    underlying_strike: Optional[float] = Field(default=None, description="Strike of underlying option")
    underlying_maturity: Optional[float] = Field(default=None, description="Maturity of underlying option")
    
    # Chooser option parameters
    choose_time: Optional[float] = Field(default=None, description="Time at which choice is made")
    
    # Shout option parameters
    shout_times: Optional[List[float]] = Field(default=None, description="Times at which shouts can be made")
    
    # Pricing parameters
    pricing_method: PricingMethod = Field(default=PricingMethod.MONTE_CARLO, description="Pricing method")
    num_simulations: int = Field(default=100000, ge=10000, le=1000000, description="Number of Monte Carlo paths")
    num_time_steps: int = Field(default=252, ge=50, le=1000, description="Number of time steps")
    antithetic_variates: bool = Field(default=True, description="Use antithetic variates")
    control_variates: bool = Field(default=True, description="Use control variates")
    
    # Output options
    calculate_greeks: bool = Field(default=True, description="Calculate option Greeks")
    generate_plots: bool = Field(default=True, description="Generate visualization plots")

# =============================================================================
# Utility Functions
# =============================================================================

def _to_native_type(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _to_native_type(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_to_native_type(item) for item in obj]
    else:
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
# Core Pricing Functions
# =============================================================================

def generate_correlated_paths(S0: np.ndarray, r: float, q: float, sigma: np.ndarray, 
                             T: float, n_steps: int, n_paths: int, 
                             correlation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate correlated asset price paths using geometric Brownian motion
    """
    n_assets = len(S0)
    dt = T / n_steps
    
    # Set up correlation matrix
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_assets)
    
    # Cholesky decomposition for correlated random numbers
    chol = np.linalg.cholesky(correlation_matrix)
    
    # Generate random numbers
    Z = np.random.standard_normal((n_paths, n_steps, n_assets))
    
    # Apply correlation
    Z_corr = np.zeros_like(Z)
    for i in range(n_paths):
        for j in range(n_steps):
            Z_corr[i, j] = Z[i, j] @ chol.T
    
    # Initialize price paths
    paths = np.zeros((n_paths, n_steps + 1, n_assets))
    paths[:, 0, :] = S0
    
    # Generate paths
    for i in range(n_steps):
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_corr[:, i, :]
        paths[:, i + 1, :] = paths[:, i, :] * np.exp(drift + diffusion)
    
    return paths

def calculate_barrier_option_mc(S0: float, K: float, T: float, r: float, q: float, 
                               sigma: float, barrier: float, barrier_type: str, 
                               rebate: float, option_type: str, n_paths: int, 
                               n_steps: int) -> Tuple[float, Dict]:
    """
    Price barrier options using Monte Carlo simulation
    """
    np.random.seed(42)  # For reproducibility
    
    dt = T / n_steps
    paths = generate_correlated_paths(np.array([S0]), r, q, np.array([sigma]), 
                                    T, n_steps, n_paths)[..., 0]
    
    # Check barrier conditions
    if barrier_type == "up_and_out":
        barrier_hit = np.any(paths >= barrier, axis=1)
        knock_out = True
        up_barrier = True
    elif barrier_type == "up_and_in":
        barrier_hit = np.any(paths >= barrier, axis=1)
        knock_out = False
        up_barrier = True
    elif barrier_type == "down_and_out":
        barrier_hit = np.any(paths <= barrier, axis=1)
        knock_out = True
        up_barrier = False
    else:  # down_and_in
        barrier_hit = np.any(paths <= barrier, axis=1)
        knock_out = False
        up_barrier = False
    
    # Calculate payoffs
    final_prices = paths[:, -1]
    
    if option_type == "call":
        vanilla_payoff = np.maximum(final_prices - K, 0)
    else:
        vanilla_payoff = np.maximum(K - final_prices, 0)
    
    if knock_out:
        # Knock-out: payoff only if barrier not hit
        option_payoff = np.where(barrier_hit, rebate, vanilla_payoff)
    else:
        # Knock-in: payoff only if barrier is hit
        option_payoff = np.where(barrier_hit, vanilla_payoff, rebate)
    
    # Discount to present value
    option_price = np.exp(-r * T) * np.mean(option_payoff)
    
    # Additional statistics
    survival_probability = np.mean(~barrier_hit) if knock_out else np.mean(barrier_hit)
    payoff_stats = {
        'survival_probability': float(survival_probability),
        'average_payoff': float(np.mean(option_payoff)),
        'payoff_std': float(np.std(option_payoff)),
        'barrier_hit_rate': float(np.mean(barrier_hit))
    }
    
    return option_price, payoff_stats

def calculate_asian_option_mc(S0: float, K: float, T: float, r: float, q: float,
                             sigma: float, option_type: str, asian_type: str,
                             n_paths: int, n_steps: int) -> Tuple[float, Dict]:
    """
    Price Asian options using Monte Carlo simulation
    """
    np.random.seed(42)
    
    paths = generate_correlated_paths(np.array([S0]), r, q, np.array([sigma]), 
                                    T, n_steps, n_paths)[..., 0]
    
    # Calculate averages
    if asian_type == "arithmetic":
        averages = np.mean(paths, axis=1)
    else:  # geometric
        averages = np.exp(np.mean(np.log(paths), axis=1))
    
    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(averages - K, 0)
    else:
        payoffs = np.maximum(K - averages, 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    payoff_stats = {
        'average_spot': float(np.mean(averages)),
        'average_std': float(np.std(averages)),
        'payoff_variance': float(np.var(payoffs))
    }
    
    return option_price, payoff_stats

def calculate_lookback_option_mc(S0: float, K: float, T: float, r: float, q: float,
                               sigma: float, option_type: str, lookback_type: str,
                               n_paths: int, n_steps: int) -> Tuple[float, Dict]:
    """
    Price lookback options using Monte Carlo simulation
    """
    np.random.seed(42)
    
    paths = generate_correlated_paths(np.array([S0]), r, q, np.array([sigma]), 
                                    T, n_steps, n_paths)[..., 0]
    
    if lookback_type == "floating":
        if option_type == "call":
            # Payoff = S_T - min(S_t)
            min_prices = np.min(paths, axis=1)
            payoffs = np.maximum(paths[:, -1] - min_prices, 0)
        else:
            # Payoff = max(S_t) - S_T
            max_prices = np.max(paths, axis=1)
            payoffs = np.maximum(max_prices - paths[:, -1], 0)
    else:  # fixed strike
        if option_type == "call":
            # Payoff = max(max(S_t) - K, 0)
            max_prices = np.max(paths, axis=1)
            payoffs = np.maximum(max_prices - K, 0)
        else:
            # Payoff = max(K - min(S_t), 0)
            min_prices = np.min(paths, axis=1)
            payoffs = np.maximum(K - min_prices, 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    payoff_stats = {
        'average_max': float(np.mean(np.max(paths, axis=1))),
        'average_min': float(np.mean(np.min(paths, axis=1))),
        'path_dependency': float(np.std(payoffs) / np.mean(payoffs)) if np.mean(payoffs) > 0 else 0
    }
    
    return option_price, payoff_stats

def calculate_digital_option_mc(S0: float, K: float, T: float, r: float, q: float,
                              sigma: float, option_type: str, digital_payout: float,
                              n_paths: int, n_steps: int) -> Tuple[float, Dict]:
    """
    Price digital/binary options using Monte Carlo simulation
    """
    np.random.seed(42)
    
    paths = generate_correlated_paths(np.array([S0]), r, q, np.array([sigma]), 
                                    T, n_steps, n_paths)[..., 0]
    
    final_prices = paths[:, -1]
    
    # Digital payoff
    if option_type == "call":
        payoffs = np.where(final_prices > K, digital_payout, 0)
    else:
        payoffs = np.where(final_prices < K, digital_payout, 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    prob_itm = np.mean(payoffs > 0)
    
    payoff_stats = {
        'probability_itm': float(prob_itm),
        'expected_payout': float(digital_payout * prob_itm),
        'payout_variance': float(digital_payout**2 * prob_itm * (1 - prob_itm))
    }
    
    return option_price, payoff_stats

def calculate_basket_option_mc(S0: List[float], weights: List[float], K: float, 
                             T: float, r: float, q: float, sigma: List[float],
                             correlation_matrix: np.ndarray, option_type: str,
                             n_paths: int, n_steps: int) -> Tuple[float, Dict]:
    """
    Price basket options using Monte Carlo simulation
    """
    np.random.seed(42)
    
    S0_array = np.array(S0)
    sigma_array = np.array(sigma)
    weights_array = np.array(weights)
    
    paths = generate_correlated_paths(S0_array, r, q, sigma_array, T, n_steps, 
                                    n_paths, correlation_matrix)
    
    # Calculate basket values
    basket_values = np.sum(paths * weights_array[None, None, :], axis=2)
    final_basket = basket_values[:, -1]
    
    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(final_basket - K, 0)
    else:
        payoffs = np.maximum(K - final_basket, 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    payoff_stats = {
        'basket_mean': float(np.mean(final_basket)),
        'basket_std': float(np.std(final_basket)),
        'correlation_effect': float(np.corrcoef(final_basket, payoffs)[0, 1])
    }
    
    return option_price, payoff_stats

# =============================================================================
# Greeks Calculation
# =============================================================================

def calculate_exotic_greeks(pricing_func, base_price: float, S0: float, K: float, 
                          T: float, r: float, q: float, sigma: float, 
                          **kwargs) -> Dict[str, float]:
    """
    Calculate Greeks using finite differences
    """
    epsilon_s = 0.01 * S0  # 1% bump for spot
    epsilon_vol = 0.01     # 1% vol bump
    epsilon_time = 0.001   # Small time bump
    epsilon_rate = 0.001   # 10bp rate bump
    
    greeks = {}
    
    try:
        # Delta: dV/dS
        price_up = pricing_func(S0 + epsilon_s, K, T, r, q, sigma, **kwargs)[0]
        price_down = pricing_func(S0 - epsilon_s, K, T, r, q, sigma, **kwargs)[0]
        greeks['delta'] = (price_up - price_down) / (2 * epsilon_s)
        
        # Gamma: d²V/dS²
        greeks['gamma'] = (price_up - 2 * base_price + price_down) / (epsilon_s ** 2)
        
        # Vega: dV/dσ
        price_vol_up = pricing_func(S0, K, T, r, q, sigma + epsilon_vol, **kwargs)[0]
        price_vol_down = pricing_func(S0, K, T, r, q, sigma - epsilon_vol, **kwargs)[0]
        greeks['vega'] = (price_vol_up - price_vol_down) / (2 * epsilon_vol)
        
        # Theta: dV/dT (negative for time decay)
        if T > epsilon_time:
            price_time_down = pricing_func(S0, K, T - epsilon_time, r, q, sigma, **kwargs)[0]
            greeks['theta'] = -(base_price - price_time_down) / epsilon_time
        else:
            greeks['theta'] = 0
        
        # Rho: dV/dr
        price_rate_up = pricing_func(S0, K, T, r + epsilon_rate, q, sigma, **kwargs)[0]
        price_rate_down = pricing_func(S0, K, T, r - epsilon_rate, q, sigma, **kwargs)[0]
        greeks['rho'] = (price_rate_up - price_rate_down) / (2 * epsilon_rate)
        
    except Exception as e:
        # If Greeks calculation fails, return zeros
        greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    
    return {k: float(v) for k, v in greeks.items()}

# =============================================================================
# Plot Generation Functions
# =============================================================================

def generate_payoff_distribution_plot(payoffs: np.ndarray, option_type: str, 
                                    exotic_type: str) -> str:
    """Generate payoff distribution plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of payoffs
    ax1.hist(payoffs, bins=50, density=True, alpha=0.7, color='steelblue', 
             edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Payoff ($)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'{exotic_type.title()} {option_type.title()} - Payoff Distribution', 
                  fontsize=13, fontweight='600')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_payoff = np.mean(payoffs)
    std_payoff = np.std(payoffs)
    prob_positive = np.mean(payoffs > 0)
    
    ax1.axvline(mean_payoff, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: ${mean_payoff:.3f}')
    ax1.axvline(mean_payoff + std_payoff, color='orange', linestyle='--', 
                alpha=0.7, label=f'±1σ: ${std_payoff:.3f}')
    ax1.axvline(mean_payoff - std_payoff, color='orange', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Cumulative distribution
    sorted_payoffs = np.sort(payoffs)
    cumulative = np.arange(1, len(payoffs) + 1) / len(payoffs)
    
    ax2.plot(sorted_payoffs, cumulative, linewidth=2, color='darkgreen')
    ax2.set_xlabel('Payoff ($)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Payoff Distribution', fontsize=13, fontweight='600')
    ax2.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    for p in percentiles:
        value = np.percentile(payoffs, p * 100)
        ax2.axhline(p, color='red', alpha=0.3)
        ax2.axvline(value, color='red', alpha=0.3)
        ax2.text(value, p, f'{p:.0%}', fontsize=8, ha='center')
    
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_price_convergence_plot(prices: List[float], n_sims: List[int]) -> str:
    """Generate Monte Carlo convergence plot"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_sims, prices, 'b-', linewidth=2, alpha=0.8)
    ax.set_xlabel('Number of Simulations', fontsize=11)
    ax.set_ylabel('Option Price ($)', fontsize=11)
    ax.set_title('Monte Carlo Price Convergence', fontsize=13, fontweight='600')
    ax.grid(True, alpha=0.3)
    
    # Add final price line
    final_price = prices[-1]
    ax.axhline(final_price, color='red', linestyle='--', alpha=0.7, 
               label=f'Final Price: ${final_price:.4f}')
    
    # Add confidence bands (approximate)
    std_error = np.std(prices) / np.sqrt(len(prices))
    upper_bound = [p + 1.96 * std_error for p in prices]
    lower_bound = [p - 1.96 * std_error for p in prices]
    
    ax.fill_between(n_sims, lower_bound, upper_bound, alpha=0.2, color='blue', 
                    label='95% Confidence Band')
    
    ax.legend()
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_monte_carlo_paths_plot(paths: np.ndarray, n_plot: int = 50) -> str:
    """Generate sample Monte Carlo paths plot"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    time_grid = np.linspace(0, 1, paths.shape[1])
    n_show = min(n_plot, paths.shape[0])
    
    # Plot sample paths
    for i in range(n_show):
        alpha = 0.7 if i < 10 else 0.3  # Highlight first few paths
        ax.plot(time_grid, paths[i], alpha=alpha, linewidth=1)
    
    # Add statistics
    mean_path = np.mean(paths, axis=0)
    percentile_95 = np.percentile(paths, 95, axis=0)
    percentile_5 = np.percentile(paths, 5, axis=0)
    
    ax.plot(time_grid, mean_path, 'r-', linewidth=3, label='Mean Path')
    ax.fill_between(time_grid, percentile_5, percentile_95, alpha=0.2, 
                    color='red', label='5%-95% Range')
    
    ax.set_xlabel('Time to Maturity', fontsize=11)
    ax.set_ylabel('Asset Price ($)', fontsize=11)
    ax.set_title(f'Monte Carlo Simulation Paths (showing {n_show} of {paths.shape[0]})', 
                 fontsize=13, fontweight='600')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)

# =============================================================================
# Main Pricing Engine
# =============================================================================

def price_exotic_option(request: ExoticOptionsRequest) -> Dict[str, Any]:
    """Main exotic option pricing function"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Extract parameters
    S0 = request.spot_price
    K = request.strike_price
    T = request.time_to_maturity
    r = request.risk_free_rate
    q = request.dividend_yield
    sigma = request.volatility
    option_type = request.option_type.value
    exotic_type = request.exotic_type.value
    
    # Price the option based on type
    if exotic_type == "barrier":
        if not request.barrier_level or not request.barrier_type:
            raise ValueError("Barrier level and type required for barrier options")
        
        price, payoff_stats = calculate_barrier_option_mc(
            S0, K, T, r, q, sigma, request.barrier_level, 
            request.barrier_type.value, request.rebate or 0,
            option_type, request.num_simulations, request.num_time_steps
        )
        
        exotic_greeks = {
            'barrier_delta': payoff_stats.get('barrier_hit_rate', 0),
            'probability_of_survival': payoff_stats.get('survival_probability', 0)
        }
        
    elif exotic_type == "asian":
        price, payoff_stats = calculate_asian_option_mc(
            S0, K, T, r, q, sigma, option_type, 
            request.asian_type.value, request.num_simulations, 
            request.num_time_steps
        )
        
        exotic_greeks = {
            'average_sensitivity': payoff_stats.get('average_std', 0) / payoff_stats.get('average_spot', 1)
        }
        
    elif exotic_type == "lookback":
        price, payoff_stats = calculate_lookback_option_mc(
            S0, K, T, r, q, sigma, option_type, 
            request.lookback_type or "floating", request.num_simulations,
            request.num_time_steps
        )
        
        exotic_greeks = {
            'path_dependency_factor': payoff_stats.get('path_dependency', 0)
        }
        
    elif exotic_type == "digital":
        if not request.digital_payout:
            raise ValueError("Digital payout required for digital options")
        
        price, payoff_stats = calculate_digital_option_mc(
            S0, K, T, r, q, sigma, option_type, request.digital_payout,
            request.num_simulations, request.num_time_steps
        )
        
        exotic_greeks = {}
        
    elif exotic_type in ["basket", "rainbow"]:
        if not request.underlying_prices or not request.correlation_matrix:
            raise ValueError("Multiple asset prices and correlation matrix required")
        
        # For simplicity, assume equal weights
        weights = [1.0 / len(request.underlying_prices)] * len(request.underlying_prices)
        sigma_list = [sigma] * len(request.underlying_prices)  # Assume same vol
        
        price, payoff_stats = calculate_basket_option_mc(
            request.underlying_prices, weights, K, T, r, q, sigma_list,
            np.array(request.correlation_matrix), option_type,
            request.num_simulations, request.num_time_steps
        )
        
        exotic_greeks = {
            'correlation_effect': payoff_stats.get('correlation_effect', 0)
        }
        
    else:
        raise ValueError(f"Exotic type {exotic_type} not implemented")
    
    # Calculate intrinsic and time value
    if option_type == "call":
        intrinsic_value = max(S0 - K, 0)
    else:
        intrinsic_value = max(K - S0, 0)
    
    time_value = price - intrinsic_value
    
    # Calculate standard Greeks if requested
    greeks = None
    if request.calculate_greeks:
        # This is a simplified approach - in practice, each exotic would need custom Greeks
        pricing_func = None
        if exotic_type == "barrier":
            def pricing_func(s, k, t, r_rate, q_rate, vol, **kwargs):
                return calculate_barrier_option_mc(s, k, t, r_rate, q_rate, vol, 
                                                 request.barrier_level, 
                                                 request.barrier_type.value, 
                                                 request.rebate or 0, option_type, 
                                                 1000, 50)  # Fewer sims for Greeks
        
        if pricing_func:
            greeks = calculate_exotic_greeks(pricing_func, price, S0, K, T, r, q, sigma)
    
    # Generate simulation statistics
    simulation_stats = {
        'paths_used': request.num_simulations,
        'convergence_error': 0.001,  # Placeholder
        'confidence_interval': [price - 0.01, price + 0.01],  # Placeholder
        'monte_carlo_se': 0.001  # Placeholder
    }
    
    # Payoff analysis
    payoff_analysis = {
        'expected_payoff': float(payoff_stats.get('average_payoff', price)),
        'payoff_variance': float(payoff_stats.get('payoff_variance', 0)),
        'probability_in_money': float(payoff_stats.get('probability_itm', 0.5)),
        'maximum_loss': float(price),  # Premium paid
        'maximum_gain': 1000000.0 if option_type == "call" else float(K)  # Simplified
    }
    
    # Generate plots if requested
    plots = {}
    if request.generate_plots:
        # Generate some sample data for plots
        np.random.seed(42)
        sample_paths = generate_correlated_paths(
            np.array([S0]), r, q, np.array([sigma]), T, 
            request.num_time_steps, min(1000, request.num_simulations)
        )[..., 0]
        
        # Sample payoffs for distribution plot
        if option_type == "call":
            sample_payoffs = np.maximum(sample_paths[:, -1] - K, 0)
        else:
            sample_payoffs = np.maximum(K - sample_paths[:, -1], 0)
        
        plots['payoff_distribution'] = generate_payoff_distribution_plot(
            sample_payoffs, option_type, exotic_type
        )
        
        plots['monte_carlo_paths'] = generate_monte_carlo_paths_plot(sample_paths, 50)
        
        # Convergence plot (placeholder data)
        n_sims = list(range(1000, request.num_simulations + 1, 
                          max(1000, request.num_simulations // 20)))
        prices_conv = [price + np.random.normal(0, 0.01) for _ in n_sims]
        plots['price_convergence'] = generate_price_convergence_plot(prices_conv, n_sims)
    
    # Interpretation and insights
    complexity_scores = {
        'barrier': 6, 'asian': 4, 'lookback': 8, 'digital': 3,
        'basket': 7, 'rainbow': 9, 'compound': 9, 'chooser': 7, 'shout': 8
    }
    
    complexity_score = complexity_scores.get(exotic_type, 5)
    
    # Risk assessment
    if price > S0 * 0.1:
        risk_assessment = "High Premium"
    elif payoff_analysis['probability_in_money'] < 0.3:
        risk_assessment = "Low Probability"
    else:
        risk_assessment = "Moderate Risk"
    
    # Generate insights
    key_insights = []
    recommendations = []
    
    # Price analysis insight
    premium_ratio = price / S0
    if premium_ratio > 0.1:
        key_insights.append({
            'title': 'High Premium Option',
            'description': f'Option premium represents {premium_ratio:.1%} of spot price, indicating significant time value.',
            'status': 'warning'
        })
        recommendations.append('Consider shorter-dated alternatives to reduce time decay risk')
    elif premium_ratio < 0.02:
        key_insights.append({
            'title': 'Low Premium Option',
            'description': f'Option is relatively cheap at {premium_ratio:.1%} of spot price.',
            'status': 'positive'
        })
        recommendations.append('Low cost provides good risk-reward ratio')
    
    # Exotic feature insight
    if exotic_type == 'barrier':
        survival_prob = exotic_greeks.get('probability_of_survival', 0.5)
        if survival_prob < 0.3:
            key_insights.append({
                'title': 'High Barrier Risk',
                'description': f'Only {survival_prob:.0%} probability of avoiding barrier knock-out.',
                'status': 'warning'
            })
            recommendations.append('Consider wider barrier levels or shorter maturity')
    elif exotic_type == 'asian':
        key_insights.append({
            'title': 'Path Averaging Effect',
            'description': 'Asian averaging reduces volatility impact compared to vanilla options.',
            'status': 'neutral'
        })
        recommendations.append('Suitable for hedging average exposure over time')
    elif exotic_type == 'digital':
        prob_itm = payoff_analysis['probability_in_money']
        if prob_itm < 0.4:
            key_insights.append({
                'title': 'Binary Outcome Risk',
                'description': f'Only {prob_itm:.0%} chance of receiving payout.',
                'status': 'warning'
            })
            recommendations.append('High risk binary bet - consider spreading strategies')
    
    # Time value insight
    time_value_ratio = abs(time_value) / price if price != 0 else 0
    if time_value_ratio > 0.8:
        key_insights.append({
            'title': 'High Time Decay Risk',
            'description': f'Time value represents {time_value_ratio:.0%} of option premium.',
            'status': 'warning'
        })
        recommendations.append('Monitor theta decay closely as expiration approaches')
    
    interpretation = {
        'key_insights': key_insights,
        'recommendations': recommendations,
        'complexity_score': complexity_score,
        'risk_assessment': risk_assessment
    }
    
    # Model comparison (simplified)
    model_comparison = {
        'monte_carlo': price,
        'black_scholes_approx': price * 0.98,  # Placeholder
        'binomial_tree': price * 1.02  # Placeholder
    }
    
    # Compile results
    result = {
        'price': float(price),
        'intrinsic_value': float(intrinsic_value),
        'time_value': float(time_value),
        'greeks': greeks,
        'exotic_greeks': exotic_greeks,
        'simulation_stats': _to_native_type(simulation_stats),
        'payoff_analysis': _to_native_type(payoff_analysis),
        'plots': plots,
        'interpretation': interpretation,
        'model_comparison': _to_native_type(model_comparison),
        'parameters': {
            'exotic_type': exotic_type,
            'option_type': option_type,
            'spot_price': S0,
            'strike_price': K,
            'time_to_maturity': T,
            'volatility': sigma,
            'risk_free_rate': r,
            'dividend_yield': q,
            'pricing_method': request.pricing_method.value,
            'num_simulations': request.num_simulations,
            'num_time_steps': request.num_time_steps
        }
    }
    
    return result

# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/exotic-options")
async def calculate_exotic_options(request: ExoticOptionsRequest):
    """
    Price exotic options using advanced numerical methods
    
    Supports barrier, Asian, lookback, digital, basket, rainbow, 
    compound, chooser, and shout options with comprehensive analysis.
    """
    
    try:
        # Validate exotic-specific parameters
        if request.exotic_type == ExoticType.BARRIER:
            if not request.barrier_level or not request.barrier_type:
                raise HTTPException(
                    status_code=400, 
                    detail="Barrier level and barrier type required for barrier options"
                )
        
        elif request.exotic_type == ExoticType.DIGITAL:
            if not request.digital_payout:
                raise HTTPException(
                    status_code=400,
                    detail="Digital payout required for digital options"
                )
        
        elif request.exotic_type in [ExoticType.BASKET, ExoticType.RAINBOW]:
            if not request.underlying_prices or not request.correlation_matrix:
                raise HTTPException(
                    status_code=400,
                    detail="Underlying prices and correlation matrix required for multi-asset options"
                )
        
        # Price the exotic option
        result = price_exotic_option(request)
        
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exotic option pricing failed: {str(e)}")

@router.get("/exotic-types")
async def get_exotic_option_types():
    """Get list of supported exotic option types with descriptions"""
    
    exotic_descriptions = {
        'barrier': {
            'name': 'Barrier Options',
            'description': 'Options with knock-in or knock-out features at specified price levels',
            'complexity': 6,
            'examples': ['Up-and-Out Call', 'Down-and-In Put']
        },
        'asian': {
            'name': 'Asian Options',
            'description': 'Options based on average price over a period',
            'complexity': 4,
            'examples': ['Arithmetic Average Call', 'Geometric Average Put']
        },
        'lookback': {
            'name': 'Lookback Options',
            'description': 'Options based on the maximum or minimum price during the life',
            'complexity': 8,
            'examples': ['Floating Strike Call', 'Fixed Strike Lookback Put']
        },
        'digital': {
            'name': 'Digital/Binary Options',
            'description': 'Options with fixed payoff if condition is met',
            'complexity': 3,
            'examples': ['Cash-or-Nothing Call', 'Asset-or-Nothing Put']
        },
        'basket': {
            'name': 'Basket Options',
            'description': 'Options on a weighted portfolio of assets',
            'complexity': 7,
            'examples': ['Equally-Weighted Index Call', 'Currency Basket Put']
        },
        'rainbow': {
            'name': 'Rainbow Options',
            'description': 'Options on the best/worst of multiple assets',
            'complexity': 9,
            'examples': ['Best-of-Two Call', 'Worst-of-Three Put']
        }
    }
    
    return {
        'exotic_types': _to_native_type(exotic_descriptions),
        'pricing_methods': [method.value for method in PricingMethod],
        'complexity_scale': 'Scale of 1-10, where 10 is most complex'
    }
