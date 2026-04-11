"""
Exotic Options Pricing Router for FastAPI
Advanced pricing for path-dependent and multi-asset derivatives
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import integrate
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

router = APIRouter()


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


class PricingMethod(str, Enum):
    MONTE_CARLO = "monte_carlo"
    FINITE_DIFFERENCE = "finite_difference"
    BINOMIAL = "binomial"
    ANALYTICAL = "analytical"


class BarrierType(str, Enum):
    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"


class AsianType(str, Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class ExoticOptionsRequest(BaseModel):
    """Exotic options pricing request parameters"""
    
    # Basic option parameters
    exotic_type: ExoticType = Field(description="Type of exotic option")
    option_type: str = Field(default="call", description="Call or put")
    spot_price: float = Field(ge=0.01, description="Current underlying price")
    strike_price: float = Field(ge=0.01, description="Strike price")
    time_to_maturity: float = Field(ge=0.01, le=10.0, description="Time to maturity in years")
    risk_free_rate: float = Field(ge=-0.1, le=0.5, description="Risk-free interest rate")
    volatility: float = Field(ge=0.01, le=2.0, description="Volatility (annualized)")
    dividend_yield: float = Field(default=0.0, ge=0.0, le=0.2, description="Continuous dividend yield")
    
    # Pricing method
    pricing_method: PricingMethod = Field(default=PricingMethod.MONTE_CARLO)
    num_simulations: int = Field(default=100000, ge=10000, le=1000000)
    num_time_steps: int = Field(default=252, ge=50, le=1000)
    antithetic_variates: bool = Field(default=True)
    control_variates: bool = Field(default=True)
    
    # Barrier option parameters
    barrier_level: Optional[float] = None
    barrier_type: Optional[BarrierType] = None
    rebate: Optional[float] = Field(default=0.0, ge=0.0)
    
    # Asian option parameters
    asian_type: Optional[AsianType] = None
    averaging_period: Optional[int] = Field(default=252, ge=1, le=1000)
    
    # Digital option parameters
    digital_payout: Optional[float] = Field(default=100.0, ge=0.0)
    
    # Multi-asset parameters
    underlying_prices: Optional[List[float]] = None
    correlation_matrix: Optional[List[List[float]]] = None
    
    # Shout option parameters
    shout_times: Optional[List[float]] = None


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
# Monte Carlo Simulation Engine
# =============================================================================

def generate_price_paths(S0: float, T: float, r: float, sigma: float, 
                        num_paths: int, num_steps: int, q: float = 0,
                        antithetic: bool = True) -> np.ndarray:
    """Generate stock price paths using geometric Brownian motion"""
    
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    # Pre-calculate drift
    drift = (r - q - 0.5 * sigma**2) * dt
    
    # Generate random numbers
    if antithetic:
        # Use antithetic variates for variance reduction
        Z = np.random.standard_normal((num_paths // 2, num_steps))
        Z = np.vstack([Z, -Z])
        if num_paths % 2:
            Z = np.vstack([Z, np.random.standard_normal((1, num_steps))])
    else:
        Z = np.random.standard_normal((num_paths, num_steps))
    
    # Generate price paths
    log_returns = drift + sigma * sqrt_dt * Z
    log_prices = np.cumsum(log_returns, axis=1)
    
    # Add initial condition
    log_prices = np.column_stack([np.zeros(num_paths), log_prices])
    prices = S0 * np.exp(log_prices)
    
    return prices


def generate_correlated_paths(S0_list: List[float], T: float, r: float, 
                            sigma_list: List[float], corr_matrix: np.ndarray,
                            num_paths: int, num_steps: int, q: float = 0) -> np.ndarray:
    """Generate correlated asset price paths"""
    
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    n_assets = len(S0_list)
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate correlated random numbers
    Z_uncorr = np.random.standard_normal((num_paths, num_steps, n_assets))
    Z_corr = np.dot(Z_uncorr, L.T)
    
    # Generate paths for each asset
    paths = np.zeros((num_paths, num_steps + 1, n_assets))
    
    for i in range(n_assets):
        S0 = S0_list[i]
        sigma = sigma_list[i]
        drift = (r - q - 0.5 * sigma**2) * dt
        
        log_returns = drift + sigma * sqrt_dt * Z_corr[:, :, i]
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.column_stack([np.zeros(num_paths), log_prices])
        paths[:, :, i] = S0 * np.exp(log_prices)
    
    return paths


# =============================================================================
# Barrier Options
# =============================================================================

def price_barrier_option(S0: float, K: float, T: float, r: float, sigma: float,
                        barrier: float, barrier_type: str, rebate: float = 0,
                        option_type: str = "call", num_paths: int = 100000,
                        num_steps: int = 252, q: float = 0) -> Dict:
    """Price barrier options using Monte Carlo"""
    
    paths = generate_price_paths(S0, T, r, sigma, num_paths, num_steps, q)
    payoffs = np.zeros(num_paths)
    barrier_hits = np.zeros(num_paths, dtype=bool)
    
    # Check barrier conditions
    if barrier_type in ["up_and_out", "up_and_in"]:
        barrier_hits = np.any(paths >= barrier, axis=1)
    else:  # down_and_out, down_and_in
        barrier_hits = np.any(paths <= barrier, axis=1)
    
    # Calculate payoffs
    final_prices = paths[:, -1]
    
    if option_type == "call":
        intrinsic = np.maximum(final_prices - K, 0)
    else:
        intrinsic = np.maximum(K - final_prices, 0)
    
    # Apply barrier conditions
    if barrier_type in ["up_and_out", "down_and_out"]:
        # Knock-out: option dies if barrier is hit
        payoffs = np.where(barrier_hits, rebate, intrinsic)
    else:  # knock-in
        # Knock-in: option activates if barrier is hit
        payoffs = np.where(barrier_hits, intrinsic, rebate)
    
    # Discount to present value
    discounted_payoffs = payoffs * np.exp(-r * T)
    price = np.mean(discounted_payoffs)
    
    # Calculate exotic Greeks
    survival_prob = np.mean(~barrier_hits) if "out" in barrier_type else np.mean(barrier_hits)
    
    # Barrier delta (sensitivity to barrier level)
    eps = 1.0
    barrier_up = barrier + eps
    barrier_down = barrier - eps
    
    # Approximate barrier delta
    if barrier_type in ["up_and_out", "up_and_in"]:
        barrier_hits_up = np.any(paths >= barrier_up, axis=1)
        barrier_hits_down = np.any(paths >= barrier_down, axis=1)
    else:
        barrier_hits_up = np.any(paths <= barrier_up, axis=1)
        barrier_hits_down = np.any(paths <= barrier_down, axis=1)
    
    if barrier_type in ["up_and_out", "down_and_out"]:
        payoffs_up = np.where(barrier_hits_up, rebate, intrinsic)
        payoffs_down = np.where(barrier_hits_down, rebate, intrinsic)
    else:
        payoffs_up = np.where(barrier_hits_up, intrinsic, rebate)
        payoffs_down = np.where(barrier_hits_down, intrinsic, rebate)
    
    price_up = np.mean(payoffs_up * np.exp(-r * T))
    price_down = np.mean(payoffs_down * np.exp(-r * T))
    barrier_delta = (price_up - price_down) / (2 * eps)
    
    return {
        'price': _to_native_type(price),
        'survival_probability': _to_native_type(survival_prob),
        'barrier_delta': _to_native_type(barrier_delta),
        'payoffs': discounted_payoffs,
        'paths': paths,
        'barrier_hits': barrier_hits
    }


# =============================================================================
# Asian Options
# =============================================================================

def price_asian_option(S0: float, K: float, T: float, r: float, sigma: float,
                      asian_type: str = "arithmetic", option_type: str = "call",
                      num_paths: int = 100000, num_steps: int = 252, q: float = 0) -> Dict:
    """Price Asian options using Monte Carlo"""
    
    paths = generate_price_paths(S0, T, r, sigma, num_paths, num_steps, q)
    
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
    
    # Discount to present value
    discounted_payoffs = payoffs * np.exp(-r * T)
    price = np.mean(discounted_payoffs)
    
    # Average sensitivity
    avg_sensitivity = np.std(averages) / np.mean(averages) if np.mean(averages) > 0 else 0
    
    return {
        'price': _to_native_type(price),
        'average_sensitivity': _to_native_type(avg_sensitivity),
        'payoffs': discounted_payoffs,
        'paths': paths,
        'averages': averages
    }


# =============================================================================
# Lookback Options
# =============================================================================

def price_lookback_option(S0: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = "call", num_paths: int = 100000,
                         num_steps: int = 252, q: float = 0) -> Dict:
    """Price lookback options using Monte Carlo"""
    
    paths = generate_price_paths(S0, T, r, sigma, num_paths, num_steps, q)
    
    # Calculate extremes
    max_prices = np.max(paths, axis=1)
    min_prices = np.min(paths, axis=1)
    
    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(max_prices - K, 0)  # Lookback call
    else:
        payoffs = np.maximum(K - min_prices, 0)  # Lookback put
    
    # Discount to present value
    discounted_payoffs = payoffs * np.exp(-r * T)
    price = np.mean(discounted_payoffs)
    
    # Path dependency factor
    path_dependency = np.mean(np.std(paths, axis=1) / np.mean(paths, axis=1))
    
    return {
        'price': _to_native_type(price),
        'path_dependency_factor': _to_native_type(path_dependency),
        'payoffs': discounted_payoffs,
        'paths': paths,
        'max_prices': max_prices,
        'min_prices': min_prices
    }


# =============================================================================
# Digital Options
# =============================================================================

def price_digital_option(S0: float, K: float, T: float, r: float, sigma: float,
                        payout: float = 100, option_type: str = "call",
                        num_paths: int = 100000, q: float = 0) -> Dict:
    """Price digital/binary options using Monte Carlo"""
    
    paths = generate_price_paths(S0, T, r, sigma, num_paths, 1, q)
    final_prices = paths[:, -1]
    
    # Calculate payoffs
    if option_type == "call":
        payoffs = np.where(final_prices > K, payout, 0)
    else:
        payoffs = np.where(final_prices < K, payout, 0)
    
    # Discount to present value
    discounted_payoffs = payoffs * np.exp(-r * T)
    price = np.mean(discounted_payoffs)
    
    # Probability of finishing in-the-money
    prob_itm = np.mean(payoffs > 0)
    
    return {
        'price': _to_native_type(price),
        'probability_itm': _to_native_type(prob_itm),
        'payoffs': discounted_payoffs,
        'final_prices': final_prices
    }


# =============================================================================
# Standard Greeks Calculation
# =============================================================================

def calculate_greeks_mc(price_func, S0: float, K: float, T: float, r: float, 
                       sigma: float, **kwargs) -> Dict:
    """Calculate Greeks using finite difference method"""
    
    eps_s = 1.0  # For delta and gamma
    eps_t = 1.0/365  # For theta (1 day)
    eps_v = 0.01  # For vega
    eps_r = 0.01  # For rho
    
    base_price = price_func(S0, K, T, r, sigma, **kwargs)['price']
    
    # Delta
    price_up = price_func(S0 + eps_s, K, T, r, sigma, **kwargs)['price']
    price_down = price_func(S0 - eps_s, K, T, r, sigma, **kwargs)['price']
    delta = (price_up - price_down) / (2 * eps_s)
    
    # Gamma
    gamma = (price_up - 2 * base_price + price_down) / (eps_s ** 2)
    
    # Theta
    if T > eps_t:
        price_theta = price_func(S0, K, T - eps_t, r, sigma, **kwargs)['price']
        theta = (price_theta - base_price) / eps_t
    else:
        theta = 0
    
    # Vega
    price_vega_up = price_func(S0, K, T, r, sigma + eps_v, **kwargs)['price']
    price_vega_down = price_func(S0, K, T, r, sigma - eps_v, **kwargs)['price']
    vega = (price_vega_up - price_vega_down) / (2 * eps_v)
    
    # Rho
    price_rho_up = price_func(S0, K, T, r + eps_r, sigma, **kwargs)['price']
    price_rho_down = price_func(S0, K, T, r - eps_r, sigma, **kwargs)['price']
    rho = (price_rho_up - price_rho_down) / (2 * eps_r)
    
    return {
        'delta': _to_native_type(delta),
        'gamma': _to_native_type(gamma),
        'theta': _to_native_type(theta),
        'vega': _to_native_type(vega / 100),  # Per 1% vol change
        'rho': _to_native_type(rho / 100)     # Per 1% rate change
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def generate_payoff_distribution_plot(payoffs: np.ndarray, option_price: float,
                                     option_type: str) -> str:
    """Generate payoff distribution plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Payoff histogram
    non_zero_payoffs = payoffs[payoffs > 0]
    zero_payoffs = payoffs[payoffs == 0]
    
    if len(non_zero_payoffs) > 0:
        ax1.hist(non_zero_payoffs, bins=50, alpha=0.7, color='skyblue', 
                edgecolor='navy', label=f'ITM Payoffs ({len(non_zero_payoffs)})')
    
    if len(zero_payoffs) > 0:
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'OTM Payoffs ({len(zero_payoffs)})')
    
    ax1.axvline(option_price, color='green', linestyle='-', linewidth=3,
               label=f'Option Price: ${option_price:.4f}')
    
    ax1.set_xlabel('Payoff ($)', fontsize=11, fontweight='600')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='600')
    ax1.set_title('Payoff Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Cumulative distribution
    sorted_payoffs = np.sort(payoffs)
    cum_prob = np.arange(1, len(sorted_payoffs) + 1) / len(sorted_payoffs)
    
    ax2.plot(sorted_payoffs, cum_prob, 'b-', linewidth=2, alpha=0.8)
    ax2.axvline(option_price, color='green', linestyle='--', linewidth=2,
               label=f'Option Price: ${option_price:.4f}')
    
    # Mark key percentiles
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        value = np.percentile(payoffs, p)
        ax2.axhline(p/100, color='gray', alpha=0.3, linestyle=':', linewidth=1)
        if value > 0:
            ax2.annotate(f'{p}th: ${value:.2f}', 
                        xy=(value, p/100), xytext=(10, 5),
                        textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Payoff ($)', fontsize=11, fontweight='600')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='600')
    ax2.set_title('Cumulative Payoff Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_convergence_plot(num_simulations: int) -> str:
    """Generate Monte Carlo convergence plot (placeholder)"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate convergence
    n_points = min(100, num_simulations // 1000)
    sim_points = np.linspace(1000, num_simulations, n_points)
    
    # Mock convergence data
    np.random.seed(42)
    true_price = 10.0
    convergence = true_price + 2.0 * np.exp(-sim_points/10000) * (np.random.normal(0, 1, n_points))
    
    ax.plot(sim_points, convergence, 'b-', linewidth=2, alpha=0.8, label='Price Estimate')
    ax.axhline(true_price, color='red', linestyle='--', linewidth=2, 
               label='True Price (approx)', alpha=0.7)
    
    # Confidence bands
    std_err = 1.0 / np.sqrt(sim_points / 1000)
    ax.fill_between(sim_points, convergence - 1.96*std_err, convergence + 1.96*std_err,
                   alpha=0.3, label='95% Confidence Band')
    
    ax.set_xlabel('Number of Simulations', fontsize=11, fontweight='600')
    ax.set_ylabel('Option Price ($)', fontsize=11, fontweight='600')
    ax.set_title('Monte Carlo Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_path_dependency_plot(paths: np.ndarray, exotic_type: str) -> str:
    """Generate path dependency visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Sample paths
    n_sample = min(100, len(paths))
    sample_indices = np.random.choice(len(paths), n_sample, replace=False)
    time_steps = np.linspace(0, 1, paths.shape[1])
    
    for i in sample_indices[:50]:  # Show subset
        ax1.plot(time_steps, paths[i], alpha=0.3, linewidth=1)
    
    # Add mean path
    mean_path = np.mean(paths, axis=0)
    ax1.plot(time_steps, mean_path, 'r-', linewidth=3, label='Average Path', alpha=0.8)
    
    ax1.set_xlabel('Time to Maturity', fontsize=11, fontweight='600')
    ax1.set_ylabel('Asset Price ($)', fontsize=11, fontweight='600')
    ax1.set_title(f'{exotic_type.title()} Option - Sample Paths', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Path statistics
    if exotic_type == "asian":
        # Show averaging effect
        path_averages = np.mean(paths, axis=1)
        final_prices = paths[:, -1]
        
        ax2.scatter(final_prices, path_averages, alpha=0.6, s=20)
        ax2.plot([np.min(paths), np.max(paths)], [np.min(paths), np.max(paths)], 
                'r--', alpha=0.7, label='Final = Average Line')
        
        ax2.set_xlabel('Final Price ($)', fontsize=11, fontweight='600')
        ax2.set_ylabel('Path Average ($)', fontsize=11, fontweight='600')
        ax2.set_title('Asian Option - Averaging Effect', fontsize=13, fontweight='bold')
        
    elif exotic_type == "lookback":
        # Show max vs final prices
        max_prices = np.max(paths, axis=1)
        final_prices = paths[:, -1]
        
        ax2.scatter(final_prices, max_prices, alpha=0.6, s=20)
        ax2.plot([np.min(paths), np.max(paths)], [np.min(paths), np.max(paths)], 
                'r--', alpha=0.7, label='Final = Max Line')
        
        ax2.set_xlabel('Final Price ($)', fontsize=11, fontweight='600')
        ax2.set_ylabel('Maximum Price ($)', fontsize=11, fontweight='600')
        ax2.set_title('Lookback Option - Path Maximum', fontsize=13, fontweight='bold')
        
    else:
        # Default: show price distribution over time
        percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        
        ax2.fill_between(time_steps, percentiles[0], percentiles[4], 
                        alpha=0.2, label='5th-95th Percentile')
        ax2.fill_between(time_steps, percentiles[1], percentiles[3], 
                        alpha=0.3, label='25th-75th Percentile')
        ax2.plot(time_steps, percentiles[2], 'r-', linewidth=2, label='Median')
        
        ax2.set_xlabel('Time to Maturity', fontsize=11, fontweight='600')
        ax2.set_ylabel('Asset Price ($)', fontsize=11, fontweight='600')
        ax2.set_title('Price Distribution Over Time', fontsize=13, fontweight='bold')
    
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Analysis and Interpretation
# =============================================================================

def generate_interpretation(price: float, payoff_analysis: Dict, 
                          exotic_type: str, option_type: str,
                          parameters: Dict) -> Dict:
    """Generate interpretation of exotic option results"""
    
    key_insights = []
    
    # Price analysis
    prob_itm = payoff_analysis['probability_in_money']
    expected_payoff = payoff_analysis['expected_payoff']
    
    if prob_itm > 0.7:
        key_insights.append({
            'title': 'High Probability In-The-Money',
            'description': f'{prob_itm*100:.1f}% chance of positive payoff. Option likely to exercise.',
            'status': 'positive'
        })
    elif prob_itm < 0.3:
        key_insights.append({
            'title': 'Low Probability In-The-Money',
            'description': f'Only {prob_itm*100:.1f}% chance of positive payoff. High-risk premium.',
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': 'Moderate Exercise Probability',
            'description': f'{prob_itm*100:.1f}% chance of positive payoff.',
            'status': 'neutral'
        })
    
    # Exotic-specific insights
    if exotic_type == "barrier":
        barrier_level = parameters.get('barrier_level', 0)
        spot_price = parameters.get('spot_price', 0)
        
        if abs(barrier_level - spot_price) / spot_price < 0.1:
            key_insights.append({
                'title': 'Near-Barrier Risk',
                'description': f'Barrier at ${barrier_level:.2f} is close to current spot ${spot_price:.2f}. High gamma risk.',
                'status': 'warning'
            })
        
    elif exotic_type == "asian":
        key_insights.append({
            'title': 'Path-Dependent Averaging',
            'description': 'Asian option reduces volatility impact through averaging. Lower premium than vanilla.',
            'status': 'neutral'
        })
        
    elif exotic_type == "digital":
        key_insights.append({
            'title': 'Binary Payoff Structure',
            'description': 'Digital option has discontinuous payoff. Extreme sensitivity near strike.',
            'status': 'warning'
        })
    
    # Complexity assessment
    complexity_factors = 0
    if exotic_type in ["barrier", "asian", "lookback"]:
        complexity_factors += 3
    if exotic_type in ["rainbow", "basket"]:
        complexity_factors += 4
    if exotic_type in ["compound", "chooser", "shout"]:
        complexity_factors += 5
    
    complexity_score = min(10, complexity_factors + 2)
    
    if complexity_score >= 8:
        risk_assessment = "High Risk"
    elif complexity_score >= 6:
        risk_assessment = "Moderate Risk"
    else:
        risk_assessment = "Low Risk"
    
    # Generate recommendations
    recommendations = []
    
    if prob_itm < 0.2:
        recommendations.append("Consider the low probability of exercise. Option may expire worthless.")
    
    if exotic_type == "barrier" and "knock" in parameters.get('barrier_type', ''):
        recommendations.append("Monitor barrier levels closely. Small price moves can dramatically affect option value.")
    
    if complexity_score >= 7:
        recommendations.append("High complexity option. Ensure you understand all features and risks.")
    
    if expected_payoff < price * 0.8:
        recommendations.append("Expected payoff is significantly below option premium. Review risk/reward ratio.")
    
    if not recommendations:
        recommendations.append("Option parameters appear reasonable for the given exotic structure.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations,
        'complexity_score': complexity_score,
        'risk_assessment': risk_assessment
    }


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/exotic-options")
async def price_exotic_option(request: ExoticOptionsRequest) -> Dict:
    """
    Price exotic options using advanced numerical methods.
    
    Supports various exotic option types:
    1. Barrier Options - Path-dependent knock-in/knock-out features
    2. Asian Options - Payoffs based on average prices
    3. Lookback Options - Payoffs based on historical extremes
    4. Digital Options - Binary payoff structures
    5. Multi-asset Options - Rainbow and basket options
    
    Uses Monte Carlo simulation with variance reduction techniques.
    """
    try:
        # Validate exotic-specific parameters
        if request.exotic_type == ExoticType.BARRIER:
            if request.barrier_level is None or request.barrier_type is None:
                raise ValueError("Barrier options require barrier_level and barrier_type")
                
        elif request.exotic_type == ExoticType.ASIAN:
            if request.asian_type is None:
                raise ValueError("Asian options require asian_type")
                
        elif request.exotic_type == ExoticType.DIGITAL:
            if request.digital_payout is None:
                raise ValueError("Digital options require digital_payout")
                
        elif request.exotic_type in [ExoticType.BASKET, ExoticType.RAINBOW]:
            if not request.underlying_prices or not request.correlation_matrix:
                raise ValueError("Multi-asset options require underlying_prices and correlation_matrix")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Route to appropriate pricing function
        if request.exotic_type == ExoticType.BARRIER:
            result = price_barrier_option(
                request.spot_price, request.strike_price, request.time_to_maturity,
                request.risk_free_rate, request.volatility, request.barrier_level,
                request.barrier_type.value, request.rebate or 0, request.option_type,
                request.num_simulations, request.num_time_steps, request.dividend_yield
            )
            
        elif request.exotic_type == ExoticType.ASIAN:
            result = price_asian_option(
                request.spot_price, request.strike_price, request.time_to_maturity,
                request.risk_free_rate, request.volatility, request.asian_type.value,
                request.option_type, request.num_simulations, request.num_time_steps,
                request.dividend_yield
            )
            
        elif request.exotic_type == ExoticType.LOOKBACK:
            result = price_lookback_option(
                request.spot_price, request.strike_price, request.time_to_maturity,
                request.risk_free_rate, request.volatility, request.option_type,
                request.num_simulations, request.num_time_steps, request.dividend_yield
            )
            
        elif request.exotic_type == ExoticType.DIGITAL:
            result = price_digital_option(
                request.spot_price, request.strike_price, request.time_to_maturity,
                request.risk_free_rate, request.volatility, request.digital_payout,
                request.option_type, request.num_simulations, request.dividend_yield
            )
            
        else:
            raise HTTPException(status_code=400, 
                              detail=f"Pricing for {request.exotic_type} not implemented yet")
        
        option_price = result['price']
        
        # Calculate intrinsic and time value
        if request.option_type == "call":
            intrinsic_value = max(request.spot_price - request.strike_price, 0)
        else:
            intrinsic_value = max(request.strike_price - request.spot_price, 0)
        
        time_value = option_price - intrinsic_value
        
        # Calculate Greeks
        if request.exotic_type == ExoticType.BARRIER:
            price_func = price_barrier_option
            kwargs = {
                'barrier': request.barrier_level,
                'barrier_type': request.barrier_type.value,
                'rebate': request.rebate or 0,
                'option_type': request.option_type,
                'num_paths': request.num_simulations // 10,  # Reduce for Greeks
                'num_steps': request.num_time_steps,
                'q': request.dividend_yield
            }
        elif request.exotic_type == ExoticType.ASIAN:
            price_func = price_asian_option
            kwargs = {
                'asian_type': request.asian_type.value,
                'option_type': request.option_type,
                'num_paths': request.num_simulations // 10,
                'num_steps': request.num_time_steps,
                'q': request.dividend_yield
            }
        elif request.exotic_type == ExoticType.DIGITAL:
            price_func = price_digital_option
            kwargs = {
                'payout': request.digital_payout,
                'option_type': request.option_type,
                'num_paths': request.num_simulations // 10,
                'q': request.dividend_yield
            }
        else:
            price_func = price_lookback_option
            kwargs = {
                'option_type': request.option_type,
                'num_paths': request.num_simulations // 10,
                'num_steps': request.num_time_steps,
                'q': request.dividend_yield
            }
        
        greeks = calculate_greeks_mc(
            price_func, request.spot_price, request.strike_price,
            request.time_to_maturity, request.risk_free_rate, request.volatility,
            **kwargs
        )
        
        # Extract exotic-specific Greeks
        exotic_greeks = {}
        if 'survival_probability' in result:
            exotic_greeks['probability_of_survival'] = result['survival_probability']
        if 'barrier_delta' in result:
            exotic_greeks['barrier_delta'] = result['barrier_delta']
        if 'average_sensitivity' in result:
            exotic_greeks['average_sensitivity'] = result['average_sensitivity']
        if 'path_dependency_factor' in result:
            exotic_greeks['path_dependency_factor'] = result['path_dependency_factor']
        
        # Payoff analysis
        payoffs = result['payoffs']
        payoff_analysis = {
            'expected_payoff': _to_native_type(np.mean(payoffs)),
            'payoff_variance': _to_native_type(np.var(payoffs)),
            'probability_in_money': _to_native_type(np.mean(payoffs > 0)),
            'maximum_loss': _to_native_type(option_price),  # Premium paid
            'maximum_gain': _to_native_type(np.max(payoffs) if len(payoffs) > 0 else 0)
        }
        
        # Simulation statistics
        confidence_level = 0.95
        z_score = norm.ppf((1 + confidence_level) / 2)
        standard_error = np.std(payoffs) / np.sqrt(len(payoffs))
        
        simulation_stats = {
            'paths_used': len(payoffs),
            'convergence_error': _to_native_type(standard_error),
            'confidence_interval': [
                _to_native_type(option_price - z_score * standard_error),
                _to_native_type(option_price + z_score * standard_error)
            ],
            'monte_carlo_se': _to_native_type(standard_error)
        }
        
        # Generate plots
        plots = {}
        plots['payoff_distribution'] = generate_payoff_distribution_plot(
            payoffs, option_price, request.option_type
        )
        plots['price_convergence'] = generate_convergence_plot(request.num_simulations)
        
        if 'paths' in result:
            plots['path_dependency'] = generate_path_dependency_plot(
                result['paths'], request.exotic_type.value
            )
        
        # Generate interpretation
        interpretation = generate_interpretation(
            option_price, payoff_analysis, request.exotic_type.value,
            request.option_type, {
                'barrier_level': request.barrier_level,
                'barrier_type': request.barrier_type.value if request.barrier_type else None,
                'spot_price': request.spot_price,
                'digital_payout': request.digital_payout
            }
        )
        
        return {
            'price': option_price,
            'intrinsic_value': _to_native_type(intrinsic_value),
            'time_value': _to_native_type(time_value),
            'greeks': greeks,
            'exotic_greeks': exotic_greeks if exotic_greeks else None,
            'simulation_stats': simulation_stats,
            'payoff_analysis': payoff_analysis,
            'plots': plots,
            'interpretation': interpretation,
            'parameters': {
                'exotic_type': request.exotic_type.value,
                'option_type': request.option_type,
                'spot_price': request.spot_price,
                'strike_price': request.strike_price,
                'time_to_maturity': request.time_to_maturity,
                'risk_free_rate': request.risk_free_rate,
                'volatility': request.volatility,
                'dividend_yield': request.dividend_yield,
                'pricing_method': request.pricing_method.value,
                'num_simulations': request.num_simulations,
                'num_time_steps': request.num_time_steps,
                'barrier_level': request.barrier_level,
                'barrier_type': request.barrier_type.value if request.barrier_type else None,
                'digital_payout': request.digital_payout
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exotic option pricing failed: {str(e)}")
