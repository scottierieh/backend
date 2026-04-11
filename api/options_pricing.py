"""
Options Pricing Analysis Router for FastAPI
Comprehensive option pricing with multiple models (Black-Scholes, Binomial, Monte Carlo)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class PricingModel(str, Enum):
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"


class OptionPricingRequest(BaseModel):
    """Option pricing request parameters"""
    
    # Option parameters
    option_type: OptionType = Field(
        default=OptionType.CALL,
        description="Option type (call or put)"
    )
    spot_price: float = Field(
        default=100.0,
        ge=0.01,
        description="Current underlying price"
    )
    strike_price: float = Field(
        default=100.0,
        ge=0.01,
        description="Strike price"
    )
    time_to_maturity: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Time to maturity in years"
    )
    risk_free_rate: float = Field(
        default=0.05,
        ge=-0.1,
        le=0.5,
        description="Risk-free interest rate"
    )
    volatility: float = Field(
        default=0.2,
        ge=0.01,
        le=2.0,
        description="Volatility (annualized)"
    )
    dividend_yield: float = Field(
        default=0.0,
        ge=0.0,
        le=0.2,
        description="Continuous dividend yield"
    )
    
    # Pricing model
    pricing_model: PricingModel = Field(
        default=PricingModel.BLACK_SCHOLES,
        description="Pricing model to use"
    )
    
    # Model-specific parameters
    binomial_steps: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of steps for binomial model"
    )
    monte_carlo_paths: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of paths for Monte Carlo"
    )
    
    # Analysis options
    calculate_greeks: bool = Field(
        default=True,
        description="Calculate option Greeks"
    )
    generate_surfaces: bool = Field(
        default=True,
        description="Generate volatility and price surfaces"
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
# Black-Scholes Model
# =============================================================================

def black_scholes_d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> tuple:
    """Calculate d1 and d2 for Black-Scholes formula"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str, q: float = 0) -> float:
    """
    Black-Scholes option pricing formula
    
    C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
    P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    """
    if T <= 0:
        # At expiration
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma, q)
    
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price


# =============================================================================
# Greeks Calculation
# =============================================================================

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str, q: float = 0) -> Dict[str, float]:
    """Calculate option Greeks using Black-Scholes"""
    
    if T <= 0:
        return {
            'delta': 1.0 if option_type == "call" and S > K else (-1.0 if option_type == "put" and S < K else 0.0),
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma, q)
    
    # Delta
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
    
    # Gamma (same for call and put)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    term1 = -S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (term1 - term2 + term3) / 365  # Per day
    else:
        term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2 + term3) / 365  # Per day
    
    # Vega (same for call and put)
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100  # Per 1% vol change
    
    # Rho
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% rate change
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': _to_native_type(delta),
        'gamma': _to_native_type(gamma),
        'theta': _to_native_type(theta),
        'vega': _to_native_type(vega),
        'rho': _to_native_type(rho)
    }


# =============================================================================
# Binomial Model
# =============================================================================

def binomial_price(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str, n_steps: int, q: float = 0) -> Dict[str, Any]:
    """
    Cox-Ross-Rubinstein Binomial Tree Model
    
    u = e^(σ√Δt)
    d = 1/u
    p = (e^((r-q)Δt) - d) / (u - d)
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # Build price tree at maturity
    stock_prices = np.zeros(n_steps + 1)
    option_values = np.zeros(n_steps + 1)
    
    for i in range(n_steps + 1):
        stock_prices[i] = S * (u ** (n_steps - i)) * (d ** i)
        if option_type == "call":
            option_values[i] = max(stock_prices[i] - K, 0)
        else:
            option_values[i] = max(K - stock_prices[i], 0)
    
    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
    
    price = option_values[0]
    
    return {
        'price': _to_native_type(price),
        'up_factor': _to_native_type(u),
        'down_factor': _to_native_type(d),
        'risk_neutral_prob': _to_native_type(p),
        'n_steps': n_steps
    }


# =============================================================================
# Monte Carlo Model
# =============================================================================

def monte_carlo_price(S: float, K: float, T: float, r: float, sigma: float,
                      option_type: str, n_paths: int, q: float = 0) -> Dict[str, Any]:
    """
    Monte Carlo simulation for option pricing
    Using Geometric Brownian Motion: dS = (r-q)Sdt + σSdW
    """
    np.random.seed(42)
    
    # Generate random paths
    Z = np.random.standard_normal(n_paths)
    
    # Simulate final stock prices
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Discounted expected payoff
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    # Confidence interval
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error
    
    return {
        'price': _to_native_type(price),
        'std_error': _to_native_type(std_error),
        'ci_lower': _to_native_type(ci_lower),
        'ci_upper': _to_native_type(ci_upper),
        'n_paths': n_paths,
        'sample_paths': ST[:100].tolist()  # Sample for visualization
    }


# =============================================================================
# Implied Volatility
# =============================================================================

def implied_volatility(market_price: float, S: float, K: float, T: float, 
                       r: float, option_type: str, q: float = 0) -> float:
    """Calculate implied volatility using Brent's method"""
    
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type, q) - market_price
    
    try:
        iv = brentq(objective, 0.001, 5.0)
        return iv
    except:
        return None


# =============================================================================
# Plot Generation
# =============================================================================

def generate_payoff_plot(S: float, K: float, option_type: str, premium: float) -> str:
    """Generate option payoff diagram"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Price range
    prices = np.linspace(S * 0.5, S * 1.5, 100)
    
    # Payoff at expiration
    if option_type == "call":
        payoff = np.maximum(prices - K, 0)
        profit = payoff - premium
    else:
        payoff = np.maximum(K - prices, 0)
        profit = payoff - premium
    
    ax.plot(prices, payoff, 'b-', linewidth=2, label='Payoff')
    ax.plot(prices, profit, 'g-', linewidth=2, label='Profit/Loss')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=K, color='gray', linestyle='--', linewidth=1, label=f'Strike (K={K})')
    ax.axvline(x=S, color='orange', linestyle='--', linewidth=1, label=f'Spot (S={S})')
    
    ax.fill_between(prices, profit, 0, where=(profit > 0), alpha=0.3, color='green')
    ax.fill_between(prices, profit, 0, where=(profit < 0), alpha=0.3, color='red')
    
    ax.set_xlabel('Underlying Price', fontsize=11)
    ax.set_ylabel('Value ($)', fontsize=11)
    ax.set_title(f'{option_type.capitalize()} Option Payoff Diagram', fontsize=13, fontweight='600')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_greeks_plot(S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str, q: float = 0) -> str:
    """Generate Greeks sensitivity plot"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Price range
    spot_range = np.linspace(S * 0.7, S * 1.3, 50)
    
    deltas, gammas, thetas, vegas, rhos, prices = [], [], [], [], [], []
    
    for s in spot_range:
        greeks = calculate_greeks(s, K, T, r, sigma, option_type, q)
        price = black_scholes_price(s, K, T, r, sigma, option_type, q)
        deltas.append(greeks['delta'])
        gammas.append(greeks['gamma'])
        thetas.append(greeks['theta'])
        vegas.append(greeks['vega'])
        rhos.append(greeks['rho'])
        prices.append(price)
    
    # Price
    ax = axes[0, 0]
    ax.plot(spot_range, prices, 'b-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Option Price')
    ax.set_title('Option Price', fontsize=11, fontweight='600')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Delta
    ax = axes[0, 1]
    ax.plot(spot_range, deltas, 'g-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Delta')
    ax.set_title('Delta (Δ)', fontsize=11, fontweight='600')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Gamma
    ax = axes[0, 2]
    ax.plot(spot_range, gammas, 'r-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Gamma')
    ax.set_title('Gamma (Γ)', fontsize=11, fontweight='600')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Theta
    ax = axes[1, 0]
    ax.plot(spot_range, thetas, 'm-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Theta (per day)')
    ax.set_title('Theta (Θ)', fontsize=11, fontweight='600')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Vega
    ax = axes[1, 1]
    ax.plot(spot_range, vegas, 'c-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Vega (per 1% vol)')
    ax.set_title('Vega (ν)', fontsize=11, fontweight='600')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Rho
    ax = axes[1, 2]
    ax.plot(spot_range, rhos, 'orange', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Rho (per 1% rate)')
    ax.set_title('Rho (ρ)', fontsize=11, fontweight='600')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_volatility_surface_plot(S: float, K: float, T: float, r: float,
                                      option_type: str, q: float = 0) -> str:
    """Generate price surface over spot and volatility"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    spot_range = np.linspace(S * 0.7, S * 1.3, 30)
    vol_range = np.linspace(0.1, 0.6, 30)
    
    X, Y = np.meshgrid(spot_range, vol_range)
    Z = np.zeros_like(X)
    
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            Z[i, j] = black_scholes_price(X[i, j], K, T, r, Y[i, j], option_type, q)
    
    contour = ax.contourf(X, Y * 100, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Option Price ($)')
    
    ax.axvline(x=S, color='white', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=r * 100, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Spot Price ($)', fontsize=11)
    ax.set_ylabel('Volatility (%)', fontsize=11)
    ax.set_title('Option Price Surface', fontsize=13, fontweight='600')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_time_decay_plot(S: float, K: float, T: float, r: float, sigma: float,
                              option_type: str, q: float = 0) -> str:
    """Generate time decay plot"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Different moneyness levels
    strikes = [K * 0.9, K, K * 1.1]
    labels = ['ITM (K=90%)', 'ATM (K=100%)', 'OTM (K=110%)'] if option_type == 'call' else ['OTM (K=90%)', 'ATM (K=100%)', 'ITM (K=110%)']
    colors = ['green', 'blue', 'red']
    
    time_range = np.linspace(T, 0.01, 50)
    
    for strike, label, color in zip(strikes, labels, colors):
        prices = [black_scholes_price(S, strike, t, r, sigma, option_type, q) for t in time_range]
        days_to_expiry = time_range * 365
        ax.plot(days_to_expiry, prices, color=color, linewidth=2, label=label)
    
    ax.set_xlabel('Days to Expiration', fontsize=11)
    ax.set_ylabel('Option Price ($)', fontsize=11)
    ax.set_title('Time Decay (Theta Effect)', fontsize=13, fontweight='600')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_monte_carlo_plot(sample_paths: List[float], K: float, option_type: str) -> str:
    """Generate Monte Carlo simulation histogram"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    paths = np.array(sample_paths)
    
    n, bins, patches = ax.hist(paths, bins=50, density=True, alpha=0.7, 
                                color='#6366f1', edgecolor='white')
    
    # Color ITM region
    for patch, b in zip(patches, bins[:-1]):
        if option_type == 'call' and b > K:
            patch.set_facecolor('#22c55e')
        elif option_type == 'put' and b < K:
            patch.set_facecolor('#22c55e')
    
    ax.axvline(x=K, color='#ef4444', linestyle='--', linewidth=2, label=f'Strike = ${K}')
    ax.axvline(x=np.mean(paths), color='#3b82f6', linestyle='-', linewidth=2, 
               label=f'Mean = ${np.mean(paths):.2f}')
    
    ax.set_xlabel('Final Stock Price ($)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Monte Carlo Simulation - Final Price Distribution', fontsize=13, fontweight='600')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Interpretation Generator
# =============================================================================

def generate_interpretation(price: float, greeks: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of option pricing results"""
    key_insights = []
    
    S, K = params['spot_price'], params['strike_price']
    option_type = params['option_type']
    
    # Moneyness
    moneyness = S / K
    if option_type == 'call':
        if moneyness > 1.05:
            status = 'ITM'
            key_insights.append({
                'title': 'In-The-Money Call',
                'description': f'Option is {((moneyness - 1) * 100):.1f}% in the money. High intrinsic value.',
                'status': 'positive'
            })
        elif moneyness < 0.95:
            status = 'OTM'
            key_insights.append({
                'title': 'Out-of-The-Money Call',
                'description': f'Option is {((1 - moneyness) * 100):.1f}% out of the money. Pure time value.',
                'status': 'neutral'
            })
        else:
            status = 'ATM'
            key_insights.append({
                'title': 'At-The-Money Call',
                'description': 'Option is near the money. Maximum time value and gamma.',
                'status': 'neutral'
            })
    else:
        if moneyness < 0.95:
            status = 'ITM'
            key_insights.append({
                'title': 'In-The-Money Put',
                'description': f'Option is {((1 - moneyness) * 100):.1f}% in the money.',
                'status': 'positive'
            })
        elif moneyness > 1.05:
            status = 'OTM'
            key_insights.append({
                'title': 'Out-of-The-Money Put',
                'description': f'Option is {((moneyness - 1) * 100):.1f}% out of the money.',
                'status': 'neutral'
            })
        else:
            status = 'ATM'
            key_insights.append({
                'title': 'At-The-Money Put',
                'description': 'Option is near the money. Maximum time value and gamma.',
                'status': 'neutral'
            })
    
    # Greeks insights
    if greeks:
        delta = greeks.get('delta', 0)
        gamma = greeks.get('gamma', 0)
        theta = greeks.get('theta', 0)
        
        key_insights.append({
            'title': 'Delta Hedge Ratio',
            'description': f'Delta of {delta:.3f} means hedge {abs(delta)*100:.1f} shares per option.',
            'status': 'neutral'
        })
        
        if abs(theta) > price * 0.01:
            key_insights.append({
                'title': 'Significant Time Decay',
                'description': f'Theta of ${theta:.4f}/day. Option loses ~${abs(theta)*7:.2f} per week.',
                'status': 'warning'
            })
    
    # Recommendations
    recommendations = []
    
    if status == 'OTM' and params['time_to_maturity'] < 0.1:
        recommendations.append("OTM option near expiry has high theta decay. Consider rolling or closing.")
    
    if greeks and greeks.get('gamma', 0) > 0.1:
        recommendations.append("High gamma indicates delta will change rapidly. Frequent rehedging may be needed.")
    
    if params['volatility'] > 0.4:
        recommendations.append("High volatility environment. Option premium is expensive relative to historical norms.")
    
    if not recommendations:
        recommendations.append("Option parameters are within normal ranges.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations,
        'moneyness_status': status
    }


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/options-pricing")
async def price_option(request: OptionPricingRequest) -> Dict[str, Any]:
    """
    Price options using multiple models and calculate Greeks.
    
    Models:
    1. Black-Scholes-Merton - Closed-form solution for European options
    2. Binomial Tree (CRR) - Discrete-time model, works for American options
    3. Monte Carlo - Simulation-based, flexible for exotic options
    
    Greeks:
    - Delta (Δ): Price sensitivity to underlying
    - Gamma (Γ): Delta sensitivity to underlying  
    - Theta (Θ): Time decay
    - Vega (ν): Volatility sensitivity
    - Rho (ρ): Interest rate sensitivity
    """
    try:
        S = request.spot_price
        K = request.strike_price
        T = request.time_to_maturity
        r = request.risk_free_rate
        sigma = request.volatility
        q = request.dividend_yield
        option_type = request.option_type.value
        
        # Black-Scholes price (always calculate as benchmark)
        bs_price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        
        # Primary model price
        if request.pricing_model == PricingModel.BLACK_SCHOLES:
            primary_price = bs_price
            model_details = {
                'd1': _to_native_type(black_scholes_d1_d2(S, K, T, r, sigma, q)[0]),
                'd2': _to_native_type(black_scholes_d1_d2(S, K, T, r, sigma, q)[1])
            }
        elif request.pricing_model == PricingModel.BINOMIAL:
            binomial_result = binomial_price(S, K, T, r, sigma, option_type, 
                                             request.binomial_steps, q)
            primary_price = binomial_result['price']
            model_details = binomial_result
        else:  # Monte Carlo
            mc_result = monte_carlo_price(S, K, T, r, sigma, option_type,
                                          request.monte_carlo_paths, q)
            primary_price = mc_result['price']
            model_details = mc_result
        
        # Calculate Greeks
        greeks = calculate_greeks(S, K, T, r, sigma, option_type, q) if request.calculate_greeks else None
        
        # Intrinsic and time value
        if option_type == "call":
            intrinsic_value = max(S - K, 0)
        else:
            intrinsic_value = max(K - S, 0)
        time_value = primary_price - intrinsic_value
        
        # Model comparison
        binomial_result = binomial_price(S, K, T, r, sigma, option_type, 100, q)
        mc_result = monte_carlo_price(S, K, T, r, sigma, option_type, 10000, q)
        
        model_comparison = {
            'black_scholes': _to_native_type(bs_price),
            'binomial_100': _to_native_type(binomial_result['price']),
            'monte_carlo_10k': _to_native_type(mc_result['price'])
        }
        
        # Generate plots
        plots = {
            'payoff': generate_payoff_plot(S, K, option_type, primary_price),
            'time_decay': generate_time_decay_plot(S, K, T, r, sigma, option_type, q)
        }
        
        if request.calculate_greeks:
            plots['greeks'] = generate_greeks_plot(S, K, T, r, sigma, option_type, q)
        
        if request.generate_surfaces:
            plots['volatility_surface'] = generate_volatility_surface_plot(S, K, T, r, option_type, q)
        
        if request.pricing_model == PricingModel.MONTE_CARLO:
            plots['monte_carlo'] = generate_monte_carlo_plot(model_details['sample_paths'], K, option_type)
        
        # Generate interpretation
        interpretation = generate_interpretation(primary_price, greeks, {
            'spot_price': S,
            'strike_price': K,
            'option_type': option_type,
            'time_to_maturity': T,
            'volatility': sigma
        })
        
        return {
            'price': _to_native_type(primary_price),
            'intrinsic_value': _to_native_type(intrinsic_value),
            'time_value': _to_native_type(time_value),
            'greeks': greeks,
            'model_details': model_details,
            'model_comparison': model_comparison,
            'plots': plots,
            'interpretation': interpretation,
            'parameters': {
                'option_type': option_type,
                'spot_price': S,
                'strike_price': K,
                'time_to_maturity': T,
                'risk_free_rate': r,
                'volatility': sigma,
                'dividend_yield': q,
                'pricing_model': request.pricing_model.value
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Option pricing failed: {str(e)}")
