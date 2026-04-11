"""
Stress Testing Analysis Router for FastAPI
Comprehensive stress testing with scenario analysis, sensitivity testing, and risk attribution
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
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

class StressScenario(str, Enum):
    FINANCIAL_CRISIS = "financial_crisis"
    COVID_SHOCK = "covid_shock"
    INFLATION_SPIKE = "inflation_spike"
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"
    CURRENCY_CRISIS = "currency_crisis"
    TECH_BUBBLE_BURST = "tech_bubble_burst"
    CUSTOM = "custom"

class RiskFactor(str, Enum):
    EQUITY = "equity"
    INTEREST_RATE = "interest_rate"
    CREDIT_SPREAD = "credit_spread"
    FX_USD = "fx_usd"
    COMMODITY = "commodity"
    VOLATILITY = "volatility"
    REAL_ESTATE = "real_estate"
    EMERGING_MARKETS = "emerging_markets"

class StressTestRequest(BaseModel):
    """Stress test request parameters"""
    
    # Portfolio configuration
    portfolio_value: float = Field(
        default=1000000.0,
        ge=1000,
        description="Current portfolio value"
    )
    portfolio_exposures: Dict[str, float] = Field(
        ...,
        description="Risk factor exposures (can be > 1 for leverage)"
    )
    
    # Scenario configuration
    scenario: StressScenario = Field(
        default=StressScenario.FINANCIAL_CRISIS,
        description="Predefined stress scenario or custom"
    )
    custom_shocks: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom shock magnitudes for each risk factor"
    )
    
    # Analysis parameters
    time_horizon: int = Field(
        default=1,
        ge=1,
        le=252,
        description="Time horizon in days"
    )
    confidence_level: float = Field(
        default=0.99,
        ge=0.9,
        le=0.999,
        description="Confidence level for tail risk analysis"
    )
    
    # Advanced options
    include_correlations: bool = Field(
        default=True,
        description="Include correlation effects"
    )
    include_liquidity: bool = Field(
        default=True,
        description="Include liquidity risk adjustments"
    )
    include_concentration: bool = Field(
        default=True,
        description="Include concentration risk analysis"
    )
    num_simulations: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo simulations"
    )

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
# Scenario Definitions
# =============================================================================

PREDEFINED_SCENARIOS = {
    StressScenario.FINANCIAL_CRISIS: {
        "name": "Financial Crisis (2008-style)",
        "description": "Global financial system stress with credit crunch",
        "shocks": {
            RiskFactor.EQUITY: -0.40,           # -40% equity markets
            RiskFactor.INTEREST_RATE: -0.03,    # -300bp rates (flight to quality)
            RiskFactor.CREDIT_SPREAD: 0.05,     # +500bp credit spreads
            RiskFactor.FX_USD: 0.15,            # +15% USD strength
            RiskFactor.COMMODITY: -0.30,        # -30% commodities
            RiskFactor.VOLATILITY: 2.0,         # +200% volatility
            RiskFactor.REAL_ESTATE: -0.25,      # -25% real estate
            RiskFactor.EMERGING_MARKETS: -0.50  # -50% EM assets
        }
    },
    StressScenario.COVID_SHOCK: {
        "name": "COVID-19 Pandemic Shock",
        "description": "Global lockdown and economic shutdown",
        "shocks": {
            RiskFactor.EQUITY: -0.35,
            RiskFactor.INTEREST_RATE: -0.02,
            RiskFactor.CREDIT_SPREAD: 0.04,
            RiskFactor.FX_USD: 0.10,
            RiskFactor.COMMODITY: -0.25,
            RiskFactor.VOLATILITY: 1.5,
            RiskFactor.REAL_ESTATE: -0.15,
            RiskFactor.EMERGING_MARKETS: -0.40
        }
    },
    StressScenario.INFLATION_SPIKE: {
        "name": "Inflation Spike Scenario",
        "description": "Persistent high inflation forcing aggressive policy",
        "shocks": {
            RiskFactor.EQUITY: -0.25,
            RiskFactor.INTEREST_RATE: 0.04,     # +400bp rates
            RiskFactor.CREDIT_SPREAD: 0.02,
            RiskFactor.FX_USD: 0.05,
            RiskFactor.COMMODITY: 0.30,         # +30% commodities
            RiskFactor.VOLATILITY: 0.8,
            RiskFactor.REAL_ESTATE: 0.30,       # Real assets benefit
            RiskFactor.EMERGING_MARKETS: -0.20
        }
    },
    StressScenario.GEOPOLITICAL_CRISIS: {
        "name": "Geopolitical Crisis",
        "description": "Major geopolitical event disrupting global trade",
        "shocks": {
            RiskFactor.EQUITY: -0.20,
            RiskFactor.INTEREST_RATE: -0.01,    # Flight to safety
            RiskFactor.CREDIT_SPREAD: 0.03,
            RiskFactor.FX_USD: 0.12,            # Safe haven demand
            RiskFactor.COMMODITY: 0.50,         # Supply disruption
            RiskFactor.VOLATILITY: 1.2,
            RiskFactor.REAL_ESTATE: -0.10,
            RiskFactor.EMERGING_MARKETS: -0.30
        }
    },
    StressScenario.CURRENCY_CRISIS: {
        "name": "Currency Crisis",
        "description": "Emerging market currency collapse",
        "shocks": {
            RiskFactor.EQUITY: -0.15,
            RiskFactor.INTEREST_RATE: 0.02,
            RiskFactor.CREDIT_SPREAD: 0.03,
            RiskFactor.FX_USD: 0.25,            # Major USD strength
            RiskFactor.COMMODITY: -0.20,
            RiskFactor.VOLATILITY: 1.0,
            RiskFactor.REAL_ESTATE: -0.20,
            RiskFactor.EMERGING_MARKETS: -0.50  # Major EM selloff
        }
    },
    StressScenario.TECH_BUBBLE_BURST: {
        "name": "Tech Bubble Burst",
        "description": "Technology sector valuation collapse",
        "shocks": {
            RiskFactor.EQUITY: -0.30,           # Heavily weighted to tech
            RiskFactor.INTEREST_RATE: -0.02,    # Deflationary pressure
            RiskFactor.CREDIT_SPREAD: 0.025,
            RiskFactor.FX_USD: 0.05,
            RiskFactor.COMMODITY: -0.15,
            RiskFactor.VOLATILITY: 1.5,
            RiskFactor.REAL_ESTATE: -0.15,
            RiskFactor.EMERGING_MARKETS: -0.25
        }
    }
}

# =============================================================================
# Risk Factor Correlation Matrix
# =============================================================================

def get_risk_factor_correlations() -> np.ndarray:
    """
    Get correlation matrix for risk factors
    Based on empirical relationships during stress periods
    """
    factors = list(RiskFactor)
    n_factors = len(factors)
    
    # Initialize correlation matrix
    corr_matrix = np.eye(n_factors)
    
    # Define correlations (stress period relationships)
    correlations = {
        (RiskFactor.EQUITY, RiskFactor.EMERGING_MARKETS): 0.85,
        (RiskFactor.EQUITY, RiskFactor.REAL_ESTATE): 0.65,
        (RiskFactor.EQUITY, RiskFactor.CREDIT_SPREAD): -0.70,
        (RiskFactor.EQUITY, RiskFactor.VOLATILITY): -0.75,
        (RiskFactor.EQUITY, RiskFactor.INTEREST_RATE): -0.30,
        (RiskFactor.INTEREST_RATE, RiskFactor.CREDIT_SPREAD): 0.40,
        (RiskFactor.CREDIT_SPREAD, RiskFactor.VOLATILITY): 0.60,
        (RiskFactor.FX_USD, RiskFactor.EMERGING_MARKETS): -0.60,
        (RiskFactor.FX_USD, RiskFactor.COMMODITY): -0.40,
        (RiskFactor.COMMODITY, RiskFactor.EMERGING_MARKETS): 0.55,
        (RiskFactor.VOLATILITY, RiskFactor.REAL_ESTATE): -0.45,
    }
    
    # Fill correlation matrix
    factor_to_idx = {factor: i for i, factor in enumerate(factors)}
    
    for (factor1, factor2), corr in correlations.items():
        i, j = factor_to_idx[factor1], factor_to_idx[factor2]
        corr_matrix[i, j] = corr
        corr_matrix[j, i] = corr
    
    return corr_matrix

# =============================================================================
# Core Stress Testing Functions
# =============================================================================

def calculate_scenario_impact(
    portfolio_exposures: Dict[str, float],
    scenario_shocks: Dict[str, float],
    portfolio_value: float,
    include_correlations: bool = True,
    include_liquidity: bool = True
) -> Dict[str, Any]:
    """Calculate portfolio impact from scenario shocks"""
    
    # Ensure all risk factors are present
    all_factors = list(RiskFactor)
    exposures = {factor.value: portfolio_exposures.get(factor.value, 0.0) for factor in all_factors}
    shocks = {factor.value: scenario_shocks.get(factor.value, 0.0) for factor in all_factors}
    
    # Base linear impact calculation
    factor_contributions = {}
    total_base_impact = 0
    
    for factor in all_factors:
        factor_key = factor.value
        exposure = exposures[factor_key]
        shock = shocks[factor_key]
        contribution = exposure * shock * portfolio_value
        
        factor_contributions[factor_key] = {
            'shock': shock,
            'exposure': exposure,
            'contribution': contribution,
            'percentage_contribution': (contribution / portfolio_value * 100) if portfolio_value != 0 else 0
        }
        
        total_base_impact += contribution
    
    # Add correlation effects if requested
    correlation_adjustment = 0
    if include_correlations:
        corr_matrix = get_risk_factor_correlations()
        exposure_vector = np.array([exposures[factor.value] for factor in all_factors])
        shock_vector = np.array([shocks[factor.value] for factor in all_factors])
        
        # Quadratic form: x'Σx where x is exposure*shock vector
        combined_vector = exposure_vector * shock_vector
        quadratic_impact = np.dot(combined_vector, np.dot(corr_matrix, combined_vector))
        linear_impact = np.sum(combined_vector**2)
        
        correlation_adjustment = (quadratic_impact - linear_impact) * portfolio_value * 0.5
    
    # Add liquidity adjustment if requested
    liquidity_adjustment = 0
    if include_liquidity:
        # Simple liquidity cost model based on market stress
        stress_intensity = np.sqrt(np.mean([abs(shocks[factor.value]) for factor in all_factors]))
        liquidity_cost_rate = min(0.05, stress_intensity * 0.1)  # Max 5% liquidity cost
        liquidity_adjustment = -abs(total_base_impact) * liquidity_cost_rate
    
    # Calculate total impact
    total_impact = total_base_impact + correlation_adjustment + liquidity_adjustment
    stressed_value = portfolio_value + total_impact
    loss_percentage = -total_impact / portfolio_value * 100 if portfolio_value != 0 else 0
    
    return {
        'base_portfolio_value': portfolio_value,
        'stressed_portfolio_value': max(0, stressed_value),  # Can't go negative
        'total_loss': -total_impact,
        'loss_percentage': loss_percentage,
        'risk_factor_contributions': factor_contributions,
        'correlation_adjustment': correlation_adjustment,
        'liquidity_adjustment': liquidity_adjustment
    }

def calculate_sensitivity_analysis(
    portfolio_exposures: Dict[str, float],
    portfolio_value: float,
    shock_range: float = 0.1
) -> Dict[str, Any]:
    """Calculate portfolio sensitivity to each risk factor"""
    
    sensitivities = {}
    all_factors = list(RiskFactor)
    
    for factor in all_factors:
        factor_key = factor.value
        exposure = portfolio_exposures.get(factor_key, 0.0)
        
        # Calculate delta (first derivative)
        delta = exposure * portfolio_value
        
        # Calculate gamma (second derivative) - simplified convexity estimate
        gamma = 0.1 * delta * exposure  # Simplified non-linear effect
        
        # Calculate theta (time decay) - for options-like exposures
        theta = -0.01 * abs(delta)  # Simple time decay estimate
        
        sensitivities[factor_key] = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta
        }
    
    # Portfolio-level greeks
    total_delta = sum([sens['delta'] for sens in sensitivities.values()])
    total_gamma = sum([sens['gamma'] for sens in sensitivities.values()])
    total_vega = abs(total_delta) * 0.01  # Volatility sensitivity estimate
    
    return {
        'risk_factor_sensitivities': sensitivities,
        'correlation_impact': {},  # Placeholder for correlation sensitivity
        'portfolio_greeks': {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_vega': total_vega
        }
    }

def calculate_reverse_stress_test(
    portfolio_exposures: Dict[str, float],
    portfolio_value: float,
    loss_targets: List[float] = [0.10, 0.20, 0.30]
) -> Dict[str, Any]:
    """Find shocks required to achieve target losses"""
    
    required_shocks = {}
    all_factors = list(RiskFactor)
    
    for target_loss in loss_targets:
        target_amount = portfolio_value * target_loss
        
        # Simple approach: scale predefined scenario to achieve target
        base_scenario = PREDEFINED_SCENARIOS[StressScenario.FINANCIAL_CRISIS]['shocks']
        
        # Calculate impact of base scenario
        base_impact = calculate_scenario_impact(
            portfolio_exposures, 
            {k.value: v for k, v in base_scenario.items()}, 
            portfolio_value,
            include_correlations=False,
            include_liquidity=False
        )
        
        if abs(base_impact['total_loss']) > 0:
            scale_factor = target_amount / base_impact['total_loss']
        else:
            scale_factor = 1.0
        
        scaled_shocks = {k.value: v * scale_factor for k, v in base_scenario.items()}
        required_shocks[f"{target_loss:.0%}"] = scaled_shocks
    
    # Historical precedents (simplified)
    historical_precedents = {
        "10%": [
            {"date": "2020-03", "magnitude": 0.12, "description": "COVID-19 Initial Shock"},
            {"date": "2018-12", "magnitude": 0.09, "description": "Fed Tightening Fears"}
        ],
        "20%": [
            {"date": "2008-09", "magnitude": 0.18, "description": "Lehman Brothers Collapse"},
            {"date": "2020-03", "magnitude": 0.22, "description": "COVID-19 Peak Selloff"}
        ],
        "30%": [
            {"date": "2008-10", "magnitude": 0.31, "description": "Financial Crisis Peak"},
            {"date": "2000-03", "magnitude": 0.28, "description": "Dot-com Bubble Burst"}
        ]
    }
    
    # Probability estimates (very simplified)
    probability_estimates = {
        "10%": 0.05,  # 5% annual probability
        "20%": 0.02,  # 2% annual probability  
        "30%": 0.005  # 0.5% annual probability
    }
    
    return {
        'loss_targets': [t * 100 for t in loss_targets],
        'required_shocks': required_shocks,
        'probability_estimates': probability_estimates,
        'historical_precedents': historical_precedents
    }

def calculate_concentration_risk(
    portfolio_exposures: Dict[str, float]
) -> Dict[str, Any]:
    """Analyze portfolio concentration across different dimensions"""
    
    # Factor concentration
    exposures = np.array([abs(exp) for exp in portfolio_exposures.values()])
    total_exposure = np.sum(exposures)
    
    if total_exposure > 0:
        weights = exposures / total_exposure
        herfindahl_index = np.sum(weights**2)
        
        # Top N concentration (simplified)
        sorted_weights = np.sort(weights)[::-1]
        top_3_concentration = np.sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else np.sum(sorted_weights)
        
        effective_diversification = 1 / herfindahl_index if herfindahl_index > 0 else 1
    else:
        herfindahl_index = 0
        top_3_concentration = 0
        effective_diversification = 1
    
    # Sector concentrations (simplified mapping)
    sector_mapping = {
        'equity': 'Equity Markets',
        'emerging_markets': 'Equity Markets',
        'real_estate': 'Real Assets',
        'commodity': 'Real Assets',
        'interest_rate': 'Fixed Income',
        'credit_spread': 'Fixed Income',
        'fx_usd': 'Currencies',
        'volatility': 'Derivatives'
    }
    
    sector_concentrations = {}
    for factor, exposure in portfolio_exposures.items():
        sector = sector_mapping.get(factor, 'Other')
        sector_concentrations[sector] = sector_concentrations.get(sector, 0) + abs(exposure)
    
    return {
        'sector_concentrations': sector_concentrations,
        'geographic_concentrations': {'Developed': 0.7, 'Emerging': 0.3},  # Simplified
        'single_name_exposures': {},  # Would require position-level data
        'concentration_metrics': {
            'herfindahl_index': float(herfindahl_index),
            'concentration_ratio': float(top_3_concentration),
            'effective_diversification': float(effective_diversification)
        }
    }

def calculate_tail_scenarios(
    portfolio_exposures: Dict[str, float],
    portfolio_value: float,
    confidence_level: float,
    num_simulations: int = 10000
) -> Dict[str, Any]:
    """Analyze tail risk scenarios using Monte Carlo simulation"""
    
    np.random.seed(42)  # For reproducibility
    
    # Generate correlated random shocks
    all_factors = list(RiskFactor)
    corr_matrix = get_risk_factor_correlations()
    
    # Base volatilities for each factor (estimated)
    factor_vols = {
        RiskFactor.EQUITY: 0.20,
        RiskFactor.INTEREST_RATE: 0.015,
        RiskFactor.CREDIT_SPREAD: 0.01,
        RiskFactor.FX_USD: 0.12,
        RiskFactor.COMMODITY: 0.25,
        RiskFactor.VOLATILITY: 0.50,
        RiskFactor.REAL_ESTATE: 0.15,
        RiskFactor.EMERGING_MARKETS: 0.30
    }
    
    vol_vector = np.array([factor_vols[factor] for factor in all_factors])
    
    # Generate random scenarios
    random_shocks = multivariate_normal.rvs(
        mean=np.zeros(len(all_factors)),
        cov=corr_matrix,
        size=num_simulations
    )
    
    # Scale by volatilities
    scaled_shocks = random_shocks * vol_vector
    
    # Calculate portfolio impacts
    portfolio_impacts = []
    for i in range(num_simulations):
        shock_dict = {factor.value: scaled_shocks[i, j] for j, factor in enumerate(all_factors)}
        impact = calculate_scenario_impact(
            portfolio_exposures, shock_dict, portfolio_value,
            include_correlations=False, include_liquidity=False
        )
        portfolio_impacts.append(impact['loss_percentage'])
    
    portfolio_impacts = np.array(portfolio_impacts)
    
    # Calculate percentile losses
    percentiles = [90, 95, 99, 99.5, 99.9]
    percentile_losses = {}
    for p in percentiles:
        percentile_losses[str(p)] = float(np.percentile(portfolio_impacts, p))
    
    # Extreme scenarios
    extreme_threshold = np.percentile(portfolio_impacts, 99)
    extreme_indices = np.where(portfolio_impacts >= extreme_threshold)[0]
    
    extreme_scenarios = [
        {
            'name': f'Tail Scenario {i+1}',
            'probability': 0.01,  # 1% probability
            'loss_amount': portfolio_impacts[idx] * portfolio_value / 100,
            'loss_percentage': portfolio_impacts[idx],
            'description': f'Extreme market stress scenario with {portfolio_impacts[idx]:.1f}% loss'
        }
        for i, idx in enumerate(extreme_indices[:3])  # Top 3 extreme scenarios
    ]
    
    # Black swan indicators
    tail_dependency = float(np.corrcoef(
        portfolio_impacts[portfolio_impacts > np.percentile(portfolio_impacts, 95)],
        np.arange(len(portfolio_impacts[portfolio_impacts > np.percentile(portfolio_impacts, 95)]))
    )[0, 1]) if len(portfolio_impacts[portfolio_impacts > np.percentile(portfolio_impacts, 95)]) > 1 else 0
    
    extreme_correlation = float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
    liquidity_risk_score = min(1.0, np.std(portfolio_impacts) / np.mean(np.abs(portfolio_impacts)) if np.mean(np.abs(portfolio_impacts)) > 0 else 0)
    
    return {
        'percentile_losses': percentile_losses,
        'extreme_scenarios': extreme_scenarios,
        'black_swan_indicators': {
            'tail_dependency': tail_dependency,
            'extreme_correlation': extreme_correlation,
            'liquidity_risk_score': liquidity_risk_score
        }
    }

def calculate_risk_attribution(
    portfolio_exposures: Dict[str, float],
    portfolio_value: float
) -> Dict[str, Any]:
    """Decompose portfolio risk into components"""
    
    all_factors = list(RiskFactor)
    corr_matrix = get_risk_factor_correlations()
    
    # Factor volatilities (annualized)
    factor_vols = {
        RiskFactor.EQUITY: 0.20,
        RiskFactor.INTEREST_RATE: 0.015,
        RiskFactor.CREDIT_SPREAD: 0.01,
        RiskFactor.FX_USD: 0.12,
        RiskFactor.COMMODITY: 0.25,
        RiskFactor.VOLATILITY: 0.50,
        RiskFactor.REAL_ESTATE: 0.15,
        RiskFactor.EMERGING_MARKETS: 0.30
    }
    
    exposures = np.array([portfolio_exposures.get(factor.value, 0.0) for factor in all_factors])
    vols = np.array([factor_vols[factor] for factor in all_factors])
    
    # Portfolio variance calculation
    portfolio_var = np.dot(exposures * vols, np.dot(corr_matrix, exposures * vols))
    portfolio_vol = np.sqrt(portfolio_var)
    
    factor_contributions = {}
    
    for i, factor in enumerate(all_factors):
        factor_key = factor.value
        exposure = exposures[i]
        vol = vols[i]
        
        # Stand-alone risk
        stand_alone_risk = abs(exposure) * vol
        
        # Marginal contribution to risk
        marginal_contrib = np.dot(corr_matrix[i], exposures * vols) * exposure * vol / portfolio_vol if portfolio_vol > 0 else 0
        
        # Component contribution
        component_risk = marginal_contrib * exposure * vol / portfolio_vol if portfolio_vol > 0 else 0
        
        # Diversification benefit
        diversification_benefit = stand_alone_risk - abs(component_risk)
        
        factor_contributions[factor_key] = {
            'stand_alone_risk': float(stand_alone_risk),
            'marginal_risk': float(marginal_contrib),
            'component_risk': float(component_risk),
            'diversification_benefit': float(diversification_benefit)
        }
    
    # Portfolio decomposition
    systematic_var = portfolio_var
    idiosyncratic_var = max(0, np.sum((exposures * vols)**2) - systematic_var)
    correlation_benefit = np.sum((exposures * vols)**2) - portfolio_var
    
    return {
        'factor_contributions': factor_contributions,
        'portfolio_decomposition': {
            'systematic_risk': float(np.sqrt(systematic_var)),
            'idiosyncratic_risk': float(np.sqrt(idiosyncratic_var)),
            'correlation_benefit': float(np.sqrt(abs(correlation_benefit))),
            'total_portfolio_risk': float(portfolio_vol)
        }
    }

# =============================================================================
# Plot Generation Functions
# =============================================================================

def generate_scenario_waterfall_plot(factor_contributions: Dict, portfolio_value: float) -> str:
    """Generate waterfall chart showing factor contributions"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    factors = list(factor_contributions.keys())
    contributions = [factor_contributions[f]['contribution'] for f in factors]
    
    # Sort by absolute contribution
    sorted_data = sorted(zip(factors, contributions), key=lambda x: abs(x[1]), reverse=True)
    factors_sorted, contributions_sorted = zip(*sorted_data)
    
    # Create waterfall chart
    x_pos = np.arange(len(factors_sorted))
    colors = ['red' if c < 0 else 'green' for c in contributions_sorted]
    
    # Calculate cumulative for connecting lines
    cumulative = np.cumsum([0] + list(contributions_sorted))
    
    bars = ax.bar(x_pos, contributions_sorted, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
    
    # Add connecting lines
    for i in range(len(x_pos)-1):
        ax.plot([x_pos[i]+0.4, x_pos[i+1]-0.4], 
                [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Risk Factors', fontsize=12, fontweight='600')
    ax.set_ylabel('Contribution ($)', fontsize=12, fontweight='600')
    ax.set_title('Stress Test Loss Attribution by Risk Factor', fontsize=14, fontweight='700')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in factors_sorted], 
                       rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, contributions_sorted)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + np.sign(height) * max(abs(height) * 0.01, portfolio_value * 0.001),
                f'${val:,.0f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='600')
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_sensitivity_heatmap(sensitivities: Dict) -> str:
    """Generate sensitivity heatmap"""
    
    factors = list(sensitivities.keys())
    metrics = ['delta', 'gamma', 'theta']
    
    # Create matrix
    data = []
    for metric in metrics:
        row = [sensitivities[f][metric] for f in factors]
        data.append(row)
    
    data = np.array(data)
    
    # Normalize for better visualization
    data_norm = np.zeros_like(data)
    for i in range(data.shape[0]):
        max_val = np.max(np.abs(data[i]))
        if max_val > 0:
            data_norm[i] = data[i] / max_val
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(data_norm, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(factors)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([f.replace('_', ' ').title() for f in factors], rotation=45, ha='right')
    ax.set_yticklabels([m.title() for m in metrics])
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(factors)):
            text = ax.text(j, i, f'{data[i, j]:.0f}', ha="center", va="center", 
                         color="white" if abs(data_norm[i, j]) > 0.5 else "black", fontweight='600')
    
    ax.set_title("Portfolio Sensitivity Analysis (Greeks)", fontsize=14, fontweight='700', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Sensitivity', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_tail_distribution_plot(percentile_losses: Dict) -> str:
    """Generate tail risk distribution plot"""
    
    percentiles = [float(p) for p in percentile_losses.keys()]
    losses = [percentile_losses[str(p)] for p in percentiles]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Percentile losses
    bars = ax1.bar(range(len(percentiles)), losses, 
                   color=['lightblue', 'orange', 'red', 'darkred', 'maroon'], 
                   edgecolor='black', linewidth=0.8, alpha=0.8)
    
    ax1.set_xlabel('Percentile', fontsize=12, fontweight='600')
    ax1.set_ylabel('Loss (%)', fontsize=12, fontweight='600')
    ax1.set_title('Tail Risk Percentiles', fontsize=13, fontweight='700')
    ax1.set_xticks(range(len(percentiles)))
    ax1.set_xticklabels([f'{p:.1f}%' for p in percentiles])
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height * 0.01,
                f'{loss:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='600')
    
    # Right plot: Risk progression
    ax2.plot(percentiles, losses, 'ro-', linewidth=2, markersize=6, alpha=0.8)
    ax2.fill_between(percentiles, losses, alpha=0.3, color='red')
    ax2.set_xlabel('Percentile', fontsize=12, fontweight='600')
    ax2.set_ylabel('Loss (%)', fontsize=12, fontweight='600')
    ax2.set_title('Tail Risk Progression', fontsize=13, fontweight='700')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_factor_attribution_plot(factor_contributions: Dict) -> str:
    """Generate factor risk attribution plot"""
    
    factors = list(factor_contributions.keys())
    
    # Extract different risk measures
    stand_alone = [factor_contributions[f]['stand_alone_risk'] for f in factors]
    component = [factor_contributions[f]['component_risk'] for f in factors]
    diversification = [factor_contributions[f]['diversification_benefit'] for f in factors]
    
    # Sort by component risk
    sorted_data = sorted(zip(factors, stand_alone, component, diversification), 
                        key=lambda x: abs(x[2]), reverse=True)
    factors_sorted, stand_alone_sorted, component_sorted, div_sorted = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(factors_sorted))
    width = 0.25
    
    bars1 = ax.bar(x_pos - width, stand_alone_sorted, width, label='Stand-alone Risk', 
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos, component_sorted, width, label='Component Risk', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x_pos + width, div_sorted, width, label='Diversification Benefit', 
                   color='forestgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Risk Factors', fontsize=12, fontweight='600')
    ax.set_ylabel('Risk Contribution', fontsize=12, fontweight='600')
    ax.set_title('Risk Attribution Analysis', fontsize=14, fontweight='700')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f.replace('_', ' ').title() for f in factors_sorted], 
                       rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_correlation_matrix_plot(portfolio_exposures: Dict) -> str:
    """Generate correlation matrix heatmap"""
    
    all_factors = list(RiskFactor)
    corr_matrix = get_risk_factor_correlations()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # Set ticks and labels
    factor_names = [f.value.replace('_', ' ').title() for f in all_factors]
    ax.set_xticks(np.arange(len(all_factors)))
    ax.set_yticks(np.arange(len(all_factors)))
    ax.set_xticklabels(factor_names, rotation=45, ha='right')
    ax.set_yticklabels(factor_names)
    
    # Add correlation values
    for i in range(len(all_factors)):
        for j in range(len(all_factors)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha="center", va="center",
                         color="white" if abs(corr_matrix[i, j]) > 0.5 else "black", fontweight='600')
    
    ax.set_title("Risk Factor Correlation Matrix\n(Stress Period Relationships)", 
                fontsize=14, fontweight='700', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_stress_surface_plot(portfolio_exposures: Dict, portfolio_value: float) -> str:
    """Generate 3D stress testing surface plot"""
    
    # For simplicity, show 2D contour plot of equity vs rates stress
    equity_shocks = np.linspace(-0.6, 0.2, 20)
    rate_shocks = np.linspace(-0.05, 0.08, 20)
    
    X, Y = np.meshgrid(equity_shocks, rate_shocks)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            shock_dict = {
                'equity': X[i, j],
                'interest_rate': Y[i, j],
                'credit_spread': 0,
                'fx_usd': 0,
                'commodity': 0,
                'volatility': 0,
                'real_estate': 0,
                'emerging_markets': 0
            }
            impact = calculate_scenario_impact(portfolio_exposures, shock_dict, portfolio_value,
                                             include_correlations=False, include_liquidity=False)
            Z[i, j] = impact['loss_percentage']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    contour = ax.contourf(X*100, Y*100, Z, levels=20, cmap='RdYlBu_r')
    contour_lines = ax.contour(X*100, Y*100, Z, levels=10, colors='black', alpha=0.4, linewidths=0.5)
    
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f%%')
    
    ax.set_xlabel('Equity Shock (%)', fontsize=12, fontweight='600')
    ax.set_ylabel('Interest Rate Shock (%)', fontsize=12, fontweight='600')
    ax.set_title('Stress Test Surface: Portfolio Loss vs Market Shocks', fontsize=14, fontweight='700')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Portfolio Loss (%)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    return _fig_to_base64(fig)

# =============================================================================
# Interpretation and Insights
# =============================================================================

def generate_stress_test_interpretation(
    scenario_analysis: Dict,
    risk_attribution: Dict,
    tail_scenarios: Dict,
    concentration_risk: Dict
) -> Dict[str, List[str]]:
    """Generate interpretation and recommendations"""
    
    key_findings = []
    recommendations = []
    risk_warnings = []
    
    loss_pct = scenario_analysis['loss_percentage']
    
    # Assess overall loss severity
    if loss_pct > 30:
        severity = "critical"
        finding_title = "Critical Portfolio Vulnerability"
        finding_desc = f"Portfolio faces extreme loss of {loss_pct:.1f}% under stress scenario. This represents catastrophic risk that requires immediate attention."
        risk_warnings.append("Portfolio may face insolvency under extreme stress conditions")
        recommendations.append("Immediately reduce risk exposures and increase hedging")
    elif loss_pct > 20:
        severity = "high"
        finding_title = "High Stress Loss Potential"
        finding_desc = f"Portfolio could lose {loss_pct:.1f}% under stress conditions. This exceeds typical risk tolerance levels."
        recommendations.append("Consider reducing leverage and diversifying risk exposures")
    elif loss_pct > 10:
        severity = "medium"
        finding_title = "Moderate Stress Impact" 
        finding_desc = f"Portfolio shows {loss_pct:.1f}% loss under stress. Within reasonable bounds but warrants monitoring."
        recommendations.append("Monitor risk concentrations and consider tactical hedging")
    else:
        severity = "low"
        finding_title = "Resilient Portfolio"
        finding_desc = f"Portfolio demonstrates good stress resilience with only {loss_pct:.1f}% loss."
    
    key_findings.append({
        "title": finding_title,
        "description": finding_desc,
        "severity": severity
    })
    
    # Analyze concentration risk
    if concentration_risk and 'concentration_metrics' in concentration_risk:
        hhi = concentration_risk['concentration_metrics'].get('herfindahl_index', 0)
        if hhi > 0.3:
            key_findings.append({
                "title": "High Portfolio Concentration",
                "description": f"Herfindahl index of {hhi:.3f} indicates significant concentration risk. Portfolio lacks sufficient diversification.",
                "severity": "high"
            })
            recommendations.append("Reduce position concentrations and increase diversification")
            risk_warnings.append("Concentrated positions amplify stress losses")
        elif hhi > 0.2:
            key_findings.append({
                "title": "Moderate Concentration Risk",
                "description": f"Portfolio shows moderate concentration with HHI of {hhi:.3f}. Some diversification improvements possible.",
                "severity": "medium"
            })
            recommendations.append("Consider rebalancing to reduce largest exposures")
    
    # Analyze tail risk
    if tail_scenarios and 'percentile_losses' in tail_scenarios:
        tail_99_9 = tail_scenarios['percentile_losses'].get('99.9', 0)
        if tail_99_9 > loss_pct * 2:
            key_findings.append({
                "title": "Significant Tail Risk",
                "description": f"99.9th percentile loss of {tail_99_9:.1f}% is much higher than scenario loss, indicating fat tail risk.",
                "severity": "high"
            })
            recommendations.append("Implement tail risk hedging strategies")
            risk_warnings.append("Extreme scenarios could result in losses far exceeding typical stress tests")
    
    # Analyze systematic risk
    if risk_attribution and 'portfolio_decomposition' in risk_attribution:
        sys_risk = risk_attribution['portfolio_decomposition'].get('systematic_risk', 0)
        total_risk = risk_attribution['portfolio_decomposition'].get('total_portfolio_risk', 1)
        
        if total_risk > 0:
            sys_pct = sys_risk / total_risk
            if sys_pct > 0.8:
                key_findings.append({
                    "title": "High Systematic Risk Exposure",
                    "description": f"Portfolio has {sys_pct:.1%} systematic risk, indicating high market beta and correlation.",
                    "severity": "medium"
                })
                recommendations.append("Add uncorrelated or negatively correlated assets")
    
    return {
        'key_findings': key_findings,
        'recommendations': recommendations,
        'risk_warnings': risk_warnings
    }

# =============================================================================
# Main API Endpoint
# =============================================================================

@router.post("/stress-test")
async def calculate_stress_test(request: StressTestRequest):
    """
    Comprehensive stress testing analysis
    
    Performs scenario analysis, sensitivity testing, reverse stress testing,
    concentration analysis, tail risk assessment, and risk attribution.
    """
    
    try:
        # Get scenario shocks
        if request.scenario == StressScenario.CUSTOM:
            if not request.custom_shocks:
                raise HTTPException(status_code=400, detail="Custom shocks required for custom scenario")
            scenario_shocks = request.custom_shocks
            scenario_name = "Custom Scenario"
        else:
            scenario_info = PREDEFINED_SCENARIOS[request.scenario]
            scenario_shocks = {k.value: v for k, v in scenario_info['shocks'].items()}
            scenario_name = scenario_info['name']
        
        # 1. Scenario Analysis
        scenario_analysis = calculate_scenario_impact(
            request.portfolio_exposures,
            scenario_shocks,
            request.portfolio_value,
            request.include_correlations,
            request.include_liquidity
        )
        scenario_analysis['scenario_name'] = scenario_name
        scenario_analysis['stress_parameters'] = scenario_shocks
        
        # 2. Sensitivity Analysis
        sensitivity_analysis = calculate_sensitivity_analysis(
            request.portfolio_exposures,
            request.portfolio_value
        )
        
        # 3. Reverse Stress Testing
        reverse_stress = calculate_reverse_stress_test(
            request.portfolio_exposures,
            request.portfolio_value
        )
        
        # 4. Concentration Risk Analysis
        concentration_risk = None
        if request.include_concentration:
            concentration_risk = calculate_concentration_risk(request.portfolio_exposures)
        
        # 5. Tail Risk Analysis
        tail_scenarios = calculate_tail_scenarios(
            request.portfolio_exposures,
            request.portfolio_value,
            request.confidence_level,
            request.num_simulations
        )
        
        # 6. Risk Attribution
        risk_attribution = calculate_risk_attribution(
            request.portfolio_exposures,
            request.portfolio_value
        )
        
        # 7. Generate Plots
        plots = {}
        
        # Scenario waterfall
        if scenario_analysis.get('risk_factor_contributions'):
            plots['scenario_waterfall'] = generate_scenario_waterfall_plot(
                scenario_analysis['risk_factor_contributions'], 
                request.portfolio_value
            )
        
        # Sensitivity heatmap
        if sensitivity_analysis.get('risk_factor_sensitivities'):
            plots['sensitivity_heatmap'] = generate_sensitivity_heatmap(
                sensitivity_analysis['risk_factor_sensitivities']
            )
        
        # Tail distribution
        if tail_scenarios.get('percentile_losses'):
            plots['tail_distribution'] = generate_tail_distribution_plot(
                tail_scenarios['percentile_losses']
            )
        
        # Factor attribution
        if risk_attribution.get('factor_contributions'):
            plots['factor_attribution'] = generate_factor_attribution_plot(
                risk_attribution['factor_contributions']
            )
        
        # Correlation matrix
        plots['correlation_matrix'] = generate_correlation_matrix_plot(
            request.portfolio_exposures
        )
        
        # Stress surface
        plots['stress_surface'] = generate_stress_surface_plot(
            request.portfolio_exposures, 
            request.portfolio_value
        )
        
        # 8. Generate Interpretation
        interpretation = generate_stress_test_interpretation(
            scenario_analysis,
            risk_attribution,
            tail_scenarios,
            concentration_risk
        )
        
        # 9. Compile Results
        result = {
            'scenario_analysis': _to_native_type(scenario_analysis),
            'sensitivity_analysis': _to_native_type(sensitivity_analysis),
            'reverse_stress': _to_native_type(reverse_stress),
            'concentration_risk': _to_native_type(concentration_risk),
            'tail_scenarios': _to_native_type(tail_scenarios),
            'risk_attribution': _to_native_type(risk_attribution),
            'plots': plots,
            'interpretation': interpretation,
            'parameters': {
                'scenario': request.scenario,
                'portfolio_value': request.portfolio_value,
                'time_horizon': request.time_horizon,
                'confidence_level': request.confidence_level,
                'include_correlations': request.include_correlations,
                'include_liquidity': request.include_liquidity,
                'include_concentration': request.include_concentration,
                'num_simulations': request.num_simulations
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress test calculation failed: {str(e)}")

# =============================================================================
# Additional Utility Endpoints
# =============================================================================

@router.get("/scenarios")
async def get_available_scenarios():
    """Get list of available predefined stress scenarios"""
    scenarios = []
    for scenario, info in PREDEFINED_SCENARIOS.items():
        scenarios.append({
            'value': scenario,
            'name': info['name'],
            'description': info['description'],
            'shocks': {k.value: v for k, v in info['shocks'].items()}
        })
    
    return {
        'scenarios': scenarios,
        'risk_factors': [factor.value for factor in RiskFactor]
    }

@router.get("/correlations")
async def get_risk_factor_correlations():
    """Get the risk factor correlation matrix"""
    corr_matrix = get_risk_factor_correlations()
    factors = [factor.value for factor in RiskFactor]
    
    return {
        'factors': factors,
        'correlation_matrix': _to_native_type(corr_matrix)
    }
