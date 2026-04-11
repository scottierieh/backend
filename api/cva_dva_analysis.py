"""
CVA/DVA (Credit & Debit Value Adjustment) Analysis Router for FastAPI
Using QuantLib for counterparty credit risk valuation
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
import QuantLib as ql
import io
import base64
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class ProductType(str, Enum):
    EUROPEAN_CALL = "european_call"
    EUROPEAN_PUT = "european_put"
    INTEREST_RATE_SWAP = "irs_payer"
    INTEREST_RATE_SWAP_RECEIVER = "irs_receiver"
    FORWARD = "forward"


class CvaRequest(BaseModel):
    """CVA/DVA calculation request parameters"""
    
    # Product parameters
    product_type: ProductType = Field(
        default=ProductType.EUROPEAN_CALL,
        description="Type of derivative product"
    )
    notional: float = Field(
        default=1000000.0,
        ge=1000,
        description="Notional amount"
    )
    spot_price: float = Field(
        default=100.0,
        ge=0.01,
        description="Current spot price (for options/forwards)"
    )
    strike_price: float = Field(
        default=100.0,
        ge=0.01,
        description="Strike price (for options/forwards)"
    )
    fixed_rate: float = Field(
        default=0.03,
        ge=0.0,
        le=0.5,
        description="Fixed rate for IRS"
    )
    
    # Market parameters
    volatility: float = Field(
        default=0.2,
        ge=0.01,
        le=2.0,
        description="Volatility (annualized)"
    )
    risk_free_rate: float = Field(
        default=0.05,
        ge=-0.1,
        le=0.5,
        description="Risk-free rate"
    )
    dividend_yield: float = Field(
        default=0.0,
        ge=0.0,
        le=0.2,
        description="Dividend yield"
    )
    maturity_years: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Time to maturity in years"
    )
    
    # Counterparty credit parameters
    hazard_rate_cp: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Counterparty hazard rate (intensity)"
    )
    recovery_rate_cp: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Counterparty recovery rate"
    )
    
    # Own credit parameters (for DVA)
    hazard_rate_own: float = Field(
        default=0.01,
        ge=0.0,
        le=0.5,
        description="Own hazard rate (intensity)"
    )
    recovery_rate_own: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Own recovery rate"
    )
    
    # Simulation parameters
    num_paths: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo paths"
    )
    time_steps: int = Field(
        default=52,
        ge=12,
        le=365,
        description="Number of time steps"
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
# QuantLib Setup Functions
# =============================================================================

def setup_ql_environment(valuation_date: ql.Date = None):
    """Setup QuantLib evaluation date and calendar"""
    if valuation_date is None:
        today = datetime.now()
        valuation_date = ql.Date(today.day, today.month, today.year)
    
    ql.Settings.instance().evaluationDate = valuation_date
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    
    return valuation_date, calendar, day_count


def build_yield_curve(rate: float, valuation_date: ql.Date, day_count: ql.DayCounter) -> ql.YieldTermStructureHandle:
    """Build flat yield curve using QuantLib"""
    flat_rate = ql.QuoteHandle(ql.SimpleQuote(rate))
    yield_curve = ql.FlatForward(valuation_date, flat_rate, day_count)
    return ql.YieldTermStructureHandle(yield_curve)


def build_hazard_rate_curve(hazard_rate: float, valuation_date: ql.Date, day_count: ql.DayCounter) -> ql.DefaultProbabilityTermStructureHandle:
    """
    Build hazard rate curve using QuantLib.
    Uses FlatHazardRate for constant hazard rate model.
    P(survival to t) = exp(-λt)
    """
    hazard_quote = ql.QuoteHandle(ql.SimpleQuote(hazard_rate))
    hazard_curve = ql.FlatHazardRate(valuation_date, hazard_quote, day_count)
    return ql.DefaultProbabilityTermStructureHandle(hazard_curve)


def build_volatility_surface(volatility: float, valuation_date: ql.Date, day_count: ql.DayCounter) -> ql.BlackVolTermStructureHandle:
    """Build flat volatility surface using QuantLib"""
    vol_quote = ql.QuoteHandle(ql.SimpleQuote(volatility))
    vol_surface = ql.BlackConstantVol(valuation_date, ql.NullCalendar(), vol_quote, day_count)
    return ql.BlackVolTermStructureHandle(vol_surface)


# =============================================================================
# Monte Carlo Path Generator using QuantLib
# =============================================================================

class QuantLibPathGenerator:
    """
    Generate Monte Carlo paths using QuantLib's path generator.
    Uses Geometric Brownian Motion process.
    """
    
    def __init__(self, spot: float, rate_handle: ql.YieldTermStructureHandle,
                 dividend_handle: ql.YieldTermStructureHandle,
                 vol_handle: ql.BlackVolTermStructureHandle,
                 maturity: float, time_steps: int):
        self.spot = spot
        self.maturity = maturity
        self.time_steps = time_steps
        self.dt = maturity / time_steps
        self.time_grid = np.linspace(0, maturity, time_steps + 1)
        
        # Store handles for later use
        self.rate_handle = rate_handle
        self.dividend_handle = dividend_handle
        self.vol_handle = vol_handle
        
        # Create Black-Scholes-Merton process
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        self.process = ql.BlackScholesMertonProcess(
            spot_handle, dividend_handle, rate_handle, vol_handle
        )
    
    def generate_paths(self, num_paths: int, seed: int = 42) -> np.ndarray:
        """Generate Monte Carlo paths using QuantLib"""
        # Create time grid for QuantLib
        times = ql.TimeGrid(self.maturity, self.time_steps)
        
        # Create path generator
        rng = ql.GaussianRandomSequenceGenerator(
            ql.UniformRandomSequenceGenerator(
                self.time_steps,
                ql.UniformRandomGenerator(seed)
            )
        )
        
        path_generator = ql.GaussianPathGenerator(
            self.process, self.maturity, self.time_steps, rng, False
        )
        
        # Generate paths
        paths = np.zeros((num_paths, self.time_steps + 1))
        
        for i in range(num_paths):
            sample_path = path_generator.next()
            path = sample_path.value()
            for j in range(self.time_steps + 1):
                paths[i, j] = path[j]
        
        return paths


# =============================================================================
# Exposure Calculator using QuantLib Pricing
# =============================================================================

class QuantLibExposureCalculator:
    """
    Calculate Expected Exposure (EE) and Expected Negative Exposure (ENE)
    using QuantLib pricing engines.
    """
    
    @staticmethod
    def european_option_exposure(
        paths: np.ndarray,
        time_grid: np.ndarray,
        strike: float,
        notional: float,
        is_call: bool,
        rate_handle: ql.YieldTermStructureHandle,
        vol_handle: ql.BlackVolTermStructureHandle,
        dividend_handle: ql.YieldTermStructureHandle,
        valuation_date: ql.Date,
        maturity_date: ql.Date
    ) -> np.ndarray:
        """
        Calculate exposure profiles for European options using QuantLib BS pricing.
        """
        num_paths, num_times = paths.shape
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        
        exposures = np.zeros((num_paths, num_times))
        
        option_type = ql.Option.Call if is_call else ql.Option.Put
        
        for t_idx, t in enumerate(time_grid):
            # Calculate valuation date for this time point
            days_forward = int(t * 365)
            current_date = valuation_date + ql.Period(days_forward, ql.Days)
            
            # Skip if past maturity
            if current_date >= maturity_date:
                # At or past maturity - intrinsic value only
                S = paths[:, t_idx]
                if is_call:
                    exposures[:, t_idx] = np.maximum(S - strike, 0) * notional
                else:
                    exposures[:, t_idx] = np.maximum(strike - S, 0) * notional
                continue
            
            # Time to maturity
            tau = day_count.yearFraction(current_date, maturity_date)
            
            if tau < 1e-6:
                S = paths[:, t_idx]
                if is_call:
                    exposures[:, t_idx] = np.maximum(S - strike, 0) * notional
                else:
                    exposures[:, t_idx] = np.maximum(strike - S, 0) * notional
                continue
            
            # For each path, price the option using Black-Scholes formula
            S = paths[:, t_idx]
            r = rate_handle.zeroRate(tau, ql.Continuous).rate()
            q = dividend_handle.zeroRate(tau, ql.Continuous).rate()
            sigma = vol_handle.blackVol(tau, strike)
            
            # Black-Scholes formula
            d1 = (np.log(S / strike) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)
            
            if is_call:
                value = S * np.exp(-q * tau) * norm.cdf(d1) - strike * np.exp(-r * tau) * norm.cdf(d2)
            else:
                value = strike * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
            
            exposures[:, t_idx] = value * notional
        
        return exposures
    
    @staticmethod
    def forward_exposure(
        paths: np.ndarray,
        time_grid: np.ndarray,
        forward_price: float,
        notional: float,
        rate_handle: ql.YieldTermStructureHandle
    ) -> np.ndarray:
        """Calculate exposure profiles for forward contracts"""
        num_paths, num_times = paths.shape
        maturity = time_grid[-1]
        
        exposures = np.zeros((num_paths, num_times))
        
        for t_idx, t in enumerate(time_grid):
            tau = maturity - t
            S = paths[:, t_idx]
            
            if tau > 1e-6:
                r = rate_handle.zeroRate(tau, ql.Continuous).rate()
                # Forward value = S - K * exp(-r * tau)
                exposures[:, t_idx] = (S - forward_price * np.exp(-r * tau)) * notional
            else:
                exposures[:, t_idx] = (S - forward_price) * notional
        
        return exposures
    
    @staticmethod
    def irs_exposure(
        rate_paths: np.ndarray,
        time_grid: np.ndarray,
        fixed_rate: float,
        notional: float,
        is_payer: bool,
        valuation_date: ql.Date,
        maturity_years: float
    ) -> np.ndarray:
        """
        Calculate exposure profiles for Interest Rate Swaps.
        Uses simplified annuity-based valuation.
        """
        num_paths, num_times = rate_paths.shape
        
        exposures = np.zeros((num_paths, num_times))
        
        for t_idx, t in enumerate(time_grid):
            tau = maturity_years - t
            if tau < 1e-6:
                continue
            
            floating_rate = rate_paths[:, t_idx]
            
            # Simplified swap valuation using annuity
            # Annuity approximation for remaining life
            annuity = tau  # Simplified continuous approximation
            
            if is_payer:
                # Payer: pay fixed, receive float
                swap_value = notional * (floating_rate - fixed_rate) * annuity
            else:
                # Receiver: receive fixed, pay float
                swap_value = notional * (fixed_rate - floating_rate) * annuity
            
            exposures[:, t_idx] = swap_value
        
        return exposures


# =============================================================================
# CVA/DVA Engine using QuantLib
# =============================================================================

class QuantLibCvaEngine:
    """
    CVA/DVA calculation engine using QuantLib curves.
    
    CVA = (1 - R_cp) * Σ EE(t_i) * PD_cp(t_{i-1}, t_i) * DF(t_i)
    DVA = (1 - R_own) * Σ ENE(t_i) * PD_own(t_{i-1}, t_i) * DF(t_i)
    """
    
    def __init__(self,
                 yield_curve_handle: ql.YieldTermStructureHandle,
                 hazard_curve_cp_handle: ql.DefaultProbabilityTermStructureHandle,
                 hazard_curve_own_handle: ql.DefaultProbabilityTermStructureHandle,
                 recovery_rate_cp: float,
                 recovery_rate_own: float,
                 valuation_date: ql.Date):
        self.yield_curve = yield_curve_handle
        self.hazard_curve_cp = hazard_curve_cp_handle
        self.hazard_curve_own = hazard_curve_own_handle
        self.recovery_rate_cp = recovery_rate_cp
        self.recovery_rate_own = recovery_rate_own
        self.valuation_date = valuation_date
    
    def calculate(self, exposures: np.ndarray, time_grid: np.ndarray) -> Dict[str, Any]:
        """
        Calculate CVA, DVA, and related metrics using QuantLib curves.
        """
        num_paths, num_times = exposures.shape
        
        # Calculate exposure profiles
        positive_exposure = np.maximum(exposures, 0)
        negative_exposure = np.maximum(-exposures, 0)
        
        # Expected Exposure (EE) profile
        ee_profile = np.mean(positive_exposure, axis=0)
        
        # Expected Negative Exposure (ENE) profile
        ene_profile = np.mean(negative_exposure, axis=0)
        
        # Potential Future Exposure (PFE) - 97.5% quantile
        pfe_profile = np.percentile(positive_exposure, 97.5, axis=0)
        
        # Expected Positive Exposure (EPE)
        epe = np.trapz(ee_profile, time_grid) / time_grid[-1] if time_grid[-1] > 0 else 0
        
        # Calculate CVA using QuantLib survival probabilities
        cva = 0.0
        lgd_cp = 1.0 - self.recovery_rate_cp
        
        for i in range(1, num_times):
            t_prev = time_grid[i - 1]
            t_curr = time_grid[i]
            
            # Midpoint EE
            ee_mid = (ee_profile[i - 1] + ee_profile[i]) / 2
            
            # Get survival probabilities from QuantLib curve
            surv_prev = self.hazard_curve_cp.survivalProbability(t_prev)
            surv_curr = self.hazard_curve_cp.survivalProbability(t_curr)
            
            # Default probability in period
            pd = surv_prev - surv_curr
            
            # Discount factor from QuantLib yield curve
            df = self.yield_curve.discount((t_prev + t_curr) / 2)
            
            cva += lgd_cp * ee_mid * pd * df
        
        # Calculate DVA using QuantLib survival probabilities
        dva = 0.0
        lgd_own = 1.0 - self.recovery_rate_own
        
        for i in range(1, num_times):
            t_prev = time_grid[i - 1]
            t_curr = time_grid[i]
            
            # Midpoint ENE
            ene_mid = (ene_profile[i - 1] + ene_profile[i]) / 2
            
            # Get survival probabilities from QuantLib curve
            surv_prev = self.hazard_curve_own.survivalProbability(t_prev)
            surv_curr = self.hazard_curve_own.survivalProbability(t_curr)
            
            # Default probability in period
            pd = surv_prev - surv_curr
            
            # Discount factor
            df = self.yield_curve.discount((t_prev + t_curr) / 2)
            
            dva += lgd_own * ene_mid * pd * df
        
        # Risk-free NPV
        base_npv = np.mean(exposures[:, 0])
        
        # Bilateral CVA (XVA)
        xva = cva - dva
        
        # Adjusted NPV
        adjusted_npv = base_npv - cva + dva
        
        return {
            'cva': _to_native_type(cva),
            'dva': _to_native_type(dva),
            'xva': _to_native_type(xva),
            'base_npv': _to_native_type(base_npv),
            'adjusted_npv': _to_native_type(adjusted_npv),
            'ee_profile': [_to_native_type(x) for x in ee_profile],
            'ene_profile': [_to_native_type(x) for x in ene_profile],
            'pfe_profile': [_to_native_type(x) for x in pfe_profile],
            'epe': _to_native_type(epe),
            'time_grid': [_to_native_type(x) for x in time_grid],
            'max_ee': _to_native_type(np.max(ee_profile)),
            'max_pfe': _to_native_type(np.max(pfe_profile)),
            'exposure_paths': exposures[:min(100, num_paths), :].tolist()
        }


# =============================================================================
# Plot Generation
# =============================================================================

def generate_exposure_profile_plot(result: Dict) -> str:
    """Generate exposure profile plot with EE, ENE, and PFE"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_grid = result['time_grid']
    ee_profile = result['ee_profile']
    ene_profile = result['ene_profile']
    pfe_profile = result['pfe_profile']
    
    ax.fill_between(time_grid, 0, ee_profile, alpha=0.3, color='#ef4444', label='Expected Exposure (EE)')
    ax.fill_between(time_grid, 0, [-x for x in ene_profile], alpha=0.3, color='#3b82f6', label='Expected Negative Exposure (ENE)')
    
    ax.plot(time_grid, ee_profile, 'r-', linewidth=2, label='EE Profile')
    ax.plot(time_grid, [-x for x in ene_profile], 'b-', linewidth=2, label='ENE Profile')
    ax.plot(time_grid, pfe_profile, 'g--', linewidth=2, label='PFE (97.5%)')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Exposure ($)')
    ax.set_title('Expected Exposure Profiles Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_exposure_distribution_plot(result: Dict) -> str:
    """Generate exposure distribution at selected time points"""
    exposure_paths = np.array(result['exposure_paths'])
    time_grid = np.array(result['time_grid'])
    
    indices = [len(time_grid) // 4, len(time_grid) // 2, 3 * len(time_grid) // 4, -1]
    time_points = [time_grid[i] for i in indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for idx, t in enumerate(time_points[:4]):
        t_idx = np.argmin(np.abs(time_grid - t))
        exposures = exposure_paths[:, t_idx]
        
        ax = axes[idx]
        ax.hist(exposures, bins=30, density=True, alpha=0.7, color='#6366f1', edgecolor='white')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=np.mean(exposures), color='green', linestyle='-', linewidth=2, label=f'Mean: ${np.mean(exposures):,.0f}')
        
        ax.set_xlabel('Exposure ($)')
        ax.set_ylabel('Density')
        ax.set_title(f't = {t:.2f} years')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_survival_probability_plot(
    hazard_curve_cp: ql.DefaultProbabilityTermStructureHandle,
    hazard_curve_own: ql.DefaultProbabilityTermStructureHandle,
    maturity: float,
    hazard_rate_cp: float,
    hazard_rate_own: float
) -> str:
    """Generate survival probability curves using QuantLib"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    time_grid = np.linspace(0.01, maturity, 100)
    
    surv_cp = [hazard_curve_cp.survivalProbability(t) for t in time_grid]
    surv_own = [hazard_curve_own.survivalProbability(t) for t in time_grid]
    
    ax.plot(time_grid, surv_cp, 'r-', linewidth=2, label=f'Counterparty (λ={hazard_rate_cp})')
    ax.plot(time_grid, surv_own, 'b-', linewidth=2, label=f'Own (λ={hazard_rate_own})')
    
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Default Probability Curves (QuantLib)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_path_simulation_plot(result: Dict) -> str:
    """Generate sample path visualization"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    exposure_paths = np.array(result['exposure_paths'])
    time_grid = result['time_grid']
    
    for i in range(min(50, len(exposure_paths))):
        ax.plot(time_grid, exposure_paths[i], alpha=0.2, color='#6366f1', linewidth=0.5)
    
    ee_profile = result['ee_profile']
    ene_profile = result['ene_profile']
    
    ax.plot(time_grid, ee_profile, 'r-', linewidth=2.5, label='Expected Exposure (EE)')
    ax.plot(time_grid, [-x for x in ene_profile], 'b-', linewidth=2.5, label='Expected Negative Exposure')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Exposure ($)')
    ax.set_title('Monte Carlo Simulation Paths (QuantLib)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_cva_breakdown_plot(result: Dict) -> str:
    """Generate CVA/DVA breakdown waterfall chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Risk-Free NPV', 'CVA Impact', 'DVA Benefit', 'Adjusted NPV']
    values = [
        result['base_npv'],
        -result['cva'],
        result['dva'],
        result['adjusted_npv']
    ]
    
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#6366f1']
    
    running_total = 0
    for i, (label, value) in enumerate(zip(labels, values)):
        if i == 0 or i == len(labels) - 1:
            bottom = 0
            height = value
        else:
            bottom = running_total
            height = value
        
        ax.bar(label, height, bottom=bottom, color=colors[i], edgecolor='white', linewidth=2)
        
        if height >= 0:
            ax.text(i, bottom + height + max(abs(v) for v in values) * 0.02, f'${value:,.0f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(i, bottom + height - max(abs(v) for v in values) * 0.02, f'${value:,.0f}', 
                   ha='center', va='top', fontsize=10, fontweight='bold')
        
        if i > 0 and i < len(labels) - 1:
            running_total += value
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Value ($)')
    ax.set_title('CVA/DVA Waterfall Analysis')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Interpretation Generator
# =============================================================================

def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of CVA/DVA results"""
    key_insights = []
    
    cva = result['cva']
    base_npv = result['base_npv']
    cva_pct = abs(cva / base_npv * 100) if base_npv != 0 else 0
    
    if cva_pct > 10:
        key_insights.append({
            'title': 'High Counterparty Risk',
            'description': f"CVA represents {cva_pct:.1f}% of the base NPV. Significant counterparty credit risk.",
            'status': 'warning'
        })
    elif cva_pct > 3:
        key_insights.append({
            'title': 'Moderate Counterparty Risk',
            'description': f"CVA represents {cva_pct:.1f}% of the base NPV. Material credit risk consideration.",
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'Low Counterparty Risk',
            'description': f"CVA represents only {cva_pct:.1f}% of the base NPV. Well contained risk.",
            'status': 'positive'
        })
    
    dva = result['dva']
    if dva > cva:
        key_insights.append({
            'title': 'DVA Exceeds CVA',
            'description': f"DVA (${dva:,.0f}) > CVA (${cva:,.0f}). Own credit benefits the trade.",
            'status': 'positive'
        })
    
    max_ee = result['max_ee']
    max_pfe = result['max_pfe']
    key_insights.append({
        'title': 'Peak Exposure Analysis',
        'description': f"Max EE: ${max_ee:,.0f}. PFE (97.5%): ${max_pfe:,.0f}.",
        'status': 'neutral'
    })
    
    xva = result['xva']
    key_insights.append({
        'title': 'Net XVA Impact',
        'description': f"Bilateral CVA impact: ${xva:,.0f}. Adjusted NPV: ${result['adjusted_npv']:,.0f}.",
        'status': 'neutral'
    })
    
    recommendations = []
    if params['hazard_rate_cp'] > 0.03:
        recommendations.append("Consider requesting collateral (CSA) to mitigate counterparty risk.")
    if cva_pct > 5:
        recommendations.append("Evaluate netting agreements to reduce gross exposure.")
    if max_pfe > 2 * max_ee:
        recommendations.append("High tail risk. Consider exposure limits or hedging.")
    if params['num_paths'] < 5000:
        recommendations.append("Increase simulation paths for accuracy.")
    if not recommendations:
        recommendations.append("Current parameters are well-balanced.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/cva-dva")
async def calculate_cva_dva(request: CvaRequest) -> Dict[str, Any]:
    """
    Calculate CVA/DVA using QuantLib for counterparty credit risk.
    
    Implementation Steps (QuantLib):
    1. Build yield curve (FlatForward)
    2. Build hazard rate curves (FlatHazardRate) 
    3. Generate paths using GaussianPathGenerator
    4. Calculate exposures using Black-Scholes pricing
    5. Compute CVA/DVA using survival probabilities
    """
    try:
        # Step 1: Setup QuantLib environment
        valuation_date, calendar, day_count = setup_ql_environment()
        
        # Step 2: Build QuantLib curves
        yield_curve_handle = build_yield_curve(request.risk_free_rate, valuation_date, day_count)
        dividend_curve_handle = build_yield_curve(request.dividend_yield, valuation_date, day_count)
        vol_handle = build_volatility_surface(request.volatility, valuation_date, day_count)
        
        # Step 3: Build hazard rate curves (Default Probability Curves)
        hazard_curve_cp = build_hazard_rate_curve(request.hazard_rate_cp, valuation_date, day_count)
        hazard_curve_own = build_hazard_rate_curve(request.hazard_rate_own, valuation_date, day_count)
        
        # Calculate maturity date
        maturity_days = int(request.maturity_years * 365)
        maturity_date = valuation_date + ql.Period(maturity_days, ql.Days)
        
        # Step 4: Generate paths and calculate exposures
        if request.product_type in [ProductType.EUROPEAN_CALL, ProductType.EUROPEAN_PUT]:
            # Create path generator
            path_gen = QuantLibPathGenerator(
                spot=request.spot_price,
                rate_handle=yield_curve_handle,
                dividend_handle=dividend_curve_handle,
                vol_handle=vol_handle,
                maturity=request.maturity_years,
                time_steps=request.time_steps
            )
            paths = path_gen.generate_paths(request.num_paths)
            
            is_call = request.product_type == ProductType.EUROPEAN_CALL
            exposures = QuantLibExposureCalculator.european_option_exposure(
                paths=paths,
                time_grid=path_gen.time_grid,
                strike=request.strike_price,
                notional=request.notional,
                is_call=is_call,
                rate_handle=yield_curve_handle,
                vol_handle=vol_handle,
                dividend_handle=dividend_curve_handle,
                valuation_date=valuation_date,
                maturity_date=maturity_date
            )
            time_grid = path_gen.time_grid
            
        elif request.product_type == ProductType.FORWARD:
            path_gen = QuantLibPathGenerator(
                spot=request.spot_price,
                rate_handle=yield_curve_handle,
                dividend_handle=dividend_curve_handle,
                vol_handle=vol_handle,
                maturity=request.maturity_years,
                time_steps=request.time_steps
            )
            paths = path_gen.generate_paths(request.num_paths)
            
            exposures = QuantLibExposureCalculator.forward_exposure(
                paths=paths,
                time_grid=path_gen.time_grid,
                forward_price=request.strike_price,
                notional=request.notional,
                rate_handle=yield_curve_handle
            )
            time_grid = path_gen.time_grid
            
        else:
            # IRS - use simplified rate paths
            np.random.seed(42)
            rate_paths = np.zeros((request.num_paths, request.time_steps + 1))
            rate_paths[:, 0] = request.risk_free_rate
            
            dt = request.maturity_years / request.time_steps
            kappa, theta, sigma = 0.1, request.risk_free_rate, 0.01
            
            for i in range(request.time_steps):
                dW = np.random.standard_normal(request.num_paths)
                dr = kappa * (theta - rate_paths[:, i]) * dt + sigma * np.sqrt(dt) * dW
                rate_paths[:, i + 1] = np.maximum(rate_paths[:, i] + dr, 0.0001)
            
            time_grid = np.linspace(0, request.maturity_years, request.time_steps + 1)
            is_payer = request.product_type == ProductType.INTEREST_RATE_SWAP
            
            exposures = QuantLibExposureCalculator.irs_exposure(
                rate_paths=rate_paths,
                time_grid=time_grid,
                fixed_rate=request.fixed_rate,
                notional=request.notional,
                is_payer=is_payer,
                valuation_date=valuation_date,
                maturity_years=request.maturity_years
            )
        
        # Step 5: Calculate CVA/DVA using QuantLib Engine
        cva_engine = QuantLibCvaEngine(
            yield_curve_handle=yield_curve_handle,
            hazard_curve_cp_handle=hazard_curve_cp,
            hazard_curve_own_handle=hazard_curve_own,
            recovery_rate_cp=request.recovery_rate_cp,
            recovery_rate_own=request.recovery_rate_own,
            valuation_date=valuation_date
        )
        
        result = cva_engine.calculate(exposures, time_grid)
        
        # Generate plots
        plots = {
            'exposure_profile': generate_exposure_profile_plot(result),
            'exposure_distribution': generate_exposure_distribution_plot(result),
            'survival_probability': generate_survival_probability_plot(
                hazard_curve_cp, hazard_curve_own, 
                request.maturity_years, request.hazard_rate_cp, request.hazard_rate_own
            ),
            'path_simulation': generate_path_simulation_plot(result),
            'cva_breakdown': generate_cva_breakdown_plot(result)
        }
        
        # Generate interpretation
        params = {
            'product_type': request.product_type.value,
            'hazard_rate_cp': request.hazard_rate_cp,
            'hazard_rate_own': request.hazard_rate_own,
            'num_paths': request.num_paths
        }
        interpretation = generate_interpretation(result, params)
        
        # Surface data for interactive visualization
        surface_data = {
            'time': result['time_grid'],
            'ee': result['ee_profile'],
            'ene': result['ene_profile'],
            'pfe': result['pfe_profile']
        }
        
        return {
            'cva': result['cva'],
            'dva': result['dva'],
            'xva': result['xva'],
            'base_npv': result['base_npv'],
            'adjusted_npv': result['adjusted_npv'],
            'ee_profile': result['ee_profile'],
            'ene_profile': result['ene_profile'],
            'pfe_profile': result['pfe_profile'],
            'epe': result['epe'],
            'max_ee': result['max_ee'],
            'max_pfe': result['max_pfe'],
            'time_grid': result['time_grid'],
            'exposure_paths': result['exposure_paths'],
            'surface_data': surface_data,
            'plots': plots,
            'interpretation': interpretation,
            'parameters': {
                'product_type': request.product_type.value,
                'notional': request.notional,
                'maturity_years': request.maturity_years,
                'hazard_rate_cp': request.hazard_rate_cp,
                'recovery_rate_cp': request.recovery_rate_cp,
                'hazard_rate_own': request.hazard_rate_own,
                'recovery_rate_own': request.recovery_rate_own,
                'num_paths': request.num_paths,
                'time_steps': request.time_steps
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CVA/DVA calculation failed: {str(e)}")
