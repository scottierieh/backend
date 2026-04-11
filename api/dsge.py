"""
Dynamic Stochastic General Equilibrium (DSGE) Simulation Router for FastAPI
Simulate macroeconomic dynamics using a basic New Keynesian DSGE model
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import linalg
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class DSGERequest(BaseModel):
    # Preference parameters
    beta: float = 0.99  # Discount factor
    sigma: float = 1.0  # Inverse elasticity of intertemporal substitution
    phi: float = 1.0  # Inverse Frisch elasticity of labor supply
    
    # Price stickiness
    theta: float = 0.75  # Calvo price stickiness (fraction not adjusting)
    
    # Monetary policy (Taylor rule)
    phi_pi: float = 1.5  # Response to inflation
    phi_y: float = 0.5  # Response to output gap
    rho_r: float = 0.7  # Interest rate smoothing
    
    # Shock persistence
    rho_a: float = 0.9  # Technology shock persistence
    rho_g: float = 0.8  # Government spending shock persistence
    rho_m: float = 0.5  # Monetary policy shock persistence
    
    # Shock standard deviations
    sigma_a: float = 0.01  # Technology shock std
    sigma_g: float = 0.01  # Government spending shock std
    sigma_m: float = 0.0025  # Monetary policy shock std
    
    # Simulation settings
    periods: int = 100  # Number of simulation periods
    n_simulations: int = 1  # Number of Monte Carlo simulations
    seed: Optional[int] = 42  # Random seed


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


class NewKeynesianDSGE:
    """
    Basic 3-equation New Keynesian DSGE Model:
    1. IS Curve (Euler equation)
    2. Phillips Curve (NKPC)
    3. Taylor Rule (Monetary policy)
    """
    
    def __init__(self, params: DSGERequest):
        self.beta = params.beta
        self.sigma = params.sigma
        self.phi = params.phi
        self.theta = params.theta
        self.phi_pi = params.phi_pi
        self.phi_y = params.phi_y
        self.rho_r = params.rho_r
        self.rho_a = params.rho_a
        self.rho_g = params.rho_g
        self.rho_m = params.rho_m
        self.sigma_a = params.sigma_a
        self.sigma_g = params.sigma_g
        self.sigma_m = params.sigma_m
        
        # Derived parameters
        self.kappa = self._compute_kappa()
        
    def _compute_kappa(self):
        """Compute slope of Phillips curve"""
        # kappa = (1-theta)(1-beta*theta)/theta * (sigma + phi)
        return ((1 - self.theta) * (1 - self.beta * self.theta) / self.theta) * (self.sigma + self.phi)
    
    def get_state_space_matrices(self):
        """
        Construct state-space representation of the linearized model.
        State vector: [y_t, pi_t, i_t, a_t, g_t, m_t]
        y_t: output gap, pi_t: inflation, i_t: interest rate
        a_t: technology shock, g_t: govt spending shock, m_t: monetary shock
        """
        # System: A * E[x_{t+1}] = B * x_t + C * epsilon_t
        
        # Simplified reduced form solution using method of undetermined coefficients
        # For the basic NK model, we can derive analytical solutions
        
        n_states = 6
        n_shocks = 3
        
        # Transition matrix (approximate solution)
        # This is a simplified linear approximation
        
        # Denominator for NK system
        denom = self.sigma + self.phi_y + self.kappa * self.phi_pi / (1 - self.beta * self.rho_a)
        
        # Policy function coefficients (simplified)
        T = np.zeros((n_states, n_states))
        
        # Output gap dynamics
        T[0, 3] = (1 + self.phi) / denom  # Response to tech shock
        T[0, 4] = self.sigma / denom  # Response to govt shock
        T[0, 5] = -1 / denom  # Response to monetary shock
        
        # Inflation dynamics
        T[1, 0] = self.kappa  # From output gap
        T[1, 1] = self.beta  # Persistence
        T[1, 3] = self.kappa * (1 + self.phi) / denom
        
        # Interest rate (Taylor rule)
        T[2, 1] = (1 - self.rho_r) * self.phi_pi
        T[2, 0] = (1 - self.rho_r) * self.phi_y
        T[2, 2] = self.rho_r
        T[2, 5] = 1  # Monetary shock
        
        # Shock persistence
        T[3, 3] = self.rho_a
        T[4, 4] = self.rho_g
        T[5, 5] = self.rho_m
        
        # Shock impact matrix
        R = np.zeros((n_states, n_shocks))
        R[3, 0] = 1  # Tech shock
        R[4, 1] = 1  # Govt shock
        R[5, 2] = 1  # Monetary shock
        
        return T, R
    
    def simulate(self, periods: int, n_sims: int = 1, seed: int = None):
        """Simulate the DSGE model"""
        if seed is not None:
            np.random.seed(seed)
        
        T, R = self.get_state_space_matrices()
        n_states = T.shape[0]
        n_shocks = R.shape[1]
        
        # Shock standard deviations
        shock_stds = np.array([self.sigma_a, self.sigma_g, self.sigma_m])
        
        # Storage for all simulations
        all_sims = []
        
        for sim in range(n_sims):
            # Initialize states
            states = np.zeros((periods + 1, n_states))
            shocks = np.zeros((periods, n_shocks))
            
            # Generate shocks
            for i, std in enumerate(shock_stds):
                shocks[:, i] = np.random.normal(0, std, periods)
            
            # Simulate
            for t in range(periods):
                states[t + 1] = T @ states[t] + R @ shocks[t]
            
            # Store results
            sim_data = {
                'output_gap': states[1:, 0],
                'inflation': states[1:, 1],
                'interest_rate': states[1:, 2],
                'tech_shock': states[1:, 3],
                'govt_shock': states[1:, 4],
                'monetary_shock': states[1:, 5],
                'shocks': shocks
            }
            all_sims.append(sim_data)
        
        return all_sims
    
    def compute_irfs(self, periods: int = 40):
        """Compute Impulse Response Functions"""
        T, R = self.get_state_space_matrices()
        n_states = T.shape[0]
        n_shocks = R.shape[1]
        
        shock_names = ['Technology', 'Government Spending', 'Monetary Policy']
        var_names = ['Output Gap', 'Inflation', 'Interest Rate']
        
        irfs = {}
        
        for shock_idx, shock_name in enumerate(shock_names):
            irf_data = {}
            
            # Initialize with unit shock
            state = np.zeros(n_states)
            shock = np.zeros(n_shocks)
            shock[shock_idx] = 1  # Unit shock
            
            responses = np.zeros((periods, n_states))
            
            # Initial impact
            state = R @ shock
            responses[0] = state
            
            # Propagate
            for t in range(1, periods):
                state = T @ state
                responses[t] = state
            
            for var_idx, var_name in enumerate(var_names):
                irf_data[var_name] = responses[:, var_idx].tolist()
            
            irfs[shock_name] = irf_data
        
        return irfs
    
    def compute_moments(self, simulations: List[Dict]) -> Dict[str, Any]:
        """Compute theoretical and simulated moments"""
        moments = {}
        
        # Variables to analyze
        var_names = ['output_gap', 'inflation', 'interest_rate']
        var_labels = ['Output Gap', 'Inflation', 'Interest Rate']
        
        for var, label in zip(var_names, var_labels):
            all_values = np.concatenate([sim[var] for sim in simulations])
            
            moments[label] = {
                'mean': _to_native_type(np.mean(all_values)),
                'std': _to_native_type(np.std(all_values)),
                'min': _to_native_type(np.min(all_values)),
                'max': _to_native_type(np.max(all_values)),
                'autocorr': _to_native_type(np.corrcoef(all_values[:-1], all_values[1:])[0, 1])
            }
        
        # Cross-correlations
        if len(simulations) > 0:
            y = simulations[0]['output_gap']
            pi = simulations[0]['inflation']
            i = simulations[0]['interest_rate']
            
            moments['correlations'] = {
                'output_inflation': _to_native_type(np.corrcoef(y, pi)[0, 1]),
                'output_interest': _to_native_type(np.corrcoef(y, i)[0, 1]),
                'inflation_interest': _to_native_type(np.corrcoef(pi, i)[0, 1])
            }
        
        return moments


def generate_simulation_plot(simulations: List[Dict], periods: int) -> str:
    """Generate time series plot of simulation"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    time = np.arange(1, periods + 1)
    
    # Use first simulation for plotting
    sim = simulations[0]
    
    # Output Gap
    axes[0].plot(time, sim['output_gap'] * 100, color='#3b82f6', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[0].fill_between(time, 0, sim['output_gap'] * 100, alpha=0.3, color='#3b82f6')
    axes[0].set_ylabel('Output Gap (%)', fontsize=11)
    axes[0].set_title('DSGE Simulation: Output Gap', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Inflation
    axes[1].plot(time, sim['inflation'] * 100, color='#ef4444', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[1].fill_between(time, 0, sim['inflation'] * 100, alpha=0.3, color='#ef4444')
    axes[1].set_ylabel('Inflation (%)', fontsize=11)
    axes[1].set_title('Inflation Rate', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # Interest Rate
    axes[2].plot(time, sim['interest_rate'] * 100, color='#22c55e', linewidth=2)
    axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[2].fill_between(time, 0, sim['interest_rate'] * 100, alpha=0.3, color='#22c55e')
    axes[2].set_ylabel('Interest Rate (%)', fontsize=11)
    axes[2].set_xlabel('Period', fontsize=11)
    axes[2].set_title('Nominal Interest Rate', fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_irf_plot(irfs: Dict, shock_name: str) -> str:
    """Generate IRF plot for a specific shock"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    irf_data = irfs[shock_name]
    periods = len(irf_data['Output Gap'])
    time = np.arange(periods)
    
    colors = ['#3b82f6', '#ef4444', '#22c55e']
    var_names = ['Output Gap', 'Inflation', 'Interest Rate']
    
    for ax, var, color in zip(axes, var_names, colors):
        values = np.array(irf_data[var]) * 100  # Convert to percentage
        ax.plot(time, values, color=color, linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.fill_between(time, 0, values, alpha=0.3, color=color)
        ax.set_xlabel('Periods', fontsize=10)
        ax.set_ylabel('% Deviation', fontsize=10)
        ax.set_title(f'{var}', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    fig.suptitle(f'Impulse Response to {shock_name} Shock', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_all_irfs_plot(irfs: Dict) -> str:
    """Generate combined IRF plot for all shocks"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    shock_names = ['Technology', 'Government Spending', 'Monetary Policy']
    var_names = ['Output Gap', 'Inflation', 'Interest Rate']
    colors = ['#3b82f6', '#ef4444', '#22c55e']
    
    for row, shock_name in enumerate(shock_names):
        irf_data = irfs[shock_name]
        periods = len(irf_data['Output Gap'])
        time = np.arange(periods)
        
        for col, (var, color) in enumerate(zip(var_names, colors)):
            ax = axes[row, col]
            values = np.array(irf_data[var]) * 100
            ax.plot(time, values, color=color, linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.fill_between(time, 0, values, alpha=0.3, color=color)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            if row == 0:
                ax.set_title(var, fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{shock_name}\nShock', fontsize=10, fontweight='bold')
            if row == 2:
                ax.set_xlabel('Periods', fontsize=10)
    
    fig.suptitle('Impulse Response Functions', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_phase_diagram(simulations: List[Dict]) -> str:
    """Generate phase diagram (output gap vs inflation)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sim = simulations[0]
    y = np.array(sim['output_gap']) * 100
    pi = np.array(sim['inflation']) * 100
    
    # Scatter with color gradient for time
    scatter = ax.scatter(y, pi, c=np.arange(len(y)), cmap='viridis', s=30, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Period')
    
    # Connect points
    ax.plot(y, pi, color='gray', alpha=0.3, linewidth=0.5)
    
    # Add axes
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Mark start and end
    ax.scatter([y[0]], [pi[0]], color='green', s=150, marker='o', zorder=5, label='Start')
    ax.scatter([y[-1]], [pi[-1]], color='red', s=150, marker='s', zorder=5, label='End')
    
    ax.set_xlabel('Output Gap (%)', fontsize=12)
    ax.set_ylabel('Inflation (%)', fontsize=12)
    ax.set_title('Phase Diagram: Output Gap vs Inflation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_shock_plot(simulations: List[Dict], periods: int) -> str:
    """Generate plot of shock realizations"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    time = np.arange(1, periods + 1)
    sim = simulations[0]
    
    shock_data = [
        (sim['tech_shock'], 'Technology Shock', '#3b82f6'),
        (sim['govt_shock'], 'Government Spending Shock', '#ef4444'),
        (sim['monetary_shock'], 'Monetary Policy Shock', '#22c55e')
    ]
    
    for ax, (data, name, color) in zip(axes, shock_data):
        ax.bar(time, data * 100, color=color, alpha=0.7, width=0.8)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax.set_ylabel('Shock (%)', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    axes[-1].set_xlabel('Period', fontsize=11)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(params: DSGERequest, moments: Dict, irfs: Dict) -> Dict[str, Any]:
    """Generate interpretation of DSGE results"""
    key_insights = []
    
    # Model specification
    key_insights.append({
        'title': 'Model Specification',
        'description': f'3-equation New Keynesian model with β={params.beta}, θ={params.theta} (Calvo), Taylor rule φ_π={params.phi_pi}, φ_y={params.phi_y}.',
        'status': 'neutral'
    })
    
    # Output volatility
    output_std = moments.get('Output Gap', {}).get('std', 0)
    key_insights.append({
        'title': 'Output Volatility',
        'description': f'Output gap standard deviation: {output_std*100:.2f}%. {"High volatility." if output_std > 0.02 else "Moderate volatility." if output_std > 0.01 else "Low volatility."}',
        'status': 'neutral'
    })
    
    # Inflation dynamics
    inflation_std = moments.get('Inflation', {}).get('std', 0)
    inflation_autocorr = moments.get('Inflation', {}).get('autocorr', 0)
    key_insights.append({
        'title': 'Inflation Dynamics',
        'description': f'Inflation std: {inflation_std*100:.2f}%, persistence: {inflation_autocorr:.2f}. {"Highly persistent" if inflation_autocorr > 0.7 else "Moderately persistent" if inflation_autocorr > 0.4 else "Low persistence"}.',
        'status': 'neutral'
    })
    
    # Monetary policy
    if params.phi_pi > 1:
        key_insights.append({
            'title': 'Taylor Principle Satisfied',
            'description': f'φ_π = {params.phi_pi} > 1. Monetary policy responds more than one-for-one to inflation, ensuring determinacy.',
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'Taylor Principle Violated',
            'description': f'φ_π = {params.phi_pi} ≤ 1. Monetary policy may not be sufficiently aggressive to stabilize inflation.',
            'status': 'warning'
        })
    
    # Price stickiness
    avg_price_duration = 1 / (1 - params.theta)
    key_insights.append({
        'title': 'Price Stickiness',
        'description': f'Calvo θ = {params.theta}. Average price duration: {avg_price_duration:.1f} quarters. {"High stickiness" if params.theta > 0.8 else "Moderate stickiness" if params.theta > 0.6 else "Low stickiness"}.',
        'status': 'neutral'
    })
    
    return {
        'key_insights': key_insights,
        'model_type': 'New Keynesian 3-Equation Model',
        'recommendation': 'Model exhibits standard NK dynamics with policy stabilization.' if params.phi_pi > 1 else 'Consider increasing monetary policy response to inflation.'
    }


@router.post("/dsge-simulation")
async def run_dsge_simulation(request: DSGERequest) -> Dict[str, Any]:
    """
    Simulate a Dynamic Stochastic General Equilibrium (DSGE) model.
    
    Uses a basic 3-equation New Keynesian framework with:
    - IS Curve (Euler equation)
    - New Keynesian Phillips Curve
    - Taylor Rule monetary policy
    """
    try:
        # Validate parameters
        if not (0 < request.beta < 1):
            raise HTTPException(status_code=400, detail="Beta must be between 0 and 1")
        if not (0 < request.theta < 1):
            raise HTTPException(status_code=400, detail="Theta must be between 0 and 1")
        if request.periods < 10 or request.periods > 1000:
            raise HTTPException(status_code=400, detail="Periods must be between 10 and 1000")
        
        # Initialize model
        model = NewKeynesianDSGE(request)
        
        # Run simulation
        simulations = model.simulate(request.periods, request.n_simulations, request.seed)
        
        # Compute IRFs
        irfs = model.compute_irfs(periods=40)
        
        # Compute moments
        moments = model.compute_moments(simulations)
        
        # Generate visualizations
        simulation_plot = generate_simulation_plot(simulations, request.periods)
        irf_tech_plot = generate_irf_plot(irfs, 'Technology')
        irf_govt_plot = generate_irf_plot(irfs, 'Government Spending')
        irf_monetary_plot = generate_irf_plot(irfs, 'Monetary Policy')
        all_irfs_plot = generate_all_irfs_plot(irfs)
        phase_plot = generate_phase_diagram(simulations)
        shock_plot = generate_shock_plot(simulations, request.periods)
        
        # Generate interpretation
        interpretation = generate_interpretation(request, moments, irfs)
        
        # Prepare simulation data for export
        sim_data = simulations[0]
        time_series = []
        for t in range(request.periods):
            time_series.append({
                'period': t + 1,
                'output_gap': _to_native_type(sim_data['output_gap'][t]),
                'inflation': _to_native_type(sim_data['inflation'][t]),
                'interest_rate': _to_native_type(sim_data['interest_rate'][t]),
                'tech_shock': _to_native_type(sim_data['tech_shock'][t]),
                'govt_shock': _to_native_type(sim_data['govt_shock'][t]),
                'monetary_shock': _to_native_type(sim_data['monetary_shock'][t])
            })
        
        # Model parameters summary
        params_summary = {
            'beta': request.beta,
            'sigma': request.sigma,
            'phi': request.phi,
            'theta': request.theta,
            'kappa': _to_native_type(model.kappa),
            'phi_pi': request.phi_pi,
            'phi_y': request.phi_y,
            'rho_r': request.rho_r,
            'rho_a': request.rho_a,
            'rho_g': request.rho_g,
            'rho_m': request.rho_m,
            'sigma_a': request.sigma_a,
            'sigma_g': request.sigma_g,
            'sigma_m': request.sigma_m
        }
        
        return {
            'parameters': params_summary,
            'moments': moments,
            'irfs': irfs,
            'time_series': time_series,
            'simulation_plot': simulation_plot,
            'irf_tech_plot': irf_tech_plot,
            'irf_govt_plot': irf_govt_plot,
            'irf_monetary_plot': irf_monetary_plot,
            'all_irfs_plot': all_irfs_plot,
            'phase_plot': phase_plot,
            'shock_plot': shock_plot,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DSGE simulation failed: {str(e)}")
