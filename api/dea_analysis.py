"""
DEA (Data Envelopment Analysis) Router for FastAPI
CCR, BCC, Super-Efficiency Models
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
from scipy.optimize import linprog
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class DEARequest(BaseModel):
    data: List[Dict[str, Any]]
    dmu_col: str  # DMU identifier column
    input_cols: List[str]  # Input variables
    output_cols: List[str]  # Output variables
    model_type: Literal["ccr", "bcc", "super"] = "ccr"
    orientation: Literal["input", "output"] = "input"
    returns_to_scale: Optional[Literal["crs", "vrs", "nirs", "ndrs"]] = None


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
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


def solve_dea_ccr_input(inputs: np.ndarray, outputs: np.ndarray, dmu_idx: int) -> Dict[str, Any]:
    """
    Solve CCR model (input-oriented) for a specific DMU
    
    Minimize θ
    Subject to:
        -θ*x_i + Xλ ≤ 0  (input constraints)
        Yλ ≥ y_i         (output constraints)
        λ ≥ 0
    """
    n_dmus, n_inputs = inputs.shape
    n_outputs = outputs.shape[1]
    
    # Decision variables: [θ, λ1, λ2, ..., λn]
    n_vars = 1 + n_dmus
    
    # Objective: minimize θ
    c = np.zeros(n_vars)
    c[0] = 1  # coefficient for θ
    
    # Inequality constraints: A_ub @ x <= b_ub
    # Input constraints: -θ*x_i + Xλ ≤ 0
    A_ub_input = np.zeros((n_inputs, n_vars))
    A_ub_input[:, 0] = -inputs[dmu_idx]  # -θ*x_i
    A_ub_input[:, 1:] = inputs.T  # Xλ
    b_ub_input = np.zeros(n_inputs)
    
    # Output constraints: -Yλ ≤ -y_i (converted from Yλ ≥ y_i)
    A_ub_output = np.zeros((n_outputs, n_vars))
    A_ub_output[:, 1:] = -outputs.T  # -Yλ
    b_ub_output = -outputs[dmu_idx]  # -y_i
    
    A_ub = np.vstack([A_ub_input, A_ub_output])
    b_ub = np.concatenate([b_ub_input, b_ub_output])
    
    # Bounds: θ ≥ 0, λ ≥ 0
    bounds = [(0, None) for _ in range(n_vars)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        theta = result.x[0]
        lambdas = result.x[1:]
        
        # Calculate slacks
        input_slack = inputs[dmu_idx] * theta - inputs.T @ lambdas
        output_slack = outputs.T @ lambdas - outputs[dmu_idx]
        
        return {
            'efficiency': _to_native_type(theta),
            'lambdas': lambdas.tolist(),
            'input_slack': input_slack.tolist(),
            'output_slack': output_slack.tolist(),
            'is_efficient': bool(theta >= 0.9999),
            'status': 'optimal'
        }
    else:
        return {
            'efficiency': None,
            'lambdas': [],
            'input_slack': [],
            'output_slack': [],
            'is_efficient': False,
            'status': 'infeasible'
        }


def solve_dea_ccr_output(inputs: np.ndarray, outputs: np.ndarray, dmu_idx: int) -> Dict[str, Any]:
    """
    Solve CCR model (output-oriented) for a specific DMU
    
    Maximize φ
    Subject to:
        Xλ ≤ x_i         (input constraints)
        -φ*y_i + Yλ ≥ 0  (output constraints)
        λ ≥ 0
    """
    n_dmus, n_inputs = inputs.shape
    n_outputs = outputs.shape[1]
    
    # Decision variables: [φ, λ1, λ2, ..., λn]
    n_vars = 1 + n_dmus
    
    # Objective: maximize φ (minimize -φ)
    c = np.zeros(n_vars)
    c[0] = -1
    
    # Input constraints: Xλ ≤ x_i
    A_ub_input = np.zeros((n_inputs, n_vars))
    A_ub_input[:, 1:] = inputs.T
    b_ub_input = inputs[dmu_idx]
    
    # Output constraints: -φ*y_i + Yλ ≥ 0 → φ*y_i - Yλ ≤ 0
    A_ub_output = np.zeros((n_outputs, n_vars))
    A_ub_output[:, 0] = outputs[dmu_idx]  # φ*y_i
    A_ub_output[:, 1:] = -outputs.T  # -Yλ
    b_ub_output = np.zeros(n_outputs)
    
    A_ub = np.vstack([A_ub_input, A_ub_output])
    b_ub = np.concatenate([b_ub_input, b_ub_output])
    
    bounds = [(0, None) for _ in range(n_vars)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        phi = result.x[0]
        lambdas = result.x[1:]
        efficiency = 1 / phi if phi > 0 else 1
        
        return {
            'efficiency': _to_native_type(efficiency),
            'phi': _to_native_type(phi),
            'lambdas': lambdas.tolist(),
            'is_efficient': bool(phi <= 1.0001),
            'status': 'optimal'
        }
    else:
        return {
            'efficiency': None,
            'phi': None,
            'lambdas': [],
            'is_efficient': False,
            'status': 'infeasible'
        }


def solve_dea_bcc_input(inputs: np.ndarray, outputs: np.ndarray, dmu_idx: int) -> Dict[str, Any]:
    """
    Solve BCC model (input-oriented, VRS) for a specific DMU
    
    Minimize θ
    Subject to:
        -θ*x_i + Xλ ≤ 0
        Yλ ≥ y_i
        Σλ = 1  (VRS constraint)
        λ ≥ 0
    """
    n_dmus, n_inputs = inputs.shape
    n_outputs = outputs.shape[1]
    
    n_vars = 1 + n_dmus
    
    c = np.zeros(n_vars)
    c[0] = 1
    
    # Input constraints
    A_ub_input = np.zeros((n_inputs, n_vars))
    A_ub_input[:, 0] = -inputs[dmu_idx]
    A_ub_input[:, 1:] = inputs.T
    b_ub_input = np.zeros(n_inputs)
    
    # Output constraints
    A_ub_output = np.zeros((n_outputs, n_vars))
    A_ub_output[:, 1:] = -outputs.T
    b_ub_output = -outputs[dmu_idx]
    
    A_ub = np.vstack([A_ub_input, A_ub_output])
    b_ub = np.concatenate([b_ub_input, b_ub_output])
    
    # Equality constraint: Σλ = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, 1:] = 1
    b_eq = np.array([1])
    
    bounds = [(0, None) for _ in range(n_vars)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        theta = result.x[0]
        lambdas = result.x[1:]
        
        return {
            'efficiency': _to_native_type(theta),
            'lambdas': lambdas.tolist(),
            'is_efficient': bool(theta >= 0.9999),
            'status': 'optimal'
        }
    else:
        return {
            'efficiency': None,
            'lambdas': [],
            'is_efficient': False,
            'status': 'infeasible'
        }


def solve_dea_bcc_output(inputs: np.ndarray, outputs: np.ndarray, dmu_idx: int) -> Dict[str, Any]:
    """Solve BCC model (output-oriented, VRS)"""
    n_dmus, n_inputs = inputs.shape
    n_outputs = outputs.shape[1]
    
    n_vars = 1 + n_dmus
    
    c = np.zeros(n_vars)
    c[0] = -1
    
    A_ub_input = np.zeros((n_inputs, n_vars))
    A_ub_input[:, 1:] = inputs.T
    b_ub_input = inputs[dmu_idx]
    
    A_ub_output = np.zeros((n_outputs, n_vars))
    A_ub_output[:, 0] = outputs[dmu_idx]
    A_ub_output[:, 1:] = -outputs.T
    b_ub_output = np.zeros(n_outputs)
    
    A_ub = np.vstack([A_ub_input, A_ub_output])
    b_ub = np.concatenate([b_ub_input, b_ub_output])
    
    A_eq = np.zeros((1, n_vars))
    A_eq[0, 1:] = 1
    b_eq = np.array([1])
    
    bounds = [(0, None) for _ in range(n_vars)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        phi = result.x[0]
        lambdas = result.x[1:]
        efficiency = 1 / phi if phi > 0 else 1
        
        return {
            'efficiency': _to_native_type(efficiency),
            'phi': _to_native_type(phi),
            'lambdas': lambdas.tolist(),
            'is_efficient': bool(phi <= 1.0001),
            'status': 'optimal'
        }
    else:
        return {
            'efficiency': None,
            'phi': None,
            'lambdas': [],
            'is_efficient': False,
            'status': 'infeasible'
        }


def solve_dea_super_efficiency(inputs: np.ndarray, outputs: np.ndarray, dmu_idx: int,
                                model_type: str = 'ccr', orientation: str = 'input') -> Dict[str, Any]:
    """
    Solve Super-Efficiency model (excludes DMU from reference set)
    """
    n_dmus = inputs.shape[0]
    
    # Create matrices excluding the evaluated DMU
    mask = np.ones(n_dmus, dtype=bool)
    mask[dmu_idx] = False
    inputs_ref = inputs[mask]
    outputs_ref = outputs[mask]
    
    n_ref = n_dmus - 1
    n_inputs = inputs.shape[1]
    n_outputs = outputs.shape[1]
    
    if orientation == 'input':
        n_vars = 1 + n_ref
        c = np.zeros(n_vars)
        c[0] = 1
        
        A_ub_input = np.zeros((n_inputs, n_vars))
        A_ub_input[:, 0] = -inputs[dmu_idx]
        A_ub_input[:, 1:] = inputs_ref.T
        b_ub_input = np.zeros(n_inputs)
        
        A_ub_output = np.zeros((n_outputs, n_vars))
        A_ub_output[:, 1:] = -outputs_ref.T
        b_ub_output = -outputs[dmu_idx]
        
        A_ub = np.vstack([A_ub_input, A_ub_output])
        b_ub = np.concatenate([b_ub_input, b_ub_output])
        
        A_eq = None
        b_eq = None
        if model_type == 'bcc':
            A_eq = np.zeros((1, n_vars))
            A_eq[0, 1:] = 1
            b_eq = np.array([1])
        
        bounds = [(0, None) for _ in range(n_vars)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            return {
                'super_efficiency': _to_native_type(result.x[0]),
                'status': 'optimal'
            }
    
    return {'super_efficiency': None, 'status': 'infeasible'}


def calculate_scale_efficiency(ccr_eff: float, bcc_eff: float) -> float:
    """Calculate scale efficiency = CCR / BCC"""
    if bcc_eff and bcc_eff > 0:
        return ccr_eff / bcc_eff if ccr_eff else None
    return None


def identify_returns_to_scale(lambdas: List[float]) -> str:
    """Identify returns to scale based on lambda sum"""
    lambda_sum = sum(lambdas)
    if abs(lambda_sum - 1) < 0.001:
        return 'CRS'  # Constant Returns to Scale
    elif lambda_sum < 1:
        return 'IRS'  # Increasing Returns to Scale
    else:
        return 'DRS'  # Decreasing Returns to Scale


def calculate_targets(dmu_inputs: np.ndarray, dmu_outputs: np.ndarray,
                      efficiency: float, orientation: str,
                      ref_inputs: np.ndarray, ref_outputs: np.ndarray,
                      lambdas: np.ndarray) -> Dict[str, Any]:
    """Calculate target inputs/outputs for inefficient DMUs"""
    if orientation == 'input':
        target_inputs = efficiency * dmu_inputs
        target_outputs = dmu_outputs.copy()
    else:
        target_inputs = dmu_inputs.copy()
        target_outputs = dmu_outputs / efficiency if efficiency > 0 else dmu_outputs
    
    return {
        'target_inputs': target_inputs.tolist(),
        'target_outputs': target_outputs.tolist(),
        'input_reduction': ((dmu_inputs - target_inputs) / dmu_inputs * 100).tolist() if orientation == 'input' else [0] * len(dmu_inputs),
        'output_increase': ((target_outputs - dmu_outputs) / dmu_outputs * 100).tolist() if orientation == 'output' else [0] * len(dmu_outputs)
    }


def create_efficiency_bar_chart(dmu_results: List[Dict], dmu_names: List[str]) -> str:
    """Create efficiency score bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    efficiencies = [r['efficiency'] or 0 for r in dmu_results]
    colors = ['#22c55e' if e >= 0.9999 else '#f59e0b' if e >= 0.8 else '#ef4444' for e in efficiencies]
    
    bars = ax.bar(range(len(dmu_names)), efficiencies, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=1.0, color='#22c55e', linestyle='--', linewidth=2, label='Efficient Frontier (1.0)')
    ax.axhline(y=0.8, color='#f59e0b', linestyle=':', linewidth=1.5, alpha=0.7, label='Threshold (0.8)')
    
    ax.set_xlabel('DMU', fontsize=11)
    ax.set_ylabel('Efficiency Score', fontsize=11)
    ax.set_title('DEA Efficiency Scores by DMU', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(dmu_names)))
    ax.set_xticklabels(dmu_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{eff:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_efficiency_distribution_chart(efficiencies: List[float]) -> str:
    """Create efficiency distribution histogram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(efficiencies, bins=10, color='#3b82f6', edgecolor='white', alpha=0.7)
    ax1.axvline(x=np.mean(efficiencies), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(efficiencies):.3f}')
    ax1.axvline(x=1.0, color='#22c55e', linestyle='-', linewidth=2, label='Efficient (1.0)')
    ax1.set_xlabel('Efficiency Score', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Efficiency Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Box plot
    bp = ax2.boxplot(efficiencies, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3b82f6')
    bp['boxes'][0].set_alpha(0.7)
    ax2.axhline(y=1.0, color='#22c55e', linestyle='--', linewidth=2, label='Efficient')
    ax2.set_ylabel('Efficiency Score', fontsize=11)
    ax2.set_title('Efficiency Score Box Plot', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['All DMUs'])
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_frontier_chart(inputs: np.ndarray, outputs: np.ndarray, 
                         efficiencies: List[float], dmu_names: List[str],
                         input_cols: List[str], output_cols: List[str]) -> str:
    """Create efficiency frontier visualization (2D)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use first input and first output for 2D visualization
    if inputs.shape[1] >= 1 and outputs.shape[1] >= 1:
        x = inputs[:, 0]
        y = outputs[:, 0]
        
        # Calculate efficiency ratio (output/input)
        ratio = y / (x + 1e-10)
        
        # Color by efficiency
        colors = ['#22c55e' if e >= 0.9999 else '#f59e0b' if e >= 0.8 else '#ef4444' for e in efficiencies]
        
        scatter = ax.scatter(x, y, c=colors, s=100, edgecolor='white', linewidth=1.5, zorder=5)
        
        # Add labels
        for i, name in enumerate(dmu_names):
            ax.annotate(name, (x[i], y[i]), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        # Draw frontier line (connect efficient DMUs)
        efficient_idx = [i for i, e in enumerate(efficiencies) if e >= 0.9999]
        if efficient_idx:
            eff_points = [(x[i], y[i]) for i in efficient_idx]
            eff_points.sort(key=lambda p: p[0])
            
            # Add origin
            frontier_x = [0] + [p[0] for p in eff_points] + [max(x) * 1.1]
            frontier_y = [0] + [p[1] for p in eff_points] + [eff_points[-1][1]]
            
            ax.plot(frontier_x[:-1], frontier_y[:-1], 'g--', linewidth=2, alpha=0.5, label='Efficient Frontier')
        
        ax.set_xlabel(f'{input_cols[0]} (Input)', fontsize=11)
        ax.set_ylabel(f'{output_cols[0]} (Output)', fontsize=11)
        ax.set_title('DEA Efficiency Frontier', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_input_output_comparison(inputs: np.ndarray, outputs: np.ndarray,
                                   efficiencies: List[float], dmu_names: List[str],
                                   input_cols: List[str], output_cols: List[str]) -> str:
    """Create input/output comparison radar or bar chart"""
    n_inputs = len(input_cols)
    n_outputs = len(output_cols)
    n_vars = n_inputs + n_outputs
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Efficient vs Inefficient comparison
    efficient_mask = [e >= 0.9999 for e in efficiencies]
    inefficient_mask = [not m for m in efficient_mask]
    
    if sum(efficient_mask) > 0 and sum(inefficient_mask) > 0:
        # Average inputs
        eff_inputs_avg = inputs[efficient_mask].mean(axis=0)
        ineff_inputs_avg = inputs[inefficient_mask].mean(axis=0)
        
        x = np.arange(n_inputs)
        width = 0.35
        
        axes[0].bar(x - width/2, eff_inputs_avg, width, label='Efficient', color='#22c55e', alpha=0.8)
        axes[0].bar(x + width/2, ineff_inputs_avg, width, label='Inefficient', color='#ef4444', alpha=0.8)
        axes[0].set_ylabel('Average Value')
        axes[0].set_title('Input Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(input_cols, rotation=45, ha='right')
        axes[0].legend()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Average outputs
        eff_outputs_avg = outputs[efficient_mask].mean(axis=0)
        ineff_outputs_avg = outputs[inefficient_mask].mean(axis=0)
        
        x = np.arange(n_outputs)
        
        axes[1].bar(x - width/2, eff_outputs_avg, width, label='Efficient', color='#22c55e', alpha=0.8)
        axes[1].bar(x + width/2, ineff_outputs_avg, width, label='Inefficient', color='#ef4444', alpha=0.8)
        axes[1].set_ylabel('Average Value')
        axes[1].set_title('Output Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(output_cols, rotation=45, ha='right')
        axes[1].legend()
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_ranking_chart(dmu_results: List[Dict], dmu_names: List[str]) -> str:
    """Create efficiency ranking chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by efficiency
    sorted_data = sorted(zip(dmu_names, dmu_results), key=lambda x: x[1]['efficiency'] or 0, reverse=True)
    sorted_names = [d[0] for d in sorted_data]
    sorted_effs = [d[1]['efficiency'] or 0 for d in sorted_data]
    
    colors = ['#22c55e' if e >= 0.9999 else '#f59e0b' if e >= 0.8 else '#ef4444' for e in sorted_effs]
    
    bars = ax.barh(range(len(sorted_names)), sorted_effs, color=colors, edgecolor='white')
    
    ax.axvline(x=1.0, color='#22c55e', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Efficiency Score', fontsize=11)
    ax.set_title('DMU Efficiency Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add rank labels
    for i, (bar, eff) in enumerate(zip(bars, sorted_effs)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'#{i+1} ({eff:.3f})', va='center', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(results: Dict, model_type: str, orientation: str) -> List[Dict[str, Any]]:
    """Generate key insights from DEA analysis"""
    insights = []
    
    dmu_results = results.get('dmu_results', [])
    efficiencies = [r['efficiency'] for r in dmu_results if r['efficiency'] is not None]
    
    if not efficiencies:
        return insights
    
    n_efficient = sum(1 for e in efficiencies if e >= 0.9999)
    n_total = len(efficiencies)
    pct_efficient = n_efficient / n_total * 100
    
    # Efficiency rate insight
    if pct_efficient >= 30:
        insights.append({
            'title': 'High Efficiency Rate',
            'description': f'{n_efficient} out of {n_total} DMUs ({pct_efficient:.1f}%) are efficient. The overall performance is strong.',
            'status': 'positive'
        })
    elif pct_efficient >= 15:
        insights.append({
            'title': 'Moderate Efficiency Rate',
            'description': f'{n_efficient} out of {n_total} DMUs ({pct_efficient:.1f}%) are efficient. There is room for improvement.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': 'Low Efficiency Rate',
            'description': f'Only {n_efficient} out of {n_total} DMUs ({pct_efficient:.1f}%) are efficient. Significant improvement needed.',
            'status': 'warning'
        })
    
    # Average efficiency
    avg_eff = np.mean(efficiencies)
    if avg_eff >= 0.9:
        insights.append({
            'title': 'High Average Efficiency',
            'description': f'Average efficiency score is {avg_eff:.3f}. Most DMUs perform close to the frontier.',
            'status': 'positive'
        })
    elif avg_eff >= 0.7:
        insights.append({
            'title': 'Moderate Average Efficiency',
            'description': f'Average efficiency score is {avg_eff:.3f}. Consider benchmarking inefficient DMUs.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': 'Low Average Efficiency',
            'description': f'Average efficiency score is {avg_eff:.3f}. Many DMUs need significant improvement.',
            'status': 'warning'
        })
    
    # Efficiency range
    min_eff = min(efficiencies)
    max_eff = max(efficiencies)
    eff_range = max_eff - min_eff
    
    if eff_range > 0.5:
        insights.append({
            'title': 'High Performance Variation',
            'description': f'Efficiency ranges from {min_eff:.3f} to {max_eff:.3f}. Large gap between best and worst performers.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Consistent Performance',
            'description': f'Efficiency ranges from {min_eff:.3f} to {max_eff:.3f}. Relatively uniform performance.',
            'status': 'neutral'
        })
    
    # Model-specific insights
    if model_type == 'bcc':
        scale_effs = [r.get('scale_efficiency') for r in dmu_results if r.get('scale_efficiency')]
        if scale_effs:
            avg_scale = np.mean(scale_effs)
            insights.append({
                'title': 'Scale Efficiency Analysis',
                'description': f'Average scale efficiency: {avg_scale:.3f}. {"Scale-efficient operations." if avg_scale > 0.95 else "Consider scale adjustments for some DMUs."}',
                'status': 'positive' if avg_scale > 0.95 else 'neutral'
            })
    
    return insights


@router.post("/dea-efficiency")
async def run_dea_analysis(request: DEARequest) -> Dict[str, Any]:
    """
    Perform DEA (Data Envelopment Analysis).
    
    Models:
    - CCR: Constant Returns to Scale (CRS)
    - BCC: Variable Returns to Scale (VRS)
    - Super: Super-efficiency model
    
    Orientations:
    - Input: Minimize inputs for given outputs
    - Output: Maximize outputs for given inputs
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.dmu_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"DMU column '{request.dmu_col}' not found")
        
        for col in request.input_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Input column '{col}' not found")
        
        for col in request.output_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Output column '{col}' not found")
        
        # Prepare data
        dmu_names = df[request.dmu_col].tolist()
        inputs = df[request.input_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        outputs = df[request.output_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        n_dmus = len(dmu_names)
        n_inputs = len(request.input_cols)
        n_outputs = len(request.output_cols)
        
        # Validate data
        if n_dmus < 3:
            raise HTTPException(status_code=400, detail="At least 3 DMUs required for DEA")
        
        if np.any(inputs <= 0) or np.any(outputs <= 0):
            # Replace zeros with small positive values
            inputs = np.where(inputs <= 0, 0.001, inputs)
            outputs = np.where(outputs <= 0, 0.001, outputs)
        
        # Run DEA for each DMU
        dmu_results = []
        
        for i in range(n_dmus):
            result = {'dmu': dmu_names[i], 'dmu_index': i}
            
            # Run appropriate model
            if request.model_type == 'ccr':
                if request.orientation == 'input':
                    dea_result = solve_dea_ccr_input(inputs, outputs, i)
                else:
                    dea_result = solve_dea_ccr_output(inputs, outputs, i)
            elif request.model_type == 'bcc':
                if request.orientation == 'input':
                    dea_result = solve_dea_bcc_input(inputs, outputs, i)
                else:
                    dea_result = solve_dea_bcc_output(inputs, outputs, i)
                
                # Also calculate CCR for scale efficiency
                ccr_result = solve_dea_ccr_input(inputs, outputs, i) if request.orientation == 'input' else solve_dea_ccr_output(inputs, outputs, i)
                result['ccr_efficiency'] = ccr_result['efficiency']
                result['scale_efficiency'] = calculate_scale_efficiency(ccr_result['efficiency'], dea_result['efficiency'])
                result['returns_to_scale'] = identify_returns_to_scale(dea_result.get('lambdas', []))
            elif request.model_type == 'super':
                # First run standard model
                if request.orientation == 'input':
                    dea_result = solve_dea_ccr_input(inputs, outputs, i)
                else:
                    dea_result = solve_dea_ccr_output(inputs, outputs, i)
                
                # Then super-efficiency for efficient DMUs
                if dea_result.get('is_efficient', False):
                    super_result = solve_dea_super_efficiency(inputs, outputs, i, 'ccr', request.orientation)
                    result['super_efficiency'] = super_result.get('super_efficiency')
            
            result.update(dea_result)
            
            # Add input/output values
            result['inputs'] = {col: _to_native_type(inputs[i, j]) for j, col in enumerate(request.input_cols)}
            result['outputs'] = {col: _to_native_type(outputs[i, j]) for j, col in enumerate(request.output_cols)}
            
            # Calculate targets for inefficient DMUs
            if result.get('efficiency') and result['efficiency'] < 0.9999:
                targets = calculate_targets(
                    inputs[i], outputs[i], result['efficiency'], request.orientation,
                    inputs, outputs, np.array(result.get('lambdas', []))
                )
                result['targets'] = targets
            
            dmu_results.append(result)
        
        # Calculate summary statistics
        efficiencies = [r['efficiency'] for r in dmu_results if r['efficiency'] is not None]
        n_efficient = sum(1 for e in efficiencies if e >= 0.9999)
        
        # Rank DMUs
        sorted_results = sorted(dmu_results, key=lambda x: x['efficiency'] or 0, reverse=True)
        for rank, result in enumerate(sorted_results, 1):
            for r in dmu_results:
                if r['dmu'] == result['dmu']:
                    r['rank'] = rank
                    break
        
        # Create visualizations
        visualizations = {}
        visualizations['efficiency_bar'] = create_efficiency_bar_chart(dmu_results, dmu_names)
        visualizations['efficiency_distribution'] = create_efficiency_distribution_chart(efficiencies)
        visualizations['frontier'] = create_frontier_chart(inputs, outputs, efficiencies, dmu_names,
                                                           request.input_cols, request.output_cols)
        visualizations['input_output_comparison'] = create_input_output_comparison(
            inputs, outputs, efficiencies, dmu_names, request.input_cols, request.output_cols)
        visualizations['ranking'] = create_ranking_chart(dmu_results, dmu_names)
        
        # Generate insights
        results = {
            'dmu_results': dmu_results,
            'summary_stats': {
                'total_dmus': n_dmus,
                'efficient_dmus': n_efficient,
                'inefficient_dmus': n_dmus - n_efficient,
                'efficiency_rate': _to_native_type(n_efficient / n_dmus * 100),
                'avg_efficiency': _to_native_type(np.mean(efficiencies)),
                'median_efficiency': _to_native_type(np.median(efficiencies)),
                'min_efficiency': _to_native_type(min(efficiencies)),
                'max_efficiency': _to_native_type(max(efficiencies)),
                'std_efficiency': _to_native_type(np.std(efficiencies))
            },
            'model_info': {
                'model_type': request.model_type.upper(),
                'orientation': request.orientation,
                'n_inputs': n_inputs,
                'n_outputs': n_outputs,
                'input_cols': request.input_cols,
                'output_cols': request.output_cols
            }
        }
        
        insights = generate_key_insights(results, request.model_type, request.orientation)
        
        summary = {
            'model': f'{request.model_type.upper()}-{request.orientation.capitalize()}',
            'total_dmus': n_dmus,
            'efficient_dmus': n_efficient,
            'avg_efficiency': _to_native_type(np.mean(efficiencies)),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DEA analysis failed: {str(e)}")
