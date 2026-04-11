"""
Process Capability Analysis Router for FastAPI
Calculates Cp, Cpk, Pp, Ppk and related metrics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import io
import base64
import time
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


# ============ REQUEST MODEL ============
class CapabilityRequest(BaseModel):
    data: List[Dict[str, Any]]
    measurement_col: str
    subgroup_col: Optional[str] = None
    usl: float
    lsl: float
    target: Optional[float] = None
    subgroup_size: int = 5


# ============ UTILITY FUNCTIONS ============
def _to_native_type(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
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


# ============ CONSTANTS ============
COLORS = {
    'in_spec': '#22c55e',
    'out_spec': '#ef4444',
    'target': '#3b82f6',
    'lsl': '#f59e0b',
    'usl': '#f59e0b',
    'distribution': '#8b5cf6',
    'control': '#06b6d4',
}

# d2 constants for subgroup sizes 2-25 (for estimating sigma from R-bar)
D2_CONSTANTS = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
    6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970,
    10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336,
    14: 3.407, 15: 3.472, 16: 3.532, 17: 3.588,
    18: 3.640, 19: 3.689, 20: 3.735, 21: 3.778,
    22: 3.819, 23: 3.858, 24: 3.895, 25: 3.931,
}

# A2 constants for X-bar chart
A2_CONSTANTS = {
    2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577,
    6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337,
    10: 0.308, 11: 0.285, 12: 0.266, 13: 0.249,
    14: 0.235, 15: 0.223, 16: 0.212, 17: 0.203,
    18: 0.194, 19: 0.187, 20: 0.180, 21: 0.173,
    22: 0.167, 23: 0.162, 24: 0.157, 25: 0.153,
}

# D3, D4 constants for R chart
D3_CONSTANTS = {
    2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076,
    8: 0.136, 9: 0.184, 10: 0.223, 11: 0.256,
    12: 0.283, 13: 0.307, 14: 0.328, 15: 0.347,
}

D4_CONSTANTS = {
    2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114,
    6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816,
    10: 1.777, 11: 1.744, 12: 1.717, 13: 1.693,
    14: 1.672, 15: 1.653,
}


# ============ CALCULATION FUNCTIONS ============
def calculate_subgroup_stats(df: pd.DataFrame, measurement_col: str,
                              subgroup_col: Optional[str], subgroup_size: int) -> Dict:
    """Calculate subgroup statistics for capability analysis"""
    values = df[measurement_col].dropna().values
    n = len(values)
    
    if n < 2:
        raise ValueError("Need at least 2 measurements for capability analysis")
    
    # Overall statistics
    mean = np.mean(values)
    std_overall = np.std(values, ddof=1)
    
    # Create subgroups
    if subgroup_col and subgroup_col in df.columns:
        # Use provided subgroup column
        groups = df.groupby(subgroup_col)[measurement_col].apply(list).values
        subgroups = [np.array(g) for g in groups if len(g) >= 2]
        actual_subgroup_size = int(np.mean([len(g) for g in subgroups])) if subgroups else subgroup_size
    else:
        # Create subgroups of specified size
        actual_subgroup_size = min(subgroup_size, n // 2)
        actual_subgroup_size = max(2, actual_subgroup_size)
        num_complete = n // actual_subgroup_size
        subgroups = [values[i*actual_subgroup_size:(i+1)*actual_subgroup_size] 
                     for i in range(num_complete)]
    
    num_subgroups = len(subgroups)
    
    if num_subgroups < 2:
        # Not enough subgroups, use overall std
        std_within = std_overall
    else:
        # Calculate R-bar (average range) method
        ranges = [np.max(sg) - np.min(sg) for sg in subgroups]
        r_bar = np.mean(ranges)
        
        # Get d2 constant
        d2 = D2_CONSTANTS.get(actual_subgroup_size, 2.326)
        
        # Estimate within-group std
        std_within = r_bar / d2 if r_bar > 0 else std_overall
    
    return {
        'mean': mean,
        'std_within': std_within,
        'std_overall': std_overall,
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
        'n': n,
        'subgroup_size': actual_subgroup_size,
        'num_subgroups': num_subgroups,
        'subgroups': subgroups,
        'values': values,
    }


def calculate_capability_indices(stats: Dict, usl: float, lsl: float, 
                                  target: Optional[float]) -> Dict:
    """Calculate all capability indices"""
    mean = stats['mean']
    std_within = stats['std_within']
    std_overall = stats['std_overall']
    
    tolerance = usl - lsl
    
    # Prevent division by zero
    if std_within <= 0:
        std_within = 0.0001
    if std_overall <= 0:
        std_overall = 0.0001
    
    # Short-term capability (using within-group variation)
    cp = tolerance / (6 * std_within)
    cpu = (usl - mean) / (3 * std_within)
    cpl = (mean - lsl) / (3 * std_within)
    cpk = min(cpu, cpl)
    
    # Long-term performance (using overall variation)
    pp = tolerance / (6 * std_overall)
    ppu = (usl - mean) / (3 * std_overall)
    ppl = (mean - lsl) / (3 * std_overall)
    ppk = min(ppu, ppl)
    
    # Cpm (Taguchi index) - if target specified
    cpm = None
    if target is not None:
        tau = np.sqrt(std_overall**2 + (mean - target)**2)
        cpm = tolerance / (6 * tau) if tau > 0 else cp
    
    return {
        'cp': cp,
        'cpk': cpk,
        'cpl': cpl,
        'cpu': cpu,
        'pp': pp,
        'ppk': ppk,
        'ppl': ppl,
        'ppu': ppu,
        'cpm': cpm,
    }


def calculate_defect_metrics(stats: Dict, usl: float, lsl: float) -> Dict:
    """Calculate defect rates and sigma level"""
    mean = stats['mean']
    std_overall = stats['std_overall']
    n = stats['n']
    values = stats['values']
    
    # Actual out-of-spec counts
    below_lsl = np.sum(values < lsl)
    above_usl = np.sum(values > usl)
    total_out = below_lsl + above_usl
    
    # Theoretical PPM based on normal distribution
    if std_overall > 0:
        z_lower = (lsl - mean) / std_overall
        z_upper = (usl - mean) / std_overall
        
        ppm_below_lsl = sp_stats.norm.cdf(z_lower) * 1000000
        ppm_above_usl = (1 - sp_stats.norm.cdf(z_upper)) * 1000000
        ppm_total = ppm_below_lsl + ppm_above_usl
    else:
        ppm_below_lsl = 0
        ppm_above_usl = 0
        ppm_total = 0
    
    # Actual percent out of spec
    percent_out_of_spec = (total_out / n * 100) if n > 0 else 0
    yield_percent = 100 - percent_out_of_spec
    
    # Sigma level (distance to nearest spec in sigma units)
    if std_overall > 0:
        z_to_upper = (usl - mean) / std_overall
        z_to_lower = (mean - lsl) / std_overall
        sigma_level = min(z_to_upper, z_to_lower)
        # Add 1.5 sigma shift for long-term
        # sigma_level_shifted = sigma_level + 1.5
    else:
        sigma_level = 6.0
    
    return {
        'ppm_total': ppm_total,
        'ppm_below_lsl': ppm_below_lsl,
        'ppm_above_usl': ppm_above_usl,
        'percent_out_of_spec': percent_out_of_spec,
        'sigma_level': max(0, min(6, sigma_level)),
        'yield_percent': yield_percent,
        'actual_below_lsl': int(below_lsl),
        'actual_above_usl': int(above_usl),
    }


def check_normality(values: np.ndarray) -> Dict:
    """Perform Shapiro-Wilk normality test"""
    # Limit sample size for Shapiro-Wilk (max 5000)
    if len(values) > 5000:
        sample = np.random.choice(values, 5000, replace=False)
    else:
        sample = values
    
    try:
        stat, p_value = sp_stats.shapiro(sample)
    except:
        stat, p_value = 0.0, 0.0
    
    return {
        'shapiro_stat': stat,
        'shapiro_p': p_value,
        'is_normal': p_value > 0.05,
    }


def assess_capability(indices: Dict) -> Dict:
    """Generate assessment based on capability indices"""
    cpk = indices['cpk']
    ppk = indices['ppk']
    cp = indices['cp']
    
    # Short-term assessment
    if cpk >= 1.67:
        short_term = "Excellent - Six Sigma capable"
    elif cpk >= 1.33:
        short_term = "Good - Meets requirements"
    elif cpk >= 1.0:
        short_term = "Marginal - Barely acceptable"
    elif cpk >= 0.67:
        short_term = "Poor - Improvement needed"
    else:
        short_term = "Very Poor - Not capable"
    
    # Long-term assessment
    if ppk >= 1.67:
        long_term = "Excellent long-term performance"
    elif ppk >= 1.33:
        long_term = "Good long-term performance"
    elif ppk >= 1.0:
        long_term = "Marginal long-term performance"
    else:
        long_term = "Poor long-term performance"
    
    # Recommendation
    recommendations = []
    
    if cpk < 1.33:
        if cp > cpk * 1.2:
            recommendations.append("Process is off-center. Adjust mean toward target.")
        else:
            recommendations.append("Reduce process variation through process improvements.")
    
    if ppk < cpk * 0.9:
        recommendations.append("Significant long-term drift detected. Investigate special causes.")
    
    if cpk >= 1.33 and ppk >= 1.33:
        recommendations.append("Process is capable. Maintain current controls and monitor.")
    
    return {
        'short_term': short_term,
        'long_term': long_term,
        'recommendation': " ".join(recommendations) if recommendations else "Continue monitoring process performance.",
    }


# ============ VISUALIZATION FUNCTIONS ============
def create_histogram(stats: Dict, usl: float, lsl: float, 
                     target: Optional[float], indices: Dict) -> str:
    """Create histogram with spec limits and normal curve"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    values = stats['values']
    mean = stats['mean']
    std = stats['std_overall']
    
    # Histogram
    n_bins = min(30, max(10, len(values) // 10))
    counts, bins, patches = ax.hist(values, bins=n_bins, density=True, 
                                     alpha=0.7, color=COLORS['distribution'],
                                     edgecolor='white', linewidth=1)
    
    # Color bars based on spec limits
    for patch, left_edge in zip(patches, bins[:-1]):
        right_edge = left_edge + (bins[1] - bins[0])
        if right_edge < lsl or left_edge > usl:
            patch.set_facecolor(COLORS['out_spec'])
        elif left_edge < lsl or right_edge > usl:
            patch.set_facecolor('#f59e0b')  # Warning color for partial
    
    # Normal curve
    x = np.linspace(min(values) - std, max(values) + std, 200)
    y = sp_stats.norm.pdf(x, mean, std)
    ax.plot(x, y, 'k-', linewidth=2, label='Normal fit')
    
    # Spec limits
    ax.axvline(lsl, color=COLORS['lsl'], linewidth=2, linestyle='--', label=f'LSL = {lsl}')
    ax.axvline(usl, color=COLORS['usl'], linewidth=2, linestyle='--', label=f'USL = {usl}')
    ax.axvline(mean, color=COLORS['target'], linewidth=2, linestyle='-', label=f'Mean = {mean:.4f}')
    
    if target is not None:
        ax.axvline(target, color='green', linewidth=2, linestyle=':', label=f'Target = {target}')
    
    # Annotations
    ax.set_xlabel('Measurement Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Process Capability Histogram\nCpk = {indices["cpk"]:.3f}, Ppk = {indices["ppk"]:.3f}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_capability_chart(stats: Dict, indices: Dict, usl: float, lsl: float,
                            target: Optional[float]) -> str:
    """Create capability summary chart with gauges"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Chart 1: Capability indices comparison
    ax1 = axes[0]
    metrics = ['Cp', 'Cpk', 'Pp', 'Ppk']
    values = [indices['cp'], indices['cpk'], indices['pp'], indices['ppk']]
    colors = [COLORS['in_spec'] if v >= 1.33 else COLORS['out_spec'] if v < 1.0 else '#f59e0b' for v in values]
    
    bars = ax1.barh(metrics, values, color=colors, edgecolor='white', linewidth=2)
    ax1.axvline(1.0, color='orange', linestyle='--', alpha=0.7, label='Min (1.0)')
    ax1.axvline(1.33, color='green', linestyle='--', alpha=0.7, label='Target (1.33)')
    
    for bar, val in zip(bars, values):
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Index Value', fontsize=11)
    ax1.set_title('Capability Indices', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(0, max(values) * 1.3)
    
    # Chart 2: Process spread vs spec
    ax2 = axes[1]
    mean = stats['mean']
    std_within = stats['std_within']
    std_overall = stats['std_overall']
    
    # Draw spec limits
    spec_mid = (usl + lsl) / 2
    ax2.axhspan(lsl, usl, alpha=0.2, color='green', label='Spec Range')
    ax2.axhline(lsl, color=COLORS['lsl'], linewidth=2, linestyle='--')
    ax2.axhline(usl, color=COLORS['usl'], linewidth=2, linestyle='--')
    ax2.axhline(mean, color=COLORS['target'], linewidth=2, label=f'Mean')
    
    # Draw 3-sigma ranges
    ax2.fill_between([0.8, 1.2], mean - 3*std_within, mean + 3*std_within, 
                     alpha=0.3, color='blue', label='±3σ (within)')
    ax2.fill_between([1.8, 2.2], mean - 3*std_overall, mean + 3*std_overall,
                     alpha=0.3, color='purple', label='±3σ (overall)')
    
    ax2.set_xlim(0, 3)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Short-term', 'Long-term'])
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('Process Spread vs Specifications', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    
    # Chart 3: Sigma gauge
    ax3 = axes[2]
    sigma = min(6, max(0, (min(usl - mean, mean - lsl) / stats['std_overall'])))
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    ax3.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=20)
    
    sections = [(0, 2, '#ef4444'), (2, 3, '#f97316'), (3, 4, '#f59e0b'), 
                (4, 5, '#84cc16'), (5, 6, '#22c55e')]
    
    for start, end, color in sections:
        mask = (theta >= np.pi * (1 - end/6)) & (theta <= np.pi * (1 - start/6))
        ax3.plot(r * np.cos(theta[mask]), r * np.sin(theta[mask]), color, linewidth=20)
    
    # Needle
    needle_angle = np.pi * (1 - sigma / 6)
    ax3.annotate('', xy=(0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle)),
                xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='black', lw=3))
    
    ax3.text(0, -0.3, f'{sigma:.2f}σ', ha='center', va='top', fontsize=24, fontweight='bold')
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-0.5, 1.3)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Sigma Level', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_control_chart(stats: Dict, usl: float, lsl: float) -> str:
    """Create X-bar R control chart"""
    subgroups = stats['subgroups']
    
    if len(subgroups) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Not enough subgroups for control chart',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return _fig_to_base64(fig)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Calculate X-bar and R for each subgroup
    x_bars = [np.mean(sg) for sg in subgroups]
    ranges = [np.max(sg) - np.min(sg) for sg in subgroups]
    
    x_bar_bar = np.mean(x_bars)
    r_bar = np.mean(ranges)
    
    subgroup_size = stats['subgroup_size']
    a2 = A2_CONSTANTS.get(subgroup_size, 0.577)
    d3 = D3_CONSTANTS.get(subgroup_size, 0)
    d4 = D4_CONSTANTS.get(subgroup_size, 2.114)
    
    # X-bar control limits
    ucl_xbar = x_bar_bar + a2 * r_bar
    lcl_xbar = x_bar_bar - a2 * r_bar
    
    # R control limits
    ucl_r = d4 * r_bar
    lcl_r = d3 * r_bar
    
    x = range(1, len(subgroups) + 1)
    
    # X-bar chart
    ax1.plot(x, x_bars, 'bo-', markersize=6, linewidth=1.5, label='X-bar')
    ax1.axhline(x_bar_bar, color='green', linewidth=2, label=f'CL = {x_bar_bar:.4f}')
    ax1.axhline(ucl_xbar, color='red', linestyle='--', linewidth=1.5, label=f'UCL = {ucl_xbar:.4f}')
    ax1.axhline(lcl_xbar, color='red', linestyle='--', linewidth=1.5, label=f'LCL = {lcl_xbar:.4f}')
    
    # Mark out-of-control points
    for i, xb in enumerate(x_bars):
        if xb > ucl_xbar or xb < lcl_xbar:
            ax1.plot(i + 1, xb, 'ro', markersize=10)
    
    ax1.set_ylabel('X-bar', fontsize=11)
    ax1.set_title('X-bar Chart', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # R chart
    ax2.plot(x, ranges, 'bo-', markersize=6, linewidth=1.5, label='Range')
    ax2.axhline(r_bar, color='green', linewidth=2, label=f'CL = {r_bar:.4f}')
    ax2.axhline(ucl_r, color='red', linestyle='--', linewidth=1.5, label=f'UCL = {ucl_r:.4f}')
    if lcl_r > 0:
        ax2.axhline(lcl_r, color='red', linestyle='--', linewidth=1.5, label=f'LCL = {lcl_r:.4f}')
    
    # Mark out-of-control points
    for i, r in enumerate(ranges):
        if r > ucl_r or r < lcl_r:
            ax2.plot(i + 1, r, 'ro', markersize=10)
    
    ax2.set_xlabel('Subgroup', fontsize=11)
    ax2.set_ylabel('Range', fontsize=11)
    ax2.set_title('R Chart', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_probability_plot(stats: Dict) -> str:
    """Create normal probability plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    values = stats['values']
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Calculate theoretical quantiles
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = sp_stats.norm.ppf(probs)
    
    # Plot
    ax.scatter(theoretical, sorted_values, alpha=0.6, color=COLORS['distribution'], 
               edgecolor='white', linewidth=0.5, s=30)
    
    # Fit line
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(theoretical, sorted_values)
    fit_line = slope * theoretical + intercept
    ax.plot(theoretical, fit_line, 'r-', linewidth=2, 
            label=f'Fit line (R² = {r_value**2:.4f})')
    
    ax.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)
    ax.set_title('Normal Probability Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add normality info
    normality = check_normality(values)
    ax.text(0.95, 0.05, f"Shapiro-Wilk p = {normality['shapiro_p']:.4f}\n" +
            f"{'Normal' if normality['is_normal'] else 'Non-normal'}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(indices: Dict, defects: Dict, stats: Dict,
                          normality: Dict, assessment: Dict) -> List[Dict]:
    """Generate key insights for the analysis"""
    insights = []
    
    # Capability assessment
    if indices['cpk'] >= 1.33:
        insights.append({
            'title': f"Process is Capable (Cpk = {indices['cpk']:.2f})",
            'description': 'Cpk ≥ 1.33 indicates the process meets capability requirements.',
            'status': 'positive'
        })
    elif indices['cpk'] >= 1.0:
        insights.append({
            'title': f"Marginal Capability (Cpk = {indices['cpk']:.2f})",
            'description': 'Cpk between 1.0 and 1.33. Process barely meets requirements.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f"Process Not Capable (Cpk = {indices['cpk']:.2f})",
            'description': 'Cpk < 1.0 indicates process does not meet capability requirements.',
            'status': 'warning'
        })
    
    # Centering check
    if indices['cp'] > indices['cpk'] * 1.2:
        insights.append({
            'title': 'Process is Off-Center',
            'description': f"Cp ({indices['cp']:.2f}) >> Cpk ({indices['cpk']:.2f}). Centering the process would improve capability.",
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Process is Well-Centered',
            'description': f"Cp ({indices['cp']:.2f}) ≈ Cpk ({indices['cpk']:.2f}). Process mean is close to target.",
            'status': 'positive'
        })
    
    # Short vs Long term
    if indices['cpk'] > indices['ppk'] * 1.2:
        insights.append({
            'title': 'Long-term Variation Detected',
            'description': f"Cpk ({indices['cpk']:.2f}) >> Ppk ({indices['ppk']:.2f}). Process shows drift over time.",
            'status': 'warning'
        })
    
    # Normality
    if not normality['is_normal']:
        insights.append({
            'title': 'Data May Not Be Normal',
            'description': f"Shapiro-Wilk p = {normality['shapiro_p']:.4f}. Capability indices assume normality.",
            'status': 'neutral'
        })
    
    # Defect rate
    if defects['ppm_total'] < 233:  # ~5 sigma
        insights.append({
            'title': f"Excellent Quality ({defects['ppm_total']:.0f} PPM)",
            'description': f"Expected yield: {defects['yield_percent']:.4f}%",
            'status': 'positive'
        })
    elif defects['ppm_total'] > 66807:  # < 3 sigma
        insights.append({
            'title': f"High Defect Rate ({defects['ppm_total']:.0f} PPM)",
            'description': f"Expected yield: {defects['yield_percent']:.2f}%. Improvement needed.",
            'status': 'warning'
        })
    
    return insights


# ============ MAIN API ENDPOINT ============
@router.post("/capability")
async def run_capability_analysis(request: CapabilityRequest) -> Dict[str, Any]:
    """
    Run Process Capability Analysis
    
    Calculates Cp, Cpk, Pp, Ppk and related metrics for process capability assessment.
    """
    try:
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        
        if request.measurement_col not in df.columns:
            raise HTTPException(status_code=400, 
                              detail=f"Measurement column '{request.measurement_col}' not found")
        
        # Convert measurement column to numeric
        df[request.measurement_col] = pd.to_numeric(df[request.measurement_col], errors='coerce')
        df = df.dropna(subset=[request.measurement_col])
        
        if len(df) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid measurements")
        
        # Validate spec limits
        if request.usl <= request.lsl:
            raise HTTPException(status_code=400, detail="USL must be greater than LSL")
        
        # Set target to midpoint if not specified
        target = request.target if request.target is not None else (request.usl + request.lsl) / 2
        
        # Calculate statistics
        calc_stats = calculate_subgroup_stats(
            df, request.measurement_col, 
            request.subgroup_col, request.subgroup_size
        )
        
        # Calculate capability indices
        indices = calculate_capability_indices(calc_stats, request.usl, request.lsl, target)
        
        # Calculate defect metrics
        defects = calculate_defect_metrics(calc_stats, request.usl, request.lsl)
        
        # Check normality
        normality = check_normality(calc_stats['values'])
        
        # Generate assessment
        assessment = assess_capability(indices)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Create visualizations
        visualizations = {
            'histogram': create_histogram(calc_stats, request.usl, request.lsl, target, indices),
            'capability_chart': create_capability_chart(calc_stats, indices, request.usl, request.lsl, target),
            'control_chart': create_control_chart(calc_stats, request.usl, request.lsl),
            'probability_plot': create_probability_plot(calc_stats),
        }
        
        # Generate insights
        key_insights = generate_key_insights(indices, defects, calc_stats, normality, assessment)
        
        # Prepare stats for response (remove numpy arrays)
        response_stats = {
            'mean': calc_stats['mean'],
            'std_within': calc_stats['std_within'],
            'std_overall': calc_stats['std_overall'],
            'min': calc_stats['min'],
            'max': calc_stats['max'],
            'range': calc_stats['range'],
            'n': calc_stats['n'],
            'subgroup_size': calc_stats['subgroup_size'],
            'num_subgroups': calc_stats['num_subgroups'],
        }
        
        # Build results
        results = {
            'indices': {k: _to_native_type(v) for k, v in indices.items()},
            'stats': {k: _to_native_type(v) for k, v in response_stats.items()},
            'defects': {k: _to_native_type(v) for k, v in defects.items()},
            'specifications': {
                'usl': request.usl,
                'lsl': request.lsl,
                'target': target,
                'tolerance': request.usl - request.lsl,
            },
            'normality': {k: _to_native_type(v) for k, v in normality.items()},
            'assessment': assessment,
        }
        
        summary = {
            'cpk': indices['cpk'],
            'ppk': indices['ppk'],
            'sigma_level': defects['sigma_level'],
            'assessment': assessment['short_term'],
            'solve_time_ms': solve_time_ms,
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capability analysis failed: {str(e)}")
