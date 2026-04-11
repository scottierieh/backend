"""
Process Capability Analysis Router for FastAPI
Cp, Cpk, Pp, Ppk, Cpm calculations with normality tests
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
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CapabilityRequest(BaseModel):
    data: List[Dict[str, Any]]
    measurement_col: str
    usl: Optional[float] = None  # Upper Specification Limit
    lsl: Optional[float] = None  # Lower Specification Limit
    target: Optional[float] = None  # Target value
    subgroup_col: Optional[str] = None  # For within-subgroup variation
    subgroup_size: int = 5
    group_col: Optional[str] = None  # For comparison
    # Auto-suggest specs if not provided
    auto_specs: bool = False
    auto_spec_sigma: float = 3.0  # For auto-suggested specs


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


def estimate_within_sigma(data: np.ndarray, subgroup_size: int = 5) -> float:
    """Estimate within-subgroup standard deviation using R-bar/d2 method"""
    d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
                7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 15: 3.472, 20: 3.735}
    
    n_subgroups = len(data) // subgroup_size
    if n_subgroups < 2:
        # Fall back to overall std if not enough subgroups
        return np.std(data, ddof=1)
    
    # Calculate ranges within subgroups
    subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
    ranges = np.ptp(subgroups, axis=1)
    r_bar = np.mean(ranges)
    
    # Get d2 constant
    d2 = d2_table.get(subgroup_size, 2.326)
    
    return r_bar / d2


def calculate_capability_indices(data: np.ndarray, usl: float, lsl: float,
                                 target: Optional[float] = None,
                                 subgroup_size: int = 5) -> Dict[str, Any]:
    """Calculate all process capability indices"""
    
    n = len(data)
    mean = np.mean(data)
    overall_std = np.std(data, ddof=1)  # Sample std for Pp, Ppk
    within_std = estimate_within_sigma(data, subgroup_size)  # For Cp, Cpk
    
    # Specification width
    spec_width = usl - lsl
    
    # Target (default to midpoint)
    if target is None:
        target = (usl + lsl) / 2
    
    # Cp (Potential Capability) - uses within-subgroup variation
    cp = spec_width / (6 * within_std) if within_std > 0 else None
    
    # Cpk (Actual Capability) - uses within-subgroup variation
    cpu = (usl - mean) / (3 * within_std) if within_std > 0 else None
    cpl = (mean - lsl) / (3 * within_std) if within_std > 0 else None
    cpk = min(cpu, cpl) if cpu is not None and cpl is not None else None
    
    # Pp (Potential Performance) - uses overall variation
    pp = spec_width / (6 * overall_std) if overall_std > 0 else None
    
    # Ppk (Actual Performance) - uses overall variation
    ppu = (usl - mean) / (3 * overall_std) if overall_std > 0 else None
    ppl = (mean - lsl) / (3 * overall_std) if overall_std > 0 else None
    ppk = min(ppu, ppl) if ppu is not None and ppl is not None else None
    
    # Cpm (Taguchi Capability) - accounts for deviation from target
    deviation_from_target = np.sqrt(np.mean((data - target) ** 2))
    cpm = spec_width / (6 * deviation_from_target) if deviation_from_target > 0 else None
    
    # Expected PPM (Parts Per Million) defective
    if within_std > 0:
        z_upper = (usl - mean) / within_std
        z_lower = (mean - lsl) / within_std
        ppm_upper = (1 - stats.norm.cdf(z_upper)) * 1000000
        ppm_lower = stats.norm.cdf(-z_lower) * 1000000
        ppm_total = ppm_upper + ppm_lower
    else:
        ppm_upper = ppm_lower = ppm_total = 0
    
    # Observed PPM (actual out-of-spec)
    out_of_spec = np.sum((data < lsl) | (data > usl))
    observed_ppm = (out_of_spec / n) * 1000000
    
    # Percent out of spec
    pct_below_lsl = (np.sum(data < lsl) / n) * 100
    pct_above_usl = (np.sum(data > usl) / n) * 100
    pct_out_of_spec = pct_below_lsl + pct_above_usl
    
    # Z-scores (sigma levels)
    z_bench = stats.norm.ppf(1 - ppm_total / 1000000) if ppm_total < 1000000 else 0
    z_bench_lt = z_bench + 1.5  # Long-term with 1.5 sigma shift
    
    return {
        'n': n,
        'mean': _to_native_type(mean),
        'std_within': _to_native_type(within_std),
        'std_overall': _to_native_type(overall_std),
        'usl': _to_native_type(usl),
        'lsl': _to_native_type(lsl),
        'target': _to_native_type(target),
        'cp': _to_native_type(cp),
        'cpk': _to_native_type(cpk),
        'cpu': _to_native_type(cpu),
        'cpl': _to_native_type(cpl),
        'pp': _to_native_type(pp),
        'ppk': _to_native_type(ppk),
        'ppu': _to_native_type(ppu),
        'ppl': _to_native_type(ppl),
        'cpm': _to_native_type(cpm),
        'ppm_upper': _to_native_type(ppm_upper),
        'ppm_lower': _to_native_type(ppm_lower),
        'ppm_total': _to_native_type(ppm_total),
        'observed_ppm': _to_native_type(observed_ppm),
        'pct_below_lsl': _to_native_type(pct_below_lsl),
        'pct_above_usl': _to_native_type(pct_above_usl),
        'pct_out_of_spec': _to_native_type(pct_out_of_spec),
        'z_bench': _to_native_type(z_bench),
        'z_bench_lt': _to_native_type(z_bench_lt),
        'out_of_spec_count': int(out_of_spec)
    }


def perform_normality_tests(data: np.ndarray) -> Dict[str, Any]:
    """Perform normality tests on data"""
    results = {}
    
    # Shapiro-Wilk (best for n < 5000)
    if len(data) <= 5000:
        try:
            stat, p_value = stats.shapiro(data[:5000])
            results['shapiro_wilk'] = {
                'statistic': _to_native_type(stat),
                'p_value': _to_native_type(p_value),
                'normal': p_value > 0.05
            }
        except:
            pass
    
    # Anderson-Darling
    try:
        result = stats.anderson(data, dist='norm')
        # Using 5% significance level
        critical_value = result.critical_values[2]  # 5% level
        results['anderson_darling'] = {
            'statistic': _to_native_type(result.statistic),
            'critical_value': _to_native_type(critical_value),
            'normal': result.statistic < critical_value
        }
    except:
        pass
    
    # D'Agostino-Pearson (requires n >= 20)
    if len(data) >= 20:
        try:
            stat, p_value = stats.normaltest(data)
            results['dagostino_pearson'] = {
                'statistic': _to_native_type(stat),
                'p_value': _to_native_type(p_value),
                'normal': p_value > 0.05
            }
        except:
            pass
    
    # Skewness and Kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    results['descriptive'] = {
        'skewness': _to_native_type(skewness),
        'kurtosis': _to_native_type(kurtosis),
        'skewness_normal': abs(skewness) < 1,
        'kurtosis_normal': abs(kurtosis) < 2
    }
    
    # Overall normality assessment
    normal_count = sum([
        results.get('shapiro_wilk', {}).get('normal', False),
        results.get('anderson_darling', {}).get('normal', False),
        results.get('dagostino_pearson', {}).get('normal', False)
    ])
    results['is_normal'] = normal_count >= 2
    
    return results


def create_capability_histogram(data: np.ndarray, usl: float, lsl: float,
                                target: float, mean: float, std: float) -> str:
    """Create histogram with normal curve and spec limits"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    n_bins = min(50, max(20, len(data) // 20))
    n, bins, patches = ax.hist(data, bins=n_bins, density=True, alpha=0.7,
                                color='steelblue', edgecolor='white', linewidth=0.5)
    
    # Color bins outside specs
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        if right < lsl or left > usl:
            patches[i].set_facecolor('#ef4444')
    
    # Normal curve
    x = np.linspace(min(data.min(), lsl - std), max(data.max(), usl + std), 200)
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y, 'b-', linewidth=2, label=f'Normal (μ={mean:.3f}, σ={std:.3f})')
    
    # Spec limits
    ax.axvline(x=lsl, color='red', linestyle='--', linewidth=2, label=f'LSL = {lsl:.3f}')
    ax.axvline(x=usl, color='red', linestyle='--', linewidth=2, label=f'USL = {usl:.3f}')
    ax.axvline(x=target, color='green', linestyle='-', linewidth=2, label=f'Target = {target:.3f}')
    ax.axvline(x=mean, color='blue', linestyle='-', linewidth=2, label=f'Mean = {mean:.3f}')
    
    # Fill areas outside specs
    x_lower = np.linspace(min(data.min(), lsl - 3*std), lsl, 100)
    x_upper = np.linspace(usl, max(data.max(), usl + 3*std), 100)
    ax.fill_between(x_lower, stats.norm.pdf(x_lower, mean, std), alpha=0.3, color='red')
    ax.fill_between(x_upper, stats.norm.pdf(x_upper, mean, std), alpha=0.3, color='red')
    
    ax.set_xlabel('Measurement Value')
    ax.set_ylabel('Density')
    ax.set_title('Process Capability Histogram', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_capability_summary_chart(indices: Dict[str, Any]) -> str:
    """Create capability indices summary chart"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    metrics = [
        ('Cp', indices['cp'], 1.33),
        ('Cpk', indices['cpk'], 1.33),
        ('Pp', indices['pp'], 1.33),
        ('Ppk', indices['ppk'], 1.33)
    ]
    
    for ax, (name, value, target) in zip(axes, metrics):
        if value is None:
            value = 0
        
        # Color based on value
        if value >= 1.67:
            color = '#22c55e'  # Excellent
        elif value >= 1.33:
            color = '#3b82f6'  # Good
        elif value >= 1.0:
            color = '#f59e0b'  # Marginal
        else:
            color = '#ef4444'  # Poor
        
        # Create gauge-like visualization
        sizes = [min(value / 2, 1) * 100, max(100 - min(value / 2, 1) * 100, 0)]
        ax.pie(sizes, colors=[color, '#e5e7eb'], startangle=90,
               wedgeprops=dict(width=0.3, edgecolor='white'))
        
        ax.text(0, 0.1, f'{value:.2f}', ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)
        ax.text(0, -0.25, name, ha='center', va='center', fontsize=12, color='#374151')
        
        # Target line indicator
        if value >= target:
            ax.text(0, -0.45, '✓ ≥1.33', ha='center', fontsize=9, color='green')
        else:
            ax.text(0, -0.45, f'Target: {target}', ha='center', fontsize=9, color='gray')
    
    plt.suptitle('Process Capability Indices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_normality_plot(data: np.ndarray) -> str:
    """Create Q-Q plot for normality assessment"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=ax1)
    ax1.set_title('Normal Probability Plot (Q-Q)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram with normal overlay
    ax2.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    x = np.linspace(data.min(), data.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, np.mean(data), np.std(data)), 'r-', linewidth=2, label='Normal fit')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution vs Normal', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_comparison_chart(comparison_data: List[Dict], group_col: str) -> str:
    """Create group comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups = [d[group_col] for d in comparison_data]
    cpk = [d['cpk'] for d in comparison_data]
    ppk = [d['ppk'] for d in comparison_data]
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cpk, width, label='Cpk', color='#3b82f6')
    bars2 = ax.bar(x + width/2, ppk, width, label='Ppk', color='#22c55e')
    
    # Target lines
    ax.axhline(y=1.33, color='orange', linestyle='--', label='Target (1.33)')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Minimum (1.0)')
    
    ax.set_ylabel('Capability Index')
    ax.set_xlabel(group_col)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_title(f'Capability Comparison by {group_col}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_within_vs_overall_chart(indices: Dict[str, Any]) -> str:
    """Create comparison of within vs overall variation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sigma comparison
    sigmas = ['Within (σ_w)', 'Overall (σ_o)']
    values = [indices['std_within'], indices['std_overall']]
    colors = ['#3b82f6', '#22c55e']
    
    ax1.bar(sigmas, values, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Within vs Overall Variation', fontsize=12, fontweight='bold')
    for i, v in enumerate(values):
        ax1.text(i, v + 0.01 * max(values), f'{v:.4f}', ha='center', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Capability comparison
    cap_labels = ['Cp', 'Cpk', 'Pp', 'Ppk']
    cap_values = [indices['cp'] or 0, indices['cpk'] or 0, 
                  indices['pp'] or 0, indices['ppk'] or 0]
    colors = ['#3b82f6', '#60a5fa', '#22c55e', '#86efac']
    
    bars = ax2.bar(cap_labels, cap_values, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(y=1.33, color='orange', linestyle='--', label='Target (1.33)')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Minimum (1.0)')
    ax2.set_ylabel('Capability Index')
    ax2.set_title('Capability vs Performance', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    for i, v in enumerate(cap_values):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(indices: Dict[str, Any], normality: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate key insights for capability analysis"""
    insights = []
    
    cpk = indices.get('cpk')
    ppk = indices.get('ppk')
    cp = indices.get('cp')
    
    # Overall capability assessment
    if cpk is not None:
        if cpk >= 1.67:
            insights.append({
                'title': 'Excellent Process Capability',
                'description': f'Cpk = {cpk:.2f} (≥1.67). Process is highly capable with very low defect risk.',
                'status': 'positive'
            })
        elif cpk >= 1.33:
            insights.append({
                'title': 'Good Process Capability',
                'description': f'Cpk = {cpk:.2f} (≥1.33). Process meets typical industry requirements.',
                'status': 'positive'
            })
        elif cpk >= 1.0:
            insights.append({
                'title': 'Marginal Process Capability',
                'description': f'Cpk = {cpk:.2f} (1.0-1.33). Process is barely capable, improvement recommended.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Poor Process Capability',
                'description': f'Cpk = {cpk:.2f} (<1.0). Process is not capable, immediate action required.',
                'status': 'warning'
            })
    
    # Centering assessment
    if cp is not None and cpk is not None:
        centering_ratio = cpk / cp if cp > 0 else 0
        if centering_ratio >= 0.9:
            insights.append({
                'title': 'Well Centered Process',
                'description': f'Cpk/Cp = {centering_ratio:.2f}. Process is well centered between spec limits.',
                'status': 'positive'
            })
        else:
            insights.append({
                'title': 'Process Not Centered',
                'description': f'Cpk/Cp = {centering_ratio:.2f}. Process mean is shifted from target. Centering could improve capability.',
                'status': 'neutral'
            })
    
    # Stability assessment (Cp vs Pp, Cpk vs Ppk)
    if cp is not None and indices.get('pp') is not None:
        stability_ratio = cp / indices['pp'] if indices['pp'] > 0 else 0
        if 0.9 <= stability_ratio <= 1.1:
            insights.append({
                'title': 'Stable Process',
                'description': f'Cp/Pp ≈ 1.0. Within and overall variation are similar, process is stable.',
                'status': 'positive'
            })
        elif stability_ratio > 1.1:
            insights.append({
                'title': 'Process Stability Concern',
                'description': f'Cp/Pp = {stability_ratio:.2f} (>1.1). Between-subgroup variation present, investigate sources.',
                'status': 'warning'
            })
    
    # Normality assessment
    if normality.get('is_normal'):
        insights.append({
            'title': 'Data is Normally Distributed',
            'description': 'Normality tests passed. Capability indices are reliable.',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': 'Non-Normal Data',
            'description': 'Data may not be normally distributed. Consider transformation or non-parametric methods.',
            'status': 'neutral'
        })
    
    # PPM insight
    ppm = indices.get('ppm_total', 0)
    if ppm is not None:
        if ppm <= 3.4:
            insights.append({
                'title': 'Six Sigma Level',
                'description': f'Expected PPM = {ppm:.1f}. Process achieves Six Sigma quality.',
                'status': 'positive'
            })
        elif ppm <= 233:
            insights.append({
                'title': 'High Quality Level',
                'description': f'Expected PPM = {ppm:.0f}. Very low defect rate expected.',
                'status': 'positive'
            })
        elif ppm > 66800:
            insights.append({
                'title': 'High Defect Risk',
                'description': f'Expected PPM = {ppm:,.0f}. Significant defect rate expected.',
                'status': 'warning'
            })
    
    return insights


@router.post("/capability")
async def run_capability_analysis(request: CapabilityRequest) -> Dict[str, Any]:
    """
    Perform Process Capability Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        if request.measurement_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{request.measurement_col}' not found")
        
        data = pd.to_numeric(df[request.measurement_col], errors='coerce').dropna().values
        
        if len(data) < 30:
            raise HTTPException(status_code=400, detail="Need at least 30 data points for reliable capability analysis")
        
        # Determine spec limits
        usl = request.usl
        lsl = request.lsl
        
        if request.auto_specs and (usl is None or lsl is None):
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            if usl is None:
                usl = mean + request.auto_spec_sigma * std
            if lsl is None:
                lsl = mean - request.auto_spec_sigma * std
        
        if usl is None or lsl is None:
            raise HTTPException(status_code=400, detail="Both USL and LSL are required (or enable auto_specs)")
        
        if usl <= lsl:
            raise HTTPException(status_code=400, detail="USL must be greater than LSL")
        
        # Target
        target = request.target if request.target is not None else (usl + lsl) / 2
        
        # Calculate capability indices
        indices = calculate_capability_indices(data, usl, lsl, target, request.subgroup_size)
        
        # Normality tests
        normality = perform_normality_tests(data)
        
        # Create visualizations
        visualizations = {}
        visualizations['histogram'] = create_capability_histogram(
            data, usl, lsl, target, indices['mean'], indices['std_overall']
        )
        visualizations['summary_chart'] = create_capability_summary_chart(indices)
        visualizations['normality_plot'] = create_normality_plot(data)
        visualizations['variation_chart'] = create_within_vs_overall_chart(indices)
        
        # Group comparison if specified
        comparison_data = None
        if request.group_col and request.group_col in df.columns:
            comparison_data = []
            for group in df[request.group_col].unique():
                group_data = pd.to_numeric(
                    df[df[request.group_col] == group][request.measurement_col],
                    errors='coerce'
                ).dropna().values
                
                if len(group_data) >= 10:
                    group_indices = calculate_capability_indices(
                        group_data, usl, lsl, target, request.subgroup_size
                    )
                    comparison_data.append({
                        request.group_col: str(group),
                        'cpk': group_indices['cpk'],
                        'ppk': group_indices['ppk'],
                        'cp': group_indices['cp'],
                        'pp': group_indices['pp'],
                        'n': group_indices['n']
                    })
            
            if comparison_data:
                visualizations['comparison_chart'] = create_comparison_chart(
                    comparison_data, request.group_col
                )
        
        # Generate insights
        insights = generate_key_insights(indices, normality)
        
        # Summary
        summary = {
            'cpk': indices['cpk'],
            'ppk': indices['ppk'],
            'cp': indices['cp'],
            'pp': indices['pp'],
            'is_capable': (indices['cpk'] or 0) >= 1.0,
            'meets_target': (indices['cpk'] or 0) >= 1.33,
            'is_normal': normality.get('is_normal', False),
            'ppm_total': indices['ppm_total'],
            'pct_out_of_spec': indices['pct_out_of_spec']
        }
        
        return {
            'success': True,
            'indices': indices,
            'normality': normality,
            'comparison_data': comparison_data,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capability analysis failed: {str(e)}")
