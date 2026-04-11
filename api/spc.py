"""
Statistical Process Control (SPC) Charts Router for FastAPI
X-bar/R, X-bar/S, I-MR, P, NP, C, U charts with Western Electric rules
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class SPCRequest(BaseModel):
    data: List[Dict[str, Any]]
    measurement_col: str
    subgroup_col: Optional[str] = None
    subgroup_size: int = 5
    chart_type: Literal["xbar_r", "xbar_s", "i_mr", "p", "np", "c", "u"] = "xbar_r"
    sample_size_col: Optional[str] = None
    defects_col: Optional[str] = None
    sigma_limit: float = 3.0
    use_specified_limits: bool = False
    specified_ucl: Optional[float] = None
    specified_lcl: Optional[float] = None
    specified_cl: Optional[float] = None


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


# Control chart constants
D2_TABLE = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704,
            8: 2.847, 9: 2.970, 10: 3.078, 15: 3.472, 20: 3.735, 25: 3.931}
D3_TABLE = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184,
            10: 0.223, 15: 0.348, 20: 0.414, 25: 0.459}
D4_TABLE = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924,
            8: 1.864, 9: 1.816, 10: 1.777, 15: 1.652, 20: 1.586, 25: 1.541}
A2_TABLE = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419,
            8: 0.373, 9: 0.337, 10: 0.308, 15: 0.223, 20: 0.180, 25: 0.153}
A3_TABLE = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182,
            8: 1.099, 9: 1.032, 10: 0.975, 15: 0.789, 20: 0.680, 25: 0.606}
B3_TABLE = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030, 7: 0.118, 8: 0.185,
            9: 0.239, 10: 0.284, 15: 0.428, 20: 0.510, 25: 0.565}
B4_TABLE = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882,
            8: 1.815, 9: 1.761, 10: 1.716, 15: 1.572, 20: 1.490, 25: 1.435}
C4_TABLE = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515,
            7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727}


def get_constant(table: dict, n: int) -> float:
    """Get constant from table, interpolating if necessary"""
    if n in table:
        return table[n]
    keys = sorted(table.keys())
    if n < keys[0]:
        return table[keys[0]]
    if n > keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] < n < keys[i + 1]:
            ratio = (n - keys[i]) / (keys[i + 1] - keys[i])
            return table[keys[i]] + ratio * (table[keys[i + 1]] - table[keys[i]])
    return table[keys[-1]]


def calculate_xbar_r(data: np.ndarray, subgroup_size: int, sigma: float = 3.0) -> Dict[str, Any]:
    """Calculate X-bar and R chart limits"""
    n_subgroups = len(data) // subgroup_size
    if n_subgroups < 2:
        raise ValueError("Not enough data for subgroups")
    
    subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
    xbar = np.mean(subgroups, axis=1)
    ranges = np.ptp(subgroups, axis=1)
    
    xbar_bar = np.mean(xbar)
    r_bar = np.mean(ranges)
    
    A2 = get_constant(A2_TABLE, subgroup_size)
    D3 = get_constant(D3_TABLE, subgroup_size)
    D4 = get_constant(D4_TABLE, subgroup_size)
    
    # X-bar chart limits
    xbar_ucl = xbar_bar + (sigma / 3) * A2 * r_bar
    xbar_lcl = xbar_bar - (sigma / 3) * A2 * r_bar
    
    # R chart limits
    r_ucl = D4 * r_bar
    r_lcl = D3 * r_bar
    
    return {
        'xbar': {'data': xbar.tolist(), 'cl': xbar_bar, 'ucl': xbar_ucl, 'lcl': xbar_lcl},
        'r': {'data': ranges.tolist(), 'cl': r_bar, 'ucl': r_ucl, 'lcl': r_lcl},
        'subgroup_size': subgroup_size,
        'n_subgroups': n_subgroups
    }


def calculate_xbar_s(data: np.ndarray, subgroup_size: int, sigma: float = 3.0) -> Dict[str, Any]:
    """Calculate X-bar and S chart limits"""
    n_subgroups = len(data) // subgroup_size
    if n_subgroups < 2:
        raise ValueError("Not enough data for subgroups")
    
    subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
    xbar = np.mean(subgroups, axis=1)
    stds = np.std(subgroups, axis=1, ddof=1)
    
    xbar_bar = np.mean(xbar)
    s_bar = np.mean(stds)
    
    A3 = get_constant(A3_TABLE, subgroup_size)
    B3 = get_constant(B3_TABLE, subgroup_size)
    B4 = get_constant(B4_TABLE, subgroup_size)
    
    # X-bar chart limits
    xbar_ucl = xbar_bar + (sigma / 3) * A3 * s_bar
    xbar_lcl = xbar_bar - (sigma / 3) * A3 * s_bar
    
    # S chart limits
    s_ucl = B4 * s_bar
    s_lcl = B3 * s_bar
    
    return {
        'xbar': {'data': xbar.tolist(), 'cl': xbar_bar, 'ucl': xbar_ucl, 'lcl': xbar_lcl},
        's': {'data': stds.tolist(), 'cl': s_bar, 'ucl': s_ucl, 'lcl': s_lcl},
        'subgroup_size': subgroup_size,
        'n_subgroups': n_subgroups
    }


def calculate_imr(data: np.ndarray, sigma: float = 3.0) -> Dict[str, Any]:
    """Calculate Individual and Moving Range chart limits"""
    individuals = data
    moving_ranges = np.abs(np.diff(data))
    
    x_bar = np.mean(individuals)
    mr_bar = np.mean(moving_ranges)
    
    d2 = get_constant(D2_TABLE, 2)
    D4 = get_constant(D4_TABLE, 2)
    D3 = get_constant(D3_TABLE, 2)
    
    # I chart limits
    i_ucl = x_bar + (sigma / 3) * (mr_bar / d2) * 3
    i_lcl = x_bar - (sigma / 3) * (mr_bar / d2) * 3
    
    # MR chart limits
    mr_ucl = D4 * mr_bar
    mr_lcl = D3 * mr_bar
    
    return {
        'i': {'data': individuals.tolist(), 'cl': x_bar, 'ucl': i_ucl, 'lcl': i_lcl},
        'mr': {'data': moving_ranges.tolist(), 'cl': mr_bar, 'ucl': mr_ucl, 'lcl': mr_lcl},
        'n_points': len(individuals)
    }


def calculate_p_chart(defectives: np.ndarray, sample_sizes: np.ndarray, sigma: float = 3.0) -> Dict[str, Any]:
    """Calculate p chart (proportion defective) limits"""
    p = defectives / sample_sizes
    p_bar = np.sum(defectives) / np.sum(sample_sizes)
    n_bar = np.mean(sample_sizes)
    
    # Variable limits for each point
    ucl = p_bar + sigma * np.sqrt(p_bar * (1 - p_bar) / sample_sizes)
    lcl = np.maximum(0, p_bar - sigma * np.sqrt(p_bar * (1 - p_bar) / sample_sizes))
    
    # Average limits
    ucl_avg = p_bar + sigma * np.sqrt(p_bar * (1 - p_bar) / n_bar)
    lcl_avg = max(0, p_bar - sigma * np.sqrt(p_bar * (1 - p_bar) / n_bar))
    
    return {
        'p': {'data': p.tolist(), 'cl': p_bar, 'ucl': ucl.tolist(), 'lcl': lcl.tolist(),
              'ucl_avg': ucl_avg, 'lcl_avg': lcl_avg},
        'sample_sizes': sample_sizes.tolist(),
        'n_samples': len(defectives)
    }


def calculate_np_chart(defectives: np.ndarray, sample_size: int, sigma: float = 3.0) -> Dict[str, Any]:
    """Calculate np chart (number defective) limits"""
    np_bar = np.mean(defectives)
    p_bar = np_bar / sample_size
    
    ucl = np_bar + sigma * np.sqrt(np_bar * (1 - p_bar))
    lcl = max(0, np_bar - sigma * np.sqrt(np_bar * (1 - p_bar)))
    
    return {
        'np': {'data': defectives.tolist(), 'cl': np_bar, 'ucl': ucl, 'lcl': lcl},
        'sample_size': sample_size,
        'p_bar': p_bar,
        'n_samples': len(defectives)
    }


def calculate_c_chart(defects: np.ndarray, sigma: float = 3.0) -> Dict[str, Any]:
    """Calculate c chart (count of defects) limits"""
    c_bar = np.mean(defects)
    
    ucl = c_bar + sigma * np.sqrt(c_bar)
    lcl = max(0, c_bar - sigma * np.sqrt(c_bar))
    
    return {
        'c': {'data': defects.tolist(), 'cl': c_bar, 'ucl': ucl, 'lcl': lcl},
        'n_samples': len(defects)
    }


def calculate_u_chart(defects: np.ndarray, sample_sizes: np.ndarray, sigma: float = 3.0) -> Dict[str, Any]:
    """Calculate u chart (defects per unit) limits"""
    u = defects / sample_sizes
    u_bar = np.sum(defects) / np.sum(sample_sizes)
    n_bar = np.mean(sample_sizes)
    
    # Variable limits
    ucl = u_bar + sigma * np.sqrt(u_bar / sample_sizes)
    lcl = np.maximum(0, u_bar - sigma * np.sqrt(u_bar / sample_sizes))
    
    # Average limits
    ucl_avg = u_bar + sigma * np.sqrt(u_bar / n_bar)
    lcl_avg = max(0, u_bar - sigma * np.sqrt(u_bar / n_bar))
    
    return {
        'u': {'data': u.tolist(), 'cl': u_bar, 'ucl': ucl.tolist(), 'lcl': lcl.tolist(),
              'ucl_avg': ucl_avg, 'lcl_avg': lcl_avg},
        'sample_sizes': sample_sizes.tolist(),
        'n_samples': len(defects)
    }


def check_western_electric_rules(data: np.ndarray, cl: float, ucl: float, lcl: float) -> List[Dict[str, Any]]:
    """Check Western Electric rules for out-of-control conditions"""
    violations = []
    sigma = (ucl - cl) / 3
    
    one_sigma_upper = cl + sigma
    one_sigma_lower = cl - sigma
    two_sigma_upper = cl + 2 * sigma
    two_sigma_lower = cl - 2 * sigma
    
    for i, point in enumerate(data):
        point_violations = []
        
        # Rule 1: Point beyond 3 sigma
        if point > ucl or point < lcl:
            point_violations.append({
                'rule': 1,
                'description': 'Point beyond 3σ limit',
                'severity': 'high'
            })
        
        # Rule 2: 2 of 3 consecutive points beyond 2 sigma (same side)
        if i >= 2:
            recent = data[i-2:i+1]
            above_2sigma = np.sum(recent > two_sigma_upper)
            below_2sigma = np.sum(recent < two_sigma_lower)
            if above_2sigma >= 2 or below_2sigma >= 2:
                point_violations.append({
                    'rule': 2,
                    'description': '2 of 3 points beyond 2σ',
                    'severity': 'medium'
                })
        
        # Rule 3: 4 of 5 consecutive points beyond 1 sigma (same side)
        if i >= 4:
            recent = data[i-4:i+1]
            above_1sigma = np.sum(recent > one_sigma_upper)
            below_1sigma = np.sum(recent < one_sigma_lower)
            if above_1sigma >= 4 or below_1sigma >= 4:
                point_violations.append({
                    'rule': 3,
                    'description': '4 of 5 points beyond 1σ',
                    'severity': 'medium'
                })
        
        # Rule 4: 8 consecutive points on same side of center
        if i >= 7:
            recent = data[i-7:i+1]
            if np.all(recent > cl) or np.all(recent < cl):
                point_violations.append({
                    'rule': 4,
                    'description': '8 consecutive points same side',
                    'severity': 'medium'
                })
        
        # Rule 5: 6 consecutive points increasing or decreasing
        if i >= 5:
            recent = data[i-5:i+1]
            diffs = np.diff(recent)
            if np.all(diffs > 0) or np.all(diffs < 0):
                point_violations.append({
                    'rule': 5,
                    'description': '6 consecutive points trending',
                    'severity': 'low'
                })
        
        # Rule 6: 14 consecutive points alternating up and down
        if i >= 13:
            recent = data[i-13:i+1]
            diffs = np.diff(recent)
            signs = np.sign(diffs)
            if np.all(np.diff(signs) != 0):
                point_violations.append({
                    'rule': 6,
                    'description': '14 points alternating',
                    'severity': 'low'
                })
        
        # Rule 7: 15 consecutive points within 1 sigma (stratification)
        if i >= 14:
            recent = data[i-14:i+1]
            if np.all((recent > one_sigma_lower) & (recent < one_sigma_upper)):
                point_violations.append({
                    'rule': 7,
                    'description': '15 points within 1σ (stratification)',
                    'severity': 'low'
                })
        
        if point_violations:
            violations.append({
                'point_index': i,
                'value': _to_native_type(point),
                'violations': point_violations
            })
    
    return violations


def create_control_chart(chart_data: Dict[str, Any], chart_type: str, title: str) -> str:
    """Create control chart visualization"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    if chart_type == "xbar_r":
        primary_key, secondary_key = 'xbar', 'r'
        primary_label, secondary_label = 'X-bar', 'Range'
    elif chart_type == "xbar_s":
        primary_key, secondary_key = 'xbar', 's'
        primary_label, secondary_label = 'X-bar', 'Std Dev'
    elif chart_type == "i_mr":
        primary_key, secondary_key = 'i', 'mr'
        primary_label, secondary_label = 'Individual', 'Moving Range'
    else:
        primary_key = chart_type.split('_')[0] if '_' in chart_type else chart_type
        secondary_key = None
        primary_label = chart_type.upper()
        secondary_label = None
    
    # Primary chart
    ax1 = axes[0]
    primary = chart_data[primary_key]
    data = primary['data']
    x = range(len(data))
    
    ax1.plot(x, data, 'b-o', markersize=4, linewidth=1)
    ax1.axhline(y=primary['cl'], color='g', linestyle='-', linewidth=2, label=f"CL = {primary['cl']:.4f}")
    
    # Handle variable limits (for p and u charts)
    if isinstance(primary['ucl'], list):
        ax1.plot(x, primary['ucl'], 'r--', linewidth=1, label='UCL')
        ax1.plot(x, primary['lcl'], 'r--', linewidth=1, label='LCL')
        ax1.axhline(y=primary.get('ucl_avg', np.mean(primary['ucl'])), color='r', linestyle=':', alpha=0.5)
        ax1.axhline(y=primary.get('lcl_avg', np.mean(primary['lcl'])), color='r', linestyle=':', alpha=0.5)
    else:
        ax1.axhline(y=primary['ucl'], color='r', linestyle='--', linewidth=1.5, label=f"UCL = {primary['ucl']:.4f}")
        ax1.axhline(y=primary['lcl'], color='r', linestyle='--', linewidth=1.5, label=f"LCL = {primary['lcl']:.4f}")
    
    # Mark out-of-control points
    ucl = primary['ucl'] if not isinstance(primary['ucl'], list) else primary['ucl']
    lcl = primary['lcl'] if not isinstance(primary['lcl'], list) else primary['lcl']
    
    if not isinstance(ucl, list):
        ooc_indices = [i for i, v in enumerate(data) if v > ucl or v < lcl]
        if ooc_indices:
            ax1.scatter([x[i] for i in ooc_indices], [data[i] for i in ooc_indices], 
                       color='red', s=100, zorder=5, marker='o', edgecolors='black')
    
    ax1.set_ylabel(primary_label)
    ax1.set_title(f'{title} - {primary_label} Chart', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Secondary chart
    if secondary_key and secondary_key in chart_data:
        ax2 = axes[1]
        secondary = chart_data[secondary_key]
        sec_data = secondary['data']
        sec_x = range(len(sec_data))
        
        ax2.plot(sec_x, sec_data, 'b-o', markersize=4, linewidth=1)
        ax2.axhline(y=secondary['cl'], color='g', linestyle='-', linewidth=2, label=f"CL = {secondary['cl']:.4f}")
        ax2.axhline(y=secondary['ucl'], color='r', linestyle='--', linewidth=1.5, label=f"UCL = {secondary['ucl']:.4f}")
        ax2.axhline(y=secondary['lcl'], color='r', linestyle='--', linewidth=1.5, label=f"LCL = {secondary['lcl']:.4f}")
        
        # Mark out-of-control points
        ooc_indices = [i for i, v in enumerate(sec_data) if v > secondary['ucl'] or v < secondary['lcl']]
        if ooc_indices:
            ax2.scatter([sec_x[i] for i in ooc_indices], [sec_data[i] for i in ooc_indices], 
                       color='red', s=100, zorder=5, marker='o', edgecolors='black')
        
        ax2.set_ylabel(secondary_label)
        ax2.set_xlabel('Sample/Subgroup')
        ax2.set_title(f'{secondary_label} Chart', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_histogram_with_limits(data: np.ndarray, cl: float, ucl: float, lcl: float) -> str:
    """Create histogram with control limits overlay"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    
    ax.axvline(x=cl, color='green', linestyle='-', linewidth=2, label=f'CL = {cl:.4f}')
    ax.axvline(x=ucl, color='red', linestyle='--', linewidth=2, label=f'UCL = {ucl:.4f}')
    ax.axvline(x=lcl, color='red', linestyle='--', linewidth=2, label=f'LCL = {lcl:.4f}')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Data Distribution with Control Limits', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(chart_data: Dict[str, Any], violations: List[Dict], 
                          chart_type: str) -> List[Dict[str, Any]]:
    """Generate key insights for SPC analysis"""
    insights = []
    
    # Get primary chart data
    if chart_type in ["xbar_r", "xbar_s"]:
        primary = chart_data['xbar']
    elif chart_type == "i_mr":
        primary = chart_data['i']
    else:
        key = chart_type.replace("_chart", "")
        primary = chart_data.get(key, chart_data.get(list(chart_data.keys())[0]))
    
    data = np.array(primary['data'])
    cl = primary['cl']
    ucl = primary['ucl'] if not isinstance(primary['ucl'], list) else np.mean(primary['ucl'])
    lcl = primary['lcl'] if not isinstance(primary['lcl'], list) else np.mean(primary['lcl'])
    
    # Process stability
    ooc_count = len([v for v in violations if any(vio['severity'] == 'high' for vio in v['violations'])])
    total_points = len(data)
    ooc_rate = ooc_count / total_points * 100 if total_points > 0 else 0
    
    if ooc_rate == 0:
        insights.append({
            'title': 'Process In Control',
            'description': f'All {total_points} points are within control limits. Process is stable.',
            'status': 'positive'
        })
    elif ooc_rate < 5:
        insights.append({
            'title': 'Minor Variations Detected',
            'description': f'{ooc_count} of {total_points} points ({ooc_rate:.1f}%) outside limits. Investigate causes.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Process Out of Control',
            'description': f'{ooc_count} of {total_points} points ({ooc_rate:.1f}%) outside limits. Immediate action needed.',
            'status': 'warning'
        })
    
    # Rule violations summary
    rule_counts = {}
    for v in violations:
        for vio in v['violations']:
            rule = vio['rule']
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    
    if rule_counts:
        most_common_rule = max(rule_counts, key=rule_counts.get)
        rule_descriptions = {
            1: 'Points beyond 3σ',
            2: '2 of 3 points beyond 2σ',
            3: '4 of 5 points beyond 1σ',
            4: '8 consecutive points same side',
            5: '6 consecutive trending points',
            6: '14 alternating points',
            7: 'Stratification (15 within 1σ)'
        }
        insights.append({
            'title': f'Most Common Violation: Rule {most_common_rule}',
            'description': f'{rule_descriptions.get(most_common_rule, "Unknown")} occurred {rule_counts[most_common_rule]} times.',
            'status': 'neutral'
        })
    
    # Process centering
    mean_val = np.mean(data)
    if abs(mean_val - cl) < (ucl - cl) * 0.1:
        insights.append({
            'title': 'Well Centered Process',
            'description': f'Process mean ({mean_val:.4f}) is close to center line ({cl:.4f}).',
            'status': 'positive'
        })
    else:
        direction = "above" if mean_val > cl else "below"
        insights.append({
            'title': 'Process Shift Detected',
            'description': f'Process mean ({mean_val:.4f}) is shifted {direction} center line ({cl:.4f}).',
            'status': 'warning'
        })
    
    # Variation assessment
    std_val = np.std(data)
    expected_std = (ucl - cl) / 3
    variation_ratio = std_val / expected_std if expected_std > 0 else 0
    
    if variation_ratio < 0.8:
        insights.append({
            'title': 'Low Variation',
            'description': f'Actual variation is {variation_ratio:.1%} of expected. Process is very consistent.',
            'status': 'positive'
        })
    elif variation_ratio > 1.2:
        insights.append({
            'title': 'High Variation',
            'description': f'Actual variation is {variation_ratio:.1%} of expected. Investigate sources of variation.',
            'status': 'warning'
        })
    
    return insights


@router.post("/spc")
async def run_spc_analysis(request: SPCRequest) -> Dict[str, Any]:
    """
    Perform Statistical Process Control analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        if request.measurement_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{request.measurement_col}' not found")
        
        data = pd.to_numeric(df[request.measurement_col], errors='coerce').dropna().values
        
        if len(data) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points")
        
        # Calculate chart based on type
        if request.chart_type == "xbar_r":
            chart_data = calculate_xbar_r(data, request.subgroup_size, request.sigma_limit)
            primary_key = 'xbar'
        elif request.chart_type == "xbar_s":
            chart_data = calculate_xbar_s(data, request.subgroup_size, request.sigma_limit)
            primary_key = 'xbar'
        elif request.chart_type == "i_mr":
            chart_data = calculate_imr(data, request.sigma_limit)
            primary_key = 'i'
        elif request.chart_type == "p":
            if not request.sample_size_col:
                raise HTTPException(status_code=400, detail="Sample size column required for p chart")
            sample_sizes = pd.to_numeric(df[request.sample_size_col], errors='coerce').dropna().values
            defectives = data[:len(sample_sizes)]
            chart_data = calculate_p_chart(defectives, sample_sizes, request.sigma_limit)
            primary_key = 'p'
        elif request.chart_type == "np":
            chart_data = calculate_np_chart(data.astype(int), request.subgroup_size, request.sigma_limit)
            primary_key = 'np'
        elif request.chart_type == "c":
            chart_data = calculate_c_chart(data.astype(int), request.sigma_limit)
            primary_key = 'c'
        elif request.chart_type == "u":
            if not request.sample_size_col:
                raise HTTPException(status_code=400, detail="Sample size column required for u chart")
            sample_sizes = pd.to_numeric(df[request.sample_size_col], errors='coerce').dropna().values
            defects = data[:len(sample_sizes)]
            chart_data = calculate_u_chart(defects, sample_sizes, request.sigma_limit)
            primary_key = 'u'
        else:
            raise HTTPException(status_code=400, detail=f"Unknown chart type: {request.chart_type}")
        
        # Apply specified limits if provided
        if request.use_specified_limits:
            if request.specified_ucl is not None:
                chart_data[primary_key]['ucl'] = request.specified_ucl
            if request.specified_lcl is not None:
                chart_data[primary_key]['lcl'] = request.specified_lcl
            if request.specified_cl is not None:
                chart_data[primary_key]['cl'] = request.specified_cl
        
        # Get limits for violation checking
        primary = chart_data[primary_key]
        cl = primary['cl']
        ucl = primary['ucl'] if not isinstance(primary['ucl'], list) else np.mean(primary['ucl'])
        lcl = primary['lcl'] if not isinstance(primary['lcl'], list) else np.mean(primary['lcl'])
        
        # Check Western Electric rules
        violations = check_western_electric_rules(np.array(primary['data']), cl, ucl, lcl)
        
        # Create visualizations
        control_chart = create_control_chart(chart_data, request.chart_type, 
                                             f"SPC Analysis - {request.chart_type.upper()}")
        histogram = create_histogram_with_limits(data, cl, ucl, lcl)
        
        # Generate insights
        insights = generate_key_insights(chart_data, violations, request.chart_type)
        
        # Calculate summary statistics
        primary_data = np.array(primary['data'])
        ooc_points = [i for i, v in enumerate(primary_data) 
                      if (not isinstance(primary['ucl'], list) and (v > primary['ucl'] or v < primary['lcl'])) or
                      (isinstance(primary['ucl'], list) and (v > primary['ucl'][i] or v < primary['lcl'][i]))]
        
        summary = {
            'chart_type': request.chart_type,
            'n_points': len(primary_data),
            'center_line': _to_native_type(cl),
            'ucl': _to_native_type(ucl),
            'lcl': _to_native_type(lcl),
            'mean': _to_native_type(np.mean(primary_data)),
            'std': _to_native_type(np.std(primary_data)),
            'min': _to_native_type(np.min(primary_data)),
            'max': _to_native_type(np.max(primary_data)),
            'out_of_control_count': len(ooc_points),
            'out_of_control_rate': _to_native_type(len(ooc_points) / len(primary_data) * 100),
            'process_in_control': len(ooc_points) == 0
        }
        
        # Convert chart_data to native types
        for key in chart_data:
            if isinstance(chart_data[key], dict):
                for subkey in chart_data[key]:
                    chart_data[key][subkey] = _to_native_type(chart_data[key][subkey])
            else:
                chart_data[key] = _to_native_type(chart_data[key])
        
        return {
            'success': True,
            'chart_data': chart_data,
            'violations': violations,
            'visualizations': {
                'control_chart': control_chart,
                'histogram': histogram
            },
            'key_insights': insights,
            'summary': summary,
            'out_of_control_points': ooc_points
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SPC analysis failed: {str(e)}")
