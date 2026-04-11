"""
Gage R&R (Measurement System Analysis) Router for FastAPI
Calculates Repeatability, Reproducibility, and related MSA metrics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
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
class GageRRRequest(BaseModel):
    data: List[Dict[str, Any]]
    measurement_col: str
    part_col: str
    operator_col: str
    trial_col: Optional[str] = None
    method: Literal["anova", "xbar_r"] = "anova"
    tolerance: Optional[float] = None


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
    'grr': '#ef4444',
    'ev': '#f97316',
    'av': '#eab308',
    'pv': '#22c55e',
    'primary': '#3b82f6',
    'muted': '#94a3b8',
}

# d2 constants for estimating sigma from range
D2_CONSTANTS = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
    6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970,
    10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336,
    14: 3.407, 15: 3.472,
}

# D3, D4 constants for R chart control limits
D3_CONSTANTS = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4_CONSTANTS = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


# ============ ANOVA METHOD ============
def calculate_anova_grr(df: pd.DataFrame, measurement_col: str, 
                        part_col: str, operator_col: str) -> Dict:
    """Calculate Gage R&R using ANOVA method"""
    
    # Get unique counts
    parts = df[part_col].unique()
    operators = df[operator_col].unique()
    n_parts = len(parts)
    n_operators = len(operators)
    n_total = len(df)
    
    # Calculate trials per part-operator combination
    trials_per_combo = df.groupby([part_col, operator_col]).size()
    n_trials = int(trials_per_combo.mean())
    
    # Grand mean
    grand_mean = df[measurement_col].mean()
    
    # Calculate sum of squares
    # SS Part
    part_means = df.groupby(part_col)[measurement_col].mean()
    ss_part = n_operators * n_trials * np.sum((part_means - grand_mean) ** 2)
    
    # SS Operator
    operator_means = df.groupby(operator_col)[measurement_col].mean()
    ss_operator = n_parts * n_trials * np.sum((operator_means - grand_mean) ** 2)
    
    # SS Part*Operator (Interaction)
    cell_means = df.groupby([part_col, operator_col])[measurement_col].mean()
    ss_interaction = 0
    for part in parts:
        for op in operators:
            if (part, op) in cell_means.index:
                cell_mean = cell_means[(part, op)]
                expected = part_means[part] + operator_means[op] - grand_mean
                ss_interaction += n_trials * (cell_mean - expected) ** 2
    
    # SS Total
    ss_total = np.sum((df[measurement_col] - grand_mean) ** 2)
    
    # SS Error (Repeatability)
    ss_error = 0
    for (part, op), group in df.groupby([part_col, operator_col]):
        cell_mean = group[measurement_col].mean()
        ss_error += np.sum((group[measurement_col] - cell_mean) ** 2)
    
    # Degrees of freedom
    df_part = n_parts - 1
    df_operator = n_operators - 1
    df_interaction = df_part * df_operator
    df_error = n_total - n_parts * n_operators
    df_total = n_total - 1
    
    # Mean squares
    ms_part = ss_part / df_part if df_part > 0 else 0
    ms_operator = ss_operator / df_operator if df_operator > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # F-statistics and p-values
    f_part = ms_part / ms_interaction if ms_interaction > 0 else None
    f_operator = ms_operator / ms_interaction if ms_interaction > 0 else None
    f_interaction = ms_interaction / ms_error if ms_error > 0 else None
    
    p_part = 1 - sp_stats.f.cdf(f_part, df_part, df_interaction) if f_part else None
    p_operator = 1 - sp_stats.f.cdf(f_operator, df_operator, df_interaction) if f_operator else None
    p_interaction = 1 - sp_stats.f.cdf(f_interaction, df_interaction, df_error) if f_interaction else None
    
    # Check if interaction is significant
    interaction_significant = p_interaction is not None and p_interaction < 0.05
    
    # Variance components
    var_repeatability = ms_error
    
    # Reproducibility variance (with interaction)
    if interaction_significant:
        var_interaction = max(0, (ms_interaction - ms_error) / n_trials)
        var_operator = max(0, (ms_operator - ms_interaction) / (n_parts * n_trials))
    else:
        var_interaction = 0
        var_operator = max(0, (ms_operator - ms_error) / (n_parts * n_trials))
    
    var_reproducibility = var_operator + var_interaction
    
    # Part variance
    var_part = max(0, (ms_part - ms_interaction) / (n_operators * n_trials))
    
    # GRR variance
    var_grr = var_repeatability + var_reproducibility
    
    # Total variance
    var_total = var_grr + var_part
    
    # Standard deviations (sigma)
    sigma_repeatability = np.sqrt(var_repeatability)
    sigma_reproducibility = np.sqrt(var_reproducibility)
    sigma_grr = np.sqrt(var_grr)
    sigma_part = np.sqrt(var_part)
    sigma_total = np.sqrt(var_total)
    
    # Study variation (6 * sigma)
    sv_repeatability = 6 * sigma_repeatability
    sv_reproducibility = 6 * sigma_reproducibility
    sv_grr = 6 * sigma_grr
    sv_part = 6 * sigma_part
    sv_total = 6 * sigma_total
    
    # Percentages of study variation
    if sv_total > 0:
        pct_repeatability = (sv_repeatability / sv_total) * 100
        pct_reproducibility = (sv_reproducibility / sv_total) * 100
        pct_grr = (sv_grr / sv_total) * 100
        pct_part = (sv_part / sv_total) * 100
    else:
        pct_repeatability = pct_reproducibility = pct_grr = pct_part = 0
    
    # Number of Distinct Categories (NDC)
    ndc = 1.41 * (sigma_part / sigma_grr) if sigma_grr > 0 else 0
    
    # ANOVA table
    anova_table = [
        {'source': 'Part', 'df': df_part, 'ss': ss_part, 'ms': ms_part, 'f': f_part, 'p': p_part},
        {'source': 'Operator', 'df': df_operator, 'ss': ss_operator, 'ms': ms_operator, 'f': f_operator, 'p': p_operator},
        {'source': 'Part × Operator', 'df': df_interaction, 'ss': ss_interaction, 'ms': ms_interaction, 'f': f_interaction, 'p': p_interaction},
        {'source': 'Repeatability', 'df': df_error, 'ss': ss_error, 'ms': ms_error, 'f': None, 'p': None},
        {'source': 'Total', 'df': df_total, 'ss': ss_total, 'ms': None, 'f': None, 'p': None},
    ]
    
    # Variance components table
    variance_components = [
        {'source': 'Gage R&R', 'variance': var_grr, 'std_dev': sigma_grr, 'study_var': sv_grr, 'pct_study': pct_grr},
        {'source': '  Repeatability', 'variance': var_repeatability, 'std_dev': sigma_repeatability, 'study_var': sv_repeatability, 'pct_study': pct_repeatability},
        {'source': '  Reproducibility', 'variance': var_reproducibility, 'std_dev': sigma_reproducibility, 'study_var': sv_reproducibility, 'pct_study': pct_reproducibility},
        {'source': 'Part-to-Part', 'variance': var_part, 'std_dev': sigma_part, 'study_var': sv_part, 'pct_study': pct_part},
        {'source': 'Total Variation', 'variance': var_total, 'std_dev': sigma_total, 'study_var': sv_total, 'pct_study': 100.0},
    ]
    
    return {
        'method': 'ANOVA',
        'ev': sigma_repeatability,
        'av': sigma_reproducibility,
        'grr': sigma_grr,
        'pv': sigma_part,
        'tv': sigma_total,
        'ev_pct': pct_repeatability,
        'av_pct': pct_reproducibility,
        'grr_pct': pct_grr,
        'pv_pct': pct_part,
        'ndc': ndc,
        'n_parts': n_parts,
        'n_operators': n_operators,
        'n_trials': n_trials,
        'n_total': n_total,
        'anova_table': anova_table,
        'variance_components': variance_components,
        'interaction_significant': interaction_significant,
        'df': df,
        'part_col': part_col,
        'operator_col': operator_col,
        'measurement_col': measurement_col,
    }


# ============ X-BAR & R METHOD ============
def calculate_xbar_r_grr(df: pd.DataFrame, measurement_col: str,
                         part_col: str, operator_col: str) -> Dict:
    """Calculate Gage R&R using X-bar & R method"""
    
    parts = df[part_col].unique()
    operators = df[operator_col].unique()
    n_parts = len(parts)
    n_operators = len(operators)
    n_total = len(df)
    
    # Calculate trials
    trials_per_combo = df.groupby([part_col, operator_col]).size()
    n_trials = int(trials_per_combo.mean())
    
    # Get d2 constant
    d2 = D2_CONSTANTS.get(n_trials, 1.128)
    
    # Calculate range for each part-operator combination
    ranges = df.groupby([part_col, operator_col])[measurement_col].apply(
        lambda x: x.max() - x.min()
    )
    r_bar = ranges.mean()
    
    # Repeatability (EV) = R-bar / d2
    sigma_repeatability = r_bar / d2 if d2 > 0 else 0
    
    # Calculate operator means
    operator_means = df.groupby(operator_col)[measurement_col].mean()
    x_diff = operator_means.max() - operator_means.min()
    
    # d2* for number of operators
    d2_star = D2_CONSTANTS.get(n_operators, 1.128)
    
    # Reproducibility (AV)
    av_squared = (x_diff / d2_star) ** 2 - (sigma_repeatability ** 2 / (n_parts * n_trials))
    sigma_reproducibility = np.sqrt(max(0, av_squared))
    
    # GRR
    sigma_grr = np.sqrt(sigma_repeatability ** 2 + sigma_reproducibility ** 2)
    
    # Part variation
    part_means = df.groupby(part_col)[measurement_col].mean()
    r_p = part_means.max() - part_means.min()
    d2_parts = D2_CONSTANTS.get(n_parts, 3.078)
    sigma_part = r_p / d2_parts if d2_parts > 0 else 0
    
    # Total variation
    sigma_total = np.sqrt(sigma_grr ** 2 + sigma_part ** 2)
    
    # Study variation
    sv_repeatability = 6 * sigma_repeatability
    sv_reproducibility = 6 * sigma_reproducibility
    sv_grr = 6 * sigma_grr
    sv_part = 6 * sigma_part
    sv_total = 6 * sigma_total
    
    # Percentages
    if sv_total > 0:
        pct_repeatability = (sv_repeatability / sv_total) * 100
        pct_reproducibility = (sv_reproducibility / sv_total) * 100
        pct_grr = (sv_grr / sv_total) * 100
        pct_part = (sv_part / sv_total) * 100
    else:
        pct_repeatability = pct_reproducibility = pct_grr = pct_part = 0
    
    # NDC
    ndc = 1.41 * (sigma_part / sigma_grr) if sigma_grr > 0 else 0
    
    # Variance components for display
    variance_components = [
        {'source': 'Gage R&R', 'variance': sigma_grr**2, 'std_dev': sigma_grr, 'study_var': sv_grr, 'pct_study': pct_grr},
        {'source': '  Repeatability', 'variance': sigma_repeatability**2, 'std_dev': sigma_repeatability, 'study_var': sv_repeatability, 'pct_study': pct_repeatability},
        {'source': '  Reproducibility', 'variance': sigma_reproducibility**2, 'std_dev': sigma_reproducibility, 'study_var': sv_reproducibility, 'pct_study': pct_reproducibility},
        {'source': 'Part-to-Part', 'variance': sigma_part**2, 'std_dev': sigma_part, 'study_var': sv_part, 'pct_study': pct_part},
        {'source': 'Total Variation', 'variance': sigma_total**2, 'std_dev': sigma_total, 'study_var': sv_total, 'pct_study': 100.0},
    ]
    
    return {
        'method': 'X-bar & R',
        'ev': sigma_repeatability,
        'av': sigma_reproducibility,
        'grr': sigma_grr,
        'pv': sigma_part,
        'tv': sigma_total,
        'ev_pct': pct_repeatability,
        'av_pct': pct_reproducibility,
        'grr_pct': pct_grr,
        'pv_pct': pct_part,
        'ndc': ndc,
        'n_parts': n_parts,
        'n_operators': n_operators,
        'n_trials': n_trials,
        'n_total': n_total,
        'anova_table': None,
        'variance_components': variance_components,
        'interaction_significant': None,
        'df': df,
        'part_col': part_col,
        'operator_col': operator_col,
        'measurement_col': measurement_col,
        'r_bar': r_bar,
    }


# ============ VISUALIZATION FUNCTIONS ============
def create_components_chart(results: Dict) -> str:
    """Create variance components bar chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of % Study Var
    components = ['Gage R&R', 'Repeatability', 'Reproducibility', 'Part-to-Part']
    values = [results['grr_pct'], results['ev_pct'], results['av_pct'], results['pv_pct']]
    colors = [COLORS['grr'], COLORS['ev'], COLORS['av'], COLORS['pv']]
    
    bars = ax1.barh(components, values, color=colors, edgecolor='white', linewidth=2)
    
    # Add threshold lines
    ax1.axvline(10, color='green', linestyle='--', alpha=0.7, label='Excellent (<10%)')
    ax1.axvline(30, color='orange', linestyle='--', alpha=0.7, label='Acceptable (<30%)')
    
    for bar, val in zip(bars, values):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('% Study Variation', fontsize=11)
    ax1.set_title('Gage R&R Components', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(0, max(values) * 1.3)
    
    # Pie chart
    pie_labels = ['Gage R&R', 'Part-to-Part']
    pie_values = [results['grr_pct'], results['pv_pct']]
    pie_colors = [COLORS['grr'], COLORS['pv']]
    
    wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
                                        colors=pie_colors, explode=[0.05, 0],
                                        startangle=90)
    ax2.set_title('Variation Breakdown', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_by_part_chart(results: Dict) -> str:
    """Create measurements by part chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = results['df']
    part_col = results['part_col']
    measurement_col = results['measurement_col']
    operator_col = results['operator_col']
    
    parts = sorted(df[part_col].unique())
    operators = df[operator_col].unique()
    
    # Box plot by part
    data_by_part = [df[df[part_col] == p][measurement_col].values for p in parts]
    bp = ax.boxplot(data_by_part, labels=parts, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['primary'])
        patch.set_alpha(0.6)
    
    # Overlay individual points with operator colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(operators)))
    for i, part in enumerate(parts):
        for j, op in enumerate(operators):
            subset = df[(df[part_col] == part) & (df[operator_col] == op)]
            x = np.random.normal(i + 1, 0.05, len(subset))
            ax.scatter(x, subset[measurement_col], alpha=0.6, s=30, 
                      color=colors[j], label=op if i == 0 else "")
    
    ax.set_xlabel('Part', fontsize=11)
    ax.set_ylabel('Measurement', fontsize=11)
    ax.set_title('Measurements by Part', fontsize=12, fontweight='bold')
    ax.legend(title='Operator', loc='upper right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_by_operator_chart(results: Dict) -> str:
    """Create measurements by operator chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = results['df']
    operator_col = results['operator_col']
    measurement_col = results['measurement_col']
    
    operators = sorted(df[operator_col].unique())
    
    # Box plot by operator
    data_by_op = [df[df[operator_col] == op][measurement_col].values for op in operators]
    bp = ax.boxplot(data_by_op, labels=operators, patch_artist=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(operators)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add mean line
    grand_mean = df[measurement_col].mean()
    ax.axhline(grand_mean, color='red', linestyle='--', label=f'Grand Mean: {grand_mean:.4f}')
    
    ax.set_xlabel('Operator', fontsize=11)
    ax.set_ylabel('Measurement', fontsize=11)
    ax.set_title('Measurements by Operator', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_interaction_chart(results: Dict) -> str:
    """Create part × operator interaction plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = results['df']
    part_col = results['part_col']
    operator_col = results['operator_col']
    measurement_col = results['measurement_col']
    
    parts = sorted(df[part_col].unique())
    operators = sorted(df[operator_col].unique())
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(operators)))
    
    # Calculate means for each part-operator combination
    for i, op in enumerate(operators):
        means = []
        for part in parts:
            subset = df[(df[part_col] == part) & (df[operator_col] == op)]
            means.append(subset[measurement_col].mean())
        ax.plot(range(len(parts)), means, 'o-', color=colors[i], 
               label=op, markersize=8, linewidth=2)
    
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(parts)
    ax.set_xlabel('Part', fontsize=11)
    ax.set_ylabel('Mean Measurement', fontsize=11)
    ax.set_title('Part × Operator Interaction', fontsize=12, fontweight='bold')
    ax.legend(title='Operator', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add note about interaction significance
    if results.get('interaction_significant') is not None:
        sig_text = 'Interaction: Significant (p < 0.05)' if results['interaction_significant'] else 'Interaction: Not Significant'
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_xbar_r_chart(results: Dict) -> str:
    """Create X-bar and R control charts"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    df = results['df']
    part_col = results['part_col']
    operator_col = results['operator_col']
    measurement_col = results['measurement_col']
    
    # Create groups (part × operator)
    groups = []
    group_labels = []
    for part in sorted(df[part_col].unique()):
        for op in sorted(df[operator_col].unique()):
            subset = df[(df[part_col] == part) & (df[operator_col] == op)]
            if len(subset) > 0:
                groups.append(subset[measurement_col].values)
                group_labels.append(f'{part}-{op}')
    
    # Calculate X-bar and R for each group
    x_bars = [np.mean(g) for g in groups]
    ranges = [np.max(g) - np.min(g) for g in groups]
    
    x_bar_bar = np.mean(x_bars)
    r_bar = np.mean(ranges)
    
    n_trials = results['n_trials']
    d2 = D2_CONSTANTS.get(n_trials, 1.128)
    d3 = D3_CONSTANTS.get(n_trials, 0)
    d4 = D4_CONSTANTS.get(n_trials, 3.267)
    a2 = 0.577 if n_trials == 5 else 1.880 / d2  # Approximation
    
    # Control limits
    ucl_xbar = x_bar_bar + a2 * r_bar
    lcl_xbar = x_bar_bar - a2 * r_bar
    ucl_r = d4 * r_bar
    lcl_r = d3 * r_bar
    
    x = range(1, len(groups) + 1)
    
    # X-bar chart
    ax1.plot(x, x_bars, 'bo-', markersize=4, linewidth=1)
    ax1.axhline(x_bar_bar, color='green', linewidth=2, label=f'CL = {x_bar_bar:.4f}')
    ax1.axhline(ucl_xbar, color='red', linestyle='--', linewidth=1.5, label=f'UCL = {ucl_xbar:.4f}')
    ax1.axhline(lcl_xbar, color='red', linestyle='--', linewidth=1.5, label=f'LCL = {lcl_xbar:.4f}')
    
    ax1.set_ylabel('X-bar', fontsize=11)
    ax1.set_title('X-bar Chart (by Part-Operator)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # R chart
    ax2.plot(x, ranges, 'bo-', markersize=4, linewidth=1)
    ax2.axhline(r_bar, color='green', linewidth=2, label=f'CL = {r_bar:.4f}')
    ax2.axhline(ucl_r, color='red', linestyle='--', linewidth=1.5, label=f'UCL = {ucl_r:.4f}')
    if lcl_r > 0:
        ax2.axhline(lcl_r, color='red', linestyle='--', linewidth=1.5, label=f'LCL = {lcl_r:.4f}')
    
    ax2.set_xlabel('Subgroup (Part-Operator)', fontsize=11)
    ax2.set_ylabel('Range', fontsize=11)
    ax2.set_title('R Chart', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(results: Dict) -> List[Dict]:
    """Generate key insights from the analysis"""
    insights = []
    
    grr_pct = results['grr_pct']
    ev_pct = results['ev_pct']
    av_pct = results['av_pct']
    ndc = results['ndc']
    
    # Overall GRR assessment
    if grr_pct < 10:
        insights.append({
            'title': f'Excellent Measurement System ({grr_pct:.1f}% GRR)',
            'description': 'Gage R&R is less than 10% of total variation. The measurement system is highly capable.',
            'status': 'positive'
        })
    elif grr_pct < 30:
        insights.append({
            'title': f'Acceptable Measurement System ({grr_pct:.1f}% GRR)',
            'description': 'Gage R&R is between 10-30%. May be acceptable depending on application requirements.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Unacceptable Measurement System ({grr_pct:.1f}% GRR)',
            'description': 'Gage R&R exceeds 30%. The measurement system requires improvement before use.',
            'status': 'warning'
        })
    
    # EV vs AV comparison
    if ev_pct > av_pct * 1.5:
        insights.append({
            'title': 'Equipment Variation Dominant',
            'description': f'Repeatability ({ev_pct:.1f}%) is significantly higher than Reproducibility ({av_pct:.1f}%). Focus on equipment maintenance, calibration, or measurement technique.',
            'status': 'warning' if ev_pct > 20 else 'neutral'
        })
    elif av_pct > ev_pct * 1.5:
        insights.append({
            'title': 'Operator Variation Dominant',
            'description': f'Reproducibility ({av_pct:.1f}%) is significantly higher than Repeatability ({ev_pct:.1f}%). Focus on operator training and standardization.',
            'status': 'warning' if av_pct > 20 else 'neutral'
        })
    else:
        insights.append({
            'title': 'Balanced Variation Sources',
            'description': f'Repeatability ({ev_pct:.1f}%) and Reproducibility ({av_pct:.1f}%) are relatively balanced.',
            'status': 'neutral'
        })
    
    # NDC assessment
    if ndc >= 5:
        insights.append({
            'title': f'Adequate Discrimination (NDC = {ndc:.1f})',
            'description': 'NDC ≥ 5 indicates the measurement system can adequately distinguish between parts.',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': f'Poor Discrimination (NDC = {ndc:.1f})',
            'description': 'NDC < 5 indicates the measurement system cannot reliably distinguish between parts.',
            'status': 'warning'
        })
    
    # Interaction (if ANOVA)
    if results.get('interaction_significant') is True:
        insights.append({
            'title': 'Significant Interaction Detected',
            'description': 'Part × Operator interaction is significant. Different operators measure different parts with varying accuracy.',
            'status': 'warning'
        })
    
    return insights


# ============ MAIN API ENDPOINT ============
@router.post("/gage-rr")
async def run_gagerr_analysis(request: GageRRRequest) -> Dict[str, Any]:
    """
    Run Gage R&R (MSA) Analysis
    
    Evaluates measurement system capability through Repeatability and Reproducibility analysis.
    """
    try:
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Validate columns
        required_cols = [request.measurement_col, request.part_col, request.operator_col]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Convert measurement to numeric
        df[request.measurement_col] = pd.to_numeric(df[request.measurement_col], errors='coerce')
        df = df.dropna(subset=[request.measurement_col])
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 measurements")
        
        # Run analysis based on method
        if request.method == "anova":
            results = calculate_anova_grr(df, request.measurement_col, 
                                          request.part_col, request.operator_col)
        else:
            results = calculate_xbar_r_grr(df, request.measurement_col,
                                           request.part_col, request.operator_col)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Create visualizations
        visualizations = {
            'components_chart': create_components_chart(results),
            'by_part_chart': create_by_part_chart(results),
            'by_operator_chart': create_by_operator_chart(results),
            'interaction_chart': create_interaction_chart(results),
            'xbar_r_chart': create_xbar_r_chart(results),
        }
        
        # Generate insights
        key_insights = generate_key_insights(results)
        
        # Prepare response (remove DataFrame from results)
        response_results = {
            'method': results['method'],
            'ev': _to_native_type(results['ev']),
            'av': _to_native_type(results['av']),
            'grr': _to_native_type(results['grr']),
            'pv': _to_native_type(results['pv']),
            'tv': _to_native_type(results['tv']),
            'ev_pct': _to_native_type(results['ev_pct']),
            'av_pct': _to_native_type(results['av_pct']),
            'grr_pct': _to_native_type(results['grr_pct']),
            'pv_pct': _to_native_type(results['pv_pct']),
            'ndc': _to_native_type(results['ndc']),
            'n_parts': results['n_parts'],
            'n_operators': results['n_operators'],
            'n_trials': results['n_trials'],
            'n_total': results['n_total'],
            'anova_table': [{k: _to_native_type(v) for k, v in row.items()} 
                           for row in results['anova_table']] if results['anova_table'] else None,
            'variance_components': [{k: _to_native_type(v) for k, v in row.items()} 
                                   for row in results['variance_components']],
            'interaction_significant': results.get('interaction_significant'),
        }
        
        # Summary
        grr_pct = results['grr_pct']
        summary = {
            'method': results['method'],
            'grr_pct': _to_native_type(grr_pct),
            'ev_pct': _to_native_type(results['ev_pct']),
            'av_pct': _to_native_type(results['av_pct']),
            'pv_pct': _to_native_type(results['pv_pct']),
            'ndc': _to_native_type(results['ndc']),
            'n_parts': results['n_parts'],
            'n_operators': results['n_operators'],
            'n_trials': results['n_trials'],
            'acceptable': grr_pct < 30,
            'excellent': grr_pct < 10,
            'solve_time_ms': solve_time_ms,
        }
        
        return {
            'success': True,
            'results': response_results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gage R&R analysis failed: {str(e)}")
