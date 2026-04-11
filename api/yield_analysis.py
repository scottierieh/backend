"""
Yield & Defect Analysis Router for FastAPI
FPY, RTY, DPU, DPMO, Sigma Level calculations with Pareto analysis
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


class YieldRequest(BaseModel):
    data: List[Dict[str, Any]]
    # For unit-level data
    pass_fail_col: Optional[str] = None  # Binary pass/fail column
    defect_count_col: Optional[str] = None  # Number of defects per unit
    # For aggregated data
    total_units_col: Optional[str] = None
    passed_units_col: Optional[str] = None
    defective_units_col: Optional[str] = None
    total_defects_col: Optional[str] = None
    # For multi-step process
    step_col: Optional[str] = None  # Process step identifier
    # Grouping
    group_col: Optional[str] = None  # For comparison (product, line, shift, etc.)
    time_col: Optional[str] = None  # For trend analysis
    defect_type_col: Optional[str] = None  # For Pareto analysis
    # Manual inputs
    opportunities_per_unit: int = 1  # For DPMO calculation
    # Aggregated input (if no data columns)
    total_units: Optional[int] = None
    passed_units: Optional[int] = None
    total_defects: Optional[int] = None


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


def dpmo_to_sigma(dpmo: float) -> float:
    """Convert DPMO to Sigma level"""
    if dpmo <= 0:
        return 6.0
    if dpmo >= 1000000:
        return 0.0
    # Using the relationship: Sigma = normsinv(1 - DPMO/1000000) + 1.5
    z = stats.norm.ppf(1 - dpmo / 1000000)
    return z + 1.5


def sigma_to_dpmo(sigma: float) -> float:
    """Convert Sigma level to DPMO"""
    z = sigma - 1.5
    dpmo = (1 - stats.norm.cdf(z)) * 1000000
    return dpmo


def calculate_yield_metrics(total_units: int, passed_units: int, 
                           total_defects: int, opportunities: int = 1) -> Dict[str, Any]:
    """Calculate all yield metrics"""
    
    defective_units = total_units - passed_units
    
    # First Pass Yield (FPY)
    fpy = passed_units / total_units if total_units > 0 else 0
    
    # Defect Rate
    defect_rate = defective_units / total_units if total_units > 0 else 0
    
    # Defects Per Unit (DPU)
    dpu = total_defects / total_units if total_units > 0 else 0
    
    # Defects Per Million Opportunities (DPMO)
    total_opportunities = total_units * opportunities
    dpmo = (total_defects / total_opportunities) * 1000000 if total_opportunities > 0 else 0
    
    # Sigma Level
    sigma_level = dpmo_to_sigma(dpmo)
    
    # Throughput Yield (using Poisson)
    # Y_tp = e^(-DPU)
    throughput_yield = np.exp(-dpu) if dpu < 100 else 0
    
    # PPM (Parts Per Million defective)
    ppm = defect_rate * 1000000
    
    return {
        'total_units': _to_native_type(total_units),
        'passed_units': _to_native_type(passed_units),
        'defective_units': _to_native_type(defective_units),
        'total_defects': _to_native_type(total_defects),
        'opportunities_per_unit': _to_native_type(opportunities),
        'total_opportunities': _to_native_type(total_opportunities),
        'fpy': _to_native_type(fpy * 100),
        'fpy_decimal': _to_native_type(fpy),
        'defect_rate': _to_native_type(defect_rate * 100),
        'dpu': _to_native_type(dpu),
        'dpmo': _to_native_type(dpmo),
        'sigma_level': _to_native_type(sigma_level),
        'throughput_yield': _to_native_type(throughput_yield * 100),
        'ppm': _to_native_type(ppm)
    }


def calculate_rty(step_yields: List[float]) -> Dict[str, Any]:
    """Calculate Rolled Throughput Yield"""
    
    # RTY = Y1 × Y2 × Y3 × ... × Yn
    rty = np.prod(step_yields)
    
    # Number of steps
    n_steps = len(step_yields)
    
    # Average step yield
    avg_yield = np.mean(step_yields)
    
    # Geometric mean
    geo_mean = np.power(rty, 1/n_steps) if n_steps > 0 else 0
    
    # Identify bottleneck (lowest yield step)
    min_yield_idx = np.argmin(step_yields)
    
    return {
        'rty': _to_native_type(rty * 100),
        'rty_decimal': _to_native_type(rty),
        'n_steps': n_steps,
        'step_yields': [_to_native_type(y * 100) for y in step_yields],
        'avg_yield': _to_native_type(avg_yield * 100),
        'geo_mean_yield': _to_native_type(geo_mean * 100),
        'bottleneck_step': min_yield_idx,
        'bottleneck_yield': _to_native_type(step_yields[min_yield_idx] * 100)
    }


def perform_pareto_analysis(defect_types: pd.Series) -> Dict[str, Any]:
    """Perform Pareto analysis on defect types"""
    
    # Count defects by type
    counts = defect_types.value_counts().sort_values(ascending=False)
    
    total = counts.sum()
    
    # Calculate percentages and cumulative
    percentages = (counts / total * 100).tolist()
    cumulative = np.cumsum(percentages).tolist()
    
    # Find 80% threshold
    threshold_idx = next((i for i, c in enumerate(cumulative) if c >= 80), len(cumulative) - 1)
    
    pareto_data = []
    for i, (defect_type, count) in enumerate(counts.items()):
        pareto_data.append({
            'defect_type': str(defect_type),
            'count': _to_native_type(count),
            'percentage': _to_native_type(percentages[i]),
            'cumulative': _to_native_type(cumulative[i]),
            'vital_few': i <= threshold_idx
        })
    
    return {
        'pareto_data': pareto_data,
        'total_defects': _to_native_type(total),
        'vital_few_count': threshold_idx + 1,
        'vital_few_percentage': _to_native_type(cumulative[threshold_idx]) if threshold_idx < len(cumulative) else 100,
        'top_defect': str(counts.index[0]) if len(counts) > 0 else None,
        'top_defect_percentage': _to_native_type(percentages[0]) if len(percentages) > 0 else 0
    }


def create_yield_gauge_chart(metrics: Dict[str, Any]) -> str:
    """Create yield gauge chart"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    gauges = [
        ('FPY', metrics['fpy'], 95),
        ('Throughput Yield', metrics['throughput_yield'], 95),
        ('Sigma Level', metrics['sigma_level'], 4),
        ('DPMO', metrics['dpmo'], 3.4)  # World class DPMO
    ]
    
    for ax, (name, value, target) in zip(axes, gauges):
        if name == 'DPMO':
            # Lower is better for DPMO
            if value <= 3.4:
                color = '#22c55e'
            elif value <= 233:
                color = '#3b82f6'
            elif value <= 6210:
                color = '#f59e0b'
            else:
                color = '#ef4444'
            display_val = f'{value:,.0f}' if value >= 1 else f'{value:.2f}'
        elif name == 'Sigma Level':
            if value >= 6:
                color = '#22c55e'
            elif value >= 4:
                color = '#3b82f6'
            elif value >= 3:
                color = '#f59e0b'
            else:
                color = '#ef4444'
            display_val = f'{value:.2f}σ'
        else:
            if value >= 99:
                color = '#22c55e'
            elif value >= 95:
                color = '#3b82f6'
            elif value >= 90:
                color = '#f59e0b'
            else:
                color = '#ef4444'
            display_val = f'{value:.1f}%'
        
        # Create donut
        if name in ['FPY', 'Throughput Yield']:
            sizes = [value, 100 - value]
        elif name == 'Sigma Level':
            sizes = [min(value / 6 * 100, 100), max(100 - value / 6 * 100, 0)]
        else:
            # For DPMO, invert (lower is better)
            dpmo_score = max(0, 100 - (value / 10000) * 100)
            sizes = [dpmo_score, 100 - dpmo_score]
        
        ax.pie(sizes, colors=[color, '#e5e7eb'], startangle=90,
               wedgeprops=dict(width=0.3, edgecolor='white'))
        ax.text(0, 0.1, display_val, ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)
        ax.text(0, -0.25, name, ha='center', va='center', fontsize=11, color='#374151')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_pareto_chart(pareto_data: List[Dict]) -> str:
    """Create Pareto chart"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    defects = [d['defect_type'] for d in pareto_data]
    counts = [d['count'] for d in pareto_data]
    cumulative = [d['cumulative'] for d in pareto_data]
    vital_few = [d['vital_few'] for d in pareto_data]
    
    # Bar colors
    colors = ['#ef4444' if v else '#94a3b8' for v in vital_few]
    
    # Bar chart
    bars = ax1.bar(range(len(defects)), counts, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Count', color='#374151')
    ax1.set_xlabel('Defect Type')
    ax1.set_xticks(range(len(defects)))
    ax1.set_xticklabels(defects, rotation=45, ha='right')
    
    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(len(defects)), cumulative, 'b-o', linewidth=2, markersize=6)
    ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    ax2.set_ylabel('Cumulative %', color='#3b82f6')
    ax2.set_ylim(0, 105)
    
    ax1.set_title('Pareto Analysis - Defect Types', fontsize=14, fontweight='bold')
    ax2.legend(loc='center right')
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_trend_chart(trend_data: List[Dict], time_col: str) -> str:
    """Create yield trend chart"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    times = [d[time_col] for d in trend_data]
    fpy = [d['fpy'] for d in trend_data]
    
    ax.plot(range(len(times)), fpy, 'b-o', linewidth=2, markersize=8, label='FPY')
    
    # Add target line
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='Target (95%)')
    
    # Add trend line
    if len(fpy) > 1:
        z = np.polyfit(range(len(fpy)), fpy, 1)
        p = np.poly1d(z)
        ax.plot(range(len(times)), p(range(len(times))), 'r--', alpha=0.5, label='Trend')
    
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, ha='right')
    ax.set_xlabel(time_col)
    ax.set_ylabel('First Pass Yield (%)')
    ax.set_title('Yield Trend Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_comparison_chart(comparison_data: List[Dict], group_col: str) -> str:
    """Create group comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups = [d[group_col] for d in comparison_data]
    fpy = [d['fpy'] for d in comparison_data]
    sigma = [d['sigma_level'] for d in comparison_data]
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fpy, width, label='FPY (%)', color='#3b82f6')
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, sigma, width, label='Sigma Level', color='#22c55e')
    
    ax.set_ylabel('FPY (%)', color='#3b82f6')
    ax2.set_ylabel('Sigma Level', color='#22c55e')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_xlabel(group_col)
    
    ax.axhline(y=95, color='#3b82f6', linestyle='--', alpha=0.5)
    ax2.axhline(y=4, color='#22c55e', linestyle='--', alpha=0.5)
    
    ax.set_title(f'Yield Comparison by {group_col}', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_rty_waterfall(step_yields: List[float], step_names: List[str] = None) -> str:
    """Create RTY waterfall chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_steps = len(step_yields)
    if step_names is None:
        step_names = [f'Step {i+1}' for i in range(n_steps)]
    
    # Calculate cumulative RTY
    cumulative = [step_yields[0]]
    for i in range(1, n_steps):
        cumulative.append(cumulative[-1] * step_yields[i])
    
    # Convert to percentage
    step_yields_pct = [y * 100 for y in step_yields]
    cumulative_pct = [y * 100 for y in cumulative]
    
    x = np.arange(n_steps)
    
    # Bar chart for step yields
    colors = ['#22c55e' if y >= 95 else '#f59e0b' if y >= 90 else '#ef4444' for y in step_yields_pct]
    bars = ax.bar(x, step_yields_pct, color=colors, edgecolor='white', linewidth=2, label='Step Yield')
    
    # Line for cumulative RTY
    ax.plot(x, cumulative_pct, 'b-o', linewidth=2, markersize=10, label='Cumulative RTY')
    
    # Add value labels
    for i, (bar, cum) in enumerate(zip(bars, cumulative_pct)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{step_yields_pct[i]:.1f}%', ha='center', fontsize=9)
        ax.text(i, cum - 3, f'{cum:.1f}%', ha='center', fontsize=9, color='blue', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=45, ha='right')
    ax.set_ylabel('Yield (%)')
    ax.set_title('Rolled Throughput Yield (RTY) Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_sigma_scale_chart(sigma_level: float) -> str:
    """Create sigma level scale chart"""
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Sigma scale
    sigma_points = [1, 2, 3, 4, 5, 6]
    dpmo_values = [690000, 308000, 66800, 6210, 233, 3.4]
    yield_values = [31, 69.2, 93.3, 99.38, 99.977, 99.9997]
    
    # Create scale
    ax.barh([0], [6], color='#e5e7eb', height=0.3)
    ax.barh([0], [sigma_level], color='#3b82f6', height=0.3)
    
    # Add markers
    for s in sigma_points:
        ax.axvline(x=s, color='white', linewidth=2)
        ax.text(s, 0.25, f'{s}σ', ha='center', fontsize=10, fontweight='bold')
    
    # Current position marker
    ax.plot(sigma_level, 0, 'rv', markersize=15)
    ax.text(sigma_level, -0.3, f'Current: {sigma_level:.2f}σ', ha='center', fontsize=11, fontweight='bold', color='red')
    
    ax.set_xlim(0, 6.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title('Sigma Level Scale', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(metrics: Dict[str, Any], pareto: Optional[Dict] = None,
                          rty: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Generate key insights for yield analysis"""
    insights = []
    
    fpy = metrics['fpy']
    sigma = metrics['sigma_level']
    dpmo = metrics['dpmo']
    
    # Overall yield assessment
    if fpy >= 99:
        insights.append({
            'title': 'Excellent Yield Performance',
            'description': f'FPY = {fpy:.2f}% indicates world-class manufacturing.',
            'status': 'positive'
        })
    elif fpy >= 95:
        insights.append({
            'title': 'Good Yield Performance',
            'description': f'FPY = {fpy:.2f}% meets industry standards.',
            'status': 'positive'
        })
    elif fpy >= 90:
        insights.append({
            'title': 'Acceptable Yield',
            'description': f'FPY = {fpy:.2f}% - room for improvement.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': 'Yield Needs Improvement',
            'description': f'FPY = {fpy:.2f}% is below acceptable threshold.',
            'status': 'warning'
        })
    
    # Sigma level assessment
    if sigma >= 6:
        insights.append({
            'title': 'Six Sigma Achieved',
            'description': f'Sigma level = {sigma:.2f}. World-class quality (DPMO = {dpmo:.1f}).',
            'status': 'positive'
        })
    elif sigma >= 4:
        insights.append({
            'title': 'Good Sigma Level',
            'description': f'Sigma level = {sigma:.2f}. DPMO = {dpmo:,.0f}.',
            'status': 'positive'
        })
    elif sigma >= 3:
        insights.append({
            'title': 'Average Sigma Level',
            'description': f'Sigma level = {sigma:.2f}. DPMO = {dpmo:,.0f}. Target 4σ or higher.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': 'Low Sigma Level',
            'description': f'Sigma level = {sigma:.2f}. DPMO = {dpmo:,.0f}. Significant improvement needed.',
            'status': 'warning'
        })
    
    # Pareto insights
    if pareto:
        insights.append({
            'title': f'Top Defect: {pareto["top_defect"]}',
            'description': f'Accounts for {pareto["top_defect_percentage"]:.1f}% of all defects. {pareto["vital_few_count"]} defect types cause {pareto["vital_few_percentage"]:.0f}% of issues.',
            'status': 'neutral'
        })
    
    # RTY insights
    if rty:
        insights.append({
            'title': f'RTY = {rty["rty"]:.2f}%',
            'description': f'Across {rty["n_steps"]} steps. Bottleneck at step {rty["bottleneck_step"]+1} ({rty["bottleneck_yield"]:.1f}%).',
            'status': 'positive' if rty['rty'] >= 90 else 'neutral' if rty['rty'] >= 80 else 'warning'
        })
    
    # Cost of poor quality estimate
    defect_rate = metrics['defect_rate']
    if defect_rate > 0:
        insights.append({
            'title': 'Cost Impact',
            'description': f'Defect rate of {defect_rate:.2f}% affects {metrics["defective_units"]:,} units. Focus on reducing defects to improve profitability.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/yield")
async def run_yield_analysis(request: YieldRequest) -> Dict[str, Any]:
    """
    Perform Yield & Defect Analysis.
    """
    try:
        df = pd.DataFrame(request.data) if request.data else pd.DataFrame()
        
        # Calculate basic metrics
        if request.total_units is not None and request.passed_units is not None:
            # Use manual inputs
            total_units = request.total_units
            passed_units = request.passed_units
            total_defects = request.total_defects or (total_units - passed_units)
        elif request.pass_fail_col and request.pass_fail_col in df.columns:
            # Unit-level pass/fail data
            pass_values = df[request.pass_fail_col].astype(str).str.lower()
            total_units = len(df)
            passed_units = pass_values.isin(['pass', 'p', '1', 'true', 'yes', 'ok', 'good']).sum()
            if request.defect_count_col and request.defect_count_col in df.columns:
                total_defects = int(pd.to_numeric(df[request.defect_count_col], errors='coerce').sum())
            else:
                total_defects = total_units - passed_units
        elif request.total_units_col and request.total_units_col in df.columns:
            # Aggregated data
            total_units = int(pd.to_numeric(df[request.total_units_col], errors='coerce').sum())
            if request.passed_units_col and request.passed_units_col in df.columns:
                passed_units = int(pd.to_numeric(df[request.passed_units_col], errors='coerce').sum())
            elif request.defective_units_col and request.defective_units_col in df.columns:
                defective = int(pd.to_numeric(df[request.defective_units_col], errors='coerce').sum())
                passed_units = total_units - defective
            else:
                passed_units = total_units
            
            if request.total_defects_col and request.total_defects_col in df.columns:
                total_defects = int(pd.to_numeric(df[request.total_defects_col], errors='coerce').sum())
            else:
                total_defects = total_units - passed_units
        else:
            # Default: count rows as units
            total_units = len(df)
            passed_units = total_units
            total_defects = 0
        
        if total_units == 0:
            raise HTTPException(status_code=400, detail="No units to analyze")
        
        # Calculate yield metrics
        metrics = calculate_yield_metrics(total_units, passed_units, total_defects, 
                                         request.opportunities_per_unit)
        
        visualizations = {}
        
        # Create gauge chart
        visualizations['gauge_chart'] = create_yield_gauge_chart(metrics)
        
        # Create sigma scale chart
        visualizations['sigma_scale'] = create_sigma_scale_chart(metrics['sigma_level'])
        
        # Pareto analysis
        pareto_result = None
        if request.defect_type_col and request.defect_type_col in df.columns:
            defect_types = df[request.defect_type_col].dropna()
            if len(defect_types) > 0:
                pareto_result = perform_pareto_analysis(defect_types)
                visualizations['pareto_chart'] = create_pareto_chart(pareto_result['pareto_data'])
        
        # RTY analysis (multi-step process)
        rty_result = None
        if request.step_col and request.step_col in df.columns:
            step_data = []
            for step in df[request.step_col].unique():
                step_df = df[df[request.step_col] == step]
                if request.pass_fail_col and request.pass_fail_col in step_df.columns:
                    pass_values = step_df[request.pass_fail_col].astype(str).str.lower()
                    step_total = len(step_df)
                    step_passed = pass_values.isin(['pass', 'p', '1', 'true', 'yes', 'ok', 'good']).sum()
                    step_yield = step_passed / step_total if step_total > 0 else 1
                    step_data.append({'step': step, 'yield': step_yield})
            
            if step_data:
                step_yields = [d['yield'] for d in step_data]
                step_names = [str(d['step']) for d in step_data]
                rty_result = calculate_rty(step_yields)
                rty_result['step_names'] = step_names
                visualizations['rty_chart'] = create_rty_waterfall(step_yields, step_names)
        
        # Trend analysis
        trend_data = None
        if request.time_col and request.time_col in df.columns:
            trend_data = []
            for time_val in df[request.time_col].unique():
                time_df = df[df[request.time_col] == time_val]
                if request.pass_fail_col and request.pass_fail_col in time_df.columns:
                    pass_values = time_df[request.pass_fail_col].astype(str).str.lower()
                    t_total = len(time_df)
                    t_passed = pass_values.isin(['pass', 'p', '1', 'true', 'yes', 'ok', 'good']).sum()
                elif request.total_units_col and request.total_units_col in time_df.columns:
                    t_total = int(pd.to_numeric(time_df[request.total_units_col], errors='coerce').sum())
                    if request.passed_units_col:
                        t_passed = int(pd.to_numeric(time_df[request.passed_units_col], errors='coerce').sum())
                    else:
                        t_passed = t_total
                else:
                    continue
                
                if t_total > 0:
                    t_metrics = calculate_yield_metrics(t_total, t_passed, t_total - t_passed, 
                                                       request.opportunities_per_unit)
                    trend_data.append({
                        request.time_col: str(time_val),
                        'fpy': t_metrics['fpy'],
                        'sigma_level': t_metrics['sigma_level'],
                        'dpmo': t_metrics['dpmo']
                    })
            
            if trend_data:
                visualizations['trend_chart'] = create_trend_chart(trend_data, request.time_col)
        
        # Group comparison
        comparison_data = None
        if request.group_col and request.group_col in df.columns:
            comparison_data = []
            for group in df[request.group_col].unique():
                group_df = df[df[request.group_col] == group]
                if request.pass_fail_col and request.pass_fail_col in group_df.columns:
                    pass_values = group_df[request.pass_fail_col].astype(str).str.lower()
                    g_total = len(group_df)
                    g_passed = pass_values.isin(['pass', 'p', '1', 'true', 'yes', 'ok', 'good']).sum()
                elif request.total_units_col and request.total_units_col in group_df.columns:
                    g_total = int(pd.to_numeric(group_df[request.total_units_col], errors='coerce').sum())
                    if request.passed_units_col:
                        g_passed = int(pd.to_numeric(group_df[request.passed_units_col], errors='coerce').sum())
                    else:
                        g_passed = g_total
                else:
                    continue
                
                if g_total > 0:
                    g_metrics = calculate_yield_metrics(g_total, g_passed, g_total - g_passed,
                                                       request.opportunities_per_unit)
                    comparison_data.append({
                        request.group_col: str(group),
                        'fpy': g_metrics['fpy'],
                        'sigma_level': g_metrics['sigma_level'],
                        'dpmo': g_metrics['dpmo'],
                        'total_units': g_total
                    })
            
            if comparison_data:
                visualizations['comparison_chart'] = create_comparison_chart(comparison_data, request.group_col)
        
        # Generate insights
        insights = generate_key_insights(metrics, pareto_result, rty_result)
        
        # Summary
        summary = {
            'fpy': metrics['fpy'],
            'sigma_level': metrics['sigma_level'],
            'dpmo': metrics['dpmo'],
            'dpu': metrics['dpu'],
            'total_units': metrics['total_units'],
            'defective_units': metrics['defective_units'],
            'total_defects': metrics['total_defects']
        }
        
        return {
            'success': True,
            'metrics': metrics,
            'pareto': pareto_result,
            'rty': rty_result,
            'trend_data': trend_data,
            'comparison_data': comparison_data,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield analysis failed: {str(e)}")
