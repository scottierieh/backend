"""
Compensation Analysis Router for FastAPI
Statistical analysis for salary and compensation data
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
import time
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CompensationRequest(BaseModel):
    data: List[Dict[str, Any]]
    employee_col: Optional[str] = None
    salary_col: str
    department_col: Optional[str] = None
    level_col: Optional[str] = None
    tenure_col: Optional[str] = None
    gender_col: Optional[str] = None
    performance_col: Optional[str] = None
    analysis_type: Literal["equity", "benchmark", "pay_gap"] = "equity"
    market_benchmark: Optional[float] = None
    equity_threshold: float = 0.05


def _to_native_type(obj):
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_salary_statistics(salaries: np.ndarray) -> Dict:
    """Calculate comprehensive salary statistics"""
    return {
        'count': int(len(salaries)),
        'mean': float(np.mean(salaries)),
        'median': float(np.median(salaries)),
        'std': float(np.std(salaries, ddof=1)) if len(salaries) > 1 else 0,
        'min': float(np.min(salaries)),
        'max': float(np.max(salaries)),
        'q25': float(np.percentile(salaries, 25)),
        'q75': float(np.percentile(salaries, 75)),
        'iqr': float(np.percentile(salaries, 75) - np.percentile(salaries, 25)),
        'range': float(np.max(salaries) - np.min(salaries)),
        'cv': float(np.std(salaries, ddof=1) / np.mean(salaries)) if np.mean(salaries) > 0 else 0
    }


def analyze_by_group(df: pd.DataFrame, salary_col: str, group_col: str) -> List[Dict]:
    """Analyze salary by group (department, level, etc.)"""
    results = []
    
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        salaries = group_df[salary_col].dropna().values
        
        if len(salaries) >= 3:
            stats_data = calculate_salary_statistics(salaries)
            stats_data['group'] = str(group)
            results.append(stats_data)
    
    return sorted(results, key=lambda x: x['mean'], reverse=True)


def perform_pay_gap_analysis(df: pd.DataFrame, salary_col: str, gender_col: str) -> Dict:
    """Perform gender pay gap analysis"""
    results = {}
    
    groups = df[gender_col].unique()
    group_stats = {}
    
    for group in groups:
        salaries = df[df[gender_col] == group][salary_col].dropna().values
        if len(salaries) >= 3:
            group_stats[str(group)] = calculate_salary_statistics(salaries)
    
    results['group_statistics'] = group_stats
    
    # Calculate pay gaps between groups
    groups_list = list(group_stats.keys())
    gaps = []
    
    for i, g1 in enumerate(groups_list):
        for g2 in groups_list[i+1:]:
            mean1 = group_stats[g1]['mean']
            mean2 = group_stats[g2]['mean']
            
            if mean2 > 0:
                gap_pct = ((mean1 - mean2) / mean2) * 100
            else:
                gap_pct = 0
            
            # T-test for significance
            salaries1 = df[df[gender_col] == g1][salary_col].dropna().values
            salaries2 = df[df[gender_col] == g2][salary_col].dropna().values
            
            if len(salaries1) >= 3 and len(salaries2) >= 3:
                t_stat, p_value = stats.ttest_ind(salaries1, salaries2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(salaries1)-1)*np.var(salaries1, ddof=1) + 
                                      (len(salaries2)-1)*np.var(salaries2, ddof=1)) / 
                                     (len(salaries1) + len(salaries2) - 2))
                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            else:
                t_stat, p_value, cohens_d = 0, 1, 0
            
            gaps.append({
                'group_1': g1,
                'group_2': g2,
                'mean_1': mean1,
                'mean_2': mean2,
                'gap_absolute': mean1 - mean2,
                'gap_percentage': gap_pct,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': p_value < 0.05
            })
    
    results['pay_gaps'] = gaps
    return results


def perform_equity_analysis(df: pd.DataFrame, salary_col: str, level_col: str, 
                            threshold: float) -> Dict:
    """Analyze pay equity within job levels"""
    results = {}
    level_analysis = []
    
    for level in df[level_col].unique():
        level_df = df[df[level_col] == level]
        salaries = level_df[salary_col].dropna().values
        
        if len(salaries) >= 3:
            stats_data = calculate_salary_statistics(salaries)
            
            # Calculate compa-ratio spread
            median = stats_data['median']
            if median > 0:
                compa_ratios = salaries / median
                spread = (np.max(compa_ratios) - np.min(compa_ratios))
                cv = stats_data['cv']
            else:
                spread, cv = 0, 0
            
            # Identify outliers
            q1, q3 = stats_data['q25'], stats_data['q75']
            iqr = stats_data['iqr']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = len(salaries[(salaries < lower_bound) | (salaries > upper_bound)])
            
            equity_score = 1 - min(cv, 1)  # Higher score = more equitable
            
            level_analysis.append({
                'level': str(level),
                'count': int(len(salaries)),
                'mean': stats_data['mean'],
                'median': stats_data['median'],
                'std': stats_data['std'],
                'cv': cv,
                'spread': spread,
                'outliers': outliers,
                'equity_score': equity_score,
                'is_equitable': cv <= threshold
            })
    
    results['level_analysis'] = sorted(level_analysis, key=lambda x: x['equity_score'], reverse=True)
    
    # Overall equity metrics
    all_scores = [l['equity_score'] for l in level_analysis]
    results['overall'] = {
        'avg_equity_score': np.mean(all_scores) if all_scores else 0,
        'levels_analyzed': len(level_analysis),
        'equitable_levels': len([l for l in level_analysis if l['is_equitable']]),
        'threshold': threshold
    }
    
    return results


def perform_benchmark_analysis(df: pd.DataFrame, salary_col: str, 
                               market_benchmark: float) -> Dict:
    """Compare salaries against market benchmark"""
    salaries = df[salary_col].dropna().values
    stats_data = calculate_salary_statistics(salaries)
    
    # Compa-ratios
    compa_ratios = salaries / market_benchmark
    
    results = {
        'statistics': stats_data,
        'benchmark': market_benchmark,
        'compa_ratio': {
            'mean': float(np.mean(compa_ratios)),
            'median': float(np.median(compa_ratios)),
            'min': float(np.min(compa_ratios)),
            'max': float(np.max(compa_ratios)),
            'std': float(np.std(compa_ratios, ddof=1)) if len(compa_ratios) > 1 else 0
        },
        'distribution': {
            'below_80': int(np.sum(compa_ratios < 0.8)),
            'between_80_90': int(np.sum((compa_ratios >= 0.8) & (compa_ratios < 0.9))),
            'between_90_110': int(np.sum((compa_ratios >= 0.9) & (compa_ratios <= 1.1))),
            'between_110_120': int(np.sum((compa_ratios > 1.1) & (compa_ratios <= 1.2))),
            'above_120': int(np.sum(compa_ratios > 1.2))
        },
        'vs_benchmark': {
            'above': int(np.sum(salaries > market_benchmark)),
            'at': int(np.sum(salaries == market_benchmark)),
            'below': int(np.sum(salaries < market_benchmark)),
            'pct_above': float(np.mean(salaries > market_benchmark) * 100),
            'pct_below': float(np.mean(salaries < market_benchmark) * 100)
        }
    }
    
    # One-sample t-test against benchmark
    t_stat, p_value = stats.ttest_1samp(salaries, market_benchmark)
    results['t_test'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'direction': 'above' if stats_data['mean'] > market_benchmark else 'below'
    }
    
    return results


def create_salary_distribution_chart(df: pd.DataFrame, salary_col: str, 
                                     benchmark: Optional[float] = None) -> str:
    """Create salary distribution chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    salaries = df[salary_col].dropna().values
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(salaries, bins=30, color='#3b82f6', alpha=0.7, edgecolor='white')
    ax1.axvline(np.mean(salaries), color='#ef4444', linestyle='--', linewidth=2, 
                label=f'Mean: ${np.mean(salaries):,.0f}')
    ax1.axvline(np.median(salaries), color='#22c55e', linestyle='--', linewidth=2,
                label=f'Median: ${np.median(salaries):,.0f}')
    if benchmark:
        ax1.axvline(benchmark, color='#f59e0b', linestyle='-', linewidth=2,
                    label=f'Benchmark: ${benchmark:,.0f}')
    ax1.set_xlabel('Salary ($)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Salary Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([salaries], patch_artist=True, labels=['All Employees'])
    bp['boxes'][0].set_facecolor('#3b82f6')
    bp['boxes'][0].set_alpha(0.7)
    if benchmark:
        ax2.axhline(benchmark, color='#f59e0b', linestyle='--', linewidth=2, label='Benchmark')
        ax2.legend()
    ax2.set_ylabel('Salary ($)', fontsize=11)
    ax2.set_title('Salary Box Plot', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_group_comparison_chart(group_data: List[Dict], title: str = "Salary by Group") -> str:
    """Create group comparison chart"""
    if not group_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups = [d['group'] for d in group_data]
    means = [d['mean'] for d in group_data]
    medians = [d['median'] for d in group_data]
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, means, width, label='Mean', color='#3b82f6', alpha=0.8)
    bars2 = ax.bar(x + width/2, medians, width, label='Median', color='#22c55e', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(g)[:15] for g in groups], rotation=45, ha='right')
    ax.set_ylabel('Salary ($)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'${height/1000:.0f}K',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_pay_gap_chart(gap_data: List[Dict]) -> str:
    """Create pay gap visualization"""
    if not gap_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No gap data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"{d['group_1']} vs {d['group_2']}" for d in gap_data]
    gaps = [d['gap_percentage'] for d in gap_data]
    significances = [d['significant'] for d in gap_data]
    
    colors = ['#ef4444' if sig else '#94a3b8' for sig in significances]
    
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, gaps, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Pay Gap (%)', fontsize=11)
    ax.set_title('Pay Gap Analysis', fontsize=14, fontweight='bold')
    ax.axvline(0, color='#1f2937', linewidth=1)
    
    for bar, gap, sig in zip(bars, gaps, significances):
        label = f'{gap:+.1f}%' + (' *' if sig else '')
        ax.annotate(label,
                    xy=(gap, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if gap >= 0 else -5, 0),
                    textcoords="offset points",
                    ha='left' if gap >= 0 else 'right',
                    va='center', fontsize=10, fontweight='bold')
    
    ax.legend([plt.Rectangle((0,0),1,1, fc='#ef4444', alpha=0.8),
               plt.Rectangle((0,0),1,1, fc='#94a3b8', alpha=0.8)],
              ['Significant (p<0.05)', 'Not Significant'], loc='lower right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_equity_chart(equity_data: List[Dict]) -> str:
    """Create equity analysis chart"""
    if not equity_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No equity data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    levels = [d['level'] for d in equity_data]
    scores = [d['equity_score'] * 100 for d in equity_data]
    cvs = [d['cv'] * 100 for d in equity_data]
    is_equitable = [d['is_equitable'] for d in equity_data]
    
    # Equity scores
    ax1 = axes[0]
    colors = ['#22c55e' if eq else '#ef4444' for eq in is_equitable]
    bars = ax1.barh(levels, scores, color=colors, alpha=0.8)
    ax1.set_xlabel('Equity Score (%)', fontsize=11)
    ax1.set_title('Pay Equity by Level', fontsize=14, fontweight='bold')
    ax1.axvline(80, color='#f59e0b', linestyle='--', linewidth=2, label='Target (80%)')
    ax1.legend()
    
    for bar, score in zip(bars, scores):
        ax1.annotate(f'{score:.1f}%',
                    xy=(score, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10)
    
    # CV comparison
    ax2 = axes[1]
    colors2 = ['#22c55e' if cv < 15 else '#f59e0b' if cv < 25 else '#ef4444' for cv in cvs]
    ax2.barh(levels, cvs, color=colors2, alpha=0.8)
    ax2.set_xlabel('Coefficient of Variation (%)', fontsize=11)
    ax2.set_title('Salary Variation by Level', fontsize=14, fontweight='bold')
    ax2.axvline(15, color='#22c55e', linestyle='--', linewidth=2, label='Good (<15%)')
    ax2.axvline(25, color='#ef4444', linestyle='--', linewidth=2, label='Concern (>25%)')
    ax2.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_compa_ratio_chart(benchmark_data: Dict) -> str:
    """Create compa-ratio distribution chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    dist = benchmark_data['distribution']
    
    # Pie chart
    ax1 = axes[0]
    labels = ['<80%', '80-90%', '90-110%', '110-120%', '>120%']
    sizes = [dist['below_80'], dist['between_80_90'], dist['between_90_110'], 
             dist['between_110_120'], dist['above_120']]
    colors = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6']
    
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if non_zero:
        labels, sizes, colors = zip(*non_zero)
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Compa-Ratio Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart
    ax2 = axes[1]
    vs = benchmark_data['vs_benchmark']
    categories = ['Below Benchmark', 'At Benchmark', 'Above Benchmark']
    values = [vs['below'], vs['at'], vs['above']]
    colors = ['#ef4444', '#f59e0b', '#22c55e']
    
    ax2.bar(categories, values, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Employees', fontsize=11)
    ax2.set_title('Position vs Market Benchmark', fontsize=12, fontweight='bold')
    
    for i, v in enumerate(values):
        ax2.annotate(str(v), xy=(i, v), xytext=(0, 5),
                    textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(analysis_results: Dict, analysis_type: str, 
                          equity_threshold: float) -> List[Dict]:
    """Generate key insights from compensation analysis"""
    insights = []
    
    if analysis_type == "pay_gap" and 'pay_gaps' in analysis_results:
        gaps = analysis_results['pay_gaps']
        sig_gaps = [g for g in gaps if g['significant']]
        
        if sig_gaps:
            largest = max(sig_gaps, key=lambda x: abs(x['gap_percentage']))
            insights.append({
                'title': f'Significant Pay Gap Detected',
                'description': f'{largest["group_1"]} vs {largest["group_2"]}: {abs(largest["gap_percentage"]):.1f}% gap (p={largest["p_value"]:.4f})',
                'status': 'warning'
            })
        else:
            insights.append({
                'title': 'No Significant Pay Gaps',
                'description': 'Pay differences between groups are not statistically significant.',
                'status': 'positive'
            })
    
    if analysis_type == "equity" and 'level_analysis' in analysis_results:
        levels = analysis_results['level_analysis']
        equitable = [l for l in levels if l['is_equitable']]
        
        if len(equitable) == len(levels):
            insights.append({
                'title': 'Strong Pay Equity',
                'description': f'All {len(levels)} job levels meet equity standards (CV < {equity_threshold*100:.0f}%).',
                'status': 'positive'
            })
        elif len(equitable) >= len(levels) / 2:
            insights.append({
                'title': 'Moderate Pay Equity',
                'description': f'{len(equitable)} of {len(levels)} levels meet equity standards.',
                'status': 'neutral'
            })
        else:
            worst = min(levels, key=lambda x: x['equity_score'])
            insights.append({
                'title': 'Pay Equity Concerns',
                'description': f'Only {len(equitable)} of {len(levels)} levels meet standards. {worst["level"]} has highest variation.',
                'status': 'warning'
            })
    
    if analysis_type == "benchmark" and 'vs_benchmark' in analysis_results:
        vs = analysis_results['vs_benchmark']
        total = vs['above'] + vs['at'] + vs['below']
        
        if vs['pct_above'] > 60:
            insights.append({
                'title': 'Above Market Compensation',
                'description': f'{vs["pct_above"]:.1f}% of employees are above market benchmark.',
                'status': 'positive'
            })
        elif vs['pct_below'] > 60:
            insights.append({
                'title': 'Below Market Compensation',
                'description': f'{vs["pct_below"]:.1f}% of employees are below market benchmark. Retention risk.',
                'status': 'warning'
            })
        else:
            insights.append({
                'title': 'Market-Competitive Compensation',
                'description': 'Salary distribution is balanced around market benchmark.',
                'status': 'neutral'
            })
    
    return insights


@router.post("/compensation")
async def run_compensation_analysis(request: CompensationRequest) -> Dict[str, Any]:
    """Run compensation analysis"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.salary_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Salary column '{request.salary_col}' not found")
        
        # Convert salary to numeric
        df[request.salary_col] = pd.to_numeric(df[request.salary_col], errors='coerce')
        df = df.dropna(subset=[request.salary_col])
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 valid salary records")
        
        # Calculate overall statistics
        overall_stats = calculate_salary_statistics(df[request.salary_col].values)
        
        # Analysis based on type
        analysis_results = {}
        
        if request.analysis_type == "pay_gap" and request.gender_col:
            if request.gender_col in df.columns:
                analysis_results = perform_pay_gap_analysis(df, request.salary_col, request.gender_col)
        
        elif request.analysis_type == "equity" and request.level_col:
            if request.level_col in df.columns:
                analysis_results = perform_equity_analysis(df, request.salary_col, 
                                                           request.level_col, request.equity_threshold)
        
        elif request.analysis_type == "benchmark" and request.market_benchmark:
            analysis_results = perform_benchmark_analysis(df, request.salary_col, request.market_benchmark)
        
        # Group analysis
        group_data = {}
        if request.department_col and request.department_col in df.columns:
            group_data['by_department'] = analyze_by_group(df, request.salary_col, request.department_col)
        if request.level_col and request.level_col in df.columns:
            group_data['by_level'] = analyze_by_group(df, request.salary_col, request.level_col)
        
        # Create visualizations
        visualizations = {
            'salary_distribution': create_salary_distribution_chart(
                df, request.salary_col, 
                request.market_benchmark if request.analysis_type == "benchmark" else None
            )
        }
        
        if group_data.get('by_department'):
            visualizations['by_department'] = create_group_comparison_chart(
                group_data['by_department'], "Salary by Department"
            )
        if group_data.get('by_level'):
            visualizations['by_level'] = create_group_comparison_chart(
                group_data['by_level'], "Salary by Level"
            )
        
        if request.analysis_type == "pay_gap" and 'pay_gaps' in analysis_results:
            visualizations['pay_gap'] = create_pay_gap_chart(analysis_results['pay_gaps'])
        
        if request.analysis_type == "equity" and 'level_analysis' in analysis_results:
            visualizations['equity'] = create_equity_chart(analysis_results['level_analysis'])
        
        if request.analysis_type == "benchmark" and analysis_results:
            visualizations['compa_ratio'] = create_compa_ratio_chart(analysis_results)
        
        # Generate insights
        key_insights = generate_key_insights(analysis_results, request.analysis_type, 
                                             request.equity_threshold)
        
        analyze_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare results
        results = {
            'overall_statistics': {k: _to_native_type(v) for k, v in overall_stats.items()},
            'analysis': {k: _to_native_type(v) if not isinstance(v, (list, dict)) else v 
                        for k, v in analysis_results.items()},
            'group_data': group_data,
            'employee_count': len(df)
        }
        
        summary = {
            'analysis_type': request.analysis_type,
            'employee_count': len(df),
            'avg_salary': overall_stats['mean'],
            'median_salary': overall_stats['median'],
            'salary_range': f"${overall_stats['min']:,.0f} - ${overall_stats['max']:,.0f}",
            'market_benchmark': request.market_benchmark,
            'analyze_time_ms': analyze_time_ms
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compensation analysis failed: {str(e)}")
