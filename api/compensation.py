"""
Compensation Analysis Router for FastAPI
Pay Equity, Salary Benchmarking, Compa-Ratio Analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CompensationRequest(BaseModel):
    data: List[Dict[str, Any]]
    salary_col: str
    employee_id_col: Optional[str] = None
    department_col: Optional[str] = None
    job_level_col: Optional[str] = None
    gender_col: Optional[str] = None
    tenure_col: Optional[str] = None
    performance_col: Optional[str] = None
    market_rate_col: Optional[str] = None  # External benchmark
    salary_min_col: Optional[str] = None  # Salary band min
    salary_max_col: Optional[str] = None  # Salary band max
    salary_mid_col: Optional[str] = None  # Salary band midpoint


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


def calculate_basic_stats(df: pd.DataFrame, salary_col: str) -> Dict[str, Any]:
    """Calculate basic salary statistics"""
    salaries = df[salary_col].dropna()
    
    return {
        'count': len(salaries),
        'mean': _to_native_type(salaries.mean()),
        'median': _to_native_type(salaries.median()),
        'std': _to_native_type(salaries.std()),
        'min': _to_native_type(salaries.min()),
        'max': _to_native_type(salaries.max()),
        'p25': _to_native_type(salaries.quantile(0.25)),
        'p75': _to_native_type(salaries.quantile(0.75)),
        'p90': _to_native_type(salaries.quantile(0.90)),
        'range': _to_native_type(salaries.max() - salaries.min()),
        'iqr': _to_native_type(salaries.quantile(0.75) - salaries.quantile(0.25))
    }


def calculate_compa_ratio(df: pd.DataFrame, salary_col: str, 
                          mid_col: Optional[str] = None,
                          market_col: Optional[str] = None) -> Dict[str, Any]:
    """Calculate compa-ratio (salary / midpoint or market rate)"""
    
    if mid_col and mid_col in df.columns:
        reference_col = mid_col
        reference_name = "Midpoint"
    elif market_col and market_col in df.columns:
        reference_col = market_col
        reference_name = "Market Rate"
    else:
        # Use median as reference
        median_salary = df[salary_col].median()
        df['_reference'] = median_salary
        reference_col = '_reference'
        reference_name = "Median"
    
    df['compa_ratio'] = df[salary_col] / df[reference_col]
    
    compa_ratios = df['compa_ratio'].dropna()
    
    # Categorize
    below_90 = (compa_ratios < 0.9).sum()
    between_90_110 = ((compa_ratios >= 0.9) & (compa_ratios <= 1.1)).sum()
    above_110 = (compa_ratios > 1.1).sum()
    
    return {
        'reference_type': reference_name,
        'mean_compa_ratio': _to_native_type(compa_ratios.mean()),
        'median_compa_ratio': _to_native_type(compa_ratios.median()),
        'std_compa_ratio': _to_native_type(compa_ratios.std()),
        'min_compa_ratio': _to_native_type(compa_ratios.min()),
        'max_compa_ratio': _to_native_type(compa_ratios.max()),
        'below_90_pct': _to_native_type(below_90 / len(compa_ratios) * 100),
        'between_90_110_pct': _to_native_type(between_90_110 / len(compa_ratios) * 100),
        'above_110_pct': _to_native_type(above_110 / len(compa_ratios) * 100),
        'below_90_count': int(below_90),
        'between_90_110_count': int(between_90_110),
        'above_110_count': int(above_110)
    }


def analyze_pay_equity(df: pd.DataFrame, salary_col: str, 
                       gender_col: str) -> Dict[str, Any]:
    """Analyze pay equity by gender"""
    
    # Clean gender column
    df[gender_col] = df[gender_col].astype(str).str.strip().str.upper()
    
    # Get unique genders
    genders = df[gender_col].unique()
    
    gender_stats = []
    for gender in genders:
        gender_df = df[df[gender_col] == gender]
        salaries = gender_df[salary_col].dropna()
        
        gender_stats.append({
            'gender': gender,
            'count': len(salaries),
            'mean_salary': _to_native_type(salaries.mean()),
            'median_salary': _to_native_type(salaries.median()),
            'std_salary': _to_native_type(salaries.std())
        })
    
    # Calculate pay gap (if binary gender)
    pay_gap = None
    pay_gap_pct = None
    
    if len(gender_stats) == 2:
        sorted_stats = sorted(gender_stats, key=lambda x: x['mean_salary'], reverse=True)
        higher = sorted_stats[0]
        lower = sorted_stats[1]
        
        pay_gap = higher['mean_salary'] - lower['mean_salary']
        pay_gap_pct = (pay_gap / higher['mean_salary']) * 100 if higher['mean_salary'] > 0 else 0
    
    # Statistical test
    if len(genders) == 2:
        group1 = df[df[gender_col] == genders[0]][salary_col].dropna()
        group2 = df[df[gender_col] == genders[1]][salary_col].dropna()
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        is_significant = p_value < 0.05
    else:
        t_stat, p_value, is_significant = None, None, None
    
    return {
        'gender_stats': gender_stats,
        'pay_gap': _to_native_type(pay_gap),
        'pay_gap_pct': _to_native_type(pay_gap_pct),
        't_statistic': _to_native_type(t_stat),
        'p_value': _to_native_type(p_value),
        'is_significant': is_significant
    }


def analyze_by_segment(df: pd.DataFrame, salary_col: str,
                       segment_col: str) -> List[Dict[str, Any]]:
    """Analyze salary by segment (department, level, etc.)"""
    
    segments = df[segment_col].unique()
    overall_mean = df[salary_col].mean()
    
    segment_stats = []
    for segment in segments:
        segment_df = df[df[segment_col] == segment]
        salaries = segment_df[salary_col].dropna()
        
        if len(salaries) > 0:
            segment_stats.append({
                'segment': str(segment),
                'count': len(salaries),
                'mean_salary': _to_native_type(salaries.mean()),
                'median_salary': _to_native_type(salaries.median()),
                'min_salary': _to_native_type(salaries.min()),
                'max_salary': _to_native_type(salaries.max()),
                'std_salary': _to_native_type(salaries.std()),
                'vs_overall_pct': _to_native_type((salaries.mean() - overall_mean) / overall_mean * 100)
            })
    
    # Sort by mean salary
    segment_stats.sort(key=lambda x: x['mean_salary'], reverse=True)
    
    return segment_stats


def analyze_salary_vs_tenure(df: pd.DataFrame, salary_col: str,
                              tenure_col: str) -> Dict[str, Any]:
    """Analyze relationship between salary and tenure"""
    
    valid_df = df[[salary_col, tenure_col]].dropna()
    
    if len(valid_df) < 10:
        return {'error': 'Insufficient data'}
    
    X = valid_df[tenure_col].values.reshape(-1, 1)
    y = valid_df[salary_col].values
    
    # Linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Correlation
    correlation, p_value = stats.pearsonr(valid_df[tenure_col], valid_df[salary_col])
    
    # Predicted values for trend line
    tenure_range = np.linspace(valid_df[tenure_col].min(), valid_df[tenure_col].max(), 50)
    predicted_salary = model.predict(tenure_range.reshape(-1, 1))
    
    return {
        'correlation': _to_native_type(correlation),
        'p_value': _to_native_type(p_value),
        'is_significant': p_value < 0.05,
        'slope': _to_native_type(model.coef_[0]),
        'intercept': _to_native_type(model.intercept_),
        'r_squared': _to_native_type(model.score(X, y)),
        'salary_increase_per_year': _to_native_type(model.coef_[0]),
        'trend_data': {
            'tenure': [_to_native_type(t) for t in tenure_range],
            'predicted_salary': [_to_native_type(s) for s in predicted_salary]
        }
    }


def analyze_salary_vs_performance(df: pd.DataFrame, salary_col: str,
                                   performance_col: str) -> Dict[str, Any]:
    """Analyze relationship between salary and performance"""
    
    valid_df = df[[salary_col, performance_col]].dropna()
    
    if len(valid_df) < 10:
        return {'error': 'Insufficient data'}
    
    # Check if performance is categorical or numeric
    if valid_df[performance_col].dtype == 'object' or valid_df[performance_col].nunique() <= 5:
        # Categorical - group analysis
        perf_groups = []
        for perf in sorted(valid_df[performance_col].unique()):
            group_df = valid_df[valid_df[performance_col] == perf]
            perf_groups.append({
                'performance': str(perf),
                'count': len(group_df),
                'mean_salary': _to_native_type(group_df[salary_col].mean()),
                'median_salary': _to_native_type(group_df[salary_col].median())
            })
        
        # ANOVA test
        groups = [valid_df[valid_df[performance_col] == p][salary_col].values 
                  for p in valid_df[performance_col].unique()]
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
        else:
            f_stat, p_value = None, None
        
        return {
            'type': 'categorical',
            'performance_groups': perf_groups,
            'f_statistic': _to_native_type(f_stat),
            'p_value': _to_native_type(p_value),
            'is_significant': p_value < 0.05 if p_value else None
        }
    else:
        # Numeric - correlation
        correlation, p_value = stats.pearsonr(valid_df[performance_col], valid_df[salary_col])
        
        return {
            'type': 'numeric',
            'correlation': _to_native_type(correlation),
            'p_value': _to_native_type(p_value),
            'is_significant': p_value < 0.05
        }


def identify_outliers(df: pd.DataFrame, salary_col: str,
                      employee_id_col: Optional[str] = None) -> Dict[str, Any]:
    """Identify salary outliers using IQR method"""
    
    salaries = df[salary_col].dropna()
    
    Q1 = salaries.quantile(0.25)
    Q3 = salaries.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df[salary_col] < lower_bound) | (df[salary_col] > upper_bound)
    outliers_df = df[outlier_mask]
    
    outliers = []
    for _, row in outliers_df.iterrows():
        salary = row[salary_col]
        outlier_type = 'high' if salary > upper_bound else 'low'
        
        outlier = {
            'salary': _to_native_type(salary),
            'type': outlier_type,
            'deviation_pct': _to_native_type(abs(salary - salaries.median()) / salaries.median() * 100)
        }
        
        if employee_id_col and employee_id_col in df.columns:
            outlier['employee_id'] = str(row[employee_id_col])
        
        outliers.append(outlier)
    
    return {
        'lower_bound': _to_native_type(lower_bound),
        'upper_bound': _to_native_type(upper_bound),
        'outlier_count': len(outliers),
        'outlier_pct': _to_native_type(len(outliers) / len(df) * 100),
        'outliers': outliers[:20]  # Top 20
    }


def create_salary_distribution_chart(df: pd.DataFrame, salary_col: str,
                                      basic_stats: Dict) -> str:
    """Create salary distribution visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    salaries = df[salary_col].dropna()
    
    # Histogram
    ax1.hist(salaries, bins=30, color='#3b82f6', edgecolor='white', linewidth=1, alpha=0.7)
    ax1.axvline(basic_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: ${basic_stats['mean']:,.0f}")
    ax1.axvline(basic_stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: ${basic_stats['median']:,.0f}")
    ax1.set_xlabel('Salary ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Salary Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Box plot
    bp = ax2.boxplot(salaries, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3b82f6')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Salary ($)')
    ax2.set_title('Salary Box Plot', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['All Employees'])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_compa_ratio_chart(compa_data: Dict, df: pd.DataFrame) -> str:
    """Create compa-ratio visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    labels = ['Below 90%', '90-110%', 'Above 110%']
    sizes = [compa_data['below_90_count'], compa_data['between_90_110_count'], compa_data['above_110_count']]
    colors = ['#ef4444', '#22c55e', '#3b82f6']
    explode = (0.05, 0, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Compa-Ratio Distribution (vs {compa_data["reference_type"]})', fontsize=12, fontweight='bold')
    
    # Histogram of compa ratios
    if 'compa_ratio' in df.columns:
        ratios = df['compa_ratio'].dropna()
        ax2.hist(ratios, bins=20, color='#3b82f6', edgecolor='white', linewidth=1, alpha=0.7)
        ax2.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)')
        ax2.axvline(0.9, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axvline(1.1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axvline(compa_data['mean_compa_ratio'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {compa_data['mean_compa_ratio']:.2f}")
    
    ax2.set_xlabel('Compa-Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Compa-Ratio Histogram', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_pay_equity_chart(equity_data: Dict) -> str:
    """Create pay equity visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    gender_stats = equity_data['gender_stats']
    genders = [g['gender'] for g in gender_stats]
    means = [g['mean_salary'] for g in gender_stats]
    
    # Bar chart
    colors = ['#3b82f6', '#ec4899'] if len(genders) == 2 else plt.cm.Set2(np.linspace(0, 1, len(genders)))
    bars = ax1.bar(genders, means, color=colors, edgecolor='white', linewidth=2)
    
    ax1.set_ylabel('Average Salary ($)')
    ax1.set_title('Average Salary by Gender', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means) * 0.02,
                f'${mean:,.0f}', ha='center', fontsize=10, fontweight='bold')
    
    # Pay gap visualization
    if equity_data['pay_gap'] is not None:
        gap_pct = equity_data['pay_gap_pct']
        
        ax2.barh(['Pay Gap'], [gap_pct], color='#ef4444' if gap_pct > 5 else '#f59e0b' if gap_pct > 2 else '#22c55e',
                edgecolor='white', linewidth=2)
        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
        ax2.set_xlabel('Pay Gap (%)')
        ax2.set_title(f'Gender Pay Gap: {gap_pct:.1f}%', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, max(10, gap_pct + 2))
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        sig_text = "Statistically Significant" if equity_data['is_significant'] else "Not Significant"
        ax2.text(0.5, -0.15, f"p-value: {equity_data['p_value']:.4f} ({sig_text})",
                transform=ax2.transAxes, ha='center', fontsize=10,
                color='red' if equity_data['is_significant'] else 'gray')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_chart(segment_data: List[Dict], segment_name: str) -> str:
    """Create segment analysis visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    segments = [s['segment'] for s in segment_data[:10]]
    means = [s['mean_salary'] for s in segment_data[:10]]
    overall_mean = sum(means) / len(means)
    
    colors = ['#22c55e' if s['vs_overall_pct'] >= 0 else '#ef4444' for s in segment_data[:10]]
    
    bars = ax.barh(segments[::-1], means[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    ax.axvline(x=overall_mean, color='blue', linestyle='--', linewidth=2, label=f'Overall Mean: ${overall_mean:,.0f}')
    
    ax.set_xlabel('Average Salary ($)')
    ax.set_title(f'Average Salary by {segment_name}', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, mean in zip(bars, means[::-1]):
        ax.text(bar.get_width() + max(means) * 0.01, bar.get_y() + bar.get_height()/2,
               f'${mean:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(basic_stats: Dict, compa_data: Optional[Dict],
                      equity_data: Optional[Dict], segment_data: Optional[List]) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    # Salary spread
    spread = (basic_stats['max'] - basic_stats['min']) / basic_stats['median'] * 100 if basic_stats['median'] > 0 else 0
    if spread > 200:
        insights.append({
            'title': f'High Salary Spread: {spread:.0f}%',
            'description': 'Large variation in compensation. Review for internal equity.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': f'Salary Range: ${basic_stats["min"]:,.0f} - ${basic_stats["max"]:,.0f}',
            'description': f'Median salary is ${basic_stats["median"]:,.0f}',
            'status': 'neutral'
        })
    
    # Compa-ratio
    if compa_data:
        mean_cr = compa_data['mean_compa_ratio']
        if mean_cr < 0.95:
            insights.append({
                'title': f'Below Market: Avg Compa-Ratio {mean_cr:.2f}',
                'description': f'{compa_data["below_90_pct"]:.1f}% of employees paid below 90% of reference.',
                'status': 'warning'
            })
        elif mean_cr > 1.05:
            insights.append({
                'title': f'Above Market: Avg Compa-Ratio {mean_cr:.2f}',
                'description': 'Compensation is competitive but monitor costs.',
                'status': 'positive'
            })
        else:
            insights.append({
                'title': f'Market Aligned: Avg Compa-Ratio {mean_cr:.2f}',
                'description': 'Compensation is well-aligned with reference.',
                'status': 'positive'
            })
    
    # Pay equity
    if equity_data and equity_data.get('pay_gap_pct') is not None:
        gap = equity_data['pay_gap_pct']
        if gap > 5 and equity_data['is_significant']:
            insights.append({
                'title': f'Pay Gap Alert: {gap:.1f}%',
                'description': 'Statistically significant gender pay gap detected.',
                'status': 'warning'
            })
        elif gap <= 2:
            insights.append({
                'title': f'Pay Equity: {gap:.1f}% Gap',
                'description': 'Gender pay gap within acceptable range.',
                'status': 'positive'
            })
    
    # Top paid segment
    if segment_data and len(segment_data) > 0:
        top = segment_data[0]
        insights.append({
            'title': f'Highest Paid: {top["segment"]}',
            'description': f'Average ${top["mean_salary"]:,.0f} ({top["vs_overall_pct"]:+.1f}% vs overall)',
            'status': 'neutral'
        })
    
    return insights


@router.post("/compensation")
async def run_compensation_analysis(request: CompensationRequest) -> Dict[str, Any]:
    """
    Perform Compensation Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 records")
        
        if request.salary_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Salary column '{request.salary_col}' not found")
        
        results = {}
        visualizations = {}
        
        # Basic statistics
        basic_stats = calculate_basic_stats(df, request.salary_col)
        results['basic_stats'] = basic_stats
        visualizations['distribution_chart'] = create_salary_distribution_chart(df, request.salary_col, basic_stats)
        
        # Compa-ratio analysis
        compa_data = calculate_compa_ratio(
            df, request.salary_col, 
            request.salary_mid_col, 
            request.market_rate_col
        )
        results['compa_ratio'] = compa_data
        visualizations['compa_ratio_chart'] = create_compa_ratio_chart(compa_data, df)
        
        # Pay equity analysis
        equity_data = None
        if request.gender_col and request.gender_col in df.columns:
            equity_data = analyze_pay_equity(df, request.salary_col, request.gender_col)
            results['pay_equity'] = equity_data
            visualizations['pay_equity_chart'] = create_pay_equity_chart(equity_data)
        
        # Segment analysis - Department
        dept_data = None
        if request.department_col and request.department_col in df.columns:
            dept_data = analyze_by_segment(df, request.salary_col, request.department_col)
            results['department_analysis'] = dept_data
            visualizations['department_chart'] = create_segment_chart(dept_data, 'Department')
        
        # Segment analysis - Job Level
        level_data = None
        if request.job_level_col and request.job_level_col in df.columns:
            level_data = analyze_by_segment(df, request.salary_col, request.job_level_col)
            results['level_analysis'] = level_data
            visualizations['level_chart'] = create_segment_chart(level_data, 'Job Level')
        
        # Tenure analysis
        if request.tenure_col and request.tenure_col in df.columns:
            tenure_analysis = analyze_salary_vs_tenure(df, request.salary_col, request.tenure_col)
            results['tenure_analysis'] = tenure_analysis
        
        # Performance analysis
        if request.performance_col and request.performance_col in df.columns:
            perf_analysis = analyze_salary_vs_performance(df, request.salary_col, request.performance_col)
            results['performance_analysis'] = perf_analysis
        
        # Outliers
        outliers = identify_outliers(df, request.salary_col, request.employee_id_col)
        results['outliers'] = outliers
        
        # Generate insights
        insights = generate_insights(basic_stats, compa_data, equity_data, dept_data or level_data)
        
        # Summary
        summary = {
            'total_employees': len(df),
            'avg_salary': basic_stats['mean'],
            'median_salary': basic_stats['median'],
            'salary_range': f"${basic_stats['min']:,.0f} - ${basic_stats['max']:,.0f}",
            'avg_compa_ratio': compa_data['mean_compa_ratio'],
            'pay_gap_pct': equity_data['pay_gap_pct'] if equity_data else None,
            'outlier_count': outliers['outlier_count']
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
        raise HTTPException(status_code=500, detail=f"Compensation analysis failed: {str(e)}")
