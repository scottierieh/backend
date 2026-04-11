"""
Absenteeism Analysis Router for FastAPI
Absence patterns, costs, prediction, and root cause analysis
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
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class AbsenteeismRequest(BaseModel):
    data: List[Dict[str, Any]]
    employee_id_col: Optional[str] = None
    absence_date_col: Optional[str] = None
    absence_hours_col: Optional[str] = None  # or days
    absence_type_col: Optional[str] = None  # Sick, Personal, etc.
    department_col: Optional[str] = None
    job_level_col: Optional[str] = None
    tenure_col: Optional[str] = None
    salary_col: Optional[str] = None  # For cost calculation
    age_col: Optional[str] = None
    gender_col: Optional[str] = None
    performance_col: Optional[str] = None
    shift_col: Optional[str] = None


def _to_native(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_absence_metrics(df: pd.DataFrame, employee_col: str, 
                               hours_col: str) -> Dict[str, Any]:
    """Calculate overall absence metrics"""
    
    # Per employee stats
    emp_absence = df.groupby(employee_col)[hours_col].sum()
    
    total_employees = df[employee_col].nunique()
    total_absence_hours = df[hours_col].sum()
    
    # Assume 2080 working hours/year (40h * 52w)
    working_hours_per_year = 2080
    total_available_hours = total_employees * working_hours_per_year
    
    absence_rate = (total_absence_hours / total_available_hours) * 100 if total_available_hours > 0 else 0
    
    # Bradford Factor: S² × D (S = number of spells, D = total days)
    # Higher = more problematic (frequent short absences worse than few long ones)
    emp_spells = df.groupby(employee_col).size()  # Number of absence records
    emp_days = df.groupby(employee_col)[hours_col].sum() / 8  # Convert to days
    bradford_factors = (emp_spells ** 2) * emp_days
    
    return {
        'total_employees': _to_native(total_employees),
        'total_absence_hours': _to_native(total_absence_hours),
        'total_absence_days': _to_native(total_absence_hours / 8),
        'absence_rate_pct': _to_native(absence_rate),
        'avg_absence_per_employee_hours': _to_native(emp_absence.mean()),
        'avg_absence_per_employee_days': _to_native(emp_absence.mean() / 8),
        'median_absence_hours': _to_native(emp_absence.median()),
        'max_absence_hours': _to_native(emp_absence.max()),
        'employees_with_absence': _to_native((emp_absence > 0).sum()),
        'zero_absence_employees': _to_native((emp_absence == 0).sum()),
        'avg_bradford_factor': _to_native(bradford_factors.mean()),
        'max_bradford_factor': _to_native(bradford_factors.max()),
        'high_bradford_count': _to_native((bradford_factors > 500).sum())  # Threshold for concern
    }


def analyze_by_type(df: pd.DataFrame, hours_col: str, 
                    type_col: str) -> List[Dict[str, Any]]:
    """Analyze absence by type"""
    
    type_stats = df.groupby(type_col).agg({
        hours_col: ['sum', 'mean', 'count']
    }).reset_index()
    type_stats.columns = ['type', 'total_hours', 'avg_hours', 'occurrences']
    
    total = type_stats['total_hours'].sum()
    
    results = []
    for _, row in type_stats.iterrows():
        results.append({
            'type': str(row['type']),
            'total_hours': _to_native(row['total_hours']),
            'total_days': _to_native(row['total_hours'] / 8),
            'avg_hours_per_occurrence': _to_native(row['avg_hours']),
            'occurrences': _to_native(row['occurrences']),
            'percentage': _to_native(row['total_hours'] / total * 100) if total > 0 else 0
        })
    
    return sorted(results, key=lambda x: x['total_hours'], reverse=True)


def analyze_by_department(df: pd.DataFrame, employee_col: str,
                          hours_col: str, dept_col: str) -> List[Dict[str, Any]]:
    """Analyze absence by department"""
    
    dept_stats = df.groupby(dept_col).agg({
        employee_col: 'nunique',
        hours_col: ['sum', 'mean']
    }).reset_index()
    dept_stats.columns = ['department', 'employees', 'total_hours', 'avg_hours']
    
    overall_avg = df.groupby(employee_col)[hours_col].sum().mean()
    
    results = []
    for _, row in dept_stats.iterrows():
        emp_avg = row['total_hours'] / row['employees'] if row['employees'] > 0 else 0
        results.append({
            'department': str(row['department']),
            'employees': _to_native(row['employees']),
            'total_hours': _to_native(row['total_hours']),
            'total_days': _to_native(row['total_hours'] / 8),
            'avg_per_employee': _to_native(emp_avg),
            'vs_overall': _to_native((emp_avg - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
        })
    
    # ANOVA test
    groups = [df[df[dept_col] == d].groupby(employee_col)[hours_col].sum().values 
              for d in df[dept_col].unique() if len(df[df[dept_col] == d]) >= 5]
    
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = None, None
    
    return {
        'departments': sorted(results, key=lambda x: x['avg_per_employee'], reverse=True),
        'f_statistic': _to_native(f_stat),
        'p_value': _to_native(p_value),
        'significant_difference': p_value < 0.05 if p_value else None
    }


def analyze_temporal_patterns(df: pd.DataFrame, date_col: str, 
                               hours_col: str) -> Dict[str, Any]:
    """Analyze temporal patterns in absences"""
    
    df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['_date'])
    
    if len(df) == 0:
        return {}
    
    # Day of week
    df['_dow'] = df['_date'].dt.dayofweek
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = df.groupby('_dow')[hours_col].sum()
    
    by_day_of_week = []
    for dow in range(7):
        hours = dow_stats.get(dow, 0)
        by_day_of_week.append({
            'day': dow_names[dow],
            'day_num': dow,
            'total_hours': _to_native(hours),
            'percentage': _to_native(hours / dow_stats.sum() * 100) if dow_stats.sum() > 0 else 0
        })
    
    # Month
    df['_month'] = df['_date'].dt.month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_stats = df.groupby('_month')[hours_col].sum()
    
    by_month = []
    for m in range(1, 13):
        hours = month_stats.get(m, 0)
        by_month.append({
            'month': month_names[m-1],
            'month_num': m,
            'total_hours': _to_native(hours),
            'percentage': _to_native(hours / month_stats.sum() * 100) if month_stats.sum() > 0 else 0
        })
    
    # Peak analysis
    peak_day = dow_names[dow_stats.idxmax()] if len(dow_stats) > 0 else None
    peak_month = month_names[month_stats.idxmax() - 1] if len(month_stats) > 0 else None
    
    # Monday/Friday pattern (potential indicator of extended weekends)
    mon_fri_pct = ((dow_stats.get(0, 0) + dow_stats.get(4, 0)) / dow_stats.sum() * 100) if dow_stats.sum() > 0 else 0
    
    return {
        'by_day_of_week': by_day_of_week,
        'by_month': by_month,
        'peak_day': peak_day,
        'peak_month': peak_month,
        'monday_friday_percentage': _to_native(mon_fri_pct),
        'potential_extended_weekends': mon_fri_pct > 50
    }


def calculate_absence_cost(df: pd.DataFrame, employee_col: str,
                           hours_col: str, salary_col: str) -> Dict[str, Any]:
    """Calculate cost of absenteeism"""
    
    # Merge salary data (assume salary is annual)
    emp_salary = df.groupby(employee_col)[salary_col].first()
    emp_absence = df.groupby(employee_col)[hours_col].sum()
    
    # Hourly rate = annual salary / 2080
    hourly_rates = emp_salary / 2080
    
    # Direct cost = hours absent × hourly rate
    direct_costs = emp_absence * hourly_rates
    
    # Indirect costs (typically 1.5-3x direct costs for replacement, overtime, productivity loss)
    indirect_multiplier = 1.5
    indirect_costs = direct_costs * indirect_multiplier
    
    total_direct = direct_costs.sum()
    total_indirect = indirect_costs.sum()
    total_cost = total_direct + total_indirect
    
    return {
        'total_direct_cost': _to_native(total_direct),
        'total_indirect_cost': _to_native(total_indirect),
        'total_cost': _to_native(total_cost),
        'avg_cost_per_employee': _to_native(total_cost / len(emp_absence)),
        'cost_per_absence_hour': _to_native(total_cost / emp_absence.sum()) if emp_absence.sum() > 0 else 0,
        'indirect_multiplier_used': indirect_multiplier
    }


def analyze_correlations(df: pd.DataFrame, employee_col: str, hours_col: str,
                         tenure_col: Optional[str] = None,
                         age_col: Optional[str] = None,
                         performance_col: Optional[str] = None) -> Dict[str, Any]:
    """Analyze correlations with absence"""
    
    emp_absence = df.groupby(employee_col)[hours_col].sum()
    
    correlations = {}
    
    if tenure_col and tenure_col in df.columns:
        emp_tenure = df.groupby(employee_col)[tenure_col].first()
        merged = pd.DataFrame({'absence': emp_absence, 'tenure': emp_tenure}).dropna()
        if len(merged) >= 10:
            corr, p = stats.pearsonr(merged['absence'], merged['tenure'])
            correlations['tenure'] = {
                'correlation': _to_native(corr),
                'p_value': _to_native(p),
                'significant': p < 0.05,
                'interpretation': 'Higher tenure → More absence' if corr > 0 else 'Higher tenure → Less absence'
            }
    
    if age_col and age_col in df.columns:
        emp_age = df.groupby(employee_col)[age_col].first()
        merged = pd.DataFrame({'absence': emp_absence, 'age': emp_age}).dropna()
        if len(merged) >= 10:
            corr, p = stats.pearsonr(merged['absence'], merged['age'])
            correlations['age'] = {
                'correlation': _to_native(corr),
                'p_value': _to_native(p),
                'significant': p < 0.05,
                'interpretation': 'Older → More absence' if corr > 0 else 'Older → Less absence'
            }
    
    if performance_col and performance_col in df.columns:
        emp_perf = df.groupby(employee_col)[performance_col].first()
        merged = pd.DataFrame({'absence': emp_absence, 'performance': emp_perf}).dropna()
        if len(merged) >= 10:
            corr, p = stats.pearsonr(merged['absence'], merged['performance'])
            correlations['performance'] = {
                'correlation': _to_native(corr),
                'p_value': _to_native(p),
                'significant': p < 0.05,
                'interpretation': 'Higher performance → More absence' if corr > 0 else 'Higher performance → Less absence'
            }
    
    return correlations


def identify_high_risk_employees(df: pd.DataFrame, employee_col: str,
                                  hours_col: str) -> Dict[str, Any]:
    """Identify employees with high absenteeism"""
    
    emp_stats = df.groupby(employee_col).agg({
        hours_col: ['sum', 'count']
    }).reset_index()
    emp_stats.columns = [employee_col, 'total_hours', 'occurrences']
    
    # Bradford Factor
    emp_stats['bradford'] = (emp_stats['occurrences'] ** 2) * (emp_stats['total_hours'] / 8)
    
    # Thresholds
    hours_threshold = emp_stats['total_hours'].quantile(0.9)  # Top 10%
    bradford_threshold = 500  # Standard threshold
    
    high_hours = emp_stats[emp_stats['total_hours'] >= hours_threshold]
    high_bradford = emp_stats[emp_stats['bradford'] >= bradford_threshold]
    
    return {
        'high_hours_threshold': _to_native(hours_threshold),
        'high_hours_count': len(high_hours),
        'high_hours_employees': high_hours[[employee_col, 'total_hours', 'occurrences']].to_dict('records'),
        'high_bradford_threshold': bradford_threshold,
        'high_bradford_count': len(high_bradford),
        'high_bradford_employees': high_bradford[[employee_col, 'bradford', 'occurrences']].to_dict('records')
    }


def create_overview_chart(metrics: Dict, temporal: Dict) -> str:
    """Create overview visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absence rate gauge
    ax1 = axes[0]
    rate = metrics.get('absence_rate_pct', 0)
    
    # Gauge background
    theta = np.linspace(0, np.pi, 100)
    ax1.fill_between(theta, 0.7, 1.0, alpha=0.3, color='lightgray')
    
    # Color zones
    ax1.fill_between(theta[:33], 0.7, 1.0, alpha=0.5, color='green')
    ax1.fill_between(theta[33:66], 0.7, 1.0, alpha=0.5, color='orange')
    ax1.fill_between(theta[66:], 0.7, 1.0, alpha=0.5, color='red')
    
    # Needle
    rate_normalized = min(rate / 10, 1)  # Normalize to 0-10% scale
    needle_angle = np.pi * (1 - rate_normalized)
    ax1.annotate('', xy=(needle_angle, 0.95), xytext=(np.pi/2, 0.3),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=3))
    
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(0, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'Absence Rate: {rate:.1f}%', fontsize=14, fontweight='bold', y=0.1)
    ax1.text(0.1, 0.65, '0%', fontsize=10)
    ax1.text(np.pi - 0.2, 0.65, '10%', fontsize=10)
    
    # Day of week distribution
    ax2 = axes[1]
    if temporal.get('by_day_of_week'):
        days = [d['day'][:3] for d in temporal['by_day_of_week'][:5]]  # Mon-Fri
        hours = [d['total_hours'] for d in temporal['by_day_of_week'][:5]]
        
        colors = ['#ef4444' if d in ['Mon', 'Fri'] else '#3b82f6' for d in days]
        bars = ax2.bar(days, hours, color=colors, edgecolor='white', linewidth=2)
        
        ax2.set_ylabel('Total Absence Hours')
        ax2.set_title('Absence by Day of Week', fontsize=12, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Annotate Mon/Fri if high
        if temporal.get('potential_extended_weekends'):
            ax2.annotate('⚠️ Extended Weekend Pattern', xy=(0.5, 0.95), 
                        xycoords='axes fraction', ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_type_chart(by_type: List[Dict]) -> str:
    """Create absence type visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    ax1 = axes[0]
    types = [t['type'] for t in by_type[:6]]
    hours = [t['total_hours'] for t in by_type[:6]]
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
    
    wedges, texts, autotexts = ax1.pie(hours, labels=types, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title('Absence by Type', fontsize=12, fontweight='bold')
    
    # Horizontal bar
    ax2 = axes[1]
    bars = ax2.barh(types[::-1], hours[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    ax2.set_xlabel('Total Hours')
    ax2.set_title('Hours by Absence Type', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, h in zip(bars, hours[::-1]):
        ax2.text(bar.get_width() + max(hours) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{h:.0f}h', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_department_chart(dept_data: Dict) -> str:
    """Create department comparison visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    depts = dept_data['departments'][:10]
    names = [d['department'][:15] for d in depts]
    avgs = [d['avg_per_employee'] for d in depts]
    
    overall_avg = np.mean(avgs)
    colors = ['#ef4444' if a > overall_avg else '#22c55e' for a in avgs]
    
    bars = ax.barh(names[::-1], avgs[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    ax.axvline(x=overall_avg, color='blue', linestyle='--', linewidth=2, 
              label=f'Overall Avg: {overall_avg:.1f}h')
    
    ax.set_xlabel('Avg Absence Hours per Employee')
    ax.set_title('Absence by Department', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if dept_data.get('significant_difference'):
        ax.annotate(f"ANOVA p={dept_data['p_value']:.4f} (Significant)", 
                   xy=(0.95, 0.02), xycoords='axes fraction', ha='right', fontsize=9, color='red')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_temporal_chart(temporal: Dict) -> str:
    """Create temporal pattern visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Monthly trend
    ax1 = axes[0]
    if temporal.get('by_month'):
        months = [m['month'] for m in temporal['by_month']]
        hours = [m['total_hours'] for m in temporal['by_month']]
        
        ax1.bar(months, hours, color='#3b82f6', edgecolor='white', linewidth=2)
        ax1.set_ylabel('Total Absence Hours')
        ax1.set_title('Monthly Absence Pattern', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        if temporal.get('peak_month'):
            ax1.annotate(f"Peak: {temporal['peak_month']}", xy=(0.95, 0.95),
                        xycoords='axes fraction', ha='right', fontsize=10, fontweight='bold')
    
    # Day of week radar
    ax2 = axes[1]
    if temporal.get('by_day_of_week'):
        dow = temporal['by_day_of_week'][:5]  # Mon-Fri
        
        labels = [d['day'][:3] for d in dow]
        values = [d['percentage'] for d in dow]
        values.append(values[0])  # Close the polygon
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles.append(angles[0])
        
        ax2 = fig.add_subplot(122, polar=True)
        ax2.fill(angles, values, alpha=0.25, color='#3b82f6')
        ax2.plot(angles, values, 'o-', color='#3b82f6', linewidth=2)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels)
        ax2.set_title('Day of Week Pattern', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(metrics: Dict, temporal: Dict, dept_data: Optional[Dict],
                      correlations: Dict, costs: Optional[Dict]) -> List[Dict[str, Any]]:
    """Generate actionable insights"""
    insights = []
    
    # Absence rate assessment
    rate = metrics.get('absence_rate_pct', 0)
    if rate < 3:
        insights.append({
            'title': f'Low Absence Rate: {rate:.1f}%',
            'description': 'Absence rate is within healthy range (<3%).',
            'status': 'positive'
        })
    elif rate < 5:
        insights.append({
            'title': f'Moderate Absence Rate: {rate:.1f}%',
            'description': 'Absence rate is slightly elevated. Industry average is 2-4%.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'High Absence Rate: {rate:.1f}%',
            'description': 'Absence rate exceeds recommended threshold. Investigation needed.',
            'status': 'warning'
        })
    
    # Bradford Factor
    if metrics.get('high_bradford_count', 0) > 0:
        insights.append({
            'title': f'{metrics["high_bradford_count"]} High Bradford Factor Employees',
            'description': 'These employees have frequent short-term absences requiring attention.',
            'status': 'warning'
        })
    
    # Extended weekends
    if temporal.get('potential_extended_weekends'):
        insights.append({
            'title': 'Extended Weekend Pattern Detected',
            'description': f'{temporal["monday_friday_percentage"]:.1f}% of absences on Monday/Friday.',
            'status': 'warning'
        })
    
    # Department differences
    if dept_data and dept_data.get('significant_difference'):
        top_dept = dept_data['departments'][0]['department']
        insights.append({
            'title': f'Department Disparity: {top_dept}',
            'description': 'Statistically significant difference in absence rates between departments.',
            'status': 'warning'
        })
    
    # Performance correlation
    if correlations.get('performance') and correlations['performance']['significant']:
        corr = correlations['performance']['correlation']
        if corr < 0:
            insights.append({
                'title': 'Performance-Absence Link',
                'description': 'Lower performers tend to have higher absence rates.',
                'status': 'neutral'
            })
    
    # Cost insight
    if costs:
        insights.append({
            'title': f'Annual Absence Cost: ${costs["total_cost"]:,.0f}',
            'description': f'Avg ${costs["avg_cost_per_employee"]:,.0f} per employee including indirect costs.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/absenteeism")
async def run_absenteeism_analysis(request: AbsenteeismRequest) -> Dict[str, Any]:
    """
    Perform Absenteeism Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 records")
        
        results = {}
        visualizations = {}
        
        employee_col = request.employee_id_col or 'employee_id'
        hours_col = request.absence_hours_col or 'absence_hours'
        
        if employee_col not in df.columns or hours_col not in df.columns:
            raise HTTPException(status_code=400, detail="Employee ID and Absence Hours columns required")
        
        # Overall metrics
        metrics = calculate_absence_metrics(df, employee_col, hours_col)
        results['metrics'] = metrics
        
        # By type
        if request.absence_type_col and request.absence_type_col in df.columns:
            by_type = analyze_by_type(df, hours_col, request.absence_type_col)
            results['by_type'] = by_type
            visualizations['type_chart'] = create_type_chart(by_type)
        else:
            by_type = None
        
        # By department
        dept_data = None
        if request.department_col and request.department_col in df.columns:
            dept_data = analyze_by_department(df, employee_col, hours_col, request.department_col)
            results['by_department'] = dept_data
            visualizations['department_chart'] = create_department_chart(dept_data)
        
        # Temporal patterns
        temporal = {}
        if request.absence_date_col and request.absence_date_col in df.columns:
            temporal = analyze_temporal_patterns(df, request.absence_date_col, hours_col)
            results['temporal_patterns'] = temporal
            if temporal:
                visualizations['temporal_chart'] = create_temporal_chart(temporal)
        
        # Overview chart
        visualizations['overview_chart'] = create_overview_chart(metrics, temporal)
        
        # Cost analysis
        costs = None
        if request.salary_col and request.salary_col in df.columns:
            costs = calculate_absence_cost(df, employee_col, hours_col, request.salary_col)
            results['costs'] = costs
        
        # Correlations
        correlations = analyze_correlations(
            df, employee_col, hours_col,
            request.tenure_col, request.age_col, request.performance_col
        )
        if correlations:
            results['correlations'] = correlations
        
        # High risk employees
        high_risk = identify_high_risk_employees(df, employee_col, hours_col)
        results['high_risk'] = high_risk
        
        # Generate insights
        insights = generate_insights(metrics, temporal, dept_data, correlations, costs)
        
        # Summary
        summary = {
            'total_employees': metrics['total_employees'],
            'absence_rate': metrics['absence_rate_pct'],
            'total_absence_days': metrics['total_absence_days'],
            'avg_per_employee_days': metrics['avg_absence_per_employee_days'],
            'total_cost': costs['total_cost'] if costs else None,
            'high_risk_count': high_risk['high_bradford_count']
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
        raise HTTPException(status_code=500, detail=f"Absenteeism analysis failed: {str(e)}")
