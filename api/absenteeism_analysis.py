"""
Absenteeism Analysis Router for FastAPI
Analyzes employee absence patterns, Bradford Factor, and cost impact
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
import time
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class AbsenteeismRequest(BaseModel):
    data: List[Dict[str, Any]]
    employee_col: str
    date_col: str
    duration_col: str
    dept_col: Optional[str] = None
    reason_col: Optional[str] = None
    cost_per_day: float = 350


def _to_native_type(obj):
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def calculate_bradford_factor(spells: int, total_days: int) -> float:
    """
    Calculate Bradford Factor: S² × D
    S = number of separate absence spells
    D = total days absent
    """
    return spells * spells * total_days


def get_risk_level(bradford_factor: float) -> str:
    """Determine risk level based on Bradford Factor"""
    if bradford_factor < 50:
        return 'Low'
    elif bradford_factor < 200:
        return 'Medium'
    elif bradford_factor < 450:
        return 'High'
    else:
        return 'Critical'


def get_season(month: int) -> str:
    """Get season from month number"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def calculate_department_stats(df: pd.DataFrame, dept_col: str, duration_col: str, 
                                employee_col: str, cost_per_day: float) -> List[Dict]:
    """Calculate statistics by department"""
    if not dept_col or dept_col not in df.columns:
        return []
    
    results = []
    total_employees = df[employee_col].nunique()
    
    for dept in df[dept_col].unique():
        dept_df = df[df[dept_col] == dept]
        
        emp_count = dept_df[employee_col].nunique()
        total_absences = len(dept_df)
        total_days = dept_df[duration_col].sum()
        avg_duration = dept_df[duration_col].mean()
        
        # Assuming 250 work days per year
        absence_rate = (total_days / (emp_count * 250)) * 100 if emp_count > 0 else 0
        
        results.append({
            'department': str(dept),
            'total_employees': emp_count,
            'total_absences': total_absences,
            'absence_rate': float(absence_rate),
            'avg_duration': float(avg_duration),
            'total_days_lost': int(total_days),
            'cost_impact': float(total_days * cost_per_day),
        })
    
    results.sort(key=lambda x: x['absence_rate'], reverse=True)
    return results


def calculate_reason_breakdown(df: pd.DataFrame, reason_col: str, duration_col: str) -> List[Dict]:
    """Calculate breakdown by absence reason"""
    if not reason_col or reason_col not in df.columns:
        return []
    
    total = len(df)
    results = []
    
    for reason in df[reason_col].unique():
        reason_df = df[df[reason_col] == reason]
        count = len(reason_df)
        total_days = reason_df[duration_col].sum()
        avg_duration = reason_df[duration_col].mean()
        
        results.append({
            'reason': str(reason),
            'count': count,
            'pct': count / total if total > 0 else 0,
            'avg_duration': float(avg_duration),
            'total_days': int(total_days),
        })
    
    results.sort(key=lambda x: x['count'], reverse=True)
    return results


def calculate_day_of_week_pattern(df: pd.DataFrame, date_col: str, duration_col: str) -> List[Dict]:
    """Calculate absence pattern by day of week"""
    results = []
    
    try:
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['_dow'] = df['_date'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        total_absences = len(df)
        
        for i, day in enumerate(DAYS_OF_WEEK):
            day_df = df[df['_dow'] == i]
            count = len(day_df)
            rate = (count / total_absences * 100) if total_absences > 0 else 0
            
            results.append({
                'period': day,
                'absence_rate': float(rate),
                'count': count,
            })
    except Exception:
        # Return default if date parsing fails
        for day in DAYS_OF_WEEK:
            results.append({'period': day, 'absence_rate': 0, 'count': 0})
    
    return results


def calculate_monthly_pattern(df: pd.DataFrame, date_col: str, duration_col: str) -> List[Dict]:
    """Calculate absence pattern by month"""
    results = []
    
    try:
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['_month'] = df['_date'].dt.month
        
        total_absences = len(df)
        
        for i, month in enumerate(MONTHS, 1):
            month_df = df[df['_month'] == i]
            count = len(month_df)
            rate = (count / total_absences * 100) if total_absences > 0 else 0
            
            results.append({
                'period': month,
                'absence_rate': float(rate),
                'count': count,
            })
    except Exception:
        for month in MONTHS:
            results.append({'period': month, 'absence_rate': 0, 'count': 0})
    
    return results


def calculate_seasonal_pattern(df: pd.DataFrame, date_col: str, duration_col: str) -> List[Dict]:
    """Calculate absence pattern by season"""
    results = []
    
    try:
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['_month'] = df['_date'].dt.month
        df['_season'] = df['_month'].apply(get_season)
        
        total_absences = len(df)
        
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_df = df[df['_season'] == season]
            count = len(season_df)
            rate = (count / total_absences * 100) if total_absences > 0 else 0
            
            results.append({
                'period': season,
                'absence_rate': float(rate),
                'count': count,
            })
    except Exception:
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            results.append({'period': season, 'absence_rate': 0, 'count': 0})
    
    return results


def identify_high_risk_employees(df: pd.DataFrame, employee_col: str, duration_col: str,
                                  dept_col: Optional[str]) -> List[Dict]:
    """Identify high-risk employees using Bradford Factor"""
    results = []
    
    employee_stats = df.groupby(employee_col).agg({
        duration_col: ['count', 'sum']
    }).reset_index()
    employee_stats.columns = ['employee_id', 'spells', 'total_days']
    
    employee_stats['bradford_factor'] = employee_stats.apply(
        lambda row: calculate_bradford_factor(row['spells'], row['total_days']), axis=1
    )
    
    # Add department if available
    if dept_col and dept_col in df.columns:
        dept_map = df.groupby(employee_col)[dept_col].first().to_dict()
        employee_stats['department'] = employee_stats['employee_id'].map(dept_map)
    else:
        employee_stats['department'] = 'Unknown'
    
    # Sort by Bradford Factor
    employee_stats = employee_stats.sort_values('bradford_factor', ascending=False)
    
    total_employees = len(employee_stats)
    
    for _, row in employee_stats.head(20).iterrows():
        # Calculate trend (simplified - would need historical data)
        trend = 'stable'
        if row['spells'] > 5:
            trend = 'increasing'
        elif row['spells'] <= 2:
            trend = 'decreasing'
        
        # Absence rate (assuming 250 work days)
        absence_rate = (row['total_days'] / 250) * 100
        
        results.append({
            'employee_id': str(row['employee_id']),
            'department': str(row['department']),
            'absence_count': int(row['spells']),
            'total_days': int(row['total_days']),
            'absence_rate': float(absence_rate),
            'risk_level': get_risk_level(row['bradford_factor']),
            'trend': trend,
        })
    
    return results


# ============ VISUALIZATION ============
def create_department_comparison_chart(department_stats: List[Dict]) -> str:
    """Create department comparison bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    depts = [d['department'][:15] for d in department_stats[:10]]
    rates = [d['absence_rate'] for d in department_stats[:10]]
    
    colors = ['#ef4444' if r > 5 else '#f59e0b' if r > 3 else '#22c55e' for r in rates]
    
    bars = ax.bar(depts, rates, color=colors, edgecolor='white')
    
    ax.axhline(y=4, color='#f59e0b', linestyle='--', alpha=0.7, label='Benchmark (4%)')
    
    ax.set_xlabel('Department', fontsize=11)
    ax.set_ylabel('Absence Rate (%)', fontsize=11)
    ax.set_title('Absence Rate by Department', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_reason_pie_chart(reason_breakdown: List[Dict]) -> str:
    """Create pie chart of absence reasons"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    reasons = [r['reason'] for r in reason_breakdown[:8]]
    counts = [r['count'] for r in reason_breakdown[:8]]
    
    colors = ['#ef4444', '#3b82f6', '#8b5cf6', '#f97316', '#22c55e', '#06b6d4', '#64748b', '#94a3b8']
    
    ax.pie(counts, labels=reasons, autopct='%1.1f%%', colors=colors[:len(reasons)], startangle=90)
    ax.set_title('Absence Reasons Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_day_of_week_chart(day_pattern: List[Dict]) -> str:
    """Create day of week bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = [d['period'][:3] for d in day_pattern]
    rates = [d['absence_rate'] for d in day_pattern]
    
    # Highlight Monday and Friday
    colors = ['#f59e0b' if d in ['Mon', 'Fri'] else '#3b82f6' for d in days]
    
    ax.bar(days, rates, color=colors, edgecolor='white')
    
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Absence Rate (%)', fontsize=11)
    ax.set_title('Absence Pattern by Day of Week', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_monthly_trend_chart(monthly_pattern: List[Dict]) -> str:
    """Create monthly trend line chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = [m['period'] for m in monthly_pattern]
    rates = [m['absence_rate'] for m in monthly_pattern]
    
    ax.plot(months, rates, marker='o', linewidth=2, color='#3b82f6', markersize=8)
    ax.fill_between(months, rates, alpha=0.2, color='#3b82f6')
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Absence Rate (%)', fontsize=11)
    ax.set_title('Monthly Absence Trend', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_bradford_distribution_chart(high_risk: List[Dict]) -> str:
    """Create Bradford Factor distribution chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    risk_levels = ['Low', 'Medium', 'High', 'Critical']
    counts = [
        len([e for e in high_risk if e['risk_level'] == level])
        for level in risk_levels
    ]
    
    colors = ['#22c55e', '#f59e0b', '#f97316', '#ef4444']
    
    ax.bar(risk_levels, counts, color=colors, edgecolor='white')
    
    ax.set_xlabel('Risk Level', fontsize=11)
    ax.set_ylabel('Number of Employees', fontsize=11)
    ax.set_title('Bradford Factor Risk Distribution', fontsize=14, fontweight='bold')
    
    for i, (level, count) in enumerate(zip(risk_levels, counts)):
        ax.text(i, count + 0.5, str(count), ha='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(summary: Dict, department_stats: List[Dict], 
                          day_pattern: List[Dict], reason_breakdown: List[Dict]) -> List[Dict]:
    """Generate key insights from absenteeism analysis"""
    insights = []
    
    # Overall absence rate
    rate = summary['overall_absence_rate']
    if rate <= 3:
        insights.append({
            'title': f"Healthy Absence Rate ({rate:.1f}%)",
            'description': "Overall absence rate is within industry benchmark (2-4%).",
            'status': 'positive'
        })
    elif rate <= 5:
        insights.append({
            'title': f"Moderate Absence Rate ({rate:.1f}%)",
            'description': "Absence rate is slightly above benchmark. Monitor trends.",
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f"High Absence Rate ({rate:.1f}%)",
            'description': "Absence rate exceeds benchmark. Investigation recommended.",
            'status': 'warning'
        })
    
    # Cost impact
    cost = summary['estimated_cost']
    insights.append({
        'title': f"Cost Impact: ${cost:,.0f}",
        'description': f"Estimated cost of {summary['total_days_lost']} days lost productivity.",
        'status': 'warning' if cost > 100000 else 'neutral'
    })
    
    # Department with highest absence
    if department_stats:
        worst_dept = department_stats[0]
        if worst_dept['absence_rate'] > 5:
            insights.append({
                'title': f"Focus Area: {worst_dept['department']}",
                'description': f"Highest absence rate at {worst_dept['absence_rate']:.1f}%. Intervention recommended.",
                'status': 'warning'
            })
    
    # Monday/Friday pattern
    if day_pattern:
        mon_rate = next((d['absence_rate'] for d in day_pattern if d['period'] == 'Monday'), 0)
        fri_rate = next((d['absence_rate'] for d in day_pattern if d['period'] == 'Friday'), 0)
        mid_rate = np.mean([d['absence_rate'] for d in day_pattern if d['period'] in ['Tuesday', 'Wednesday', 'Thursday']])
        
        if mon_rate > mid_rate * 1.3 or fri_rate > mid_rate * 1.3:
            insights.append({
                'title': "Monday/Friday Pattern Detected",
                'description': "Higher absences on Mondays/Fridays may indicate 'long weekend' behavior.",
                'status': 'warning'
            })
    
    # Top reason
    if reason_breakdown:
        top_reason = reason_breakdown[0]
        insights.append({
            'title': f"Top Reason: {top_reason['reason']}",
            'description': f"Accounts for {top_reason['pct']*100:.0f}% of all absences.",
            'status': 'neutral'
        })
    
    return insights


@router.post("/absenteeism")
async def run_absenteeism_analysis(request: AbsenteeismRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        for col in [request.employee_col, request.date_col, request.duration_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Convert duration to numeric
        df[request.duration_col] = pd.to_numeric(df[request.duration_col], errors='coerce').fillna(1)
        
        # Calculate summary statistics
        total_employees = df[request.employee_col].nunique()
        total_absences = len(df)
        total_days_lost = int(df[request.duration_col].sum())
        avg_duration = float(df[request.duration_col].mean())
        
        # Absence rate (assuming 250 work days per year)
        overall_absence_rate = (total_days_lost / (total_employees * 250)) * 100 if total_employees > 0 else 0
        
        # Bradford Factor average
        emp_stats = df.groupby(request.employee_col)[request.duration_col].agg(['count', 'sum'])
        emp_stats['bradford'] = emp_stats.apply(lambda r: calculate_bradford_factor(r['count'], r['sum']), axis=1)
        bradford_avg = float(emp_stats['bradford'].mean())
        
        summary_data = {
            'total_employees': total_employees,
            'total_absences': total_absences,
            'total_days_lost': total_days_lost,
            'overall_absence_rate': overall_absence_rate,
            'avg_absence_duration': avg_duration,
            'estimated_cost': total_days_lost * request.cost_per_day,
            'bradford_factor_avg': bradford_avg,
        }
        
        # Department statistics
        dept_stats = calculate_department_stats(df, request.dept_col, request.duration_col, 
                                                 request.employee_col, request.cost_per_day)
        
        # Reason breakdown
        reason_breakdown = calculate_reason_breakdown(df, request.reason_col, request.duration_col)
        
        # Time patterns
        day_pattern = calculate_day_of_week_pattern(df, request.date_col, request.duration_col)
        monthly_pattern = calculate_monthly_pattern(df, request.date_col, request.duration_col)
        seasonal_pattern = calculate_seasonal_pattern(df, request.date_col, request.duration_col)
        
        # High-risk employees
        high_risk = identify_high_risk_employees(df, request.employee_col, request.duration_col, request.dept_col)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {}
        if dept_stats:
            visualizations['department_comparison'] = create_department_comparison_chart(dept_stats)
        if reason_breakdown:
            visualizations['reason_pie'] = create_reason_pie_chart(reason_breakdown)
        visualizations['day_of_week'] = create_day_of_week_chart(day_pattern)
        visualizations['monthly_trend'] = create_monthly_trend_chart(monthly_pattern)
        if high_risk:
            visualizations['bradford_distribution'] = create_bradford_distribution_chart(high_risk)
        
        # Key insights
        key_insights = generate_key_insights(summary_data, dept_stats, day_pattern, reason_breakdown)
        
        # Find highest absence department and peak day
        highest_dept = dept_stats[0]['department'] if dept_stats else 'N/A'
        peak_day = max(day_pattern, key=lambda x: x['absence_rate'])['period'] if day_pattern else 'N/A'
        top_reason = reason_breakdown[0]['reason'] if reason_breakdown else 'N/A'
        
        # Determine analysis period
        try:
            dates = pd.to_datetime(df[request.date_col], errors='coerce').dropna()
            period = f"{dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}"
        except:
            period = "Unknown"
        
        results = {
            'summary': {k: _to_native_type(v) for k, v in summary_data.items()},
            'department_stats': [{k: _to_native_type(v) for k, v in d.items()} for d in dept_stats],
            'reason_breakdown': [{k: _to_native_type(v) for k, v in r.items()} for r in reason_breakdown],
            'day_of_week_pattern': [{k: _to_native_type(v) for k, v in d.items()} for d in day_pattern],
            'monthly_pattern': [{k: _to_native_type(v) for k, v in m.items()} for m in monthly_pattern],
            'seasonal_pattern': [{k: _to_native_type(v) for k, v in s.items()} for s in seasonal_pattern],
            'high_risk_employees': [{k: _to_native_type(v) for k, v in e.items()} for e in high_risk],
            'trends': [],
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'analysis_period': period,
                'highest_absence_dept': highest_dept,
                'top_reason': top_reason,
                'peak_day': peak_day,
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Absenteeism analysis failed: {str(e)}")
