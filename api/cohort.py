"""
Cohort Analysis Router for FastAPI
Customer retention, revenue cohorts, behavioral analysis
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
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CohortRequest(BaseModel):
    data: List[Dict[str, Any]]
    user_col: str  # User/customer identifier
    date_col: str  # Transaction/event date
    value_col: Optional[str] = None  # Revenue/value column (optional)
    event_col: Optional[str] = None  # Event type column (optional)
    cohort_period: Literal["day", "week", "month", "quarter", "year"] = "month"
    analysis_type: Literal["retention", "revenue", "cumulative", "average"] = "retention"
    max_periods: int = 12  # Maximum periods to analyze


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if obj is None:
        return None
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
    if isinstance(obj, (pd.Timestamp, datetime)):
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


def get_cohort_period(date: pd.Timestamp, period: str) -> str:
    """Get cohort period string from date"""
    if period == "day":
        return date.strftime("%Y-%m-%d")
    elif period == "week":
        return date.strftime("%Y-W%W")
    elif period == "month":
        return date.strftime("%Y-%m")
    elif period == "quarter":
        return f"{date.year}-Q{(date.month - 1) // 3 + 1}"
    elif period == "year":
        return str(date.year)
    return date.strftime("%Y-%m")


def calculate_period_number(cohort_date: pd.Timestamp, current_date: pd.Timestamp, 
                            period: str) -> int:
    """Calculate the period number from cohort date"""
    if period == "day":
        return (current_date - cohort_date).days
    elif period == "week":
        return (current_date - cohort_date).days // 7
    elif period == "month":
        return (current_date.year - cohort_date.year) * 12 + (current_date.month - cohort_date.month)
    elif period == "quarter":
        cohort_q = (cohort_date.year * 4) + (cohort_date.month - 1) // 3
        current_q = (current_date.year * 4) + (current_date.month - 1) // 3
        return current_q - cohort_q
    elif period == "year":
        return current_date.year - cohort_date.year
    return 0


def build_cohort_data(df: pd.DataFrame, user_col: str, date_col: str,
                      value_col: Optional[str], cohort_period: str,
                      max_periods: int) -> Dict[str, Any]:
    """Build cohort analysis data"""
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get first transaction date for each user (cohort assignment)
    user_cohorts = df.groupby(user_col)[date_col].min().reset_index()
    user_cohorts.columns = [user_col, 'cohort_date']
    
    # Merge cohort info back to main data
    df = df.merge(user_cohorts, on=user_col)
    
    # Create cohort period labels
    df['cohort'] = df['cohort_date'].apply(lambda x: get_cohort_period(x, cohort_period))
    df['transaction_period'] = df[date_col].apply(lambda x: get_cohort_period(x, cohort_period))
    
    # Calculate period number
    df['period_number'] = df.apply(
        lambda row: calculate_period_number(row['cohort_date'], row[date_col], cohort_period),
        axis=1
    )
    
    # Filter to max periods
    df = df[df['period_number'] <= max_periods]
    
    # Get unique cohorts sorted
    cohorts = sorted(df['cohort'].unique())
    
    # Build cohort metrics
    cohort_data = {}
    
    for cohort in cohorts:
        cohort_df = df[df['cohort'] == cohort]
        cohort_users = cohort_df[user_col].nunique()
        
        cohort_data[cohort] = {
            'cohort_size': cohort_users,
            'periods': {}
        }
        
        for period in range(max_periods + 1):
            period_df = cohort_df[cohort_df['period_number'] == period]
            
            if len(period_df) > 0:
                active_users = period_df[user_col].nunique()
                retention_rate = active_users / cohort_users if cohort_users > 0 else 0
                
                period_data = {
                    'active_users': active_users,
                    'retention_rate': retention_rate,
                    'transactions': len(period_df)
                }
                
                if value_col and value_col in period_df.columns:
                    total_value = period_df[value_col].sum()
                    avg_value = period_df[value_col].mean()
                    period_data['total_value'] = _to_native_type(total_value)
                    period_data['avg_value'] = _to_native_type(avg_value)
                    period_data['value_per_user'] = _to_native_type(total_value / active_users) if active_users > 0 else 0
                
                cohort_data[cohort]['periods'][period] = period_data
    
    return cohort_data


def build_retention_matrix(cohort_data: Dict, max_periods: int) -> pd.DataFrame:
    """Build retention rate matrix"""
    cohorts = sorted(cohort_data.keys())
    
    matrix_data = []
    for cohort in cohorts:
        row = {'cohort': cohort, 'cohort_size': cohort_data[cohort]['cohort_size']}
        for period in range(max_periods + 1):
            if period in cohort_data[cohort]['periods']:
                row[f'period_{period}'] = cohort_data[cohort]['periods'][period]['retention_rate']
            else:
                row[f'period_{period}'] = None
        matrix_data.append(row)
    
    return pd.DataFrame(matrix_data)


def build_revenue_matrix(cohort_data: Dict, max_periods: int, 
                         metric: str = "total_value") -> pd.DataFrame:
    """Build revenue/value matrix"""
    cohorts = sorted(cohort_data.keys())
    
    matrix_data = []
    for cohort in cohorts:
        row = {'cohort': cohort, 'cohort_size': cohort_data[cohort]['cohort_size']}
        cumulative = 0
        for period in range(max_periods + 1):
            if period in cohort_data[cohort]['periods']:
                value = cohort_data[cohort]['periods'][period].get(metric, 0)
                if value is not None:
                    cumulative += value
                    row[f'period_{period}'] = value if metric != 'cumulative' else cumulative
                else:
                    row[f'period_{period}'] = None
            else:
                row[f'period_{period}'] = None
        matrix_data.append(row)
    
    return pd.DataFrame(matrix_data)


def calculate_cohort_metrics(cohort_data: Dict, max_periods: int) -> Dict[str, Any]:
    """Calculate aggregate cohort metrics"""
    
    cohorts = list(cohort_data.keys())
    total_users = sum(c['cohort_size'] for c in cohort_data.values())
    
    # Average retention by period
    avg_retention = {}
    for period in range(max_periods + 1):
        rates = []
        for cohort in cohorts:
            if period in cohort_data[cohort]['periods']:
                rates.append(cohort_data[cohort]['periods'][period]['retention_rate'])
        avg_retention[period] = np.mean(rates) if rates else None
    
    # Churn analysis
    period_1_retention = avg_retention.get(1)
    period_3_retention = avg_retention.get(3)
    period_6_retention = avg_retention.get(6)
    period_12_retention = avg_retention.get(12)
    
    # Customer Lifetime calculation (simplified)
    retention_rates = [r for r in avg_retention.values() if r is not None]
    if retention_rates and len(retention_rates) > 1:
        # Estimate using average retention (excluding period 0)
        non_zero_rates = [avg_retention.get(p) for p in range(1, max_periods + 1) 
                         if avg_retention.get(p) is not None]
        if non_zero_rates:
            avg_ret = np.mean(non_zero_rates)
            estimated_lifetime = 1 / (1 - avg_ret) if avg_ret < 1 else float('inf')
        else:
            estimated_lifetime = None
    else:
        estimated_lifetime = None
    
    # Best and worst cohorts
    cohort_performance = []
    for cohort in cohorts:
        periods = cohort_data[cohort]['periods']
        if periods:
            avg_ret = np.mean([p['retention_rate'] for p in periods.values()])
            cohort_performance.append((cohort, avg_ret, cohort_data[cohort]['cohort_size']))
    
    cohort_performance.sort(key=lambda x: x[1], reverse=True)
    best_cohort = cohort_performance[0] if cohort_performance else None
    worst_cohort = cohort_performance[-1] if cohort_performance else None
    
    return {
        'total_cohorts': len(cohorts),
        'total_users': total_users,
        'avg_cohort_size': total_users / len(cohorts) if cohorts else 0,
        'avg_retention_by_period': {k: _to_native_type(v) for k, v in avg_retention.items()},
        'period_1_retention': _to_native_type(period_1_retention),
        'period_3_retention': _to_native_type(period_3_retention),
        'period_6_retention': _to_native_type(period_6_retention),
        'period_12_retention': _to_native_type(period_12_retention),
        'estimated_lifetime_periods': _to_native_type(estimated_lifetime),
        'best_cohort': {'name': best_cohort[0], 'retention': _to_native_type(best_cohort[1]), 
                        'size': best_cohort[2]} if best_cohort else None,
        'worst_cohort': {'name': worst_cohort[0], 'retention': _to_native_type(worst_cohort[1]),
                         'size': worst_cohort[2]} if worst_cohort else None
    }


def create_retention_heatmap(retention_matrix: pd.DataFrame, max_periods: int) -> str:
    """Create retention heatmap visualization"""
    fig, ax = plt.subplots(figsize=(14, max(8, len(retention_matrix) * 0.5)))
    
    # Prepare data for heatmap
    cohorts = retention_matrix['cohort'].values
    period_cols = [f'period_{i}' for i in range(max_periods + 1)]
    heatmap_data = retention_matrix[period_cols].values.astype(float) * 100  # Convert to percentage
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu',
                xticklabels=[f'P{i}' for i in range(max_periods + 1)],
                yticklabels=cohorts, ax=ax, cbar_kws={'label': 'Retention %'},
                vmin=0, vmax=100, mask=pd.isna(heatmap_data))
    
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Cohort', fontsize=12)
    ax.set_title('Cohort Retention Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_retention_curves(cohort_data: Dict, max_periods: int) -> str:
    """Create retention curves visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cohorts = sorted(cohort_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(cohorts)))
    
    for cohort, color in zip(cohorts, colors):
        periods = []
        retention = []
        for p in range(max_periods + 1):
            if p in cohort_data[cohort]['periods']:
                periods.append(p)
                retention.append(cohort_data[cohort]['periods'][p]['retention_rate'] * 100)
        
        if periods:
            ax.plot(periods, retention, '-o', color=color, label=cohort,
                    linewidth=2, markersize=4, alpha=0.7)
    
    # Average line
    avg_retention = []
    for p in range(max_periods + 1):
        rates = []
        for cohort in cohorts:
            if p in cohort_data[cohort]['periods']:
                rates.append(cohort_data[cohort]['periods'][p]['retention_rate'] * 100)
        if rates:
            avg_retention.append((p, np.mean(rates)))
    
    if avg_retention:
        periods, rates = zip(*avg_retention)
        ax.plot(periods, rates, 'k-', linewidth=3, label='Average', zorder=10)
    
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Retention Rate (%)', fontsize=12)
    ax.set_title('Cohort Retention Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cohort_size_chart(cohort_data: Dict) -> str:
    """Create cohort size bar chart"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    cohorts = sorted(cohort_data.keys())
    sizes = [cohort_data[c]['cohort_size'] for c in cohorts]
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(cohorts)))
    bars = ax.bar(cohorts, sizes, color=colors, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Cohort', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title('Cohort Sizes Over Time', fontsize=14, fontweight='bold')
    ax.set_xticklabels(cohorts, rotation=45, ha='right')
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                f'{size:,}', ha='center', va='bottom', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_average_retention_chart(metrics: Dict) -> str:
    """Create average retention by period chart"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    avg_retention = metrics.get('avg_retention_by_period', {})
    
    # Filter out None values and ensure proper conversion
    valid_data = []
    for k, v in avg_retention.items():
        if v is not None:
            try:
                period = int(k)
                rate = float(v) * 100
                valid_data.append((period, rate))
            except (ValueError, TypeError):
                continue
    
    if not valid_data:
        ax.text(0.5, 0.5, 'No retention data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        plt.tight_layout()
        return _fig_to_base64(fig)
    
    # Sort by period
    valid_data.sort(key=lambda x: x[0])
    periods = [d[0] for d in valid_data]
    rates = [d[1] for d in valid_data]
    
    # Bar chart
    colors = ['#22c55e' if r >= 50 else '#3b82f6' if r >= 30 else '#f59e0b' if r >= 15 else '#ef4444' for r in rates]
    bars = ax.bar(periods, rates, color=colors, edgecolor='white', linewidth=2)
    
    # Trend line
    if len(periods) > 1:
        z = np.polyfit(periods, rates, 2)
        p = np.poly1d(z)
        x_line = np.linspace(min(periods), max(periods), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Trend')
    
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Average Retention Rate (%)', fontsize=12)
    ax.set_title('Average Retention by Period', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_revenue_heatmap(revenue_matrix: pd.DataFrame, max_periods: int) -> str:
    """Create revenue heatmap visualization"""
    fig, ax = plt.subplots(figsize=(14, max(8, len(revenue_matrix) * 0.5)))
    
    cohorts = revenue_matrix['cohort'].values
    period_cols = [f'period_{i}' for i in range(max_periods + 1)]
    heatmap_data = revenue_matrix[period_cols].values.astype(float)
    
    # Format numbers for display
    def format_value(x):
        if pd.isna(x):
            return ''
        if x >= 1000000:
            return f'{x/1000000:.1f}M'
        elif x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x:.0f}'
    
    annot = np.vectorize(format_value)(heatmap_data)
    
    sns.heatmap(heatmap_data, annot=annot, fmt='', cmap='YlOrRd',
                xticklabels=[f'P{i}' for i in range(max_periods + 1)],
                yticklabels=cohorts, ax=ax, cbar_kws={'label': 'Revenue'},
                mask=pd.isna(heatmap_data))
    
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Cohort', fontsize=12)
    ax.set_title('Cohort Revenue Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(metrics: Dict, cohort_data: Dict) -> List[Dict[str, Any]]:
    """Generate key insights for cohort analysis"""
    insights = []
    
    # Overall retention assessment
    p1_ret = metrics.get('period_1_retention')
    if p1_ret is not None:
        if p1_ret >= 0.7:
            insights.append({
                'title': 'Strong Initial Retention',
                'description': f'Period 1 retention is {p1_ret*100:.1f}%. Users show high engagement after acquisition.',
                'status': 'positive'
            })
        elif p1_ret >= 0.4:
            insights.append({
                'title': 'Moderate Initial Retention',
                'description': f'Period 1 retention is {p1_ret*100:.1f}%. Room for improvement in early engagement.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Low Initial Retention',
                'description': f'Period 1 retention is {p1_ret*100:.1f}%. Focus on improving onboarding experience.',
                'status': 'warning'
            })
    
    # Long-term retention
    p6_ret = metrics.get('period_6_retention')
    if p6_ret is not None:
        if p6_ret >= 0.3:
            insights.append({
                'title': 'Good Long-term Retention',
                'description': f'Period 6 retention is {p6_ret*100:.1f}%. Strong user loyalty.',
                'status': 'positive'
            })
        elif p6_ret >= 0.15:
            insights.append({
                'title': 'Moderate Long-term Retention',
                'description': f'Period 6 retention is {p6_ret*100:.1f}%. Consider loyalty programs.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Low Long-term Retention',
                'description': f'Period 6 retention is {p6_ret*100:.1f}%. Implement re-engagement strategies.',
                'status': 'warning'
            })
    
    # Cohort comparison
    best = metrics.get('best_cohort')
    worst = metrics.get('worst_cohort')
    if best and worst and best['name'] != worst['name']:
        best_ret = best.get('retention')
        worst_ret = worst.get('retention')
        if best_ret is not None and worst_ret is not None:
            diff = (best_ret - worst_ret) * 100
            insights.append({
                'title': f'Best Cohort: {best["name"]}',
                'description': f'Avg retention: {best_ret*100:.1f}%. {diff:.1f}pp better than worst cohort ({worst["name"]}).',
                'status': 'neutral'
            })
    
    # Lifetime estimate
    lifetime = metrics.get('estimated_lifetime_periods')
    if lifetime is not None and lifetime < float('inf'):
        insights.append({
            'title': f'Estimated Customer Lifetime',
            'description': f'Approximately {lifetime:.1f} periods based on retention rates.',
            'status': 'neutral'
        })
    
    # Cohort size trend
    cohorts = sorted(cohort_data.keys())
    if len(cohorts) >= 3:
        sizes = [cohort_data[c]['cohort_size'] for c in cohorts]
        recent_avg = np.mean(sizes[-3:])
        earlier_avg = np.mean(sizes[:3])
        if earlier_avg > 0:
            if recent_avg > earlier_avg * 1.2:
                insights.append({
                    'title': 'Growing User Acquisition',
                    'description': f'Recent cohorts are {((recent_avg/earlier_avg)-1)*100:.0f}% larger than earlier cohorts.',
                    'status': 'positive'
                })
            elif recent_avg < earlier_avg * 0.8:
                insights.append({
                    'title': 'Declining User Acquisition',
                    'description': f'Recent cohorts are {(1-(recent_avg/earlier_avg))*100:.0f}% smaller than earlier cohorts.',
                    'status': 'warning'
                })
    
    return insights


@router.post("/cohort")
async def run_cohort_analysis(request: CohortRequest) -> Dict[str, Any]:
    """
    Perform Cohort Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.user_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"User column '{request.user_col}' not found")
        if request.date_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{request.date_col}' not found")
        
        # Build cohort data
        cohort_data = build_cohort_data(
            df, request.user_col, request.date_col,
            request.value_col, request.cohort_period, request.max_periods
        )
        
        if not cohort_data:
            raise HTTPException(status_code=400, detail="No cohort data could be generated")
        
        # Build matrices
        retention_matrix = build_retention_matrix(cohort_data, request.max_periods)
        
        revenue_matrix = None
        if request.value_col:
            if request.analysis_type == "cumulative":
                revenue_matrix = build_revenue_matrix(cohort_data, request.max_periods, "cumulative")
            elif request.analysis_type == "average":
                revenue_matrix = build_revenue_matrix(cohort_data, request.max_periods, "avg_value")
            else:
                revenue_matrix = build_revenue_matrix(cohort_data, request.max_periods, "total_value")
        
        # Calculate metrics
        metrics = calculate_cohort_metrics(cohort_data, request.max_periods)
        
        # Create visualizations
        visualizations = {}
        visualizations['retention_heatmap'] = create_retention_heatmap(retention_matrix, request.max_periods)
        visualizations['retention_curves'] = create_retention_curves(cohort_data, request.max_periods)
        visualizations['cohort_sizes'] = create_cohort_size_chart(cohort_data)
        visualizations['avg_retention'] = create_average_retention_chart(metrics)
        
        if revenue_matrix is not None:
            visualizations['revenue_heatmap'] = create_revenue_heatmap(revenue_matrix, request.max_periods)
        
        # Generate insights
        insights = generate_key_insights(metrics, cohort_data)
        
        # Prepare cohort summary for response
        cohort_summary = []
        for cohort in sorted(cohort_data.keys()):
            data = cohort_data[cohort]
            periods = data['periods']
            p1_ret = periods.get(1, {}).get('retention_rate')
            p3_ret = periods.get(3, {}).get('retention_rate')
            
            cohort_summary.append({
                'cohort': cohort,
                'size': data['cohort_size'],
                'period_1_retention': _to_native_type(p1_ret) if p1_ret else None,
                'period_3_retention': _to_native_type(p3_ret) if p3_ret else None,
                'periods_active': len(periods)
            })
        
        # Convert retention matrix to serializable format
        retention_table = []
        for _, row in retention_matrix.iterrows():
            table_row = {'cohort': row['cohort'], 'size': row['cohort_size']}
            for i in range(request.max_periods + 1):
                val = row.get(f'period_{i}')
                if val is not None and not pd.isna(val):
                    table_row[f'P{i}'] = _to_native_type(float(val) * 100)
                else:
                    table_row[f'P{i}'] = None
            retention_table.append(table_row)
        
        # Summary
        summary = {
            'total_cohorts': metrics['total_cohorts'],
            'total_users': metrics['total_users'],
            'avg_cohort_size': _to_native_type(metrics['avg_cohort_size']),
            'period_1_retention': metrics['period_1_retention'],
            'period_6_retention': metrics['period_6_retention'],
            'cohort_period': request.cohort_period,
            'analysis_type': request.analysis_type
        }
        
        return {
            'success': True,
            'metrics': metrics,
            'cohort_summary': cohort_summary,
            'retention_table': retention_table,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohort analysis failed: {str(e)}")
