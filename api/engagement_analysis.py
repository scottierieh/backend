"""
User Engagement Change Analysis FastAPI Endpoint
Detect and analyze changes in user engagement patterns over time
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from io import BytesIO
import base64
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class EngagementRequest(BaseModel):
    """Request model for Engagement Change Analysis"""
    data: List[Dict[str, Any]]
    user_id_col: str
    date_col: str
    engagement_metric: str = Field(default="event_count")
    custom_metric_col: Optional[str] = None
    period: str = Field(default="weekly", pattern="^(daily|weekly|monthly)$")
    lookback_periods: int = Field(default=12, ge=4, le=52)


def detect_changepoints(series: pd.Series, min_periods: int = 4):
    """Detect significant changepoints in time series"""
    changepoints = []
    
    if len(series) < min_periods * 2:
        return changepoints
    
    # Rolling statistics
    window = max(min_periods, len(series) // 4)
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    
    # Z-score based detection
    z_scores = np.abs((series - rolling_mean) / (rolling_std + 1e-10))
    
    # Find points where z-score exceeds threshold
    threshold = 2.0
    significant = z_scores > threshold
    
    # Group consecutive points
    in_change = False
    start_idx = None
    
    for idx, is_sig in enumerate(significant):
        if is_sig and not in_change:
            in_change = True
            start_idx = idx
        elif not is_sig and in_change:
            in_change = False
            if start_idx is not None:
                mid_idx = (start_idx + idx) // 2
                if mid_idx < len(series):
                    changepoints.append({
                        'index': mid_idx,
                        'date': series.index[mid_idx],
                        'value': float(series.iloc[mid_idx]),
                        'z_score': float(z_scores.iloc[mid_idx])
                    })
    
    return changepoints


def calculate_growth_rate(series: pd.Series):
    """Calculate period-over-period growth rates"""
    growth_rates = series.pct_change() * 100
    return growth_rates


def segment_users_by_engagement(df: pd.DataFrame, user_id_col: str, date_col: str):
    """Segment users into engagement tiers"""
    
    # Make a copy
    df = df.copy()
    
    # Calculate total engagement per user (count events)
    user_engagement = df.groupby(user_id_col).size().reset_index(name='total_engagement')
    
    # Define quartiles
    q25 = user_engagement['total_engagement'].quantile(0.25)
    q50 = user_engagement['total_engagement'].quantile(0.50)
    q75 = user_engagement['total_engagement'].quantile(0.75)
    
    def assign_segment(value):
        if value >= q75:
            return 'Power Users'
        elif value >= q50:
            return 'Regular Users'
        elif value >= q25:
            return 'Casual Users'
        else:
            return 'Low Engagement'
    
    user_engagement['segment'] = user_engagement['total_engagement'].apply(assign_segment)
    
    # Merge back
    df = df.merge(user_engagement[[user_id_col, 'segment']], on=user_id_col, how='left')
    
    return df, user_engagement


def calculate_engagement_metrics(df: pd.DataFrame, user_id_col: str, date_col: str, 
                                 engagement_metric: str, period: str = 'weekly',
                                 lookback_periods: int = 12):
    """Calculate engagement metrics over time"""
    
    # Make a copy
    df = df.copy()
    
    # Convert date to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Group by period
    if period == 'daily':
        df['period'] = df[date_col].dt.date
    elif period == 'weekly':
        df['period'] = df[date_col].dt.to_period('W').astype(str)
    else:  # monthly
        df['period'] = df[date_col].dt.to_period('M').astype(str)
    
    # Calculate metrics based on engagement_metric type
    if engagement_metric == 'event_count':
        # Count events per period
        period_metrics = df.groupby('period').size().reset_index(name='metric_value')
    elif engagement_metric == 'active_users':
        # Count unique active users per period
        period_metrics = df.groupby('period')[user_id_col].nunique().reset_index(name='metric_value')
    elif engagement_metric == 'events_per_user':
        # Average events per user
        events = df.groupby('period').size()
        users = df.groupby('period')[user_id_col].nunique()
        period_metrics = pd.DataFrame({
            'period': events.index,
            'metric_value': (events / users).values
        })
    elif engagement_metric == 'session_duration':
        # This would require session data - placeholder
        period_metrics = df.groupby('period').size().reset_index(name='metric_value')
    else:
        # Custom metric from column
        period_metrics = df.groupby('period').size().reset_index(name='metric_value')
    
    # Sort by period
    period_metrics = period_metrics.sort_values('period')
    
    # Take last N periods
    period_metrics = period_metrics.tail(lookback_periods)
    
    return period_metrics


def analyze_trends(period_metrics: pd.DataFrame):
    """Analyze engagement trends"""
    
    if len(period_metrics) < 2:
        return {}
    
    values = period_metrics['metric_value'].values
    periods = np.arange(len(values))
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(periods, values)
    
    # Overall trend
    trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
    trend_strength = abs(r_value)
    
    # Recent vs historical comparison
    split_point = len(values) // 2
    recent_avg = np.mean(values[split_point:])
    historical_avg = np.mean(values[:split_point])
    
    change_pct = ((recent_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
    
    # Volatility
    volatility = np.std(values) / np.mean(values) * 100 if np.mean(values) > 0 else 0
    
    # Growth rate
    growth_rates = pd.Series(values).pct_change() * 100
    avg_growth_rate = growth_rates.mean()
    
    return {
        'trend_direction': trend_direction,
        'trend_strength': float(trend_strength),
        'slope': float(slope),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'recent_avg': float(recent_avg),
        'historical_avg': float(historical_avg),
        'change_percentage': float(change_pct),
        'volatility': float(volatility),
        'avg_growth_rate': float(avg_growth_rate),
        'current_value': float(values[-1]),
        'peak_value': float(np.max(values)),
        'trough_value': float(np.min(values))
    }


def generate_visualizations(period_metrics: pd.DataFrame, engagement_metric: str,
                           changepoints: list, segment_data: Optional[pd.DataFrame] = None):
    """Generate engagement visualizations"""
    visualizations = {}
    
    # 1. Engagement Trend Line
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(period_metrics))
    y = period_metrics['metric_value'].values
    
    ax.plot(x, y, marker='o', linewidth=2, markersize=6, color='#4A90E2', label='Actual')
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "--", color='#E74C3C', linewidth=2, alpha=0.7, label='Trend')
    
    # Mark changepoints
    for cp in changepoints:
        idx = cp['index']
        if idx < len(period_metrics):
            ax.axvline(x=idx, color='orange', linestyle='--', alpha=0.5)
            ax.scatter([idx], [y[idx]], color='orange', s=100, zorder=5, marker='v')
    
    ax.set_title(f'Engagement Trend - {engagement_metric.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Engagement Metric', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set x-axis labels
    labels = [p.split('-')[-1] if '-' in str(p) else str(p) for p in period_metrics['period'].values]
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels(labels[::max(1, len(labels)//10)], rotation=45, ha='right')
    
    plt.tight_layout()
    visualizations['trend_line'] = fig_to_base64(fig)
    
    # 2. Growth Rate Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    growth_rates = calculate_growth_rate(period_metrics['metric_value'])
    colors = ['green' if x > 0 else 'red' for x in growth_rates]
    
    ax.bar(x[1:], growth_rates[1:], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_title('Period-over-Period Growth Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Growth Rate (%)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_xticks(x[1::max(1, len(x)//10)])
    ax.set_xticklabels(labels[1::max(1, len(labels)//10)], rotation=45, ha='right')
    
    plt.tight_layout()
    visualizations['growth_rate'] = fig_to_base64(fig)
    
    # 3. Moving Average
    fig, ax = plt.subplots(figsize=(12, 6))
    
    window = min(4, len(period_metrics) // 3)
    if window >= 2:
        ma = period_metrics['metric_value'].rolling(window=window).mean()
        
        ax.plot(x, y, marker='o', linewidth=1, markersize=4, alpha=0.5, 
               color='#95A5A6', label='Actual')
        ax.plot(x, ma, linewidth=3, color='#4A90E2', label=f'{window}-Period MA')
        
        ax.set_title('Engagement with Moving Average', fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=11)
        ax.set_ylabel('Engagement Metric', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels(labels[::max(1, len(labels)//10)], rotation=45, ha='right')
        
        plt.tight_layout()
        visualizations['moving_average'] = fig_to_base64(fig)
    
    # 4. Segment Breakdown (if available)
    if segment_data is not None and len(segment_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        segment_counts = segment_data['segment'].value_counts()
        colors_map = {
            'Power Users': '#2ECC71',
            'Regular Users': '#3498DB',
            'Casual Users': '#F39C12',
            'Low Engagement': '#E74C3C'
        }
        colors = [colors_map.get(s, '#95A5A6') for s in segment_counts.index]
        
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
              colors=colors, startangle=90)
        ax.set_title('User Engagement Segmentation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        visualizations['segmentation'] = fig_to_base64(fig)
    
    # 5. Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(y, bins=min(15, len(y)), color='#4A90E2', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(y), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y):.1f}')
    ax.axvline(np.median(y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(y):.1f}')
    
    ax.set_title('Engagement Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Engagement Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    visualizations['distribution'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(trend_analysis: dict, changepoints: list):
    """Generate key insights"""
    insights = []
    
    # Trend insight
    if trend_analysis['trend_direction'] == 'increasing':
        if trend_analysis['change_percentage'] > 20:
            insights.append({
                'title': 'Strong Growth Trend',
                'description': f"Engagement has increased by {trend_analysis['change_percentage']:.1f}% with consistent upward momentum. Recent performance significantly outpaces historical average.",
                'status': 'positive'
            })
        elif trend_analysis['change_percentage'] > 0:
            insights.append({
                'title': 'Moderate Growth',
                'description': f"Engagement is growing at {trend_analysis['change_percentage']:.1f}% with positive trend direction (R²={trend_analysis['r_squared']:.2f}).",
                'status': 'positive'
            })
    elif trend_analysis['trend_direction'] == 'decreasing':
        if trend_analysis['change_percentage'] < -20:
            insights.append({
                'title': 'Significant Decline',
                'description': f"Engagement has dropped {abs(trend_analysis['change_percentage']):.1f}%. Recent average is substantially below historical levels. Immediate investigation needed.",
                'status': 'warning'
            })
        else:
            insights.append({
                'title': 'Declining Engagement',
                'description': f"Engagement is down {abs(trend_analysis['change_percentage']):.1f}% from historical average. Monitor closely and consider retention initiatives.",
                'status': 'warning'
            })
    else:
        insights.append({
            'title': 'Stable Engagement',
            'description': f"Engagement remains relatively stable with {trend_analysis['change_percentage']:.1f}% change. Focus on optimization rather than recovery.",
            'status': 'neutral'
        })
    
    # Volatility insight
    if trend_analysis['volatility'] > 30:
        insights.append({
            'title': 'High Volatility',
            'description': f"Engagement shows high volatility ({trend_analysis['volatility']:.1f}% coefficient of variation). Consider stabilizing factors or seasonal patterns.",
            'status': 'neutral'
        })
    
    # Changepoint insights
    if len(changepoints) > 0:
        insights.append({
            'title': f'{len(changepoints)} Significant Change{"s" if len(changepoints) > 1 else ""} Detected',
            'description': f"Identified {len(changepoints)} period(s) with significant engagement shifts. Review product changes, marketing campaigns, or external factors during these times.",
            'status': 'neutral'
        })
    
    return insights


@router.post("/engagement-change-analysis")
async def analyze_engagement_changes(request: EngagementRequest):
    """
    User Engagement Change Analysis Endpoint
    
    Analyzes changes in user engagement patterns over time
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 20:
            raise HTTPException(400, "Insufficient data (need at least 20 records)")
        
        df = pd.DataFrame(request.data)
        
        required_cols = [request.user_id_col, request.date_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns: {missing}")
        
        # Calculate engagement metrics
        period_metrics = calculate_engagement_metrics(
            df, request.user_id_col, request.date_col, 
            request.engagement_metric, request.period, request.lookback_periods
        )
        
        # Analyze trends
        trend_analysis = analyze_trends(period_metrics)
        
        # Detect changepoints
        metric_series = period_metrics.set_index('period')['metric_value']
        changepoints = detect_changepoints(metric_series)
        
        # Segment users
        df_segmented, user_segments = segment_users_by_engagement(
            df, request.user_id_col, request.date_col
        )
        
        # Generate visualizations
        visualizations = generate_visualizations(
            period_metrics, request.engagement_metric, changepoints, user_segments
        )
        
        # Generate insights
        insights = generate_insights(trend_analysis, changepoints)
        
        # Prepare response
        response_data = {
            'success': True,
            'results': {
                'engagement_metric': request.engagement_metric,
                'period': request.period,
                'periods_analyzed': len(period_metrics),
                'trend_analysis': trend_analysis,
                'changepoints': changepoints,
                'segment_distribution': user_segments['segment'].value_counts().to_dict()
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'engagement_change_analysis',
                'total_users': int(df[request.user_id_col].nunique()),
                'total_events': int(len(df)),
                'trend_direction': trend_analysis['trend_direction'],
                'change_percentage': round(trend_analysis['change_percentage'], 1),
                'num_changepoints': len(changepoints)
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
