"""
Cohort Analysis FastAPI Endpoint
Retention, Revenue, and Behavioral Cohort Analysis
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
import json

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class CohortRequest(BaseModel):
    """Request model for Cohort Analysis"""
    data: List[Dict[str, Any]]
    user_id_col: str
    date_col: str
    cohort_type: str = Field(default="retention", pattern="^(retention|revenue|behavioral)$")
    revenue_col: Optional[str] = None
    event_col: Optional[str] = None
    cohort_period: str = Field(default="monthly", pattern="^(weekly|monthly|quarterly)$")


def prepare_cohort_data(df: pd.DataFrame, user_id_col: str, date_col: str, 
                       revenue_col: Optional[str] = None, event_col: Optional[str] = None):
    """Prepare data for cohort analysis"""
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=[date_col])
    
    # Get first purchase/event date for each user (cohort assignment)
    cohort_data = df.groupby(user_id_col, as_index=False)[date_col].min()
    cohort_data.columns = [user_id_col, 'cohort_date']
    
    # Merge back to original data
    df = df.merge(cohort_data, on=user_id_col, how='left')
    
    return df


def calculate_retention_cohorts(df: pd.DataFrame, user_id_col: str, date_col: str, period: str = 'monthly'):
    """Calculate retention cohort analysis"""
    
    # Make a copy to avoid issues
    df = df.copy()
    
    # Define period grouping
    if period == 'weekly':
        df['cohort_period'] = df['cohort_date'].dt.to_period('W').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('W').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('W')
        df['activity_period_obj'] = df[date_col].dt.to_period('W')
    elif period == 'monthly':
        df['cohort_period'] = df['cohort_date'].dt.to_period('M').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('M').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('M')
        df['activity_period_obj'] = df[date_col].dt.to_period('M')
    else:  # quarterly
        df['cohort_period'] = df['cohort_date'].dt.to_period('Q').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('Q').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('Q')
        df['activity_period_obj'] = df[date_col].dt.to_period('Q')
    
    # Calculate period index (periods since cohort)
    df['period_index'] = (df['activity_period_obj'] - df['cohort_period_obj']).apply(lambda x: x.n)
    
    # Group by cohort and period index
    cohort_data = df.groupby(['cohort_period', 'period_index'], as_index=False)[user_id_col].nunique()
    cohort_data.columns = ['cohort_period', 'period_index', 'users']
    
    # Pivot to create cohort table
    cohort_table = cohort_data.pivot(index='cohort_period', columns='period_index', values='users')
    
    # Fill NaN with 0
    cohort_table = cohort_table.fillna(0)
    
    # Calculate retention percentages
    cohort_sizes = cohort_table.iloc[:, 0]
    cohort_sizes = cohort_sizes.replace(0, 1)  # Avoid division by zero
    
    retention_table = cohort_table.div(cohort_sizes, axis=0) * 100
    
    return cohort_table, retention_table


def calculate_revenue_cohorts(df: pd.DataFrame, user_id_col: str, date_col: str, 
                              revenue_col: str, period: str = 'monthly'):
    """Calculate revenue cohort analysis"""
    
    # Make a copy
    df = df.copy()
    
    # Define period grouping
    if period == 'weekly':
        df['cohort_period'] = df['cohort_date'].dt.to_period('W').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('W').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('W')
        df['activity_period_obj'] = df[date_col].dt.to_period('W')
    elif period == 'monthly':
        df['cohort_period'] = df['cohort_date'].dt.to_period('M').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('M').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('M')
        df['activity_period_obj'] = df[date_col].dt.to_period('M')
    else:  # quarterly
        df['cohort_period'] = df['cohort_date'].dt.to_period('Q').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('Q').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('Q')
        df['activity_period_obj'] = df[date_col].dt.to_period('Q')
    
    df['period_index'] = (df['activity_period_obj'] - df['cohort_period_obj']).apply(lambda x: x.n)
    
    # Group by cohort and period index - sum revenue
    cohort_data = df.groupby(['cohort_period', 'period_index'], as_index=False)[revenue_col].sum()
    cohort_data.columns = ['cohort_period', 'period_index', 'revenue']
    
    # Pivot to create revenue table
    revenue_table = cohort_data.pivot(index='cohort_period', columns='period_index', values='revenue')
    revenue_table = revenue_table.fillna(0)
    
    # Calculate cohort sizes for average revenue per user
    cohort_sizes = df.groupby('cohort_period')[user_id_col].nunique()
    cohort_sizes = cohort_sizes.replace(0, 1)
    
    avg_revenue_table = revenue_table.div(cohort_sizes, axis=0)
    
    # Calculate cumulative revenue
    cumulative_revenue_table = revenue_table.cumsum(axis=1)
    cumulative_avg_revenue_table = cumulative_revenue_table.div(cohort_sizes, axis=0)
    
    return revenue_table, avg_revenue_table, cumulative_avg_revenue_table


def calculate_behavioral_cohorts(df: pd.DataFrame, user_id_col: str, date_col: str, 
                                 event_col: str, period: str = 'monthly'):
    """Calculate behavioral cohort analysis"""
    
    # Make a copy
    df = df.copy()
    
    # Define period grouping
    if period == 'weekly':
        df['cohort_period'] = df['cohort_date'].dt.to_period('W').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('W').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('W')
        df['activity_period_obj'] = df[date_col].dt.to_period('W')
    elif period == 'monthly':
        df['cohort_period'] = df['cohort_date'].dt.to_period('M').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('M').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('M')
        df['activity_period_obj'] = df[date_col].dt.to_period('M')
    else:  # quarterly
        df['cohort_period'] = df['cohort_date'].dt.to_period('Q').astype(str)
        df['activity_period'] = df[date_col].dt.to_period('Q').astype(str)
        df['cohort_period_obj'] = df['cohort_date'].dt.to_period('Q')
        df['activity_period_obj'] = df[date_col].dt.to_period('Q')
    
    df['period_index'] = (df['activity_period_obj'] - df['cohort_period_obj']).apply(lambda x: x.n)
    
    # Count events by cohort and period
    cohort_data = df.groupby(['cohort_period', 'period_index'], as_index=False).size()
    cohort_data.columns = ['cohort_period', 'period_index', 'events']
    
    # Pivot
    events_table = cohort_data.pivot(index='cohort_period', columns='period_index', values='events')
    events_table = events_table.fillna(0)
    
    # Calculate average events per user
    cohort_sizes = df.groupby('cohort_period')[user_id_col].nunique()
    cohort_sizes = cohort_sizes.replace(0, 1)
    
    avg_events_table = events_table.div(cohort_sizes, axis=0)
    
    return events_table, avg_events_table


def calculate_cohort_metrics(retention_table: pd.DataFrame, revenue_table: Optional[pd.DataFrame] = None):
    """Calculate summary metrics for cohorts"""
    
    metrics = {}
    
    # Retention metrics
    if retention_table is not None and not retention_table.empty:
        # Overall retention by period
        avg_retention = retention_table.mean(axis=0)
        
        # First period retention (Day 0/Week 0/Month 0)
        if 0 in retention_table.columns:
            metrics['initial_retention'] = float(retention_table[0].mean())
        
        # Period 1 retention (first return)
        if 1 in retention_table.columns:
            metrics['period_1_retention'] = float(retention_table[1].mean())
        
        # Long-term retention (if available)
        max_period = retention_table.columns.max()
        if max_period >= 3:
            metrics['long_term_retention'] = float(retention_table[max_period].mean())
        
        # Retention curve (list of averages)
        metrics['retention_curve'] = [float(x) for x in avg_retention.values]
        metrics['retention_periods'] = [int(x) for x in avg_retention.index]
    
    # Revenue metrics
    if revenue_table is not None and not revenue_table.empty:
        metrics['total_revenue'] = float(revenue_table.sum().sum())
        metrics['avg_revenue_per_cohort'] = float(revenue_table.sum(axis=1).mean())
        
        # Revenue by period
        revenue_by_period = revenue_table.sum(axis=0)
        metrics['revenue_by_period'] = [float(x) for x in revenue_by_period.values]
    
    return metrics


def generate_visualizations(retention_table: pd.DataFrame, cohort_type: str, 
                           revenue_table: Optional[pd.DataFrame] = None,
                           cumulative_revenue_table: Optional[pd.DataFrame] = None):
    """Generate cohort visualizations"""
    visualizations = {}
    
    # 1. Retention Heatmap
    if retention_table is not None and not retention_table.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Limit to reasonable number of cohorts and periods
        plot_data = retention_table.iloc[:12, :12]  # Last 12 cohorts, first 12 periods
        
        sns.heatmap(plot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=50, vmin=0, vmax=100, cbar_kws={'label': 'Retention %'},
                   ax=ax, linewidths=0.5, linecolor='gray')
        
        ax.set_title('Cohort Retention Heatmap (%)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Periods Since Cohort Start', fontsize=11)
        ax.set_ylabel('Cohort Period', fontsize=11)
        
        # Format y-axis labels
        yticklabels = [str(label) for label in plot_data.index]
        ax.set_yticklabels(yticklabels, rotation=0)
        
        plt.tight_layout()
        visualizations['retention_heatmap'] = fig_to_base64(fig)
    
    # 2. Retention Curves
    if retention_table is not None and not retention_table.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot retention curves for each cohort
        for idx in retention_table.index[:8]:  # Plot up to 8 cohorts
            cohort_data = retention_table.loc[idx].dropna()
            ax.plot(cohort_data.index, cohort_data.values, marker='o', label=str(idx), linewidth=2)
        
        ax.set_title('Retention Curves by Cohort', fontsize=14, fontweight='bold')
        ax.set_xlabel('Periods Since Cohort Start', fontsize=11)
        ax.set_ylabel('Retention Rate (%)', fontsize=11)
        ax.legend(title='Cohort', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        visualizations['retention_curves'] = fig_to_base64(fig)
    
    # 3. Average Retention Curve
    if retention_table is not None and not retention_table.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        avg_retention = retention_table.mean(axis=0)
        ax.plot(avg_retention.index, avg_retention.values, marker='o', linewidth=3, 
               color='#4A90E2', markersize=8)
        ax.fill_between(avg_retention.index, 0, avg_retention.values, alpha=0.3, color='#4A90E2')
        
        ax.set_title('Average Retention Curve Across All Cohorts', fontsize=14, fontweight='bold')
        ax.set_xlabel('Periods Since Cohort Start', fontsize=11)
        ax.set_ylabel('Average Retention Rate (%)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Add value labels
        for x, y in zip(avg_retention.index, avg_retention.values):
            ax.text(x, y + 2, f'{y:.1f}%', ha='center', fontsize=9)
        
        plt.tight_layout()
        visualizations['avg_retention_curve'] = fig_to_base64(fig)
    
    # 4. Revenue Heatmap (if revenue data available)
    if revenue_table is not None and not revenue_table.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plot_data = revenue_table.iloc[:12, :12]
        
        sns.heatmap(plot_data, annot=True, fmt='.0f', cmap='YlGnBu', 
                   cbar_kws={'label': 'Revenue ($)'}, ax=ax, 
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title('Cohort Revenue Heatmap ($)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Periods Since Cohort Start', fontsize=11)
        ax.set_ylabel('Cohort Period', fontsize=11)
        
        yticklabels = [str(label) for label in plot_data.index]
        ax.set_yticklabels(yticklabels, rotation=0)
        
        plt.tight_layout()
        visualizations['revenue_heatmap'] = fig_to_base64(fig)
    
    # 5. Cumulative Revenue (if available)
    if cumulative_revenue_table is not None and not cumulative_revenue_table.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx in cumulative_revenue_table.index[:8]:
            cohort_data = cumulative_revenue_table.loc[idx].dropna()
            ax.plot(cohort_data.index, cohort_data.values, marker='o', label=str(idx), linewidth=2)
        
        ax.set_title('Cumulative Revenue per User by Cohort', fontsize=14, fontweight='bold')
        ax.set_xlabel('Periods Since Cohort Start', fontsize=11)
        ax.set_ylabel('Cumulative Revenue per User ($)', fontsize=11)
        ax.legend(title='Cohort', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        visualizations['cumulative_revenue'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(metrics: dict, cohort_type: str):
    """Generate key insights"""
    insights = []
    
    if cohort_type == 'retention':
        # Initial retention
        if 'period_1_retention' in metrics:
            p1_retention = metrics['period_1_retention']
            if p1_retention > 40:
                insights.append({
                    'title': 'Strong First-Period Retention',
                    'description': f"Period 1 retention of {p1_retention:.1f}% indicates strong product-market fit. Users find value quickly and return.",
                    'status': 'positive'
                })
            elif p1_retention > 20:
                insights.append({
                    'title': 'Moderate First-Period Retention',
                    'description': f"Period 1 retention of {p1_retention:.1f}% is acceptable but has room for improvement. Focus on onboarding and early engagement.",
                    'status': 'neutral'
                })
            else:
                insights.append({
                    'title': 'Low First-Period Retention',
                    'description': f"Period 1 retention of {p1_retention:.1f}% is concerning. Users are churning quickly. Immediate action needed on onboarding and value proposition.",
                    'status': 'warning'
                })
        
        # Long-term retention
        if 'long_term_retention' in metrics:
            lt_retention = metrics['long_term_retention']
            if lt_retention > 20:
                insights.append({
                    'title': 'Solid Long-term Retention',
                    'description': f"Long-term retention of {lt_retention:.1f}% shows strong loyalty. Core user base is stable.",
                    'status': 'positive'
                })
            else:
                insights.append({
                    'title': 'Long-term Retention Opportunity',
                    'description': f"Long-term retention of {lt_retention:.1f}% suggests users churn over time. Improve engagement loops and introduce new features.",
                    'status': 'neutral'
                })
    
    elif cohort_type == 'revenue':
        if 'total_revenue' in metrics:
            insights.append({
                'title': 'Revenue Performance',
                'description': f"Total cohort revenue of ${metrics['total_revenue']:,.0f} across analyzed periods. Monitor cohort LTV trends for monetization health.",
                'status': 'neutral'
            })
    
    return insights


@router.post("/cohort-analysis")
async def analyze_cohorts(request: CohortRequest):
    """
    Cohort Analysis Endpoint
    
    Analyzes user cohorts for retention, revenue, or behavioral patterns
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 50:
            raise HTTPException(400, "Insufficient data (need at least 50 records)")
        
        df = pd.DataFrame(request.data)
        
        required_cols = [request.user_id_col, request.date_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns: {missing}")
        
        # Check cohort type specific requirements
        if request.cohort_type == 'revenue' and not request.revenue_col:
            raise HTTPException(400, "revenue_col required for revenue cohort analysis")
        if request.cohort_type == 'behavioral' and not request.event_col:
            raise HTTPException(400, "event_col required for behavioral cohort analysis")
        
        # Prepare cohort data
        df = prepare_cohort_data(df, request.user_id_col, request.date_col, 
                                request.revenue_col, request.event_col)
        
        # Calculate cohorts based on type
        retention_table = None
        revenue_table = None
        cumulative_revenue_table = None
        
        if request.cohort_type == 'retention':
            cohort_counts, retention_table = calculate_retention_cohorts(
                df, request.user_id_col, request.date_col, request.cohort_period
            )
        elif request.cohort_type == 'revenue':
            revenue_table, avg_revenue_table, cumulative_revenue_table = calculate_revenue_cohorts(
                df, request.user_id_col, request.date_col, request.revenue_col, request.cohort_period
            )
            # Also calculate retention for revenue analysis
            cohort_counts, retention_table = calculate_retention_cohorts(
                df, request.user_id_col, request.date_col, request.cohort_period
            )
        elif request.cohort_type == 'behavioral':
            events_table, avg_events_table = calculate_behavioral_cohorts(
                df, request.user_id_col, request.date_col, request.event_col, request.cohort_period
            )
            retention_table = avg_events_table  # Use for visualization
        
        # Calculate metrics
        metrics = calculate_cohort_metrics(retention_table, revenue_table)
        
        # Generate visualizations
        visualizations = generate_visualizations(retention_table, request.cohort_type, 
                                                revenue_table, cumulative_revenue_table)
        
        # Generate insights
        insights = generate_insights(metrics, request.cohort_type)
        
        # Prepare cohort table data (first 10 cohorts, first 12 periods)
        if retention_table is not None:
            cohort_table_data = retention_table.iloc[:10, :12].fillna(0).round(2)
            cohort_table_json = json.loads(cohort_table_data.to_json(orient='split'))
        else:
            cohort_table_json = {'index': [], 'columns': [], 'data': []}
        
        response_data = {
            'success': True,
            'results': {
                'cohort_type': request.cohort_type,
                'cohort_period': request.cohort_period,
                'metrics': metrics,
                'cohort_table': cohort_table_json
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'cohort_analysis',
                'cohort_type': request.cohort_type,
                'total_cohorts': int(len(retention_table)) if retention_table is not None else 0,
                'total_users': int(df[request.user_id_col].nunique()),
                'date_range': f"{df[request.date_col].min().strftime('%Y-%m-%d')} to {df[request.date_col].max().strftime('%Y-%m-%d')}"
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
