"""
Sales Forecast Analysis FastAPI Endpoint
Time series forecasting using Prophet and Simple algorithms
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
from io import BytesIO
import base64
import warnings

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class ForecastRequest(BaseModel):
    """Request model for Sales Forecast"""
    data: List[Dict[str, Any]]
    date_col: str
    sales_col: str
    forecast_days: int = Field(default=30, ge=7, le=90)


class ForecastData(BaseModel):
    """Individual forecast data point"""
    date: str
    forecast: float
    lower_bound: float
    upper_bound: float


class Metrics(BaseModel):
    """Forecast metrics"""
    total_historical_sales: float
    avg_daily_sales: float
    median_daily_sales: float
    total_forecast_sales: float
    avg_forecast_daily: float
    growth_rate: float
    trend_direction: str
    confidence_score: float
    model_used: str
    historical_days: int
    forecast_days: int


class KeyInsight(BaseModel):
    """Key insight"""
    title: str
    description: str
    status: str


class Summary(BaseModel):
    """Analysis summary"""
    analysis_type: str
    forecast_days: int
    total_forecast_sales: float
    growth_rate: float
    confidence_score: float


class ForecastResponse(BaseModel):
    """Response model for Sales Forecast"""
    success: bool
    results: Dict[str, Any]
    visualizations: Dict[str, Optional[str]]
    key_insights: List[KeyInsight]
    summary: Summary


def forecast_simple(daily_sales: pd.DataFrame, periods: int):
    """
    Simple forecasting using trend and seasonality
    """
    y = daily_sales['y'].values
    dates = pd.to_datetime(daily_sales['ds'])
    
    # Linear trend
    x = np.arange(len(y))
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    
    # Exponential smoothing
    alpha = 0.3
    ema = [y[0]]
    for i in range(1, len(y)):
        ema.append(alpha * y[i] + (1 - alpha) * ema[-1])
    ema = np.array(ema)
    
    # Future dates
    last_date = dates.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Forecast
    future_x = np.arange(len(y), len(y) + periods)
    trend_forecast = slope * future_x + intercept
    
    # Weekly seasonality
    last_7_days = y[-7:]
    weekly_pattern = last_7_days / last_7_days.mean()
    seasonal_factor = np.tile(weekly_pattern, (periods // 7) + 1)[:periods]
    
    forecast_values = trend_forecast * seasonal_factor
    forecast_lower = forecast_values * 0.8
    forecast_upper = forecast_values * 1.2
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values,
        'yhat_lower': forecast_lower,
        'yhat_upper': forecast_upper,
        'trend': trend_forecast
    })
    
    # Historical forecast for full dataframe
    historical_forecast = pd.DataFrame({
        'ds': dates,
        'yhat': ema,
        'yhat_lower': ema * 0.9,
        'yhat_upper': ema * 1.1,
        'trend': slope * x + intercept
    })
    
    full_forecast = pd.concat([historical_forecast, forecast_df], ignore_index=True)
    
    return full_forecast, forecast_df, r_value ** 2


def calculate_metrics(historical: pd.DataFrame, future: pd.DataFrame, model_type: str, r_squared: float = None):
    """Calculate forecast metrics"""
    total_historical = historical['y'].sum()
    avg_daily_sales = historical['y'].mean()
    median_daily_sales = historical['y'].median()
    
    total_forecast = future['yhat'].sum()
    avg_forecast_daily = future['yhat'].mean()
    
    # Growth
    last_7_days_avg = historical['y'].tail(7).mean()
    first_7_forecast_avg = future['yhat'].head(7).mean()
    
    if last_7_days_avg > 0:
        growth_rate = ((first_7_forecast_avg - last_7_days_avg) / last_7_days_avg) * 100
    else:
        growth_rate = 0
    
    # Trend
    trend_values = future['trend'].values if 'trend' in future.columns else future['yhat'].values
    trend_slope = (trend_values[-1] - trend_values[0]) / len(trend_values)
    
    if trend_slope > avg_daily_sales * 0.01:
        trend_direction = "Increasing"
    elif trend_slope < -avg_daily_sales * 0.01:
        trend_direction = "Decreasing"
    else:
        trend_direction = "Stable"
    
    # Confidence
    if r_squared is not None:
        confidence = r_squared * 100
    else:
        confidence = 60.0  # Default for Prophet
    
    return {
        'total_historical_sales': float(total_historical),
        'avg_daily_sales': float(avg_daily_sales),
        'median_daily_sales': float(median_daily_sales),
        'total_forecast_sales': float(total_forecast),
        'avg_forecast_daily': float(avg_forecast_daily),
        'growth_rate': float(growth_rate),
        'trend_direction': trend_direction,
        'confidence_score': float(confidence),
        'model_used': model_type,
        'historical_days': len(historical),
        'forecast_days': len(future)
    }


def generate_visualizations(historical: pd.DataFrame, forecast: pd.DataFrame, future: pd.DataFrame):
    """Generate all visualizations"""
    visualizations = {}
    
    # 1. Forecast Overview
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(historical['ds'], historical['y'], linewidth=2, label='Historical Sales', color='#2C3E50')
    forecast_dates = forecast['ds'].tail(len(future))
    forecast_values = forecast['yhat'].tail(len(future))
    ax.plot(forecast_dates, forecast_values, linewidth=2, label='Forecast', color='#3498DB', linestyle='--')
    if 'yhat_lower' in forecast.columns:
        ax.fill_between(forecast_dates, forecast['yhat_lower'].tail(len(future)), 
                        forecast['yhat_upper'].tail(len(future)), alpha=0.2, color='#3498DB', label='Confidence Interval')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sales', fontsize=11, fontweight='bold')
    ax.set_title('Sales Forecast Overview', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['forecast_overview'] = fig_to_base64(fig)
    
    # 2. Trend Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    if 'trend' in forecast.columns:
        ax.plot(forecast['ds'], forecast['trend'], linewidth=2, color='#2C3E50', label='Trend')
        forecast_start = historical['ds'].max()
        ax.axvline(forecast_start, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Forecast Start')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Trend Value', fontsize=11, fontweight='bold')
    ax.set_title('Sales Trend Analysis', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['trend_analysis'] = fig_to_base64(fig)
    
    # 3. Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    last_30_days = historical.tail(30)
    hist_avg = last_30_days['y'].mean()
    forecast_avg = future['yhat'].mean()
    categories = ['Last 30 Days\n(Historical)', 'Next Period\n(Forecast)']
    values = [hist_avg, forecast_avg]
    colors = ['#2C3E50', '#3498DB']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'${height:,.0f}',
               ha='center', va='bottom', fontweight='bold', fontsize=11)
    change_pct = ((forecast_avg - hist_avg) / hist_avg) * 100
    ax.text(0.5, 0.95, f'Change: {change_pct:+.1f}%', transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Daily Sales', fontsize=11, fontweight='bold')
    ax.set_title('Historical vs Forecast Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    visualizations['comparison'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(metrics: dict):
    """Generate key insights"""
    insights = []
    
    growth_rate = metrics['growth_rate']
    trend = metrics['trend_direction']
    
    if growth_rate > 10:
        insights.append({
            'title': 'Strong Growth Expected',
            'description': f"Forecast shows {growth_rate:.1f}% growth in average daily sales. {trend} trend detected in the forecast period.",
            'status': 'positive'
        })
    elif growth_rate < -10:
        insights.append({
            'title': 'Declining Sales Forecast',
            'description': f"Forecast indicates {abs(growth_rate):.1f}% decline in average daily sales. {trend} trend detected. Consider promotional strategies.",
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Stable Sales Forecast',
            'description': f"Sales expected to remain relatively stable with {growth_rate:+.1f}% change. {trend} trend pattern observed.",
            'status': 'neutral'
        })
    
    confidence = metrics['confidence_score']
    if confidence > 70:
        insights.append({
            'title': 'High Forecast Confidence',
            'description': f"Model confidence score of {confidence:.1f}% indicates reliable predictions. {metrics['model_used']} algorithm used for analysis.",
            'status': 'positive'
        })
    elif confidence > 50:
        insights.append({
            'title': 'Moderate Forecast Confidence',
            'description': f"Model confidence score of {confidence:.1f}%. Forecast provides reasonable estimates but monitor actual performance closely.",
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': 'Lower Forecast Confidence',
            'description': f"Model confidence score of {confidence:.1f}%. High variability in historical data. Use forecast as directional guidance.",
            'status': 'warning'
        })
    
    total_forecast = metrics['total_forecast_sales']
    forecast_days = metrics['forecast_days']
    
    insights.append({
        'title': 'Forecast Period Outlook',
        'description': f"Next {forecast_days} days forecasted at ${total_forecast:,.0f} total sales.",
        'status': 'positive'
    })
    
    return insights


@router.post("/forecast")
async def sales_forecast(request: ForecastRequest):
    """
    Sales Forecast Analysis
    
    Time series forecasting for future sales prediction
    """
    try:
        # Validate
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 30:
            raise HTTPException(400, "Insufficient data (need at least 30 observations)")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        df[request.date_col] = pd.to_datetime(df[request.date_col])
        df[request.sales_col] = pd.to_numeric(df[request.sales_col])
        
        # Sort by date
        df = df.sort_values(request.date_col).reset_index(drop=True)
        
        # Aggregate by date
        daily_sales = df.groupby(request.date_col)[request.sales_col].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        # Try Prophet first
        try:
            from prophet import Prophet
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(daily_sales)
            future = model.make_future_dataframe(periods=request.forecast_days)
            forecast = model.predict(future)
            future_only = forecast.tail(request.forecast_days)
            model_type = 'Prophet'
            r_squared = None
        except:
            # Fallback to simple
            forecast, future_only, r_squared = forecast_simple(daily_sales, request.forecast_days)
            model_type = 'Simple Trend + Seasonality'
        
        # Calculate metrics
        metrics = calculate_metrics(daily_sales, future_only, model_type, r_squared)
        
        # Generate visualizations
        visualizations = generate_visualizations(daily_sales, forecast, future_only)
        
        # Generate insights
        insights = generate_insights(metrics)
        
        # Prepare forecast data
        forecast_data = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_data['ds'] = forecast_data['ds'].dt.strftime('%Y-%m-%d')
        forecast_data.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        
        return ForecastResponse(
            success=True,
            results={
                'forecast_data': forecast_data.to_dict('records'),
                'metrics': metrics
            },
            visualizations=visualizations,
            key_insights=insights,
            summary=Summary(
                analysis_type='sales_forecast',
                forecast_days=request.forecast_days,
                total_forecast_sales=metrics['total_forecast_sales'],
                growth_rate=metrics['growth_rate'],
                confidence_score=metrics['confidence_score']
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

