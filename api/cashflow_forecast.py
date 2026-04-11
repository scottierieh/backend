"""
Cash Flow Forecasting Router for FastAPI
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import io
import base64
import time
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CashFlowRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    inflow_cols: List[str]
    outflow_cols: List[str]
    method: Literal["moving_average", "exponential", "seasonal", "regression"] = "moving_average"
    starting_balance: float = 0
    forecast_periods: int = 6


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


COLORS = {
    'inflow': '#22c55e',
    'outflow': '#ef4444',
    'net': '#3b82f6',
    'balance': '#8b5cf6',
    'forecast': '#f59e0b',
}


def moving_average_forecast(series: pd.Series, periods: int, window: int = 3) -> np.ndarray:
    """Simple moving average forecast"""
    ma = series.rolling(window=window).mean().iloc[-1]
    return np.full(periods, ma)


def exponential_smoothing_forecast(series: pd.Series, periods: int, alpha: float = 0.3) -> np.ndarray:
    """Exponential smoothing forecast"""
    result = [series.iloc[0]]
    for i in range(1, len(series)):
        result.append(alpha * series.iloc[i] + (1 - alpha) * result[-1])
    
    forecast = []
    last_value = result[-1]
    for _ in range(periods):
        forecast.append(last_value)
    
    return np.array(forecast)


def linear_regression_forecast(series: pd.Series, periods: int) -> np.ndarray:
    """Linear regression trend forecast"""
    x = np.arange(len(series))
    slope, intercept, _, _, _ = stats.linregress(x, series.values)
    
    future_x = np.arange(len(series), len(series) + periods)
    forecast = slope * future_x + intercept
    
    return np.maximum(forecast, 0)


def seasonal_forecast(series: pd.Series, periods: int) -> np.ndarray:
    """Seasonal decomposition forecast"""
    n = len(series)
    
    # Simple seasonal factors (assuming monthly data)
    seasonal_period = min(12, n // 2)
    if seasonal_period < 2:
        return moving_average_forecast(series, periods)
    
    # Calculate seasonal factors
    seasonal_factors = []
    for i in range(seasonal_period):
        indices = list(range(i, n, seasonal_period))
        factor = series.iloc[indices].mean() / series.mean() if series.mean() > 0 else 1
        seasonal_factors.append(factor)
    
    # Trend
    trend = series.rolling(window=seasonal_period, min_periods=1).mean()
    trend_slope = (trend.iloc[-1] - trend.iloc[0]) / n if n > 1 else 0
    
    # Forecast
    forecast = []
    last_trend = trend.iloc[-1]
    for i in range(periods):
        seasonal_idx = (n + i) % seasonal_period
        trend_value = last_trend + trend_slope * (i + 1)
        forecast.append(max(0, trend_value * seasonal_factors[seasonal_idx]))
    
    return np.array(forecast)


def calculate_trend(series: pd.Series) -> float:
    """Calculate trend as percentage change"""
    if len(series) < 2 or series.iloc[0] == 0:
        return 0.0
    
    x = np.arange(len(series))
    slope, _, _, _, _ = stats.linregress(x, series.values)
    
    avg = series.mean()
    if avg == 0:
        return 0.0
    
    return slope / avg


def create_forecast_chart(historical: List[Dict], forecast: List[Dict]) -> str:
    """Create cash flow forecast chart"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Historical data
    hist_periods = [h['period'] for h in historical]
    hist_inflow = [h['inflow'] / 1000 for h in historical]
    hist_outflow = [h['outflow'] / 1000 for h in historical]
    
    # Forecast data
    fcst_periods = [f['period'] for f in forecast]
    fcst_inflow = [f['inflow'] / 1000 for f in forecast]
    fcst_outflow = [f['outflow'] / 1000 for f in forecast]
    
    all_periods = hist_periods + fcst_periods
    x_hist = range(len(hist_periods))
    x_fcst = range(len(hist_periods), len(all_periods))
    
    # Plot historical
    ax.bar([x - 0.2 for x in x_hist], hist_inflow, width=0.4, color=COLORS['inflow'], 
           alpha=0.8, label='Inflow (Historical)')
    ax.bar([x + 0.2 for x in x_hist], hist_outflow, width=0.4, color=COLORS['outflow'], 
           alpha=0.8, label='Outflow (Historical)')
    
    # Plot forecast
    ax.bar([x - 0.2 for x in x_fcst], fcst_inflow, width=0.4, color=COLORS['inflow'], 
           alpha=0.4, hatch='//', label='Inflow (Forecast)')
    ax.bar([x + 0.2 for x in x_fcst], fcst_outflow, width=0.4, color=COLORS['outflow'], 
           alpha=0.4, hatch='//', label='Outflow (Forecast)')
    
    # Divider line
    ax.axvline(x=len(hist_periods) - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(len(hist_periods) - 0.3, ax.get_ylim()[1] * 0.95, 'Forecast →', fontsize=10, alpha=0.7)
    
    ax.set_xticks(range(len(all_periods)))
    ax.set_xticklabels(all_periods, rotation=45, ha='right')
    ax.set_ylabel('Amount ($K)', fontsize=11)
    ax.set_title('Cash Flow Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cumulative_chart(periods: List[Dict]) -> str:
    """Create cumulative cash balance chart"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    period_names = [p['period'] for p in periods]
    balances = [p['closing_balance'] / 1000 for p in periods]
    
    # Determine forecast start
    fcst_start = next((i for i, p in enumerate(periods) if 'F' in p['period']), len(periods))
    
    colors = [COLORS['balance'] if i < fcst_start else COLORS['forecast'] for i in range(len(periods))]
    
    ax.fill_between(range(len(periods)), balances, alpha=0.3, color=COLORS['balance'])
    ax.plot(range(len(periods)), balances, 'o-', color=COLORS['balance'], linewidth=2, markersize=6)
    
    # Highlight forecast
    if fcst_start < len(periods):
        ax.fill_between(range(fcst_start, len(periods)), 
                        balances[fcst_start:], 
                        alpha=0.3, color=COLORS['forecast'])
        ax.axvline(x=fcst_start - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    # Zero line
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    
    # Min balance marker
    min_idx = balances.index(min(balances))
    ax.scatter([min_idx], [balances[min_idx]], color='red', s=100, zorder=5, marker='v')
    ax.annotate(f'Min: ${balances[min_idx]:.0f}K', (min_idx, balances[min_idx]),
                textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
    
    ax.set_xticks(range(len(period_names)))
    ax.set_xticklabels(period_names, rotation=45, ha='right')
    ax.set_ylabel('Cash Balance ($K)', fontsize=11)
    ax.set_title('Projected Cash Balance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_category_breakdown_chart(categories: List[Dict]) -> str:
    """Create category breakdown chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Inflows
    inflows = [c for c in categories if c['type'] == 'inflow']
    if inflows:
        labels = [c['category'] for c in inflows]
        values = [c['total'] / 1000 for c in inflows]
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(inflows)))
        
        ax1.barh(labels, values, color=colors, edgecolor='white')
        for i, v in enumerate(values):
            ax1.text(v + max(values) * 0.02, i, f'${v:.0f}K', va='center', fontsize=9)
        ax1.set_xlabel('Amount ($K)', fontsize=11)
        ax1.set_title('Cash Inflows', fontsize=14, fontweight='bold', color=COLORS['inflow'])
        ax1.invert_yaxis()
    
    # Outflows
    outflows = [c for c in categories if c['type'] == 'outflow']
    if outflows:
        labels = [c['category'] for c in outflows]
        values = [c['total'] / 1000 for c in outflows]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(outflows)))
        
        ax2.barh(labels, values, color=colors, edgecolor='white')
        for i, v in enumerate(values):
            ax2.text(v + max(values) * 0.02, i, f'${v:.0f}K', va='center', fontsize=9)
        ax2.set_xlabel('Amount ($K)', fontsize=11)
        ax2.set_title('Cash Outflows', fontsize=14, fontweight='bold', color=COLORS['outflow'])
        ax2.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(summary: Dict, risk_metrics: Dict, trend_analysis: Dict,
                          min_balance: float) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Cash position
    if summary['ending_balance'] > summary['total_inflow'] * 0.2:
        insights.append({
            'title': f'Strong Cash Position: ${summary["ending_balance"]:,.0f}',
            'description': 'Projected ending balance is healthy. Consider strategic investments.',
            'status': 'positive'
        })
    elif summary['ending_balance'] < 0:
        insights.append({
            'title': f'Cash Shortfall Projected: ${summary["ending_balance"]:,.0f}',
            'description': 'Forecast shows negative cash balance. Immediate action required.',
            'status': 'warning'
        })
    
    # Runway
    if risk_metrics['cash_runway_months'] < 6:
        insights.append({
            'title': f'Low Cash Runway: {risk_metrics["cash_runway_months"]:.1f} months',
            'description': 'Cash runway below 6 months. Review expenses and secure funding.',
            'status': 'warning'
        })
    elif risk_metrics['cash_runway_months'] >= 12:
        insights.append({
            'title': f'Healthy Runway: {risk_metrics["cash_runway_months"]:.1f} months',
            'description': 'Cash runway exceeds 12 months. Good financial buffer.',
            'status': 'positive'
        })
    
    # Trend
    if trend_analysis['net_trend'] < -0.05:
        insights.append({
            'title': 'Declining Net Cash Flow',
            'description': f'Net cash flow trending down {abs(trend_analysis["net_trend"]*100):.1f}%. Monitor closely.',
            'status': 'warning'
        })
    elif trend_analysis['net_trend'] > 0.05:
        insights.append({
            'title': 'Improving Net Cash Flow',
            'description': f'Net cash flow trending up {trend_analysis["net_trend"]*100:.1f}%.',
            'status': 'positive'
        })
    
    # Minimum balance warning
    if min_balance < 0:
        insights.append({
            'title': f'Minimum Balance Alert: ${min_balance:,.0f}',
            'description': 'Forecast shows periods with negative balance. Plan for financing.',
            'status': 'warning'
        })
    
    return insights


@router.post("/cashflow")
async def run_cashflow_forecast(request: CashFlowRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.date_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{request.date_col}' not found")
        
        for col in request.inflow_cols + request.outflow_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Sort by date
        df = df.sort_values(request.date_col).reset_index(drop=True)
        
        # Calculate totals
        for col in request.inflow_cols + request.outflow_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['total_inflow'] = df[request.inflow_cols].sum(axis=1)
        df['total_outflow'] = df[request.outflow_cols].sum(axis=1)
        df['net_cash_flow'] = df['total_inflow'] - df['total_outflow']
        
        # Forecast
        forecast_func = {
            'moving_average': moving_average_forecast,
            'exponential': exponential_smoothing_forecast,
            'seasonal': seasonal_forecast,
            'regression': linear_regression_forecast,
        }.get(request.method, moving_average_forecast)
        
        inflow_forecast = forecast_func(df['total_inflow'], request.forecast_periods)
        outflow_forecast = forecast_func(df['total_outflow'], request.forecast_periods)
        
        # Build periods
        historical_periods = []
        balance = request.starting_balance
        
        for idx, row in df.iterrows():
            opening = balance
            net = row['net_cash_flow']
            balance = opening + net
            
            historical_periods.append({
                'period': str(row[request.date_col]),
                'inflow': row['total_inflow'],
                'outflow': row['total_outflow'],
                'net_cash_flow': net,
                'cumulative_cash': balance,
                'opening_balance': opening,
                'closing_balance': balance,
            })
        
        # Forecast periods
        forecast_periods = []
        last_date = df[request.date_col].iloc[-1]
        
        for i in range(request.forecast_periods):
            opening = balance
            inflow = inflow_forecast[i]
            outflow = outflow_forecast[i]
            net = inflow - outflow
            balance = opening + net
            
            forecast_periods.append({
                'period': f'F{i+1}',
                'inflow': inflow,
                'outflow': outflow,
                'net_cash_flow': net,
                'cumulative_cash': balance,
                'opening_balance': opening,
                'closing_balance': balance,
            })
        
        all_periods = historical_periods + forecast_periods
        
        # Categories
        categories = []
        for col in request.inflow_cols:
            total = df[col].sum()
            trend = calculate_trend(df[col])
            forecast_val = forecast_func(df[col], request.forecast_periods).sum()
            categories.append({
                'category': col,
                'type': 'inflow',
                'total': total,
                'average': df[col].mean(),
                'trend': trend,
                'forecast': forecast_val,
            })
        
        for col in request.outflow_cols:
            total = df[col].sum()
            trend = calculate_trend(df[col])
            forecast_val = forecast_func(df[col], request.forecast_periods).sum()
            categories.append({
                'category': col,
                'type': 'outflow',
                'total': total,
                'average': df[col].mean(),
                'trend': trend,
                'forecast': forecast_val,
            })
        
        # Summary
        total_inflow = df['total_inflow'].sum() + sum(inflow_forecast)
        total_outflow = df['total_outflow'].sum() + sum(outflow_forecast)
        all_balances = [p['closing_balance'] for p in all_periods]
        
        summary = {
            'total_inflow': total_inflow,
            'total_outflow': total_outflow,
            'total_net': total_inflow - total_outflow,
            'avg_monthly_inflow': df['total_inflow'].mean(),
            'avg_monthly_outflow': df['total_outflow'].mean(),
            'avg_monthly_net': df['net_cash_flow'].mean(),
            'min_balance': min(all_balances),
            'max_balance': max(all_balances),
            'ending_balance': all_periods[-1]['closing_balance'],
        }
        
        # Trend analysis
        trend_analysis = {
            'inflow_trend': calculate_trend(df['total_inflow']),
            'outflow_trend': calculate_trend(df['total_outflow']),
            'net_trend': calculate_trend(df['net_cash_flow']),
        }
        
        # Risk metrics
        burn_rate = max(0, -df['net_cash_flow'].mean())
        ending_balance = all_periods[-1]['closing_balance']
        runway = ending_balance / burn_rate if burn_rate > 0 else 999
        
        negative_periods = sum(1 for p in forecast_periods if p['closing_balance'] < 0)
        shortfall_prob = negative_periods / len(forecast_periods) if forecast_periods else 0
        
        risk_metrics = {
            'cash_runway_months': min(runway, 999),
            'burn_rate': burn_rate,
            'volatility': df['net_cash_flow'].std() / df['net_cash_flow'].mean() if df['net_cash_flow'].mean() != 0 else 0,
            'shortfall_probability': shortfall_prob,
        }
        
        # Scenarios
        scenarios = [
            {
                'name': 'Base Case',
                'ending_balance': ending_balance,
                'min_balance': min(all_balances),
            },
            {
                'name': 'Optimistic (+10%)',
                'ending_balance': ending_balance + sum(inflow_forecast) * 0.1,
                'min_balance': min(all_balances) + sum(inflow_forecast) * 0.1 / len(forecast_periods),
            },
            {
                'name': 'Pessimistic (-10%)',
                'ending_balance': ending_balance - sum(inflow_forecast) * 0.1,
                'min_balance': min(all_balances) - sum(inflow_forecast) * 0.1 / len(forecast_periods),
            },
        ]
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {
            'forecast_chart': create_forecast_chart(historical_periods[-12:], forecast_periods),
            'cumulative_chart': create_cumulative_chart(all_periods[-18:]),
            'category_breakdown': create_category_breakdown_chart(categories),
        }
        
        # Key insights
        key_insights = generate_key_insights(summary, risk_metrics, trend_analysis, summary['min_balance'])
        
        results = {
            'method': request.method,
            'forecast_periods': [{k: _to_native_type(v) for k, v in p.items()} for p in all_periods],
            'categories': [{k: _to_native_type(v) for k, v in c.items()} for c in categories],
            'summary': {k: _to_native_type(v) for k, v in summary.items()},
            'trend_analysis': {k: _to_native_type(v) for k, v in trend_analysis.items()},
            'risk_metrics': {k: _to_native_type(v) for k, v in risk_metrics.items()},
            'scenarios': [{k: _to_native_type(v) for k, v in s.items()} for s in scenarios],
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'method': request.method,
                'forecast_months': request.forecast_periods,
                'ending_balance': ending_balance,
                'min_balance': summary['min_balance'],
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cash flow forecast failed: {str(e)}")
