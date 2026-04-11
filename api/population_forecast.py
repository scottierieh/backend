"""
Population Forecast Router for FastAPI
Time Series Forecasting with Multiple Methods — Optimized for Annual Demographic Data
Endpoint: POST /population-forecast
Based on forecast.py with bug-fixes, no matplotlib overhead in response
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
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class PopulationForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    value_col: str
    forecast_periods: int = 10
    frequency: Literal["D", "W", "M", "Q", "Y"] = "Y"
    method: Literal["auto", "moving_average", "exponential", "holt", "holt_winters", "linear", "ensemble"] = "auto"
    seasonality: Optional[int] = None
    confidence_level: float = 0.95
    group_col: Optional[str] = None


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


# ── Forecasting Methods ──────────────────────────────────────────────

def moving_average_forecast(series: np.ndarray, periods: int, window: int = 3) -> Dict[str, Any]:
    """Simple Moving Average forecast"""
    forecast = []
    history = list(series)

    for _ in range(periods):
        avg = np.mean(history[-window:])
        forecast.append(avg)
        history.append(avg)

    fitted = []
    for i in range(len(series)):
        if i < window:
            fitted.append(np.mean(series[:i + 1]))
        else:
            fitted.append(np.mean(series[i - window:i]))

    return {
        'forecast': np.array(forecast),
        'fitted': np.array(fitted),
        'method': f'Moving Average (window={window})'
    }


def exponential_smoothing_forecast(series: np.ndarray, periods: int, alpha: float = None) -> Dict[str, Any]:
    """Simple Exponential Smoothing"""
    if alpha is None:
        def sse(alpha_arr):
            a = float(alpha_arr[0])
            fitted = [float(series[0])]
            for i in range(1, len(series)):
                fitted.append(a * series[i - 1] + (1 - a) * fitted[-1])
            return np.sum((series - np.array(fitted)) ** 2)

        result = minimize(sse, x0=[0.5], bounds=[(0.01, 0.99)])
        alpha = float(result.x[0])

    fitted = [float(series[0])]
    for i in range(1, len(series)):
        fitted.append(alpha * series[i - 1] + (1 - alpha) * fitted[-1])

    last_level = alpha * series[-1] + (1 - alpha) * fitted[-1]
    forecast = np.array([float(last_level)] * periods)

    return {
        'forecast': forecast,
        'fitted': np.array(fitted),
        'alpha': _to_native_type(alpha),
        'method': f'Exponential Smoothing (α={alpha:.3f})'
    }


def holt_forecast(series: np.ndarray, periods: int, alpha: float = None, beta: float = None) -> Dict[str, Any]:
    """Holt's Linear Trend Method"""
    n = len(series)

    if alpha is None or beta is None:
        def sse(params):
            a = float(params[0])
            b = float(params[1])
            level = float(series[0])
            trend = float(series[1] - series[0]) if n > 1 else 0.0
            fitted = []
            for i in range(n):
                fitted.append(level + trend)
                if i < n - 1:
                    new_level = a * series[i] + (1 - a) * (level + trend)
                    new_trend = b * (new_level - level) + (1 - b) * trend
                    level, trend = float(new_level), float(new_trend)
            return np.sum((series - np.array(fitted)) ** 2)

        result = minimize(sse, x0=[0.5, 0.5], bounds=[(0.01, 0.99), (0.01, 0.99)])
        alpha, beta = float(result.x[0]), float(result.x[1])

    level = float(series[0])
    trend = float(series[1] - series[0]) if n > 1 else 0.0
    fitted = []

    for i in range(n):
        fitted.append(level + trend)
        if i < n - 1:
            new_level = alpha * series[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level, trend = float(new_level), float(new_trend)

    level = alpha * series[-1] + (1 - alpha) * (level + trend)
    trend = beta * (level - (fitted[-1] - trend)) + (1 - beta) * trend

    forecast = np.array([level + trend * (i + 1) for i in range(periods)])

    return {
        'forecast': forecast,
        'fitted': np.array(fitted),
        'alpha': _to_native_type(alpha),
        'beta': _to_native_type(beta),
        'method': f'Holt Linear (α={alpha:.3f}, β={beta:.3f})'
    }


def holt_winters_forecast(series: np.ndarray, periods: int, seasonal_period: int = 12,
                          alpha: float = None, beta: float = None, gamma: float = None) -> Dict[str, Any]:
    """Holt-Winters Seasonal Method (Additive)"""
    n = len(series)

    if n < 2 * seasonal_period:
        return holt_forecast(series, periods, alpha, beta)

    seasonal = np.zeros(seasonal_period)
    for i in range(seasonal_period):
        indices = range(i, min(n, seasonal_period * 2), seasonal_period)
        seasonal[i] = np.mean([series[j] for j in indices]) - np.mean(series[:seasonal_period * 2])

    if alpha is None or beta is None or gamma is None:
        def sse(params):
            a = float(params[0])
            b = float(params[1])
            g = float(params[2])
            level = float(np.mean(series[:seasonal_period]))
            trend = float((np.mean(series[seasonal_period:2 * seasonal_period]) - np.mean(
                series[:seasonal_period])) / seasonal_period)
            seas = seasonal.copy()
            fitted = []

            for i in range(n):
                si = i % seasonal_period
                fitted.append(level + trend + seas[si])
                if i < n - 1:
                    new_level = a * (series[i] - seas[si]) + (1 - a) * (level + trend)
                    new_trend = b * (new_level - level) + (1 - b) * trend
                    seas[si] = g * (series[i] - new_level) + (1 - g) * seas[si]
                    level, trend = float(new_level), float(new_trend)

            return np.sum((series - np.array(fitted)) ** 2)

        result = minimize(sse, x0=[0.5, 0.1, 0.5], bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)])
        alpha, beta, gamma = float(result.x[0]), float(result.x[1]), float(result.x[2])

    level = float(np.mean(series[:seasonal_period]))
    trend = float(
        (np.mean(series[seasonal_period:2 * seasonal_period]) - np.mean(series[:seasonal_period])) / seasonal_period)
    seas = seasonal.copy()
    fitted = []

    for i in range(n):
        si = i % seasonal_period
        fitted.append(level + trend + seas[si])
        if i < n - 1:
            new_level = alpha * (series[i] - seas[si]) + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            seas[si] = gamma * (series[i] - new_level) + (1 - gamma) * seas[si]
            level, trend = float(new_level), float(new_trend)

    si = (n - 1) % seasonal_period
    level = alpha * (series[-1] - seas[si]) + (1 - alpha) * (level + trend)
    trend = beta * (level - (fitted[-1] - trend - seas[si])) + (1 - beta) * trend
    seas[si] = gamma * (series[-1] - level) + (1 - gamma) * seas[si]

    forecast = []
    for i in range(periods):
        si = (n + i) % seasonal_period
        forecast.append(level + trend * (i + 1) + seas[si])

    return {
        'forecast': np.array(forecast),
        'fitted': np.array(fitted),
        'alpha': _to_native_type(alpha),
        'beta': _to_native_type(beta),
        'gamma': _to_native_type(gamma),
        'seasonal_period': seasonal_period,
        'method': f'Holt-Winters (α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f})'
    }


def linear_trend_forecast(series: np.ndarray, periods: int) -> Dict[str, Any]:
    """Linear Trend Forecast"""
    n = len(series)
    x = np.arange(n)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)

    fitted = intercept + slope * x
    forecast_x = np.arange(n, n + periods)
    forecast = intercept + slope * forecast_x

    return {
        'forecast': forecast,
        'fitted': fitted,
        'slope': _to_native_type(slope),
        'intercept': _to_native_type(intercept),
        'r_squared': _to_native_type(r_value ** 2),
        'method': f'Linear Trend (R²={r_value ** 2:.3f})'
    }


def ensemble_forecast(series: np.ndarray, periods: int, seasonal_period: int = 12) -> Dict[str, Any]:
    """Ensemble of multiple methods"""
    methods = []
    fitted_list = []

    ma_result = moving_average_forecast(series, periods)
    methods.append(('MA', np.array(ma_result['forecast']).flatten()))
    fitted_list.append(np.array(ma_result['fitted']).flatten())

    exp_result = exponential_smoothing_forecast(series, periods)
    methods.append(('ES', np.array(exp_result['forecast']).flatten()))
    fitted_list.append(np.array(exp_result['fitted']).flatten())

    holt_result = holt_forecast(series, periods)
    methods.append(('Holt', np.array(holt_result['forecast']).flatten()))
    fitted_list.append(np.array(holt_result['fitted']).flatten())

    linear_result = linear_trend_forecast(series, periods)
    methods.append(('Linear', np.array(linear_result['forecast']).flatten()))
    fitted_list.append(np.array(linear_result['fitted']).flatten())

    if len(series) >= 2 * seasonal_period:
        hw_result = holt_winters_forecast(series, periods, seasonal_period)
        methods.append(('HW', np.array(hw_result['forecast']).flatten()))
        fitted_list.append(np.array(hw_result['fitted']).flatten())

    forecast_arrays = [m[1] for m in methods]
    ensemble_fc = np.mean(np.vstack(forecast_arrays), axis=0)
    ensemble_fitted = np.mean(np.vstack(fitted_list), axis=0)

    return {
        'forecast': ensemble_fc,
        'fitted': ensemble_fitted,
        'components': {name: _to_native_type(fc) for name, fc in methods},
        'method': f'Ensemble ({len(methods)} models)'
    }


# ── Accuracy & CI ────────────────────────────────────────────────────

def calculate_forecast_accuracy(actual: np.ndarray, fitted: np.ndarray) -> Dict[str, float]:
    """Calculate forecast accuracy metrics"""
    errors = actual - fitted
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / actual) * 100
    pct_errors = pct_errors[~np.isinf(pct_errors)]

    return {
        'mae': _to_native_type(np.mean(abs_errors)),
        'mse': _to_native_type(np.mean(errors ** 2)),
        'rmse': _to_native_type(np.sqrt(np.mean(errors ** 2))),
        'mape': _to_native_type(np.mean(pct_errors)) if len(pct_errors) > 0 else None,
        'bias': _to_native_type(np.mean(errors)),
        'std_error': _to_native_type(np.std(errors))
    }


def calculate_confidence_intervals(series: np.ndarray, forecast: np.ndarray,
                                   fitted: np.ndarray, confidence: float) -> Dict[str, np.ndarray]:
    """Calculate confidence intervals for forecast"""
    errors = series - fitted
    std_error = np.std(errors)

    z_score = stats.norm.ppf((1 + confidence) / 2)
    horizon_factor = np.sqrt(np.arange(1, len(forecast) + 1))
    margin = z_score * std_error * horizon_factor

    return {
        'lower': forecast - margin,
        'upper': forecast + margin,
        'std_error': std_error
    }


def detect_seasonality(series: np.ndarray, max_lag: int = 24) -> Dict[str, Any]:
    """Detect seasonality in time series"""
    n = len(series)
    if n < max_lag * 2:
        max_lag = n // 2

    acf = []
    mean = np.mean(series)
    var = np.var(series)

    for lag in range(max_lag + 1):
        if var == 0:
            acf.append(0)
        else:
            cov = np.mean((series[:n - lag] - mean) * (series[lag:] - mean)) if lag > 0 else var
            acf.append(cov / var)

    peaks = []
    for i in range(2, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.2:
            peaks.append((i, acf[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    seasonal_period = peaks[0][0] if peaks else None

    return {
        'acf': [_to_native_type(a) for a in acf],
        'seasonal_period': seasonal_period,
        'peaks': [(p[0], _to_native_type(p[1])) for p in peaks[:5]],
        'has_seasonality': seasonal_period is not None
    }


def decompose_series(series: np.ndarray, period: int) -> Dict[str, np.ndarray]:
    """Simple additive decomposition"""
    n = len(series)

    if period is None or n < 2 * period:
        return {
            'trend': series,
            'seasonal': np.zeros(n),
            'residual': np.zeros(n)
        }

    trend = np.convolve(series, np.ones(period) / period, mode='same')

    half = period // 2
    for i in range(half):
        trend[i] = np.mean(series[:period])
        trend[-(i + 1)] = np.mean(series[-period:])

    detrended = series - trend

    seasonal = np.zeros(n)
    for i in range(period):
        indices = range(i, n, period)
        seasonal_mean = np.mean([detrended[j] for j in indices])
        for j in indices:
            seasonal[j] = seasonal_mean

    residual = series - trend - seasonal

    return {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual
    }


def auto_select_method(series: np.ndarray, seasonal_period: Optional[int]) -> str:
    """Automatically select best forecasting method"""
    n = len(series)

    x = np.arange(n)
    slope, _, r_value, _, _ = stats.linregress(x, series)
    has_trend = abs(r_value) > 0.5

    has_seasonality = seasonal_period is not None and n >= 2 * seasonal_period

    if has_seasonality and has_trend:
        return "holt_winters"
    elif has_trend:
        return "holt"
    elif n < 10:
        return "moving_average"
    else:
        return "exponential"


# ── Visualizations (matplotlib — frontend ignores these, uses Recharts) ─

def create_forecast_chart(dates, actual, fitted, forecast, forecast_dates,
                          lower, upper, method_name) -> str:
    """Generate main forecast chart"""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, actual, 'b-o', markersize=4, label='Historical', linewidth=2)
    ax.plot(dates, fitted, 'g--', alpha=0.6, label='Fitted', linewidth=1.5)
    ax.plot(forecast_dates, forecast, 'r-s', markersize=5, label='Forecast', linewidth=2)
    ax.fill_between(forecast_dates, lower, upper, alpha=0.15, color='red', label='CI Band')

    ax.set_title(f'Population Forecast — {method_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_components_chart(dates, decomposition) -> str:
    """Generate decomposition chart"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(dates, decomposition['trend'], 'b-', linewidth=2)
    axes[0].set_title('Trend', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    axes[1].plot(dates, decomposition['seasonal'], 'g-', linewidth=1.5)
    axes[1].set_title('Seasonal', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    axes[2].plot(dates, decomposition['residual'], 'r-', linewidth=1)
    axes[2].set_title('Residual', fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def create_accuracy_chart(actual, fitted, dates) -> str:
    """Generate accuracy comparison chart"""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, actual, 'b-o', markersize=3, label='Actual', linewidth=2)
    ax.plot(dates, fitted, 'g--', label='Fitted', linewidth=1.5, alpha=0.8)

    errors = actual - fitted
    ax.fill_between(dates, actual, fitted, alpha=0.15,
                    where=(errors >= 0), color='green', label='Under-forecast')
    ax.fill_between(dates, actual, fitted, alpha=0.15,
                    where=(errors < 0), color='red', label='Over-forecast')

    ax.set_title('Actual vs Fitted', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_seasonality_chart(acf_values, seasonal_period) -> str:
    """Generate ACF / seasonality chart"""
    fig, ax = plt.subplots(figsize=(12, 5))

    lags = range(len(acf_values))
    ax.bar(lags, acf_values, color='#3b82f6', alpha=0.7)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Threshold (0.2)')
    if seasonal_period:
        ax.axvline(x=seasonal_period, color='green', linestyle='--', linewidth=2,
                   label=f'Detected period = {seasonal_period}')

    ax.set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


# ── Key Insights ─────────────────────────────────────────────────────

def generate_key_insights(forecast_result, accuracy, seasonality, series, forecast) -> List[Dict[str, str]]:
    """Generate interpretive insights"""
    insights = []

    # Trend direction
    if len(forecast) > 1:
        pct_change = ((forecast[-1] - series[-1]) / series[-1]) * 100
        direction = 'growth' if pct_change > 0 else 'decline'
        insights.append({
            'title': f'Population {direction.title()} Projected',
            'description': f'The model projects a {abs(pct_change):.1f}% {direction} from the last observed value to the end of the forecast horizon ({forecast_result["method"]}).',
            'type': 'trend'
        })

    # Accuracy assessment
    mape = accuracy.get('mape')
    if mape is not None:
        if mape < 2:
            quality = 'excellent'
        elif mape < 5:
            quality = 'good'
        else:
            quality = 'moderate'
        insights.append({
            'title': f'{quality.title()} Model Fit',
            'description': f'MAPE = {mape:.2f}%, RMSE = {accuracy["rmse"]:.2f}. The model provides {quality} in-sample accuracy for historical observations.',
            'type': 'accuracy'
        })

    # Seasonality note
    if seasonality.get('has_seasonality'):
        insights.append({
            'title': 'Seasonality Detected',
            'description': f'ACF analysis detected a seasonal cycle of {seasonality["seasonal_period"]} periods.',
            'type': 'seasonality'
        })
    else:
        insights.append({
            'title': 'No Seasonality Detected',
            'description': 'No significant seasonal patterns found — appropriate for annual demographic data.',
            'type': 'seasonality'
        })

    # Growth rate analysis
    if len(series) > 5:
        recent_growth = np.mean(np.diff(series[-5:])) / np.mean(series[-5:]) * 100
        earlier_growth = np.mean(np.diff(series[:len(series) // 2])) / np.mean(series[:len(series) // 2]) * 100
        if recent_growth < earlier_growth - 0.2:
            insights.append({
                'title': 'Growth Deceleration Observed',
                'description': f'Recent 5-period avg growth ({recent_growth:.2f}%/yr) is slower than the historical average ({earlier_growth:.2f}%/yr), suggesting a demographic transition.',
                'type': 'deceleration'
            })

    # Methodology
    insights.append({
        'title': 'Methodology Note',
        'description': f'Method: {forecast_result["method"]}. The model assumes structural continuity — major policy changes, migration shocks, or pandemics are not modeled. Confidence intervals widen with forecast horizon.',
        'type': 'methodology'
    })

    return insights


# ── Main Endpoint ────────────────────────────────────────────────────

@router.post("/population-forecast")
async def run_population_forecast(request: PopulationForecastRequest) -> Dict[str, Any]:
    """
    Population Time Series Forecasting.
    
    Supports: Moving Average, Exponential Smoothing, Holt Linear Trend,
    Holt-Winters Seasonal, Linear Trend, Ensemble.
    Optimized for annual demographic data (frequency=Y, seasonality=null).
    """
    try:
        df = pd.DataFrame(request.data)

        if request.date_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Date column '{request.date_col}' not found")
        if request.value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Value column '{request.value_col}' not found")

        # Parse dates
        df[request.date_col] = pd.to_datetime(df[request.date_col])
        df = df.sort_values(request.date_col)

        # Aggregate by date
        df_agg = df.groupby(request.date_col)[request.value_col].sum().reset_index()

        dates = pd.DatetimeIndex(df_agg[request.date_col])
        series = df_agg[request.value_col].values.astype(float)

        if len(series) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 data points for forecasting")

        # Detect seasonality
        seasonality = detect_seasonality(series, max_lag=min(24, len(series) // 2))
        seasonal_period = request.seasonality or seasonality.get('seasonal_period')

        # Method selection
        method = request.method
        if method == "auto":
            method = auto_select_method(series, seasonal_period)

        # Run forecast
        if method == "moving_average":
            result = moving_average_forecast(series, request.forecast_periods)
        elif method == "exponential":
            result = exponential_smoothing_forecast(series, request.forecast_periods)
        elif method == "holt":
            result = holt_forecast(series, request.forecast_periods)
        elif method == "holt_winters":
            result = holt_winters_forecast(series, request.forecast_periods, seasonal_period or 12)
        elif method == "linear":
            result = linear_trend_forecast(series, request.forecast_periods)
        elif method == "ensemble":
            result = ensemble_forecast(series, request.forecast_periods, seasonal_period or 12)
        else:
            result = exponential_smoothing_forecast(series, request.forecast_periods)

        forecast = result['forecast']
        fitted = result['fitted']

        # Confidence intervals
        ci = calculate_confidence_intervals(series, forecast, fitted, request.confidence_level)

        # Generate forecast dates
        freq_map = {'D': 'D', 'W': 'W', 'M': 'MS', 'Q': 'QS', 'Y': 'YS'}
        freq = freq_map.get(request.frequency, 'MS')

        last_date = dates[-1]
        if request.frequency == 'Y':
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=request.forecast_periods, freq=freq
            )
        elif request.frequency == 'M':
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=request.forecast_periods, freq=freq
            )
        else:
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=request.forecast_periods, freq=freq
            )

        # Accuracy metrics
        accuracy = calculate_forecast_accuracy(series, fitted)

        # Decomposition
        decomposition = decompose_series(series, seasonal_period)

        # Matplotlib visualizations (frontend can ignore these)
        visualizations = {}
        try:
            visualizations['forecast_chart'] = create_forecast_chart(
                dates, series, fitted, forecast, forecast_dates,
                ci['lower'], ci['upper'], result['method']
            )
            visualizations['components_chart'] = create_components_chart(dates, decomposition)
            visualizations['accuracy_chart'] = create_accuracy_chart(series, fitted, dates)
            visualizations['seasonality_chart'] = create_seasonality_chart(
                seasonality['acf'], seasonal_period
            )
        except Exception:
            pass  # matplotlib failures should not block the response

        # Key insights
        insights = generate_key_insights(result, accuracy, seasonality, series, forecast)

        # Build forecast table
        forecast_table = []
        for i, (date, val) in enumerate(zip(forecast_dates, forecast)):
            forecast_table.append({
                'period': i + 1,
                'date': date.strftime('%Y-%m-%d'),
                'forecast': _to_native_type(val),
                'lower': _to_native_type(ci['lower'][i]),
                'upper': _to_native_type(ci['upper'][i])
            })

        # Summary
        summary = {
            'method': result['method'],
            'forecast_periods': request.forecast_periods,
            'seasonal_period': seasonal_period,
            'total_forecast': _to_native_type(np.sum(forecast)),
            'avg_forecast': _to_native_type(np.mean(forecast)),
            'historical_periods': len(series),
            'historical_total': _to_native_type(np.sum(series)),
            'historical_avg': _to_native_type(np.mean(series))
        }

        return {
            'success': True,
            'forecast': forecast_table,
            'accuracy': accuracy,
            'seasonality': seasonality,
            'model_params': {k: v for k, v in result.items() if k not in ['forecast', 'fitted']},
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Population forecast failed: {str(e)}")
