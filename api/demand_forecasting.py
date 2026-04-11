from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()


class DemandForecastingRequest(BaseModel):
    data: List[Dict[str, Any]]
    time_col: str
    value_col: str
    forecast_periods: int = 12
    confidence_level: float = 95


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=None):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except:
        return default


def detect_frequency(df: pd.DataFrame) -> Optional[str]:
    try:
        inferred = pd.infer_freq(df.index)
        if inferred:
            return inferred
        if len(df.index) >= 2:
            diffs = pd.Series(df.index).diff().dropna()
            median_diff = diffs.median()
            if median_diff <= pd.Timedelta(days=1):
                return 'D'
            elif median_diff <= pd.Timedelta(days=7):
                return 'W'
            elif median_diff <= pd.Timedelta(days=31):
                return 'MS'
            elif median_diff <= pd.Timedelta(days=92):
                return 'QS'
            else:
                return 'YS'
    except:
        pass
    return None


def get_seasonal_periods(freq: Optional[str]) -> int:
    freq_map = {'D': 7, 'W': 52, 'MS': 12, 'M': 12, 'QS': 4, 'Q': 4, 'YS': 1, 'Y': 1}
    return freq_map.get(freq, 12)


def fit_arima_model(series: pd.Series, seasonal_periods: int):
    orders_to_try = [
        ((1, 1, 1), (1, 1, 1, seasonal_periods)),
        ((1, 1, 1), (0, 1, 1, seasonal_periods)),
        ((2, 1, 2), (1, 1, 1, seasonal_periods)),
        ((1, 0, 1), (1, 0, 1, seasonal_periods)),
        ((0, 1, 1), (0, 1, 1, seasonal_periods)),
        ((1, 1, 0), (1, 1, 0, seasonal_periods)),
    ]
    best_model = None
    best_aic = np.inf
    
    for order, seasonal_order in orders_to_try:
        try:
            if len(series) < seasonal_order[3] * 2:
                seasonal_order = (0, 0, 0, 0)
            model = ARIMA(series, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
        except:
            continue
    
    if best_model is None:
        try:
            model = ARIMA(series, order=(1, 1, 1))
            best_model = model.fit()
        except:
            pass
    
    return best_model


def fit_exponential_smoothing(series: pd.Series, seasonal_periods: int):
    configs = []
    if len(series) >= seasonal_periods * 2:
        configs.extend([
            {'seasonal': 'add', 'seasonal_periods': seasonal_periods, 'trend': 'add'},
            {'seasonal': 'add', 'seasonal_periods': seasonal_periods, 'trend': None},
            {'seasonal': 'mul', 'seasonal_periods': seasonal_periods, 'trend': 'add'},
        ])
    configs.extend([
        {'seasonal': None, 'trend': 'add'},
        {'seasonal': None, 'trend': 'mul'},
        {'seasonal': None, 'trend': None},
    ])
    
    best_model = None
    best_aic = np.inf
    
    for config in configs:
        try:
            model = ExponentialSmoothing(
                series, seasonal=config['seasonal'],
                seasonal_periods=config.get('seasonal_periods'),
                trend=config['trend'], initialization_method='estimated'
            )
            fitted = model.fit(optimized=True)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
        except:
            continue
    
    return best_model


def calculate_forecast_accuracy(actual: pd.Series, fitted_values: pd.Series) -> Dict[str, float]:
    try:
        common_idx = actual.index.intersection(fitted_values.index)
        actual_aligned = actual.loc[common_idx]
        fitted_aligned = fitted_values.loc[common_idx]
        mask = ~(actual_aligned.isna() | fitted_aligned.isna())
        actual_clean = actual_aligned[mask]
        fitted_clean = fitted_aligned[mask]
        if len(actual_clean) == 0:
            return {}
        errors = actual_clean - fitted_clean
        rmse = np.sqrt(np.mean(errors ** 2))
        non_zero_mask = actual_clean != 0
        mape = np.mean(np.abs(errors[non_zero_mask] / actual_clean[non_zero_mask])) * 100 if non_zero_mask.any() else None
        mae = np.mean(np.abs(errors))
        return {'rmse': rmse, 'mape': mape, 'mae': mae}
    except:
        return {}


@router.post("/demand-forecasting")
async def demand_forecasting(request: DemandForecastingRequest):
    try:
        df = pd.DataFrame(request.data)
        time_col = request.time_col
        value_col = request.value_col
        forecast_periods = request.forecast_periods
        confidence_level = request.confidence_level / 100

        if time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{time_col}' not found")
        if value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{value_col}' not found")

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df = df.set_index(time_col).sort_index()
        series = df[value_col].astype(float).dropna()

        if len(series) < 10:
            raise HTTPException(status_code=400, detail=f"Insufficient data points: {len(series)}. Need at least 10.")

        freq = detect_frequency(df)
        if freq:
            try:
                series = series.asfreq(freq)
                series = series.fillna(method='ffill', limit=2)
            except:
                pass

        seasonal_periods = get_seasonal_periods(freq)
        arima_model = fit_arima_model(series, seasonal_periods)
        es_model = fit_exponential_smoothing(series, seasonal_periods)

        if arima_model is None and es_model is None:
            raise HTTPException(status_code=400, detail="Both ARIMA and Exponential Smoothing models failed to fit")

        arima_aic = arima_model.aic if arima_model else np.inf
        es_aic = es_model.aic if es_model else np.inf

        if arima_aic < es_aic and arima_model:
            best_model = arima_model
            best_model_name = 'ARIMA'
            best_aic = arima_aic
            best_bic = getattr(arima_model, 'bic', arima_aic)
        else:
            best_model = es_model
            best_model_name = 'Exponential Smoothing'
            best_aic = es_aic
            best_bic = getattr(es_model, 'bic', es_aic)

        alpha = 1 - confidence_level
        try:
            forecast_obj = best_model.get_forecast(steps=forecast_periods)
            forecast_mean = forecast_obj.predicted_mean
            forecast_ci = forecast_obj.conf_int(alpha=alpha)
            ci_lower = forecast_ci.iloc[:, 0]
            ci_upper = forecast_ci.iloc[:, 1]
        except AttributeError:
            forecast_mean = best_model.forecast(steps=forecast_periods)
            residuals_std = best_model.resid.std()
            z_score = 1.96 if confidence_level == 0.95 else 1.645 if confidence_level == 0.90 else 2.576
            ci_lower = forecast_mean - z_score * residuals_std
            ci_upper = forecast_mean + z_score * residuals_std

        last_date = series.index[-1]
        if freq:
            forecast_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                                          periods=forecast_periods, freq=freq)
        else:
            avg_diff = (series.index[-1] - series.index[0]) / (len(series) - 1)
            forecast_dates = pd.date_range(start=last_date + avg_diff, periods=forecast_periods, freq=avg_diff)

        forecast_records = []
        for i in range(forecast_periods):
            forecast_records.append({
                'forecast_date': forecast_dates[i].strftime('%Y-%m-%d'),
                'mean': safe_float(forecast_mean.values[i] if hasattr(forecast_mean, 'values') else forecast_mean[i]),
                'mean_ci_lower': safe_float(ci_lower.values[i] if hasattr(ci_lower, 'values') else ci_lower[i]),
                'mean_ci_upper': safe_float(ci_upper.values[i] if hasattr(ci_upper, 'values') else ci_upper[i])
            })

        try:
            fitted_values = best_model.fittedvalues
            accuracy = calculate_forecast_accuracy(series, fitted_values)
        except:
            accuracy = {}

        original_data = []
        for idx, val in series.items():
            original_data.append({
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'value': safe_float(val)
            })

        if len(series) >= 3:
            recent_avg = series.iloc[-3:].mean()
            earlier_avg = series.iloc[:3].mean()
            pct_change = (recent_avg - earlier_avg) / earlier_avg * 100 if earlier_avg != 0 else 0
            trend_direction = 'up' if pct_change > 5 else 'down' if pct_change < -5 else 'stable'
        else:
            trend_direction = 'stable'

        response = {
            'results': {
                'best_model': best_model_name,
                'aic': safe_float(best_aic),
                'bic': safe_float(best_bic),
                'mape': safe_float(accuracy.get('mape')),
                'rmse': safe_float(accuracy.get('rmse')),
                'mae': safe_float(accuracy.get('mae')),
                'forecast': forecast_records,
                'original_data': original_data,
                'model_comparison': {
                    'arima_aic': safe_float(arima_aic) if arima_aic != np.inf else None,
                    'es_aic': safe_float(es_aic) if es_aic != np.inf else None,
                },
                'seasonality_detected': seasonal_periods > 1 and len(series) >= seasonal_periods * 2,
                'trend_direction': trend_direction,
                'data_frequency': freq,
                'seasonal_periods': seasonal_periods,
                'confidence_level': confidence_level * 100
            }
        }

        return _to_native(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
