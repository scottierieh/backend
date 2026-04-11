from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class ForecastRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nPeriods: int = 104  # weeks of history
    seed: Optional[int] = None
    # Column mapping
    colDate: Optional[str] = None
    colTarget: Optional[str] = None
    colExogenous: Optional[List[str]] = None
    # Config
    forecastHorizon: int = 12
    testSize: int = 12
    frequency: str = 'W'  # W, D, M
    confidenceLevel: float = 0.95


def _to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception: return default


# ══════════════════════════════════════════════════════════════
# Data Generation
# ══════════════════════════════════════════════════════════════

def generate_demand(n_periods: int, freq: str = 'W', seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if freq == 'D':
        dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
        season_period = 365
        base = 500
    elif freq == 'M':
        dates = pd.date_range(start='2021-01-01', periods=n_periods, freq='MS')
        season_period = 12
        base = 15000
    else:  # W
        dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='W-MON')
        season_period = 52
        base = 3000

    t = np.arange(n_periods)

    # Trend: gentle upward
    trend = base + t * (base * 0.003)

    # Seasonality
    yearly = base * 0.15 * np.sin(2 * np.pi * t / season_period)
    if freq != 'M':
        weekly_effect = base * 0.05 * np.sin(2 * np.pi * t / 7) if freq == 'D' else 0
    else:
        weekly_effect = 0

    # Holiday spikes
    holidays = np.zeros(n_periods)
    for i in range(n_periods):
        week_of_year = dates[i].isocalendar()[1] if hasattr(dates[i], 'isocalendar') else 0
        if week_of_year in [48, 49, 50, 51]:  # Holiday season
            holidays[i] = base * rng.uniform(0.2, 0.5)
        elif week_of_year in [1, 2]:  # New year dip
            holidays[i] = -base * rng.uniform(0.05, 0.15)

    # Promotions (random events)
    promo = np.zeros(n_periods)
    promo_flag = np.zeros(n_periods)
    for i in range(n_periods):
        if rng.random() < 0.1:
            promo[i] = base * rng.uniform(0.1, 0.3)
            promo_flag[i] = 1

    # Noise
    noise = rng.normal(0, base * 0.08, n_periods)

    demand = trend + yearly + weekly_effect + holidays + promo + noise
    demand = np.maximum(demand, base * 0.3)  # floor

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'demand': np.round(demand, 0).astype(int),
        'promotion': promo_flag.astype(int),
        'temperature': np.round(15 + 10 * np.sin(2 * np.pi * t / season_period + 1) + rng.normal(0, 2, n_periods), 1),
    })

    return df


# ══════════════════════════════════════════════════════════════
# Forecasting Models
# ══════════════════════════════════════════════════════════════

def fit_sarima(train: np.ndarray, test_size: int, horizon: int, freq: str, conf: float):
    """Auto ARIMA via pmdarima."""
    import pmdarima as pm

    seasonal_period = {'D': 7, 'W': 52, 'M': 12}.get(freq, 52)
    # Cap seasonal period for performance
    m = min(seasonal_period, 52)

    model = pm.auto_arima(
        train,
        seasonal=True, m=m,
        stepwise=True, suppress_warnings=True,
        max_p=3, max_q=3, max_P=2, max_Q=2, max_d=2, max_D=1,
        error_action='ignore', trace=False,
        n_jobs=1,
    )

    # In-sample fitted
    fitted = model.predict_in_sample()

    # Forecast
    total_ahead = test_size + horizon
    fc, conf_int = model.predict(n_periods=total_ahead, return_conf_int=True, alpha=1 - conf)

    order = model.order
    seasonal_order = model.seasonal_order

    return {
        'name': 'SARIMA',
        'fitted': fitted.tolist(),
        'forecast': fc.tolist(),
        'conf_lower': conf_int[:, 0].tolist(),
        'conf_upper': conf_int[:, 1].tolist(),
        'params': {'order': str(order), 'seasonal_order': str(seasonal_order)},
    }


def fit_ets(train: np.ndarray, test_size: int, horizon: int, freq: str, conf: float):
    """Exponential Smoothing (Holt-Winters)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal_period = {'D': 7, 'W': 52, 'M': 12}.get(freq, 52)

    # Choose seasonal type based on data length
    if len(train) < 2 * seasonal_period:
        # Not enough data for full seasonality
        model = ExponentialSmoothing(train, trend='add', seasonal=None).fit(optimized=True)
        s_type = 'None'
    else:
        try:
            model = ExponentialSmoothing(
                train, trend='add', seasonal='add',
                seasonal_periods=seasonal_period,
            ).fit(optimized=True)
            s_type = f'add(p={seasonal_period})'
        except Exception:
            model = ExponentialSmoothing(train, trend='add', seasonal=None).fit(optimized=True)
            s_type = 'None'

    fitted = model.fittedvalues

    total_ahead = test_size + horizon
    fc = model.forecast(total_ahead)

    # Approximate confidence intervals using residual std
    residuals = train - fitted.values
    std = np.std(residuals)
    from scipy import stats
    z = stats.norm.ppf((1 + conf) / 2)
    steps = np.arange(1, total_ahead + 1)
    margin = z * std * np.sqrt(steps)

    return {
        'name': 'Exp Smoothing',
        'fitted': fitted.tolist(),
        'forecast': fc.tolist(),
        'conf_lower': (fc.values - margin).tolist(),
        'conf_upper': (fc.values + margin).tolist(),
        'params': {'seasonal': s_type, 'aic': safe_float(model.aic)},
    }


def fit_xgboost(train: np.ndarray, test_size: int, horizon: int, freq: str, conf: float,
                exog_train=None, exog_full=None):
    """XGBoost with lag features."""
    import xgboost as xgb

    seasonal_period = {'D': 7, 'W': 52, 'M': 12}.get(freq, 52)

    def create_features(series, exog=None):
        n = len(series)
        X = []
        for i in range(n):
            feats = []
            # Lags
            for lag in [1, 2, 3, 4, 8, 12]:
                feats.append(series[i - lag] if i >= lag else series[0])
            # Rolling stats
            for window in [4, 8, 12]:
                start = max(0, i - window)
                feats.append(np.mean(series[start:i + 1]))
                feats.append(np.std(series[start:i + 1]) if i > start else 0)
            # Time features
            feats.append(i % seasonal_period)  # season position
            feats.append(i)  # trend index
            # Exogenous
            if exog is not None and i < len(exog):
                feats.extend(exog[i].tolist() if hasattr(exog[i], 'tolist') else [float(exog[i])])
            X.append(feats)
        return np.array(X)

    X_train = create_features(train, exog_train)
    y_train = train

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)

    # Fitted values
    fitted = model.predict(X_train)

    # Recursive forecast
    total_ahead = test_size + horizon
    extended = list(train)
    forecasts = []

    for step in range(total_ahead):
        feats = []
        n_ext = len(extended)
        for lag in [1, 2, 3, 4, 8, 12]:
            feats.append(extended[n_ext - lag] if n_ext >= lag else extended[0])
        for window in [4, 8, 12]:
            start = max(0, n_ext - window)
            feats.append(np.mean(extended[start:]))
            feats.append(np.std(extended[start:]) if n_ext > start + 1 else 0)
        feats.append((n_ext) % seasonal_period)
        feats.append(n_ext)
        if exog_full is not None:
            idx = len(train) + step
            if idx < len(exog_full):
                row = exog_full[idx]
                feats.extend(row.tolist() if hasattr(row, 'tolist') else [float(row)])
            else:
                feats.extend([0] * (exog_full.shape[1] if exog_full.ndim > 1 else 1))

        pred = float(model.predict(np.array([feats]))[0])
        forecasts.append(pred)
        extended.append(pred)

    # Confidence intervals from training residuals
    residuals = train - fitted
    std = np.std(residuals)
    from scipy import stats
    z = stats.norm.ppf((1 + conf) / 2)
    steps_arr = np.arange(1, total_ahead + 1)
    margin = z * std * np.sqrt(np.log(steps_arr + 1))

    fc_arr = np.array(forecasts)

    return {
        'name': 'XGBoost',
        'fitted': fitted.tolist(),
        'forecast': fc_arr.tolist(),
        'conf_lower': (fc_arr - margin).tolist(),
        'conf_upper': (fc_arr + margin).tolist(),
        'params': {'n_estimators': 200, 'max_depth': 4},
    }


# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════

def calc_metrics(actual: np.ndarray, predicted: np.ndarray):
    mask = actual != 0
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else 0
    return {'mae': safe_float(mae), 'rmse': safe_float(rmse), 'mape': safe_float(mape)}


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/demand-forecast")
async def demand_forecast(request: ForecastRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_demand(request.nPeriods, request.frequency, request.seed)
            col_date = 'date'
            col_target = 'demand'
            exog_cols = ['promotion', 'temperature']
        else:
            df = pd.DataFrame(request.data)
            col_date = request.colDate or next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'week' in c.lower() or 'month' in c.lower()), None)
            col_target = request.colTarget or next((c for c in df.columns if 'demand' in c.lower() or 'sales' in c.lower() or 'quantity' in c.lower() or 'revenue' in c.lower() or 'orders' in c.lower()), None)

            if not col_target:
                raise HTTPException(status_code=400, detail="Cannot find target column (demand/sales/orders).")

            if request.colExogenous:
                exog_cols = request.colExogenous
            else:
                exog_cols = [c for c in df.columns if c not in [col_date, col_target] and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        df[col_target] = pd.to_numeric(df[col_target], errors='coerce')
        df = df.dropna(subset=[col_target])

        if col_date and col_date in df.columns:
            df[col_date] = pd.to_datetime(df[col_date])
            df = df.sort_values(col_date)

        n = len(df)
        if n < 30:
            raise HTTPException(status_code=400, detail=f"Need >=30 periods. Got {n}.")

        y = df[col_target].values.astype(float)
        dates = df[col_date].dt.strftime('%Y-%m-%d').tolist() if col_date and col_date in df.columns else [str(i) for i in range(n)]

        test_size = min(request.testSize, n // 4)
        horizon = request.forecastHorizon
        freq = request.frequency
        conf = request.confidenceLevel

        train = y[:n - test_size]
        test = y[n - test_size:]

        # Exogenous
        exog_train = None
        exog_full = None
        valid_exog = [c for c in exog_cols if c in df.columns]
        if valid_exog:
            exog_data = df[valid_exog].values.astype(float)
            exog_train = exog_data[:n - test_size]
            exog_full = exog_data

        # ── 2. Fit Models ──
        models = {}
        errors = {}

        # SARIMA
        try:
            models['SARIMA'] = fit_sarima(train, test_size, horizon, freq, conf)
        except Exception as e:
            errors['SARIMA'] = str(e)

        # ETS
        try:
            models['Exp Smoothing'] = fit_ets(train, test_size, horizon, freq, conf)
        except Exception as e:
            errors['Exp Smoothing'] = str(e)

        # XGBoost
        try:
            models['XGBoost'] = fit_xgboost(train, test_size, horizon, freq, conf, exog_train, exog_full)
        except Exception as e:
            errors['XGBoost'] = str(e)

        if not models:
            raise HTTPException(status_code=500, detail=f"All models failed: {errors}")

        # ── 3. Evaluate on Test Set ──
        model_metrics = {}
        for name, m in models.items():
            fc = np.array(m['forecast'][:test_size])
            if len(fc) == len(test):
                metrics = calc_metrics(test, fc)
                model_metrics[name] = metrics
            else:
                model_metrics[name] = {'mae': 999999, 'rmse': 999999, 'mape': 999999}

        # Best model by MAPE
        best_name = min(model_metrics, key=lambda k: model_metrics[k]['mape'])

        # ── 4. Ensemble (1/MAPE weighted) ──
        weights = {}
        total_inv_mape = 0
        for name, met in model_metrics.items():
            inv = 1.0 / (met['mape'] + 1e-6)
            weights[name] = inv
            total_inv_mape += inv
        for name in weights:
            weights[name] /= total_inv_mape

        ensemble_forecast = np.zeros(test_size + horizon)
        ensemble_lower = np.zeros(test_size + horizon)
        ensemble_upper = np.zeros(test_size + horizon)
        for name, m in models.items():
            w = weights.get(name, 0)
            fc = np.array(m['forecast'][:test_size + horizon])
            lo = np.array(m['conf_lower'][:test_size + horizon])
            up = np.array(m['conf_upper'][:test_size + horizon])
            pad = test_size + horizon - len(fc)
            if pad > 0:
                fc = np.pad(fc, (0, pad), mode='edge')
                lo = np.pad(lo, (0, pad), mode='edge')
                up = np.pad(up, (0, pad), mode='edge')
            ensemble_forecast += w * fc
            ensemble_lower += w * lo
            ensemble_upper += w * up

        ens_test = ensemble_forecast[:test_size]
        ens_metrics = calc_metrics(test, ens_test)
        model_metrics['Ensemble'] = ens_metrics
        if ens_metrics['mape'] < model_metrics[best_name]['mape']:
            best_name = 'Ensemble'

        # ── 5. Build Charts ──

        # Actual + fitted overlay
        fit_chart = []
        for i in range(n):
            entry: Dict[str, Any] = {'date': dates[i], 'actual': safe_float(y[i])}
            for name, m in models.items():
                if i < len(m['fitted']):
                    entry[name] = safe_float(m['fitted'][i])
            if i >= n - test_size:
                idx = i - (n - test_size)
                entry['Ensemble'] = safe_float(ensemble_forecast[idx])
            fit_chart.append(entry)

        # Future forecast
        future_chart = []
        last_date = pd.to_datetime(dates[-1]) if col_date else pd.Timestamp('2025-01-01')
        freq_offset = {'D': pd.Timedelta(days=1), 'W': pd.Timedelta(weeks=1), 'M': pd.DateOffset(months=1)}.get(freq, pd.Timedelta(weeks=1))

        for step in range(horizon):
            fut_date = last_date + freq_offset * (step + 1)
            entry: Dict[str, Any] = {'date': fut_date.strftime('%Y-%m-%d')}
            fc_idx = test_size + step
            for name, m in models.items():
                if fc_idx < len(m['forecast']):
                    entry[name] = safe_float(m['forecast'][fc_idx])
                    entry[f'{name}_lower'] = safe_float(m['conf_lower'][fc_idx])
                    entry[f'{name}_upper'] = safe_float(m['conf_upper'][fc_idx])
            if fc_idx < len(ensemble_forecast):
                entry['Ensemble'] = safe_float(ensemble_forecast[fc_idx])
                entry['Ensemble_lower'] = safe_float(ensemble_lower[fc_idx])
                entry['Ensemble_upper'] = safe_float(ensemble_upper[fc_idx])
            future_chart.append(entry)

        # Model comparison
        comparison = []
        for name, met in model_metrics.items():
            comparison.append({
                'model': name,
                'mae': met['mae'],
                'rmse': met['rmse'],
                'mape': met['mape'],
                'is_best': name == best_name,
                'weight': safe_float(weights.get(name, 0) * 100) if name != 'Ensemble' else 0,
            })
        comparison.sort(key=lambda x: x['mape'])

        # Residuals (best model)
        residual_chart = []
        if best_name in models and best_name != 'Ensemble':
            fitted_vals = models[best_name]['fitted']
            for i in range(min(len(fitted_vals), len(train))):
                residual_chart.append({
                    'date': dates[i],
                    'residual': safe_float(train[i] - fitted_vals[i]),
                })

        # Decomposition (simple)
        decomp = []
        if n >= 20:
            from scipy.ndimage import uniform_filter1d
            trend_line = uniform_filter1d(y, size=min(13, n // 3))
            seasonal = y - trend_line
            for i in range(n):
                decomp.append({
                    'date': dates[i],
                    'actual': safe_float(y[i]),
                    'trend': safe_float(trend_line[i]),
                    'seasonal': safe_float(seasonal[i]),
                })

        # ── Response ──
        results = {
            'n_periods': n,
            'train_size': len(train),
            'test_size': test_size,
            'forecast_horizon': horizon,
            'frequency': freq,
            'best_model': best_name,
            'columns_used': {
                'date': col_date, 'target': col_target,
                'exogenous': valid_exog if valid_exog else None,
            },
            'summary': {
                'mean_demand': safe_float(y.mean()),
                'std_demand': safe_float(y.std()),
                'min_demand': safe_float(y.min()),
                'max_demand': safe_float(y.max()),
                'trend_direction': 'Up' if y[-1] > y[0] else 'Down',
                'trend_pct': safe_float((y[-12:].mean() - y[:12].mean()) / y[:12].mean() * 100),
            },
            'model_metrics': model_metrics,
            'ensemble_weights': {k: safe_float(v * 100) for k, v in weights.items()},
            'errors': errors,
            'charts': {
                'fit': fit_chart,
                'future': future_chart,
                'comparison': comparison,
                'residuals': residual_chart,
                'decomposition': decomp,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
