from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()


class ForecastEvaluationRequest(BaseModel):
    data: List[Dict[str, Any]]
    timeCol: str
    valueCol: str
    rolling_origins: int = 3            # how many rolling evaluation origins (1 = single split)
    rolling_window_type: str = 'expanding'  # 'expanding' | 'rolling'
    rolling_window_size: Optional[int] = None  # fixed window for rolling (None = auto)
    test_size: Optional[int] = None     # override auto test size per origin


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
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


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality=1):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae_pred = mean_absolute_error(y_true, y_pred)
    train_vals = np.array(y_train).flatten()
    if len(train_vals) <= seasonality:
        return None
    naive_forecast = train_vals[:-seasonality]
    actual_train = train_vals[seasonality:]
    mae_naive = mean_absolute_error(actual_train, naive_forecast)
    if mae_naive == 0:
        return None
    return mae_pred / mae_naive


def evaluate_sarima(train, test, seasonal_period=12):
    try:
        if len(train) < seasonal_period * 2:
            model = SARIMAX(train, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period), 
                          enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False, maxiter=100)
        forecast_result = model_fit.get_forecast(steps=len(test))
        forecast = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int()
        test_vals = np.array(test).flatten()
        rmse = np.sqrt(mean_squared_error(test_vals, forecast))
        mae = mean_absolute_error(test_vals, forecast)
        mape = mean_absolute_percentage_error(test_vals, forecast)
        mase = mean_absolute_scaled_error(test_vals, forecast, train, seasonality=seasonal_period)
        ci_lower = conf_int.iloc[:, 0].values
        ci_upper = conf_int.iloc[:, 1].values
        coverage = np.mean((test_vals >= ci_lower) & (test_vals <= ci_upper)) * 100
        return {"Method": "SARIMA", "RMSE": safe_float(rmse), "MAE": safe_float(mae), 
                "MAPE (%)": safe_float(mape), "MASE": safe_float(mase), "Coverage (95% PI)": safe_float(coverage)}
    except Exception as e:
        return {"Method": "SARIMA", "RMSE": None, "MAE": None, "MAPE (%)": None, "MASE": None, "Coverage (95% PI)": None, "error": str(e)}


def evaluate_holt_winters(train, test, seasonal_period=12):
    try:
        train_vals = np.array(train).flatten()
        test_vals = np.array(test).flatten()
        if len(train_vals) < seasonal_period * 2:
            raise ValueError(f"Requires at least {seasonal_period * 2} data points")
        train_series = pd.Series(train_vals)
        model = ExponentialSmoothing(train_series, seasonal_periods=seasonal_period, trend='add', seasonal='add', 
                                     initialization_method="estimated", use_boxcox=False)
        model_fit = model.fit(optimized=True)
        forecast = model_fit.forecast(steps=len(test_vals))
        forecast_vals = np.array(forecast).flatten()
        residuals = model_fit.fittedvalues.values - train_vals
        std_resid = np.std(residuals)
        ci_lower = forecast_vals - 1.96 * std_resid
        ci_upper = forecast_vals + 1.96 * std_resid
        rmse = np.sqrt(mean_squared_error(test_vals, forecast_vals))
        mae = mean_absolute_error(test_vals, forecast_vals)
        mape = mean_absolute_percentage_error(test_vals, forecast_vals)
        mase = mean_absolute_scaled_error(test_vals, forecast_vals, train_vals, seasonality=seasonal_period)
        coverage = np.mean((test_vals >= ci_lower) & (test_vals <= ci_upper)) * 100
        return {"Method": "Holt-Winters Add", "RMSE": safe_float(rmse), "MAE": safe_float(mae), 
                "MAPE (%)": safe_float(mape), "MASE": safe_float(mase), "Coverage (95% PI)": safe_float(coverage)}
    except Exception as e:
        return {"Method": "Holt-Winters Add", "RMSE": None, "MAE": None, "MAPE (%)": None, "MASE": None, "Coverage (95% PI)": None, "error": str(e)}


def evaluate_simple_exp_smoothing(train, test):
    try:
        train_vals = np.array(train).flatten()
        test_vals = np.array(test).flatten()
        train_series = pd.Series(train_vals)
        model = SimpleExpSmoothing(train_series, initialization_method="estimated")
        model_fit = model.fit(optimized=True)
        forecast = model_fit.forecast(steps=len(test_vals))
        forecast_vals = np.array(forecast).flatten()
        residuals = model_fit.fittedvalues.values - train_vals
        std_resid = np.std(residuals)
        ci_lower = forecast_vals - 1.96 * std_resid
        ci_upper = forecast_vals + 1.96 * std_resid
        rmse = np.sqrt(mean_squared_error(test_vals, forecast_vals))
        mae = mean_absolute_error(test_vals, forecast_vals)
        mape = mean_absolute_percentage_error(test_vals, forecast_vals)
        mase = mean_absolute_scaled_error(test_vals, forecast_vals, train_vals, seasonality=1)
        coverage = np.mean((test_vals >= ci_lower) & (test_vals <= ci_upper)) * 100
        return {"Method": "Simple Exp Smoothing", "RMSE": safe_float(rmse), "MAE": safe_float(mae), 
                "MAPE (%)": safe_float(mape), "MASE": safe_float(mase), "Coverage (95% PI)": safe_float(coverage)}
    except Exception as e:
        return {"Method": "Simple Exp Smoothing", "RMSE": None, "MAE": None, "MAPE (%)": None, "MASE": None, "Coverage (95% PI)": None, "error": str(e)}


def evaluate_holt_linear(train, test):
    try:
        train_vals = np.array(train).flatten()
        test_vals = np.array(test).flatten()
        train_series = pd.Series(train_vals)
        model = Holt(train_series, initialization_method="estimated")
        model_fit = model.fit(optimized=True)
        forecast = model_fit.forecast(steps=len(test_vals))
        forecast_vals = np.array(forecast).flatten()
        residuals = model_fit.fittedvalues.values - train_vals
        std_resid = np.std(residuals)
        ci_lower = forecast_vals - 1.96 * std_resid
        ci_upper = forecast_vals + 1.96 * std_resid
        rmse = np.sqrt(mean_squared_error(test_vals, forecast_vals))
        mae = mean_absolute_error(test_vals, forecast_vals)
        mape = mean_absolute_percentage_error(test_vals, forecast_vals)
        mase = mean_absolute_scaled_error(test_vals, forecast_vals, train_vals, seasonality=1)
        coverage = np.mean((test_vals >= ci_lower) & (test_vals <= ci_upper)) * 100
        return {"Method": "Holt's Linear", "RMSE": safe_float(rmse), "MAE": safe_float(mae), 
                "MAPE (%)": safe_float(mape), "MASE": safe_float(mase), "Coverage (95% PI)": safe_float(coverage)}
    except Exception as e:
        return {"Method": "Holt's Linear", "RMSE": None, "MAE": None, "MAPE (%)": None, "MASE": None, "Coverage (95% PI)": None, "error": str(e)}


def evaluate_naive_seasonal(train, test, seasonal_period=12):
    try:
        train_vals = np.array(train).flatten()
        test_vals = np.array(test).flatten()
        if len(train_vals) < seasonal_period:
            raise ValueError(f"Not enough data for {seasonal_period}-period seasonal naive forecast")
        naive_base = train_vals[-seasonal_period:]
        if len(test_vals) == seasonal_period:
            naive_forecast = naive_base
        else:
            repeats = int(np.ceil(len(test_vals) / seasonal_period))
            naive_forecast = np.tile(naive_base, repeats)[:len(test_vals)]
        rmse = np.sqrt(mean_squared_error(test_vals, naive_forecast))
        mae = mean_absolute_error(test_vals, naive_forecast)
        mape = mean_absolute_percentage_error(test_vals, naive_forecast)
        mase = mean_absolute_scaled_error(test_vals, naive_forecast, train_vals, seasonality=seasonal_period)
        return {"Method": "Naive Seasonal", "RMSE": safe_float(rmse), "MAE": safe_float(mae), 
                "MAPE (%)": safe_float(mape), "MASE": safe_float(mase), "Coverage (95% PI)": None}
    except Exception as e:
        return {"Method": "Naive Seasonal", "RMSE": None, "MAE": None, "MAPE (%)": None, "MASE": None, "Coverage (95% PI)": None, "error": str(e)}



# ══════════════════════════════════════════════════════════════════
# ① Additional baseline models: Drift + Naive (non-seasonal)
# ══════════════════════════════════════════════════════════════════

def evaluate_naive(train, test):
    """
    Random-walk (naïve) forecast: last observed value repeated h steps.
    Benchmark for non-seasonal series.
    """
    try:
        train_vals = np.array(train).flatten()
        test_vals  = np.array(test).flatten()
        h          = len(test_vals)
        forecast   = np.full(h, train_vals[-1])
        rmse  = float(np.sqrt(np.mean((test_vals - forecast) ** 2)))
        mae   = float(np.mean(np.abs(test_vals - forecast)))
        mape  = mean_absolute_percentage_error(test_vals, forecast)
        mase  = mean_absolute_scaled_error(test_vals, forecast, train_vals, seasonality=1)
        return {"Method": "Naïve", "RMSE": safe_float(rmse), "MAE": safe_float(mae),
                "MAPE (%)": safe_float(mape), "MASE": safe_float(mase),
                "Coverage (95% PI)": None}
    except Exception as e:
        return {"Method": "Naïve", "RMSE": None, "MAE": None,
                "MAPE (%)": None, "MASE": None, "Coverage (95% PI)": None, "error": str(e)}


def evaluate_drift(train, test):
    """
    Drift forecast: extrapolate the line between the first and last
    training observation.  Equivalent to random walk with drift.
    Ref: Hyndman & Athanasopoulos (2021) §3.3.
    """
    try:
        train_vals = np.array(train).flatten()
        test_vals  = np.array(test).flatten()
        m  = len(train_vals)
        h  = len(test_vals)
        if m < 2:
            raise ValueError("Need ≥ 2 training observations for Drift")
        slope    = (train_vals[-1] - train_vals[0]) / (m - 1)
        forecast = train_vals[-1] + slope * np.arange(1, h + 1)
        rmse  = float(np.sqrt(np.mean((test_vals - forecast) ** 2)))
        mae   = float(np.mean(np.abs(test_vals - forecast)))
        mape  = mean_absolute_percentage_error(test_vals, forecast)
        mase  = mean_absolute_scaled_error(test_vals, forecast, train_vals, seasonality=1)
        return {"Method": "Drift", "RMSE": safe_float(rmse), "MAE": safe_float(mae),
                "MAPE (%)": safe_float(mape), "MASE": safe_float(mase),
                "Coverage (95% PI)": None}
    except Exception as e:
        return {"Method": "Drift", "RMSE": None, "MAE": None,
                "MAPE (%)": None, "MASE": None, "Coverage (95% PI)": None, "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# ② Rolling-origin evaluation engine
# ══════════════════════════════════════════════════════════════════

def _rolling_origin_evaluate(
    series: pd.Series,
    seasonal_period: int,
    n_origins: int,
    window_type: str,
    window_size: int,
    h: int,
) -> dict:
    """
    Evaluate all models across multiple forecast origins.

    For each origin k = 1..n_origins:
      expanding : train = series[:n - (n_origins-k)*h]
      rolling   : train = series[max(0, end-window_size):end]
      test      = series[end : end+h]

    Each model is called on every origin.  Per-model, per-origin errors
    are aggregated by taking the mean across origins.

    Returns
    -------
    dict:
      per_origin   : list of {origin, train_end_date, errors per model}
      aggregated   : {model: {RMSE, MAE, MAPE, MASE, Coverage, n_origins}}
    """
    n       = len(series)
    origins = []

    # Evaluator registry: name → callable(train, test[, period])
    def _run(evaluator, train, test):
        return evaluator(train.values, test.values)
    def _run_sp(evaluator, train, test):
        return evaluator(train.values, test.values, seasonal_period)

    registry = [
        ("SARIMA",              lambda tr, te: evaluate_sarima(tr.values, te.values, seasonal_period)),
        ("Holt-Winters Add",    lambda tr, te: evaluate_holt_winters(tr.values, te.values, seasonal_period)),
        ("Simple Exp Smooth",   lambda tr, te: evaluate_simple_exp_smoothing(tr.values, te.values)),
        ("Holt's Linear",       lambda tr, te: evaluate_holt_linear(tr.values, te.values)),
        ("Seasonal Naïve",      lambda tr, te: evaluate_naive_seasonal(tr.values, te.values, seasonal_period)),
        ("Naïve",               lambda tr, te: evaluate_naive(tr.values, te.values)),
        ("Drift",               lambda tr, te: evaluate_drift(tr.values, te.values)),
    ]
    metric_keys = ["RMSE", "MAE", "MAPE (%)", "MASE", "Coverage (95% PI)"]

    # Accumulator: {method: {metric: [values]}}
    accum = {name: {m: [] for m in metric_keys} for name, _ in registry}

    for k in range(n_origins):
        end = n - (n_origins - 1 - k) * h
        if end + h > n:
            end = n - h
        if window_type == 'rolling' and window_size:
            start = max(0, end - window_size)
        else:
            start = 0
        train = series.iloc[start:end]
        test  = series.iloc[end:end + h]

        if len(train) < max(12, seasonal_period * 2) or len(test) == 0:
            continue

        origin_errors = {}
        for name, fn in registry:
            res = fn(train, test)
            origin_errors[name] = {m: res.get(m) for m in metric_keys}
            for m in metric_keys:
                v = res.get(m)
                if v is not None:
                    accum[name][m].append(v)

        origins.append({
            'origin':         k + 1,
            'train_size':     len(train),
            'test_size':      len(test),
            'train_end_date': str(train.index[-1])[:10],
            'test_start_date': str(test.index[0])[:10],
            'errors':         origin_errors,
        })

    # Aggregate: mean ± std across origins
    aggregated = []
    for name, _ in registry:
        row = {'Method': name, 'n_origins': len(origins)}
        for m in metric_keys:
            vals = accum[name][m]
            row[m]              = round(float(np.mean(vals)),  4) if vals else None
            row[f'{m}_std']     = round(float(np.std(vals)),   4) if len(vals) > 1 else None
            row[f'{m}_values']  = [round(v, 4) for v in vals]
        aggregated.append(row)

    return {'per_origin': origins, 'aggregated': aggregated}


# ══════════════════════════════════════════════════════════════════
# ③ Ranking
# ══════════════════════════════════════════════════════════════════

def _rank_models(aggregated: list) -> list:
    """
    Rank models on 4 metrics (RMSE, MAE, MAPE, MASE).
    Score = mean rank across available metrics (lower = better).
    Returns list sorted by composite_rank ascending, with rank fields added.
    """
    metric_keys = ["RMSE", "MAE", "MAPE (%)", "MASE"]
    valid = [r for r in aggregated if any(r.get(m) is not None for m in metric_keys)]

    # Per-metric rank (1 = best)
    for m in metric_keys:
        vals = [(i, r[m]) for i, r in enumerate(valid) if r.get(m) is not None]
        vals.sort(key=lambda x: x[1])
        for rank, (i, _) in enumerate(vals, 1):
            valid[i][f'rank_{m}'] = rank

    # Composite: mean of available ranks
    for r in valid:
        ranks = [r[f'rank_{m}'] for m in metric_keys if f'rank_{m}' in r]
        r['composite_rank']  = round(float(np.mean(ranks)), 2) if ranks else None
        r['rank_n_metrics']  = len(ranks)

    valid.sort(key=lambda x: (x['composite_rank'] is None, x.get('composite_rank', 9999)))
    for i, r in enumerate(valid, 1):
        r['overall_rank'] = i
        r['is_best']      = (i == 1)

    # Mark best per metric
    for m in metric_keys:
        best_val = min((r[m] for r in valid if r.get(m) is not None), default=None)
        for r in valid:
            r[f'is_best_{m}'] = (r.get(m) is not None and r[m] == best_val)

    return valid


@router.post("/forecast-evaluation")
async def forecast_evaluation(request: ForecastEvaluationRequest):
    try:
        df        = pd.DataFrame(request.data)
        time_col  = request.timeCol
        value_col = request.valueCol
        n_origins = max(1, int(request.rolling_origins))
        win_type  = request.rolling_window_type.lower().strip()
        if win_type not in ('expanding', 'rolling'):
            win_type = 'expanding'

        if time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{time_col}' not found")
        if value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{value_col}' not found")

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])
        series = df.set_index(time_col)[value_col].sort_index()

        n = len(series)
        if n < 24:
            raise HTTPException(status_code=400,
                detail=f"At least 24 data points required. Found: {n}")

        # ── Infer frequency + seasonal period ────────────────────────────────
        seasonal_period = 12
        try:
            freq = pd.infer_freq(series.index)
            if freq:
                if 'D' in freq or 'B' in freq: seasonal_period = 7
                elif 'W' in freq:              seasonal_period = 52
                elif 'Q' in freq:              seasonal_period = 4
                elif 'A' in freq or 'Y' in freq: seasonal_period = 1
        except Exception:
            pass

        # ── Test horizon per origin ───────────────────────────────────────────
        h = request.test_size if request.test_size else min(seasonal_period,
                                                             max(1, int(n * 0.15 / n_origins)))
        h = max(1, h)

        # Rolling window size (for rolling mode)
        win_size = request.rolling_window_size or max(seasonal_period * 3, n // 2)
        win_size = max(seasonal_period * 2, int(win_size))

        # Guard: enough data for origins
        needed = h * n_origins + seasonal_period * 2
        if n < needed:
            raise HTTPException(status_code=400,
                detail=f"Need ≥{needed} observations for {n_origins} origins × h={h}. Have {n}.")

        # ── Rolling-origin evaluation ─────────────────────────────────────────
        roll = _rolling_origin_evaluate(
            series, seasonal_period, n_origins, win_type, win_size, h
        )

        # ── Ranking ───────────────────────────────────────────────────────────
        ranked = _rank_models(roll['aggregated'])
        best   = next((r for r in ranked if r.get('is_best')), None)

        return _to_native({
            'results':        ranked,            # backward compat key
            'ranked':         ranked,
            'per_origin':     roll['per_origin'],
            'best_model':     best['Method'] if best else None,
            'evaluation_config': {
                'n_origins':          n_origins,
                'window_type':        win_type,
                'window_size':        win_size if win_type == 'rolling' else None,
                'horizon_h':          h,
                'seasonal_period':    seasonal_period,
                'n_observations':     n,
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
