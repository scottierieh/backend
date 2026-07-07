"""
Generic LSTM Forecasting Router for FastAPI
Univariate time-series forecasting with an LSTM network — not tied to any
specific business domain (works for demand, sales, traffic, sensor data, etc.)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class LSTMForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    value_col: str
    forecast_periods: int = 12
    window_size: int = 12  # look-back window
    lstm_units: int = 50
    epochs: int = 100
    batch_size: int = 16
    test_size: float = 0.2
    random_state: int = 42


def _to_native_type(obj):
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
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def _make_sequences(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def fit_lstm_forecaster(values: np.ndarray, window_size: int, lstm_units: int,
                         epochs: int, batch_size: int, test_size: float, random_state: int):
    import tensorflow as tf
    from tensorflow import keras

    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(random_state)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    X, y = _make_sequences(scaled, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_test) == 0:
        raise ValueError("Test set is empty — reduce window_size or test_size.")

    model = keras.Sequential([
        keras.layers.LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=(window_size, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(max(lstm_units // 2, 8), activation='tanh'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        verbose=0, callbacks=[early_stop], shuffle=False
    )

    y_pred_test_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_train_scaled = model.predict(X_train, verbose=0).flatten()

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_test_actual = scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_pred_train_actual = scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()

    return {
        'model': model, 'scaler': scaler, 'history': history,
        'y_test_actual': y_test_actual, 'y_pred_test_actual': y_pred_test_actual,
        'y_train_actual': y_train_actual, 'y_pred_train_actual': y_pred_train_actual,
        'scaled_values': scaled, 'n_train': len(X_train), 'n_test': len(X_test)
    }


def recursive_forecast(model, scaler, scaled_values: np.ndarray, window_size: int, forecast_periods: int) -> np.ndarray:
    last_window = scaled_values[-window_size:].tolist()
    future_scaled = []
    for _ in range(forecast_periods):
        x_input = np.array(last_window[-window_size:]).reshape((1, window_size, 1))
        next_val = model.predict(x_input, verbose=0)[0, 0]
        future_scaled.append(next_val)
        last_window.append(next_val)
    return scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()


def generate_loss_plot(history) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['loss'], color='#3b82f6', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss (scaled)', fontsize=11)
    ax.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_forecast_plot(dates, values, value_col: str, test_dates, y_pred_test_actual,
                            future_dates, future_values, test_r2: float) -> str:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(dates, values, label='Historical', color='#3b82f6', linewidth=1.8)
    ax.plot(test_dates, y_pred_test_actual, label='Test Predictions', color='#f59e0b', linewidth=1.8)
    ax.plot(future_dates, future_values, label='Future Forecast', color='#22c55e', linewidth=1.8, linestyle='--', marker='o', markersize=3)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel(value_col, fontsize=11)
    ax.set_title(f'LSTM Forecast (Test R² = {test_r2:.3f})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(train_r2: float, test_r2: float, test_rmse: float, epochs_run: int,
                             epochs_requested: int, n_obs: int) -> Dict[str, Any]:
    key_insights = []
    r2_diff = train_r2 - test_r2

    if test_r2 > 0.8 and r2_diff < 0.2:
        status, desc = 'positive', 'Good Fit — train and test R² are both high and close together, suggesting the model generalizes well to unseen periods.'
    elif train_r2 > 0.7 and r2_diff > 0.3:
        status, desc = 'warning', 'Overfitting Warning — the model fits training data much better than test data. Consider a shorter window, fewer LSTM units, or stronger dropout.'
    elif train_r2 < 0.5 and test_r2 < 0.5:
        status, desc = 'warning', 'Underfitting Possible — both train and test R² are low. Try more LSTM units, a longer window_size, or more training epochs.'
    else:
        status, desc = 'neutral', 'Moderate performance — review RMSE/MAE alongside the forecast plot before relying on this model.'

    key_insights.append({'title': 'Model Fit', 'description': desc, 'status': status})
    key_insights.append({
        'title': 'Performance Metrics',
        'description': f'Test R² = {test_r2:.3f}, Test RMSE = {test_rmse:.3f}, Train R² = {train_r2:.3f}',
        'status': 'neutral'
    })

    if epochs_run < epochs_requested:
        key_insights.append({
            'title': 'Early Stopping Triggered',
            'description': f'Training stopped after {epochs_run} of {epochs_requested} requested epochs — the loss curve plateaued.',
            'status': 'positive'
        })

    key_insights.append({
        'title': 'Forecast Uncertainty Compounds',
        'description': 'Multi-step forecasts beyond the observed data are generated recursively (each prediction feeds the next step), so uncertainty compounds — treat forecasts far beyond the training range with caution.',
        'status': 'warning'
    })

    if n_obs < 200:
        key_insights.append({
            'title': 'Limited Training Data',
            'description': f'With only {n_obs} observations, LSTM forecasts may be unstable across re-runs; classical time-series models (ARIMA, exponential smoothing) may be more data-efficient.',
            'status': 'warning'
        })

    return {
        'key_insights': key_insights,
        'recommendation': 'Compare against a simple baseline (e.g. naive last-value or ARIMA) to confirm the LSTM\'s added complexity is actually paying off.'
    }


@router.post("/lstm-forecast")
async def run_lstm_forecast(request: LSTMForecastRequest) -> Dict[str, Any]:
    """
    Forecast a generic univariate time series using an LSTM network.
    """
    try:
        data = request.data
        date_col = request.date_col
        value_col = request.value_col

        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")

        df = pd.DataFrame(data)
        if date_col not in df.columns or value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Columns not found: {date_col}, {value_col}")

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col]).sort_values(date_col).reset_index(drop=True)

        window_size = request.window_size
        if len(df) < window_size + 10:
            raise HTTPException(status_code=400, detail=f"Not enough data. Need at least {window_size + 10} observations, got {len(df)}.")

        values = df[value_col].values.astype(float)

        result = fit_lstm_forecaster(
            values, window_size, request.lstm_units, request.epochs,
            request.batch_size, request.test_size, request.random_state
        )

        test_metrics = {
            'rmse': _to_native_type(np.sqrt(mean_squared_error(result['y_test_actual'], result['y_pred_test_actual']))),
            'mae': _to_native_type(mean_absolute_error(result['y_test_actual'], result['y_pred_test_actual'])),
            'r2': _to_native_type(r2_score(result['y_test_actual'], result['y_pred_test_actual'])),
        }
        train_metrics = {
            'rmse': _to_native_type(np.sqrt(mean_squared_error(result['y_train_actual'], result['y_pred_train_actual']))),
            'mae': _to_native_type(mean_absolute_error(result['y_train_actual'], result['y_pred_train_actual'])),
            'r2': _to_native_type(r2_score(result['y_train_actual'], result['y_pred_train_actual'])),
        }

        future_values = recursive_forecast(result['model'], result['scaler'], result['scaled_values'], window_size, request.forecast_periods)

        freq = pd.infer_freq(df[date_col]) or 'D'
        last_date = df[date_col].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=request.forecast_periods + 1, freq=freq)[1:]

        forecast_records = [
            {'forecast_date': d.isoformat(), 'forecast_value': _to_native_type(v)}
            for d, v in zip(future_dates, future_values)
        ]

        test_dates = df[date_col].iloc[-len(result['y_test_actual']):]

        loss_plot = generate_loss_plot(result['history'])
        forecast_plot = generate_forecast_plot(
            df[date_col], df[value_col], value_col, test_dates, result['y_pred_test_actual'],
            future_dates, future_values, test_metrics['r2']
        )

        interpretation = generate_interpretation(
            train_metrics['r2'], test_metrics['r2'], test_metrics['rmse'],
            len(result['history'].history['loss']), request.epochs, len(df)
        )

        test_predictions = [
            {
                'date': d.isoformat(),
                'actual': _to_native_type(a),
                'predicted': _to_native_type(p),
                'error': _to_native_type(p - a)
            }
            for d, a, p in zip(test_dates, result['y_test_actual'], result['y_pred_test_actual'])
        ]

        return {
            'n_observations': len(df),
            'window_size': window_size,
            'n_train': result['n_train'],
            'n_test': result['n_test'],
            'epochs_run': len(result['history'].history['loss']),
            'final_training_loss': _to_native_type(result['history'].history['loss'][-1]),
            'metrics': {'train': train_metrics, 'test': test_metrics},
            'forecast': forecast_records,
            'test_predictions': test_predictions,
            'loss_plot': loss_plot,
            'forecast_plot': forecast_plot,
            'interpretation': interpretation
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM forecasting failed: {str(e)}")
