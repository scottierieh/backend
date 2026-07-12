
import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
import warnings

warnings.filterwarnings('ignore')

import os
# Quiet TensorFlow/oneDNN/absl startup chatter on stderr (must be set before the
# tensorflow import). Otherwise "oneDNN custom operations are on ..." and
# "absl::InitializeLog ..." leak into the endpoint's error surface.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
try:
    tf.get_logger().setLevel('ERROR')
    from absl import logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass

tf.random.set_seed(42)

def _generate_interpretation(train_metrics, test_metrics, epochs_run, epochs_requested,
                              window_size, forecast_periods, n_obs):
    parts = []
    r2_diff = train_metrics['r2_score'] - test_metrics['r2_score']

    parts.append("**Overall Assessment**")
    parts.append(
        f"→ LSTM trained on {n_obs} observations using a look-back window of {window_size} steps, "
        f"forecasting {forecast_periods} steps ahead. Training ran for {epochs_run} of {epochs_requested} "
        "requested epochs" + (" (stopped early)." if epochs_run < epochs_requested else ".")
    )

    if test_metrics['r2_score'] > 0.8 and r2_diff < 0.2:
        fit_desc = "**Good Fit**. Train and test R² are both high and close together, suggesting the model generalizes well to unseen periods."
    elif train_metrics['r2_score'] > 0.7 and r2_diff > 0.3:
        fit_desc = "**Overfitting Warning**. The model fits training data much better than test data — consider a shorter window, fewer LSTM units, or stronger dropout."
    elif train_metrics['r2_score'] < 0.5 and test_metrics['r2_score'] < 0.5:
        fit_desc = "**Underfitting Possible**. Both train and test R² are low — try more LSTM units, a longer window_size, or more training epochs."
    else:
        fit_desc = "Model performance is moderate; review the RMSE/MAE alongside the forecast plot before relying on this model."
    parts.append(f"→ {fit_desc}")

    parts.append("")
    parts.append("**Statistical Insights**")
    parts.append(
        f"→ Test RMSE = {test_metrics['rmse']:.3f} | Test MAE = {test_metrics['mae']:.3f} | "
        f"Test R² = {test_metrics['r2_score']:.3f}"
    )
    parts.append(
        f"→ Train RMSE = {train_metrics['rmse']:.3f} | Train R² = {train_metrics['r2_score']:.3f}"
    )

    parts.append("")
    parts.append("**Recommendations**")
    parts.append(
        "→ Multi-step forecasts beyond the observed data are generated recursively (each prediction "
        "feeds the next step), so uncertainty compounds — treat forecasts far beyond the training range "
        "with caution."
    )
    parts.append("→ Compare against a simple baseline (e.g. naive last-value or ARIMA) to confirm the LSTM's added complexity is actually paying off.")
    if n_obs < 200:
        parts.append("→ With a relatively small number of observations, LSTM forecasts may be unstable across re-runs; consider classical time-series models as a more data-efficient alternative.")

    return "\n".join(parts)

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def _make_sequences(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)

def main():
    try:
        payload = json.load(sys.stdin)

        # Accept both camelCase and snake_case keys. The statistica frontend
        # sends snake_case (date_col/value_col/window_size/...), while earlier
        # callers used camelCase (timeCol/valueCol/windowSize/...). Supporting
        # both keeps every caller working and fixes the "Missing required
        # parameters" failure without a lock-step frontend/backend deploy.
        def _p(*keys, default=None):
            for k in keys:
                v = payload.get(k)
                if v is not None:
                    return v
            return default

        data = _p('data')
        time_col = _p('timeCol', 'time_col', 'date_col', 'dateCol')
        value_col = _p('valueCol', 'value_col')
        window_size = int(_p('windowSize', 'window_size', default=12))
        forecast_periods = int(_p('forecastPeriods', 'forecast_periods', default=12))
        lstm_units = int(_p('lstmUnits', 'lstm_units', default=50))
        epochs = int(_p('epochs', default=100))
        batch_size = int(_p('batchSize', 'batch_size', default=16))
        test_size = float(_p('testSize', 'test_size', default=0.2))

        if not all([data, time_col, value_col]):
            raise ValueError("Missing required parameters: data, timeCol/date_col, or valueCol/value_col")

        df = pd.DataFrame(data)
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col, value_col]).sort_values(time_col).reset_index(drop=True)

        if len(df) < window_size + 10:
            raise ValueError(f"Not enough data. Need at least {window_size + 10} observations.")

        values = df[[value_col]].values.astype(float)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values).flatten()

        X, y = _make_sequences(scaled_values, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_test) == 0:
            raise ValueError("Test set is empty; reduce window_size or testSize.")

        model = Sequential([
            LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=(window_size, 1)),
            Dropout(0.2),
            LSTM(max(lstm_units // 2, 8), activation='tanh'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop],
            shuffle=False
        )

        y_pred_test_scaled = model.predict(X_test, verbose=0).flatten()
        y_pred_train_scaled = model.predict(X_train, verbose=0).flatten()

        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_test_actual = scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_pred_train_actual = scaler.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()

        test_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))),
            'mae': float(mean_absolute_error(y_test_actual, y_pred_test_actual)),
            'r2_score': float(r2_score(y_test_actual, y_pred_test_actual))
        }
        train_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual))),
            'mae': float(mean_absolute_error(y_train_actual, y_pred_train_actual)),
            'r2_score': float(r2_score(y_train_actual, y_pred_train_actual))
        }

        # Recursive multi-step forecast beyond the observed data
        last_window = scaled_values[-window_size:].tolist()
        future_scaled = []
        for _ in range(forecast_periods):
            x_input = np.array(last_window[-window_size:]).reshape((1, window_size, 1))
            next_val = model.predict(x_input, verbose=0)[0, 0]
            future_scaled.append(next_val)
            last_window.append(next_val)

        future_values = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()

        freq = pd.infer_freq(df[time_col]) or 'D'
        last_date = df[time_col].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]

        forecast_records = [
            {'forecast_date': d, 'forecast_value': float(v)}
            for d, v in zip(future_dates, future_values)
        ]

        results = {
            'metrics': {'train': train_metrics, 'test': test_metrics},
            'window_size': window_size,
            'epochs_run': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'forecast': forecast_records,
            'n_train_samples': int(len(X_train)),
            'n_test_samples': int(len(X_test)),
            # Raw test-set actuals/predictions (inverse-scaled) so the reported
            # RMSE/MAE can be independently recomputed by the validation harness.
            '_validation': {
                'y_test_actual': [float(v) for v in y_test_actual],
                'y_pred_test': [float(v) for v in y_pred_test_actual],
            }
        }
        results['interpretation'] = _generate_interpretation(
            train_metrics, test_metrics, len(history.history['loss']), epochs,
            window_size, forecast_periods, len(df)
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        fig.suptitle('LSTM Forecasting Analysis', fontsize=16)

        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss (scaled)')
        axes[0].set_title('Training Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        test_dates = df[time_col].iloc[-len(y_test_actual):]
        axes[1].plot(df[time_col], df[value_col], label='Historical', color='steelblue')
        axes[1].plot(test_dates, y_pred_test_actual, label='Test Predictions', color='darkorange')
        axes[1].plot(future_dates, future_values, label='Future Forecast', color='green', linestyle='--', marker='o', markersize=3)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel(value_col)
        axes[1].set_title(f"Actual vs Predicted (Test R² = {test_metrics['r2_score']:.3f})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')

        response = {
            'results': results,
            'plot': f"data:image/png;base64,{plot_image}"
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
