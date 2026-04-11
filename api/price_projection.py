from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class PriceProjectionRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    dateCol: Optional[str] = None
    priceCol: Optional[str] = None
    # Generate mode
    generate: bool = False
    ticker: str = "AAPL"
    nDays: int = 750               # ~3 years of trading days
    seed: Optional[int] = None
    # Model config
    forecastDays: int = 60         # days to forecast
    lstmEpochs: int = 50
    lstmLookback: int = 60        # lookback window
    garchP: int = 1
    garchQ: int = 1


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Stock Price Data Generator
# ══════════════════════════════════════════════════════════════

STOCK_PROFILES = {
    'AAPL':  {'name': 'Apple Inc.',           'base': 180.0, 'drift': 0.0008, 'vol': 0.018, 'mean_rev': 0.02},
    'MSFT':  {'name': 'Microsoft Corp.',      'base': 380.0, 'drift': 0.0007, 'vol': 0.016, 'mean_rev': 0.02},
    'GOOGL': {'name': 'Alphabet Inc.',        'base': 140.0, 'drift': 0.0006, 'vol': 0.019, 'mean_rev': 0.02},
    'AMZN':  {'name': 'Amazon.com Inc.',      'base': 175.0, 'drift': 0.0007, 'vol': 0.020, 'mean_rev': 0.02},
    'TSLA':  {'name': 'Tesla Inc.',           'base': 250.0, 'drift': 0.0010, 'vol': 0.035, 'mean_rev': 0.01},
    'JPM':   {'name': 'JPMorgan Chase',       'base': 190.0, 'drift': 0.0005, 'vol': 0.015, 'mean_rev': 0.03},
    'JNJ':   {'name': 'Johnson & Johnson',    'base': 155.0, 'drift': 0.0003, 'vol': 0.010, 'mean_rev': 0.04},
    'NVDA':  {'name': 'NVIDIA Corp.',         'base': 800.0, 'drift': 0.0015, 'vol': 0.030, 'mean_rev': 0.01},
    'META':  {'name': 'Meta Platforms',       'base': 500.0, 'drift': 0.0008, 'vol': 0.022, 'mean_rev': 0.02},
    'WMT':   {'name': 'Walmart Inc.',         'base': 170.0, 'drift': 0.0003, 'vol': 0.010, 'mean_rev': 0.04},
    'XOM':   {'name': 'Exxon Mobil',          'base': 110.0, 'drift': 0.0004, 'vol': 0.018, 'mean_rev': 0.03},
    'V':     {'name': 'Visa Inc.',            'base': 280.0, 'drift': 0.0005, 'vol': 0.013, 'mean_rev': 0.03},
    'NFLX':  {'name': 'Netflix Inc.',         'base': 620.0, 'drift': 0.0009, 'vol': 0.025, 'mean_rev': 0.02},
    'AMD':   {'name': 'AMD Inc.',             'base': 160.0, 'drift': 0.0010, 'vol': 0.028, 'mean_rev': 0.01},
    'DIS':   {'name': 'Walt Disney Co.',      'base':  95.0, 'drift': 0.0002, 'vol': 0.018, 'mean_rev': 0.03},
}


def generate_price_data(
    ticker: str = 'AAPL',
    n_days: int = 750,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate realistic daily stock price data using GBM with stochastic volatility.
    Includes volume and simple technical features.
    """
    rng = np.random.default_rng(seed)

    profile = STOCK_PROFILES.get(ticker.upper(), {
        'name': ticker, 'base': 100.0, 'drift': 0.0005, 'vol': 0.02, 'mean_rev': 0.02,
    })

    base_price = profile['base']
    drift = profile['drift']
    base_vol = profile['vol']
    mean_rev = profile['mean_rev']

    # Stochastic volatility (mean-reverting)
    vol = np.zeros(n_days)
    vol[0] = base_vol
    for i in range(1, n_days):
        vol[i] = vol[i-1] + mean_rev * (base_vol - vol[i-1]) + 0.002 * rng.normal()
        vol[i] = max(vol[i], 0.005)

    # GBM price path
    log_returns = drift + vol * rng.normal(size=n_days)
    prices = base_price * np.exp(np.cumsum(log_returns))

    # Volume (correlated with absolute returns)
    base_volume = rng.integers(5_000_000, 30_000_000)
    abs_ret = np.abs(log_returns)
    volume = base_volume * (1 + 5 * abs_ret / abs_ret.mean()) * (0.8 + 0.4 * rng.random(n_days))
    volume = volume.astype(int)

    # OHLC from close
    high = prices * (1 + np.abs(rng.normal(0, 0.008, n_days)))
    low = prices * (1 - np.abs(rng.normal(0, 0.008, n_days)))
    open_p = prices * (1 + rng.normal(0, 0.003, n_days))

    # Dates
    end_date = pd.Timestamp('2025-04-30')
    dates = pd.bdate_range(end=end_date, periods=n_days)

    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.round(open_p, 2),
        'high': np.round(high, 2),
        'low': np.round(low, 2),
        'close': np.round(prices, 2),
        'volume': volume,
    })

    return df, profile


# ══════════════════════════════════════════════════════════════
# GARCH Volatility Model (arch package)
# ══════════════════════════════════════════════════════════════

def fit_garch_model(
    prices: np.ndarray,
    p: int = 1,
    q: int = 1,
    forecast_days: int = 60,
):
    """
    Fit GARCH(p,q) model using the arch package.
    Returns conditional volatility, forecast, and model diagnostics.
    """
    from arch import arch_model

    # Log returns in %
    returns = np.diff(np.log(prices)) * 100

    # Fit GARCH(p,q)
    model = arch_model(returns, vol='Garch', p=p, q=q, mean='Constant', dist='normal')
    result = model.fit(disp='off', show_warning=False)

    # Conditional volatility (in-sample)
    cond_vol = result.conditional_volatility

    # Forecast
    forecasts = result.forecast(horizon=forecast_days)
    forecast_variance = forecasts.variance.iloc[-1].values
    forecast_vol = np.sqrt(forecast_variance)

    # Standardized residuals
    std_resids = result.std_resid

    # Model parameters
    params = {}
    for name, val in result.params.items():
        params[name] = safe_float(val)

    return {
        'returns': returns.tolist(),
        'cond_vol': cond_vol.tolist(),
        'forecast_vol': forecast_vol.tolist(),
        'forecast_variance': forecast_variance.tolist(),
        'std_resids': std_resids.tolist() if std_resids is not None else [],
        'params': params,
        'aic': safe_float(result.aic),
        'bic': safe_float(result.bic),
        'log_likelihood': safe_float(result.loglikelihood),
        'n_obs': int(len(returns)),
        'p': p,
        'q': q,
    }


# ══════════════════════════════════════════════════════════════
# LSTM Price Prediction (tensorflow/keras)
# ══════════════════════════════════════════════════════════════

def fit_lstm_model(
    prices: np.ndarray,
    lookback: int = 60,
    forecast_days: int = 60,
    epochs: int = 50,
):
    """
    Fit LSTM model for price prediction using tensorflow/keras.
    Uses MinMax scaling and a simple LSTM architecture.
    """
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler

    tf.get_logger().setLevel('ERROR')

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    # In-sample predictions (test set)
    train_pred = model.predict(X_train, verbose=0).flatten()
    test_pred = model.predict(X_test, verbose=0).flatten()

    # Inverse transform
    train_pred_prices = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    test_pred_prices = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Future forecast (recursive)
    last_sequence = scaled[-lookback:].flatten()
    future_preds = []
    for _ in range(forecast_days):
        input_seq = last_sequence[-lookback:].reshape(1, lookback, 1)
        next_val = model.predict(input_seq, verbose=0)[0, 0]
        future_preds.append(next_val)
        last_sequence = np.append(last_sequence, next_val)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    test_rmse = float(np.sqrt(mean_squared_error(y_test_actual, test_pred_prices)))
    test_mae = float(mean_absolute_error(y_test_actual, test_pred_prices))
    test_r2 = float(r2_score(y_test_actual, test_pred_prices))
    test_mape = float(np.mean(np.abs((y_test_actual - test_pred_prices) / y_test_actual)) * 100)

    train_loss = [safe_float(v) for v in history.history['loss']]
    val_loss = [safe_float(v) for v in history.history.get('val_loss', [])]

    return {
        'train_pred': train_pred_prices.tolist(),
        'test_pred': test_pred_prices.tolist(),
        'train_actual': y_train_actual.tolist(),
        'test_actual': y_test_actual.tolist(),
        'future_forecast': future_prices.tolist(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': {
            'rmse': safe_float(test_rmse),
            'mae': safe_float(test_mae),
            'r2': safe_float(test_r2),
            'mape': safe_float(test_mape),
        },
        'lookback': lookback,
        'epochs': epochs,
        'train_size': split,
        'test_size': len(X_test),
        'split_index': lookback + split,
    }


# ══════════════════════════════════════════════════════════════
# Confidence Bands — combine LSTM + GARCH
# ══════════════════════════════════════════════════════════════

def build_confidence_bands(
    last_price: float,
    lstm_forecast: List[float],
    garch_forecast_vol: List[float],
    n_days: int,
):
    """
    Combine LSTM point forecast with GARCH volatility for confidence bands.

    LSTM provides the expected price path.
    GARCH provides the conditional volatility (σ) for uncertainty bands.
    Bands: forecast ± z × σ × √t (scaled daily vol to price level)
    """
    bands = []
    for i in range(n_days):
        price = lstm_forecast[i] if i < len(lstm_forecast) else lstm_forecast[-1]
        vol = garch_forecast_vol[i] if i < len(garch_forecast_vol) else garch_forecast_vol[-1]

        # Convert % vol to price-level vol
        price_vol = price * (vol / 100.0)

        bands.append({
            'day': i + 1,
            'forecast': safe_float(price),
            'upper_95': safe_float(price + 1.96 * price_vol),
            'lower_95': safe_float(price - 1.96 * price_vol),
            'upper_80': safe_float(price + 1.28 * price_vol),
            'lower_80': safe_float(price - 1.28 * price_vol),
            'volatility': safe_float(vol),
        })

    return bands


# ══════════════════════════════════════════════════════════════
# Technical Indicators
# ══════════════════════════════════════════════════════════════

def compute_technicals(prices: np.ndarray, dates: list):
    """Compute SMA, EMA, RSI, Bollinger Bands for visualization."""
    n = len(prices)
    close = pd.Series(prices)

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_20 = close.ewm(span=20).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    bb_mid = sma_20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    result = []
    for i in range(n):
        result.append({
            'date': str(dates[i]),
            'close': safe_float(prices[i]),
            'sma_20': safe_float(sma_20.iloc[i]),
            'sma_50': safe_float(sma_50.iloc[i]),
            'ema_20': safe_float(ema_20.iloc[i]),
            'rsi': safe_float(rsi.iloc[i]),
            'bb_upper': safe_float(bb_upper.iloc[i]),
            'bb_lower': safe_float(bb_lower.iloc[i]),
        })
    return result


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/price-projection")
async def price_projection(request: PriceProjectionRequest):
    try:
        profile = None

        # ── 1. Get Data ──
        if request.generate or not request.data:
            df, profile = generate_price_data(
                ticker=request.ticker,
                n_days=request.nDays,
                seed=request.seed,
            )
            price_col = 'close'
            date_col = 'date'
        else:
            df = pd.DataFrame(request.data)
            price_col = request.priceCol or 'close'
            date_col = request.dateCol or 'date'
            if price_col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{price_col}' not found.")
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            df = df.dropna(subset=[price_col])

        prices = df[price_col].values.astype(np.float64)
        dates = df[date_col].values.tolist() if date_col in df.columns else [str(i) for i in range(len(df))]
        n = len(prices)

        if n < 100:
            raise HTTPException(status_code=400, detail=f"Need at least 100 data points, got {n}")

        # ── 2. GARCH Model ──
        garch_results = fit_garch_model(
            prices=prices,
            p=request.garchP,
            q=request.garchQ,
            forecast_days=request.forecastDays,
        )

        # ── 3. LSTM Model ──
        lstm_results = fit_lstm_model(
            prices=prices,
            lookback=min(request.lstmLookback, n // 3),
            forecast_days=request.forecastDays,
            epochs=request.lstmEpochs,
        )

        # ── 4. Confidence Bands ──
        confidence_bands = build_confidence_bands(
            last_price=prices[-1],
            lstm_forecast=lstm_results['future_forecast'],
            garch_forecast_vol=garch_results['forecast_vol'],
            n_days=request.forecastDays,
        )

        # ── 5. Technical Indicators ──
        technicals = compute_technicals(prices, dates)

        # ── 6. Forecast Dates ──
        last_date = pd.Timestamp(dates[-1])
        forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=request.forecastDays)
        forecast_date_strs = forecast_dates.strftime('%Y-%m-%d').tolist()

        # ── 7. Chart Data — Price + LSTM overlay ──
        lookback = lstm_results['lookback']
        split_idx = lstm_results['split_index']

        price_chart = []
        for i in range(n):
            entry: Dict[str, Any] = {
                'date': str(dates[i]),
                'actual': safe_float(prices[i]),
                'type': 'historical',
            }
            # Map LSTM train predictions
            train_offset = i - lookback
            if 0 <= train_offset < len(lstm_results['train_pred']):
                entry['lstm_train'] = safe_float(lstm_results['train_pred'][train_offset])
            # Map LSTM test predictions
            test_offset = i - split_idx
            if 0 <= test_offset < len(lstm_results['test_pred']):
                entry['lstm_test'] = safe_float(lstm_results['test_pred'][test_offset])

            price_chart.append(entry)

        # Forecast entries
        for i in range(request.forecastDays):
            price_chart.append({
                'date': forecast_date_strs[i],
                'forecast': safe_float(lstm_results['future_forecast'][i]),
                'upper_95': safe_float(confidence_bands[i]['upper_95']),
                'lower_95': safe_float(confidence_bands[i]['lower_95']),
                'upper_80': safe_float(confidence_bands[i]['upper_80']),
                'lower_80': safe_float(confidence_bands[i]['lower_80']),
                'type': 'forecast',
            })

        # ── 8. GARCH volatility chart ──
        vol_chart = []
        for i in range(len(garch_results['cond_vol'])):
            vol_chart.append({
                'date': str(dates[i + 1]) if i + 1 < len(dates) else str(i),
                'cond_vol': safe_float(garch_results['cond_vol'][i]),
                'type': 'historical',
            })
        for i in range(request.forecastDays):
            vol_chart.append({
                'date': forecast_date_strs[i],
                'forecast_vol': safe_float(garch_results['forecast_vol'][i]),
                'type': 'forecast',
            })

        # ── 9. Training loss chart ──
        loss_chart = []
        for i in range(len(lstm_results['train_loss'])):
            entry = {'epoch': i + 1, 'train_loss': lstm_results['train_loss'][i]}
            if i < len(lstm_results['val_loss']):
                entry['val_loss'] = lstm_results['val_loss'][i]
            loss_chart.append(entry)

        # ── 10. Price statistics ──
        returns = np.diff(np.log(prices))
        price_stats = {
            'current_price': safe_float(prices[-1]),
            'price_change_pct': safe_float((prices[-1] / prices[0] - 1) * 100),
            'daily_vol': safe_float(np.std(returns) * 100),
            'annual_vol': safe_float(np.std(returns) * np.sqrt(252) * 100),
            'max_price': safe_float(np.max(prices)),
            'min_price': safe_float(np.min(prices)),
            'avg_volume': safe_float(df['volume'].mean()) if 'volume' in df.columns else 0,
            'sharpe': safe_float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0,
        }

        # ── 11. Forecast summary ──
        forecast_summary = {
            'last_price': safe_float(prices[-1]),
            'forecast_end_price': safe_float(lstm_results['future_forecast'][-1]),
            'forecast_change_pct': safe_float((lstm_results['future_forecast'][-1] / prices[-1] - 1) * 100),
            'forecast_high': safe_float(max(confidence_bands[i]['upper_95'] for i in range(len(confidence_bands)))),
            'forecast_low': safe_float(min(confidence_bands[i]['lower_95'] for i in range(len(confidence_bands)))),
            'avg_forecast_vol': safe_float(np.mean(garch_results['forecast_vol'])),
        }

        # ── 12. Build Response ──
        results = {
            'ticker': request.ticker,
            'stock_profile': profile,
            'price_stats': price_stats,
            'forecast_summary': forecast_summary,
            'lstm': {
                'metrics': lstm_results['metrics'],
                'lookback': lstm_results['lookback'],
                'epochs': lstm_results['epochs'],
                'train_size': lstm_results['train_size'],
                'test_size': lstm_results['test_size'],
            },
            'garch': {
                'params': garch_results['params'],
                'aic': garch_results['aic'],
                'bic': garch_results['bic'],
                'log_likelihood': garch_results['log_likelihood'],
                'p': garch_results['p'],
                'q': garch_results['q'],
            },
            'confidence_bands': confidence_bands,
            'data_summary': {
                'n_observations': n,
                'date_range': f"{dates[0]} to {dates[-1]}",
                'forecast_days': request.forecastDays,
                'forecast_range': f"{forecast_date_strs[0]} to {forecast_date_strs[-1]}",
            },
            'charts': {
                'price_chart': price_chart,
                'vol_chart': vol_chart,
                'loss_chart': loss_chart,
                'technicals': technicals,
            },
            'available_tickers': list(STOCK_PROFILES.keys()),
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
