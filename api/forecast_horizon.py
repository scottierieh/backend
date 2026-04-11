from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class ForecastHorizonRequest(BaseModel):
    data: List[Dict[str, Any]]
    variable: str
    model_type: str = 'linear'
    horizons: List[int] = [1, 3, 5, 7, 10, 14]
    n_lags: int = 10
    train_ratio: float = 0.7


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
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


def create_lagged_features(series, lags):
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    return df.dropna()


def get_model(model_type):
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    return models.get(model_type, LinearRegression())


def walk_forward_validation(series, model_type, horizon, n_lags, train_ratio=0.7):
    n = len(series)
    train_size = int(n * train_ratio)
    if train_size < n_lags + horizon + 10:
        return None
    df = create_lagged_features(series, n_lags)
    if len(df) < train_size:
        return None
    y = df['y'].values
    X = df.drop('y', axis=1).values
    predictions, actuals, prediction_indices = [], [], []
    
    for i in range(train_size, len(df) - horizon + 1):
        X_train, y_train = X[:i], y[:i]
        model = get_model(model_type)
        model.fit(X_train, y_train)
        current_features = X[i].copy()
        for h in range(horizon):
            pred = model.predict(current_features.reshape(1, -1))[0]
            if h == horizon - 1:
                predictions.append(pred)
                if i + horizon - 1 < len(y):
                    actuals.append(y[i + horizon - 1])
                    prediction_indices.append(i + horizon - 1)
            if h < horizon - 1:
                current_features = np.roll(current_features, 1)
                current_features[0] = pred
    
    if len(predictions) == 0 or len(actuals) == 0:
        return None
    
    predictions, actuals = np.array(predictions), np.array(actuals)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100 if mask.sum() > 0 else None
    r2 = r2_score(actuals, predictions) if len(actuals) > 1 else None
    
    if len(actuals) > 1:
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        directional_accuracy = None
    
    return {
        'horizon': horizon, 'n_predictions': len(predictions), 'mse': float(mse), 'rmse': float(rmse),
        'mae': float(mae), 'mape': float(mape) if mape else None, 'r2': float(r2) if r2 else None,
        'directional_accuracy': float(directional_accuracy) if directional_accuracy else None,
        'predictions': predictions.tolist(), 'actuals': actuals.tolist(), 'indices': prediction_indices
    }


def calculate_decay_rate(results):
    if len(results) < 2:
        return None
    horizons = [r['horizon'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    try:
        log_rmse = np.log(rmse_values)
        slope, intercept, r_value, p_value, std_err = stats.linregress(horizons, log_rmse)
        return {'decay_rate': float(slope), 'r_squared': float(r_value ** 2), 'p_value': float(p_value),
                'interpretation': 'exponential_growth' if slope > 0 else 'exponential_decay'}
    except:
        return None


def find_optimal_horizon(results, metric='rmse', threshold_pct=20):
    if len(results) < 2:
        return None
    base_value = results[0][metric]
    for r in results[1:]:
        current_value = r[metric]
        if base_value > 0:
            pct_change = ((current_value - base_value) / base_value) * 100
            if pct_change > threshold_pct:
                return {'optimal_horizon': r['horizon'] - 1, 'threshold_exceeded_at': r['horizon'],
                        'base_value': float(base_value), 'exceeded_value': float(current_value), 'pct_increase': float(pct_change)}
    return {'optimal_horizon': results[-1]['horizon'], 'threshold_exceeded_at': None, 'note': 'No significant degradation detected'}


def create_metrics_vs_horizon_plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    horizons = [r['horizon'] for r in results]
    rmse = [r['rmse'] for r in results]
    mae = [r['mae'] for r in results]
    mape = [r['mape'] for r in results if r['mape'] is not None]
    r2 = [r['r2'] for r in results if r['r2'] is not None]
    
    axes[0, 0].plot(horizons, rmse, 'o-', color='#4C72B0', linewidth=2, markersize=8)
    axes[0, 0].fill_between(horizons, 0, rmse, alpha=0.2, color='#4C72B0')
    axes[0, 0].set_xlabel('Forecast Horizon'); axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE vs Forecast Horizon', fontweight='bold'); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(horizons, mae, 's-', color='#55A868', linewidth=2, markersize=8)
    axes[0, 1].fill_between(horizons, 0, mae, alpha=0.2, color='#55A868')
    axes[0, 1].set_xlabel('Forecast Horizon'); axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE vs Forecast Horizon', fontweight='bold'); axes[0, 1].grid(True, alpha=0.3)
    
    if mape:
        h_mape = [r['horizon'] for r in results if r['mape'] is not None]
        axes[1, 0].plot(h_mape, mape, '^-', color='#C44E52', linewidth=2, markersize=8)
        axes[1, 0].fill_between(h_mape, 0, mape, alpha=0.2, color='#C44E52')
    axes[1, 0].set_xlabel('Forecast Horizon'); axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].set_title('MAPE vs Forecast Horizon', fontweight='bold'); axes[1, 0].grid(True, alpha=0.3)
    
    if r2:
        h_r2 = [r['horizon'] for r in results if r['r2'] is not None]
        axes[1, 1].plot(h_r2, r2, 'd-', color='#8172B3', linewidth=2, markersize=8)
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Forecast Horizon'); axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('R² vs Forecast Horizon', fontweight='bold'); axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_forecast_comparison_plot(series, results):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(series))
    ax.plot(x, series, 'b-', linewidth=1, alpha=0.5, label='Actual')
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for r, color in zip(results, colors):
        if r['indices'] and r['predictions']:
            ax.scatter(r['indices'], r['predictions'], color=color, s=30, alpha=0.7,
                      label=f"Horizon {r['horizon']} (RMSE={r['rmse']:.2f})")
    ax.set_xlabel('Index'); ax.set_ylabel('Value')
    ax.set_title('Forecast Comparison Across Horizons', fontweight='bold')
    ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_insights(results, decay_info, optimal_horizon):
    insights, recommendations = [], []
    if not results:
        return [{"type": "warning", "title": "Insufficient Data", "description": "Could not generate forecasts."}], []
    
    first_rmse, last_rmse = results[0]['rmse'], results[-1]['rmse']
    pct_increase = ((last_rmse - first_rmse) / first_rmse) * 100 if first_rmse > 0 else 0
    insights.append({"type": "info", "title": "Error Growth Summary",
                    "description": f"RMSE increases from {first_rmse:.4f} (h=1) to {last_rmse:.4f} (h={results[-1]['horizon']}), a {pct_increase:.1f}% increase."})
    
    if decay_info:
        if decay_info['decay_rate'] > 0.1:
            insights.append({"type": "warning", "title": "Rapid Performance Decay",
                            "description": f"Exponential decay rate of {decay_info['decay_rate']:.4f} indicates rapidly deteriorating forecasts."})
        elif decay_info['decay_rate'] > 0:
            insights.append({"type": "info", "title": "Gradual Performance Decay",
                            "description": f"Moderate decay rate of {decay_info['decay_rate']:.4f} suggests stable short-term forecasts."})
    
    if optimal_horizon and optimal_horizon.get('optimal_horizon'):
        insights.append({"type": "info", "title": f"Recommended Horizon: {optimal_horizon['optimal_horizon']}",
                        "description": f"Beyond horizon {optimal_horizon.get('threshold_exceeded_at', 'N/A')}, error increases significantly."})
    
    r2_values = [r['r2'] for r in results if r['r2'] is not None]
    if r2_values:
        if r2_values[0] > 0.7:
            insights.append({"type": "info", "title": "Good Short-term Predictability ✓",
                            "description": f"R² of {r2_values[0]:.3f} at horizon 1 indicates strong short-term forecasting ability."})
        if r2_values[-1] < 0:
            insights.append({"type": "warning", "title": "Long Horizon Unreliable",
                            "description": f"Negative R² ({r2_values[-1]:.3f}) at longest horizon means forecasts are worse than mean prediction."})
    
    recommendations = ["Use shorter forecast horizons for critical decisions.", "Consider ensemble methods to improve longer-horizon forecasts.",
                      "Regularly retrain models as new data becomes available.", "Add exogenous variables to improve multi-step forecasts."]
    if pct_increase > 100:
        recommendations.insert(0, "Limit forecasts to short horizons due to high error growth.")
    
    return insights, recommendations


@router.post("/forecast-horizon")
async def forecast_horizon(request: ForecastHorizonRequest):
    try:
        df = pd.DataFrame(request.data)
        variable = request.variable
        model_type = request.model_type
        horizons = request.horizons
        n_lags = request.n_lags
        train_ratio = request.train_ratio

        if variable not in df.columns:
            raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found")

        series = pd.to_numeric(df[variable], errors='coerce').dropna().reset_index(drop=True)
        arr = series.values
        n = len(arr)

        if n < 50:
            raise HTTPException(status_code=400, detail=f"Need at least 50 observations, got {n}")

        max_horizon = min(max(horizons), n // 4)
        valid_horizons = [h for h in horizons if h <= max_horizon]
        if not valid_horizons:
            valid_horizons = [1, 2, 3]

        results = []
        for h in valid_horizons:
            result = walk_forward_validation(arr, model_type, h, n_lags, train_ratio)
            if result:
                results.append(result)

        if not results:
            raise HTTPException(status_code=400, detail="Could not generate forecasts. Try with more data or fewer horizons.")

        decay_info = calculate_decay_rate(results)
        optimal_horizon = find_optimal_horizon(results)

        plots = {'metrics': create_metrics_vs_horizon_plot(results), 'forecast_comparison': create_forecast_comparison_plot(arr, results)}
        insights, recommendations = generate_insights(results, decay_info, optimal_horizon)

        summary = {'n_observations': n, 'model_type': model_type, 'n_lags': n_lags, 'train_ratio': train_ratio,
                  'horizons_tested': [r['horizon'] for r in results],
                  'best_horizon_rmse': min(results, key=lambda x: x['rmse'])['horizon'],
                  'worst_horizon_rmse': max(results, key=lambda x: x['rmse'])['horizon']}

        results_simplified = [{'horizon': r['horizon'], 'n_predictions': r['n_predictions'], 'rmse': r['rmse'],
                              'mae': r['mae'], 'mape': r['mape'], 'r2': r['r2'], 'directional_accuracy': r['directional_accuracy']} for r in results]

        return _to_native({'variable': variable, 'summary': summary, 'results': results_simplified, 'decay_info': decay_info,
                          'optimal_horizon': optimal_horizon, 'insights': insights, 'recommendations': recommendations, 'plots': plots})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
