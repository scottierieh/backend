from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class SeasonalStrengthRequest(BaseModel):
    data: List[Dict[str, Any]]
    variable: str
    period: int = 12
    test_periods: List[int] = [4, 7, 12, 24]
    auto_detect: bool = True


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


def calculate_seasonal_strength(series, period):
    try:
        if len(series) < period * 2:
            return None, None, None
        decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        seasonal = decomposition.seasonal
        trend = decomposition.trend
        resid = decomposition.resid
        valid_idx = ~(np.isnan(resid) | np.isnan(seasonal) | np.isnan(trend))
        if valid_idx.sum() < period:
            return None, None, decomposition
        detrended = seasonal[valid_idx] + resid[valid_idx]
        var_resid = np.var(resid[valid_idx])
        var_detrended = np.var(detrended)
        ssi = max(0, 1 - var_resid / var_detrended) if var_detrended > 0 else 0
        deseasonalized = trend[valid_idx] + resid[valid_idx]
        var_deseasonalized = np.var(deseasonalized)
        tsi = max(0, 1 - var_resid / var_deseasonalized) if var_deseasonalized > 0 else 0
        return ssi, tsi, decomposition
    except:
        return None, None, None


def detect_dominant_period(series, max_period=None):
    try:
        n = len(series)
        if max_period is None:
            max_period = n // 2
        detrended = series - np.linspace(series.iloc[0], series.iloc[-1], n)
        yf = fft(detrended.values)
        xf = fftfreq(n, 1)
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        power = np.abs(yf[pos_mask]) ** 2
        periods = 1 / xf_pos
        valid_mask = (periods >= 2) & (periods <= max_period)
        if valid_mask.sum() == 0:
            return None, [], []
        periods_valid = periods[valid_mask]
        power_valid = power[valid_mask]
        dominant_idx = np.argmax(power_valid)
        dominant_period = int(round(periods_valid[dominant_idx]))
        sorted_idx = np.argsort(power_valid)[::-1][:5]
        fourier_components = []
        for idx in sorted_idx:
            fourier_components.append({
                'period': float(periods_valid[idx]), 'frequency': float(1 / periods_valid[idx]),
                'amplitude': float(np.sqrt(power_valid[idx])), 'power': float(power_valid[idx])
            })
        return dominant_period, fourier_components, (periods_valid, power_valid)
    except:
        return None, [], []


def calculate_seasonal_indices(series, period):
    indices = []
    for pos in range(period):
        values = series.iloc[pos::period]
        if len(values) > 0:
            mean_val = values.mean()
            overall_mean = series.mean()
            index = mean_val / overall_mean if overall_mean != 0 else 1
            indices.append({'position': pos + 1, 'index': float(index), 'std': float(values.std()) if len(values) > 1 else 0, 'n': len(values)})
    return indices


def create_decomposition_plot(decomposition, variable_name, period):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(decomposition.observed, 'b-', linewidth=1)
    axes[0].set_ylabel('Observed'); axes[0].set_title(f'Time Series Decomposition: {variable_name} (period={period})', fontweight='bold'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(decomposition.trend, 'g-', linewidth=1.5); axes[1].set_ylabel('Trend'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(decomposition.seasonal, 'r-', linewidth=1); axes[2].set_ylabel('Seasonal'); axes[2].grid(True, alpha=0.3)
    axes[3].plot(decomposition.resid, 'purple', linewidth=1, alpha=0.7); axes[3].set_ylabel('Residual'); axes[3].set_xlabel('Index'); axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_seasonal_pattern_plot(seasonal_indices, variable_name, period):
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = [s['position'] for s in seasonal_indices]
    indices = [s['index'] for s in seasonal_indices]
    colors = ['#55A868' if idx > 1 else '#C44E52' for idx in indices]
    ax.bar(positions, indices, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='Baseline (1.0)')
    ax.set_xlabel('Season Position'); ax.set_ylabel('Seasonal Index')
    ax.set_title(f'Seasonal Pattern: {variable_name}', fontweight='bold')
    ax.set_xticks(positions); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_gauge_plot(ssi, tsi, variable_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, value, title in zip(axes, [ssi, tsi], ['Seasonal Strength', 'Trend Strength']):
        if value is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=24, color='gray'); ax.axis('off'); ax.set_title(title, fontweight='bold'); continue
        theta = np.linspace(np.pi, 0, 100)
        for start, end, c in [(0, 0.4, '#C44E52'), (0.4, 0.7, '#CCCC00'), (0.7, 1, '#55A868')]:
            mask = (theta >= np.pi * (1 - end)) & (theta <= np.pi * (1 - start))
            ax.fill_between(theta[mask], 0.6, 1, alpha=0.3, color=c)
        needle_theta = np.pi * (1 - value)
        ax.arrow(0, 0, 0.5 * np.cos(needle_theta), 0.5 * np.sin(needle_theta), head_width=0.08, head_length=0.05, fc='black', ec='black')
        ax.plot(0, 0, 'ko', markersize=10)
        ax.text(0, -0.3, f'{value:.3f}', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontweight='bold')
    plt.suptitle(f'Strength Indices: {variable_name}', fontweight='bold', y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_strength_comparison_plot(period_comparison):
    fig, ax = plt.subplots(figsize=(12, 6))
    periods = [p['period'] for p in period_comparison]
    ssi_values = [p['seasonal_strength'] for p in period_comparison]
    x = np.arange(len(periods))
    bars = ax.bar(x, ssi_values, color='#4C72B0', alpha=0.7, edgecolor='black')
    best_idx = np.argmax(ssi_values)
    bars[best_idx].set_color('#55A868')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong (0.7)')
    ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.4)')
    ax.set_xlabel('Period'); ax.set_ylabel('Seasonal Strength Index')
    ax.set_title('Seasonal Strength by Period', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(periods); ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_insights(ssi, tsi, period, dominant_period, seasonal_indices):
    insights, recommendations = [], []
    if ssi is not None:
        if ssi > 0.7:
            insights.append({"type": "info", "title": "Strong Seasonality ✓", "description": f"SSI of {ssi:.3f} indicates strong, predictable seasonal patterns."})
            recommendations.append("Use seasonal models (SARIMA, Holt-Winters) for forecasting.")
        elif ssi > 0.4:
            insights.append({"type": "info", "title": "Moderate Seasonality", "description": f"SSI of {ssi:.3f} suggests noticeable but not dominant seasonal effects."})
        else:
            insights.append({"type": "warning", "title": "Weak Seasonality", "description": f"SSI of {ssi:.3f} indicates minimal seasonal patterns."})
    if tsi is not None:
        if tsi > 0.7:
            insights.append({"type": "info", "title": "Strong Trend Component", "description": f"TSI of {tsi:.3f} indicates significant trend in the data."})
    if dominant_period and dominant_period != period:
        insights.append({"type": "warning", "title": "Period Mismatch", "description": f"Detected period ({dominant_period}) differs from specified ({period})."})
    if seasonal_indices:
        max_idx = max(seasonal_indices, key=lambda x: x['index'])
        min_idx = min(seasonal_indices, key=lambda x: x['index'])
        if max_idx['index'] > 1.2:
            insights.append({"type": "info", "title": f"Peak Season: Position {max_idx['position']}", "description": f"Index of {max_idx['index']:.3f}."})
        if min_idx['index'] < 0.8:
            insights.append({"type": "info", "title": f"Low Season: Position {min_idx['position']}", "description": f"Index of {min_idx['index']:.3f}."})
    recommendations.extend(["Compare multiple periods to find optimal seasonal cycle.", "Check for structural breaks."])
    return insights, list(dict.fromkeys(recommendations))


@router.post("/seasonal-strength")
async def seasonal_strength(request: SeasonalStrengthRequest):
    try:
        df = pd.DataFrame(request.data)
        variable = request.variable
        period = request.period
        test_periods = request.test_periods
        auto_detect = request.auto_detect
        if variable not in df.columns:
            raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found")
        series = pd.to_numeric(df[variable], errors='coerce').dropna().reset_index(drop=True)
        n = len(series)
        if n < period * 2:
            raise HTTPException(status_code=400, detail=f"Need at least {period * 2} observations for period={period}, got {n}")
        ssi, tsi, decomposition = calculate_seasonal_strength(series, period)
        if ssi is None:
            raise HTTPException(status_code=400, detail="Could not calculate seasonal strength.")
        dominant_period, fourier_components, periodogram_data = (None, [], [])
        if auto_detect:
            dominant_period, fourier_components, periodogram_data = detect_dominant_period(series, max_period=min(n//2, 100))
        seasonal_indices = calculate_seasonal_indices(series, period)
        period_comparison = []
        for test_period in [p for p in test_periods if n >= p * 2 and p >= 2]:
            test_ssi, test_tsi, _ = calculate_seasonal_strength(series, test_period)
            if test_ssi is not None:
                period_comparison.append({'period': test_period, 'seasonal_strength': test_ssi, 'trend_strength': test_tsi, 'combined_strength': (test_ssi + (test_tsi or 0)) / 2})
        plots = {'decomposition': create_decomposition_plot(decomposition, variable, period), 'seasonal_pattern': create_seasonal_pattern_plot(seasonal_indices, variable, period), 'gauge': create_gauge_plot(ssi, tsi, variable)}
        if len(period_comparison) > 1:
            plots['strength_comparison'] = create_strength_comparison_plot(period_comparison)
        insights, recommendations = generate_insights(ssi, tsi, period, dominant_period, seasonal_indices)
        seasonality_interp = "Strong" if ssi > 0.7 else "Moderate" if ssi > 0.4 else "Weak"
        trend_interp = "Strong" if tsi and tsi > 0.7 else "Moderate" if tsi and tsi > 0.4 else "Weak" if tsi else "N/A"
        return _to_native({'variable': variable, 'n_observations': n, 'period': period, 'seasonal_strength_index': ssi, 'trend_strength_index': tsi, 'dominant_period_detected': dominant_period, 'seasonal_indices': seasonal_indices, 'period_comparison': period_comparison, 'fourier_components': fourier_components, 'interpretation': {'seasonality': seasonality_interp, 'trend': trend_interp}, 'insights': insights, 'recommendations': recommendations, 'plots': plots})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
