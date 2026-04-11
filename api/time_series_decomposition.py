from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from statsmodels.tsa.seasonal import seasonal_decompose

router = APIRouter()

class TimeSeriesDecompositionRequest(BaseModel):
    data: List[dict] = Field(...)
    timeCol: str = Field(...)
    valueCol: str = Field(...)
    model: Optional[str] = 'additive'  # 'additive' or 'multiplicative'
    period: Optional[int] = 7

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except: return default

def generate_interpretation(df, trend, seasonal, resid, model, period, value_col):
    parts = []
    
    # Overall analysis
    parts.append("**Overall Analysis**")
    parts.append(f"→ Time series decomposition performed using the **{model}** model with period **{period}**.")
    parts.append(f"→ Analyzed **{len(df)}** observations of variable **{value_col}**.")
    
    # Trend analysis
    trend_direction = "upward" if trend.iloc[-1] > trend.iloc[0] else "downward" if trend.iloc[-1] < trend.iloc[0] else "stable"
    trend_change = ((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100) if trend.iloc[0] != 0 else 0
    parts.append(f"→ Overall trend is **{trend_direction}** ({trend_change:+.1f}% change from start to end).")
    
    parts.append("")
    parts.append("**Key Insights**")
    
    # Seasonal strength
    seasonal_var = seasonal.var()
    resid_var = resid.var()
    total_var = df[value_col].var()
    
    if total_var > 0:
        seasonal_strength = 1 - (resid_var / (seasonal_var + resid_var)) if (seasonal_var + resid_var) > 0 else 0
        if seasonal_strength > 0.7:
            parts.append(f"→ **Strong seasonality** detected (strength: {seasonal_strength:.2f}). Seasonal patterns are highly predictable.")
        elif seasonal_strength > 0.4:
            parts.append(f"→ **Moderate seasonality** detected (strength: {seasonal_strength:.2f}). Some regular patterns present.")
        else:
            parts.append(f"→ **Weak seasonality** (strength: {seasonal_strength:.2f}). Seasonal patterns are minimal.")
    
    # Residual analysis
    resid_mean = resid.mean()
    resid_std = resid.std()
    parts.append(f"→ Residuals have mean **{resid_mean:.4f}** and std **{resid_std:.4f}**.")
    
    if abs(resid_mean) < resid_std * 0.1:
        parts.append("→ Residuals are centered near zero, indicating good model fit.")
    else:
        parts.append("→ Residuals show some bias; consider alternative decomposition model.")
    
    # Outliers in residuals
    outlier_threshold = 2 * resid_std
    n_outliers = len(resid[abs(resid - resid_mean) > outlier_threshold])
    if n_outliers > 0:
        parts.append(f"→ **{n_outliers}** potential outliers detected in residuals (beyond 2σ).")
    
    parts.append("")
    parts.append("**Recommendations**")
    
    if model == 'additive':
        parts.append("→ Additive model assumes constant seasonal amplitude. If variance increases with level, try **multiplicative** model.")
    else:
        parts.append("→ Multiplicative model used. If the series contains zero or negative values, consider **additive** model.")
    
    parts.append(f"→ Current period is {period}. Adjust if data has different seasonality (e.g., 12 for monthly, 4 for quarterly).")
    parts.append("→ Use the **trend** component for long-term forecasting and the **seasonal** component for short-term patterns.")
    parts.append("→ Large residuals indicate unexplained variation; consider adding external variables or using advanced models (SARIMA, Prophet).")
    
    return "\n".join(parts)

@router.post("/time-series-decomposition")
def time_series_decomposition(req: TimeSeriesDecompositionRequest):
    try:
        df = pd.DataFrame(req.data)
        time_col = req.timeCol
        value_col = req.valueCol
        model = req.model or 'additive'
        period = req.period or 7
        
        # Validate columns
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found")
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found")
        
        # Prepare data
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[time_col, value_col]).set_index(time_col).sort_index()
        
        if len(df) < period * 2:
            raise ValueError(f"Not enough data for the given period. Need at least {period * 2} data points, but have {len(df)}.")
        
        # Check for multiplicative model requirements
        if model == 'multiplicative' and (df[value_col] <= 0).any():
            raise ValueError("Multiplicative model requires all positive values. Use 'additive' model instead.")
        
        # Perform decomposition
        decomposition = seasonal_decompose(df[value_col], model=model, period=period)
        
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        resid = decomposition.resid.dropna()
        
        # Create plots
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Original Series
        axes[0].plot(df.index, df[value_col], color='#1f77b4', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('Observed', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Time Series Decomposition ({model.capitalize()} Model, Period={period})', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(trend.index, trend.values, color='#ff7f0e', linewidth=2)
        axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonality
        axes[2].plot(seasonal.index, seasonal.values, color='#2ca02c', linewidth=1.5)
        axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Residuals
        axes[3].scatter(resid.index, resid.values, alpha=0.5, color='#d62728', s=15)
        axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
        axes[3].set_xlabel(time_col, fontsize=11)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
        
        # Generate interpretation
        interpretation = generate_interpretation(df, trend, seasonal, resid, model, period, value_col)
        
        # Prepare results
        def series_to_records(series, col_name):
            return [
                {time_col: idx.isoformat() if hasattr(idx, 'isoformat') else str(idx), col_name: safe_float(val)}
                for idx, val in series.items()
            ]
        
        response = {
            'results': {
                'n_observations': len(df),
                'period': period,
                'model': model,
                'time_column': time_col,
                'value_column': value_col,
                'trend': series_to_records(trend, 'trend'),
                'seasonal': series_to_records(seasonal, 'seasonal'),
                'residual': series_to_records(resid, 'residual'),
                'statistics': {
                    'original_mean': safe_float(df[value_col].mean()),
                    'original_std': safe_float(df[value_col].std()),
                    'trend_start': safe_float(trend.iloc[0]),
                    'trend_end': safe_float(trend.iloc[-1]),
                    'seasonal_amplitude': safe_float(seasonal.max() - seasonal.min()),
                    'residual_mean': safe_float(resid.mean()),
                    'residual_std': safe_float(resid.std()),
                },
                'interpretation': interpretation
            },
            'plot': plot
        }
        
        return _to_native(response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
