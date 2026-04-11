from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import base64
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class TrendAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    timeCol: str
    valueCol: str
    model: str = 'additive'
    period: int = 7


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


def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except:
        return default


@router.post("/trend")
async def trend_analysis(request: TrendAnalysisRequest):
    try:
        df = pd.DataFrame(request.data)
        time_col = request.timeCol
        value_col = request.valueCol
        model = request.model
        period = request.period

        if time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{time_col}' not found")
        if value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{value_col}' not found")

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[time_col, value_col]).set_index(time_col).sort_index()

        if len(df) < period * 2:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough data. Need at least {period * 2} points, have {len(df)}."
            )

        if model == 'multiplicative' and (df[value_col] <= 0).any():
            raise HTTPException(
                status_code=400,
                detail="Multiplicative model requires all positive values. Use 'additive' instead."
            )

        decomposition = seasonal_decompose(df[value_col], model=model, period=period)

        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        resid = decomposition.resid.dropna()

        original_values = df[value_col].values
        trend_values = trend.values
        seasonal_values = seasonal.values
        resid_values = resid.values

        if len(trend_values) >= 2:
            trend_change = trend_values[-1] - trend_values[0]
            if trend_change > 0:
                trend_direction = 'increasing'
            elif trend_change < 0:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'

        var_resid = np.var(resid_values) if len(resid_values) > 0 else 0
        var_trend_resid = np.var(trend_values) + var_resid if len(trend_values) > 0 else 1
        var_seasonal_resid = np.var(seasonal_values) + var_resid if len(seasonal_values) > 0 else 1

        strength_trend = max(0, 1 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0
        strength_seasonal = max(0, 1 - var_resid / var_seasonal_resid) if var_seasonal_resid > 0 else 0

        statistics = {
            'original_mean': safe_float(np.mean(original_values)),
            'original_std': safe_float(np.std(original_values)),
            'original_min': safe_float(np.min(original_values)),
            'original_max': safe_float(np.max(original_values)),
            'trend_start': safe_float(trend_values[0]) if len(trend_values) > 0 else None,
            'trend_end': safe_float(trend_values[-1]) if len(trend_values) > 0 else None,
            'trend_direction': trend_direction,
            'strength_trend': safe_float(strength_trend),
            'strength_seasonal': safe_float(strength_seasonal),
            'residual_mean': safe_float(np.mean(resid_values)),
            'residual_std': safe_float(np.std(resid_values)),
        }

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Time Series Decomposition ({model.capitalize()} Model, Period={period})', 
                     fontsize=14, fontweight='bold')

        axes[0].plot(df.index, df[value_col], color='#1f77b4', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('Observed', fontsize=11)
        axes[0].set_title('Original Time Series', fontsize=11, loc='left')

        axes[1].plot(trend.index, trend.values, color='#ff7f0e', linewidth=2)
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].set_title(f'Trend Component (Strength: {strength_trend:.1%})', fontsize=11, loc='left')

        axes[2].plot(seasonal.index, seasonal.values, color='#2ca02c', linewidth=1.5)
        axes[2].set_ylabel('Seasonal', fontsize=11)
        axes[2].set_title(f'Seasonal Component (Strength: {strength_seasonal:.1%})', fontsize=11, loc='left')

        axes[3].scatter(resid.index, resid.values, alpha=0.5, color='#d62728', s=15)
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[3].set_ylabel('Residual', fontsize=11)
        axes[3].set_xlabel(time_col, fontsize=11)
        axes[3].set_title('Residual Component', fontsize=11, loc='left')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        def series_to_records(series, col_name):
            records = []
            for idx, val in series.items():
                records.append({
                    time_col: idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    col_name: safe_float(val)
                })
            return records

        results = {
            'trend': series_to_records(trend, 'trend'),
            'seasonal': series_to_records(seasonal, 'seasonal'),
            'resid': series_to_records(resid, 'resid'),
        }

        return _to_native({
            'results': results,
            'statistics': statistics,
            'plot': f"data:image/png;base64,{plot_b64}"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
