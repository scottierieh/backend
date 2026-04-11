from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_arch
import io
import base64
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class ArchLmRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]]
    valueCol: Optional[str] = None
    lags: int = 10


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


def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except:
        return default


@router.post("/arch-lm-test")
async def arch_lm_test(request: ArchLmRequest):
    try:
        data = request.data
        value_col = request.valueCol
        lags = request.lags

        # Handle both formats: array of numbers OR array of objects
        if len(data) > 0 and isinstance(data[0], (int, float)):
            series = pd.Series(data).dropna()
        else:
            df = pd.DataFrame(data)
            if value_col and value_col in df.columns:
                series = pd.to_numeric(df[value_col], errors='coerce').dropna()
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise HTTPException(status_code=400, detail="No numeric columns found")
                series = df[numeric_cols[0]].dropna()

        if len(series) <= lags:
            raise HTTPException(
                status_code=400,
                detail=f"Need more than {lags} observations. Have {len(series)}."
            )

        # Perform ARCH-LM test
        test_result = het_arch(series.values, nlags=lags)
        lm_stat = safe_float(test_result[0])
        p_value = safe_float(test_result[1])
        f_stat = safe_float(test_result[2])
        f_p_value = safe_float(test_result[3])

        is_significant = p_value < 0.05

        # Build interpretation
        interpretation = (
            f"The ARCH-LM test examines whether the variance is constant over time. "
            f"With {lags} lags, LM statistic = {lm_stat:.2f}, p = {p_value:.4f}. "
        )
        if is_significant:
            interpretation += "ARCH effects detected - volatility clustering present. Consider GARCH models."
        else:
            interpretation += "No ARCH effects - constant variance assumption holds."

        # Create squared residuals plot for visualization
        squared_residuals = series.values ** 2
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('ARCH-LM Test: Volatility Analysis', fontsize=14, fontweight='bold')

        # Original series
        axes[0].plot(series.values, color='#1f77b4', linewidth=1, alpha=0.8)
        axes[0].set_title('Original Series', fontsize=11)
        axes[0].set_ylabel('Value')
        axes[0].axhline(y=series.mean(), color='red', linestyle='--', alpha=0.5, label=f'Mean: {series.mean():.2f}')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.6)

        # Squared residuals (proxy for volatility)
        axes[1].plot(squared_residuals, color='#d62728', linewidth=1, alpha=0.8)
        axes[1].set_title('Squared Values (Volatility Proxy)', fontsize=11)
        axes[1].set_xlabel('Observation')
        axes[1].set_ylabel('Squared Value')
        axes[1].axhline(y=np.mean(squared_residuals), color='blue', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.6)

        # Add test result annotation
        result_text = f"LM = {lm_stat:.2f}, p = {p_value:.4f}\n"
        result_text += "ARCH Effects: " + ("Yes" if is_significant else "No")
        axes[1].annotate(result_text,
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=10, ha='left', va='top',
                        bbox=dict(boxstyle='round', 
                                 facecolor='#ffcccc' if is_significant else '#ccffcc', 
                                 alpha=0.8))

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        response = {
            'results': {
                'lm_statistic': lm_stat,
                'p_value': p_value,
                'f_statistic': f_stat,
                'f_p_value': f_p_value,
                'lags': lags,
                'is_significant': is_significant,
                'interpretation': interpretation,
                'n_observations': len(series)
            },
            'plot': f"data:image/png;base64,{plot_b64}"
        }

        return _to_native(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
