from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from scipy import stats
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class NormalityTestRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    variables: List[str] = Field(..., description="List of variables to test")
    alpha: float = Field(default=0.05)


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


@router.post("/normality-test")
def normality_test(req: NormalityTestRequest):
    try:
        df = pd.DataFrame(req.data)
        variables = req.variables
        alpha = req.alpha
        
        results = {}
        
        for var in variables:
            if var not in df.columns:
                results[var] = {"error": f"Variable '{var}' not found"}
                continue
            
            series = pd.to_numeric(df[var], errors='coerce').dropna()
            
            if len(series) < 3:
                results[var] = {"error": f"Not enough data for '{var}' (minimum 3)"}
                continue
            
            n = len(series)
            
            # Shapiro-Wilk
            sw_stat, sw_p = stats.shapiro(series)
            
            # Jarque-Bera
            jb_stat, jb_p = stats.jarque_bera(series)
            
            # Kolmogorov-Smirnov
            standardized = (series - series.mean()) / series.std()
            ks_stat, ks_p = stats.kstest(standardized, 'norm')
            
            # Select primary test
            if n < 50:
                primary_test = 'shapiro_wilk'
                primary_p = sw_p
                test_name = 'Shapiro-Wilk'
                test_reason = f'Best for small samples (n={n} < 50)'
            elif n < 300:
                primary_test = 'kolmogorov_smirnov'
                primary_p = ks_p
                test_name = 'Kolmogorov-Smirnov'
                test_reason = f'Recommended for medium samples (n={n})'
            else:
                primary_test = 'jarque_bera'
                primary_p = jb_p
                test_name = 'Jarque-Bera'
                test_reason = f'Efficient for large samples (n={n})'
            
            is_normal = primary_p > alpha
            
            interpretation = f"Using {test_name} ({test_reason}). "
            if is_normal:
                interpretation += f"Data does not deviate from normal (p = {primary_p:.4f} > {alpha}). Normality assumption met."
            else:
                interpretation += f"Data deviates from normal (p = {primary_p:.4f} < {alpha}). Consider non-parametric tests."
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            sns.histplot(series, kde=True, ax=axes[0], color='#5B9BD5')
            axes[0].set_title(f'Distribution of {var}')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')
            
            stats.probplot(series, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            
            results[var] = {
                'n': n,
                'primary_test': primary_test,
                'primary_test_name': test_name,
                'shapiro_wilk': {'statistic': float(sw_stat), 'p_value': float(sw_p)},
                'jarque_bera': {'statistic': float(jb_stat), 'p_value': float(jb_p)},
                'kolmogorov_smirnov': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
                'is_normal': bool(is_normal),
                'is_normal_shapiro': bool(sw_p > alpha),
                'is_normal_jarque': bool(jb_p > alpha),
                'is_normal_ks': bool(ks_p > alpha),
                'interpretation': interpretation,
                'plot': plot
            }
        
        return _to_native({'results': results})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
