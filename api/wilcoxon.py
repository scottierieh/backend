from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats
import io
import base64
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class WilcoxonRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    var1: str = Field(..., description="First variable (e.g., pre-test)")
    var2: str = Field(..., description="Second variable (e.g., post-test)")
    alternative: str = Field(default='two-sided', description="Alternative hypothesis")
    alpha: float = Field(default=0.05, description="Significance level")


def _to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def get_effect_size_interpretation(r: float) -> dict:
    """Interpret effect size r"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return {"magnitude": "Negligible", "text": "Negligible effect (r < 0.1)"}
    elif abs_r < 0.3:
        return {"magnitude": "Small", "text": "Small effect (0.1 ≤ r < 0.3)"}
    elif abs_r < 0.5:
        return {"magnitude": "Medium", "text": "Medium effect (0.3 ≤ r < 0.5)"}
    else:
        return {"magnitude": "Large", "text": "Large effect (r ≥ 0.5)"}


@router.post("/wilcoxon")
def wilcoxon_test(req: WilcoxonRequest):
    try:
        df = pd.DataFrame(req.data)
        var1 = req.var1
        var2 = req.var2
        alternative = req.alternative
        alpha = req.alpha
        
        # Clean data - remove rows with missing values
        clean_df = df[[var1, var2]].dropna()
        n_dropped = len(df) - len(clean_df)
        
        if len(clean_df) < 10:
            raise ValueError(f"Need at least 10 valid pairs, but only {len(clean_df)} pairs after removing missing values")
        
        # Extract data
        d1 = clean_df[var1].values
        d2 = clean_df[var2].values
        n = len(d1)
        
        # Calculate differences
        differences = d1 - d2
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(d1, d2, alternative=alternative)
        
        # Calculate W+ and W- (sum of positive and negative ranks)
        abs_diffs = np.abs(differences)
        non_zero_mask = differences != 0
        non_zero_diffs = differences[non_zero_mask]
        non_zero_abs_diffs = abs_diffs[non_zero_mask]
        
        if len(non_zero_diffs) > 0:
            ranks = stats.rankdata(non_zero_abs_diffs)
            W_plus = np.sum(ranks[non_zero_diffs > 0])
            W_minus = np.sum(ranks[non_zero_diffs < 0])
            n_diff = len(non_zero_diffs)
        else:
            W_plus = 0
            W_minus = 0
            n_diff = 0
        
        # Calculate Z-score and effect size
        if n_diff > 0:
            mean_W = n_diff * (n_diff + 1) / 4
            std_W = np.sqrt(n_diff * (n_diff + 1) * (2 * n_diff + 1) / 24)
            z_score = (statistic - mean_W) / std_W if std_W > 0 else 0
            effect_size = abs(z_score) / np.sqrt(n)
        else:
            z_score = 0
            effect_size = 0
        
        # Descriptive statistics
        descriptive_stats = {
            var1: {
                'n': int(n),
                'mean': float(np.mean(d1)),
                'median': float(np.median(d1)),
                'std': float(np.std(d1, ddof=1)) if n > 1 else 0,
                'min': float(np.min(d1)),
                'max': float(np.max(d1))
            },
            var2: {
                'n': int(n),
                'mean': float(np.mean(d2)),
                'median': float(np.median(d2)),
                'std': float(np.std(d2, ddof=1)) if n > 1 else 0,
                'min': float(np.min(d2)),
                'max': float(np.max(d2))
            },
            'difference': {
                'n': int(n),
                'mean': float(np.mean(differences)),
                'median': float(np.median(differences)),
                'std': float(np.std(differences, ddof=1)) if n > 1 else 0,
                'min': float(np.min(differences)),
                'max': float(np.max(differences))
            }
        }
        
        # Interpretation
        is_significant = p_value < alpha
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.4f}"
        effect_interpretation = get_effect_size_interpretation(effect_size)
        
        if is_significant:
            decision = "Reject null hypothesis"
            conclusion = f"Wilcoxon W = {statistic:.1f}, Z = {z_score:.3f}, {p_text}. There is a statistically significant difference between {var1} and {var2}. Effect size r = {effect_size:.3f} ({effect_interpretation['magnitude'].lower()})."
        else:
            decision = "Fail to reject null hypothesis"
            conclusion = f"Wilcoxon W = {statistic:.1f}, Z = {z_score:.3f}, {p_text}. There is no statistically significant difference between {var1} and {var2}. Effect size r = {effect_size:.3f} ({effect_interpretation['magnitude'].lower()})."
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Box plot for paired variables
        box_data = pd.DataFrame({var1: d1, var2: d2})
        sns.boxplot(data=box_data, palette='crest', ax=axes[0])
        axes[0].set_title('Paired Variables Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Value')
        
        # Histogram of differences with KDE
        sns.histplot(differences, kde=True, ax=axes[1], color='steelblue')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        axes[1].axvline(np.median(differences), color='green', linestyle='-', linewidth=2, label=f'Median = {np.median(differences):.2f}')
        axes[1].set_title('Distribution of Differences', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(f'Difference ({var1} - {var2})')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # Paired scatter plot
        axes[2].scatter(d1, d2, alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.5)
        min_val = min(d1.min(), d2.min())
        max_val = max(d1.max(), d2.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x (no change)')
        axes[2].set_title('Paired Scatter Plot', fontsize=12, fontweight='bold')
        axes[2].set_xlabel(var1)
        axes[2].set_ylabel(var2)
        axes[2].legend()
        axes[2].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # Build response
        result = {
            'test_type': 'Wilcoxon Signed-Rank Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'z_score': float(z_score),
            'effect_size': float(effect_size),
            'effect_size_interpretation': effect_interpretation,
            'is_significant': is_significant,
            'W_plus': float(W_plus),
            'W_minus': float(W_minus),
            'n': int(n),
            'n_valid': int(n),
            'interpretation': {
                'decision': decision,
                'conclusion': conclusion
            },
            'descriptive_stats': descriptive_stats,
            'variables': [var1, var2]
        }
        
        return _to_native({
            'results': result,
            'plot': plot_base64,
            'n_dropped': n_dropped
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
