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


class KruskalWallisRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    group_col: str = Field(..., description="Column name for grouping variable")
    value_col: str = Field(..., description="Column name for numeric value")
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


def get_effect_size_interpretation(epsilon_sq: float) -> dict:
    """Interpret effect size epsilon-squared"""
    if epsilon_sq < 0.01:
        return {"magnitude": "Negligible", "text": "Negligible effect (ε² < 0.01)"}
    elif epsilon_sq < 0.06:
        return {"magnitude": "Small", "text": "Small effect (0.01 ≤ ε² < 0.06)"}
    elif epsilon_sq < 0.14:
        return {"magnitude": "Medium", "text": "Medium effect (0.06 ≤ ε² < 0.14)"}
    else:
        return {"magnitude": "Large", "text": "Large effect (ε² ≥ 0.14)"}


@router.post("/kruskal-wallis")
def kruskal_wallis_test(req: KruskalWallisRequest):
    try:
        df = pd.DataFrame(req.data)
        group_col = req.group_col
        value_col = req.value_col
        alpha = req.alpha
        
        # Clean data - remove rows with missing values
        clean_df = df[[group_col, value_col]].dropna()
        n_dropped = len(df) - len(clean_df)
        
        # Get unique groups
        groups = clean_df[group_col].dropna().unique()
        if len(groups) < 3:
            raise ValueError(f"Kruskal-Wallis test requires at least 3 groups, found {len(groups)}")
        
        # Extract data for each group
        groups_data = [clean_df[clean_df[group_col] == g][value_col].values for g in groups]
        
        # Check minimum group size
        min_group_size = min(len(g) for g in groups_data)
        if min_group_size < 2:
            raise ValueError(f"Each group must have at least 2 observations, smallest group has {min_group_size}")
        
        # Perform Kruskal-Wallis test
        statistic, p_value = stats.kruskal(*groups_data)
        
        # Calculate effect size (epsilon-squared)
        n_total = sum(len(g) for g in groups_data)
        k = len(groups_data)
        epsilon_sq = (statistic - k + 1) / (n_total - k) if (n_total - k) > 0 else 0
        epsilon_sq = max(0, epsilon_sq)  # Ensure non-negative
        
        # Degrees of freedom
        df_value = k - 1
        
        # Group statistics
        group_stats = {}
        for i, g in enumerate(groups):
            gd = groups_data[i]
            group_stats[str(g)] = {
                'count': int(len(gd)),
                'n': int(len(gd)),
                'mean': float(np.mean(gd)),
                'median': float(np.median(gd)),
                'std': float(np.std(gd, ddof=1)) if len(gd) > 1 else 0,
                'min': float(np.min(gd)),
                'max': float(np.max(gd))
            }
        
        # Interpretation
        is_significant = p_value < alpha
        effect_interpretation = get_effect_size_interpretation(epsilon_sq)
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.4f}"
        
        if is_significant:
            decision = "Reject null hypothesis"
            conclusion = f"Kruskal-Wallis H({df_value}) = {statistic:.3f}, {p_text}. There are statistically significant differences among the {k} groups. Effect size ε² = {epsilon_sq:.3f} ({effect_interpretation['magnitude'].lower()})."
        else:
            decision = "Fail to reject null hypothesis"
            conclusion = f"Kruskal-Wallis H({df_value}) = {statistic:.3f}, {p_text}. There are no statistically significant differences among the {k} groups. Effect size ε² = {epsilon_sq:.3f} ({effect_interpretation['magnitude'].lower()})."
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Box plot
        sns.boxplot(x=group_col, y=value_col, data=clean_df, palette='crest', ax=axes[0])
        axes[0].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(group_col)
        axes[0].set_ylabel(value_col)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(x=group_col, y=value_col, data=clean_df, palette='crest', ax=axes[1])
        axes[1].set_title('Violin Plot Comparison', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(group_col)
        axes[1].set_ylabel(value_col)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Bar plot of medians
        medians = [np.median(gd) for gd in groups_data]
        colors = sns.color_palette('crest', len(groups))
        axes[2].bar(range(len(groups)), medians, color=colors)
        axes[2].set_xticks(range(len(groups)))
        axes[2].set_xticklabels([str(g) for g in groups], rotation=45, ha='right')
        axes[2].set_title('Median Comparison', fontsize=12, fontweight='bold')
        axes[2].set_xlabel(group_col)
        axes[2].set_ylabel(f'Median {value_col}')
        
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # Build response
        result = {
            'test_type': 'Kruskal-Wallis H Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'df': int(df_value),
            'effect_size': float(epsilon_sq),
            'effect_size_interpretation': effect_interpretation,
            'is_significant': is_significant,
            'interpretation': {
                'decision': decision,
                'conclusion': conclusion
            },
            'group_stats': group_stats,
            'groups': [str(g) for g in groups],
            'n_groups': int(k),
            'n_total': int(n_total)
        }
        
        return _to_native({
            'results': result,
            'plot': plot_base64,
            'n_dropped': n_dropped
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
