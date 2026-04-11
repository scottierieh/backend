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


class FriedmanRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    variables: List[str] = Field(..., description="List of column names representing repeated conditions")
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


def get_concordance_interpretation(w: float) -> dict:
    """Interpret Kendall's W concordance coefficient"""
    if w < 0.1:
        return {"level": "very_weak", "text": "Very weak agreement (W < 0.1)"}
    elif w < 0.3:
        return {"level": "weak", "text": "Weak agreement (0.1 ≤ W < 0.3)"}
    elif w < 0.5:
        return {"level": "moderate", "text": "Moderate agreement (0.3 ≤ W < 0.5)"}
    elif w < 0.7:
        return {"level": "strong", "text": "Strong agreement (0.5 ≤ W < 0.7)"}
    else:
        return {"level": "very_strong", "text": "Very strong agreement (W ≥ 0.7)"}


@router.post("/friedman")
def friedman_test(req: FriedmanRequest):
    try:
        df = pd.DataFrame(req.data)
        variables = req.variables
        alpha = req.alpha
        
        # Validate
        if len(variables) < 3:
            raise ValueError(f"Friedman test requires at least 3 conditions, got {len(variables)}")
        
        # Check all variables exist
        missing_vars = [v for v in variables if v not in df.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in data: {missing_vars}")
        
        # Clean data - remove rows with any missing values in selected variables
        clean_df = df[variables].dropna()
        n_dropped = len(df) - len(clean_df)
        
        if len(clean_df) < 3:
            raise ValueError(f"Need at least 3 complete observations, but only {len(clean_df)} remain after removing missing values")
        
        # Extract data matrix
        data_matrix = clean_df.values
        n_subjects = len(data_matrix)
        k_conditions = len(variables)
        
        # Perform Friedman test
        statistic, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(k_conditions)])
        
        # Calculate Kendall's W (coefficient of concordance)
        # W = chi_squared / (n * (k - 1))
        kendall_w = statistic / (n_subjects * (k_conditions - 1)) if n_subjects > 0 and k_conditions > 1 else 0
        kendall_w = min(1.0, max(0.0, kendall_w))  # Clamp to [0, 1]
        
        # Degrees of freedom
        df_value = k_conditions - 1
        
        # Calculate mean ranks for each condition
        ranks = np.zeros_like(data_matrix, dtype=float)
        for i in range(n_subjects):
            ranks[i, :] = stats.rankdata(data_matrix[i, :])
        mean_ranks = np.mean(ranks, axis=0)
        
        # Condition statistics
        condition_stats = {}
        for i, var in enumerate(variables):
            col_data = data_matrix[:, i]
            condition_stats[var] = {
                'count': int(n_subjects),
                'mean': float(np.mean(col_data)),
                'median': float(np.median(col_data)),
                'std': float(np.std(col_data, ddof=1)) if n_subjects > 1 else 0,
                'min': float(np.min(col_data)),
                'max': float(np.max(col_data)),
                'mean_rank': float(mean_ranks[i])
            }
        
        # Interpretation
        is_significant = p_value < alpha
        effect_interpretation = get_concordance_interpretation(kendall_w)
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.4f}"
        
        if is_significant:
            decision = "Reject null hypothesis"
            conclusion = f"Friedman χ²({df_value}) = {statistic:.3f}, {p_text}. There are statistically significant differences among the {k_conditions} repeated conditions. Kendall's W = {kendall_w:.3f} ({effect_interpretation['text'].split('(')[0].strip()})."
        else:
            decision = "Fail to reject null hypothesis"
            conclusion = f"Friedman χ²({df_value}) = {statistic:.3f}, {p_text}. There are no statistically significant differences among the {k_conditions} repeated conditions. Kendall's W = {kendall_w:.3f} ({effect_interpretation['text'].split('(')[0].strip()})."
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Melt data for plotting
        melted = clean_df.melt(var_name='Condition', value_name='Value')
        
        # Box plot
        sns.boxplot(x='Condition', y='Value', data=melted, palette='crest', ax=axes[0])
        axes[0].set_title('Box Plot by Condition', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(x='Condition', y='Value', data=melted, palette='crest', ax=axes[1])
        axes[1].set_title('Violin Plot by Condition', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Mean ranks bar plot
        colors = sns.color_palette('crest', k_conditions)
        axes[2].bar(range(k_conditions), mean_ranks, color=colors)
        axes[2].set_xticks(range(k_conditions))
        axes[2].set_xticklabels(variables, rotation=45, ha='right')
        axes[2].set_title('Mean Ranks by Condition', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Condition')
        axes[2].set_ylabel('Mean Rank')
        
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # Build response
        result = {
            'test_type': 'Friedman Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'df': int(df_value),
            'effect_size': float(kendall_w),
            'effect_size_interpretation': effect_interpretation,
            'is_significant': is_significant,
            'interpretation': {
                'decision': decision,
                'conclusion': conclusion
            },
            'condition_stats': condition_stats,
            'variables': variables,
            'n_subjects': int(n_subjects),
            'n_conditions': int(k_conditions),
            'mean_ranks': {var: float(mean_ranks[i]) for i, var in enumerate(variables)}
        }
        
        return _to_native({
            'results': result,
            'plot': plot_base64,
            'n_dropped': n_dropped
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
