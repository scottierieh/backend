from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
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


class HomogeneityTestRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    valueVar: str = Field(..., description="Numeric value variable")
    groupVar: str = Field(..., description="Grouping variable")
    alpha: float = Field(default=0.05)


@router.post("/homogeneity-test")
def homogeneity_test(req: HomogeneityTestRequest):
    try:
        df = pd.DataFrame(req.data)
        value_var = req.valueVar
        group_var = req.groupVar
        alpha = req.alpha
        
        # Clean data
        clean_data = df[[value_var, group_var]].dropna()
        clean_data[value_var] = pd.to_numeric(clean_data[value_var], errors='coerce')
        clean_data = clean_data.dropna()
        
        groups = clean_data[group_var].unique()
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups")
        
        samples = [clean_data[clean_data[group_var] == g][value_var].values for g in groups]
        
        k = len(groups)
        n_total = sum(len(s) for s in samples)
        df_between = k - 1
        df_within = n_total - k
        
        # Levene's Test
        levene_stat, levene_p = stats.levene(*samples, center='median')
        
        # Bartlett's Test
        bartlett_stat, bartlett_p = stats.bartlett(*samples)
        
        # Descriptives
        descriptives = {}
        for i, group in enumerate(groups):
            sample = samples[i]
            descriptives[str(group)] = {
                'n': int(len(sample)),
                'mean': float(np.mean(sample)),
                'variance': float(np.var(sample, ddof=1)),
                'std_dev': float(np.std(sample, ddof=1))
            }
        
        assumption_met = levene_p > alpha
        
        if assumption_met:
            interpretation = f"Levene's test not significant (p = {levene_p:.4f} > {alpha}). Variances are equal. Assumption met."
        else:
            interpretation = f"Levene's test significant (p = {levene_p:.4f} ≤ {alpha}). Variances are unequal. Assumption violated."
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=group_var, y=value_var, data=clean_data, palette='crest', ax=ax)
        ax.set_title('Distribution by Group')
        ax.set_xlabel(group_var)
        ax.set_ylabel(value_var)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        return {
            'results': {
                'levene_test': {
                    'statistic': float(levene_stat),
                    'p_value': float(levene_p),
                    'df_between': df_between,
                    'df_within': df_within
                },
                'bartlett_test': {
                    'statistic': float(bartlett_stat),
                    'p_value': float(bartlett_p),
                    'df': df_between
                },
                'descriptives': descriptives,
                'assumption_met': bool(assumption_met),
                'interpretation': interpretation,
                'plot': plot
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
