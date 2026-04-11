from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class OutlierDetectionRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    variables: List[str] = Field(..., description="List of numeric variables to check for outliers")


def detect_outliers_zscore(series, threshold=3):
    """Detect outliers using Z-score method"""
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return []
    
    z_scores = (series - mean) / std
    outliers = []
    
    for idx, (value, z) in enumerate(zip(series, z_scores)):
        if abs(z) > threshold:
            outliers.append({
                'index': int(idx),
                'value': float(value),
                'z_score': float(z)
            })
    
    return outliers


def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = []
    
    for idx, value in enumerate(series):
        if value < lower_bound or value > upper_bound:
            outliers.append({
                'index': int(idx),
                'value': float(value)
            })
    
    return outliers


def create_outlier_plot(series, variable_name):
    """Create box plot with outliers highlighted"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    ax1 = axes[0]
    bp = ax1.boxplot(series.dropna(), patch_artist=True)
    bp['boxes'][0].set_facecolor('#4C72B0')
    bp['boxes'][0].set_alpha(0.7)
    ax1.set_ylabel(variable_name)
    ax1.set_title(f'Box Plot: {variable_name}', fontweight='bold')
    ax1.set_xticklabels([variable_name])
    
    # Histogram with outlier boundaries
    ax2 = axes[1]
    data = series.dropna()
    
    # Calculate IQR bounds
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Calculate Z-score bounds
    mean = data.mean()
    std = data.std()
    z_lower = mean - 3 * std
    z_upper = mean + 3 * std
    
    ax2.hist(data, bins=30, color='#4C72B0', alpha=0.7, edgecolor='white')
    ax2.axvline(lower_bound, color='#C44E52', linestyle='--', linewidth=2, label=f'IQR bounds')
    ax2.axvline(upper_bound, color='#C44E52', linestyle='--', linewidth=2)
    ax2.axvline(z_lower, color='#FFB347', linestyle=':', linewidth=2, label=f'Z-score bounds (±3σ)')
    ax2.axvline(z_upper, color='#FFB347', linestyle=':', linewidth=2)
    ax2.set_xlabel(variable_name)
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution with Outlier Boundaries', fontweight='bold')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


@router.post("/outlier-detection")
def outlier_detection(req: OutlierDetectionRequest):
    try:
        df = pd.DataFrame(req.data)
        variables = req.variables
        
        results = {}
        
        for variable in variables:
            if variable not in df.columns:
                results[variable] = {'error': f"Variable '{variable}' not found in data"}
                continue
            
            # Convert to numeric and clean
            series = pd.to_numeric(df[variable], errors='coerce').dropna()
            
            if len(series) < 3:
                results[variable] = {'error': f"Not enough valid data points for '{variable}'"}
                continue
            
            # Detect outliers
            z_score_outliers = detect_outliers_zscore(series)
            iqr_outliers = detect_outliers_iqr(series)
            
            # Create plot
            plot = create_outlier_plot(series, variable)
            
            results[variable] = {
                'z_score_outliers': z_score_outliers,
                'iqr_outliers': iqr_outliers,
                'summary': {
                    'total_count': int(len(series)),
                    'z_score_count': len(z_score_outliers),
                    'iqr_count': len(iqr_outliers)
                },
                'plot': plot
            }
        
        return {'results': results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
