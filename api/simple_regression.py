# -*- coding: utf-8 -*-
"""
Simple Linear Regression API Endpoint
- Fixed unicode encoding (R², √, etc.)
- Added input validation with Pydantic
- Improved diagnostics and interpretation
- Correlation analysis included
- Better error handling
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import io
import base64
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


# =============================================================================
# Request Model with Validation
# =============================================================================

class SimpleRegressionRequest(BaseModel):
    """Request model for Simple Linear Regression"""
    data: List[dict[str, Any]] = Field(..., min_length=10, description="Dataset with at least 10 rows")
    targetVar: str = Field(..., min_length=1, description="Target (dependent) variable name")
    feature: str = Field(..., min_length=1, description="Feature (independent) variable name")
    confidence_level: float = Field(default=0.95, ge=0.80, le=0.99, description="Confidence level for intervals")

    @field_validator('confidence_level')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.80 <= v <= 0.99:
            raise ValueError('Confidence level must be between 0.80 and 0.99')
        return v


# =============================================================================
# Utility Functions
# =============================================================================

def _to_native(obj: Any) -> Any:
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


def generate_interpretation(
    r2: float,
    adj_r2: float,
    f_pvalue: float,
    slope: float,
    slope_pvalue: float,
    correlation: float,
    feature_name: str,
    target_name: str,
    n: int,
    normality_ok: bool,
    homoscedasticity_ok: bool
) -> str:
    """Generate human-readable interpretation of regression results"""
    lines = []
    
    # Effect size interpretation
    if r2 >= 0.75:
        effect = "excellent"
        effect_desc = "very strong"
    elif r2 >= 0.50:
        effect = "good"
        effect_desc = "strong"
    elif r2 >= 0.25:
        effect = "moderate"
        effect_desc = "moderate"
    elif r2 >= 0.10:
        effect = "weak"
        effect_desc = "weak"
    else:
        effect = "very weak"
        effect_desc = "very weak"
    
    # Main findings
    lines.append("**Overall Assessment**")
    lines.append(f"→ Model explains {r2*100:.1f}% of variance in {target_name} (R² = {r2:.4f}).")
    lines.append(f"→ This represents {effect} explanatory power ({effect_desc} relationship).")
    
    # Significance
    is_significant = f_pvalue < 0.05
    if is_significant:
        lines.append(f"→ ✓ Model is statistically significant (F-test p = {f_pvalue:.4f}).")
    else:
        lines.append(f"→ ✗ Model is NOT statistically significant (F-test p = {f_pvalue:.4f}).")
    
    # Direction and relationship
    lines.append("")
    lines.append("**Relationship Details**")
    direction = "positive" if slope > 0 else "negative"
    lines.append(f"→ {feature_name} has a {direction} relationship with {target_name}.")
    lines.append(f"→ Pearson correlation: r = {correlation:.4f}")
    
    if slope_pvalue < 0.05:
        lines.append(f"→ For each 1-unit increase in {feature_name}, {target_name} changes by {slope:.4f} units.")
    else:
        lines.append(f"→ The slope coefficient is not statistically significant (p = {slope_pvalue:.4f}).")
    
    # Assumption checks
    lines.append("")
    lines.append("**Assumption Diagnostics**")
    
    if normality_ok:
        lines.append("→ ✓ Residuals appear normally distributed.")
    else:
        lines.append("→ ⚠ Residuals may not be normally distributed (check Q-Q plot).")
    
    if homoscedasticity_ok:
        lines.append("→ ✓ Homoscedasticity assumption appears satisfied.")
    else:
        lines.append("→ ⚠ Potential heteroscedasticity detected (variance not constant).")
    
    # Recommendations
    lines.append("")
    lines.append("**Recommendations**")
    
    if not is_significant:
        lines.append("→ The relationship is not significant. Consider:")
        lines.append("  • Adding more observations")
        lines.append("  • Trying different predictors")
        lines.append("  • Checking for non-linear relationships")
    elif r2 < 0.25:
        lines.append("→ Low R² suggests other factors influence the outcome.")
        lines.append("→ Consider multiple regression with additional predictors.")
    else:
        lines.append("→ Model performs well for prediction purposes.")
        if not normality_ok or not homoscedasticity_ok:
            lines.append("→ Address assumption violations for more reliable inference.")
    
    return "\n".join(lines)


def generate_plot(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    residuals: pd.Series, 
    sm_model: Any, 
    X: pd.DataFrame,
    feature_name: str,
    target_name: str
) -> str:
    """Generate diagnostic plots for simple regression"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    line_color = '#C44E52'
    point_color = '#5B9BD5'
    
    # Plot 1: Scatter with regression line
    sns.scatterplot(x=X.iloc[:, 0], y=y_true, alpha=0.6, color=point_color, ax=axes[0, 0])
    sorted_idx = X.iloc[:, 0].argsort()
    axes[0, 0].plot(X.iloc[:, 0].iloc[sorted_idx], y_pred.iloc[sorted_idx], color=line_color, linewidth=2)
    
    # Add confidence band
    try:
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        X_const = sm.add_constant(X)
        _, ci_lower, ci_upper = wls_prediction_std(sm_model, alpha=0.05)
        axes[0, 0].fill_between(
            X.iloc[:, 0].iloc[sorted_idx], 
            ci_lower[sorted_idx], 
            ci_upper[sorted_idx], 
            alpha=0.2, 
            color=line_color
        )
    except:
        pass
    
    axes[0, 0].set_title(f"Regression Line (R² = {sm_model.rsquared:.4f})", fontweight='bold')
    axes[0, 0].set_xlabel(feature_name)
    axes[0, 0].set_ylabel(target_name)
    
    # Plot 2: Residuals vs Fitted
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color=point_color, ax=axes[0, 1])
    axes[0, 1].axhline(0, linestyle='--', color=line_color, linewidth=2)
    
    # Add lowess smoother for pattern detection
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, y_pred, frac=0.6)
        axes[0, 1].plot(smoothed[:, 0], smoothed[:, 1], color='#2ECC71', linewidth=2, linestyle='-')
    except:
        pass
    
    axes[0, 1].set_title('Residuals vs Fitted', fontweight='bold')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    
    # Plot 3: Q-Q Plot
    sm.qqplot(residuals, line='s', ax=axes[1, 0], markerfacecolor=point_color, alpha=0.6)
    axes[1, 0].set_title('Normal Q-Q Plot', fontweight='bold')
    axes[1, 0].get_lines()[0].set_color(point_color)
    axes[1, 0].get_lines()[1].set_color(line_color)
    
    # Plot 4: Scale-Location (Spread-Location)
    std_resid = sm_model.get_influence().resid_studentized_internal
    sqrt_abs = np.sqrt(np.abs(std_resid))
    sns.scatterplot(x=y_pred, y=sqrt_abs, alpha=0.6, color=point_color, ax=axes[1, 1])
    
    # Add lowess smoother
    try:
        smoothed = lowess(sqrt_abs, y_pred, frac=0.6)
        axes[1, 1].plot(smoothed[:, 0], smoothed[:, 1], color='#2ECC71', linewidth=2)
    except:
        pass
    
    axes[1, 1].set_title('Scale-Location', fontweight='bold')
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Standardized Residuals|')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


# =============================================================================
# Main Endpoint
# =============================================================================

@router.post("/simple-regression")
def simple_regression_analysis(req: SimpleRegressionRequest):
    """
    Perform Simple Linear Regression analysis.
    
    Simple regression models the relationship between one independent variable (X)
    and one dependent variable (Y) using the equation: Y = β₀ + β₁X + ε
    
    Returns:
    - Model fit metrics (R², RMSE, MAE)
    - Coefficient estimates with significance tests
    - Diagnostic tests for assumptions
    - Visualization plots
    """
    try:
        df = pd.DataFrame(req.data)
        target_var = req.targetVar
        feature = req.feature
        confidence_level = req.confidence_level
        
        # Validate columns exist
        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data")
        if feature not in df.columns:
            raise ValueError(f"Feature variable '{feature}' not found in data")
        if target_var == feature:
            raise ValueError("Target and feature variables must be different")
        
        # Store original names for display
        original_target = target_var
        original_feature = feature
        
        # Clean column names for statsmodels
        sanitized = {col: re.sub(r'[^A-Za-z0-9_]', '_', str(col)) for col in df.columns}
        original_names = {v: k for k, v in sanitized.items()}
        df.rename(columns=sanitized, inplace=True)
        target_clean = sanitized[target_var]
        feature_clean = sanitized[feature]
        
        # Convert to numeric
        df[target_clean] = pd.to_numeric(df[target_clean], errors='coerce')
        df[feature_clean] = pd.to_numeric(df[feature_clean], errors='coerce')
        
        # Handle missing values
        original_len = len(df)
        df_clean = df.dropna(subset=[target_clean, feature_clean])
        n_dropped = original_len - len(df_clean)
        
        if len(df_clean) < 10:
            raise ValueError(f"Not enough valid observations after removing missing values (n={len(df_clean)}, minimum=10)")
        
        y = df_clean[target_clean]
        X = df_clean[[feature_clean]]
        n = len(y)
        
        # =================================================================
        # Fit OLS Model
        # =================================================================
        X_const = sm.add_constant(X)
        sm_model = sm.OLS(y, X_const).fit()
        y_pred = sm_model.predict(X_const)
        residuals = sm_model.resid
        
        # =================================================================
        # Calculate Metrics
        # =================================================================
        mse = mean_squared_error(y, y_pred)
        metrics = {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred)),
            'adj_r2': float(sm_model.rsquared_adj),
            'n_observations': n
        }
        
        # =================================================================
        # Correlation Analysis
        # =================================================================
        correlation, corr_pvalue = stats.pearsonr(X.iloc[:, 0], y)
        
        # =================================================================
        # Coefficient Diagnostics
        # =================================================================
        def clean_name(name: str) -> str:
            """Restore original variable names"""
            name = re.sub(r'Q\("([^"]+)"\)', r'\1', str(name).strip())
            return original_names.get(name, name)
        
        # Get confidence intervals
        alpha = 1 - confidence_level
        conf_int = sm_model.conf_int(alpha=alpha)
        
        coefficient_tests = {
            'params': {},
            'pvalues': {},
            'bse': {},
            'tvalues': {},
            'conf_int_lower': {},
            'conf_int_upper': {}
        }
        
        for idx, (name, value) in enumerate(sm_model.params.items()):
            clean = clean_name(name)
            coefficient_tests['params'][clean] = float(value)
            coefficient_tests['pvalues'][clean] = float(sm_model.pvalues[name])
            coefficient_tests['bse'][clean] = float(sm_model.bse[name])
            coefficient_tests['tvalues'][clean] = float(sm_model.tvalues[name])
            coefficient_tests['conf_int_lower'][clean] = float(conf_int.iloc[idx, 0])
            coefficient_tests['conf_int_upper'][clean] = float(conf_int.iloc[idx, 1])
        
        diagnostics = {
            'f_statistic': float(sm_model.fvalue) if sm_model.fvalue else None,
            'f_pvalue': float(sm_model.f_pvalue) if sm_model.f_pvalue else None,
            'coefficient_tests': coefficient_tests,
            'durbin_watson': float(durbin_watson(residuals)),
            'correlation': {
                'pearson_r': float(correlation),
                'p_value': float(corr_pvalue)
            },
            'confidence_level': confidence_level
        }
        
        # =================================================================
        # Normality Tests
        # =================================================================
        normality_ok = True
        try:
            jb_stat, jb_p, _, _ = jarque_bera(residuals)
            sw_stat, sw_p = stats.shapiro(residuals) if n <= 5000 else (None, None)
            
            diagnostics['normality_tests'] = {
                'jarque_bera': {'statistic': float(jb_stat), 'p_value': float(jb_p)},
            }
            if sw_stat is not None:
                diagnostics['normality_tests']['shapiro_wilk'] = {
                    'statistic': float(sw_stat), 
                    'p_value': float(sw_p)
                }
            
            # Check if normality assumption is violated
            normality_ok = jb_p > 0.05
        except Exception:
            diagnostics['normality_tests'] = {}
        
        # =================================================================
        # Heteroscedasticity Test
        # =================================================================
        homoscedasticity_ok = True
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, sm_model.model.exog)
            diagnostics['heteroscedasticity_tests'] = {
                'breusch_pagan': {
                    'statistic': float(bp_stat), 
                    'p_value': float(bp_p)
                }
            }
            homoscedasticity_ok = bp_p > 0.05
        except Exception:
            diagnostics['heteroscedasticity_tests'] = {}
        
        # =================================================================
        # Generate Interpretation
        # =================================================================
        slope = coefficient_tests['params'].get(original_feature, 0)
        slope_pvalue = coefficient_tests['pvalues'].get(original_feature, 1)
        
        interpretation = generate_interpretation(
            r2=metrics['r2'],
            adj_r2=metrics['adj_r2'],
            f_pvalue=diagnostics.get('f_pvalue', 1),
            slope=slope,
            slope_pvalue=slope_pvalue,
            correlation=correlation,
            feature_name=original_feature,
            target_name=original_target,
            n=n,
            normality_ok=normality_ok,
            homoscedasticity_ok=homoscedasticity_ok
        )
        
        # =================================================================
        # Generate Plot
        # =================================================================
        plot = generate_plot(
            y_true=y, 
            y_pred=y_pred, 
            residuals=residuals, 
            sm_model=sm_model, 
            X=X,
            feature_name=original_feature,
            target_name=original_target
        )
        
        # =================================================================
        # Build Response
        # =================================================================
        return _to_native({
            'results': {
                'model_name': 'Simple Linear Regression',
                'model_type': 'simple',
                'features': [original_feature],
                'target': original_target,
                'metrics': {'all_data': metrics},
                'diagnostics': diagnostics,
                'interpretation': interpretation,
                'equation': f"{original_target} = {coefficient_tests['params'].get('const', 0):.4f} + {slope:.4f} × {original_feature}",
                'n_dropped': n_dropped
            },
            'model_name': 'simple',
            'model_type': 'regression',
            'plot': plot
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
