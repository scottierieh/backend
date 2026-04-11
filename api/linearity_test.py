from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class LinearityTestRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    dependent: str = Field(...)
    independents: List[str] = Field(...)


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
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


def rainbow_test(X, y, fitted):
    """Rainbow test for linearity - simplified version"""
    n = len(y)
    
    # Sort by fitted values
    sort_idx = np.argsort(fitted)
    y_sorted = y[sort_idx]
    X_sorted = X[sort_idx]
    
    # Use middle portion (central 50%)
    start = n // 4
    end = n - n // 4
    
    if end - start < 5:
        return 0.0, 1.0
    
    y_center = y_sorted[start:end]
    X_center = X_sorted[start:end]
    
    try:
        # Fit model on center portion
        model_center = LinearRegression()
        model_center.fit(X_center, y_center)
        fitted_center = model_center.predict(X_center)
        rss_center = np.sum((y_center - fitted_center) ** 2)
        
        # Fit model on full data
        model_full = LinearRegression()
        model_full.fit(X, y)
        fitted_full = model_full.predict(X)
        rss_full = np.sum((y - fitted_full) ** 2)
        
        # Calculate F statistic
        df_center = len(y_center) - X_center.shape[1] - 1
        df_full = n - X.shape[1] - 1
        
        if df_center <= 0 or df_full <= 0 or rss_center <= 0:
            return 0.0, 1.0
        
        # F = (RSS_full - RSS_center) / (df_full - df_center) / (RSS_center / df_center)
        df_diff = df_full - df_center
        if df_diff <= 0:
            return 0.0, 1.0
            
        f_stat = ((rss_full - rss_center) / df_diff) / (rss_center / df_center)
        
        if f_stat < 0:
            f_stat = 0
            
        p_value = 1 - stats.f.cdf(f_stat, df_diff, df_center)
        
        return float(f_stat), float(p_value)
    except:
        return 0.0, 1.0


def calculate_metrics(residuals, fitted, X, y):
    metrics = {}
    
    # Residual-fitted correlation
    corr, p_value = stats.pearsonr(residuals, fitted)
    metrics['residual_fitted_corr'] = float(corr)
    metrics['residual_fitted_corr_pvalue'] = float(p_value)
    
    # Runs test
    signs = np.sign(residuals)
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)
    n = len(residuals)
    
    if n_pos > 0 and n_neg > 0:
        expected_runs = (2 * n_pos * n_neg) / n + 1
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
        z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
    else:
        z_runs, p_runs, expected_runs = 0, 1.0, runs
    
    metrics['runs_observed'] = int(runs)
    metrics['runs_expected'] = float(expected_runs)
    metrics['runs_z_statistic'] = float(z_runs)
    metrics['runs_p_value'] = float(p_runs)
    
    # Rainbow test
    rainbow_f, rainbow_p = rainbow_test(X, y, fitted)
    metrics['rainbow_f_statistic'] = rainbow_f
    metrics['rainbow_p_value'] = rainbow_p
    
    # Curvature test
    x_norm = (fitted - np.mean(fitted)) / (np.std(fitted) + 1e-10)
    try:
        linear_coef = np.polyfit(x_norm, residuals, 1)
        linear_pred = np.polyval(linear_coef, x_norm)
        ss_linear = np.sum((residuals - linear_pred)**2)
        
        quad_coef = np.polyfit(x_norm, residuals, 2)
        quad_pred = np.polyval(quad_coef, x_norm)
        ss_quad = np.sum((residuals - quad_pred)**2)
        
        df_quad = n - 3
        if df_quad > 0 and ss_quad > 0:
            f_curve = ((ss_linear - ss_quad) / 1) / (ss_quad / df_quad)
            p_curve = 1 - stats.f.cdf(f_curve, 1, df_quad)
        else:
            f_curve, p_curve = 0, 1.0
        
        metrics['curvature_f_statistic'] = float(f_curve)
        metrics['curvature_p_value'] = float(p_curve)
        metrics['quadratic_coef'] = float(quad_coef[0])
    except:
        metrics['curvature_f_statistic'] = 0
        metrics['curvature_p_value'] = 1.0
        metrics['quadratic_coef'] = 0
    
    metrics['residual_mean'] = float(np.mean(residuals))
    metrics['residual_std'] = float(np.std(residuals))
    
    return metrics


def generate_insights(metrics, n):
    insights = []
    recommendations = []
    
    corr = abs(metrics['residual_fitted_corr'])
    if corr > 0.3:
        insights.append({'type': 'warning', 'title': 'Strong Residual-Fitted Correlation', 'description': f'Correlation = {metrics["residual_fitted_corr"]:.3f}. Suggests non-linearity.'})
        recommendations.append('Consider adding polynomial terms or transforming variables.')
    elif corr > 0.1:
        insights.append({'type': 'info', 'title': 'Moderate Correlation', 'description': f'Correlation = {metrics["residual_fitted_corr"]:.3f}. Some pattern may exist.'})
    else:
        insights.append({'type': 'info', 'title': 'Low Correlation ✓', 'description': f'Correlation = {metrics["residual_fitted_corr"]:.3f}. Supports linearity.'})
    
    if metrics['runs_p_value'] < 0.05:
        insights.append({'type': 'warning', 'title': 'Non-Random Residual Pattern', 'description': f'Runs test p = {metrics["runs_p_value"]:.4f}. Residuals show systematic patterns.'})
        recommendations.append('Check for missing predictors or non-linear relationships.')
    else:
        insights.append({'type': 'info', 'title': 'Random Residual Pattern ✓', 'description': f'Runs test p = {metrics["runs_p_value"]:.4f}. Supports linearity.'})
    
    if metrics['curvature_p_value'] < 0.05:
        insights.append({'type': 'warning', 'title': 'Curvature Detected', 'description': f'Curvature test p = {metrics["curvature_p_value"]:.4f}. Relationship may be non-linear.'})
        recommendations.append('Try polynomial regression or variable transformations.')
    else:
        insights.append({'type': 'info', 'title': 'No Significant Curvature ✓', 'description': f'Curvature test p = {metrics["curvature_p_value"]:.4f}.'})
    
    # Overall assessment
    violations = 0
    if corr > 0.3: violations += 2
    elif corr > 0.1: violations += 1
    if metrics['runs_p_value'] < 0.05: violations += 2
    if metrics['curvature_p_value'] < 0.05: violations += 2
    
    if violations == 0:
        insights.append({'type': 'info', 'title': '✅ Linearity Assumption Satisfied', 'description': 'All tests pass. Linear model appears appropriate.'})
    elif violations <= 2:
        insights.append({'type': 'info', 'title': '⚠️ Minor Linearity Concerns', 'description': 'Some weak evidence of non-linearity.'})
    else:
        insights.append({'type': 'warning', 'title': '🚨 Linearity Violation', 'description': 'Strong evidence of non-linearity. Consider non-linear models.'})
    
    return insights, list(dict.fromkeys(recommendations))


def create_plots(fitted, residuals):
    plots = {}
    
    # Residual vs Fitted
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(fitted, residuals, alpha=0.6, c='#4C72B0', edgecolors='white', s=60)
    ax.axhline(0, color='#C44E52', linestyle='--', linewidth=1.5)
    
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.6, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='#55A868', linewidth=2.5, label='LOWESS')
        ax.legend()
    except:
        pass
    
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual vs Fitted Plot', fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    plots['residual_vs_fitted'] = base64.b64encode(buf.read()).decode('utf-8')
    
    # Scale-Location
    fig, ax = plt.subplots(figsize=(10, 6))
    std_resid = residuals / (np.std(residuals) + 1e-10)
    sqrt_abs = np.sqrt(np.abs(std_resid))
    ax.scatter(fitted, sqrt_abs, alpha=0.6, c='#8172B3', edgecolors='white', s=60)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Standardized Residuals|')
    ax.set_title('Scale-Location Plot', fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    plots['scale_location'] = base64.b64encode(buf.read()).decode('utf-8')
    
    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins='auto', density=True, alpha=0.7, color='#4C72B0', edgecolor='white')
    mu, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), color='#C44E52', linewidth=2.5, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Residuals', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    plots['residual_histogram'] = base64.b64encode(buf.read()).decode('utf-8')
    
    return plots


@router.post("/linearity-test")
def linearity_test(req: LinearityTestRequest):
    try:
        df = pd.DataFrame(req.data)
        dependent = req.dependent if isinstance(req.dependent, str) else req.dependent[0]
        independents = req.independents if isinstance(req.independents, list) else [req.independents]
        
        all_vars = [dependent] + independents
        df_clean = df[all_vars].dropna()
        
        for col in all_vars:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean = df_clean.dropna()
        
        n = len(df_clean)
        if n < 5:
            raise ValueError(f"Need at least 5 observations, got {n}")
        
        X = df_clean[independents].values.astype(float)
        y = df_clean[dependent].values.astype(float).ravel()
        
        model = LinearRegression()
        model.fit(X, y)
        
        fitted = model.predict(X).flatten()
        residuals = (y - fitted).flatten()
        
        metrics = calculate_metrics(residuals, fitted, X, y)
        metrics['n_observations'] = n
        metrics['n_predictors'] = len(independents)
        metrics['r_squared'] = float(model.score(X, y))
        metrics['coefficients'] = {name: float(coef) for name, coef in zip(independents, model.coef_)}
        metrics['intercept'] = float(model.intercept_)
        
        insights, recommendations = generate_insights(metrics, n)
        plots = create_plots(fitted, residuals)
        
        # Residual data
        residual_data = [{'index': i+1, 'fitted': float(fitted[i]), 'residual': float(residuals[i]), 'std_residual': float(residuals[i] / (np.std(residuals) + 1e-10))} for i in range(min(n, 100))]
        
        return _to_native({
            'metrics': metrics,
            'insights': insights,
            'recommendations': recommendations,
            'plots': plots,
            'residual_data': residual_data,
            'model_summary': {
                'dependent': dependent,
                'independents': independents,
                'equation': f"{dependent} = {metrics['intercept']:.4f} + " + " + ".join([f"{metrics['coefficients'][v]:.4f}*{v}" for v in independents])
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
