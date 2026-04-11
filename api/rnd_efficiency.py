"""
R&D Investment Efficiency Analysis API
- DEA (Data Envelopment Analysis)
- SFA (Stochastic Frontier Analysis)
- ROI Analysis
- Cost-Performance Regression
- Marginal Effect Analysis
- Nonlinear Regression (Optimal Investment)
- Threshold Effect Analysis
- Economies of Scale Test
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

warnings.filterwarnings('ignore')
router = APIRouter()
sns.set_theme(style="whitegrid")

class RndEfficiencyRequest(BaseModel):
    data: List[Dict[str, Any]]
    dmu_var: str  # Decision Making Unit (company/project name)
    input_vars: List[str]  # Input variables (R&D investment, personnel, etc.)
    output_vars: List[str]  # Output variables (patents, revenue, etc.)
    cost_var: Optional[str] = None  # Cost variable for ROI
    performance_var: Optional[str] = None  # Performance variable
    threshold_var: Optional[str] = None  # Variable for threshold analysis

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def safe_float(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return float(val)

# ============ DEA Analysis ============
def analyze_dea(df, dmu_var, input_vars, output_vars):
    """
    Data Envelopment Analysis (CCR Model - Constant Returns to Scale)
    """
    try:
        dmus = df[dmu_var].tolist()
        inputs = df[input_vars].values
        outputs = df[output_vars].values
        n_dmus = len(dmus)
        
        efficiencies = []
        
        for i in range(n_dmus):
            # Simplified DEA: efficiency = weighted outputs / weighted inputs
            # Using equal weights for simplicity
            input_sum = inputs[i].sum() if inputs[i].sum() > 0 else 1
            output_sum = outputs[i].sum()
            efficiency = output_sum / input_sum
            efficiencies.append(efficiency)
        
        # Normalize to max = 1
        max_eff = max(efficiencies) if max(efficiencies) > 0 else 1
        normalized_eff = [e / max_eff for e in efficiencies]
        
        results = []
        for i, dmu in enumerate(dmus):
            results.append({
                'dmu': str(dmu),
                'efficiency': safe_float(normalized_eff[i]),
                'efficient': normalized_eff[i] >= 0.99,
                'rank': None,
                'input_total': safe_float(inputs[i].sum()),
                'output_total': safe_float(outputs[i].sum())
            })
        
        # Add ranks
        sorted_results = sorted(results, key=lambda x: x['efficiency'], reverse=True)
        for rank, r in enumerate(sorted_results, 1):
            r['rank'] = rank
        
        # Statistics
        eff_values = [r['efficiency'] for r in results]
        n_efficient = sum(1 for r in results if r['efficient'])
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Efficiency bar chart
        sorted_by_eff = sorted(results, key=lambda x: x['efficiency'])
        names = [r['dmu'][:15] for r in sorted_by_eff]
        effs = [r['efficiency'] for r in sorted_by_eff]
        colors = ['#4CAF50' if e >= 0.99 else '#FF9800' if e >= 0.8 else '#F44336' for e in effs]
        axes[0].barh(names, effs, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Efficient Frontier')
        axes[0].axvline(x=0.8, color='orange', linestyle=':', linewidth=1.5, label='80% Threshold')
        axes[0].set_xlabel('Efficiency Score'); axes[0].set_title('DEA Efficiency Scores')
        axes[0].legend(); axes[0].set_xlim(0, 1.1)
        
        # Input-Output scatter
        if len(input_vars) >= 1 and len(output_vars) >= 1:
            axes[1].scatter(inputs[:, 0], outputs[:, 0], c=normalized_eff, cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
            for i, dmu in enumerate(dmus):
                axes[1].annotate(str(dmu)[:8], (inputs[i, 0], outputs[i, 0]), fontsize=8)
            axes[1].set_xlabel(input_vars[0]); axes[1].set_ylabel(output_vars[0])
            axes[1].set_title('Input vs Output (Color = Efficiency)')
            plt.colorbar(axes[1].collections[0], ax=axes[1], label='Efficiency')
        
        plt.tight_layout()
        
        return {
            'method': 'DEA (Data Envelopment Analysis)',
            'model': 'CCR (Constant Returns to Scale)',
            'n_dmus': n_dmus,
            'n_efficient': n_efficient,
            'efficiency_rate': safe_float(n_efficient / n_dmus * 100),
            'mean_efficiency': safe_float(np.mean(eff_values)),
            'std_efficiency': safe_float(np.std(eff_values)),
            'min_efficiency': safe_float(min(eff_values)),
            'max_efficiency': safe_float(max(eff_values)),
            'results': sorted(results, key=lambda x: x['rank']),
            'efficient_dmus': [r['dmu'] for r in results if r['efficient']],
            'inefficient_dmus': [r['dmu'] for r in results if not r['efficient']],
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ SFA Analysis ============
def analyze_sfa(df, dmu_var, input_vars, output_vars):
    """
    Stochastic Frontier Analysis (Production Function Approach)
    """
    try:
        if len(output_vars) < 1 or len(input_vars) < 1:
            return None
            
        y = np.log(df[output_vars[0]].replace(0, 0.001).values)
        X = np.log(df[input_vars].replace(0, 0.001).values)
        X = sm.add_constant(X)
        
        # OLS as base
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        
        # Estimate inefficiency (negative skewness in residuals indicates inefficiency)
        # Simplified: use normalized residuals as efficiency proxy
        max_resid = residuals.max()
        technical_efficiency = np.exp(residuals - max_resid)
        
        dmus = df[dmu_var].tolist()
        results = []
        for i, dmu in enumerate(dmus):
            results.append({
                'dmu': str(dmu),
                'efficiency': safe_float(technical_efficiency[i]),
                'log_output': safe_float(y[i]),
                'predicted': safe_float(model.fittedvalues[i]),
                'residual': safe_float(residuals[i])
            })
        
        # Sort by efficiency
        results = sorted(results, key=lambda x: x['efficiency'], reverse=True)
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Efficiency distribution
        eff_vals = [r['efficiency'] for r in results]
        axes[0].hist(eff_vals, bins=15, color='#4A90A4', alpha=0.8, edgecolor='black')
        axes[0].axvline(np.mean(eff_vals), color='red', linestyle='--', label=f'Mean: {np.mean(eff_vals):.3f}')
        axes[0].set_xlabel('Technical Efficiency'); axes[0].set_ylabel('Frequency')
        axes[0].set_title('SFA Efficiency Distribution'); axes[0].legend()
        
        # Actual vs Frontier
        axes[1].scatter(model.fittedvalues, y, alpha=0.7, c='#4A90A4', edgecolors='black')
        axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Frontier')
        axes[1].set_xlabel('Predicted (Frontier)'); axes[1].set_ylabel('Actual (Log Output)')
        axes[1].set_title('Actual vs Frontier'); axes[1].legend()
        
        plt.tight_layout()
        
        # Coefficients interpretation
        coef_names = ['Intercept'] + input_vars
        coefficients = [{'variable': coef_names[i], 'coefficient': safe_float(model.params[i]),
                        'std_error': safe_float(model.bse[i]), 'p_value': safe_float(model.pvalues[i]),
                        'significant': bool(model.pvalues[i] < 0.05)} for i in range(len(model.params))]
        
        return {
            'method': 'SFA (Stochastic Frontier Analysis)',
            'model': 'Cobb-Douglas Production Function',
            'r_squared': safe_float(model.rsquared),
            'adj_r_squared': safe_float(model.rsquared_adj),
            'mean_efficiency': safe_float(np.mean(eff_vals)),
            'std_efficiency': safe_float(np.std(eff_vals)),
            'coefficients': coefficients,
            'results': results,
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ ROI Analysis ============
def analyze_roi(df, dmu_var, cost_var, performance_var):
    """
    Return on Investment Analysis
    """
    try:
        if not cost_var or not performance_var:
            return None
            
        df_clean = df[[dmu_var, cost_var, performance_var]].dropna()
        df_clean['roi'] = (df_clean[performance_var] / df_clean[cost_var].replace(0, np.nan)) * 100
        df_clean = df_clean.dropna()
        
        results = []
        for _, row in df_clean.iterrows():
            results.append({
                'dmu': str(row[dmu_var]),
                'cost': safe_float(row[cost_var]),
                'performance': safe_float(row[performance_var]),
                'roi': safe_float(row['roi'])
            })
        
        results = sorted(results, key=lambda x: x['roi'] if x['roi'] else 0, reverse=True)
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
        
        roi_values = [r['roi'] for r in results if r['roi'] is not None]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROI bar chart
        names = [r['dmu'][:12] for r in results[:15]]
        rois = [r['roi'] for r in results[:15]]
        colors = ['#4CAF50' if r > np.mean(roi_values) else '#FF9800' if r > np.median(roi_values) else '#F44336' for r in rois]
        axes[0].barh(names[::-1], rois[::-1], color=colors[::-1], alpha=0.8, edgecolor='black')
        axes[0].axvline(np.mean(roi_values), color='blue', linestyle='--', label=f'Mean: {np.mean(roi_values):.1f}%')
        axes[0].set_xlabel('ROI (%)'); axes[0].set_title('ROI Rankings (Top 15)')
        axes[0].legend()
        
        # Cost vs Performance scatter
        costs = [r['cost'] for r in results]
        perfs = [r['performance'] for r in results]
        axes[1].scatter(costs, perfs, c=roi_values[:len(costs)], cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
        axes[1].set_xlabel(cost_var); axes[1].set_ylabel(performance_var)
        axes[1].set_title('Cost vs Performance')
        plt.colorbar(axes[1].collections[0], ax=axes[1], label='ROI (%)')
        
        # Add trend line
        z = np.polyfit(costs, perfs, 1)
        p = np.poly1d(z)
        axes[1].plot(sorted(costs), p(sorted(costs)), 'r--', alpha=0.8, label='Trend')
        axes[1].legend()
        
        plt.tight_layout()
        
        return {
            'method': 'ROI (Return on Investment)',
            'n_observations': len(results),
            'mean_roi': safe_float(np.mean(roi_values)),
            'median_roi': safe_float(np.median(roi_values)),
            'std_roi': safe_float(np.std(roi_values)),
            'min_roi': safe_float(min(roi_values)),
            'max_roi': safe_float(max(roi_values)),
            'top_performers': results[:5],
            'bottom_performers': results[-5:] if len(results) >= 5 else results,
            'results': results,
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ Cost-Performance Regression ============
def analyze_cost_performance_regression(df, cost_var, performance_var, input_vars):
    """
    Regression analysis of cost vs performance
    """
    try:
        if not cost_var or not performance_var:
            return None
        
        predictors = [cost_var] + [v for v in input_vars if v != cost_var]
        df_clean = df[[performance_var] + predictors].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_clean) < len(predictors) + 2:
            return None
        
        X = df_clean[predictors]
        y = df_clean[performance_var]
        X_const = sm.add_constant(X)
        
        model = sm.OLS(y, X_const).fit()
        
        # Standardized coefficients
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        y_std = (y - y.mean()) / y.std()
        beta_model = LinearRegression().fit(X_std, y_std)
        
        coefficients = []
        for i, var in enumerate(['const'] + predictors):
            coefficients.append({
                'variable': var,
                'coefficient': safe_float(model.params[i]),
                'std_error': safe_float(model.bse[i]),
                't_value': safe_float(model.tvalues[i]),
                'p_value': safe_float(model.pvalues[i]),
                'significant': bool(model.pvalues[i] < 0.05),
                'beta': safe_float(beta_model.coef_[i-1]) if i > 0 else None
            })
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Coefficient plot
        pred_coefs = [c for c in coefficients if c['variable'] != 'const' and c['beta'] is not None]
        if pred_coefs:
            vars_n = [c['variable'] for c in pred_coefs]
            betas = [c['beta'] for c in pred_coefs]
            colors = ['#4CAF50' if b > 0 else '#F44336' for b in betas]
            axes[0].barh(vars_n, betas, color=colors, alpha=0.8, edgecolor='black')
            axes[0].axvline(0, color='black', linewidth=0.5)
            axes[0].set_xlabel('Standardized Coefficient (β)')
            axes[0].set_title('Cost-Performance Drivers')
        
        # Actual vs Predicted
        axes[1].scatter(model.fittedvalues, y, alpha=0.7, c='#4A90A4', edgecolors='black')
        axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
        axes[1].set_title(f'Model Fit (R² = {model.rsquared:.3f})'); axes[1].legend()
        
        plt.tight_layout()
        
        return {
            'method': 'Cost-Performance Regression',
            'dependent_var': performance_var,
            'r_squared': safe_float(model.rsquared),
            'adj_r_squared': safe_float(model.rsquared_adj),
            'f_statistic': safe_float(model.fvalue),
            'f_pvalue': safe_float(model.f_pvalue),
            'n_observations': len(df_clean),
            'coefficients': coefficients,
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ Marginal Effect Analysis ============
def analyze_marginal_effects(df, cost_var, performance_var):
    """
    Marginal effect of additional investment
    """
    try:
        if not cost_var or not performance_var:
            return None
        
        df_clean = df[[cost_var, performance_var]].apply(pd.to_numeric, errors='coerce').dropna()
        df_clean = df_clean.sort_values(cost_var)
        
        x = df_clean[cost_var].values
        y = df_clean[performance_var].values
        
        # Quadratic fit for marginal effects
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, y)
        
        # Marginal effect = derivative = b1 + 2*b2*x
        b0, b1, b2 = model.intercept_, model.coef_[1], model.coef_[2]
        
        # Calculate marginal effects at different investment levels
        x_range = np.linspace(x.min(), x.max(), 100)
        marginal_effects = b1 + 2 * b2 * x_range
        predicted = model.predict(poly.transform(x_range.reshape(-1, 1)))
        
        # Find optimal investment (where marginal effect = 0, if b2 < 0)
        optimal_investment = None
        if b2 < 0:
            optimal_investment = -b1 / (2 * b2)
            if optimal_investment < x.min() or optimal_investment > x.max():
                optimal_investment = None
        
        # Diminishing returns point (where marginal effect starts decreasing significantly)
        diminishing_point = x_range[np.argmax(marginal_effects < marginal_effects[0] * 0.5)] if b2 < 0 else None
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Performance curve
        axes[0].scatter(x, y, alpha=0.6, c='#4A90A4', edgecolors='black', label='Actual')
        axes[0].plot(x_range, predicted, 'r-', linewidth=2, label='Fitted Curve')
        if optimal_investment:
            axes[0].axvline(optimal_investment, color='green', linestyle='--', label=f'Optimal: {optimal_investment:.0f}')
        axes[0].set_xlabel(cost_var); axes[0].set_ylabel(performance_var)
        axes[0].set_title('Investment-Performance Curve'); axes[0].legend()
        
        # Marginal effects
        axes[1].plot(x_range, marginal_effects, 'b-', linewidth=2)
        axes[1].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[1].fill_between(x_range, marginal_effects, 0, where=(marginal_effects > 0), alpha=0.3, color='green', label='Positive Returns')
        axes[1].fill_between(x_range, marginal_effects, 0, where=(marginal_effects < 0), alpha=0.3, color='red', label='Negative Returns')
        axes[1].set_xlabel(cost_var); axes[1].set_ylabel('Marginal Effect')
        axes[1].set_title('Marginal Effect of Investment'); axes[1].legend()
        
        plt.tight_layout()
        
        return {
            'method': 'Marginal Effect Analysis',
            'model': 'Quadratic Regression',
            'coefficients': {
                'intercept': safe_float(b0),
                'linear': safe_float(b1),
                'quadratic': safe_float(b2)
            },
            'optimal_investment': safe_float(optimal_investment),
            'diminishing_returns_point': safe_float(diminishing_point),
            'marginal_effect_at_mean': safe_float(b1 + 2 * b2 * x.mean()),
            'marginal_effect_at_min': safe_float(b1 + 2 * b2 * x.min()),
            'marginal_effect_at_max': safe_float(b1 + 2 * b2 * x.max()),
            'interpretation': 'Diminishing returns detected' if b2 < 0 else 'Increasing returns detected',
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ Nonlinear Regression (Optimal Investment) ============
def analyze_nonlinear_optimization(df, cost_var, performance_var):
    """
    Find optimal investment level using nonlinear regression
    """
    try:
        if not cost_var or not performance_var:
            return None
        
        df_clean = df[[cost_var, performance_var]].apply(pd.to_numeric, errors='coerce').dropna()
        x = df_clean[cost_var].values
        y = df_clean[performance_var].values
        
        # Try different polynomial degrees
        results = []
        for degree in [1, 2, 3]:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(x.reshape(-1, 1))
            model = LinearRegression().fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            # Calculate metrics
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            n = len(y)
            k = degree + 1
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
            aic = n * np.log(ss_res / n) + 2 * k
            
            results.append({
                'degree': degree,
                'r_squared': safe_float(r2),
                'adj_r_squared': safe_float(adj_r2),
                'aic': safe_float(aic),
                'coefficients': [safe_float(c) for c in model.coef_] + [safe_float(model.intercept_)]
            })
        
        # Select best model by AIC
        best_model = min(results, key=lambda x: x['aic'])
        best_degree = best_model['degree']
        
        # Refit best model
        poly = PolynomialFeatures(degree=best_degree)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, y)
        
        # Find optimal point
        x_range = np.linspace(x.min(), x.max(), 1000)
        y_pred = model.predict(poly.transform(x_range.reshape(-1, 1)))
        optimal_idx = np.argmax(y_pred)
        optimal_x = x_range[optimal_idx]
        optimal_y = y_pred[optimal_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.6, c='#4A90A4', edgecolors='black', s=80, label='Actual Data')
        ax.plot(x_range, y_pred, 'r-', linewidth=2.5, label=f'Polynomial (degree={best_degree})')
        ax.scatter([optimal_x], [optimal_y], c='green', s=200, marker='*', zorder=5, label=f'Optimal: ({optimal_x:.0f}, {optimal_y:.1f})')
        ax.set_xlabel(cost_var, fontsize=12)
        ax.set_ylabel(performance_var, fontsize=12)
        ax.set_title('Optimal Investment Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        
        return {
            'method': 'Nonlinear Regression Optimization',
            'best_model_degree': best_degree,
            'model_comparison': results,
            'optimal_investment': safe_float(optimal_x),
            'expected_performance': safe_float(optimal_y),
            'current_mean_investment': safe_float(x.mean()),
            'current_mean_performance': safe_float(y.mean()),
            'potential_improvement': safe_float((optimal_y - y.mean()) / y.mean() * 100) if y.mean() != 0 else None,
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ Threshold Effect Analysis ============
def analyze_threshold_effects(df, cost_var, performance_var):
    """
    Analyze threshold effects in investment-performance relationship
    """
    try:
        if not cost_var or not performance_var:
            return None
        
        df_clean = df[[cost_var, performance_var]].apply(pd.to_numeric, errors='coerce').dropna()
        df_clean = df_clean.sort_values(cost_var)
        x = df_clean[cost_var].values
        y = df_clean[performance_var].values
        
        # Test different threshold points
        potential_thresholds = np.percentile(x, [25, 33, 50, 66, 75])
        best_threshold = None
        best_improvement = 0
        threshold_results = []
        
        for threshold in potential_thresholds:
            below = df_clean[df_clean[cost_var] <= threshold]
            above = df_clean[df_clean[cost_var] > threshold]
            
            if len(below) < 5 or len(above) < 5:
                continue
            
            # Fit separate regressions
            X_below = sm.add_constant(below[cost_var])
            X_above = sm.add_constant(above[cost_var])
            
            model_below = sm.OLS(below[performance_var], X_below).fit()
            model_above = sm.OLS(above[performance_var], X_above).fit()
            
            # Combined R² improvement
            y_pred_below = model_below.predict(X_below)
            y_pred_above = model_above.predict(X_above)
            y_pred_combined = np.concatenate([y_pred_below, y_pred_above])
            y_actual = np.concatenate([below[performance_var].values, above[performance_var].values])
            
            ss_res = np.sum((y_actual - y_pred_combined) ** 2)
            ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
            r2_split = 1 - (ss_res / ss_tot)
            
            # Single model R²
            X_full = sm.add_constant(df_clean[cost_var])
            model_full = sm.OLS(df_clean[performance_var], X_full).fit()
            
            improvement = r2_split - model_full.rsquared
            
            threshold_results.append({
                'threshold': safe_float(threshold),
                'n_below': len(below),
                'n_above': len(above),
                'slope_below': safe_float(model_below.params[1]) if len(model_below.params) > 1 else None,
                'slope_above': safe_float(model_above.params[1]) if len(model_above.params) > 1 else None,
                'r2_split': safe_float(r2_split),
                'r2_improvement': safe_float(improvement)
            })
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = threshold
        
        # Plot with best threshold
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter with threshold
        colors = ['#4CAF50' if xi > best_threshold else '#F44336' for xi in x]
        axes[0].scatter(x, y, c=colors, alpha=0.7, edgecolors='black', s=80)
        if best_threshold:
            axes[0].axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold:.0f}')
        axes[0].set_xlabel(cost_var); axes[0].set_ylabel(performance_var)
        axes[0].set_title('Threshold Effect Visualization'); axes[0].legend()
        
        # R² improvement by threshold
        if threshold_results:
            thresholds = [t['threshold'] for t in threshold_results]
            improvements = [t['r2_improvement'] for t in threshold_results]
            axes[1].bar(range(len(thresholds)), improvements, color='#4A90A4', alpha=0.8, edgecolor='black')
            axes[1].set_xticks(range(len(thresholds)))
            axes[1].set_xticklabels([f'{t:.0f}' for t in thresholds])
            axes[1].set_xlabel('Threshold Value'); axes[1].set_ylabel('R² Improvement')
            axes[1].set_title('Model Improvement by Threshold')
        
        plt.tight_layout()
        
        return {
            'method': 'Threshold Effect Analysis',
            'best_threshold': safe_float(best_threshold),
            'r2_improvement': safe_float(best_improvement),
            'threshold_significant': best_improvement > 0.05,
            'threshold_results': threshold_results,
            'interpretation': f"Investment above {best_threshold:.0f} shows different efficiency pattern" if best_threshold else "No significant threshold detected",
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ Economies of Scale Test ============
def analyze_economies_of_scale(df, cost_var, performance_var, dmu_var):
    """
    Test for economies of scale in R&D investment
    """
    try:
        if not cost_var or not performance_var:
            return None
        
        df_clean = df[[dmu_var, cost_var, performance_var]].apply(
            lambda col: pd.to_numeric(col, errors='coerce') if col.name != dmu_var else col
        ).dropna()
        
        x = np.log(df_clean[cost_var].replace(0, 0.001).values)
        y = np.log(df_clean[performance_var].replace(0, 0.001).values)
        
        # Log-log regression: ln(Y) = a + b*ln(X)
        # b > 1: Increasing returns to scale
        # b = 1: Constant returns to scale
        # b < 1: Decreasing returns to scale
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        elasticity = model.params[1]
        se = model.bse[1]
        
        # Test if elasticity is significantly different from 1
        t_stat = (elasticity - 1) / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x) - 2))
        
        if elasticity > 1 and p_value < 0.05:
            scale_type = 'Increasing Returns to Scale'
            interpretation = 'Larger investments yield proportionally higher returns. Consider consolidating or scaling up.'
        elif elasticity < 1 and p_value < 0.05:
            scale_type = 'Decreasing Returns to Scale'
            interpretation = 'Larger investments yield proportionally lower returns. Consider diversifying or limiting investment size.'
        else:
            scale_type = 'Constant Returns to Scale'
            interpretation = 'Investment size does not affect efficiency. Focus on other factors for optimization.'
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Log-log scatter
        axes[0].scatter(x, y, alpha=0.7, c='#4A90A4', edgecolors='black', s=80)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = model.params[0] + model.params[1] * x_line
        axes[0].plot(x_line, y_line, 'r-', linewidth=2, label=f'Elasticity = {elasticity:.3f}')
        axes[0].plot(x_line, model.params[0] + 1 * x_line, 'g--', linewidth=1.5, alpha=0.7, label='CRS (β=1)')
        axes[0].set_xlabel(f'Log({cost_var})'); axes[0].set_ylabel(f'Log({performance_var})')
        axes[0].set_title('Log-Log Regression (Economies of Scale)'); axes[0].legend()
        
        # Scale interpretation visual
        scale_values = ['Decreasing', 'Constant', 'Increasing']
        scale_colors = ['#F44336', '#FF9800', '#4CAF50']
        highlight = 0 if elasticity < 1 else (2 if elasticity > 1 else 1)
        bar_colors = ['lightgray'] * 3
        bar_colors[highlight] = scale_colors[highlight]
        axes[1].barh(scale_values, [1, 1, 1], color=bar_colors, edgecolor='black')
        axes[1].set_xlabel(''); axes[1].set_title(f'Returns to Scale: {scale_type}')
        axes[1].annotate(f'β = {elasticity:.3f}\np = {p_value:.4f}', xy=(0.5, 1.5), fontsize=12, ha='center')
        
        plt.tight_layout()
        
        return {
            'method': 'Economies of Scale Analysis',
            'model': 'Log-Log Regression',
            'elasticity': safe_float(elasticity),
            'std_error': safe_float(se),
            't_statistic': safe_float(t_stat),
            'p_value': safe_float(p_value),
            'r_squared': safe_float(model.rsquared),
            'scale_type': scale_type,
            'significant': bool(p_value < 0.05),
            'interpretation': interpretation,
            'plot': fig_to_base64(fig)
        }
    except Exception as e:
        return {'error': str(e)}

# ============ Generate Conclusion ============
def generate_conclusion(results):
    """Generate comprehensive conclusion with insights"""
    section_summaries = {}
    numbered_findings = []
    
    # DEA Summary
    if results.get('dea_analysis') and not results['dea_analysis'].get('error'):
        dea = results['dea_analysis']
        eff_rate = dea.get('efficiency_rate', 0)
        section_summaries['dea'] = {
            'title': 'Efficiency Analysis (DEA)',
            'summary': f"{dea['n_efficient']} out of {dea['n_dmus']} units are efficient ({eff_rate:.1f}%).",
            'meaning': f"{'Most units operate efficiently. Focus on maintaining best practices.' if eff_rate >= 70 else 'Significant efficiency gaps exist. Benchmark against top performers.' if eff_rate >= 40 else 'Widespread inefficiency detected. Major restructuring may be needed.'}",
            'interpretation': 'POSITIVE' if eff_rate >= 70 else ('MODERATE' if eff_rate >= 40 else 'NEGATIVE')
        }
        numbered_findings.append(f"DEA shows {eff_rate:.0f}% of units are operating at the efficient frontier.")
    
    # SFA Summary
    if results.get('sfa_analysis') and not results['sfa_analysis'].get('error'):
        sfa = results['sfa_analysis']
        mean_eff = sfa.get('mean_efficiency', 0)
        section_summaries['sfa'] = {
            'title': 'Technical Efficiency (SFA)',
            'summary': f"Average technical efficiency is {mean_eff:.1%}.",
            'meaning': f"{'High technical efficiency indicates good resource utilization.' if mean_eff >= 0.8 else 'Moderate efficiency suggests room for operational improvement.' if mean_eff >= 0.6 else 'Low efficiency indicates significant resource waste.'}",
            'interpretation': 'POSITIVE' if mean_eff >= 0.8 else ('MODERATE' if mean_eff >= 0.6 else 'NEGATIVE')
        }
        numbered_findings.append(f"Technical efficiency averages {mean_eff:.1%} across all units.")
    
    # ROI Summary
    if results.get('roi_analysis') and not results['roi_analysis'].get('error'):
        roi = results['roi_analysis']
        mean_roi = roi.get('mean_roi', 0)
        section_summaries['roi'] = {
            'title': 'Return on Investment',
            'summary': f"Average ROI is {mean_roi:.1f}%.",
            'meaning': f"{'Strong returns justify continued investment.' if mean_roi >= 100 else 'Moderate returns suggest selective investment.' if mean_roi >= 50 else 'Low returns indicate need for investment strategy review.'}",
            'interpretation': 'POSITIVE' if mean_roi >= 100 else ('MODERATE' if mean_roi >= 50 else 'NEGATIVE')
        }
        numbered_findings.append(f"Average ROI is {mean_roi:.1f}%, with top performers achieving significantly higher returns.")
    
    # Marginal Effects Summary
    if results.get('marginal_analysis') and not results['marginal_analysis'].get('error'):
        marg = results['marginal_analysis']
        optimal = marg.get('optimal_investment')
        section_summaries['marginal'] = {
            'title': 'Marginal Effect Analysis',
            'summary': f"Optimal investment level: {optimal:.0f}" if optimal else "Linear relationship detected.",
            'meaning': f"{marg.get('interpretation', '')}. {'Investment beyond optimal point shows diminishing returns.' if optimal else 'Each additional investment unit yields consistent returns.'}",
            'interpretation': 'ACTION_NEEDED' if optimal else 'NEUTRAL'
        }
        if optimal:
            numbered_findings.append(f"Optimal investment level is {optimal:.0f}. Beyond this, returns diminish.")
    
    # Economies of Scale Summary
    if results.get('scale_analysis') and not results['scale_analysis'].get('error'):
        scale = results['scale_analysis']
        section_summaries['scale'] = {
            'title': 'Economies of Scale',
            'summary': f"{scale['scale_type']} (elasticity = {scale['elasticity']:.3f}).",
            'meaning': scale['interpretation'],
            'interpretation': 'POSITIVE' if 'Increasing' in scale['scale_type'] else ('NEGATIVE' if 'Decreasing' in scale['scale_type'] else 'NEUTRAL')
        }
        numbered_findings.append(f"Analysis indicates {scale['scale_type'].lower()}.")
    
    # Threshold Summary
    if results.get('threshold_analysis') and not results['threshold_analysis'].get('error'):
        thresh = results['threshold_analysis']
        if thresh.get('threshold_significant'):
            section_summaries['threshold'] = {
                'title': 'Threshold Effects',
                'summary': f"Significant threshold at {thresh['best_threshold']:.0f}.",
                'meaning': thresh['interpretation'],
                'interpretation': 'ACTION_NEEDED'
            }
            numbered_findings.append(thresh['interpretation'])
    
    # Overall recommendation
    recommendation = "Based on the analysis: "
    if results.get('scale_analysis') and 'Increasing' in results['scale_analysis'].get('scale_type', ''):
        recommendation += "Consider scaling up investments to leverage increasing returns. "
    elif results.get('scale_analysis') and 'Decreasing' in results['scale_analysis'].get('scale_type', ''):
        recommendation += "Consider diversifying investments across multiple smaller projects. "
    
    if results.get('marginal_analysis') and results['marginal_analysis'].get('optimal_investment'):
        recommendation += f"Optimal investment level is around {results['marginal_analysis']['optimal_investment']:.0f}. "
    
    if results.get('dea_analysis') and results['dea_analysis'].get('inefficient_dmus'):
        recommendation += f"Focus improvement efforts on underperforming units: {', '.join(results['dea_analysis']['inefficient_dmus'][:3])}."
    
    return {
        'section_summaries': section_summaries,
        'numbered_findings': numbered_findings,
        'recommendation': recommendation,
        'overall_efficiency': 'HIGH' if len([s for s in section_summaries.values() if s['interpretation'] == 'POSITIVE']) > len(section_summaries) / 2 else 'MODERATE'
    }

@router.post("/rnd-efficiency")
async def rnd_efficiency_analysis(request: RndEfficiencyRequest):
    try:
        df = pd.DataFrame(request.data)
        dmu_var = request.dmu_var
        input_vars = request.input_vars
        output_vars = request.output_vars
        cost_var = request.cost_var or (input_vars[0] if input_vars else None)
        performance_var = request.performance_var or (output_vars[0] if output_vars else None)
        
        results = {
            'summary_statistics': {
                'n_observations': len(df),
                'n_inputs': len(input_vars),
                'n_outputs': len(output_vars),
                'dmu_var': dmu_var,
                'input_vars': input_vars,
                'output_vars': output_vars
            }
        }
        
        # Run analyses
        results['dea_analysis'] = analyze_dea(df, dmu_var, input_vars, output_vars)
        results['sfa_analysis'] = analyze_sfa(df, dmu_var, input_vars, output_vars)
        results['roi_analysis'] = analyze_roi(df, dmu_var, cost_var, performance_var)
        results['regression_analysis'] = analyze_cost_performance_regression(df, cost_var, performance_var, input_vars)
        results['marginal_analysis'] = analyze_marginal_effects(df, cost_var, performance_var)
        results['optimization_analysis'] = analyze_nonlinear_optimization(df, cost_var, performance_var)
        results['threshold_analysis'] = analyze_threshold_effects(df, cost_var, performance_var)
        results['scale_analysis'] = analyze_economies_of_scale(df, cost_var, performance_var, dmu_var)
        
        # Generate conclusion
        results['conclusion'] = generate_conclusion(results)
        
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
