from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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


class RegressionRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    targetVar: str = Field(...)
    features: List[str] = Field(...)
    modelType: str = Field(default="multiple")
    selectionMethod: str = Field(default="enter")
    degree: int = Field(default=2)


def _to_native(obj):
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


def stepwise_selection(X, y, method='stepwise', p_enter=0.05, p_remove=0.1):
    initial_cols = X.columns.tolist()
    included = []
    log = []
    
    if method == 'forward':
        while True:
            excluded = list(set(initial_cols) - set(included))
            best_pvalue, best_feature = p_enter, None
            for col in excluded:
                try:
                    model = sm.OLS(y, sm.add_constant(X[included + [col]])).fit()
                    pval = model.pvalues.get(col, 1.0)
                    if pval < best_pvalue:
                        best_pvalue, best_feature = pval, col
                except:
                    continue
            if best_feature:
                included.append(best_feature)
                log.append(f"Add '{best_feature}' (p={best_pvalue:.4f})")
            else:
                break
    elif method == 'backward':
        included = initial_cols.copy()
        while included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.drop('const', errors='ignore')
            worst_pvalue, worst_feature = p_remove, None
            for feat, pval in pvalues.items():
                if pval > worst_pvalue:
                    worst_pvalue, worst_feature = pval, feat
            if worst_feature:
                included.remove(worst_feature)
                log.append(f"Remove '{worst_feature}' (p={worst_pvalue:.4f})")
            else:
                break
    elif method == 'stepwise':
        while True:
            changed = False
            excluded = list(set(initial_cols) - set(included))
            best_pvalue, best_feature = p_enter, None
            for col in excluded:
                try:
                    model = sm.OLS(y, sm.add_constant(X[included + [col]])).fit()
                    pval = model.pvalues.get(col, 1.0)
                    if pval < best_pvalue:
                        best_pvalue, best_feature = pval, col
                except:
                    continue
            if best_feature:
                included.append(best_feature)
                log.append(f"Add '{best_feature}' (p={best_pvalue:.4f})")
                changed = True
            
            if included:
                model = sm.OLS(y, sm.add_constant(X[included])).fit()
                pvalues = model.pvalues.drop('const', errors='ignore')
                worst_pvalue, worst_feature = p_remove, None
                for feat, pval in pvalues.items():
                    if pval > worst_pvalue:
                        worst_pvalue, worst_feature = pval, feat
                if worst_feature:
                    included.remove(worst_feature)
                    log.append(f"Remove '{worst_feature}' (p={worst_pvalue:.4f})")
                    changed = True
            
            if not changed:
                break
    
    return included, log


def generate_plot(y_true, y_pred, residuals, sm_model):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    line_color = '#C44E52'
    
    # Actual vs Predicted
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#5B9BD5', ax=axes[0, 0])
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color=line_color, linewidth=2)
    axes[0, 0].set_title(f"Actual vs Predicted (R² = {sm_model.rsquared:.4f})", fontweight='bold')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    
    # Residuals vs Fitted
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='#5B9BD5', ax=axes[0, 1])
    axes[0, 1].axhline(0, linestyle='--', color=line_color, linewidth=2)
    axes[0, 1].set_title('Residuals vs Fitted', fontweight='bold')
    axes[0, 1].set_xlabel('Fitted')
    axes[0, 1].set_ylabel('Residuals')
    
    # Q-Q Plot
    sm.qqplot(residuals, line='s', ax=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
    
    # Scale-Location
    std_resid = sm_model.get_influence().resid_studentized_internal
    sqrt_abs = np.sqrt(np.abs(std_resid))
    sns.scatterplot(x=y_pred, y=sqrt_abs, alpha=0.6, color='#5B9BD5', ax=axes[1, 1])
    axes[1, 1].set_title('Scale-Location', fontweight='bold')
    axes[1, 1].set_xlabel('Fitted')
    axes[1, 1].set_ylabel('√|Std Residuals|')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/regression")
def regression_analysis(req: RegressionRequest):
    try:
        df = pd.DataFrame(req.data)
        target_var = req.targetVar
        features = req.features
        model_type = req.modelType
        selection_method = req.selectionMethod
        degree = req.degree
        
        # Clean column names
        sanitized = {col: re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns}
        original_names = {v: k for k, v in sanitized.items()}
        df.rename(columns=sanitized, inplace=True)
        target_clean = sanitized[target_var]
        features_clean = [sanitized[f] for f in features if sanitized[f] in df.columns]
        
        # Prepare data
        df[target_clean] = pd.to_numeric(df[target_clean], errors='coerce')
        for f in features_clean:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        
        original_len = len(df)
        df = df.dropna(subset=[target_clean] + features_clean)
        dropped_rows = list(set(range(original_len)) - set(df.index.tolist()))
        
        y = df[target_clean]
        X = df[features_clean]
        
        stepwise_log = []
        if selection_method not in ['none', 'enter']:
            final_features, stepwise_log = stepwise_selection(X, y, method=selection_method)
            if not final_features:
                raise ValueError("No features selected")
            X = X[final_features]
        
        # Polynomial
        if model_type == 'polynomial':
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            poly_names = poly.get_feature_names_out(X.columns)
            X = pd.DataFrame(X_poly, columns=poly_names, index=X.index)
        
        # Fit model
        X_const = sm.add_constant(X)
        sm_model = sm.OLS(y, X_const).fit()
        y_pred = sm_model.predict(X_const)
        residuals = sm_model.resid
        
        # Standardized model
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_std = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)
        y_std = pd.Series(scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)
        X_std_const = sm.add_constant(X_std)
        sm_model_std = sm.OLS(y_std, X_std_const).fit()
        
        # Metrics
        n = len(y)
        n_features = X.shape[1]
        mse = mean_squared_error(y, y_pred)
        metrics = {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred)),
            'adj_r2': float(1 - (1 - sm_model.rsquared) * (n - 1) / (n - n_features - 1)) if (n - n_features - 1) > 0 else 0
        }
        
        # Diagnostics
        def clean_name(name):
            name = re.sub(r'Q\("([^"]+)"\)', r'\1', str(name).strip())
            return original_names.get(name, name)
        
        diagnostics = {
            'f_statistic': float(sm_model.fvalue) if sm_model.fvalue else None,
            'f_pvalue': float(sm_model.f_pvalue) if sm_model.f_pvalue else None,
            'coefficient_tests': {
                'params': {clean_name(k): float(v) for k, v in sm_model.params.items()},
                'pvalues': {clean_name(k): float(v) for k, v in sm_model.pvalues.items()},
                'bse': {clean_name(k): float(v) for k, v in sm_model.bse.items()},
                'tvalues': {clean_name(k): float(v) for k, v in sm_model.tvalues.items()}
            },
            'standardized_coefficients': {
                'params': {clean_name(k): float(v) for k, v in sm_model_std.params.items()},
                'pvalues': {clean_name(k): float(v) for k, v in sm_model_std.pvalues.items()}
            },
            'durbin_watson': float(durbin_watson(residuals))
        }
        
        # VIF
        try:
            X_vif = sm.add_constant(X)
            vif = {clean_name(X_vif.columns[i]): float(variance_inflation_factor(X_vif.values, i)) for i in range(X_vif.shape[1])}
            diagnostics['vif'] = vif
        except:
            diagnostics['vif'] = {}
        
        # Normality
        try:
            jb_stat, jb_p, _, _ = jarque_bera(residuals)
            sw_stat, sw_p = stats.shapiro(residuals)
            diagnostics['normality_tests'] = {
                'jarque_bera': {'statistic': float(jb_stat), 'p_value': float(jb_p)},
                'shapiro_wilk': {'statistic': float(sw_stat), 'p_value': float(sw_p)}
            }
        except:
            diagnostics['normality_tests'] = {}
        
        # Heteroscedasticity
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, sm_model.model.exog)
            diagnostics['heteroscedasticity_tests'] = {'breusch_pagan': {'statistic': float(bp_stat), 'p_value': float(bp_p)}}
        except:
            diagnostics['heteroscedasticity_tests'] = {}
        
        # Interpretation
        r2 = metrics['r2']
        f_pvalue = diagnostics.get('f_pvalue', 1)
        sig = "significant" if f_pvalue and f_pvalue < 0.05 else "not significant"
        interpretation = f"Regression model explains {r2*100:.1f}% of variance (R² = {r2:.3f}). Model is {sig} (F p = {f_pvalue:.4f})."
        
        # Plot
        plot = generate_plot(y, y_pred, residuals, sm_model)
        
        return _to_native({
            'results': {
                'metrics': {'all_data': metrics},
                'diagnostics': diagnostics,
                'stepwise_log': stepwise_log,
                'interpretation': {'overall_analysis': interpretation},
                'n_dropped': len(dropped_rows),
                'dropped_rows': dropped_rows
            },
            'model_name': model_type,
            'model_type': 'regression',
            'plot': plot
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
