# -*- coding: utf-8 -*-
"""
Ridge Regression API Endpoint
- Fixed unicode encoding
- Added input validation
- Performance optimization with ridge path
- Cross-validation support with RidgeCV
- VIF (Variance Inflation Factor) calculation
- Better error handling
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from scipy import stats as _scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import io
import base64
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


# =============================================================================
# Request/Response Models with Validation
# =============================================================================

class RidgeRequest(BaseModel):
    """Request model with comprehensive validation"""
    data: List[dict[str, Any]] = Field(..., min_length=10, description="Dataset with at least 10 rows")
    target: str = Field(..., min_length=1, description="Target variable name")
    features: List[str] = Field(..., min_length=1, description="List of feature variable names")
    alpha: float = Field(default=1.0, gt=0, le=1000, description="Regularization strength (0 < alpha <= 1000)")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set proportion (0.1 to 0.5)")
    use_cv: bool = Field(default=False, description="Use cross-validation to find optimal alpha")
    cv_folds: int = Field(default=5, ge=3, le=10, description="Number of CV folds (3 to 10)")

    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Alpha must be positive')
        if v > 1000:
            raise ValueError('Alpha should not exceed 1000 for numerical stability')
        return v

    @field_validator('test_size')
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if v < 0.1:
            raise ValueError('Test size too small (minimum 0.1)')
        if v > 0.5:
            raise ValueError('Test size too large (maximum 0.5)')
        return v


# =============================================================================
# Utility Functions
# =============================================================================

def _to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization"""
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


def calculate_vif(X: pd.DataFrame) -> dict[str, float]:
    """
    Calculate Variance Inflation Factor for multicollinearity detection.
    VIF > 5 suggests moderate multicollinearity
    VIF > 10 suggests severe multicollinearity
    """
    vif_data = {}
    
    if X.shape[1] < 2:
        return vif_data
    
    try:
        X_with_const = X.copy()
        X_with_const['_const'] = 1
        
        for i, col in enumerate(X.columns):
            try:
                vif_value = variance_inflation_factor(X_with_const.values, i)
                vif_data[col] = float(vif_value) if not np.isinf(vif_value) else None
            except Exception:
                vif_data[col] = None
    except Exception:
        pass
    
    return vif_data


def generate_interpretation(
    train_r2: float, 
    test_r2: float, 
    alpha: float, 
    coefficients: np.ndarray, 
    feature_names: List[str],
    used_cv: bool = False,
    cv_alpha: Optional[float] = None,
    vif_data: Optional[dict] = None
) -> str:
    """
    Generate human-readable interpretation of Ridge regression results.
    Fixed unicode characters for proper display.
    """
    r2_diff = train_r2 - test_r2
    n_features = len(coefficients)
    
    # Determine effect size
    if test_r2 >= 0.75:
        effect_size = "large"
    elif test_r2 >= 0.50:
        effect_size = "good"
    elif test_r2 >= 0.25:
        effect_size = "moderate"
    else:
        effect_size = "small"
    
    # Build interpretation
    lines = []
    
    # Overall Assessment
    lines.append("**Overall Assessment**")
    lines.append(f"→ Ridge regression: R² = {test_r2:.2f} ({effect_size} effect), {test_r2*100:.1f}% variance explained.")
    lines.append(f"→ L2 regularization (α = {alpha:.3f}): coefficients shrunk toward zero.")
    lines.append(f"→ Train R² = {train_r2:.2f}, Test R² = {test_r2:.2f}, ΔR² = {r2_diff:.2f}.")
    
    # Cross-validation info
    if used_cv and cv_alpha is not None:
        lines.append(f"→ Optimal α = {cv_alpha:.4f} selected via cross-validation.")
    
    # Overfitting assessment
    if r2_diff > 0.15:
        lines.append("→ ⚠ Potential overfitting detected (train-test gap > 0.15).")
    elif r2_diff > 0.10:
        lines.append("→ Moderate generalization gap detected.")
    elif r2_diff < 0.05:
        lines.append("→ ✓ Strong generalization capability.")
    
    # Statistical Insights
    lines.append("")
    lines.append("**Statistical Insights**")
    
    # Top predictors by magnitude
    top_indices = np.argsort(np.abs(coefficients))[::-1][:5]
    lines.append("→ Top predictors by coefficient magnitude:")
    for i in top_indices[:5]:
        coef = coefficients[i]
        direction = "positive" if coef > 0 else "negative"
        lines.append(f"  • {feature_names[i]}: {coef:.4f} ({direction})")
    
    # VIF warnings (multicollinearity)
    if vif_data:
        high_vif = [(k, v) for k, v in vif_data.items() if v is not None and v > 5]
        if high_vif:
            lines.append("")
            lines.append("**Multicollinearity Note**")
            lines.append("→ Ridge regression handles multicollinearity well, but high VIF detected:")
            for feat, vif in sorted(high_vif, key=lambda x: -x[1])[:3]:
                severity = "severe" if vif > 10 else "moderate"
                lines.append(f"  • {feat}: VIF = {vif:.2f} ({severity})")
    
    # Recommendations
    lines.append("")
    lines.append("**Recommendations**")
    
    if test_r2 < 0.25:
        lines.append("→ Low R² - consider adding more features or reducing alpha.")
    elif test_r2 < 0.50:
        lines.append("→ Moderate R² - model is useful but could be improved.")
    
    if r2_diff > 0.15:
        lines.append("→ Increase alpha to reduce overfitting.")
    elif r2_diff < 0.05 and test_r2 >= 0.50:
        lines.append("→ Model performs well with current settings.")
    
    lines.append("→ Ridge keeps all features (unlike Lasso) — good for multicollinearity.")
    
    if not used_cv:
        lines.append("→ Consider using cross-validation (use_cv=True) for automatic alpha selection.")
    
    return "\n".join(lines)


# =============================================================================
# Main Endpoint
# =============================================================================

@router.post("/ridge-regression")
def ridge_regression(req: RidgeRequest):
    """
    Perform Ridge (L2-regularized) regression analysis.
    
    Features:
    - L2 penalty shrinks coefficients toward zero
    - Optional cross-validation for alpha tuning
    - VIF calculation for multicollinearity detection
    - Comprehensive diagnostics and visualizations
    """
    try:
        # =================================================================
        # Data Preparation
        # =================================================================
        df = pd.DataFrame(req.data)
        target = req.target
        features = req.features
        alpha = req.alpha
        test_size = req.test_size
        use_cv = req.use_cv
        cv_folds = req.cv_folds
        
        # Validate target exists
        if target not in df.columns:
            raise ValueError(f"Target variable '{target}' not found in data")
        
        # Validate features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        
        # Prepare feature matrix with dummy encoding for categorical variables
        X = pd.get_dummies(df[features], drop_first=True)
        y = pd.to_numeric(df[target], errors='coerce')
        final_features = X.columns.tolist()
        
        # Handle missing values
        combined = pd.concat([X, y], axis=1).dropna()
        n_dropped = len(df) - len(combined)
        
        X = combined[final_features]
        y = combined[target]
        
        if len(X) < 10:
            raise ValueError(f"Not enough valid data after removing missing values (n={len(X)}, minimum=10)")
        
        # Validate sample size vs features
        if len(X) < len(final_features) * 2:
            warnings.warn(f"Low sample size ({len(X)}) relative to features ({len(final_features)}). Results may be unstable.")
        
        # =================================================================
        # Train/Test Split
        # =================================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # =================================================================
        # Model Training
        # =================================================================
        cv_alpha_used = None
        
        if use_cv:
            # Use RidgeCV to find optimal alpha
            ridge_cv = RidgeCV(
                alphas=np.logspace(-4, 4, 100),
                cv=cv_folds,
                scoring='r2'
            )
            ridge_cv.fit(X_train_scaled, y_train)
            alpha = ridge_cv.alpha_
            cv_alpha_used = alpha
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_scaled, y_train)
        else:
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_scaled, y_train)
        
        # =================================================================
        # Predictions and Metrics
        # =================================================================
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_metrics = {
            'r2_score': float(r2_score(y_train, y_pred_train)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'mae': float(mean_absolute_error(y_train, y_pred_train)),
            'n_samples': int(len(y_train))
        }
        
        test_metrics = {
            'r2_score': float(r2_score(y_test, y_pred_test)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'n_samples': int(len(y_test))
        }
        
        # =================================================================
        # VIF Calculation (multicollinearity detection)
        # =================================================================
        vif_data = calculate_vif(X_train)
        
        # =================================================================
        # Cross-Validation Scores
        # =================================================================
        X_all_scaled = scaler.transform(X)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_r2   = cross_val_score(model, X_all_scaled, y, cv=kf, scoring='r2')
        cv_nmse = cross_val_score(model, X_all_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        cv_results = {
            'r2_mean':   round(float(cv_r2.mean()), 4),
            'r2_std':    round(float(cv_r2.std()),  4),
            'rmse_mean': round(float(np.sqrt((-cv_nmse).mean())), 4),
            'rmse_std':  round(float(np.sqrt((-cv_nmse).std())),  4),
            'n_folds':   cv_folds,
            'scores':    [round(float(s), 4) for s in cv_r2],
        }

        # =================================================================
        # OLS Comparison
        # =================================================================
        ols = LinearRegression().fit(X_train_scaled, y_train)
        ols_pred_tr = ols.predict(X_train_scaled)
        ols_pred_te = ols.predict(X_test_scaled)
        ols_comparison = {
            'ols_train_r2':   round(float(r2_score(y_train, ols_pred_tr)), 4),
            'ols_test_r2':    round(float(r2_score(y_test,  ols_pred_te)), 4),
            'ols_test_rmse':  round(float(np.sqrt(mean_squared_error(y_test, ols_pred_te))), 4),
            'ridge_train_r2': round(float(train_metrics['r2_score']), 4),
            'ridge_test_r2':  round(float(test_metrics['r2_score']),  4),
            'ridge_test_rmse':round(float(test_metrics['rmse']), 4),
            'delta_test_r2':  round(float(r2_score(y_test, ols_pred_te) - test_metrics['r2_score']), 4),
            'note': 'Positive delta = OLS beats Ridge on test R² (Ridge may be over-regularised)',
        }

        # =================================================================
        # Residual Diagnostics
        # =================================================================
        residuals   = y_test.values - y_pred_test
        fitted_vals = y_pred_test

        sw_stat, sw_p = (_scipy_stats.shapiro(residuals)
                         if len(residuals) <= 5000 else (None, None))
        bp_corr, bp_p = _scipy_stats.pearsonr(fitted_vals, np.abs(residuals))
        residual_diagnostics = {
            'mean':     round(float(residuals.mean()), 4),
            'std':      round(float(residuals.std()),  4),
            'skewness': round(float(_scipy_stats.skew(residuals)), 4),
            'kurtosis': round(float(_scipy_stats.kurtosis(residuals)), 4),
            'shapiro_wilk': {
                'statistic': round(float(sw_stat), 4) if sw_stat is not None else None,
                'p_value':   round(float(sw_p),   4) if sw_p   is not None else None,
                'normal':    bool(sw_p > 0.05)        if sw_p   is not None else None,
            },
            'heteroscedasticity': {
                'corr_fitted_abs_resid': round(float(bp_corr), 4),
                'p_value':              round(float(bp_p),    4),
                'detected':             bool(bp_p < 0.05),
            },
        }

        # =================================================================
        # Generate Interpretation
        # =================================================================
        interpretation = generate_interpretation(
            train_r2=train_metrics['r2_score'],
            test_r2=test_metrics['r2_score'],
            alpha=alpha,
            coefficients=model.coef_,
            feature_names=final_features,
            used_cv=use_cv,
            cv_alpha=cv_alpha_used,
            vif_data=vif_data
        )
        interpretation += f"\n\n**Cross-Validation ({cv_folds}-fold)**"
        interpretation += f"\n→ CV R² = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}"
        interpretation += f"\n→ CV RMSE = {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}"
        interpretation += f"\n\n**OLS vs Ridge (test set)**"
        interpretation += f"\n→ OLS R² = {ols_comparison['ols_test_r2']:.3f}  |  Ridge R² = {ols_comparison['ridge_test_r2']:.3f}"
        interpretation += f"\n→ Δ R² = {ols_comparison['delta_test_r2']:+.3f}"
        interpretation += f"\n\n**Residual Diagnostics**"
        _sw_p_str  = str(residual_diagnostics['shapiro_wilk']['p_value']) if residual_diagnostics['shapiro_wilk']['p_value'] is not None else 'N/A'
        _sw_label  = ('normal' if residual_diagnostics['shapiro_wilk']['normal'] else 'non-normal') if residual_diagnostics['shapiro_wilk']['normal'] is not None else 'N/A'
        interpretation += f"\n→ Shapiro-Wilk p = {_sw_p_str} ({_sw_label})"
        if residual_diagnostics['heteroscedasticity']['detected']:
            interpretation += f"\n→ Heteroscedasticity detected (r={residual_diagnostics['heteroscedasticity']['corr_fitted_abs_resid']:.3f}, p={residual_diagnostics['heteroscedasticity']['p_value']:.3f})"
        
        # =================================================================
        # Ridge Path Calculation
        # =================================================================
        alpha_list = np.logspace(-3, 3, 50)
        coefs = []
        train_scores = []
        test_scores = []
        
        for a in alpha_list:
            ridge = Ridge(alpha=a, random_state=42)
            ridge.fit(X_train_scaled, y_train)
            coefs.append(ridge.coef_)
            train_scores.append(ridge.score(X_train_scaled, y_train))
            test_scores.append(ridge.score(X_test_scaled, y_test))
        
        coefs_array = np.array(coefs)
        
        # =================================================================
        # Visualization: Main Diagnostic Plot (2x2 grid)
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Train predictions vs actual
        sns.scatterplot(x=y_train, y=y_pred_train, alpha=0.6, color='#5B9BD5', ax=axes[0, 0])
        axes[0, 0].plot(
            [y_train.min(), y_train.max()], 
            [y_train.min(), y_train.max()], 
            '--', color='#C44E52', lw=2
        )
        axes[0, 0].set_title(f"Train (R² = {train_metrics['r2_score']:.3f})", fontweight='bold')
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        
        # Plot 2: Test predictions vs actual
        sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, color='#5B9BD5', ax=axes[0, 1])
        axes[0, 1].plot(
            [y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 
            '--', color='#C44E52', lw=2
        )
        axes[0, 1].set_title(f"Test (R² = {test_metrics['r2_score']:.3f})", fontweight='bold')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        
        # Plot 3: R² vs Alpha
        axes[1, 0].plot(alpha_list, train_scores, label='Train', color='#5B9BD5', lw=2)
        axes[1, 0].plot(alpha_list, test_scores, label='Test', color='#F4A582', lw=2)
        axes[1, 0].axvline(x=alpha, color='#C44E52', linestyle='--', label=f'α = {alpha:.4f}')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel('Alpha (log scale)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('R² vs Alpha', fontweight='bold')
        axes[1, 0].legend(loc='best')
        
        # Plot 4: Ridge Path
        for i in range(coefs_array.shape[1]):
            axes[1, 1].plot(alpha_list, coefs_array[:, i], lw=1.5)
        axes[1, 1].axvline(x=alpha, color='#C44E52', linestyle='--', label=f'α = {alpha:.4f}')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_xlabel('Alpha (log scale)')
        axes[1, 1].set_ylabel('Coefficients')
        axes[1, 1].set_title('Ridge Path', fontweight='bold')
        axes[1, 1].legend(loc='best')
        
        plt.tight_layout()
        
        # Save main plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # =================================================================
        # Visualization: Detailed Ridge Path Plot
        # =================================================================
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        
        for i in range(coefs_array.shape[1]):
            label = final_features[i] if len(final_features) <= 10 else None
            ax2.plot(alpha_list, coefs_array[:, i], lw=1.5, label=label)
        
        ax2.axvline(x=alpha, color='#C44E52', linestyle='--', linewidth=2, label=f'Current α = {alpha:.4f}')
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha (log scale)', fontsize=12)
        ax2.set_ylabel('Coefficients', fontsize=12)
        ax2.set_title('Ridge Regularization Path', fontweight='bold', fontsize=14)
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        if len(final_features) <= 10:
            ax2.legend(loc='best', fontsize=9)
        else:
            ax2.legend([f'Current α = {alpha:.4f}'], loc='best', fontsize=9)
        
        plt.tight_layout()
        
        # Save path plot
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig2)
        buf2.seek(0)
        path_plot = f"data:image/png;base64,{base64.b64encode(buf2.read()).decode('utf-8')}"
        
        # =================================================================
        # Plot 3: Residuals vs Fitted  +  Normal Q-Q
        # =================================================================
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

        axes3[0].scatter(fitted_vals, residuals, alpha=0.55, color='#5B9BD5', s=28, edgecolors='none')
        axes3[0].axhline(0, color='red', linestyle='--', lw=1.5)
        _xs = np.linspace(fitted_vals.min(), fitted_vals.max(), 200)
        axes3[0].plot(_xs, np.poly1d(np.polyfit(fitted_vals, residuals, 2))(_xs),
                      color='orange', lw=1.5, label='Trend (poly2)')
        axes3[0].set_xlabel('Fitted Values'); axes3[0].set_ylabel('Residuals')
        axes3[0].set_title('Residuals vs Fitted', fontweight='bold')
        axes3[0].legend(fontsize=8)

        _scipy_stats.probplot(residuals, dist='norm', plot=axes3[1])
        axes3[1].set_title('Normal Q-Q Plot of Residuals', fontweight='bold')
        axes3[1].get_lines()[0].set(markersize=4, alpha=0.6, color='#5B9BD5')
        axes3[1].get_lines()[1].set(color='red', lw=1.5)

        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig3); buf3.seek(0)
        residual_plot = f"data:image/png;base64,{base64.b64encode(buf3.read()).decode('utf-8')}"

        # =================================================================
        # Plot 4: OLS vs Ridge bar comparison
        # =================================================================
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        _labels = ['Train R²', 'Test R²', 'Test RMSE']
        _ols_v  = [ols_comparison['ols_train_r2'],   ols_comparison['ols_test_r2'],   ols_comparison['ols_test_rmse']]
        _rdg_v  = [ols_comparison['ridge_train_r2'], ols_comparison['ridge_test_r2'], ols_comparison['ridge_test_rmse']]
        _xp = np.arange(len(_labels))
        b1 = ax4.bar(_xp - 0.2, _ols_v, 0.38, label='OLS',   color='#F4A582', edgecolor='gray', linewidth=0.7)
        b2 = ax4.bar(_xp + 0.2, _rdg_v, 0.38, label='Ridge', color='#5B9BD5', edgecolor='gray', linewidth=0.7)
        for bar, val in list(zip(b1, _ols_v)) + list(zip(b2, _rdg_v)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax4.set_xticks(_xp); ax4.set_xticklabels(_labels)
        ax4.set_title('OLS vs Ridge (test set)', fontweight='bold')
        ax4.legend(); ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf4 = io.BytesIO()
        plt.savefig(buf4, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig4); buf4.seek(0)
        ols_plot = f"data:image/png;base64,{base64.b64encode(buf4.read()).decode('utf-8')}"

        # =================================================================
        # Build Response
        # =================================================================
        return _to_native({
            'results': {
                'metrics': {
                    'train': train_metrics,
                    'test': test_metrics
                },
                'coefficients': dict(zip(final_features, model.coef_.tolist())),
                'intercept': float(model.intercept_),
                'alpha': float(alpha),
                'alpha_source': 'cross_validation' if use_cv else 'user_specified',
                'cv_folds': cv_folds if use_cv else None,
                'interpretation': interpretation,
                'n_dropped': n_dropped,
                'n_features': len(final_features),
                'vif': vif_data if vif_data else None,
                'cv_results':           cv_results,
                'ols_comparison':       ols_comparison,
                'residual_diagnostics': residual_diagnostics,
            },
            'plot': plot,
            'path_plot': path_plot,
            'residual_plot': residual_plot,
            'ols_plot': ols_plot,
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
