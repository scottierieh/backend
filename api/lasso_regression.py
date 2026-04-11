# -*- coding: utf-8 -*-
"""
Lasso Regression API Endpoint
- Fixed unicode encoding
- Added input validation
- Performance optimization with lasso_path
- Cross-validation support with LassoCV
- VIF (Variance Inflation Factor) calculation
- Better error handling
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from scipy import stats as _scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, lasso_path
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

class LassoRequest(BaseModel):
    """Request model with comprehensive validation"""
    data: List[dict[str, Any]] = Field(..., min_length=10, description="Dataset with at least 10 rows")
    target: str = Field(..., min_length=1, description="Target variable name")
    features: List[str] = Field(..., min_length=1, description="List of feature variable names")
    alpha: float = Field(default=0.1, gt=0, le=100, description="Regularization strength (0 < alpha <= 100)")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set proportion (0.1 to 0.5)")
    use_cv: bool = Field(default=False, description="Use cross-validation to find optimal alpha")
    cv_folds: int = Field(default=5, ge=3, le=10, description="Number of CV folds (3 to 10)")

    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Alpha must be positive')
        if v > 100:
            raise ValueError('Alpha should not exceed 100 for numerical stability')
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
        # Add constant for VIF calculation
        X_with_const = X.copy()
        X_with_const['_const'] = 1
        
        for i, col in enumerate(X.columns):
            try:
                vif_value = variance_inflation_factor(X_with_const.values, i)
                vif_data[col] = float(vif_value) if not np.isinf(vif_value) else None
            except Exception:
                vif_data[col] = None
    except Exception:
        # Return empty dict if VIF calculation fails
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
    Generate human-readable interpretation of Lasso regression results.
    Fixed unicode characters for proper display.
    """
    r2_diff = train_r2 - test_r2
    non_zero_mask = np.abs(coefficients) >= 1e-6
    n_selected = int(np.sum(non_zero_mask))
    n_total = len(coefficients)
    n_excluded = n_total - n_selected
    
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
    lines.append(f"→ Lasso regression: R² = {test_r2:.2f} ({effect_size} effect), {test_r2*100:.1f}% variance explained.")
    lines.append(f"→ L1 regularization (α = {alpha:.3f}): retained {n_selected}/{n_total} predictors.")
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
    
    # Top predictors
    top_indices = np.argsort(np.abs(coefficients))[::-1][:5]
    top_features = [(feature_names[i], coefficients[i]) for i in top_indices if abs(coefficients[i]) >= 1e-6]
    
    if top_features:
        lines.append("→ Top predictors:")
        for feat, coef in top_features:
            direction = "positive" if coef > 0 else "negative"
            lines.append(f"  • {feat}: {coef:.4f} ({direction})")
    
    if n_excluded > 0:
        lines.append(f"→ {n_excluded} feature(s) excluded (coefficients = 0).")
    
    # VIF warnings (multicollinearity)
    if vif_data:
        high_vif = [(k, v) for k, v in vif_data.items() if v is not None and v > 5]
        if high_vif:
            lines.append("")
            lines.append("**Multicollinearity Warning**")
            for feat, vif in sorted(high_vif, key=lambda x: -x[1])[:3]:
                severity = "severe" if vif > 10 else "moderate"
                lines.append(f"→ {feat}: VIF = {vif:.2f} ({severity})")
    
    # Recommendations
    lines.append("")
    lines.append("**Recommendations**")
    
    if test_r2 < 0.25:
        lines.append("→ Low R² - consider adding more features or reducing alpha.")
    elif test_r2 < 0.50:
        lines.append("→ Moderate R² - model is useful but could be improved.")
    
    if r2_diff > 0.15:
        lines.append("→ Increase alpha or use cross-validation to reduce overfitting.")
    elif r2_diff < 0.05 and test_r2 >= 0.50:
        lines.append("→ Model performs well with current settings.")
    
    if n_selected < 3 and n_total > 5:
        lines.append("→ Many features eliminated - consider reducing alpha if important predictors were excluded.")
    
    if not used_cv:
        lines.append("→ Consider using cross-validation (use_cv=True) for automatic alpha selection.")
    
    return "\n".join(lines)


# =============================================================================
# Main Endpoint
# =============================================================================

@router.post("/lasso-regression")
def lasso_regression(req: LassoRequest):
    """
    Perform Lasso (L1-regularized) regression analysis.
    
    Features:
    - Automatic feature selection via L1 penalty
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
            # Use LassoCV to find optimal alpha
            lasso_cv = LassoCV(
                alphas=np.logspace(-4, 2, 100),
                cv=cv_folds,
                random_state=42,
                max_iter=10000
            )
            lasso_cv.fit(X_train_scaled, y_train)
            alpha = lasso_cv.alpha_
            cv_alpha_used = alpha
            model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            model.fit(X_train_scaled, y_train)
        else:
            model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
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
        # Cross-Validation Scores (full dataset, fitted scaler)
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
            'ols_train_r2':    round(float(r2_score(y_train, ols_pred_tr)), 4),
            'ols_test_r2':     round(float(r2_score(y_test,  ols_pred_te)), 4),
            'ols_test_rmse':   round(float(np.sqrt(mean_squared_error(y_test, ols_pred_te))), 4),
            'lasso_train_r2':  round(float(train_metrics['r2_score']), 4),
            'lasso_test_r2':   round(float(test_metrics['r2_score']),  4),
            'lasso_test_rmse': round(float(test_metrics['rmse']), 4),
            'n_features_ols':   len(final_features),
            'n_features_lasso': int(np.sum(np.abs(model.coef_) >= 1e-6)),
            'delta_test_r2':   round(float(r2_score(y_test, ols_pred_te) - test_metrics['r2_score']), 4),
            'note': 'Positive delta = OLS beats Lasso on test R² (Lasso may be over-regularised)',
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
        interpretation += f"\n\n**OLS vs Lasso (test set)**"
        interpretation += f"\n→ OLS R² = {ols_comparison['ols_test_r2']:.3f}  |  Lasso R² = {ols_comparison['lasso_test_r2']:.3f}"
        interpretation += f"\n→ Δ R² = {ols_comparison['delta_test_r2']:+.3f}  ({ols_comparison['n_features_lasso']}/{ols_comparison['n_features_ols']} features kept)"
        interpretation += f"\n\n**Residual Diagnostics**"
        sw_p_str = str(residual_diagnostics['shapiro_wilk']['p_value']) if residual_diagnostics['shapiro_wilk']['p_value'] is not None else 'N/A'
        sw_label = ('normal' if residual_diagnostics['shapiro_wilk']['normal'] else 'non-normal') if residual_diagnostics['shapiro_wilk']['normal'] is not None else 'N/A'
        interpretation += f"\n→ Shapiro-Wilk p = {sw_p_str} ({sw_label})"
        if residual_diagnostics['heteroscedasticity']['detected']:
            interpretation += f"\n→ Heteroscedasticity detected (r={residual_diagnostics['heteroscedasticity']['corr_fitted_abs_resid']:.3f}, p={residual_diagnostics['heteroscedasticity']['p_value']:.3f})"
        
        # =================================================================
        # Lasso Path Calculation (Optimized with sklearn.lasso_path)
        # =================================================================
        alpha_list = np.logspace(-4, 2, 50)
        
        # Use sklearn's optimized lasso_path instead of fitting 50 models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alphas_path, coefs_path, _ = lasso_path(
                X_train_scaled, y_train,
                alphas=alpha_list,
                max_iter=10000
            )
        
        # coefs_path shape: (n_features, n_alphas), transpose for plotting
        coefs_array = coefs_path.T
        
        # Calculate R² scores along the path
        train_scores = []
        test_scores = []
        for i, a in enumerate(alphas_path):
            coef = coefs_path[:, i]
            # Calculate predictions using path coefficients
            y_pred_train_path = X_train_scaled @ coef
            y_pred_test_path = X_test_scaled @ coef
            
            train_scores.append(r2_score(y_train, y_pred_train_path))
            test_scores.append(r2_score(y_test, y_pred_test_path))
        
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
        axes[1, 0].plot(alphas_path, train_scores, label='Train', color='#5B9BD5', lw=2)
        axes[1, 0].plot(alphas_path, test_scores, label='Test', color='#F4A582', lw=2)
        axes[1, 0].axvline(x=alpha, color='#C44E52', linestyle='--', label=f'α = {alpha:.4f}')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel('Alpha (log scale)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('R² vs Alpha', fontweight='bold')
        axes[1, 0].legend(loc='best')
        axes[1, 0].set_ylim(bottom=min(0, min(test_scores) - 0.1))
        
        # Plot 4: Lasso Path
        for i in range(coefs_array.shape[1]):
            axes[1, 1].plot(alphas_path, coefs_array[:, i], lw=1.5)
        axes[1, 1].axvline(x=alpha, color='#C44E52', linestyle='--', label=f'α = {alpha:.4f}')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_xlabel('Alpha (log scale)')
        axes[1, 1].set_ylabel('Coefficients')
        axes[1, 1].set_title('Lasso Path', fontweight='bold')
        axes[1, 1].legend(loc='best')
        
        plt.tight_layout()
        
        # Save main plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
        # =================================================================
        # Visualization: Detailed Lasso Path Plot
        # =================================================================
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        
        for i in range(coefs_array.shape[1]):
            label = final_features[i] if len(final_features) <= 10 else None
            ax2.plot(alphas_path, coefs_array[:, i], lw=1.5, label=label)
        
        ax2.axvline(x=alpha, color='#C44E52', linestyle='--', linewidth=2, label=f'Current α = {alpha:.4f}')
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha (log scale)', fontsize=12)
        ax2.set_ylabel('Coefficients', fontsize=12)
        ax2.set_title('Lasso Regularization Path', fontweight='bold', fontsize=14)
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
        _trend = np.poly1d(np.polyfit(fitted_vals, residuals, 2))
        _xsort = np.linspace(fitted_vals.min(), fitted_vals.max(), 200)
        axes3[0].plot(_xsort, _trend(_xsort), color='orange', lw=1.5, label='Trend (poly2)')
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
        plt.close(fig3)
        buf3.seek(0)
        residual_plot = f"data:image/png;base64,{base64.b64encode(buf3.read()).decode('utf-8')}"

        # =================================================================
        # Plot 4: OLS vs Lasso bar comparison
        # =================================================================
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        _labels   = ['Train R²', 'Test R²', 'Test RMSE']
        _ols_v    = [ols_comparison['ols_train_r2'],   ols_comparison['ols_test_r2'],   ols_comparison['ols_test_rmse']]
        _lasso_v  = [ols_comparison['lasso_train_r2'], ols_comparison['lasso_test_r2'], ols_comparison['lasso_test_rmse']]
        _xp = np.arange(len(_labels))
        ax4.bar(_xp - 0.2, _ols_v,   0.38, label='OLS',   color='#F4A582', edgecolor='gray', linewidth=0.7)
        ax4.bar(_xp + 0.2, _lasso_v, 0.38, label='Lasso', color='#5B9BD5', edgecolor='gray', linewidth=0.7)
        for _i, (_o, _l) in enumerate(zip(_ols_v, _lasso_v)):
            ax4.text(_i - 0.2, _o + 0.005, f'{_o:.3f}', ha='center', va='bottom', fontsize=8)
            ax4.text(_i + 0.2, _l + 0.005, f'{_l:.3f}', ha='center', va='bottom', fontsize=8)
        ax4.set_xticks(_xp); ax4.set_xticklabels(_labels)
        ax4.set_title(f'OLS vs Lasso  (Lasso kept {ols_comparison["n_features_lasso"]}/{ols_comparison["n_features_ols"]} features)', fontweight='bold')
        ax4.legend(); ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf4 = io.BytesIO()
        plt.savefig(buf4, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig4)
        buf4.seek(0)
        ols_plot = f"data:image/png;base64,{base64.b64encode(buf4.read()).decode('utf-8')}"

        # =================================================================
        # Build Response
        # =================================================================
        # Feature selection summary
        non_zero_mask = np.abs(model.coef_) >= 1e-6
        selected_features = [f for f, m in zip(final_features, non_zero_mask) if m]
        excluded_features = [f for f, m in zip(final_features, non_zero_mask) if not m]
        
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
                'feature_selection': {
                    'n_total': len(final_features),
                    'n_selected': len(selected_features),
                    'n_excluded': len(excluded_features),
                    'selected': selected_features,
                    'excluded': excluded_features
                },
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


# =============================================================================
# ElasticNet Regression
# =============================================================================
from sklearn.linear_model import ElasticNet, ElasticNetCV

class ElasticNetRequest(BaseModel):
    data: List[dict[str, Any]] = Field(..., min_length=10)
    target: str = Field(..., min_length=1)
    features: List[str] = Field(..., min_length=1)
    alpha: float = Field(default=0.1, gt=0, le=100)
    l1_ratio: float = Field(default=0.5, ge=0.0, le=1.0,
                            description="Mix ratio: 0=Ridge, 1=Lasso, 0.5=equal mix")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    use_cv: bool = Field(default=True)
    cv_folds: int = Field(default=5, ge=3, le=10)


@router.post("/elasticnet-regression")
def elasticnet_regression(req: ElasticNetRequest):
    try:
        # ── Data prep ────────────────────────────────────────────────────
        df = pd.DataFrame(req.data)
        if req.target not in df.columns:
            raise ValueError(f"Target '{req.target}' not found")
        missing = [f for f in req.features if f not in df.columns]
        if missing:
            raise ValueError(f"Features not found: {missing}")

        X = pd.get_dummies(df[req.features], drop_first=True)
        y = pd.to_numeric(df[req.target], errors='coerce')
        final_features = X.columns.tolist()

        combined = pd.concat([X, y], axis=1).dropna()
        n_dropped = len(df) - len(combined)
        X = combined[final_features]
        y = combined[req.target]

        if len(X) < 10:
            raise ValueError(f"Not enough data after dropna (n={len(X)})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=42)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        # ── Model fit ────────────────────────────────────────────────────
        alpha = req.alpha
        l1_ratio = req.l1_ratio
        cv_alpha_used = None
        cv_l1_used = None

        if req.use_cv:
            l1_ratios_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
            en_cv = ElasticNetCV(
                l1_ratio=l1_ratios_grid,
                alphas=np.logspace(-4, 2, 60),
                cv=req.cv_folds,
                random_state=42,
                max_iter=10000,
            )
            en_cv.fit(X_tr, y_train)
            alpha    = en_cv.alpha_
            l1_ratio = en_cv.l1_ratio_
            cv_alpha_used = alpha
            cv_l1_used    = l1_ratio

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           random_state=42, max_iter=10000)
        model.fit(X_tr, y_train)

        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)

        # ── Metrics ──────────────────────────────────────────────────────
        train_metrics = {
            'r2_score': round(float(r2_score(y_train, y_pred_tr)), 4),
            'rmse':     round(float(np.sqrt(mean_squared_error(y_train, y_pred_tr))), 4),
            'mae':      round(float(mean_absolute_error(y_train, y_pred_tr)), 4),
            'n_samples': int(len(y_train)),
        }
        test_metrics = {
            'r2_score': round(float(r2_score(y_test, y_pred_te)), 4),
            'rmse':     round(float(np.sqrt(mean_squared_error(y_test, y_pred_te))), 4),
            'mae':      round(float(mean_absolute_error(y_test, y_pred_te)), 4),
            'n_samples': int(len(y_test)),
        }

        # ── CV scores (full data) ────────────────────────────────────────
        X_all = scaler.transform(X)
        kf = KFold(n_splits=req.cv_folds, shuffle=True, random_state=42)
        cv_r2   = cross_val_score(model, X_all, y, cv=kf, scoring='r2')
        cv_nmse = cross_val_score(model, X_all, y, cv=kf, scoring='neg_mean_squared_error')
        cv_results = {
            'r2_mean':   round(float(cv_r2.mean()), 4),
            'r2_std':    round(float(cv_r2.std()),  4),
            'rmse_mean': round(float(np.sqrt((-cv_nmse).mean())), 4),
            'rmse_std':  round(float(np.sqrt((-cv_nmse).std())),  4),
            'n_folds':   req.cv_folds,
            'scores':    [round(float(s), 4) for s in cv_r2],
        }

        # ── OLS + Lasso comparison ───────────────────────────────────────
        ols = LinearRegression().fit(X_tr, y_train)
        lasso_m = Lasso(alpha=alpha, random_state=42, max_iter=10000).fit(X_tr, y_train)

        def _r2te(m): return round(float(r2_score(y_test, m.predict(X_te))), 4)
        def _rmse(m):  return round(float(np.sqrt(mean_squared_error(y_test, m.predict(X_te)))), 4)
        def _nfeat(m): return int(np.sum(np.abs(m.coef_) >= 1e-6))

        model_comparison = {
            'ols':        {'test_r2': _r2te(ols),     'test_rmse': _rmse(ols),     'n_features': len(final_features)},
            'lasso':      {'test_r2': _r2te(lasso_m), 'test_rmse': _rmse(lasso_m), 'n_features': _nfeat(lasso_m)},
            'elasticnet': {'test_r2': test_metrics['r2_score'], 'test_rmse': test_metrics['rmse'], 'n_features': _nfeat(model)},
            'note': 'All models trained on same train/test split for fair comparison',
        }

        # ── Residual diagnostics ─────────────────────────────────────────
        residuals   = y_test.values - y_pred_te
        fitted_vals = y_pred_te
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

        # ── Feature selection summary ────────────────────────────────────
        non_zero = np.abs(model.coef_) >= 1e-6
        selected  = [f for f, m in zip(final_features, non_zero) if m]
        excluded  = [f for f, m in zip(final_features, non_zero) if not m]

        # ── l1_ratio interpretation ──────────────────────────────────────
        if l1_ratio >= 0.9:
            mix_label = "near-Lasso (strong sparsity)"
        elif l1_ratio >= 0.5:
            mix_label = "Lasso-leaning (moderate sparsity)"
        elif l1_ratio >= 0.1:
            mix_label = "Ridge-leaning (grouped selection)"
        else:
            mix_label = "near-Ridge (minimal sparsity)"

        # ── Interpretation ───────────────────────────────────────────────
        interp  = "**ElasticNet Regression**\n"
        interp += f"→ α = {alpha:.4f}, l1_ratio = {l1_ratio:.2f} ({mix_label})\n"
        interp += f"→ {'CV-optimised' if req.use_cv else 'User-specified'} hyperparameters\n"
        interp += f"→ Test R² = {test_metrics['r2_score']:.3f}  |  RMSE = {test_metrics['rmse']:.3f}\n"
        interp += f"→ CV R² = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}\n"
        interp += f"→ Features kept: {len(selected)}/{len(final_features)}\n"
        interp += "\n**Model Comparison (test set)**\n"
        for name, vals in [('OLS', model_comparison['ols']),
                           ('Lasso', model_comparison['lasso']),
                           ('ElasticNet', model_comparison['elasticnet'])]:
            interp += f"→ {name}: R²={vals['test_r2']:.3f}, RMSE={vals['test_rmse']:.3f}, features={vals['n_features']}\n"
        interp += "\n**Why ElasticNet over Lasso?**\n"
        interp += "→ Lasso arbitrarily picks one variable from a correlated group.\n"
        interp += f"→ ElasticNet (l1_ratio={l1_ratio:.2f}) retains grouped correlated features.\n"
        if model_comparison['elasticnet']['test_r2'] > model_comparison['lasso']['test_r2']:
            delta = model_comparison['elasticnet']['test_r2'] - model_comparison['lasso']['test_r2']
            interp += f"→ ElasticNet outperforms Lasso by ΔR²={delta:+.3f} on this dataset.\n"

        # ── Plots ────────────────────────────────────────────────────────
        # Plot 1: Predicted vs Actual (train & test) + Residuals vs Fitted + Q-Q
        fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))

        for ax, ytr, ypr, label, color in [
            (axes1[0], y_train, y_pred_tr, 'Train', '#5B9BD5'),
            (axes1[1], y_test,  y_pred_te, 'Test',  '#F4A582'),
        ]:
            ax.scatter(ytr, ypr, alpha=0.55, color=color, s=28, edgecolors='none')
            lims = [min(ytr.min(), ypr.min()), max(ytr.max(), ypr.max())]
            ax.plot(lims, lims, 'r--', lw=1.5)
            ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
            ax.set_title(f'{label}  R²={r2_score(ytr, ypr):.3f}', fontweight='bold')

        axes1[2].scatter(fitted_vals, residuals, alpha=0.55, color='#8DA0CB', s=28, edgecolors='none')
        axes1[2].axhline(0, color='red', linestyle='--', lw=1.5)
        _xs = np.linspace(fitted_vals.min(), fitted_vals.max(), 200)
        axes1[2].plot(_xs, np.poly1d(np.polyfit(fitted_vals, residuals, 2))(_xs),
                      color='orange', lw=1.5, label='Trend')
        axes1[2].set_xlabel('Fitted'); axes1[2].set_ylabel('Residuals')
        axes1[2].set_title('Residuals vs Fitted', fontweight='bold')
        axes1[2].legend(fontsize=8)
        plt.tight_layout()
        buf1 = io.BytesIO(); fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig1); buf1.seek(0)
        plot_main = f"data:image/png;base64,{base64.b64encode(buf1.read()).decode()}"

        # Plot 2: OLS vs Lasso vs ElasticNet bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        _models  = ['OLS', 'Lasso', 'ElasticNet']
        _r2s     = [model_comparison[k.lower()]['test_r2']   for k in _models]
        _rmses   = [model_comparison[k.lower()]['test_rmse']  for k in _models]
        _colors  = ['#F4A582', '#ABDDA4', '#5B9BD5']
        _xp      = np.arange(len(_models))
        b1 = ax2.bar(_xp - 0.2, _r2s,  0.35, label='Test R²',   color=_colors, alpha=0.85, edgecolor='gray', linewidth=0.7)
        b2 = ax2.bar(_xp + 0.2, _rmses, 0.35, label='Test RMSE', color=_colors, alpha=0.5,  edgecolor='gray', linewidth=0.7, hatch='//')
        for bar, val in list(zip(b1, _r2s)) + list(zip(b2, _rmses)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.set_xticks(_xp); ax2.set_xticklabels(_models)
        ax2.set_title('OLS vs Lasso vs ElasticNet (test set)', fontweight='bold')
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf2 = io.BytesIO(); fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig2); buf2.seek(0)
        plot_compare = f"data:image/png;base64,{base64.b64encode(buf2.read()).decode()}"

        # Plot 3: Coefficient comparison (Lasso vs ElasticNet)
        fig3, ax3 = plt.subplots(figsize=(8, max(4, len(final_features) * 0.45 + 2)))
        _ypos = np.arange(len(final_features))
        ax3.barh(_ypos - 0.18, lasso_m.coef_, 0.33, label='Lasso',      color='#ABDDA4', edgecolor='gray', linewidth=0.6)
        ax3.barh(_ypos + 0.18, model.coef_,   0.33, label='ElasticNet', color='#5B9BD5', edgecolor='gray', linewidth=0.6)
        ax3.axvline(0, color='red', linestyle='--', lw=1)
        ax3.set_yticks(_ypos); ax3.set_yticklabels(final_features, fontsize=9)
        ax3.set_xlabel('Coefficient (standardised)')
        ax3.set_title('Coefficient Comparison: Lasso vs ElasticNet', fontweight='bold')
        ax3.legend()
        plt.tight_layout()
        buf3 = io.BytesIO(); fig3.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig3); buf3.seek(0)
        plot_coef = f"data:image/png;base64,{base64.b64encode(buf3.read()).decode()}"

        # ── Response ─────────────────────────────────────────────────────
        return _to_native({
            'results': {
                'metrics':     {'train': train_metrics, 'test': test_metrics},
                'alpha':       float(alpha),
                'l1_ratio':    float(l1_ratio),
                'l1_ratio_interpretation': mix_label,
                'alpha_source': 'cross_validation' if req.use_cv else 'user_specified',
                'cv_results':  cv_results,
                'coefficients': dict(zip(final_features, model.coef_.tolist())),
                'intercept':   float(model.intercept_),
                'feature_selection': {
                    'n_total':    len(final_features),
                    'n_selected': len(selected),
                    'n_excluded': len(excluded),
                    'selected':   selected,
                    'excluded':   excluded,
                },
                'model_comparison':     model_comparison,
                'residual_diagnostics': residual_diagnostics,
                'interpretation': interp,
                'n_dropped': n_dropped,
                'n_total':   len(y),
                'n_train':   len(y_train),
                'n_test':    len(y_test),
            },
            'plot':         plot_main,
            'compare_plot': plot_compare,
            'coef_plot':    plot_coef,
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ElasticNet failed: {str(e)}")
