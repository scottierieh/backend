
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import io
import base64
import warnings

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def _generate_interpretation(train_r2, test_r2):
    interpretation = ""
    r2_diff = train_r2 - test_r2
    
    if train_r2 > 0.8 and r2_diff < 0.2:
        interpretation = "The model shows a **Good Fit**. Both training and testing R-squared scores are high and close to each other, indicating that the model generalizes well to new data."
    elif train_r2 > 0.7 and r2_diff > 0.3:
        interpretation = "**Overfitting Warning**. The model performs significantly better on the training data than on the test data. This suggests that the model has learned the training data's noise and may not perform well on unseen data. Consider increasing the alpha value to add more regularization and potentially simplify the model."
    elif train_r2 < 0.5 and test_r2 < 0.5:
        interpretation = "**Underfitting Possible**. Both training and testing R-squared scores are low, suggesting the model is too simple to capture the underlying patterns in the data. The model may not be complex enough, or the features may not have a strong linear relationship with the target."
    else:
        interpretation = "The model's performance is moderate. Review the R-squared values and residuals to assess if the model is sufficient for your needs. The difference between train and test scores suggests some degree of overfitting might be present."
        
    return interpretation.strip()


def _compute_cv_results(X, y, alpha, cv_folds=5):
    """K-fold CV of the Lasso model (with its own scaler) on the FULL dataset.
    Mirrors the {r2_mean, r2_std, rmse_mean, rmse_std, n_folds, scores} shape
    expected by the lasso-regression-page.tsx frontend."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=alpha, random_state=42, max_iter=10000)),
    ])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    r2_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    neg_mse_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    return {
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'rmse_mean': float(np.mean(rmse_scores)),
        'rmse_std': float(np.std(rmse_scores)),
        'n_folds': cv_folds,
        'scores': [float(s) for s in r2_scores],
        'mean_cv_score': float(np.mean(r2_scores)),
        'std_cv_score': float(np.std(r2_scores)),
    }


def _compute_ols_comparison(X_train_scaled, X_test_scaled, y_train, y_test,
                             lasso_train_r2, lasso_test_r2, lasso_test_rmse,
                             n_features_ols, n_features_lasso):
    """Fit a plain (unregularized) LinearRegression on the same standardized
    train/test split as Lasso, for a fair head-to-head comparison."""
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_pred_train = ols.predict(X_train_scaled)
    ols_pred_test = ols.predict(X_test_scaled)

    ols_train_r2 = float(r2_score(y_train, ols_pred_train))
    ols_test_r2 = float(r2_score(y_test, ols_pred_test))
    ols_test_rmse = float(np.sqrt(mean_squared_error(y_test, ols_pred_test)))
    delta_test_r2 = ols_test_r2 - float(lasso_test_r2)

    return {
        'ols_train_r2': ols_train_r2,
        'ols_test_r2': ols_test_r2,
        'ols_test_rmse': ols_test_rmse,
        'lasso_train_r2': float(lasso_train_r2),
        'lasso_test_r2': float(lasso_test_r2),
        'lasso_test_rmse': float(lasso_test_rmse),
        'n_features_ols': int(n_features_ols),
        'n_features_lasso': int(n_features_lasso),
        'delta_test_r2': delta_test_r2,
        'note': (
            'Positive ΔTest R² (OLS − Lasso) means unregularized OLS outperformed Lasso on the '
            'held-out test set; Lasso trades some R² for a sparser, more interpretable model.'
        ),
    }


def _compute_residual_diagnostics(residuals, fitted):
    """Standard residual diagnostics on test-set residuals: mean/std, skewness,
    kurtosis, a Shapiro-Wilk normality test (guarded for sample size), and a
    simple heteroscedasticity check (correlation of |residual| with fitted values)."""
    residuals = np.asarray(residuals, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    n = len(residuals)

    mean_r = float(np.mean(residuals)) if n > 0 else 0.0
    std_r = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
    skewness = float(scipy_stats.skew(residuals)) if n > 2 else 0.0
    kurtosis = float(scipy_stats.kurtosis(residuals)) if n > 3 else 0.0

    if 3 <= n <= 5000:
        sw_stat, sw_p = scipy_stats.shapiro(residuals)
        shapiro_wilk = {
            'statistic': float(sw_stat),
            'p_value': float(sw_p),
            'normal': bool(sw_p > 0.05),
        }
    else:
        shapiro_wilk = {'statistic': None, 'p_value': None, 'normal': None}

    if n > 2 and np.std(fitted) > 0 and np.std(np.abs(residuals)) > 0:
        corr, p_val = scipy_stats.pearsonr(fitted, np.abs(residuals))
        heteroscedasticity = {
            'corr_fitted_abs_resid': float(corr),
            'p_value': float(p_val),
            'detected': bool(p_val < 0.05),
        }
    else:
        heteroscedasticity = {'corr_fitted_abs_resid': 0.0, 'p_value': 1.0, 'detected': False}

    return {
        'mean': mean_r,
        'std': std_r,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_wilk': shapiro_wilk,
        'heteroscedasticity': heteroscedasticity,
    }

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target = payload.get('target')
        features = payload.get('features')
        alpha = float(payload.get('alpha', 1.0))
        test_size = float(payload.get('test_size', 0.2))
        auto_tune = bool(payload.get('use_cv', False))
        cv_folds = int(payload.get('cv_folds', 5) or 5)
        cv_folds = max(2, cv_folds)

        if not all([data, target, features]):
            raise ValueError("Missing data, target, or features")

        df = pd.DataFrame(data)
        
        X = df[features]
        y = df[target]
        
        X = pd.get_dummies(X, drop_first=True)
        final_features = X.columns.tolist()

        y = pd.to_numeric(y, errors='coerce')
        
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[final_features]
        y = combined[target]
        
        if X.empty or y.empty:
            raise ValueError("Not enough valid data after cleaning.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        alpha_source = 'user_specified'
        if auto_tune:
            cv_model = LassoCV(alphas=np.logspace(-3, 2, 100), cv=cv_folds, random_state=42, max_iter=10000)
            cv_model.fit(X_train_scaled, y_train)
            alpha = float(cv_model.alpha_)
            alpha_source = 'cross_validation'

        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred_test = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)
        
        test_metrics = {
            'r2_score': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test)
        }
        
        train_metrics = {
            'r2_score': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train)
        }
        
        interpretation = _generate_interpretation(train_metrics['r2_score'], test_metrics['r2_score'])

        n_features_total = len(final_features)
        n_features_selected = int(np.sum(model.coef_ != 0))

        residuals_test = np.asarray(y_test) - np.asarray(y_pred_test)
        residual_diagnostics = _compute_residual_diagnostics(residuals_test, y_pred_test)
        ols_comparison = _compute_ols_comparison(
            X_train_scaled, X_test_scaled, y_train, y_test,
            train_metrics['r2_score'], test_metrics['r2_score'], test_metrics['rmse'],
            n_features_total, n_features_selected,
        )
        cv_results = _compute_cv_results(X, y, alpha, cv_folds=cv_folds)

        results = {
            'metrics': { 'test': test_metrics, 'train': train_metrics },
            'coefficients': dict(zip(final_features, model.coef_)),
            'intercept': model.intercept_,
            'alpha': alpha,
            'alpha_source': alpha_source,
            'cv_folds': cv_results['n_folds'],
            'interpretation': interpretation,
            'feature_selection': {
                'n_total': n_features_total,
                'n_selected': n_features_selected,
                'n_excluded': n_features_total - n_features_selected,
                'selected': [f for f, c in zip(final_features, model.coef_) if c != 0],
                'excluded': [f for f, c in zip(final_features, model.coef_) if c == 0],
            },
            'ols_comparison': ols_comparison,
            'cv_results': cv_results,
            'residual_diagnostics': residual_diagnostics,
        }

        fig_main, axes = plt.subplots(2, 1, figsize=(8, 12))
        fig_main.suptitle(f'Lasso Regression Performance (alpha={alpha})', fontsize=16)

        axes[0].scatter(y_train, y_pred_train, alpha=0.5, label='(Actual, Predicted)')
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='45° Line (Perfect Fit)')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Train Set Performance')
        axes[0].legend()
        axes[0].grid(True)
        train_text = (
            f"Train R²: {train_metrics['r2_score']:.4f}\n"
            f"Train RMSE: {train_metrics['rmse']:.4f}"
        )
        axes[0].text(0.05, 0.95, train_text, transform=axes[0].transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        axes[1].scatter(y_test, y_pred_test, alpha=0.5, label='(Actual, Predicted)')
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='45° Line (Perfect Fit)')
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        axes[1].set_title('Test Set Performance')
        axes[1].legend()
        axes[1].grid(True)
        test_text = (
            f"Test R²: {test_metrics['r2_score']:.4f}\n"
            f"Test RMSE: {test_metrics['rmse']:.4f}"
        )
        axes[1].text(0.05, 0.95, test_text, transform=axes[1].transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_image = fig_to_base64(fig_main)

        alpha_list = np.logspace(-3, 2, 100)
        coefs = []
        train_scores, test_scores = [], []
        for a in alpha_list:
            lasso_iter = Lasso(alpha=a, random_state=42, max_iter=1000)
            lasso_iter.fit(X_train_scaled, y_train)
            coefs.append(lasso_iter.coef_)
            train_scores.append(lasso_iter.score(X_train_scaled, y_train))
            test_scores.append(lasso_iter.score(X_test_scaled, y_test))

        fig_path, axes_path = plt.subplots(2, 1, figsize=(8, 12))
        fig_path.suptitle('Lasso Model Behavior vs. Alpha', fontsize=16)

        axes_path[0].plot(alpha_list, train_scores, label='Train R²')
        axes_path[0].plot(alpha_list, test_scores, label='Test R²')
        axes_path[0].set_xlabel('Alpha')
        axes_path[0].set_ylabel('R-squared')
        axes_path[0].set_xscale('log')
        axes_path[0].set_title('R-squared vs. Regularization Strength (alpha)')
        axes_path[0].legend()
        axes_path[0].grid(True)

        axes_path[1].plot(alpha_list, coefs)
        axes_path[1].set_xscale('log')
        axes_path[1].set_xlabel('Alpha')
        axes_path[1].set_ylabel('Coefficients')
        axes_path[1].set_title('Lasso Coefficients Path')
        axes_path[1].grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path_plot_image = fig_to_base64(fig_path)

        # Residual diagnostics plot: residuals vs fitted (with quadratic trend) + Normal Q-Q
        fig_resid, axes_resid = plt.subplots(1, 2, figsize=(12, 5))
        fig_resid.suptitle('Residual Diagnostics (Test Set)', fontsize=14)

        axes_resid[0].scatter(y_pred_test, residuals_test, alpha=0.5)
        axes_resid[0].axhline(0, color='r', linestyle='--', lw=1.5)
        if len(y_pred_test) > 2:
            order = np.argsort(y_pred_test)
            trend_coefs = np.polyfit(np.asarray(y_pred_test)[order], np.asarray(residuals_test)[order], 2)
            trend = np.poly1d(trend_coefs)
            axes_resid[0].plot(np.asarray(y_pred_test)[order], trend(np.asarray(y_pred_test)[order]), color='orange', lw=2, label='Trend')
            axes_resid[0].legend()
        axes_resid[0].set_xlabel('Fitted Values')
        axes_resid[0].set_ylabel('Residuals')
        axes_resid[0].set_title('Residuals vs Fitted')
        axes_resid[0].grid(True)

        scipy_stats.probplot(residuals_test, dist='norm', plot=axes_resid[1])
        axes_resid[1].set_title('Normal Q-Q Plot')
        axes_resid[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        residual_plot_image = fig_to_base64(fig_resid)

        # OLS vs Lasso comparison plot
        fig_ols, ax_ols = plt.subplots(figsize=(7, 5))
        metric_labels = ['Train R²', 'Test R²', 'Test RMSE']
        ols_vals = [ols_comparison['ols_train_r2'], ols_comparison['ols_test_r2'], ols_comparison['ols_test_rmse']]
        lasso_vals = [ols_comparison['lasso_train_r2'], ols_comparison['lasso_test_r2'], ols_comparison['lasso_test_rmse']]
        x_pos = np.arange(len(metric_labels))
        width = 0.35
        ax_ols.bar(x_pos - width / 2, ols_vals, width, label='OLS')
        ax_ols.bar(x_pos + width / 2, lasso_vals, width, label='Lasso')
        ax_ols.set_xticks(x_pos)
        ax_ols.set_xticklabels(metric_labels)
        ax_ols.set_title('OLS vs Lasso Comparison')
        ax_ols.legend()
        ax_ols.grid(True, axis='y')

        plt.tight_layout()
        ols_plot_image = fig_to_base64(fig_ols)

        response = {
            'results': results,
            'plot': plot_image,
            'path_plot': path_plot_image,
            'residual_plot': residual_plot_image,
            'ols_plot': ols_plot_image,
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
