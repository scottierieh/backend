
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, Lasso
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
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def _generate_interpretation(train_r2, test_r2, l1_ratio):
    r2_diff = train_r2 - test_r2

    if train_r2 > 0.8 and r2_diff < 0.2:
        fit_desc = "The model shows a **Good Fit**. Both training and testing R-squared scores are high and close to each other, indicating that the model generalizes well to new data."
    elif train_r2 > 0.7 and r2_diff > 0.3:
        fit_desc = "**Overfitting Warning**. The model performs significantly better on the training data than on the test data. Consider increasing alpha to add more regularization."
    elif train_r2 < 0.5 and test_r2 < 0.5:
        fit_desc = "**Underfitting Possible**. Both training and testing R-squared scores are low, suggesting the model is too simple to capture the underlying patterns, or the features lack a strong relationship with the target."
    else:
        fit_desc = "The model's performance is moderate. Review the R-squared values and residuals to assess if the model is sufficient for your needs."

    if l1_ratio >= 0.9:
        mix_desc = "With an L1 ratio close to 1, the model behaves almost like Lasso, favoring sparse solutions that can zero out weak predictors."
    elif l1_ratio <= 0.1:
        mix_desc = "With an L1 ratio close to 0, the model behaves almost like Ridge, shrinking coefficients smoothly without eliminating them."
    else:
        mix_desc = f"With an L1 ratio of {l1_ratio:.2f}, the model blends Lasso's variable selection with Ridge's coefficient shrinkage."

    return f"{fit_desc.strip()} {mix_desc}"

def _l1_ratio_interpretation(l1_ratio):
    """Short practical label for where l1_ratio sits between pure Ridge (0) and pure Lasso (1)."""
    if l1_ratio >= 0.9:
        return "near-Lasso (favors sparse, individually-selected predictors)"
    elif l1_ratio >= 0.55:
        return "Lasso-leaning (blends selection and shrinkage, tilted toward sparsity)"
    elif l1_ratio > 0.45:
        return "balanced (equal mix of Lasso-style selection and Ridge-style shrinkage)"
    elif l1_ratio > 0.1:
        return "Ridge-leaning (blends shrinkage and selection, tilted toward smoothness)"
    else:
        return "near-Ridge (favors grouped shrinkage of correlated predictors)"

def perform_cross_validation(X, y, alpha, l1_ratio, cv_folds=5):
    """K-fold CV on the full (unscaled) dataset via a scaler+model Pipeline, so each
    fold's scaler only ever sees its own training portion. Shape matches the
    {r2_mean, r2_std, rmse_mean, rmse_std, n_folds, scores} contract the frontend's
    CvResults interface and the DOCX export route already expect for this analysis."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)),
    ])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_out = cross_validate(
        pipeline, X, y, cv=kf,
        scoring={'r2': 'r2', 'rmse': 'neg_root_mean_squared_error'},
    )
    r2_scores = cv_out['test_r2']
    rmse_scores = -cv_out['test_rmse']
    return {
        'r2_mean': _to_native_type(np.mean(r2_scores)),
        'r2_std': _to_native_type(np.std(r2_scores)),
        'rmse_mean': _to_native_type(np.mean(rmse_scores)),
        'rmse_std': _to_native_type(np.std(rmse_scores)),
        'n_folds': cv_folds,
        'scores': [_to_native_type(s) for s in r2_scores],
    }

def compute_residual_diagnostics(y_test, y_pred_test):
    """Residual mean/std/skew/kurtosis, Shapiro-Wilk normality (guarded for sample
    size — scipy requires n>=3 and is unreliable/slow well past a few thousand),
    and a heteroscedasticity check via corr(fitted, |resid|)."""
    residuals = np.asarray(y_test) - np.asarray(y_pred_test)
    n = len(residuals)

    mean = float(np.mean(residuals)) if n > 0 else None
    std = float(np.std(residuals)) if n > 0 else None
    skewness = _to_native_type(scipy_stats.skew(residuals)) if n >= 3 else None
    kurt = _to_native_type(scipy_stats.kurtosis(residuals)) if n >= 3 else None

    if 3 <= n <= 5000:
        try:
            sw_stat, sw_p = scipy_stats.shapiro(residuals)
            shapiro_wilk = {
                'statistic': _to_native_type(sw_stat),
                'p_value': _to_native_type(sw_p),
                'normal': bool(sw_p > 0.05),
            }
        except Exception:
            shapiro_wilk = {'statistic': None, 'p_value': None, 'normal': None}
    else:
        shapiro_wilk = {'statistic': None, 'p_value': None, 'normal': None}

    try:
        corr, het_p = scipy_stats.pearsonr(np.asarray(y_pred_test), np.abs(residuals))
        heteroscedasticity = {
            'corr_fitted_abs_resid': _to_native_type(corr),
            'p_value': _to_native_type(het_p),
            'detected': bool(het_p < 0.05),
        }
    except Exception:
        heteroscedasticity = {'corr_fitted_abs_resid': None, 'p_value': None, 'detected': None}

    return {
        'mean': mean, 'std': std, 'skewness': skewness, 'kurtosis': kurt,
        'shapiro_wilk': shapiro_wilk, 'heteroscedasticity': heteroscedasticity,
    }

def generate_compare_plot(model_comparison):
    """Bar chart: Test R² and Test RMSE for OLS vs. Lasso vs. Elastic Net."""
    models = ['OLS', 'Lasso', 'Elastic Net']
    r2_vals = [model_comparison['ols']['test_r2'], model_comparison['lasso']['test_r2'], model_comparison['elasticnet']['test_r2']]
    rmse_vals = [model_comparison['ols']['test_rmse'], model_comparison['lasso']['test_rmse'], model_comparison['elasticnet']['test_rmse']]
    colors = sns.color_palette('husl', n_colors=3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Comparison: OLS vs. Lasso vs. Elastic Net', fontsize=15)

    axes[0].bar(models, r2_vals, color=colors)
    axes[0].set_title('Test R² by Model')
    axes[0].set_ylabel('R²')
    axes[0].grid(True, axis='y')
    for i, v in enumerate(r2_vals):
        axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

    axes[1].bar(models, rmse_vals, color=colors)
    axes[1].set_title('Test RMSE by Model')
    axes[1].set_ylabel('RMSE')
    axes[1].grid(True, axis='y')
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig_to_base64(fig)

def generate_coef_plot(final_features, en_coefs, lasso_coefs, max_bars=20):
    """Horizontal grouped bar chart: Lasso vs. Elastic Net standardized coefficients.
    OLS is intentionally excluded here (see model_comparison for that) — OLS
    coefficients are unregularized and on a very different scale, so this panel
    focuses on the two sparsity-inducing methods, matching the DOCX caption
    ('Coefficient comparison: Lasso vs. Elastic Net')."""
    pairs = list(zip(final_features, en_coefs, lasso_coefs))
    pairs.sort(key=lambda p: max(abs(p[1]), abs(p[2])), reverse=True)
    pairs = pairs[:max_bars]
    feats = [p[0] for p in pairs][::-1]
    en_v = [p[1] for p in pairs][::-1]
    la_v = [p[2] for p in pairs][::-1]
    y_pos = np.arange(len(feats))
    height = 0.35
    colors = sns.color_palette('husl', n_colors=3)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feats) * 0.4)))
    ax.barh(y_pos - height / 2, la_v, height=height, label='Lasso', color=colors[0])
    ax.barh(y_pos + height / 2, en_v, height=height, label='Elastic Net', color=colors[2])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Standardized Coefficient')
    ax.set_title('Coefficients: Lasso vs. Elastic Net')
    ax.legend()
    ax.grid(True, axis='x')
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_l1_ratio_path_plot(X_train_scaled, X_test_scaled, y_train, y_test, alpha):
    """Scans l1_ratio at the fitted alpha — Elastic Net's second tuning axis, which
    (unlike the alpha path) had no visualization at all before this."""
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    coefs = []
    test_r2s = []
    for lr in l1_ratios:
        m = ElasticNet(alpha=alpha, l1_ratio=lr, random_state=42, max_iter=10000)
        m.fit(X_train_scaled, y_train)
        coefs.append(m.coef_)
        test_r2s.append(m.score(X_test_scaled, y_test))
    coefs_arr = np.array(coefs)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle(f'Elastic Net Behavior vs. l1_ratio (alpha={alpha:.4f})', fontsize=15)

    axes[0].plot(l1_ratios, test_r2s, marker='o', color=sns.color_palette('husl', 1)[0])
    axes[0].set_xlabel('l1_ratio (0 = Ridge, 1 = Lasso)')
    axes[0].set_ylabel('Test R²')
    axes[0].set_title('Test R² vs. l1_ratio')
    axes[0].grid(True)

    for j in range(coefs_arr.shape[1]):
        axes[1].plot(l1_ratios, coefs_arr[:, j], marker='o', markersize=3)
    axes[1].set_xlabel('l1_ratio (0 = Ridge, 1 = Lasso)')
    axes[1].set_ylabel('Coefficients')
    axes[1].set_title('Coefficient Path vs. l1_ratio')
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig_to_base64(fig)

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target = payload.get('target')
        features = payload.get('features')
        alpha = float(payload.get('alpha', 1.0))
        l1_ratio = float(payload.get('l1_ratio', 0.5))
        test_size = float(payload.get('test_size', 0.2))
        auto_tune = bool(payload.get('use_cv', payload.get('auto_tune', False)))
        cv_folds = int(payload.get('cv_folds', 5))

        if not all([data, target, features]):
            raise ValueError("Missing data, target, or features")

        df = pd.DataFrame(data)
        n_total_rows = len(df)

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

        n_dropped = n_total_rows - len(combined)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        alpha_source = 'user_specified'
        if auto_tune:
            l1_ratio_grid = [.1, .3, .5, .7, .9, .95, .99, 1]
            cv_model = ElasticNetCV(l1_ratio=l1_ratio_grid, alphas=np.logspace(-3, 2, 50), cv=cv_folds, random_state=42, max_iter=10000)
            cv_model.fit(X_train_scaled, y_train)
            alpha = float(cv_model.alpha_)
            l1_ratio = float(cv_model.l1_ratio_)
            alpha_source = 'cross_validation'

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
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

        interpretation = _generate_interpretation(train_metrics['r2_score'], test_metrics['r2_score'], l1_ratio)
        cv_result = perform_cross_validation(X, y, alpha, l1_ratio, cv_folds)

        # ── Feature selection: which coefficients ElasticNet shrunk to exactly zero ──
        selected_features = [f for f, c in zip(final_features, model.coef_) if c != 0]
        excluded_features = [f for f in final_features if f not in selected_features]
        feature_selection = {
            'n_total': len(final_features),
            'n_selected': len(selected_features),
            'n_excluded': len(excluded_features),
            'selected': selected_features,
            'excluded': excluded_features,
        }

        # ── Model comparison: OLS and pure Lasso baselines on the same split ──
        ols_model = LinearRegression()
        ols_model.fit(X_train_scaled, y_train)
        ols_pred_test = ols_model.predict(X_test_scaled)
        ols_test_r2 = r2_score(y_test, ols_pred_test)
        ols_test_rmse = np.sqrt(mean_squared_error(y_test, ols_pred_test))

        lasso_model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        lasso_model.fit(X_train_scaled, y_train)
        lasso_pred_test = lasso_model.predict(X_test_scaled)
        lasso_test_r2 = r2_score(y_test, lasso_pred_test)
        lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred_test))
        lasso_n_features = int(np.sum(lasso_model.coef_ != 0))
        en_n_features = int(np.sum(model.coef_ != 0))

        if test_metrics['r2_score'] >= ols_test_r2:
            mc_note = (f"Elastic Net matched or exceeded OLS on test R² "
                       f"({test_metrics['r2_score']:.3f} vs {ols_test_r2:.3f}) while using "
                       f"{en_n_features} of {len(final_features)} features.")
        else:
            mc_note = (f"OLS outperformed Elastic Net on test R² "
                       f"({ols_test_r2:.3f} vs {test_metrics['r2_score']:.3f}); the regularization "
                       f"may be too strong for this data.")

        model_comparison = {
            'ols': {
                'test_r2': _to_native_type(ols_test_r2),
                'test_rmse': _to_native_type(ols_test_rmse),
                'n_features': len(final_features),
            },
            'lasso': {
                'test_r2': _to_native_type(lasso_test_r2),
                'test_rmse': _to_native_type(lasso_test_rmse),
                'n_features': lasso_n_features,
            },
            'elasticnet': {
                'test_r2': _to_native_type(test_metrics['r2_score']),
                'test_rmse': _to_native_type(test_metrics['rmse']),
                'n_features': en_n_features,
            },
            'note': mc_note,
        }

        residual_diagnostics = compute_residual_diagnostics(y_test, y_pred_test)

        try:
            n_iter_val = int(np.max(model.n_iter_)) if getattr(model, 'n_iter_', None) is not None else None
        except Exception:
            n_iter_val = None
        max_iter_val = 10000
        converged_val = None if n_iter_val is None else bool(n_iter_val < max_iter_val)

        results = {
            'metrics': {'test': test_metrics, 'train': train_metrics},
            'cv_results': cv_result,
            'coefficients': dict(zip(final_features, model.coef_)),
            'intercept': model.intercept_,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'l1_ratio_interpretation': _l1_ratio_interpretation(l1_ratio),
            'alpha_source': alpha_source,
            'n_nonzero_coefficients': en_n_features,
            'feature_selection': feature_selection,
            'model_comparison': model_comparison,
            'residual_diagnostics': residual_diagnostics,
            'interpretation': interpretation,
            'n_dropped': int(n_dropped),
            'n_total': int(len(combined)),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'n_iter': n_iter_val,
            'max_iter': max_iter_val,
            'converged': converged_val,
        }

        fig_main, axes = plt.subplots(2, 1, figsize=(8, 12))
        fig_main.suptitle(f'Elastic Net Regression Performance (alpha={alpha:.4f}, l1_ratio={l1_ratio:.2f})', fontsize=16)

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
            en_iter = ElasticNet(alpha=a, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
            en_iter.fit(X_train_scaled, y_train)
            coefs.append(en_iter.coef_)
            train_scores.append(en_iter.score(X_train_scaled, y_train))
            test_scores.append(en_iter.score(X_test_scaled, y_test))

        fig_path, axes_path = plt.subplots(2, 1, figsize=(8, 12))
        fig_path.suptitle(f'Elastic Net Model Behavior vs. Alpha (l1_ratio={l1_ratio:.2f})', fontsize=16)

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
        axes_path[1].set_title('Elastic Net Coefficients Path')
        axes_path[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path_plot_image = fig_to_base64(fig_path)

        compare_plot_image = generate_compare_plot(model_comparison)
        coef_plot_image = generate_coef_plot(final_features, model.coef_, lasso_model.coef_)
        l1_ratio_path_plot_image = generate_l1_ratio_path_plot(X_train_scaled, X_test_scaled, y_train, y_test, alpha)

        try:
            from guardrails import compute_guardrails
            guardrails = compute_guardrails(X, y, features, 'regression', {'r2': test_metrics['r2_score']})
        except Exception:
            guardrails = []

        response = {
            'results': results,
            'guardrails': guardrails,
            'plot': plot_image,
            'path_plot': path_plot_image,
            'compare_plot': compare_plot_image,
            'coef_plot': coef_plot_image,
            'l1_ratio_path_plot': l1_ratio_path_plot_image,
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
