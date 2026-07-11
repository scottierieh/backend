import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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


def _metric_set(y_true, y_pred):
    return {
        'r2_score': r2_score(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': mean_absolute_error(y_true, y_pred),
        'n_samples': int(len(y_true)),
    }


def _l1_ratio_interpretation(l1_ratio):
    if l1_ratio >= 0.9:
        return "Close to pure Lasso (L1) — strong variable selection, many coefficients driven to exactly zero."
    if l1_ratio <= 0.1:
        return "Close to pure Ridge (L2) — coefficients are shrunk smoothly, few (if any) set to exactly zero."
    return f"A balanced L1/L2 mix (l1_ratio = {l1_ratio:.2f}) — combines Lasso's variable selection with Ridge's stable shrinkage."


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target = payload.get('target')
        features = payload.get('features')
        alpha = float(payload.get('alpha', 1.0))
        l1_ratio = float(payload.get('l1_ratio', 0.5))
        test_size = float(payload.get('test_size', 0.2))
        use_cv = bool(payload.get('use_cv', False))
        cv_folds = int(payload.get('cv_folds', 5))

        if not all([data, target, features]):
            raise ValueError("Missing data, target, or features")

        df = pd.DataFrame(data)
        X = df[features]
        y = df[target]
        X = pd.get_dummies(X, drop_first=True)
        final_features = X.columns.tolist()
        y = pd.to_numeric(y, errors='coerce')

        n_original = len(df)
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[final_features]
        y = combined[target]
        n_total = int(len(y))
        n_dropped = int(n_original - n_total)
        if X.empty or y.empty:
            raise ValueError("Not enough valid data after cleaning.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        n_train, n_test = int(len(y_train)), int(len(y_test))

        # ---- hyperparameter selection ----
        alpha_source = 'user_specified'
        if use_cv:
            l1_grid = [.1, .3, .5, .7, .9, .95, .99, 1]
            cv_model = ElasticNetCV(l1_ratio=l1_grid, alphas=np.logspace(-3, 2, 50),
                                    cv=cv_folds, random_state=42, max_iter=10000)
            cv_model.fit(X_train_scaled, y_train)
            alpha = float(cv_model.alpha_)
            l1_ratio = float(cv_model.l1_ratio_)
            alpha_source = 'cross_validation'

        # ---- fit the Elastic Net ----
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        train_metrics = _metric_set(y_train, y_pred_train)
        test_metrics = _metric_set(y_test, y_pred_test)

        # ---- cross-validated performance of the chosen model ----
        r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
        rmse_scores = np.sqrt(-cross_val_score(model, X_train_scaled, y_train, cv=cv_folds,
                                               scoring='neg_mean_squared_error'))
        cv_results = {
            'r2_mean': float(np.mean(r2_scores)), 'r2_std': float(np.std(r2_scores)),
            'rmse_mean': float(np.mean(rmse_scores)), 'rmse_std': float(np.std(rmse_scores)),
            'n_folds': int(cv_folds), 'scores': r2_scores.tolist(),
        }

        # ---- coefficients + feature selection ----
        coefs = dict(zip(final_features, model.coef_))
        selected = [f for f, c in coefs.items() if c != 0]
        excluded = [f for f in final_features if f not in selected]
        feature_selection = {
            'n_total': len(final_features), 'n_selected': len(selected),
            'n_excluded': len(excluded), 'selected': selected, 'excluded': excluded,
        }

        # ---- model comparison: OLS vs pure Lasso vs Elastic Net (test set) ----
        ols = LinearRegression().fit(X_train_scaled, y_train)
        ols_pred = ols.predict(X_test_scaled)
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000).fit(X_train_scaled, y_train)
        lasso_pred = lasso.predict(X_test_scaled)
        model_comparison = {
            'ols': {'test_r2': r2_score(y_test, ols_pred),
                    'test_rmse': float(np.sqrt(mean_squared_error(y_test, ols_pred))),
                    'n_features': int(np.sum(ols.coef_ != 0))},
            'lasso': {'test_r2': r2_score(y_test, lasso_pred),
                      'test_rmse': float(np.sqrt(mean_squared_error(y_test, lasso_pred))),
                      'n_features': int(np.sum(lasso.coef_ != 0))},
            'elasticnet': {'test_r2': test_metrics['r2_score'], 'test_rmse': test_metrics['rmse'],
                           'n_features': int(np.sum(model.coef_ != 0))},
            'note': (f"OLS uses all {len(final_features)} feature(s); "
                     f"Lasso retained {int(np.sum(lasso.coef_ != 0))}; "
                     f"Elastic Net (l1_ratio={l1_ratio:.2f}) retained {len(selected)}."),
        }

        # ---- residual diagnostics (training residuals) ----
        resid_train = np.asarray(y_train) - y_pred_train
        if 3 <= n_train <= 5000 and len(np.unique(resid_train)) > 1:
            sw_stat, sw_p = stats.shapiro(resid_train)
            shapiro = {'statistic': float(sw_stat), 'p_value': float(sw_p), 'normal': bool(sw_p > 0.05)}
        else:
            shapiro = {'statistic': None, 'p_value': None, 'normal': None}
        abs_resid = np.abs(resid_train)
        if len(np.unique(y_pred_train)) > 1:
            het_corr, het_p = stats.pearsonr(y_pred_train, abs_resid)
        else:
            het_corr, het_p = 0.0, 1.0
        residual_diagnostics = {
            'mean': float(np.mean(resid_train)), 'std': float(np.std(resid_train, ddof=1)),
            'skewness': float(stats.skew(resid_train)), 'kurtosis': float(stats.kurtosis(resid_train)),
            'shapiro_wilk': shapiro,
            'heteroscedasticity': {'corr_fitted_abs_resid': float(het_corr), 'p_value': float(het_p),
                                   'detected': bool(het_p < 0.05)},
        }

        gap = train_metrics['r2_score'] - test_metrics['r2_score']
        interpretation = (
            f"Elastic Net (alpha={alpha:.4g}, l1_ratio={l1_ratio:.2f}) explains "
            f"{test_metrics['r2_score']*100:.1f}% of test variance (train R²={train_metrics['r2_score']:.3f}, "
            f"gap={gap:.3f}); test RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}. "
            f"{len(selected)} of {len(final_features)} feature(s) retained. {model_comparison['note']}"
        )

        results = {
            'metrics': {'train': train_metrics, 'test': test_metrics},
            'alpha': alpha, 'l1_ratio': l1_ratio,
            'l1_ratio_interpretation': _l1_ratio_interpretation(l1_ratio),
            'alpha_source': alpha_source,
            'cv_results': cv_results,
            'coefficients': coefs, 'intercept': float(model.intercept_),
            'feature_selection': feature_selection,
            'model_comparison': model_comparison,
            'residual_diagnostics': residual_diagnostics,
            'n_nonzero_coefficients': int(np.sum(model.coef_ != 0)),
            'interpretation': interpretation,
            'n_dropped': n_dropped, 'n_total': n_total, 'n_train': n_train, 'n_test': n_test,
            'max_iter': 10000, 'converged': bool(model.n_iter_ < 10000), 'n_iter': int(model.n_iter_),
        }

        # ---- plot: actual vs predicted (train/test) ----
        fig_main, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, yt, yp, name, m in ((axes[0], y_train, y_pred_train, 'Train', train_metrics),
                                    (axes[1], y_test, y_pred_test, 'Test', test_metrics)):
            ax.scatter(yt, yp, alpha=0.5)
            lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            ax.plot(lims, lims, 'r--', lw=2)
            ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
            ax.set_title(f"{name} (R²={m['r2_score']:.3f}, RMSE={m['rmse']:.3f})")
            ax.grid(True)
        fig_main.suptitle(f'Elastic Net Performance (alpha={alpha:.4g}, l1_ratio={l1_ratio:.2f})')
        plt.tight_layout()
        plot_image = fig_to_base64(fig_main)

        # ---- compare_plot: OLS vs Lasso vs Elastic Net test R² ----
        fig_cmp, axc = plt.subplots(figsize=(7, 5))
        names = ['OLS', 'Lasso', 'Elastic Net']
        r2s = [model_comparison['ols']['test_r2'], model_comparison['lasso']['test_r2'],
               model_comparison['elasticnet']['test_r2']]
        bars = axc.bar(names, r2s, color=['#9E9E9E', '#5B9BD5', '#2E7D32'])
        for b, v in zip(bars, r2s):
            axc.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha='center', va='bottom')
        axc.set_ylabel('Test R²'); axc.set_title('Model Comparison: OLS vs Lasso vs Elastic Net'); axc.grid(True, axis='y')
        compare_plot = fig_to_base64(fig_cmp)

        # ---- coef_plot: Lasso vs Elastic Net coefficients ----
        order = np.argsort(np.abs(model.coef_))[::-1][:15]
        labs = [final_features[i] for i in order][::-1]
        en_c = [model.coef_[i] for i in order][::-1]
        la_c = [lasso.coef_[i] for i in order][::-1]
        fig_cf, axf = plt.subplots(figsize=(8, max(4, 0.4*len(labs)+1)))
        yv = np.arange(len(labs))
        axf.barh(yv-0.2, la_c, height=0.4, color='#5B9BD5', label='Lasso')
        axf.barh(yv+0.2, en_c, height=0.4, color='#2E7D32', label='Elastic Net')
        axf.set_yticks(yv); axf.set_yticklabels(labs); axf.axvline(0, color='k', lw=0.8)
        axf.set_xlabel('Standardized coefficient'); axf.set_title('Coefficients: Lasso vs Elastic Net'); axf.legend()
        coef_plot = fig_to_base64(fig_cf)

        response = {'results': results, 'plot': plot_image,
                    'compare_plot': compare_plot, 'coef_plot': coef_plot}
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
