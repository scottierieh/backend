"""
Hyperparameter Tuning — CLI script
Grid/Random search over a chosen model's hyperparameters, for classification
or regression. Standalone analysis (not tied to Model Lab's model registry —
see docs/model-lab-plugin-architecture.md §6a for that separate, deferred
piece). Started from src/backend's existing classification-only script and
extended with regression support + the target_col/feature_cols naming
convention used across the rest of the app's analysis pages.
"""

import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error,
)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

warnings.filterwarnings('ignore')


def _to_native_type(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    return obj


def fig_to_base64(fig):
    """Returns a raw base64 PNG string (no data-URI prefix) -- the frontend
    prepends `data:image/png;base64,` itself, matching every other ported page."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _generate_interpretation(model_name, task_type, search_method, baseline_score, tuned_score, best_params,
                              best_cv_score, cv_folds, n_candidates, std_at_best, scoring):
    improvement = tuned_score - baseline_score
    score_label = 'accuracy' if task_type == 'classification' else scoring

    key_insights = []
    key_insights.append({
        'title': 'Tuning Result',
        'description': (
            f"{search_method.title()} search evaluated {n_candidates} hyperparameter candidate(s) for "
            f"'{model_name}' using {cv_folds}-fold cross-validation. Tuned test {score_label} = "
            f"{tuned_score:.4f} vs. baseline (default hyperparameters) = {baseline_score:.4f} "
            f"(Δ = {improvement:+.4f})."
        ),
        'status': 'positive' if improvement > 0.05 else 'warning' if abs(improvement) < 0.005 else 'neutral',
    })
    if abs(improvement) < 0.005:
        key_insights.append({
            'title': 'Little Improvement Over Defaults',
            'description': 'Either the default hyperparameters were already near-optimal, or the search grid/space needs to be widened.',
            'status': 'warning',
        })
    elif improvement > 0.05:
        key_insights.append({
            'title': 'Substantial Improvement',
            'description': 'The default hyperparameters were poorly suited to this dataset — tuning found a meaningfully better configuration.',
            'status': 'positive',
        })

    key_insights.append({
        'title': 'Cross-Validation Stability',
        'description': f'Best cross-validated score: {best_cv_score:.4f} (std across folds: {std_at_best:.4f}).' + (
            ' This is relatively high variance — performance is sensitive to which data ends up in each fold, so treat the reported score as an estimate, not a guarantee.'
            if std_at_best > 0.05 else ''
        ),
        'status': 'warning' if std_at_best > 0.05 else 'neutral',
    })

    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    key_insights.append({
        'title': 'Best Hyperparameters',
        'description': f'Found: {param_str}',
        'status': 'neutral',
    })

    if search_method == 'grid':
        recommendation = (
            "Grid search is exhaustive but scales poorly with more hyperparameters — consider random search "
            "for larger search spaces. Refit the chosen hyperparameters on the full training set and validate "
            "on a held-out or newly collected sample before deploying."
        )
    else:
        recommendation = (
            "Random search sampled a subset of the space; increasing n_iter may find further improvements. "
            "Refit the chosen hyperparameters on the full training set and validate on a held-out or newly "
            "collected sample before deploying."
        )

    return {'key_insights': key_insights, 'recommendation': recommendation}


MODEL_REGISTRY = {
    'classification': {
        'logistic_regression': {
            'estimator': lambda: LogisticRegression(max_iter=1000, random_state=42),
            'param_grid': {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']}
        },
        'decision_tree': {
            'estimator': lambda: DecisionTreeClassifier(random_state=42),
            'param_grid': {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        },
        'random_forest': {
            'estimator': lambda: RandomForestClassifier(random_state=42),
            'param_grid': {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10]}
        },
        'gradient_boosting': {
            'estimator': lambda: GradientBoostingClassifier(random_state=42),
            'param_grid': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [2, 3, 4]}
        },
        'svm': {
            'estimator': lambda: SVC(probability=True, random_state=42),
            'param_grid': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
        },
        'knn': {
            'estimator': lambda: KNeighborsClassifier(),
            'param_grid': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
        },
        'mlp': {
            'estimator': lambda: MLPClassifier(max_iter=1000, random_state=42),
            'param_grid': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]}
        },
    },
    'regression': {
        'ridge': {
            'estimator': lambda: Ridge(random_state=42),
            'param_grid': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        },
        'lasso': {
            'estimator': lambda: Lasso(random_state=42, max_iter=5000),
            'param_grid': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        },
        'decision_tree': {
            'estimator': lambda: DecisionTreeRegressor(random_state=42),
            'param_grid': {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        },
        'random_forest': {
            'estimator': lambda: RandomForestRegressor(random_state=42),
            'param_grid': {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10]}
        },
        'gradient_boosting': {
            'estimator': lambda: GradientBoostingRegressor(random_state=42),
            'param_grid': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [2, 3, 4]}
        },
        'svm': {
            'estimator': lambda: SVR(),
            'param_grid': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
        },
        'knn': {
            'estimator': lambda: KNeighborsRegressor(),
            'param_grid': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
        },
        'mlp': {
            'estimator': lambda: MLPRegressor(max_iter=1000, random_state=42),
            'param_grid': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]}
        },
    },
}


def _detect_task_type(y: pd.Series) -> str:
    unique_ratio = len(y.unique()) / len(y)
    if not pd.api.types.is_numeric_dtype(y) or y.dtype.name == 'category':
        return 'classification'
    elif len(y.unique()) <= 10 or unique_ratio < 0.05:
        return 'classification'
    else:
        return 'regression'


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target = payload.get('target_col') or payload.get('target')
        features = payload.get('feature_cols') or payload.get('features')
        task_type_req = payload.get('task_type', 'auto')

        model_name = payload.get('model_type', payload.get('model', 'random_forest'))
        search_method = payload.get('search_method', 'grid')  # 'grid' or 'random'
        param_grid_override = payload.get('param_grid')
        cv_folds = int(payload.get('cv_folds', 5))
        n_iter = int(payload.get('n_iter', 20))
        test_size = float(payload.get('test_size', 0.3))
        random_state = int(payload.get('random_state', 42))

        if not all([data, features, target]):
            raise ValueError("Missing data, features, or target")

        df = pd.DataFrame(data)
        df = df.dropna(subset=[target])

        if task_type_req in ('classification', 'regression'):
            task_type = task_type_req
        else:
            task_type = _detect_task_type(df[target])

        if model_name not in MODEL_REGISTRY[task_type]:
            raise ValueError(f"Unsupported model '{model_name}' for task_type '{task_type}'")

        scoring = payload.get('scoring') or ('accuracy' if task_type == 'classification' else 'r2')

        X_raw = df[features].copy()
        y = pd.to_numeric(df[target], errors='coerce') if task_type == 'regression' else df[target]
        valid_mask = y.notna()
        X_raw = X_raw[valid_mask]
        y = y[valid_mask].reset_index(drop=True)

        X = pd.get_dummies(X_raw, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state,
                stratify=y if task_type == 'classification' else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )

        registry_entry = MODEL_REGISTRY[task_type][model_name]
        estimator = registry_entry['estimator']()
        param_grid = param_grid_override if param_grid_override else registry_entry['param_grid']

        baseline_model = registry_entry['estimator']()
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_score = accuracy_score(y_test, baseline_pred) if task_type == 'classification' else r2_score(y_test, baseline_pred)

        if search_method == 'random':
            search = RandomizedSearchCV(
                estimator, param_distributions=param_grid, n_iter=n_iter,
                cv=cv_folds, scoring=scoring, random_state=random_state, n_jobs=-1, refit=True
            )
        else:
            search = GridSearchCV(
                estimator, param_grid=param_grid,
                cv=cv_folds, scoring=scoring, n_jobs=-1, refit=True
            )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        tuned_score = accuracy_score(y_test, y_pred) if task_type == 'classification' else r2_score(y_test, y_pred)

        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_summary = cv_results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']] \
            .sort_values('rank_test_score').head(20).to_dict('records')

        response = {
            'model': model_name,
            'task_type': task_type,
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'search_method': search_method,
            'scoring': scoring,
            'cv_folds': cv_folds,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'baseline_test_score': baseline_score,
            'tuned_test_score': tuned_score,
            'improvement': tuned_score - baseline_score,
            'cv_results_top20': cv_results_summary,
            'n_candidates_evaluated': len(cv_results_df),
        }

        if task_type == 'classification':
            response['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            response['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            response['class_names'] = [str(c) for c in best_model.classes_] if hasattr(best_model, 'classes_') else []
        else:
            response['mse'] = _to_native_type(mean_squared_error(y_test, y_pred))
            response['rmse'] = _to_native_type(np.sqrt(mean_squared_error(y_test, y_pred)))
            response['mae'] = _to_native_type(mean_absolute_error(y_test, y_pred))

        std_at_best = float(cv_results_df.loc[cv_results_df['rank_test_score'] == 1, 'std_test_score'].iloc[0])
        response['interpretation'] = _generate_interpretation(
            model_name, task_type, search_method, baseline_score, tuned_score, search.best_params_,
            search.best_score_, cv_folds, len(cv_results_df), std_at_best, scoring
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Hyperparameter Tuning ({search_method.title()} Search) — {model_name} ({task_type})', fontsize=16)

        if task_type == 'classification':
            cm = np.array(response['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix (Tuned Model)')
            axes[0, 0].set_xlabel('Predicted Label')
            axes[0, 0].set_ylabel('True Label')
        else:
            axes[0, 0].scatter(y_test, y_pred, alpha=0.5, color='#3b82f6')
            lo, hi = float(min(min(y_test), min(y_pred))), float(max(max(y_test), max(y_pred)))
            axes[0, 0].plot([lo, hi], [lo, hi], 'r--', linewidth=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted (Tuned Model)')

        score_label = 'Accuracy' if task_type == 'classification' else scoring.upper()
        axes[0, 1].bar(['Baseline', 'Tuned'], [baseline_score, tuned_score], color=['steelblue', 'darkorange'])
        axes[0, 1].set_ylabel(f'Test {score_label}')
        axes[0, 1].set_title(f'Baseline vs. Tuned Model {score_label}')
        for i, v in enumerate([baseline_score, tuned_score]):
            axes[0, 1].text(i, v + 0.01 * (1 if v >= 0 else -1), f"{v:.3f}", ha='center')
        axes[0, 1].grid(True, axis='y', alpha=0.3)

        top_scores = cv_results_df.sort_values('rank_test_score').head(15)
        axes[1, 0].barh(range(len(top_scores)), top_scores['mean_test_score'], xerr=top_scores['std_test_score'], color='seagreen')
        axes[1, 0].set_yticks(range(len(top_scores)))
        axes[1, 0].set_yticklabels([f"#{r}" for r in top_scores['rank_test_score']])
        axes[1, 0].set_xlabel(f'Mean CV Score ({scoring})')
        axes[1, 0].set_ylabel('Candidate Rank')
        axes[1, 0].set_title('Top Candidate CV Scores (with std)')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, axis='x', alpha=0.3)

        axes[1, 1].axis('off')
        summary_text = (
            f"Model: {model_name} ({task_type})\n"
            f"Search Method: {search_method}\n"
            f"CV Folds: {cv_folds}\n"
            f"Candidates Evaluated: {len(cv_results_df)}\n\n"
            f"Best Params:\n" +
            "\n".join(f"  {k}: {v}" for k, v in search.best_params_.items()) +
            f"\n\nBest CV Score: {search.best_score_:.4f}\n"
            f"Test {score_label} (Tuned): {tuned_score:.4f}"
        )
        axes[1, 1].text(0.02, 0.98, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        response['tuning_plot'] = fig_to_base64(fig)

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
