
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def _generate_interpretation(model_name, search_method, baseline_acc, tuned_acc, best_params,
                              best_cv_score, cv_folds, n_candidates, std_at_best):
    parts = []
    improvement = tuned_acc - baseline_acc
    parts.append("**Overall Assessment**")
    parts.append(
        f"→ {search_method.title()} search evaluated {n_candidates} hyperparameter candidate(s) for "
        f"'{model_name}' using {cv_folds}-fold cross-validation."
    )
    parts.append(
        f"→ Tuned test accuracy = {tuned_acc*100:.1f}% vs. baseline (default hyperparameters) = "
        f"{baseline_acc*100:.1f}% (Δ = {improvement*100:+.1f}pp)."
    )
    if improvement < 0.005:
        parts.append(
            "→ Tuning produced little to no improvement over the defaults. Either the default "
            "hyperparameters were already near-optimal, or the search grid/space needs to be widened."
        )
    elif improvement > 0.05:
        parts.append("→ Tuning produced a substantial improvement — the default hyperparameters were poorly suited to this dataset.")

    parts.append("")
    parts.append("**Statistical Insights**")
    parts.append(f"→ Best cross-validated score: {best_cv_score:.4f} (std across folds: {std_at_best:.4f})")
    if std_at_best > 0.05:
        parts.append(
            "→ The standard deviation across CV folds is relatively high, meaning performance is "
            "sensitive to which data ends up in each fold — treat the reported score as an estimate, "
            "not a guarantee."
        )
    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    parts.append(f"→ Best hyperparameters found: {param_str}")

    parts.append("")
    parts.append("**Recommendations**")
    if search_method == 'grid':
        parts.append("→ Grid search is exhaustive but scales poorly with more hyperparameters — consider random search or Bayesian optimization (e.g. optuna) for larger search spaces.")
    else:
        parts.append("→ Random search sampled a subset of the space; increasing n_iter (or switching to Bayesian optimization) may find further improvements.")
    parts.append("→ Refit the chosen hyperparameters on the full training set and validate on a held-out or newly collected sample before deploying.")

    return "\n".join(parts)

MODEL_REGISTRY = {
    'logistic_regression': {
        'estimator': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'param_grid': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    },
    'decision_tree': {
        'estimator': lambda: DecisionTreeClassifier(random_state=42),
        'param_grid': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'random_forest': {
        'estimator': lambda: RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'gradient_boosting': {
        'estimator': lambda: GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4]
        }
    },
    'svm': {
        'estimator': lambda: SVC(probability=True, random_state=42),
        'param_grid': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    },
    'knn': {
        'estimator': lambda: KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    },
}

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        features = payload.get('features')
        target = payload.get('target')

        model_name = payload.get('model', 'random_forest')
        search_method = payload.get('search_method', 'grid')  # 'grid' or 'random'
        param_grid_override = payload.get('param_grid')
        cv_folds = int(payload.get('cv_folds', 5))
        scoring = payload.get('scoring', 'accuracy')
        n_iter = int(payload.get('n_iter', 20))

        if not all([data, features, target]):
            raise ValueError("Missing data, features, or target")

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {model_name}")

        df = pd.DataFrame(data)
        X = df[features]
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

        registry_entry = MODEL_REGISTRY[model_name]
        estimator = registry_entry['estimator']()
        param_grid = param_grid_override if param_grid_override else registry_entry['param_grid']

        baseline_model = registry_entry['estimator']()
        baseline_model.fit(X_train, y_train)
        baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))

        if search_method == 'random':
            search = RandomizedSearchCV(
                estimator, param_distributions=param_grid, n_iter=n_iter,
                cv=cv_folds, scoring=scoring, random_state=42, n_jobs=-1, refit=True
            )
        else:
            search = GridSearchCV(
                estimator, param_grid=param_grid,
                cv=cv_folds, scoring=scoring, n_jobs=-1, refit=True
            )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        tuned_accuracy = accuracy_score(y_test, y_pred)

        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_summary = cv_results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']] \
            .sort_values('rank_test_score').head(20).to_dict('records')

        results = {
            'model': model_name,
            'search_method': search_method,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'baseline_test_accuracy': baseline_accuracy,
            'tuned_test_accuracy': tuned_accuracy,
            'improvement': tuned_accuracy - baseline_accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'cv_results_top20': cv_results_summary,
            'n_candidates_evaluated': len(cv_results_df),
            'class_names': [str(c) for c in best_model.classes_] if hasattr(best_model, 'classes_') else []
        }
        std_at_best = float(cv_results_df.loc[cv_results_df['rank_test_score'] == 1, 'std_test_score'].iloc[0])
        results['interpretation'] = _generate_interpretation(
            model_name, search_method, baseline_accuracy, tuned_accuracy, search.best_params_,
            search.best_score_, cv_folds, len(cv_results_df), std_at_best
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Hyperparameter Tuning ({search_method.title()} Search) — {model_name}', fontsize=16)

        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix (Tuned Model)')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_ylabel('True Label')

        axes[0, 1].bar(['Baseline', 'Tuned'], [baseline_accuracy, tuned_accuracy], color=['steelblue', 'darkorange'])
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('Baseline vs. Tuned Model Accuracy')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate([baseline_accuracy, tuned_accuracy]):
            axes[0, 1].text(i, v + 0.01, f"{v:.3f}", ha='center')
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
            f"Model: {model_name}\n"
            f"Search Method: {search_method}\n"
            f"CV Folds: {cv_folds}\n"
            f"Candidates Evaluated: {len(cv_results_df)}\n\n"
            f"Best Params:\n" +
            "\n".join(f"  {k}: {v}" for k, v in search.best_params_.items()) +
            f"\n\nBest CV Score: {search.best_score_:.4f}\n"
            f"Test Accuracy (Tuned): {tuned_accuracy:.4f}"
        )
        axes[1, 1].text(0.02, 0.98, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_image = fig_to_base64(fig)

        response = {
            'results': results,
            'plot': plot_image
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
