
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve, auc
)
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

def _generate_interpretation(problem_type, results, evals_result, n_estimators, best_iteration, n_obs):
    parts = []
    parts.append("**Overall Assessment**")
    parts.append(
        f"→ LightGBM was trained with up to {n_estimators} boosting rounds on {n_obs} observations."
    )

    overfit_gap = None
    if evals_result:
        metric_key = list(evals_result['train'].keys())[0]
        train_curve = evals_result['train'][metric_key]
        valid_curve = evals_result['valid'][metric_key]
        lower_is_better = metric_key in ('rmse', 'binary_logloss', 'multi_logloss', 'l2', 'l1')
        if lower_is_better:
            best_idx = int(np.argmin(valid_curve))
        else:
            best_idx = int(np.argmax(valid_curve))
        overfit_gap = valid_curve[-1] - valid_curve[best_idx] if lower_is_better else valid_curve[best_idx] - valid_curve[-1]
        parts.append(
            f"→ Best validation {metric_key} reached at iteration {best_idx + 1} of {len(valid_curve)}."
        )
        if best_idx < len(valid_curve) * 0.6:
            parts.append(
                "→ Validation performance peaked well before the final iteration, suggesting the "
                "model may be overfitting in later rounds. Consider early stopping or fewer estimators."
            )

    if problem_type == 'regression':
        r2 = results['metrics']['r2_score']
        rmse = results['metrics']['rmse']
        if r2 >= 0.8:
            fit_desc = "excellent"
        elif r2 >= 0.6:
            fit_desc = "good"
        elif r2 >= 0.4:
            fit_desc = "moderate"
        else:
            fit_desc = "weak"
        parts.append(f"→ Test R² = {r2:.3f} ({fit_desc} fit) | RMSE = {rmse:.3f}")
    else:
        accuracy = results['metrics']['accuracy']
        report = results['metrics']['classification_report']
        macro_f1 = report.get('macro avg', {}).get('f1-score')
        acc_desc = "excellent" if accuracy >= 0.9 else "good" if accuracy >= 0.75 else "fair" if accuracy >= 0.6 else "poor"
        parts.append(f"→ Test accuracy = {accuracy*100:.1f}% ({acc_desc})" + (f" | Macro F1 = {macro_f1:.3f}" if macro_f1 is not None else ""))
        if 'roc_auc_data' in results:
            auc_val = results['roc_auc_data']['auc']
            auc_desc = "excellent" if auc_val >= 0.9 else "good" if auc_val >= 0.8 else "fair" if auc_val >= 0.7 else "poor"
            parts.append(f"→ ROC AUC = {auc_val:.3f} ({auc_desc})")

    top_features = sorted(results['feature_importance'].items(), key=lambda kv: kv[1], reverse=True)[:3]
    parts.append("")
    parts.append("**Statistical Insights**")
    if top_features:
        feat_str = ", ".join(f"{name} ({imp:.0f})" for name, imp in top_features)
        parts.append(f"→ Most influential features by split count: {feat_str}")

    parts.append("")
    parts.append("**Recommendations**")
    if overfit_gap is not None and overfit_gap > 0:
        parts.append("→ Consider enabling early stopping (monitor a validation set) to stop training at the best iteration automatically.")
    parts.append("→ Tune num_leaves and max_depth together — LightGBM's leaf-wise growth can overfit quickly on small datasets if num_leaves is too large relative to max_depth.")
    parts.append("→ Cross-validate the final hyperparameters before deploying, since a single train/test split can be optimistic or pessimistic by chance.")

    return "\n".join(parts)

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        features = payload.get('features')
        target = payload.get('target')
        problem_type = payload.get('problemType')  # 'regression' or 'classification'

        n_estimators = int(payload.get('nEstimators', 100))
        learning_rate = float(payload.get('learningRate', 0.1))
        max_depth = int(payload.get('maxDepth', -1))
        num_leaves = int(payload.get('numLeaves', 31))

        if not all([data, features, target, problem_type]):
            raise ValueError("Missing data, features, target, or problemType")

        df = pd.DataFrame(data)

        X = df[features]
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)
        feature_names = X.columns.tolist()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        common_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            verbosity=-1
        )

        if problem_type == 'regression':
            model = lgb.LGBMRegressor(**common_params)
        else:
            model = lgb.LGBMClassifier(**common_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results = {}
        if problem_type == 'regression':
            results['metrics'] = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        else:
            y_prob = model.predict_proba(X_test)
            results['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            if len(np.unique(y)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=model.classes_[1])
                results['roc_auc_data'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': auc(fpr, tpr)
                }

        results['feature_importance'] = dict(zip(feature_names, model.feature_importances_.tolist()))
        results['best_iteration'] = int(getattr(model, 'best_iteration_', n_estimators) or n_estimators)

        evals_result = {}
        n_obs = len(X_train) + len(X_test)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        eval_metric = 'rmse' if problem_type == 'regression' else ('binary_logloss' if len(np.unique(y)) == 2 else 'multi_logloss')
        eval_model_cls = lgb.LGBMRegressor if problem_type == 'regression' else lgb.LGBMClassifier
        eval_model = eval_model_cls(**common_params)
        eval_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=['train', 'valid'],
            eval_metric=eval_metric,
            callbacks=[lgb.record_evaluation(evals_result)]
        )

        results['interpretation'] = _generate_interpretation(
            problem_type, results, evals_result, n_estimators, results['best_iteration'], n_obs
        )

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('LightGBM Analysis', fontsize=18, fontweight='bold')

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        sns.barplot(x='importance', y='feature', data=importance_df, ax=axes[0], palette='viridis')
        axes[0].set_title('Top 15 Feature Importances (Split Count)')

        if evals_result:
            metric_key = list(evals_result['train'].keys())[0]
            axes[1].plot(evals_result['train'][metric_key], label='Train')
            axes[1].plot(evals_result['valid'][metric_key], label='Validation')
            axes[1].set_xlabel('Boosting Iterations')
            axes[1].set_ylabel(metric_key)
            axes[1].set_title('Learning Curve')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        if problem_type == 'regression':
            axes[2].scatter(y_test, y_pred, alpha=0.6)
            axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[2].set_xlabel('Actual Values')
            axes[2].set_ylabel('Predicted Values')
            axes[2].set_title(f"Actual vs Predicted (R² = {results['metrics']['r2_score']:.3f})")
            axes[2].grid(True, alpha=0.3)
        else:
            cm = np.array(results['metrics']['confusion_matrix'])
            class_names = sorted(y.unique())
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], xticklabels=class_names, yticklabels=class_names)
            axes[2].set_xlabel('Predicted')
            axes[2].set_ylabel('Actual')
            axes[2].set_title('Confusion Matrix')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
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
