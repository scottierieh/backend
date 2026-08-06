
import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from scipy import stats
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

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        features = payload.get('features')
        target = payload.get('target')
        problem_type = payload.get('problemType') # 'regression' or 'classification'
        
        # Hyperparameters
        n_estimators = int(payload.get('nEstimators', 100))
        learning_rate = float(payload.get('learningRate', 0.1))
        max_depth = int(payload.get('maxDepth', 3))

        if not all([data, features, target, problem_type]):
            raise ValueError("Missing data, features, target, or problemType")

        df = pd.DataFrame(data)

        # --- Data Preparation ---
        X = df[features]
        y = df[target]

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)
        feature_names = X.columns.tolist()

        try:
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None
            )
        except ValueError:
            # Fallback for small classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # --- Model Training ---
        if problem_type == 'regression':
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                validation_fraction=0.1,
                n_iter_no_change=5, 
                tol=0.01
            )
        else: # classification
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Evaluation ---
        results = {}
        prediction_examples = []
        if problem_type == 'regression':
            results['metrics'] = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            residuals = y_test - y_pred
            errors = np.abs(residuals)
            
            n_examples = min(10, len(y_test))
            example_indices = np.random.choice(y_test.index, n_examples, replace=False)
            
            for idx in example_indices:
                actual = y_test.loc[idx]
                predicted = model.predict(X_test.loc[[idx]])[0]
                error = abs(actual - predicted)
                error_pct = (error / actual) * 100 if actual != 0 else 0
                prediction_examples.append({
                    "actual": actual,
                    "predicted": predicted,
                    "error": error,
                    "error_percent": error_pct
                })

        else:
            results['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            n_examples = min(10, len(y_test))
            example_indices = np.random.choice(y_test.index, n_examples, replace=False)

            for idx in example_indices:
                actual = y_test.loc[idx]
                predicted = model.predict(X_test.loc[[idx]])[0]
                proba = model.predict_proba(X_test.loc[[idx]])[0]
                prediction_examples.append({
                    "actual": actual,
                    "predicted": predicted,
                    "status": "✅" if actual == predicted else "❌",
                    "confidence": max(proba)
                })

        # Array of {feature, importance, importance_pct}, sorted descending -- matches
        # every other model's contract. Was a plain {feature: value} dict before, which
        # crashed the frontend (`results.feature_importance.slice(...)` expects an array).
        _imp = np.asarray(model.feature_importances_, dtype=float)
        _total = _imp.sum() or 1.0
        feature_importance = [
            {'feature': name, 'importance': _to_native_type(imp), 'importance_pct': _to_native_type(imp / _total * 100)}
            for name, imp in zip(feature_names, _imp)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        results['feature_importance'] = feature_importance
        results['prediction_examples'] = prediction_examples

        # --- Plotting ---
        def _fig_to_data_url(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        plots = []
        if problem_type == 'regression':
            residuals = y_test - y_pred

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # 1. Actual vs Predicted
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f"Actual vs Predicted (R² = {results['metrics']['r2_score']:.3f})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Actual vs Predicted', 'image': _fig_to_data_url(fig)})

            # 2. Feature Importance
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.barplot(x='importance', y='feature', data=importance_df.head(10), ax=ax, palette='viridis')
            ax.set_title('Top 10 Feature Importance')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Feature Importance', 'image': _fig_to_data_url(fig)})

            # 3. Residuals vs Predicted
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs. Predicted')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Residuals vs Predicted', 'image': _fig_to_data_url(fig)})

            # 4. Residual Distribution
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.histplot(residuals, kde=True, ax=ax, bins=15)
            ax.set_title('Residuals Distribution')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Residuals Distribution', 'image': _fig_to_data_url(fig)})

            # 5. Q-Q Plot
            fig, ax = plt.subplots(figsize=(7, 6))
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot of Residuals')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Q-Q Plot of Residuals', 'image': _fig_to_data_url(fig)})

            # 6. Learning Curve
            train_scores = np.zeros(n_estimators)
            for i, y_pred_train in enumerate(model.staged_predict(X_train)):
                train_scores[i] = mean_squared_error(y_train, y_pred_train)

            test_scores = np.zeros(n_estimators)
            for i, y_pred_test in enumerate(model.staged_predict(X_test)):
                test_scores[i] = mean_squared_error(y_test, y_pred_test)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(train_scores, 'b-', label='Train MSE')
            ax.plot(test_scores, 'r-', label='Test MSE')
            ax.set_xlabel('Boosting Iterations')
            ax.set_ylabel('Mean Squared Error')
            ax.set_title('Learning Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Learning Curve', 'image': _fig_to_data_url(fig)})

            # 7. Prediction Error Distribution
            errors = np.abs(residuals)
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.histplot(errors, kde=False, ax=ax, bins=15)
            ax.set_title(f'Prediction Error Distribution (MAE={errors.mean():.2f})')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Prediction Error Distribution', 'image': _fig_to_data_url(fig)})

            # 8. Top Feature vs Target
            top_feature = importance_df.iloc[0]['feature']
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.scatterplot(x=X_test[top_feature], y=y_test, ax=ax, alpha=0.6, label='Actual')
            sns.scatterplot(x=X_test[top_feature], y=y_pred, ax=ax, alpha=0.6, label='Predicted')
            ax.set_title(f'Top Feature ({top_feature}) vs Target')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Top Feature vs Target', 'image': _fig_to_data_url(fig)})

            # 9. Summary Text
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.axis('off')
            summary_text = (
                f"Model Performance:\n"
                f"  R² Score: {results['metrics']['r2_score']:.4f}\n"
                f"  MSE: {results['metrics']['mse']:,.2f}\n"
                f"  RMSE: {results['metrics']['rmse']:,.2f}\n"
                f"  MAE: {errors.mean():,.2f}\n\n"
                f"Residuals Summary:\n"
                f"  Mean: {residuals.mean():.2f}\n"
                f"  Std Dev: {residuals.std():.2f}\n"
                f"  Min: {residuals.min():,.2f}\n"
                f"  Max: {residuals.max():,.2f}"
            )
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3))
            plots.append({'label': 'Model Performance Summary', 'image': _fig_to_data_url(fig)})

        else: # Classification
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            # 1. Feature Importance
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='importance', y='feature', data=importance_df, ax=ax, palette='viridis')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            plots.append({'label': 'Feature Importance', 'image': _fig_to_data_url(fig)})

            # 2. Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            class_names = sorted(y.unique())
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            plots.append({'label': 'Confusion Matrix', 'image': _fig_to_data_url(fig)})

            # 3. Precision-Recall Curve
            y_proba = model.predict_proba(X_test)
            classes_ = model.classes_
            n_classes = len(classes_)

            fig, ax = plt.subplots(figsize=(7, 6))
            if n_classes <= 2:
                pos_label = classes_[1] if n_classes == 2 else classes_[0]
                pos_idx = 1 if n_classes == 2 else 0
                precision, recall, _ = precision_recall_curve(y_test, y_proba[:, pos_idx], pos_label=pos_label)
                ap = average_precision_score((y_test == pos_label).astype(int), y_proba[:, pos_idx])
                base_rate = (y_test == pos_label).mean()
                ax.plot(recall, precision, color='C0', lw=2)
                ax.axhline(y=base_rate, color='r', linestyle='--', lw=1, label=f'Base rate = {base_rate:.3f}')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision-Recall Curve (AP = {ap:.3f})')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            else:
                y_test_bin = label_binarize(y_test, classes=classes_)
                for i, cls in enumerate(classes_):
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                    ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
                    ax.plot(recall, precision, lw=1.5, label=f'{cls} (AP={ap:.3f})')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve (One-vs-Rest)')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.legend(loc='best', fontsize='small')
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append({'label': 'Precision-Recall Curve', 'image': _fig_to_data_url(fig)})

        try:
            from guardrails import compute_guardrails
            _norm_metrics = {'accuracy': results['metrics'].get('accuracy'), 'r2': results['metrics'].get('r2_score')}
            guardrails = compute_guardrails(X, y, features, problem_type, _norm_metrics)
        except Exception:
            guardrails = []

        response = {
            'results': results,
            'guardrails': guardrails,
            'plots': plots
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
