
import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

        results['feature_importance'] = dict(zip(feature_names, model.feature_importances_))
        results['prediction_examples'] = prediction_examples

        # --- Plotting ---
        plot_image = None
        if problem_type == 'regression':
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle('GBM Regression Analysis', fontsize=20, fontweight='bold')
            residuals = y_test - y_pred

            # 1. Actual vs Predicted
            sns.scatterplot(x=y_test, y=y_pred, ax=axes[0, 0], alpha=0.6)
            axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title(f"Actual vs Predicted (R² = {results['metrics']['r2_score']:.3f})")
            axes[0,0].grid(True, alpha=0.3)

            # 2. Feature Importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            sns.barplot(x='importance', y='feature', data=importance_df.head(10), ax=axes[0, 1], palette='viridis')
            axes[0, 1].set_title('Top 10 Feature Importance')
            axes[0,1].grid(True, alpha=0.3)

            # 3. Residuals vs Predicted
            sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 2], alpha=0.6)
            axes[0, 2].axhline(y=0, color='r', linestyle='--')
            axes[0, 2].set_xlabel('Predicted Values')
            axes[0, 2].set_ylabel('Residuals')
            axes[0, 2].set_title('Residuals vs. Predicted')
            axes[0,2].grid(True, alpha=0.3)

            # 4. Residual Distribution
            sns.histplot(residuals, kde=True, ax=axes[1, 0], bins=15)
            axes[1, 0].set_title('Residuals Distribution')
            axes[1,0].grid(True, alpha=0.3)

            # 5. Q-Q Plot
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot of Residuals')
            axes[1,1].grid(True, alpha=0.3)
            
            # 6. Learning Curve
            train_scores = np.zeros(n_estimators)
            for i, y_pred_train in enumerate(model.staged_predict(X_train)):
                train_scores[i] = mean_squared_error(y_train, y_pred_train)
            
            test_scores = np.zeros(n_estimators)
            for i, y_pred_test in enumerate(model.staged_predict(X_test)):
                test_scores[i] = mean_squared_error(y_test, y_pred_test)

            axes[1, 2].plot(train_scores, 'b-', label='Train MSE')
            axes[1, 2].plot(test_scores, 'r-', label='Test MSE')
            axes[1, 2].set_xlabel('Boosting Iterations')
            axes[1, 2].set_ylabel('Mean Squared Error')
            axes[1, 2].set_title('Learning Curve')
            axes[1, 2].legend()
            axes[1,2].grid(True, alpha=0.3)

            # 7. Prediction Error Distribution
            errors = np.abs(residuals)
            sns.histplot(errors, kde=False, ax=axes[2, 0], bins=15)
            axes[2, 0].set_title(f'Prediction Error Distribution (MAE={errors.mean():.2f})')
            axes[2,0].grid(True, alpha=0.3)
            
            # 8. Top Feature vs Target
            top_feature = importance_df.iloc[0]['feature']
            sns.scatterplot(x=X_test[top_feature], y=y_test, ax=axes[2, 1], alpha=0.6, label='Actual')
            sns.scatterplot(x=X_test[top_feature], y=y_pred, ax=axes[2, 1], alpha=0.6, label='Predicted')
            axes[2, 1].set_title(f'Top Feature ({top_feature}) vs Target')
            axes[2,1].legend()
            axes[2,1].grid(True, alpha=0.3)

            # 9. Summary Text
            axes[2, 2].axis('off')
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
            axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, fontsize=12,
                            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3))

        else: # Classification
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            sns.barplot(x='importance', y='feature', data=importance_df, ax=axes[0], palette='viridis')
            axes[0].set_title('Feature Importance')

            cm = confusion_matrix(y_test, y_pred)
            class_names = sorted(y.unique())
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            axes[1].set_title('Confusion Matrix')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')
        
        response = {
            'results': results,
            'plot': f"data:image/png;base64,{plot_image}"
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
