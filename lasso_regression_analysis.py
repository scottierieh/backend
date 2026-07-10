
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target = payload.get('target')
        features = payload.get('features')
        alpha = float(payload.get('alpha', 1.0))
        test_size = float(payload.get('test_size', 0.2))

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
        
        results = {
            'metrics': { 'test': test_metrics, 'train': train_metrics },
            'coefficients': dict(zip(final_features, model.coef_)),
            'intercept': model.intercept_,
            'alpha': alpha,
            'interpretation': interpretation,
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
        
        response = {
            'results': results,
            'plot': plot_image,
            'path_plot': path_plot_image
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
