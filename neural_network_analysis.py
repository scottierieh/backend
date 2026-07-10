
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
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

def _generate_interpretation(problem_type, results, hidden_layer_sizes, activation, solver,
                              n_iter, max_iter, n_obs, n_features):
    parts = []
    parts.append("**Overall Assessment**")
    parts.append(
        f"→ MLP architecture: {n_features} inputs → {' → '.join(str(h) for h in hidden_layer_sizes)} "
        f"→ output, activation='{activation}', solver='{solver}'."
    )
    parts.append(f"→ Training ran for {n_iter} of {max_iter} allowed iterations (final loss = {results['loss']:.5f}).")
    if n_iter >= max_iter:
        parts.append(
            "→ Training used the full iteration budget without early stopping — the loss curve may "
            "not have fully converged. Consider increasing max_iter."
        )

    parts.append("")
    parts.append("**Statistical Insights**")
    if problem_type == 'regression':
        r2 = results['metrics']['r2_score']
        fit_desc = "excellent" if r2 >= 0.8 else "good" if r2 >= 0.6 else "moderate" if r2 >= 0.4 else "weak"
        parts.append(f"→ Test R² = {r2:.3f} ({fit_desc} fit) | RMSE = {results['metrics']['rmse']:.3f}")
    else:
        accuracy = results['metrics']['accuracy']
        nir = results['metrics'].get('nir')
        acc_desc = "excellent" if accuracy >= 0.9 else "good" if accuracy >= 0.75 else "fair" if accuracy >= 0.6 else "poor"
        lift_str = f" (lift = {(accuracy - nir)*100:+.1f}pp over NIR)" if nir is not None else ""
        parts.append(f"→ Test accuracy = {accuracy*100:.1f}% ({acc_desc}){lift_str}")
        if 'roc_auc_data' in results:
            parts.append(f"→ ROC AUC = {results['roc_auc_data']['auc']:.3f}")

    n_weights = sum(hidden_layer_sizes) * n_features if hidden_layer_sizes else 0
    if n_obs < 10 * max(n_weights, 1) and n_weights > 0:
        parts.append(
            f"→ With {n_obs} observations and a network of this size, the model may be prone to "
            "overfitting; a smaller architecture or stronger L2 regularization (alpha) may help."
        )

    parts.append("")
    parts.append("**Recommendations**")
    parts.append("→ Compare the loss curve's trajectory: a plateau early on suggests the learning_rate or architecture may need adjustment.")
    parts.append("→ Neural networks are sensitive to feature scaling (already applied here) and random initialization — run multiple seeds to check stability.")
    if problem_type != 'regression':
        parts.append("→ For tabular data with limited samples, tree ensembles (Random Forest, LightGBM) often match or beat MLPs with less tuning.")

    return "\n".join(parts)

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        features = payload.get('features')
        target = payload.get('target')
        problem_type = payload.get('problemType', 'classification')  # 'regression' or 'classification'

        hidden_layer_sizes = payload.get('hiddenLayerSizes', [100])
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        activation = payload.get('activation', 'relu')
        solver = payload.get('solver', 'adam')
        alpha = float(payload.get('alpha', 0.0001))
        learning_rate_init = float(payload.get('learningRateInit', 0.001))
        max_iter = int(payload.get('maxIter', 500))

        if not all([data, features, target]):
            raise ValueError("Missing data, features, or target")

        df = pd.DataFrame(data)
        X = df[features]
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)
        feature_names = X.columns.tolist()

        if problem_type == 'regression':
            y = pd.to_numeric(y, errors='coerce')
            combined = pd.concat([X, y], axis=1).dropna()
            X = combined[feature_names]
            y = combined[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        stratify = y if problem_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=stratify)

        common_params = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            random_state=42
        )

        if problem_type == 'regression':
            model = MLPRegressor(**common_params)
        else:
            model = MLPClassifier(**common_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results = {
            'hidden_layer_sizes': list(hidden_layer_sizes),
            'n_layers': model.n_layers_,
            'n_iterations': model.n_iter_,
            'loss': model.loss_,
            'loss_curve': model.loss_curve_,
        }

        if problem_type == 'regression':
            results['metrics'] = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        else:
            y_prob = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results['metrics'] = {
                'accuracy': accuracy,
                'nir': y_test.value_counts(normalize=True).max(),
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
            results['class_names'] = [str(c) for c in model.classes_]

        results['interpretation'] = _generate_interpretation(
            problem_type, results, hidden_layer_sizes, activation, solver,
            model.n_iter_, max_iter, len(X_train) + len(X_test), len(feature_names)
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Artificial Neural Network (MLP) Analysis', fontsize=16)

        axes[0, 0].plot(model.loss_curve_)
        axes[0, 0].set_xlabel('Iterations')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Curve')
        axes[0, 0].grid(True, alpha=0.3)

        if problem_type == 'regression':
            axes[0, 1].scatter(y_test, y_pred, alpha=0.6)
            axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual Values')
            axes[0, 1].set_ylabel('Predicted Values')
            axes[0, 1].set_title(f"Actual vs Predicted (R² = {results['metrics']['r2_score']:.3f})")
            axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
        else:
            cm = np.array(results['metrics']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], xticklabels=model.classes_, yticklabels=model.classes_)
            axes[0, 1].set_title('Confusion Matrix')
            axes[0, 1].set_xlabel('Predicted Label')
            axes[0, 1].set_ylabel('True Label')

            if 'roc_auc_data' in results:
                roc = results['roc_auc_data']
                axes[1, 0].plot(roc['fpr'], roc['tpr'], color='darkorange', lw=2, label=f"ROC curve (area = {roc['auc']:.2f})")
                axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('ROC Curve')
                axes[1, 0].legend(loc="lower right")
            else:
                axes[1, 0].set_visible(False)

            axes[1, 1].axis('off')
            summary_text = (
                f"Architecture: {feature_names.__len__()} inputs -> {hidden_layer_sizes} -> {len(model.classes_)} outputs\n"
                f"Activation: {activation}\n"
                f"Solver: {solver}\n"
                f"Iterations run: {model.n_iter_}\n"
                f"Final Loss: {model.loss_:.5f}\n"
                f"Accuracy: {results['metrics']['accuracy']:.4f}"
            )
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=12,
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
