"""
Artificial Neural Network (Multi-Layer Perceptron) — CLI script
Standalone MLP classifier/regressor with loss-curve tracking. Ported from
scottierieh/backend's api/mlp_analysis.py (FastAPI router) to the
stdin/stdout CLI contract used by src/backend/main.py's generic script runner.
"""

import sys
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def detect_task_type(y: pd.Series) -> str:
    unique_ratio = len(y.unique()) / len(y)
    if not pd.api.types.is_numeric_dtype(y) or y.dtype.name == 'category':
        return 'classification'
    elif len(y.unique()) <= 10 or unique_ratio < 0.05:
        return 'classification'
    else:
        return 'regression'


def _common_params(params: dict) -> dict:
    return dict(
        hidden_layer_sizes=tuple(params['hidden_layer_sizes']),
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        early_stopping=params['early_stopping'] and params['solver'] in ('sgd', 'adam'),
        random_state=params['random_state']
    )


def train_mlp_classifier(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    n_classes = len(le.classes_)

    model = MLPClassifier(**_common_params(params))
    model.fit(X_train, y_train_encoded)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred)),
        'train_accuracy': _to_native_type(accuracy_score(y_train_encoded, model.predict(X_train))),
        'precision_macro': _to_native_type(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'recall_macro': _to_native_type(recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
    }

    class_report = classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True, zero_division=0)
    per_class_metrics = []
    for cls in le.classes_:
        cls_str = str(cls)
        if cls_str in class_report:
            per_class_metrics.append({
                'class': cls_str,
                'precision': _to_native_type(class_report[cls_str]['precision']),
                'recall': _to_native_type(class_report[cls_str]['recall']),
                'f1_score': _to_native_type(class_report[cls_str]['f1-score']),
                'support': int(class_report[cls_str]['support'])
            })

    cm = confusion_matrix(y_test_encoded, y_pred)

    roc_data = {}
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_data['binary'] = {'fpr': [_to_native_type(x) for x in fpr], 'tpr': [_to_native_type(x) for x in tpr], 'auc': _to_native_type(roc_auc)}
        metrics['auc'] = _to_native_type(roc_auc)
    else:
        for i, cls in enumerate(le.classes_):
            y_binary = (y_test_encoded == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[str(cls)] = {'fpr': [_to_native_type(x) for x in fpr], 'tpr': [_to_native_type(x) for x in tpr], 'auc': _to_native_type(roc_auc)}

    return {
        'model': model, 'metrics': metrics, 'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(), 'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data, 'label_encoder': le,
        'y_test_encoded': y_test_encoded, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba
    }


def train_mlp_regressor(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    model = MLPRegressor(**_common_params(params))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        'mse': _to_native_type(mse),
        'rmse': _to_native_type(np.sqrt(mse)),
        'mae': _to_native_type(mean_absolute_error(y_test, y_pred)),
        'r2': _to_native_type(r2_score(y_test, y_pred)),
        'train_r2': _to_native_type(r2_score(y_train, y_train_pred)),
    }

    return {'model': model, 'metrics': metrics, 'y_test': y_test.values if hasattr(y_test, 'values') else y_test, 'y_pred': y_pred}


def compute_permutation_importance(model, X_test, y_test, feature_names: List[str],
                                    n_repeats: int = 10, random_state: int = 42) -> List[Dict[str, Any]]:
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
        result = []
        for name, mean, std in zip(feature_names, perm.importances_mean, perm.importances_std):
            result.append({'feature': name, 'importance_mean': _to_native_type(mean), 'importance_std': _to_native_type(std)})
        result.sort(key=lambda x: x['importance_mean'], reverse=True)
        for i, item in enumerate(result):
            item['rank'] = i + 1
        return result
    except Exception:
        return []


def compute_shap(model, X_test: np.ndarray, feature_names: List[str], task_type: str,
                  max_background: int = 50, max_samples: int = 100) -> Dict:
    """Model-agnostic SHAP (Permutation explainer) — MLP has no tree structure for TreeExplainer."""
    try:
        try:
            import shap as _shap
        except ImportError:
            return {'shap_importance': [], 'shap_plot': None, 'error': 'shap package not installed. Run: pip install shap'}

        X_arr = np.asarray(X_test)
        n = len(X_arr)
        rng = np.random.RandomState(42)
        background = X_arr[rng.choice(n, size=min(max_background, n), replace=False)]
        X_sample = X_arr[rng.choice(n, size=min(max_samples, n), replace=False)]

        predict_fn = model.predict_proba if task_type == 'classification' else model.predict
        explainer = _shap.Explainer(predict_fn, _shap.maskers.Independent(background))
        shap_values = explainer(X_sample)

        sv = np.array(shap_values.values)
        mean_shap = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)

        shap_importance = [
            {'feature': name, 'mean_abs_shap': _to_native_type(val)}
            for name, val in sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)
        ]

        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
        feats = [d['feature'] for d in shap_importance][::-1]
        vals = [d['mean_abs_shap'] for d in shap_importance][::-1]
        ax.barh(feats, vals, color='#f59e0b', edgecolor='none')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, axis='x')
        fig.subplots_adjust(left=0.20)
        shap_plot = _fig_to_base64(fig)

        return {'shap_importance': shap_importance, 'shap_plot': shap_plot, 'error': None}
    except Exception as e:
        return {'shap_importance': [], 'shap_plot': None, 'error': str(e)}


def perform_cross_validation(X, y, params: dict, task_type: str, cv_folds: int) -> Dict[str, Any]:
    cv_params = _common_params(params)
    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model = MLPClassifier(**cv_params)
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=params['random_state'])
        scores = cross_val_score(model, X, y_encoded, cv=cv_splitter, scoring='accuracy')
    else:
        model = MLPRegressor(**cv_params)
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=params['random_state'])
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='r2')

    return {
        'cv_scores': [_to_native_type(s) for s in scores],
        'cv_mean': _to_native_type(np.mean(scores)),
        'cv_std': _to_native_type(np.std(scores)),
        'cv_folds': cv_folds
    }


def generate_loss_curve_plot(model) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_curve_, color='#3b82f6', linewidth=2, label='Training Loss')
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, color='#e53e3e', linewidth=2, label='Validation Score')
        ax2.set_ylabel('Validation Score', fontsize=11, color='#e53e3e')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_confusion_matrix_plot(cm: List[List[int]], class_labels: List[str]) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_array = np.array(cm)
    im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_labels)
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, str(cm_array[i, j]), ha='center', va='center',
                    color='white' if cm_array[i, j] > thresh else 'black', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_roc_plot(roc_data: Dict) -> Optional[str]:
    if not roc_data:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    for (label, data), color in zip(roc_data.items(), colors):
        ax.plot(data['fpr'], data['tpr'], color=color, linewidth=2, label=f'{label} (AUC = {data["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11); ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_feature_importance_plot(importance_data: List[Dict], top_n: int = 15) -> str:
    fig, ax = plt.subplots(figsize=(10, max(5, len(importance_data[:top_n]) * 0.4)))
    top_features = importance_data[:top_n]
    features = [d['feature'] for d in top_features][::-1]
    importances = [d['importance_mean'] for d in top_features][::-1]
    stds = [d.get('importance_std', 0) for d in top_features][::-1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, importances, xerr=stds, color=colors, edgecolor='black', alpha=0.8, capsize=3)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Permutation Importance', fontsize=11)
    ax.set_title('MLP Feature Importance (Permutation)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_regression_plot(y_test, y_pred) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, color='#3b82f6', s=30)
    min_val = min(min(y_test), min(y_pred)); max_val = max(max(y_test), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual', fontsize=11); ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = axes[1]
    residuals = np.array(y_test) - np.array(y_pred)
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#22c55e', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=11); ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, task_type: str, model, params: dict, n_obs: int, n_features: int) -> Dict[str, Any]:
    key_insights = []

    n_weights = sum(params['hidden_layer_sizes']) * n_features
    key_insights.append({
        'title': 'Network Architecture',
        'description': f"{n_features} inputs → {' → '.join(str(h) for h in params['hidden_layer_sizes'])} → output, "
                        f"activation='{params['activation']}', solver='{params['solver']}'.",
        'status': 'neutral'
    })

    n_iter = model.n_iter_
    if n_iter >= params['max_iter']:
        key_insights.append({
            'title': 'Training Did Not Converge',
            'description': f'Training used the full budget of {params["max_iter"]} iterations without early stopping. Consider increasing max_iter.',
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': 'Training Converged',
            'description': f'Training stopped after {n_iter} of {params["max_iter"]} iterations (final loss = {model.loss_:.5f}).',
            'status': 'positive'
        })

    if task_type == 'classification':
        accuracy = result['metrics']['accuracy']
        train_accuracy = result['metrics'].get('train_accuracy')
        status, perf_desc = ('positive', 'Excellent classification performance') if accuracy >= 0.9 else \
                             ('neutral', 'Good classification performance') if accuracy >= 0.75 else \
                             ('warning', 'Moderate performance — consider tuning architecture or regularization')
        train_str = f', Train: {train_accuracy:.1%}' if train_accuracy is not None else ''
        key_insights.append({
            'title': 'Classification Performance',
            'description': f'{perf_desc}. Test Accuracy: {accuracy:.1%}{train_str}',
            'status': status
        })
        if train_accuracy is not None and (train_accuracy - accuracy) > 0.15:
            key_insights.append({
                'title': 'Overfitting Warning',
                'description': f'Train accuracy exceeds test accuracy by {(train_accuracy - accuracy):.1%}. Increase alpha (L2 penalty) or reduce network size.',
                'status': 'warning'
            })
    else:
        r2 = result['metrics']['r2']
        status, perf_desc = ('positive', 'Excellent fit') if r2 >= 0.8 else ('neutral', 'Moderate fit') if r2 >= 0.5 else ('warning', 'Weak fit')
        key_insights.append({'title': 'Regression Performance', 'description': f'{perf_desc}. R² = {r2:.3f}, RMSE = {result["metrics"]["rmse"]:.4f}', 'status': status})

    if n_obs < 10 * max(n_weights, 1):
        key_insights.append({
            'title': 'Small Sample Relative to Network Size',
            'description': f'With {n_obs} observations and this network size, overfitting risk is elevated. Consider a smaller architecture or more data.',
            'status': 'warning'
        })

    return {
        'key_insights': key_insights,
        'recommendation': (
            'MLP trained successfully. Neural networks are sensitive to feature scaling (already applied) '
            'and random initialization — for tabular data with limited samples, tree ensembles often match '
            'or beat MLPs with less tuning effort.'
        )
    }


def generate_prediction_examples(result: Dict, task_type: str, n_examples: int = 15, random_state: int = 42) -> List[Dict[str, Any]]:
    """Sample prediction examples from the test set (regression: actual/predicted/error; classification: actual/predicted/correct/confidence)."""
    try:
        rng = np.random.RandomState(random_state)
        if task_type == 'regression':
            y_test = np.array(result['y_test'])
            y_pred = np.array(result['y_pred'])
            n = min(n_examples, len(y_test))
            idx = rng.choice(len(y_test), size=n, replace=False)
            examples = []
            for i in idx:
                actual = _to_native_type(y_test[i])
                predicted = _to_native_type(y_pred[i])
                error = _to_native_type(y_pred[i] - y_test[i])
                error_pct = _to_native_type(abs(error) / abs(actual) * 100) if actual not in (0, None) else None
                examples.append({'actual': actual, 'predicted': predicted, 'error': error, 'error_pct': error_pct})
            return examples
        else:
            le = result['label_encoder']
            y_test_enc = np.array(result['y_test_encoded'])
            y_pred_enc = np.array(result['y_pred'])
            y_pred_proba = np.array(result['y_pred_proba'])
            n = min(n_examples, len(y_test_enc))
            idx = rng.choice(len(y_test_enc), size=n, replace=False)
            examples = []
            for i in idx:
                actual_label = str(le.inverse_transform([y_test_enc[i]])[0])
                predicted_label = str(le.inverse_transform([y_pred_enc[i]])[0])
                confidence = _to_native_type(float(y_pred_proba[i].max()))
                examples.append({
                    'actual': actual_label, 'predicted': predicted_label,
                    'correct': bool(y_test_enc[i] == y_pred_enc[i]), 'confidence': confidence
                })
            return examples
    except Exception:
        return []


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target_col = payload.get('target_col') or payload.get('target')
        feature_cols = payload.get('feature_cols') or payload.get('features')
        task_type = payload.get('task_type', 'auto')
        test_size = float(payload.get('test_size', 0.2))

        hidden_layer_sizes = payload.get('hidden_layer_sizes', [100])
        activation = payload.get('activation', 'relu')
        solver = payload.get('solver', 'adam')
        alpha = float(payload.get('alpha', 0.0001))
        learning_rate_init = float(payload.get('learning_rate_init', 0.001))
        max_iter = int(payload.get('max_iter', 500))
        early_stopping = bool(payload.get('early_stopping', True))
        random_state = int(payload.get('random_state', 42))
        cv_folds = int(payload.get('cv_folds', 5))

        if not data:
            raise ValueError("Data not provided.")
        if not target_col or not feature_cols:
            raise ValueError("Missing data, features, or target")

        df = pd.DataFrame(data)

        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        feature_cols = list(X.columns)

        if len(X) < 30:
            raise ValueError("At least 30 valid samples required.")

        if not hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must be a non-empty list.")

        if task_type == 'auto':
            task_type = detect_task_type(y)

        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
            'max_iter': max_iter,
            'early_stopping': early_stopping,
            'random_state': random_state
        }

        scaler = StandardScaler()
        X_array = scaler.fit_transform(X.values)

        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y, test_size=test_size, random_state=random_state,
            stratify=y if task_type == 'classification' else None
        )

        if task_type == 'classification':
            result = train_mlp_classifier(X_train, X_test, y_train, y_test, params)
        else:
            result = train_mlp_regressor(X_train, X_test, y_train, y_test, params)

        model = result['model']

        y_test_for_perm = result['label_encoder'].transform(y_test) if task_type == 'classification' else (y_test.values if hasattr(y_test, 'values') else y_test)
        perm_importance = compute_permutation_importance(model, X_test, y_test_for_perm, feature_cols)
        shap_result = compute_shap(model, X_test, feature_cols, task_type)

        cv_result = perform_cross_validation(X_array, y, params, task_type, cv_folds)

        loss_plot = generate_loss_curve_plot(model)
        importance_plot = generate_feature_importance_plot(perm_importance)

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data'])
            regression_plot = None
        else:
            cm_plot = None
            roc_plot = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])

        interpretation = generate_interpretation(result, task_type, model, params, len(X), len(feature_cols))
        prediction_examples = generate_prediction_examples(result, task_type)

        response = {
            'task_type': task_type,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'parameters': params,
            'n_iterations': int(model.n_iter_),
            'final_loss': _to_native_type(model.loss_),
            'metrics': result['metrics'],
            'perm_importance': perm_importance,
            'shap_importance': shap_result.get('shap_importance'),
            'shap_plot': shap_result.get('shap_plot'),
            'shap_error': shap_result.get('error'),
            'cv_results': cv_result,
            'loss_plot': loss_plot,
            'importance_plot': importance_plot,
            'interpretation': interpretation,
            'prediction_examples': prediction_examples
        }

        if task_type == 'classification':
            response['per_class_metrics'] = result['per_class_metrics']
            response['confusion_matrix'] = result['confusion_matrix']
            response['class_labels'] = result['class_labels']
            response['cm_plot'] = cm_plot
            response['roc_plot'] = roc_plot
        else:
            response['regression_plot'] = regression_plot

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
