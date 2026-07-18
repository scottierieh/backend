"""
K-Nearest Neighbors (KNN) Classification and Regression — CLI script
Instance-based learning with distance metrics. Ported from
scottierieh/backend's api/knn.py (FastAPI router) to the stdin/stdout CLI
contract used by src/backend/main.py's generic script runner.
"""

import sys
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from regression_diag import regression_sample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
import warnings


def _compute_multiclass_auc(y_true, y_pred_proba):
    """Macro-average ROC-AUC: binary uses the positive-class column; multiclass uses One-vs-Rest macro averaging."""
    try:
        n_classes = y_pred_proba.shape[1]
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'))
    except Exception:
        return None


warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

VALID_METRICS = {'euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine', 'hamming'}


def validate_inputs(metric: str, k_range: List[int], n_neighbors: int) -> List[int]:
    """Validate request parameters, raise ValueError on invalid input. Returns cleaned k_range."""
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}'. "
            f"Supported: {sorted(VALID_METRICS)}"
        )
    cleaned = [k for k in k_range if isinstance(k, int) and 1 <= k <= 500]
    if not cleaned:
        raise ValueError("k_range must contain at least one integer in [1, 500].")
    cleaned = sorted(set(cleaned))
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be ≥ 1.")
    return cleaned


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
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
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def detect_task_type(y: pd.Series) -> str:
    """Auto-detect classification vs regression"""
    unique_ratio = len(y.unique()) / len(y)
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    elif len(y.unique()) <= 10 or unique_ratio < 0.05:
        return 'classification'
    else:
        return 'regression'


def find_optimal_k(X_train, y_train, k_range: List[int], task_type: str,
                   weights: str, metric: str, p: int,
                   cv_folds: int = 5) -> Dict[str, Any]:
    """Find optimal K using cross-validation (respects cv_folds setting)."""
    k_scores = []
    # Cap k_range to avoid k > n_train
    max_k = len(X_train) - 1
    valid_k_range = [k for k in k_range if k <= max_k]
    if not valid_k_range:
        valid_k_range = [1]

    for k in valid_k_range:
        if task_type == 'classification':
            model = KNeighborsClassifier(n_neighbors=k, weights=weights,
                                          metric=metric, p=p, n_jobs=-1)
            cv = StratifiedKFold(n_splits=min(cv_folds, len(X_train) // k + 1),
                                 shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        else:
            model = KNeighborsRegressor(n_neighbors=k, weights=weights,
                                         metric=metric, p=p, n_jobs=-1)
            scores = cross_val_score(model, X_train, y_train,
                                     cv=min(cv_folds, len(X_train) // k + 1),
                                     scoring='r2')

        k_scores.append({
            'k': k,
            'mean_score': _to_native_type(np.mean(scores)),
            'std_score': _to_native_type(np.std(scores))
        })

    # Find best K
    best_idx = np.argmax([s['mean_score'] for s in k_scores])
    optimal_k = k_scores[best_idx]['k']

    return {
        'k_scores': k_scores,
        'optimal_k': optimal_k,
        'optimal_score': k_scores[best_idx]['mean_score']
    }


def train_knn_classifier(X_train, X_test, y_train, y_test, params: dict,
                          feature_names: List[str]) -> Dict[str, Any]:
    """Train KNN classifier"""
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(le.classes_)

    model = KNeighborsClassifier(
        n_neighbors=params['n_neighbors'],
        weights=params['weights'],
        metric=params['metric'],
        p=params['p'],
        algorithm=params['algorithm'],
        n_jobs=-1
    )

    model.fit(X_train, y_train_encoded)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_train_pred = model.predict(X_train)

    # Metrics
    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred)),
        'train_accuracy': _to_native_type(accuracy_score(y_train_encoded, y_train_pred)),
        'precision_macro': _to_native_type(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'recall_macro': _to_native_type(recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0))
    }

    # Per-class metrics
    class_report = classification_report(y_test_encoded, y_pred,
                                          target_names=[str(c) for c in le.classes_],
                                          output_dict=True, zero_division=0)
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

    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)

    # ROC curves
    roc_data = {}
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_data['binary'] = {
            'fpr': [_to_native_type(x) for x in fpr],
            'tpr': [_to_native_type(x) for x in tpr],
            'auc': _to_native_type(roc_auc)
        }
        metrics['auc'] = _to_native_type(roc_auc)
    else:
        for i, cls in enumerate(le.classes_):
            y_binary = (y_test_encoded == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[str(cls)] = {
                'fpr': [_to_native_type(x) for x in fpr],
                'tpr': [_to_native_type(x) for x in tpr],
                'auc': _to_native_type(roc_auc)
            }
        macro_auc = _compute_multiclass_auc(y_test_encoded, y_pred_proba)
        if macro_auc is not None:
            metrics['auc'] = macro_auc

    # Feature importance via permutation
    perm_importance = permutation_importance(model, X_test, y_test_encoded,
                                              n_repeats=10, random_state=42, n_jobs=-1)
    feature_importance = []
    for name, imp, std in zip(feature_names, perm_importance.importances_mean,
                               perm_importance.importances_std):
        feature_importance.append({
            'feature': name,
            'importance': _to_native_type(imp),
            'std': _to_native_type(std)
        })
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    _pos = [max(f['importance'], 0.0) for f in feature_importance]
    _total = sum(_pos) or 1.0
    for f, p in zip(feature_importance, _pos):
        f['importance_pct'] = _to_native_type(p / _total * 100)

    return {
        'model': model,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'feature_importance': feature_importance,
        'label_encoder': le,
        'y_test_encoded': y_test_encoded,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }


def train_knn_regressor(X_train, X_test, y_train, y_test, params: dict,
                         feature_names: List[str]) -> Dict[str, Any]:
    """Train KNN regressor"""
    model = KNeighborsRegressor(
        n_neighbors=params['n_neighbors'],
        weights=params['weights'],
        metric=params['metric'],
        p=params['p'],
        algorithm=params['algorithm'],
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        'mse': _to_native_type(mse),
        'rmse': _to_native_type(np.sqrt(mse)),
        'mae': _to_native_type(mean_absolute_error(y_test, y_pred)),
        'r2': _to_native_type(r2_score(y_test, y_pred)),
        'train_r2': _to_native_type(r2_score(y_train, y_train_pred))
    }

    # Feature importance
    perm_importance = permutation_importance(model, X_test, y_test,
                                              n_repeats=10, random_state=42, n_jobs=-1)
    feature_importance = []
    for name, imp, std in zip(feature_names, perm_importance.importances_mean,
                               perm_importance.importances_std):
        feature_importance.append({
            'feature': name,
            'importance': _to_native_type(imp),
            'std': _to_native_type(std)
        })
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    _pos = [max(f['importance'], 0.0) for f in feature_importance]
    _total = sum(_pos) or 1.0
    for f, p in zip(feature_importance, _pos):
        f['importance_pct'] = _to_native_type(p / _total * 100)

    return {
        'model': model,
        'metrics': metrics,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred,
        'feature_importance': feature_importance
    }


def perform_cross_validation(X, y, params: dict, task_type: str, cv_folds: int) -> Dict[str, Any]:
    """Perform cross-validation (StratifiedKFold for classification)."""
    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model = KNeighborsClassifier(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            metric=params['metric'],
            p=params['p'],
            n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')
    else:
        model = KNeighborsRegressor(
            n_neighbors=params['n_neighbors'],
            weights=params['weights'],
            metric=params['metric'],
            p=params['p'],
            n_jobs=-1
        )
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

    return {
        'cv_scores': [_to_native_type(s) for s in scores],
        'cv_mean': _to_native_type(np.mean(scores)),
        'cv_std': _to_native_type(np.std(scores)),
        'cv_folds': cv_folds,
        'cv_metric': 'accuracy' if task_type == 'classification' else 'r2'
    }


def generate_prediction_examples(result: Dict, task_type: str, n_examples: int = 15) -> List[Dict]:
    """Generate sample prediction examples (mirrors SVM implementation)."""
    examples = []
    try:
        if task_type == 'classification':
            y_test_enc = result.get('y_test_encoded')
            y_pred = result.get('y_pred')
            y_pred_proba = result.get('y_pred_proba')
            class_labels = result.get('class_labels', [])
            if y_test_enc is None or y_pred is None:
                return []
            indices = list(range(len(y_test_enc)))
            np.random.seed(42)
            chosen = np.random.choice(indices, size=min(n_examples, len(indices)), replace=False)
            for i in chosen:
                conf = float(np.max(y_pred_proba[i])) if y_pred_proba is not None else None
                examples.append({
                    'actual': class_labels[int(y_test_enc[i])] if class_labels else int(y_test_enc[i]),
                    'predicted': class_labels[int(y_pred[i])] if class_labels else int(y_pred[i]),
                    'correct': bool(y_test_enc[i] == y_pred[i]),
                    'confidence': round(conf, 4) if conf is not None else None
                })
        else:
            y_test = result.get('y_test')
            y_pred = result.get('y_pred')
            if y_test is None or y_pred is None:
                return []
            y_test_arr = np.array(y_test)
            indices = list(range(len(y_test_arr)))
            np.random.seed(42)
            chosen = np.random.choice(indices, size=min(n_examples, len(indices)), replace=False)
            for i in chosen:
                actual = float(y_test_arr[i])
                predicted = float(y_pred[i])
                error = predicted - actual
                error_pct = (error / actual * 100) if actual != 0 else None
                examples.append({
                    'actual': round(actual, 4),
                    'predicted': round(predicted, 4),
                    'error': round(error, 4),
                    'error_pct': round(error_pct, 2) if error_pct is not None else None
                })
    except Exception:
        pass
    return examples


def generate_k_selection_plot(k_search_result: Dict) -> str:
    """Generate K selection plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    k_scores = k_search_result['k_scores']
    ks = [s['k'] for s in k_scores]
    means = [s['mean_score'] for s in k_scores]
    stds = [s['std_score'] for s in k_scores]

    ax.plot(ks, means, 'b-o', linewidth=2, markersize=8, label='CV Score')
    ax.fill_between(ks,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='blue')

    # Mark optimal K
    optimal_k = k_search_result['optimal_k']
    optimal_score = k_search_result['optimal_score']
    ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
    ax.scatter([optimal_k], [optimal_score], color='red', s=150, zorder=5, edgecolors='black')

    ax.set_xlabel('K (Number of Neighbors)', fontsize=11)
    ax.set_ylabel('Cross-Validation Score', fontsize=11)
    ax.set_title('K Selection via Cross-Validation', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xticks(ks)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_decision_boundary_plot(model, X, y, feature_names: List[str], le=None) -> str:
    """Generate 2D decision boundary plot"""
    if X.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    X_2d = X[:, :2]

    # Create mesh
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Train a 2D model for visualization
    if le is not None:
        y_encoded = le.transform(y) if hasattr(y, 'values') else y
        model_2d = KNeighborsClassifier(n_neighbors=model.n_neighbors,
                                         weights=model.weights)
        model_2d.fit(X_2d, y_encoded)
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        model_2d = KNeighborsRegressor(n_neighbors=model.n_neighbors,
                                        weights=model.weights)
        model_2d.fit(X_2d, y)
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        y_encoded = y

    Z = Z.reshape(xx.shape)

    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap='coolwarm',
                         edgecolors='black', s=50, alpha=0.7)

    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title(
        f'KNN Decision Boundary (K={model.n_neighbors}) — "{feature_names[0]}" vs "{feature_names[1]}"',
        fontsize=13, fontweight='bold'
    )
    if len(feature_names) > 2:
        ax.set_xlabel(
            f'{feature_names[0]}  (2-feature projection — other features set to 0)',
            fontsize=10
        )

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_confusion_matrix_plot(cm: List[List[int]], class_labels: List[str]) -> str:
    """Generate confusion matrix heatmap (pure matplotlib, no seaborn)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_array = np.array(cm)

    im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_labels)

    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, format(cm_array[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm_array[i, j] > thresh else 'black',
                    fontsize=12)

    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_roc_plot(roc_data: Dict) -> str:
    """Generate ROC curve plot"""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))

    for (label, data), color in zip(roc_data.items(), colors):
        ax.plot(data['fpr'], data['tpr'], color=color, linewidth=2,
                label=f'{label} (AUC = {data["auc"]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_feature_importance_plot(importance_data: List[Dict], top_n: int = 15) -> str:
    """Generate feature importance plot"""
    fig, ax = plt.subplots(figsize=(10, max(5, len(importance_data[:top_n]) * 0.4)))

    top_features = importance_data[:top_n]
    features = [d['feature'] for d in top_features][::-1]
    importances = [d['importance'] for d in top_features][::-1]
    stds = [d.get('std', 0) for d in top_features][::-1]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importances, xerr=stds, color=colors,
                   edgecolor='black', alpha=0.8, capsize=3)

    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Permutation Importance', fontsize=11)
    ax.set_title('KNN Feature Importance (Permutation)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_regression_plot(y_test, y_pred) -> str:
    """Generate actual vs predicted plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, color='#3b82f6', s=30)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = axes[1]
    residuals = np.array(y_test) - np.array(y_pred)
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#22c55e', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, task_type: str, params: dict,
                            k_search_result: Optional[Dict] = None,
                            n_features: int = 0) -> Dict[str, Any]:
    """Generate interpretation of KNN results."""
    key_insights = []

    # ── 1. Performance (분류/회귀 기준 분리) ────────────────────────
    if task_type == 'classification':
        accuracy = result['metrics']['accuracy']
        train_accuracy = result['metrics'].get('train_accuracy', None)

        if accuracy >= 0.90:
            status = 'positive'
            perf_desc = 'Excellent classification performance'
        elif accuracy >= 0.75:
            status = 'neutral'
            perf_desc = 'Good classification performance'
        elif accuracy >= 0.60:
            status = 'warning'
            perf_desc = 'Moderate performance — consider tuning K or distance metric'
        else:
            status = 'warning'
            perf_desc = 'Low accuracy — model may need significant tuning'

        train_str = f', Train: {train_accuracy:.1%}' if train_accuracy is not None else ''
        key_insights.append({
            'title': 'Classification Performance',
            'description': (
                f'{perf_desc}. Test Accuracy: {accuracy:.1%}{train_str}, '
                f'F1-macro: {result["metrics"]["f1_macro"]:.3f}'
            ),
            'status': status
        })
    else:
        r2 = result['metrics']['r2']
        train_r2 = result['metrics'].get('train_r2', None)
        rmse = result['metrics']['rmse']

        if r2 >= 0.75:
            status = 'positive'
            perf_desc = 'Strong predictive fit'
        elif r2 >= 0.50:
            status = 'neutral'
            perf_desc = 'Moderate fit'
        elif r2 >= 0.25:
            status = 'warning'
            perf_desc = 'Weak fit — consider feature engineering or different metric'
        else:
            status = 'warning'
            perf_desc = 'Poor fit — KNN may not suit this data distribution'

        train_str = f', Train R²: {train_r2:.3f}' if train_r2 is not None else ''
        key_insights.append({
            'title': 'Regression Performance',
            'description': (
                f'{perf_desc}. Test R²: {r2:.3f}{train_str}, '
                f'RMSE: {rmse:.4f}, MAE: {result["metrics"]["mae"]:.4f}'
            ),
            'status': status
        })

    # ── 2. K value ──────────────────────────────────────────────────
    k = params['n_neighbors']
    if k_search_result:
        key_insights.append({
            'title': f'Optimal K = {k}',
            'description': f'Selected via cross-validation. CV Score: {k_search_result["optimal_score"]:.3f}',
            'status': 'positive'
        })
    else:
        if k == 1:
            k_desc = 'K=1 may overfit — high variance, sensitive to noise'
        elif k > 20:
            k_desc = 'Large K increases bias — boundary may be too smooth'
        else:
            k_desc = 'Reasonable K value for most datasets'
        key_insights.append({
            'title': f'K = {k} Neighbors',
            'description': k_desc,
            'status': 'neutral'
        })

    # ── 3. High-dimensionality warning ──────────────────────────────
    if n_features > 20:
        key_insights.append({
            'title': 'High Dimensionality Warning',
            'description': (
                f'{n_features} features detected. KNN performance degrades in high dimensions '
                '(curse of dimensionality) — distances become less meaningful. '
                'Consider feature selection or PCA before applying KNN.'
            ),
            'status': 'warning'
        })

    # ── 4. Feature importance note ──────────────────────────────────
    key_insights.append({
        'title': 'Feature Importance (Approximate)',
        'description': (
            'Importance is estimated via permutation — shuffling each feature and measuring '
            'accuracy drop. KNN distance-based importance can be unstable with correlated '
            'or high-dimensional features. Treat as approximate guidance.'
        ),
        'status': 'neutral'
    })

    # ── 5. Distance metric ──────────────────────────────────────────
    metric_desc = {
        'euclidean': 'Euclidean (L2) — standard straight-line distance',
        'manhattan': 'Manhattan (L1) — sum of absolute differences; robust to outliers',
        'minkowski': f'Minkowski (p={params["p"]}) — generalizes Euclidean/Manhattan',
        'chebyshev': 'Chebyshev (L∞) — maximum coordinate difference',
        'cosine': 'Cosine similarity — angle-based; good for text/sparse data',
        'hamming': 'Hamming — fraction of differing positions; for categorical data'
    }
    key_insights.append({
        'title': f'Metric: {params["metric"]}',
        'description': metric_desc.get(params['metric'], params['metric']),
        'status': 'neutral'
    })

    # ── 6. Algorithm hint for high dimensions ───────────────────────
    algo = params.get('algorithm', 'auto')
    if n_features > 15 and algo != 'brute':
        algo_note = (
            f'algorithm="{algo}" may fall back to brute force for high-dimensional data. '
            'Explicitly setting algorithm="brute" can be faster when n_features > 15.'
        )
        key_insights.append({
            'title': 'Algorithm Note',
            'description': algo_note,
            'status': 'neutral'
        })

    # ── 7. Recommendation ──────────────────────────────────────────
    if task_type == 'classification':
        rec = (
            'KNN classifier trained. For improvement: tune K via cross-validation, '
            'try distance-weighted voting (weights="distance"), '
            'and ensure features are scaled. '
            'For high-dimensional data consider feature selection first.'
        )
    else:
        rec = (
            'KNN regressor trained. For improvement: tune K, try weights="distance" '
            'to reduce influence of distant neighbors, and verify feature scaling. '
            'KNN regression can be sensitive to outliers in the target variable.'
        )

    return {
        'key_insights': key_insights,
        'recommendation': rec
    }


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target_col = payload.get('target_col') or payload.get('target')
        feature_cols = payload.get('feature_cols') or payload.get('features')
        task_type = payload.get('task_type', 'auto')
        test_size = float(payload.get('test_size', 0.2))

        n_neighbors_req = int(payload.get('n_neighbors', 5))
        weights = payload.get('weights', 'uniform')
        metric = payload.get('metric', 'minkowski')
        p = int(payload.get('p', 2))
        algorithm = payload.get('algorithm', 'auto')
        random_state = int(payload.get('random_state', 42))
        cv_folds = int(payload.get('cv_folds', 5))
        scale_features = bool(payload.get('scale_features', True))
        find_optimal_k_flag = bool(payload.get('find_optimal_k', True))
        k_range = payload.get('k_range', [1, 3, 5, 7, 9, 11, 13, 15])

        if not data:
            raise ValueError("Data not provided.")
        if not target_col or not feature_cols:
            raise ValueError("Missing data, features, or target")

        df = pd.DataFrame(data)

        # Validate columns
        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")

        # ── Input validation ────────────────────────────────────────
        k_range = validate_inputs(metric, k_range, n_neighbors_req)

        # ── Prepare features ────────────────────────────────────────
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Categorical → one-hot (safer than LabelEncoder for distance-based KNN)
        cat_cols = [c for c in X.columns if X[c].dtype == 'object']
        num_cols = [c for c in X.columns if X[c].dtype != 'object']
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)

        # Drop NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        feature_cols = list(X.columns)  # update after one-hot expansion

        if len(X) < 30:
            raise ValueError("At least 30 valid samples required.")

        # ── Auto-detect task type ───────────────────────────────────
        if task_type == 'auto':
            task_type = detect_task_type(y)

        # ── Scale features ──────────────────────────────────────────
        scaler = None
        X_array = X.values
        if scale_features:
            scaler = StandardScaler()
            X_array = scaler.fit_transform(X_array)

        # ── Split data ──────────────────────────────────────────────
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y, test_size=test_size, random_state=random_state,
                stratify=y if task_type == 'classification' else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y, test_size=test_size, random_state=random_state
            )

        # ── Clamp n_neighbors to valid range ────────────────────────
        max_k = len(X_train) - 1
        n_neighbors = min(max(1, n_neighbors_req), max_k)

        # ── Find optimal K (pass cv_folds) ──────────────────────────
        k_search_result = None
        if find_optimal_k_flag:
            k_search_result = find_optimal_k(
                X_train,
                y_train if task_type == 'regression' else LabelEncoder().fit_transform(y_train),
                k_range, task_type, weights, metric, p,
                cv_folds=cv_folds   # ← was hardcoded to 5
            )
            n_neighbors = k_search_result['optimal_k']

        # ── Parameters ──────────────────────────────────────────────
        params = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'metric': metric,
            'p': p,
            'algorithm': algorithm
        }

        # ── Train model ─────────────────────────────────────────────
        if task_type == 'classification':
            result = train_knn_classifier(X_train, X_test, y_train, y_test, params, feature_cols)
        else:
            result = train_knn_regressor(X_train, X_test, y_train, y_test, params, feature_cols)

        model = result['model']

        # ── Cross-validation ────────────────────────────────────────
        cv_result = perform_cross_validation(X_array, y, params, task_type, cv_folds)

        # ── Prediction examples ─────────────────────────────────────
        prediction_examples = generate_prediction_examples(result, task_type, n_examples=15)

        # ── Visualizations ──────────────────────────────────────────
        importance_plot = generate_feature_importance_plot(result['feature_importance'])
        k_plot = generate_k_selection_plot(k_search_result) if k_search_result else None

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            decision_plot = generate_decision_boundary_plot(
                model, X_array, y, feature_cols, result['label_encoder']
            )
            regression_plot = None
        else:
            cm_plot = None
            roc_plot = None
            decision_plot = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])

        # ── Interpretation ──────────────────────────────────────────
        interpretation = generate_interpretation(
            result, task_type, params, k_search_result,
            n_features=len(feature_cols)   # ← for high-dim warning
        )

        # ── Response ────────────────────────────────────────────────
        try:
            from guardrails import compute_guardrails
            guardrails = compute_guardrails(X, y, feature_cols, task_type, result['metrics'])
        except Exception:
            guardrails = []

        response = {
            'guardrails': guardrails,
            'task_type': task_type,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'parameters': {
                'n_neighbors': params['n_neighbors'],
                'weights': params['weights'],
                'metric': params['metric'],
                'p': params['p'],
                'scaled': scale_features
            },
            'metrics': result['metrics'],
            'feature_importance': result['feature_importance'],
            'cv_results': cv_result,
            'k_search_result': k_search_result,
            'importance_plot': importance_plot,
            'k_plot': k_plot,
            'interpretation': interpretation,
            'prediction_examples': prediction_examples
        }

        if task_type == 'classification':
            response['per_class_metrics'] = result['per_class_metrics']
            response['confusion_matrix'] = result['confusion_matrix']
            response['class_labels'] = result['class_labels']
            response['cm_plot'] = cm_plot
            response['roc_plot'] = roc_plot
            response['decision_plot'] = decision_plot
        else:
            response['regression_plot'] = regression_plot
            _rd = regression_sample(result['y_test'], result['y_pred'])
            if _rd:
                response.update(_rd)

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
