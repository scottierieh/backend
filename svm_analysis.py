"""
Support Vector Machine (SVM) Classification and Regression — CLI script
Linear and non-linear classification/regression with kernel methods. Ported from
scottierieh/backend's api/svm.py (FastAPI router) to the stdin/stdout CLI
contract used by src/backend/main.py's generic script runner.
"""

import sys
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from cv_strategy import run_cv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
import warnings
from model_diagnostics import bootstrap_ci, calibration_curve, pr_curve, error_examples


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


def train_svm_classifier(X_train, X_test, y_train, y_test, params: dict,
                          feature_names: List[str]) -> Dict[str, Any]:
    """Train SVM classifier"""
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(le.classes_)

    # Parse gamma
    gamma = params['gamma']
    if gamma not in ['scale', 'auto']:
        try:
            gamma = float(gamma)
        except:
            gamma = 'scale'

    model = SVC(
        kernel=params['kernel'],
        C=params['C'],
        gamma=gamma,
        degree=params['degree'],
        coef0=params['coef0'],
        random_state=params['random_state'],
        probability=True
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
        'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'n_support_vectors': int(sum(model.n_support_))
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

    # Support vectors per class
    support_per_class = []
    for i, (cls, n_sv) in enumerate(zip(le.classes_, model.n_support_)):
        support_per_class.append({
            'class': str(cls),
            'n_support_vectors': int(n_sv)
        })

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
        'support_per_class': support_per_class,
        'feature_importance': feature_importance,
        'label_encoder': le,
        'y_test_encoded': y_test_encoded,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'bootstrap_ci': bootstrap_ci(y_test_encoded, y_pred, 'classification'),
        'calibration': calibration_curve(y_test_encoded, y_pred_proba),
        'pr_curve': pr_curve(y_test_encoded, y_pred_proba),
        'error_examples': error_examples(le.inverse_transform(y_test_encoded), le.inverse_transform(y_pred), y_pred_proba, list(X_test.columns) if hasattr(X_test,'columns') else None, X_test),
    }


def train_svm_regressor(X_train, X_test, y_train, y_test, params: dict,
                         feature_names: List[str]) -> Dict[str, Any]:
    """Train SVM regressor"""
    # Parse gamma
    gamma = params['gamma']
    if gamma not in ['scale', 'auto']:
        try:
            gamma = float(gamma)
        except:
            gamma = 'scale'

    model = SVR(
        kernel=params['kernel'],
        C=params['C'],
        gamma=gamma,
        degree=params['degree'],
        coef0=params['coef0'],
        epsilon=params['epsilon']
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
        'train_r2': _to_native_type(r2_score(y_train, y_train_pred)),
        'n_support_vectors': int(len(model.support_))
    }

    # Feature importance via permutation
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
        'feature_importance': feature_importance,
        'bootstrap_ci': bootstrap_ci(y_test, y_pred, 'regression'),
    }


def perform_cross_validation(X, y, params: dict, task_type: str, cv_folds: int) -> Dict[str, Any]:
    """Perform cross-validation"""
    gamma = params['gamma']
    if gamma not in ['scale', 'auto']:
        try:
            gamma = float(gamma)
        except:
            gamma = 'scale'

    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = SVC(
            kernel=params['kernel'],
            C=params['C'],
            gamma=gamma,
            degree=params['degree'],
            random_state=params['random_state']
        )
        cv_target, cv_task = y_encoded, 'classification'
    else:
        model = SVR(
            kernel=params['kernel'],
            C=params['C'],
            gamma=gamma,
            degree=params['degree'],
            epsilon=params['epsilon']
        )
        cv_target, cv_task = y, 'regression'

    cv = run_cv(model, X, cv_target, cv_task, cv_folds, params['random_state'])
    cv['cv_metric'] = cv['cv_scoring']  # preserve svm's original field name
    return cv


def generate_prediction_examples(
    result: Dict,
    task_type: str,
    n_examples: int = 15,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample prediction examples from the test set.

    Regression     → actual, predicted, error, error_pct
    Classification → actual (label), predicted (label), correct, confidence
    """
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
                error_pct = (
                    _to_native_type(abs(error) / abs(actual) * 100)
                    if actual != 0 else None
                )
                examples.append({
                    'actual': actual,
                    'predicted': predicted,
                    'error': error,
                    'error_pct': error_pct,
                })
            return examples

        else:  # classification
            le: LabelEncoder = result['label_encoder']
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
                    'actual': actual_label,
                    'predicted': predicted_label,
                    'correct': bool(y_test_enc[i] == y_pred_enc[i]),
                    'confidence': confidence,
                })
            return examples

    except Exception:
        return []


def generate_decision_boundary_plot(model, X, y, feature_names: List[str],
                                     scaler=None, le=None) -> str:
    """Generate 2D decision boundary plot (for first 2 features)"""
    if X.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use first two features
    X_2d = X[:, :2]

    # Create mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Pad with zeros for other features
    if X.shape[1] > 2:
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        padding = np.zeros((mesh_points.shape[0], X.shape[1] - 2))
        mesh_full = np.hstack([mesh_points, padding])
    else:
        mesh_full = np.c_[xx.ravel(), yy.ravel()]

    # Predict
    try:
        Z = model.predict(mesh_full)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
    except:
        pass

    # Plot points
    if le is not None:
        y_encoded = le.transform(y) if hasattr(y, 'values') else y
    else:
        y_encoded = y

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap='coolwarm',
                         edgecolors='black', s=50, alpha=0.7)

    # Mark support vectors if available
    if hasattr(model, 'support_'):
        sv = X[model.support_, :2]
        ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none',
                   edgecolors='green', linewidths=2, label='Support Vectors')

    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title(
        f'SVM Decision Boundary — "{feature_names[0]}" vs "{feature_names[1]}"',
        fontsize=13, fontweight='bold'
    )
    if len(feature_names) > 2:
        ax.set_xlabel(
            f'{feature_names[0]}  (2-feature projection — other features set to 0)',
            fontsize=10
        )
    ax.legend()

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_confusion_matrix_plot(cm: List[List[int]], class_labels: List[str]) -> str:
    """Generate confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))

    cm_array = np.array(cm)
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)

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
    ax.set_title('SVM Feature Importance (Permutation)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_regression_plot(y_test, y_pred) -> str:
    """Generate actual vs predicted plot for regression"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, color='#3b82f6', s=30)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Residuals
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


def generate_support_vectors_plot(support_per_class: List[Dict]) -> str:
    """Generate support vectors distribution plot"""
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = [d['class'] for d in support_per_class]
    n_svs = [d['n_support_vectors'] for d in support_per_class]

    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, n_svs, color=colors, edgecolor='black', alpha=0.8)

    for bar, n in zip(bars, n_svs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(n), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Number of Support Vectors', fontsize=11)
    ax.set_title('Support Vectors per Class', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, task_type: str, params: dict,
                            scale_features: bool = True) -> Dict[str, Any]:
    """Generate interpretation of SVM results"""
    key_insights = []

    # ── 1. Model performance (분류/회귀 기준 분리) ──────────────────
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
            perf_desc = 'Moderate performance — consider tuning C or kernel'
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

        # 회귀는 분류보다 낮은 기준 적용
        if r2 >= 0.75:
            status = 'positive'
            perf_desc = 'Strong predictive fit'
        elif r2 >= 0.50:
            status = 'neutral'
            perf_desc = 'Moderate fit'
        elif r2 >= 0.25:
            status = 'warning'
            perf_desc = 'Weak fit — consider feature engineering or kernel change'
        else:
            status = 'warning'
            perf_desc = 'Poor fit — SVR may not suit this data distribution'

        train_str = f', Train R²: {train_r2:.3f}' if train_r2 is not None else ''
        key_insights.append({
            'title': 'Regression Performance',
            'description': (
                f'{perf_desc}. Test R²: {r2:.3f}{train_str}, RMSE: {rmse:.4f}, '
                f'MAE: {result["metrics"]["mae"]:.4f}'
            ),
            'status': status
        })

    # ── 2. Scaling warning ──────────────────────────────────────────
    if not scale_features:
        key_insights.append({
            'title': 'Feature Scaling Disabled',
            'description': (
                'SVM is highly sensitive to feature scale. Without standardization, '
                'features with large ranges will dominate the kernel computation and '
                'may severely degrade performance. Enable "Scale Features" for reliable results.'
            ),
            'status': 'warning'
        })

    # ── 3. Kernel info ──────────────────────────────────────────────
    kernel_desc = {
        'linear': 'Linear kernel — best for linearly separable data; fastest to train.',
        'rbf': 'RBF kernel — handles non-linear boundaries; good default choice.',
        'poly': f'Polynomial kernel (degree={params["degree"]}) — captures higher-order interactions.',
        'sigmoid': 'Sigmoid kernel — similar to a single-layer neural network; use with care.'
    }
    key_insights.append({
        'title': f'Kernel: {params["kernel"].upper()}',
        'description': kernel_desc.get(params['kernel'], 'Custom kernel'),
        'status': 'neutral'
    })

    # ── 4. Support vectors (완화된 해석) ───────────────────────────
    n_sv = result['metrics']['n_support_vectors']
    key_insights.append({
        'title': 'Support Vectors',
        'description': (
            f'{n_sv} support vectors define the decision boundary. '
            'A moderate number is typical; very few may indicate underfitting, '
            'while very many suggests a complex or noisy boundary.'
        ),
        'status': 'neutral'
    })

    # ── 5. Regularization C (로그 스케일 해석) ────────────────────
    C = params['C']
    if C >= 100:
        c_desc = 'Very high C — minimal regularization; high risk of overfitting'
        c_status = 'warning'
    elif C >= 10:
        c_desc = 'High C — less regularization; watch for overfitting'
        c_status = 'neutral'
    elif C >= 0.1:
        c_desc = 'Moderate C — balanced bias-variance trade-off'
        c_status = 'neutral'
    else:
        c_desc = 'Low C — strong regularization; may underfit complex boundaries'
        c_status = 'neutral'

    key_insights.append({
        'title': f'Regularization (C={C})',
        'description': c_desc,
        'status': c_status
    })

    # ── 6. Recommendation ──────────────────────────────────────────
    if task_type == 'classification':
        rec = (
            'SVM classifier trained successfully. '
            'For further improvement: try grid search over C and gamma, '
            'or switch kernels (rbf ↔ linear). '
            'Ensure features are scaled for best results.'
        )
    else:
        rec = (
            'SVR model trained successfully. '
            'For further improvement: tune epsilon alongside C — epsilon controls '
            'the width of the insensitive tube and directly impacts regression accuracy. '
            'Feature scaling is especially important for SVR.'
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

        # Prepare features
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle categorical features — one-hot encoding (safer than LabelEncoder for nominals)
        cat_cols = [c for c in X.columns if X[c].dtype == 'object']
        num_cols = [c for c in X.columns if X[c].dtype != 'object']

        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)

        # Drop rows with NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        # Update feature_cols to reflect one-hot expanded columns
        feature_cols = list(X.columns)

        if len(X) < 30:
            raise ValueError("At least 30 valid samples required.")

        # Auto-detect task type
        if task_type == 'auto':
            task_type = detect_task_type(y)

        # Parameters
        params = {
            'kernel': payload.get('kernel', 'rbf'),
            'C': float(payload.get('C', 1.0)),
            'gamma': payload.get('gamma', 'scale'),
            'degree': int(payload.get('degree', 3)),
            'coef0': float(payload.get('coef0', 0.0)),
            'epsilon': float(payload.get('epsilon', 0.1)),
            'random_state': int(payload.get('random_state', 42))
        }
        cv_folds = int(payload.get('cv_folds', 5))
        scale_features = bool(payload.get('scale_features', True))

        # Scale features
        scaler = None
        X_array = X.values
        if scale_features:
            scaler = StandardScaler()
            X_array = scaler.fit_transform(X_array)

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y, test_size=test_size, random_state=params['random_state'],
                stratify=y if task_type == 'classification' else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_array, y, test_size=test_size, random_state=params['random_state']
            )

        # Train model
        if task_type == 'classification':
            result = train_svm_classifier(X_train, X_test, y_train, y_test, params, feature_cols)
        else:
            result = train_svm_regressor(X_train, X_test, y_train, y_test, params, feature_cols)

        model = result['model']

        # Cross-validation
        cv_result = perform_cross_validation(X_array, y, params, task_type, cv_folds)

        # Generate visualizations
        importance_plot = generate_feature_importance_plot(result['feature_importance'])

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            sv_plot = generate_support_vectors_plot(result['support_per_class'])
            regression_plot = None
            decision_plot = generate_decision_boundary_plot(
                model, X_array, y, feature_cols, scaler, result['label_encoder']
            )
        else:
            cm_plot = None
            roc_plot = None
            sv_plot = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])
            decision_plot = None

        # Interpretation
        interpretation = generate_interpretation(result, task_type, params,
                                                 scale_features=scale_features)

        # Prediction examples
        prediction_examples = generate_prediction_examples(result, task_type)

        # Prepare response
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
                'kernel': params['kernel'],
                'C': params['C'],
                'gamma': str(params['gamma']),
                'degree': params['degree'],
                'scaled': scale_features
            },
            'metrics': result['metrics'],
            'bootstrap_ci': result.get('bootstrap_ci'),
            'feature_importance': result['feature_importance'],
            'cv_results': cv_result,
            'importance_plot': importance_plot,
            'interpretation': interpretation,
            'prediction_examples': prediction_examples
        }

        if task_type == 'classification':
            response['per_class_metrics'] = result['per_class_metrics']
            response['confusion_matrix'] = result['confusion_matrix']
            response['class_labels'] = result['class_labels']
            response['support_per_class'] = result['support_per_class']
            response['cm_plot'] = cm_plot
            response['roc_plot'] = roc_plot
            response['calibration'] = result.get('calibration')
            response['pr_curve'] = result.get('pr_curve')
            response['error_examples'] = result.get('error_examples')
            response['sv_plot'] = sv_plot
            response['decision_plot'] = decision_plot
        else:
            response['regression_plot'] = regression_plot

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
