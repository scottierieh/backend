"""
Decision Tree Classification and Regression — CLI script
Includes: Normalized Importance, SHAP, Partial Dependence Plot, Tree Rule
Extraction. Ported from scottierieh/backend's api/decision_tree_analysis.py
(FastAPI router) to the stdin/stdout CLI contract used by
src/backend/main.py's generic script runner.
"""

import sys
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import io
import base64
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    plot_tree, export_text
)
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import shap


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


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _fig_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def detect_task_type(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    if len(y.unique()) <= 10 or len(y.unique()) / len(y) < 0.05:
        return 'classification'
    return 'regression'


def _fix_criterion(criterion: str, task_type: str) -> str:
    clf_valid = {'gini', 'entropy', 'log_loss'}
    reg_valid = {'squared_error', 'friedman_mse', 'absolute_error', 'poisson'}
    if task_type == 'classification':
        return criterion if criterion in clf_valid else 'gini'
    return criterion if criterion in reg_valid else 'squared_error'


# ─────────────────────────────────────────────
# Feature Importance  (raw + normalized + %)
# ─────────────────────────────────────────────

def get_feature_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    raw = model.feature_importances_
    max_imp = raw.max() if raw.max() > 0 else 1.0

    data = []
    for name, imp in zip(feature_names, raw):
        data.append({
            'feature': name,
            'importance': _to_native(imp),
            'normalized_importance': _to_native(imp / max_imp),
            'importance_pct': _to_native(imp * 100),
        })

    data.sort(key=lambda x: x['importance'], reverse=True)
    for rank, row in enumerate(data, 1):
        row['rank'] = rank
    return data


# ─────────────────────────────────────────────
# Permutation Importance (unbiased, handles high-cardinality
# features — Gini/impurity importance above is biased toward
# them). Mirrors random_forest_analysis.py's implementation and
# response field shape (feature/importance_mean/importance_std/rank)
# for consistency across the tree-based model pages.
# ─────────────────────────────────────────────

def compute_permutation_importance(
    model, X_test: np.ndarray, y_test, feature_names: List[str],
    n_repeats: int = 10, random_state: int = 42
) -> List[Dict[str, Any]]:
    try:
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        result = []
        for name, mean, std in zip(feature_names, perm.importances_mean, perm.importances_std):
            result.append({
                'feature': name,
                'importance_mean': _to_native(mean),
                'importance_std': _to_native(std),
            })
        result.sort(key=lambda x: x['importance_mean'], reverse=True)
        for i, item in enumerate(result):
            item['rank'] = i + 1
        return result
    except Exception:
        return []


# ─────────────────────────────────────────────
# max_depth Validation Curve — train/test score at each candidate
# depth, reusing the existing train/test split (no re-splitting).
# ─────────────────────────────────────────────

def compute_max_depth_validation_curve(X_train, X_test, y_train, y_test,
                                        task_type: str, params: dict,
                                        max_depth_cap: int = 15) -> Optional[Dict[str, Any]]:
    try:
        # Probe the tree's natural (unrestricted) depth so the sweep covers a
        # sensible range instead of always going all the way to the cap.
        probe = (DecisionTreeClassifier if task_type == 'classification' else DecisionTreeRegressor)(
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['random_state']
        )
        probe.fit(X_train, y_train)
        natural_depth = probe.get_depth()
        max_depth_sweep = min(max_depth_cap, max(natural_depth, 5))

        depths = list(range(1, max_depth_sweep + 1))
        train_scores, test_scores = [], []
        for d in depths:
            Model = DecisionTreeClassifier if task_type == 'classification' else DecisionTreeRegressor
            m = Model(
                max_depth=d,
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                criterion=params['criterion'],
                splitter=params['splitter'],
                random_state=params['random_state']
            )
            m.fit(X_train, y_train)
            if task_type == 'classification':
                train_scores.append(_to_native(accuracy_score(y_train, m.predict(X_train))))
                test_scores.append(_to_native(accuracy_score(y_test, m.predict(X_test))))
            else:
                train_scores.append(_to_native(r2_score(y_train, m.predict(X_train))))
                test_scores.append(_to_native(r2_score(y_test, m.predict(X_test))))

        return {'depth': depths, 'train_score': train_scores, 'test_score': test_scores}
    except Exception:
        return None


def generate_validation_curve_plot(vc: Dict[str, Any], task_type: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(vc['depth'], vc['train_score'], color='#2563eb', linewidth=2, marker='o', markersize=4, label='Train')
    ax.plot(vc['depth'], vc['test_score'], color='#dc2626', linewidth=2, marker='o', markersize=4, label='Test')
    ax.set_xlabel('max_depth', fontsize=11)
    ax.set_ylabel('Accuracy' if task_type == 'classification' else 'R² Score', fontsize=11)
    ax.set_title('Validation Curve: max_depth', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────

def _shap_samples_from_matrix(sv, X_arr, feature_names: List[str], expected_value, max_samples: int = 8):
    """Turns an already-computed SHAP matrix into a small set of per-sample (feature
    value, contribution) pairs for a Force Plot. Reuses the SHAP values the caller
    already computed for shap_importance -- no extra explainer calls. For binary
    classification (3D output with 2 classes), picks the positive class's slice, same
    convention as the confusion matrix / ROC curve elsewhere in this app. True
    multiclass (3+ classes, or the older list-of-arrays shape) is skipped rather than
    guess which class to show."""
    try:
        if isinstance(sv, list):
            return None
        sv = np.asarray(sv)
        base = expected_value
        if sv.ndim == 3:
            n_classes = sv.shape[2]
            if n_classes != 2:
                return None
            sv = sv[:, :, 1]
            base = base[1] if isinstance(base, (list, np.ndarray)) else base
        elif sv.ndim != 2:
            return None
        n = min(max_samples, sv.shape[0])
        if n == 0:
            return None
        base = float(base[0]) if isinstance(base, (list, np.ndarray)) else float(base)
        X_arr = np.asarray(X_arr)
        return [
            {
                'base_value': _to_native(base),
                'contributions': [
                    {'feature': feature_names[j], 'value': _to_native(X_arr[i, j]), 'shap': _to_native(sv[i, j])}
                    for j in range(len(feature_names))
                ],
            }
            for i in range(n)
        ]
    except Exception:
        return None

def compute_shap(model, X_train: np.ndarray, X_test: np.ndarray,
                 feature_names: List[str], task_type: str) -> Dict[str, Any]:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # shap_values shape varies by version and task:
        # - list of arrays (one per class): each (n_samples, n_features)
        # - 3D array: (n_samples, n_features, n_classes)
        # - 2D array: (n_samples, n_features) for binary/regression
        if isinstance(shap_values, list):
            # older shap: list of (n_samples, n_features)
            mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        elif shap_values.ndim == 3:
            # newer shap: (n_samples, n_features, n_classes)
            mean_abs = np.abs(shap_values).mean(axis=2)
        else:
            # binary/regression: (n_samples, n_features)
            mean_abs = np.abs(shap_values)

        mean_shap = mean_abs.mean(axis=0)

        shap_importance = []
        for name, val in zip(feature_names, mean_shap):
            shap_importance.append({'feature': name, 'mean_abs_shap': _to_native(val)})
        shap_importance.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
        shap_samples = _shap_samples_from_matrix(shap_values, X_test, feature_names, explainer.expected_value)

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.4)))
        feats  = [d['feature'] for d in shap_importance][::-1]
        values = [d['mean_abs_shap'] for d in shap_importance][::-1]
        max_val = max(values) if values else 1
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feats)))
        bars = ax.barh(feats, values, color=colors, edgecolor='black', alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.4f}', va='center', fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, axis='x')
        plt.tight_layout()
        shap_plot = _fig_to_b64(fig)

        return {'shap_importance': shap_importance, 'shap_plot': shap_plot, 'shap_samples': shap_samples}
    except Exception as e:
        return {'shap_importance': [], 'shap_plot': None, 'shap_samples': None, 'error': str(e)}


# ─────────────────────────────────────────────
# Partial Dependence Plot
# ─────────────────────────────────────────────

def compute_pdp(model, X_train: np.ndarray, feature_names: List[str],
                top_n: int = 6) -> Optional[str]:
    try:
        n = min(top_n, len(feature_names))
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

        if n == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n):
            ax = axes[i // ncols][i % ncols]
            pd_result = partial_dependence(model, X_train, [i], kind='average')
            ax.plot(pd_result['grid_values'][0], pd_result['average'][0],
                    color='#16a34a', linewidth=2)
            ax.set_xlabel(feature_names[i], fontsize=10)
            ax.set_ylabel('Partial Dependence', fontsize=9)
            ax.set_title(f'PDP: {feature_names[i]}', fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)

        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        plt.suptitle('Partial Dependence Plots (Top Features)', fontsize=13,
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        return _fig_to_b64(fig)
    except Exception:
        return None

def compute_pdp_json(model, X_train: np.ndarray, feature_names: List[str],
                      top_n: int = 6) -> Optional[List[Dict]]:
    """Same top-N feature selection as compute_pdp, but returns the {grid, average} curve
    data as JSON instead of a PNG -- for an interactive PDP/ICE chart on the frontend."""
    try:
        n = min(top_n, len(feature_names))
        out = []
        for i in range(n):
            pd_result = partial_dependence(model, X_train, [i], kind='average')
            out.append({
                'feature': feature_names[i],
                'grid': [_to_native(v) for v in pd_result['grid_values'][0]],
                'average': [_to_native(v) for v in pd_result['average'][0]],
            })
        return out
    except Exception:
        return None


# ─────────────────────────────────────────────
# Tree Rule Extraction
# ─────────────────────────────────────────────

def extract_tree_rules(model, feature_names: List[str],
                       class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        text_rules = export_text(
            model, feature_names=feature_names,
            max_depth=10, decimals=3, show_weights=True
        )

        tree      = model.tree_
        feature   = tree.feature
        threshold = tree.threshold
        n_samples = tree.n_node_samples
        value     = tree.value

        leaf_rules = []

        children_left  = tree.children_left
        children_right = tree.children_right

        def recurse(node, path):
            if feature[node] == -2:   # leaf node
                if class_names is not None:
                    cls_idx    = int(np.argmax(value[node][0]))
                    prediction = class_names[cls_idx]
                    confidence = float(value[node][0][cls_idx] / n_samples[node])
                else:
                    prediction = float(value[node][0][0])
                    confidence = None
                leaf_rules.append({
                    'conditions': list(path),
                    'prediction': prediction,
                    'confidence': round(confidence, 3) if confidence is not None else None,
                    'n_samples':  int(n_samples[node])
                })
            else:
                fname = feature_names[feature[node]]
                thr   = round(float(threshold[node]), 3)
                recurse(children_left[node],  path + [f'{fname} <= {thr}'])
                recurse(children_right[node], path + [f'{fname} > {thr}'])

        if tree.node_count <= 511:
            recurse(0, [])
            leaf_rules.sort(key=lambda x: x['n_samples'], reverse=True)

        return {
            'text_rules':      text_rules,
            'leaf_rules':      leaf_rules[:30],
            'n_leaves':        int(model.get_n_leaves()),
            'rules_truncated': tree.node_count > 511
        }
    except Exception as e:
        return {'text_rules': '', 'leaf_rules': [], 'error': str(e)}


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────

def generate_importance_plot(importance_data: List[Dict], top_n: int = 20) -> str:
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_data[:top_n]) * 0.4)))
    top   = importance_data[:top_n]
    feats = [d['feature'] for d in top][::-1]
    imps  = [d['importance'] for d in top][::-1]
    norms = [d['normalized_importance'] for d in top][::-1]
    max_imp = max(imps) if imps else 1

    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(feats)))
    bars = ax.barh(feats, imps, color=colors, edgecolor='black', alpha=0.8)
    for bar, imp, norm in zip(bars, imps, norms):
        ax.text(bar.get_width() + max_imp * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{imp:.3f}  ({norm * 100:.0f}%)', va='center', fontsize=9)
    ax.set_xlabel('Feature Importance (Gini / Variance Reduction)', fontsize=11)
    ax.set_title('Decision Tree Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_tree_plot(model, feature_names, class_names=None, max_depth=4) -> str:
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(model, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=8, max_depth=max_depth, ax=ax)
    ax.set_title(f'Decision Tree Structure (depth ≤ {max_depth})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_confusion_matrix_plot(cm, class_labels) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Greens',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_roc_plot(roc_data: Dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    for (label, data), color in zip(roc_data.items(), colors):
        ax.plot(data['fpr'], data['tpr'], color=color, linewidth=2,
                label=f'{label} (AUC = {data["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_pr_plot(pr_data: Dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data)))
    for (label, data), color in zip(pr_data.items(), colors):
        ax.plot(data['recall'], data['precision'], color=color, linewidth=2,
                label=f'{label} (AP = {data["ap"]:.3f})')
    # Positive-class base-rate reference line (only meaningful for a single/binary curve —
    # in multiclass each class has its own prevalence, so we skip it to avoid clutter,
    # mirroring how the ROC plot only draws one shared "Random" diagonal).
    if len(pr_data) == 1:
        base_rate = list(pr_data.values())[0].get('base_rate')
        if base_rate is not None:
            ax.axhline(y=base_rate, color='k', linestyle='--', linewidth=1,
                       label=f'Base Rate ({base_rate:.3f})')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_regression_plot(y_test, y_pred) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_test = np.array(y_test); y_pred = np.array(y_pred)

    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, color='#22c55e', s=30)
    lo = min(y_test.min(), y_pred.min()); hi = max(y_test.max(), y_pred.max())
    ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual', fontsize=11); ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#16a34a', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=11); ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_b64(fig)


def generate_regression_plots(y_test, y_pred) -> List[Dict[str, str]]:
    """Same diagnostics as generate_regression_plot, but as separate
    single-panel figures so the frontend can show each as its own tab."""
    y_test = np.array(y_test); y_pred = np.array(y_pred)
    plots = []

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.scatter(y_test, y_pred, alpha=0.5, color='#22c55e', s=30)
    lo = min(y_test.min(), y_pred.min()); hi = max(y_test.max(), y_pred.max())
    ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual', fontsize=11); ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plots.append({'label': 'Actual vs Predicted', 'image': _fig_to_data_url(fig1)})

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#16a34a', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=11); ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plots.append({'label': 'Residual Plot', 'image': _fig_to_data_url(fig2)})

    return plots


# ─────────────────────────────────────────────
# Classification Training
# ─────────────────────────────────────────────

def train_classifier(X_train, X_test, y_train, y_test,
                     params: dict, feature_names: List[str]) -> Dict[str, Any]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    n_classes   = len(le.classes_)

    model = DecisionTreeClassifier(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        criterion=params['criterion'],
        splitter=params['splitter'],
        max_leaf_nodes=params['max_leaf_nodes'],
        random_state=params['random_state']
    )
    model.fit(X_train, y_train_enc)

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_train_pred = model.predict(X_train)

    metrics = {
        'accuracy':        _to_native(accuracy_score(y_test_enc, y_pred)),
        'train_accuracy':  _to_native(accuracy_score(y_train_enc, y_train_pred)),
        'precision_macro': _to_native(precision_score(y_test_enc, y_pred, average='macro', zero_division=0)),
        'recall_macro':    _to_native(recall_score(y_test_enc, y_pred, average='macro', zero_division=0)),
        'f1_macro':        _to_native(f1_score(y_test_enc, y_pred, average='macro', zero_division=0)),
    }

    report = classification_report(y_test_enc, y_pred,
                                   target_names=[str(c) for c in le.classes_],
                                   output_dict=True)
    per_class = []
    for cls in le.classes_:
        s = str(cls)
        if s in report:
            per_class.append({
                'class':     s,
                'precision': _to_native(report[s]['precision']),
                'recall':    _to_native(report[s]['recall']),
                'f1_score':  _to_native(report[s]['f1-score']),
                'support':   int(report[s]['support'])
            })

    cm = confusion_matrix(y_test_enc, y_pred)

    roc_data = {}
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test_enc, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_data['binary'] = {
            'fpr': [_to_native(x) for x in fpr],
            'tpr': [_to_native(x) for x in tpr],
            'auc': _to_native(roc_auc)
        }
        metrics['auc'] = _to_native(roc_auc)
    else:
        for i, cls in enumerate(le.classes_):
            y_bin = (y_test_enc == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_pred_proba[:, i])
            roc_data[str(cls)] = {
                'fpr': [_to_native(x) for x in fpr],
                'tpr': [_to_native(x) for x in tpr],
                'auc': _to_native(auc(fpr, tpr))
            }
        macro_auc = _compute_multiclass_auc(y_test_enc, y_pred_proba)
        if macro_auc is not None:
            metrics['auc'] = macro_auc

    pr_data = {}
    if n_classes == 2:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_enc, y_pred_proba[:, 1])
        ap = average_precision_score(y_test_enc, y_pred_proba[:, 1])
        pr_data['binary'] = {
            'precision': [_to_native(x) for x in precision_curve],
            'recall':    [_to_native(x) for x in recall_curve],
            'ap':        _to_native(ap),
            'base_rate': _to_native(np.mean(y_test_enc))
        }
        metrics['average_precision'] = _to_native(ap)
    else:
        class_aps = []
        for i, cls in enumerate(le.classes_):
            y_bin = (y_test_enc == i).astype(int)
            precision_curve, recall_curve, _ = precision_recall_curve(y_bin, y_pred_proba[:, i])
            ap = average_precision_score(y_bin, y_pred_proba[:, i])
            pr_data[str(cls)] = {
                'precision': [_to_native(x) for x in precision_curve],
                'recall':    [_to_native(x) for x in recall_curve],
                'ap':        _to_native(ap),
                'base_rate': _to_native(np.mean(y_bin))
            }
            class_aps.append(ap)
        if class_aps:
            metrics['average_precision_macro'] = _to_native(float(np.mean(class_aps)))

    return {
        'model': model, 'metrics': metrics,
        'per_class_metrics': per_class,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'pr_data': pr_data,
        'label_encoder': le,
        'tree_info': {
            'n_nodes':          int(model.tree_.node_count),
            'max_depth_actual': int(model.get_depth()),
            'n_leaves':         int(model.get_n_leaves())
        }
    }


# ─────────────────────────────────────────────
# Regression Training
# ─────────────────────────────────────────────

def train_regressor(X_train, X_test, y_train, y_test,
                    params: dict, feature_names: List[str]) -> Dict[str, Any]:
    model = DecisionTreeRegressor(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        criterion=params['criterion'],
        splitter=params['splitter'],
        max_leaf_nodes=params['max_leaf_nodes'],
        random_state=params['random_state']
    )
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    mse          = mean_squared_error(y_test, y_pred)

    return {
        'model': model,
        'metrics': {
            'mse':      _to_native(mse),
            'rmse':     _to_native(np.sqrt(mse)),
            'mae':      _to_native(mean_absolute_error(y_test, y_pred)),
            'r2':       _to_native(r2_score(y_test, y_pred)),
            'train_r2': _to_native(r2_score(y_train, y_train_pred)),
        },
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred,
        'tree_info': {
            'n_nodes':          int(model.tree_.node_count),
            'max_depth_actual': int(model.get_depth()),
            'n_leaves':         int(model.get_n_leaves())
        }
    }


# ─────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────

def perform_cv(X, y, params: dict, task_type: str, cv_folds: int) -> Dict[str, Any]:
    if task_type == 'classification':
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        model = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            criterion=params['criterion'],
            splitter=params['splitter'],
            max_leaf_nodes=params['max_leaf_nodes'],
            random_state=params['random_state']
        )
        scores = cross_val_score(model, X, y_enc, cv=cv_folds, scoring='accuracy')
    else:
        model = DecisionTreeRegressor(
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            criterion=params['criterion'],
            splitter=params['splitter'],
            max_leaf_nodes=params['max_leaf_nodes'],
            random_state=params['random_state']
        )
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

    return {
        'cv_scores': [_to_native(s) for s in scores],
        'cv_mean':   _to_native(float(np.mean(scores))),
        'cv_std':    _to_native(float(np.std(scores))),
        'cv_folds':  cv_folds
    }


# ─────────────────────────────────────────────
# Interpretation
# ─────────────────────────────────────────────

def generate_interpretation(result: Dict, task_type: str,
                             feature_importance: List[Dict],
                             tree_info: Dict) -> Dict[str, Any]:
    key_insights = []

    if task_type == 'classification':
        accuracy = result['metrics']['accuracy']
        f1 = result['metrics']['f1_macro']
        if accuracy >= 0.9:
            status, perf_desc = 'positive', 'Excellent classification performance'
        elif accuracy >= 0.7:
            status, perf_desc = 'neutral', 'Good classification performance'
        else:
            status, perf_desc = 'warning', 'Model performance may need improvement'
        key_insights.append({
            'title': 'Classification Performance',
            'description': f'{perf_desc}. Accuracy: {accuracy:.1%}, F1-macro: {f1:.3f}',
            'status': status
        })
        if 'auc' in result['metrics']:
            auc_val  = result['metrics']['auc']
            auc_desc = ('Excellent discrimination ability' if auc_val > 0.9
                        else 'Good discrimination ability' if auc_val > 0.7
                        else 'Fair discrimination ability')
            key_insights.append({
                'title': 'AUC Score',
                'description': f'Area Under ROC Curve: {auc_val:.3f}. {auc_desc}.',
                'status': 'positive' if auc_val > 0.8 else 'neutral'
            })
    else:
        r2   = result['metrics']['r2']
        rmse = result['metrics']['rmse']
        if r2 >= 0.8:
            status, perf_desc = 'positive', 'Excellent fit'
        elif r2 >= 0.5:
            status, perf_desc = 'neutral', 'Moderate fit'
        else:
            status, perf_desc = 'warning', 'Weak fit — consider adding features or adjusting depth'
        key_insights.append({
            'title': 'Regression Performance',
            'description': f'{perf_desc}. R² = {r2:.3f}, RMSE = {rmse:.4f}',
            'status': status
        })

    key_insights.append({
        'title': 'Tree Complexity',
        'description': (f'Tree has {tree_info["n_nodes"]} nodes, '
                        f'{tree_info["n_leaves"]} leaves, '
                        f'and a depth of {tree_info["max_depth_actual"]}.'),
        'status': 'neutral'
    })

    top3 = feature_importance[:3]
    feature_str = ', '.join([f"{f['feature']} ({f['importance']:.3f})" for f in top3])
    key_insights.append({
        'title': 'Top Discriminating Features',
        'description': f'Top features: {feature_str}',
        'status': 'neutral'
    })

    top3_total = sum(f['importance'] for f in top3)
    if top3_total > 0.7:
        key_insights.append({
            'title': 'Feature Concentration',
            'description': (f'Top 3 features account for {top3_total:.1%} of total importance. '
                            f'The model relies heavily on a small subset of features.'),
            'status': 'neutral'
        })

    train_m = (result['metrics'].get('train_accuracy', 1.0)
               if task_type == 'classification'
               else result['metrics'].get('train_r2', 1.0))
    test_m  = (result['metrics']['accuracy']
               if task_type == 'classification'
               else result['metrics']['r2'])
    gap = train_m - test_m
    if gap > 0.15:
        key_insights.append({
            'title': 'Overfitting Warning',
            'description': (f'Train–test gap of {gap:.3f} ({train_m:.3f} vs {test_m:.3f}) '
                            f'suggests overfitting. Consider reducing max_depth or '
                            f'increasing min_samples_leaf.'),
            'status': 'warning'
        })

    return {
        'key_insights': key_insights,
        'recommendation': (
            'Decision Tree trained successfully. '
            'For better generalization, consider pruning (max_depth, min_samples_leaf) '
            'or ensemble methods such as Random Forest or Gradient Boosting.'
        )
    }


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target_col = payload.get('target_col') or payload.get('target')
        feature_cols = payload.get('feature_cols') or payload.get('features')
        task_type = payload.get('task_type', 'auto')
        test_size = float(payload.get('test_size', 0.2))
        max_depth = payload.get('max_depth', None)
        if max_depth is not None:
            max_depth = int(max_depth)
        min_samples_split = int(payload.get('min_samples_split', 2))
        min_samples_leaf = int(payload.get('min_samples_leaf', 1))
        max_features = payload.get('max_features', None)
        criterion = payload.get('criterion', 'gini')
        splitter = payload.get('splitter', 'best')
        max_leaf_nodes = payload.get('max_leaf_nodes', None)
        if max_leaf_nodes is not None:
            max_leaf_nodes = int(max_leaf_nodes)
        random_state = int(payload.get('random_state', 42))
        cv_folds = int(payload.get('cv_folds', 5))

        if not data:
            raise ValueError("Data not provided.")
        if not target_col or not feature_cols:
            raise ValueError("Missing data, features, or target")

        df           = pd.DataFrame(data)

        missing = [c for c in [target_col] + feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Encode categorical features
        categorical_features = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                categorical_features.append(col)
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

        if len(X) < 50:
            raise ValueError("At least 50 valid samples required.")

        if task_type == 'auto':
            task_type = detect_task_type(y)

        if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            y = pd.Series(LabelEncoder().fit_transform(y))

        criterion = _fix_criterion(criterion, task_type)

        params = {
            'max_depth':         max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf':  min_samples_leaf,
            'max_features':      (max_features
                                  if max_features not in (None, 'None')
                                  else None),
            'criterion':         criterion,
            'splitter':          splitter,
            'max_leaf_nodes':    max_leaf_nodes,
            'random_state':      random_state,
        }

        X_arr = X.values.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if task_type == 'classification' else None
        )

        if task_type == 'classification':
            result = train_classifier(X_train, X_test, y_train, y_test,
                                      params, feature_cols)
        else:
            result = train_regressor(X_train, X_test, y_train, y_test,
                                     params, feature_cols)

        model     = result['model']
        tree_info = result['tree_info']

        # ── Feature Importance (normalized) ──
        feature_importance = get_feature_importance(model, feature_cols)

        # ── CV ──
        cv_result = perform_cv(X_arr, y, params, task_type, cv_folds)

        # ── Permutation Importance (unbiased vs. Gini above) ──
        if task_type == 'classification':
            y_test_for_perm = result['label_encoder'].transform(y_test)
        else:
            y_test_for_perm = y_test.values if hasattr(y_test, 'values') else y_test
        perm_importance = compute_permutation_importance(model, X_test, y_test_for_perm, feature_cols)

        # ── max_depth Validation Curve (reuses the existing train/test split) ──
        validation_curve = compute_max_depth_validation_curve(
            X_train, X_test, y_train, y_test, task_type, params)
        validation_curve_plot = (generate_validation_curve_plot(validation_curve, task_type)
                                  if validation_curve else None)

        # ── SHAP ──
        shap_result = compute_shap(model, X_train, X_test, feature_cols, task_type)
        pdp_data = compute_pdp_json(model, X_train, feature_cols, top_n=6)

        # ── PDP (top 6 features) ──
        top6_names = [d['feature'] for d in feature_importance[:6]]
        top6_idx   = [feature_cols.index(n) for n in top6_names if n in feature_cols]
        # Reorder X_train to top6 order for PDP labels
        pdp_plot = compute_pdp(model, X_train, feature_cols, top_n=6)

        # ── Tree Rules ──
        class_names = result.get('class_labels') if task_type == 'classification' else None
        tree_rules  = extract_tree_rules(model, feature_cols, class_names)

        # ── Visualizations ──
        importance_plot = generate_importance_plot(feature_importance)
        tree_plot       = generate_tree_plot(model, feature_cols, class_names, max_depth=4)

        if task_type == 'classification':
            cm_plot          = generate_confusion_matrix_plot(
                result['confusion_matrix'], result['class_labels'])
            roc_plot         = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            pr_plot          = generate_pr_plot(result['pr_data']) if result.get('pr_data') else None
            regression_plot  = None
            regression_plots = None
        else:
            cm_plot          = None
            roc_plot         = None
            pr_plot          = None
            regression_plot  = generate_regression_plot(result['y_test'], result['y_pred'])
            regression_plots = generate_regression_plots(result['y_test'], result['y_pred'])

        # ── Interpretation ──
        interpretation = generate_interpretation(
            result, task_type, feature_importance, tree_info)

        # ── Build response ──
        try:
            from guardrails import compute_guardrails
            guardrails = compute_guardrails(X, y, feature_cols, task_type, result['metrics'])
        except Exception:
            guardrails = []

        response = {
            'guardrails': guardrails,
            'task_type':          task_type,
            'n_samples':          len(X),
            'n_features':         len(feature_cols),
            'n_train':            len(X_train),
            'n_test':             len(X_test),
            'parameters':         {k: _to_native(v) for k, v in params.items()},
            'metrics':            result['metrics'],
            'feature_importance': feature_importance,
            'perm_importance':    perm_importance,
            'cv_results':         cv_result,
            'tree_info':          tree_info,
            'importance_plot':    importance_plot,
            'tree_plot':          tree_plot,
            'validation_curve':      validation_curve,
            'validation_curve_plot': validation_curve_plot,
            'shap_importance':    shap_result.get('shap_importance', []),
            'shap_plot':          shap_result.get('shap_plot'),
            'shap_samples':       shap_result.get('shap_samples'),
            'pdp_plot':           pdp_plot,
            'pdp':                pdp_data,
            'tree_rules':         tree_rules,
            'interpretation':     interpretation,
        }

        if categorical_features:
            response['data_warnings'] = {
                'has_categorical': True,
                'categorical_features': categorical_features
            }

        if task_type == 'classification':
            response['per_class_metrics'] = result['per_class_metrics']
            response['confusion_matrix']  = result['confusion_matrix']
            response['class_labels']      = result['class_labels']
            response['cm_plot']           = cm_plot
            response['roc_plot']          = roc_plot
            response['pr_plot']           = pr_plot
        else:
            response['regression_plot']  = regression_plot
            response['regression_plots'] = regression_plots

        print(json.dumps(response, default=_to_native))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
