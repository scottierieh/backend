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
import io
import base64
import warnings
from model_diagnostics import bootstrap_ci, calibration_curve, pr_curve, error_examples

from sklearn.model_selection import train_test_split, cross_val_score
from cv_strategy import run_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    plot_tree, export_text
)
from sklearn.inspection import partial_dependence
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
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


def detect_task_type(y: pd.Series) -> str:
    if y.dtype == 'object' or y.dtype.name == 'category':
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
# SHAP
# ─────────────────────────────────────────────

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

        return {'shap_importance': shap_importance, 'shap_plot': shap_plot}
    except Exception as e:
        return {'shap_importance': [], 'shap_plot': None, 'error': str(e)}


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

    return {
        'model': model, 'metrics': metrics,
        'per_class_metrics': per_class,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'bootstrap_ci': bootstrap_ci(y_test_enc, y_pred, 'classification'),
        'calibration': calibration_curve(y_test_enc, y_pred_proba),
        'pr_curve': pr_curve(y_test_enc, y_pred_proba),
        'error_examples': error_examples(le.inverse_transform(y_test_enc), le.inverse_transform(y_pred), y_pred_proba, list(X_test.columns) if hasattr(X_test,'columns') else None, X_test),
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
        'bootstrap_ci': bootstrap_ci(y_test, y_pred, 'regression'),
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
        cv_target, cv_task = y_enc, 'classification'
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
        cv_target, cv_task = y, 'regression'

    return run_cv(model, X, cv_target, cv_task, cv_folds, params['random_state'])


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
            if X[col].dtype == 'object':
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

        if task_type == 'classification' and y.dtype == 'object':
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

        # ── SHAP ──
        shap_result = compute_shap(model, X_train, X_test, feature_cols, task_type)

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
            cm_plot         = generate_confusion_matrix_plot(
                result['confusion_matrix'], result['class_labels'])
            roc_plot        = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            regression_plot = None
        else:
            cm_plot         = None
            roc_plot        = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])

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
            'bootstrap_ci':       result.get('bootstrap_ci'),
            'feature_importance': feature_importance,
            'cv_results':         cv_result,
            'tree_info':          tree_info,
            'importance_plot':    importance_plot,
            'tree_plot':          tree_plot,
            'shap_importance':    shap_result.get('shap_importance', []),
            'shap_plot':          shap_result.get('shap_plot'),
            'pdp_plot':           pdp_plot,
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
            response['calibration']       = result.get('calibration')
            response['pr_curve']          = result.get('pr_curve')
            response['error_examples']    = result.get('error_examples')
        else:
            response['regression_plot'] = regression_plot

        print(json.dumps(response, default=_to_native))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
