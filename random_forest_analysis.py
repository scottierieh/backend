"""
Random Forest Classification and Regression — CLI script
Ensemble learning with decision trees. Ported from scottierieh/backend's
api/random_forest.py (FastAPI router) to the stdin/stdout CLI contract used
by src/backend/main.py's generic script runner.
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from cv_strategy import run_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.tree import export_text
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
sns.set_style('darkgrid')
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


def train_rf_classifier(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    """Train Random Forest classifier"""
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(le.classes_)

    max_features = params.get('max_features', 'sqrt')
    if max_features == 'None':
        max_features = None

    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=max_features,
        bootstrap=params['bootstrap'],
        oob_score=params['oob_score'] if params['bootstrap'] else False,
        random_state=params['random_state'],
        n_jobs=-1
    )

    model.fit(X_train, y_train_encoded)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_train_pred = model.predict(X_train)

    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred)),
        'precision_macro': _to_native_type(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'recall_macro': _to_native_type(recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'train_accuracy': _to_native_type(accuracy_score(y_train_encoded, y_train_pred))
    }

    if params['bootstrap'] and params['oob_score']:
        metrics['oob_score'] = _to_native_type(model.oob_score_)

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

    return {
        'model': model,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'label_encoder': le
    }


def train_rf_regressor(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    """Train Random Forest regressor"""
    max_features = params.get('max_features', 'sqrt')
    if max_features == 'None':
        max_features = None

    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=max_features,
        bootstrap=params['bootstrap'],
        oob_score=params['oob_score'] if params['bootstrap'] else False,
        random_state=params['random_state'],
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        'mse': _to_native_type(mse),
        'rmse': _to_native_type(np.sqrt(mse)),
        'mae': _to_native_type(mean_absolute_error(y_test, y_pred)),
        'r2': _to_native_type(r2_score(y_test, y_pred)),
        'train_r2': _to_native_type(r2_score(y_train, y_train_pred))
    }

    if params['bootstrap'] and params['oob_score']:
        metrics['oob_score'] = _to_native_type(model.oob_score_)

    return {
        'model': model,
        'metrics': metrics,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred
    }


def get_feature_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Extract Gini feature importance in unified format (DT/GBM compatible)"""
    importance = model.feature_importances_
    total = importance.sum() if importance.sum() > 0 else 1.0
    max_imp = importance.max() if importance.max() > 0 else 1.0

    importance_data = []
    for name, imp in zip(feature_names, importance):
        importance_data.append({
            'feature': name,
            'importance': _to_native_type(imp),
            'importance_pct': _to_native_type(imp / total * 100),
            'normalized_importance': _to_native_type(imp / max_imp),
        })

    importance_data.sort(key=lambda x: x['importance'], reverse=True)
    for i, item in enumerate(importance_data):
        item['rank'] = i + 1

    return importance_data


def compute_permutation_importance(
    model, X_test: np.ndarray, y_test, feature_names: List[str],
    n_repeats: int = 10, random_state: int = 42
) -> List[Dict[str, Any]]:
    """Permutation importance — unbiased, handles high-cardinality features"""
    try:
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        result = []
        for name, mean, std in zip(feature_names, perm.importances_mean, perm.importances_std):
            result.append({
                'feature': name,
                'importance_mean': _to_native_type(mean),
                'importance_std': _to_native_type(std),
            })
        result.sort(key=lambda x: x['importance_mean'], reverse=True)
        for i, item in enumerate(result):
            item['rank'] = i + 1
        return result
    except Exception:
        return []


def compute_shap(model, X_train: np.ndarray, X_test: np.ndarray,
                 feature_names: List[str]) -> Dict:
    """SHAP TreeExplainer — lazy import, graceful fallback"""
    try:
        try:
            import shap as _shap
        except ImportError:
            return {'shap_importance': [], 'shap_plot': None,
                    'error': 'shap package not installed. Run: pip install shap'}

        explainer = _shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        sv = np.array(shap_values)
        if sv.ndim == 3:
            mean_shap = np.abs(sv).mean(axis=(0, 2))
        elif sv.ndim == 2:
            mean_shap = np.abs(sv).mean(axis=0)
        else:
            mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
        shap_importance = [
            {'feature': name, 'mean_abs_shap': _to_native_type(val)}
            for name, val in sorted(zip(feature_names, mean_shap),
                                    key=lambda x: x[1], reverse=True)
        ]

        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
        feats = [d['feature'] for d in shap_importance][::-1]
        vals = [d['mean_abs_shap'] for d in shap_importance][::-1]
        colors = sns.color_palette('husl', n_colors=len(feats))
        ax.barh(feats, vals, color=colors, edgecolor='none')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
        ax.grid(False)
        fig.subplots_adjust(left=0.20)
        shap_plot = _fig_to_base64(fig)

        return {'shap_importance': shap_importance, 'shap_plot': shap_plot, 'error': None}
    except Exception as e:
        return {'shap_importance': [], 'shap_plot': None, 'error': str(e)}


def compute_pdp(model, X_train: np.ndarray, feature_names: List[str],
                feature_importance: Optional[List[Dict]] = None,
                top_n: int = 6) -> Optional[str]:
    """Partial Dependence Plots for top-n features by importance"""
    try:
        if feature_importance:
            sorted_indices = [
                feature_names.index(f['feature'])
                for f in feature_importance
                if f['feature'] in feature_names
            ][:top_n]
        else:
            sorted_indices = list(range(min(top_n, len(feature_names))))

        n = len(sorted_indices)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = np.array(axes).reshape(-1) if n > 1 else [axes]

        for plot_idx, feat_idx in enumerate(sorted_indices):
            ax = axes_flat[plot_idx]
            pd_res = partial_dependence(model, X_train, [feat_idx], kind='average')
            grid_vals = pd_res.get('grid_values', pd_res.get('values', [None]))
            ax.plot(grid_vals[0], pd_res['average'][0],
                    color=sns.color_palette('husl', n_colors=1)[0], linewidth=2)
            ax.set_xlabel(feature_names[feat_idx], fontsize=10)
            ax.set_ylabel('Partial Dependence', fontsize=9)
            ax.set_title(f'PDP: {feature_names[feat_idx]}', fontsize=10, fontweight='bold')
            ax.grid(False)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle('Partial Dependence Plots (Top Features)', fontsize=13, fontweight='bold', y=1.01)
        fig.subplots_adjust(left=0.12, hspace=0.5, wspace=0.4)
        return _fig_to_base64(fig)
    except Exception:
        return None


def extract_tree_rules(model, feature_names: List[str], task_type: str,
                       class_names: Optional[List[str]] = None,
                       max_leaf_rules: int = 30) -> Optional[Dict]:
    """Extract decision rules from one representative tree"""
    try:
        sample_tree = model.estimators_[0]

        text_rules = export_text(
            sample_tree,
            feature_names=list(feature_names),
            max_depth=5
        )

        tree_ = sample_tree.tree_

        leaf_rules = []

        def traverse(node, conditions):
            if tree_.feature[node] == -2:  # leaf
                if task_type == 'classification':
                    class_idx = int(np.argmax(tree_.value[node][0]))
                    confidence = float(tree_.value[node][0][class_idx] / tree_.value[node][0].sum())
                    pred = class_names[class_idx] if class_names else str(class_idx)
                    leaf_rules.append({
                        'conditions': list(conditions),
                        'prediction': pred,
                        'confidence': round(confidence, 3),
                        'n_samples': int(tree_.n_node_samples[node])
                    })
                else:
                    leaf_rules.append({
                        'conditions': list(conditions),
                        'prediction': round(float(tree_.value[node][0][0]), 4),
                        'n_samples': int(tree_.n_node_samples[node])
                    })
                return
            feat = feature_names[tree_.feature[node]]
            thresh = round(float(tree_.threshold[node]), 4)
            traverse(tree_.children_left[node], conditions + [f'{feat} <= {thresh}'])
            traverse(tree_.children_right[node], conditions + [f'{feat} > {thresh}'])

        try:
            traverse(0, [])
        except RecursionError:
            pass

        leaf_rules.sort(key=lambda x: x['n_samples'], reverse=True)
        rules_truncated = len(leaf_rules) > max_leaf_rules
        leaf_rules = leaf_rules[:max_leaf_rules]

        return {
            'text_rules': text_rules,
            'leaf_rules': leaf_rules,
            'n_leaves': int(tree_.n_node_samples[0]),
            'rules_truncated': rules_truncated,
            'note': f'Rules extracted from 1 representative tree (of {len(model.estimators_)} total).'
        }
    except Exception:
        return None


def perform_cross_validation(X, y, params: dict, task_type: str, cv_folds: int) -> Dict[str, Any]:
    """Perform cross-validation"""
    max_features = params.get('max_features', 'sqrt')
    if max_features == 'None':
        max_features = None

    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'],
            max_features=max_features, bootstrap=params['bootstrap'],
            random_state=params['random_state'], n_jobs=-1
        )
        cv_target = y_encoded
        cv_task = 'classification'
    else:
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'],
            max_features=max_features, bootstrap=params['bootstrap'],
            random_state=params['random_state'], n_jobs=-1
        )
        cv_target = y
        cv_task = 'regression'

    # Shared CV (cv_strategy.py) — same StratifiedKFold(clf)/KFold(reg) behavior as
    # before, now centralized so time/group splits can be added in one place.
    return run_cv(model, X, cv_target, cv_task, cv_folds, params['random_state'])


def generate_feature_importance_plot(importance_data: List[Dict], top_n: int = 20) -> str:
    """Generate feature importance plot"""
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_data[:top_n]) * 0.4)))

    top_features = importance_data[:top_n]
    features = [d['feature'] for d in top_features][::-1]
    importances = [d['importance'] for d in top_features][::-1]

    colors = sns.color_palette('husl', n_colors=len(features))
    bars = ax.barh(features, importances, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(False)

    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{imp:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_confusion_matrix_plot(cm: List[List[int]], class_labels: List[str]) -> str:
    """Generate confusion matrix heatmap — pure matplotlib"""
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
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_roc_plot(roc_data: Dict) -> str:
    plot_data = {k: v for k, v in roc_data.items() if k != '__macro_auc__'}
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette('husl', n_colors=len(plot_data))
    for (label, data), color in zip(plot_data.items(), colors):
        ax.plot(data['fpr'], data['tpr'], color=color, linewidth=2,
                label=f'{label} (AUC = {data["auc"]:.3f})')
    if '__macro_auc__' in roc_data:
        ax.plot([], [], ' ', label=f'Macro AUC = {roc_data["__macro_auc__"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(False)
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.15)
    return _fig_to_base64(fig)


def generate_tree_count_plot(model, X_test, y_test, task_type: str) -> str:
    """Generate tree count vs performance plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    n_trees = len(model.estimators_)
    tree_counts = list(range(1, n_trees + 1, max(1, n_trees // 20)))
    if tree_counts[-1] != n_trees:
        tree_counts.append(n_trees)

    scores = []
    for n in tree_counts:
        if task_type == 'classification':
            proba = np.mean([est.predict_proba(X_test) for est in model.estimators_[:n]], axis=0)
            pred = np.argmax(proba, axis=1)
            scores.append(accuracy_score(y_test, pred))
        else:
            pred = np.mean([est.predict(X_test) for est in model.estimators_[:n]], axis=0)
            scores.append(r2_score(y_test, pred))

    color = sns.color_palette('husl', n_colors=1)[0]
    ax.plot(tree_counts, scores, color=color, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Number of Trees', fontsize=11)
    ylabel = 'Accuracy' if task_type == 'classification' else 'R² Score'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('Performance vs Number of Trees', fontsize=13, fontweight='bold')
    ax.grid(False)
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.15)
    return _fig_to_base64(fig)


def generate_actual_vs_predicted_plot(y_test, y_pred) -> str:
    """Generate actual vs predicted scatter plot"""
    fig, ax = plt.subplots(figsize=(7, 5))
    color = sns.color_palette('husl', n_colors=1)[0]
    ax.scatter(y_test, y_pred, alpha=0.5, color=color, s=30)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(False)
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.15)
    return _fig_to_base64(fig)


def generate_residual_plot(y_test, y_pred) -> str:
    """Generate residual plot"""
    fig, ax = plt.subplots(figsize=(7, 5))
    residuals = y_test - y_pred
    color = sns.color_palette('husl', n_colors=3)[1]
    ax.scatter(y_pred, residuals, alpha=0.5, color=color, s=30)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(False)
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.15)
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, task_type: str, feature_importance: List[Dict], params: dict) -> Dict[str, Any]:
    """Generate interpretation of Random Forest results"""
    key_insights = []

    if task_type == 'classification':
        accuracy = result['metrics']['accuracy']
        f1 = result['metrics']['f1_macro']

        if accuracy >= 0.9:
            status = 'positive'
            perf_desc = 'Excellent classification performance'
        elif accuracy >= 0.7:
            status = 'neutral'
            perf_desc = 'Good classification performance'
        else:
            status = 'warning'
            perf_desc = 'Model may need improvement'

        key_insights.append({
            'title': 'Classification Performance',
            'description': f'{perf_desc}. Accuracy: {accuracy:.1%}, F1-macro: {f1:.3f}',
            'status': status
        })

        if 'auc' in result['metrics']:
            auc_val = result['metrics']['auc']
            key_insights.append({
                'title': 'AUC Score',
                'description': f'Area Under ROC Curve: {auc_val:.3f}. {"Excellent discrimination" if auc_val > 0.9 else "Good discrimination" if auc_val > 0.7 else "Fair discrimination"}',
                'status': 'positive' if auc_val > 0.8 else 'neutral'
            })
    else:
        r2 = result['metrics']['r2']
        rmse = result['metrics']['rmse']

        if r2 >= 0.8:
            status = 'positive'
            perf_desc = 'Excellent fit'
        elif r2 >= 0.5:
            status = 'neutral'
            perf_desc = 'Moderate fit'
        else:
            status = 'warning'
            perf_desc = 'Weak fit'

        key_insights.append({
            'title': 'Regression Performance',
            'description': f'{perf_desc}. R² = {r2:.3f}, RMSE = {rmse:.4f}',
            'status': status
        })

    if 'oob_score' in result['metrics']:
        oob = result['metrics']['oob_score']
        key_insights.append({
            'title': 'Out-of-Bag Score',
            'description': f'OOB estimate: {oob:.3f}. This is an unbiased estimate of generalization error.',
            'status': 'positive' if oob > 0.8 else 'neutral'
        })

    top_features = feature_importance[:3]
    feature_str = ', '.join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
    key_insights.append({
        'title': 'Key Predictors',
        'description': f'Top features: {feature_str}',
        'status': 'neutral'
    })

    key_insights.append({
        'title': 'Ensemble Size',
        'description': f'{params["n_estimators"]} trees in the forest. {"Good ensemble size for stability." if params["n_estimators"] >= 100 else "Consider increasing trees for more stable predictions."}',
        'status': 'positive' if params["n_estimators"] >= 100 else 'neutral'
    })

    return {
        'key_insights': key_insights,
        'recommendation': 'Random Forest model trained successfully. The ensemble approach provides robust predictions with built-in feature importance.'
    }


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target_col = payload.get('target_col') or payload.get('target')
        feature_cols = payload.get('feature_cols') or payload.get('features')
        task_type = payload.get('task_type', 'auto')
        test_size = float(payload.get('test_size', 0.2))

        n_estimators = int(payload.get('n_estimators', 100))
        max_depth_raw = payload.get('max_depth')
        max_depth = int(max_depth_raw) if max_depth_raw not in (None, '', 'None') else None
        min_samples_split = int(payload.get('min_samples_split', 2))
        min_samples_leaf = int(payload.get('min_samples_leaf', 1))
        max_features = payload.get('max_features', 'sqrt')
        bootstrap = bool(payload.get('bootstrap', True))
        oob_score_flag = bool(payload.get('oob_score', True))
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

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 50:
            raise ValueError("At least 50 valid samples required.")

        if task_type == 'auto':
            task_type = detect_task_type(y)

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'oob_score': oob_score_flag,
            'random_state': random_state,
        }

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if task_type == 'classification' else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        if task_type == 'classification':
            result = train_rf_classifier(X_train, X_test, y_train, y_test, params)
            le = result['label_encoder']
            y_test_encoded = le.transform(y_test)
        else:
            result = train_rf_regressor(X_train, X_test, y_train, y_test, params)
            y_test_encoded = y_test

        model = result['model']

        feature_importance = get_feature_importance(model, feature_cols)

        if task_type == 'classification':
            y_test_for_perm = result['label_encoder'].transform(y_test)
        else:
            y_test_for_perm = y_test.values if hasattr(y_test, 'values') else y_test
        perm_importance = compute_permutation_importance(
            model, X_test.values, y_test_for_perm, feature_cols
        )

        cv_result = perform_cross_validation(X, y, params, task_type, cv_folds)

        shap_result = compute_shap(model, X_train.values, X_test.values, feature_cols)

        pdp_plot = compute_pdp(model, X_train.values, feature_cols, feature_importance, top_n=6)

        class_names = result.get('class_labels') if task_type == 'classification' else None
        tree_rules = extract_tree_rules(model, feature_cols, task_type, class_names)

        importance_plot = generate_feature_importance_plot(feature_importance)
        tree_count_plot = generate_tree_count_plot(model, X_test, y_test_encoded, task_type)

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            actual_vs_predicted_plot = None
            residual_plot = None
        else:
            cm_plot = None
            roc_plot = None
            actual_vs_predicted_plot = generate_actual_vs_predicted_plot(result['y_test'], result['y_pred'])
            residual_plot = generate_residual_plot(result['y_test'], result['y_pred'])

        interpretation = generate_interpretation(result, task_type, feature_importance, params)

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
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'] if params['max_depth'] else 'None',
                'min_samples_split': params['min_samples_split'],
                'min_samples_leaf': params['min_samples_leaf'],
                'max_features': params['max_features'],
                'bootstrap': params['bootstrap'],
                'test_size': test_size
            },
            'metrics': result['metrics'],
            'feature_importance': feature_importance,
            'perm_importance': perm_importance,
            'cv_results': cv_result,
            'importance_plot': importance_plot,
            'tree_count_plot': tree_count_plot,
            'shap_plot': shap_result.get('shap_plot'),
            'shap_importance': shap_result.get('shap_importance'),
            'shap_error': shap_result.get('error'),
            'pdp_plot': pdp_plot,
            'tree_rules': tree_rules,
            'interpretation': interpretation
        }

        if task_type == 'classification':
            response['per_class_metrics'] = result['per_class_metrics']
            response['confusion_matrix'] = result['confusion_matrix']
            response['class_labels'] = result['class_labels']
            response['cm_plot'] = cm_plot
            response['roc_plot'] = roc_plot
        else:
            response['actual_vs_predicted_plot'] = actual_vs_predicted_plot
            response['residual_plot'] = residual_plot

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
