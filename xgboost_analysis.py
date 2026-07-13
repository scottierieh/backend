"""
XGBoost Classification and Regression — CLI script
Gradient boosting with tree-based learners. Ported from a standalone FastAPI
router to the stdin/stdout CLI contract used by src/backend/main.py's generic
script runner — same contract as lightgbm_analysis.py / catboost_analysis.py.
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from cv_strategy import run_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
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
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    elif len(y.unique()) <= 10 or unique_ratio < 0.05:
        return 'classification'
    else:
        return 'regression'


def train_xgboost_classifier(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(le.classes_)

    if n_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'

    model = xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        objective=objective,
        eval_metric=eval_metric,
        random_state=params['random_state'],
        n_jobs=-1
    )

    eval_set = [(X_train, y_train_encoded), (X_test, y_test_encoded)]
    model.fit(
        X_train, y_train_encoded,
        eval_set=eval_set,
        verbose=False
    )

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred)),
        'precision_macro': _to_native_type(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'recall_macro': _to_native_type(recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0))
    }

    class_report = classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
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

    train_history = {
        'train': model.evals_result()['validation_0'][eval_metric],
        'test': model.evals_result()['validation_1'][eval_metric]
    }

    return {
        'model': model,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'train_history': train_history,
        'label_encoder': le,
        # Tier-B diagnostics (model_diagnostics.py)
        'bootstrap_ci': bootstrap_ci(y_test_encoded, y_pred, 'classification'),
        'calibration': calibration_curve(y_test_encoded, y_pred_proba),
        'pr_curve': pr_curve(y_test_encoded, y_pred_proba),
        'error_examples': error_examples(
            le.inverse_transform(y_test_encoded), le.inverse_transform(y_pred), y_pred_proba,
            list(X_test.columns) if hasattr(X_test, 'columns') else None, X_test,
        ),
    }


def train_xgboost_regressor(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        objective='reg:squarederror',
        random_state=params['random_state'],
        n_jobs=-1
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

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

    train_history = {
        'train': model.evals_result()['validation_0']['rmse'],
        'test': model.evals_result()['validation_1']['rmse']
    }

    return {
        'model': model,
        'metrics': metrics,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred,
        'train_history': train_history,
        'bootstrap_ci': bootstrap_ci(y_test, y_pred, 'regression'),
    }


def get_feature_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
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
    model, X_test: np.ndarray, y_test,
    feature_names: List[str], n_repeats: int = 10, random_state: int = 42
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
                'importance_mean': _to_native_type(mean),
                'importance_std': _to_native_type(std),
            })
        result.sort(key=lambda x: x['importance_mean'], reverse=True)
        for i, item in enumerate(result):
            item['rank'] = i + 1
        return result
    except Exception:
        return []


def compute_shap(model, X_test: np.ndarray, feature_names: List[str]) -> Dict:
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
            mean_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)

        shap_importance = [
            {'feature': name, 'mean_abs_shap': _to_native_type(val)}
            for name, val in sorted(zip(feature_names, mean_shap),
                                    key=lambda x: x[1], reverse=True)
        ]

        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
        feats = [d['feature'] for d in shap_importance][::-1]
        vals  = [d['mean_abs_shap'] for d in shap_importance][::-1]
        ax.barh(feats, vals, color='#f59e0b', edgecolor='none')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
        ax.set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, axis='x')
        fig.subplots_adjust(left=0.20)
        shap_plot = _fig_to_base64(fig)

        return {'shap_importance': shap_importance, 'shap_plot': shap_plot, 'error': None}
    except Exception as e:
        return {'shap_importance': [], 'shap_plot': None, 'error': str(e)}


def compute_pdp(model, X_train: np.ndarray, feature_names: List[str],
                feature_importance: Optional[List[Dict]] = None,
                top_n: int = 6) -> Optional[str]:
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
            ax.plot(grid_vals[0], pd_res['average'][0], color='#2563eb', linewidth=2)
            ax.set_xlabel(feature_names[feat_idx], fontsize=10)
            ax.set_ylabel('Partial Dependence', fontsize=9)
            ax.set_title(f'PDP: {feature_names[feat_idx]}', fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)

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
    try:
        dump = model.get_booster().get_dump(with_stats=False)
        if not dump:
            return None
        text_rules = dump[0]

        lines = text_rules.strip().split('\n')
        leaf_rules = []
        stack: List[tuple] = []

        for line in lines:
            stripped = line.lstrip('\t')
            depth = len(line) - len(stripped)
            stack = [(d, c) for d, c in stack if d < depth]

            if 'leaf' in stripped:
                val_str = stripped.split('leaf=')[-1].split(',')[0].strip()
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
                conditions = [c for _, c in stack]
                if task_type == 'classification':
                    leaf_rules.append({
                        'conditions': conditions,
                        'prediction': f'score={val:.4f}',
                        'n_samples': len(conditions)
                    })
                else:
                    leaf_rules.append({
                        'conditions': conditions,
                        'prediction': round(val, 4),
                        'n_samples': len(conditions)
                    })
            else:
                import re
                m = re.search(r'\[(.+?)<(.+?)\]', stripped)
                if m:
                    feat_raw, thresh = m.group(1), m.group(2)
                    fn_m = re.match(r'f(\d+)', feat_raw)
                    if fn_m:
                        idx = int(fn_m.group(1))
                        fname = feature_names[idx] if idx < len(feature_names) else feat_raw
                    else:
                        fname = feat_raw
                    stack.append((depth, f'{fname} < {float(thresh):.4f}'))

        leaf_rules = leaf_rules[:max_leaf_rules]

        return {
            'text_rules': text_rules,
            'leaf_rules': leaf_rules,
            'n_leaves': len([l for l in lines if 'leaf' in l]),
            'rules_truncated': len(leaf_rules) == max_leaf_rules,
            'note': f'Rules extracted from tree 0 of {model.n_estimators} total trees.'
        }
    except Exception:
        return None


def perform_cross_validation(X, y, params: dict, task_type: str, cv_folds: int) -> Dict[str, Any]:
    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        objective = 'binary:logistic' if n_classes == 2 else 'multi:softprob'
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            learning_rate=params['learning_rate'], subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'], objective=objective,
            random_state=params['random_state'], n_jobs=-1
        )
        min_class_count = int(np.min(np.bincount(y_encoded)))
        cv_folds = max(2, min(cv_folds, min_class_count))
        cv_target, cv_task = y_encoded, 'classification'
    else:
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            learning_rate=params['learning_rate'], subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'], objective='reg:squarederror',
            random_state=params['random_state'], n_jobs=-1
        )
        cv_target, cv_task = y, 'regression'

    return run_cv(model, X, cv_target, cv_task, cv_folds, params['random_state'])


def generate_feature_importance_plot(importance_data: List[Dict], top_n: int = 20) -> str:
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_data[:top_n]) * 0.4)))

    top_features = importance_data[:top_n]
    features = [d['feature'] for d in top_features][::-1]
    importances = [d['importance'] for d in top_features][::-1]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('XGBoost Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9)

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
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_roc_plot(roc_data: Dict) -> str:
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


def generate_learning_curve_plot(train_history: Dict, task_type: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_history['train']) + 1)

    ax.plot(epochs, train_history['train'], 'b-', linewidth=2, label='Train')
    ax.plot(epochs, train_history['test'], 'r-', linewidth=2, label='Test')

    ax.set_xlabel('Boosting Round', fontsize=11)
    ylabel = 'Log Loss' if task_type == 'classification' else 'RMSE'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('Training History', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_regression_plot(y_test, y_pred) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, color='#3b82f6', s=30)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#22c55e', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, task_type: str, feature_importance: List[Dict], language: str = 'en') -> Dict[str, Any]:
    key_insights = []

    if language == 'ko':
        if task_type == 'classification':
            accuracy = result['metrics']['accuracy']
            f1 = result['metrics']['f1_macro']
            if accuracy >= 0.9:
                status = 'positive'
                perf_desc = '분류 성능이 우수합니다'
            elif accuracy >= 0.7:
                status = 'neutral'
                perf_desc = '분류 성능이 양호합니다'
            else:
                status = 'warning'
                perf_desc = '모델 개선이 필요할 수 있습니다'
            key_insights.append({
                'title': '분류 성능',
                'description': f'{perf_desc}. 정확도: {accuracy:.1%}, F1-macro: {f1:.3f}',
                'status': status
            })
            if 'auc' in result['metrics']:
                auc_val = result['metrics']['auc']
                auc_label = "우수한" if auc_val > 0.9 else "양호한" if auc_val > 0.7 else "보통 수준의"
                key_insights.append({
                    'title': 'AUC 점수',
                    'description': f'ROC-AUC: {auc_val:.3f}. {auc_label} 판별력.',
                    'status': 'positive' if auc_val > 0.8 else 'neutral'
                })
        else:
            r2 = result['metrics']['r2']
            rmse = result['metrics']['rmse']
            if r2 >= 0.8:
                status = 'positive'
                perf_desc = '적합도가 우수합니다'
            elif r2 >= 0.5:
                status = 'neutral'
                perf_desc = '보통 수준의 적합도입니다'
            else:
                status = 'warning'
                perf_desc = '적합도가 낮습니다'
            key_insights.append({
                'title': '회귀 성능',
                'description': f'{perf_desc}. R² = {r2:.3f}, RMSE = {rmse:.4f}',
                'status': status
            })

        top_features = feature_importance[:3]
        feature_str = ', '.join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
        key_insights.append({
            'title': '주요 예측 변수',
            'description': f'상위 피처: {feature_str}',
            'status': 'neutral'
        })

        top3_importance = sum(f['importance'] for f in feature_importance[:3])
        if top3_importance > 0.7:
            key_insights.append({
                'title': '피처 집중도',
                'description': f'상위 3개 피처가 중요도의 {top3_importance:.1%}를 차지합니다. 모델이 소수의 피처에 크게 의존합니다.',
                'status': 'neutral'
            })

        return {
            'key_insights': key_insights,
            'recommendation': 'XGBoost 모델이 성공적으로 학습되었습니다. 추가 성능 향상을 위해 하이퍼파라미터 튜닝을 고려하세요.'
        }

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

    top_features = feature_importance[:3]
    feature_str = ', '.join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
    key_insights.append({
        'title': 'Key Predictors',
        'description': f'Top features: {feature_str}',
        'status': 'neutral'
    })

    top3_importance = sum(f['importance'] for f in feature_importance[:3])
    if top3_importance > 0.7:
        key_insights.append({
            'title': 'Feature Concentration',
            'description': f'Top 3 features account for {top3_importance:.1%} of importance. Model relies heavily on few features.',
            'status': 'neutral'
        })

    return {
        'key_insights': key_insights,
        'recommendation': 'XGBoost model trained successfully. Consider hyperparameter tuning for further improvement.'
    }


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target_col = payload.get('target_col')
        feature_cols = payload.get('feature_cols') or []
        task_type = payload.get('task_type', 'auto')
        test_size = float(payload.get('test_size', 0.2))
        random_state = int(payload.get('random_state', 42))
        cv_folds = int(payload.get('cv_folds', 5))
        language = payload.get('language', 'en')

        if not data:
            raise ValueError("Data not provided.")

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
            'n_estimators': int(payload.get('n_estimators', 100)),
            'max_depth': int(payload.get('max_depth', 6)),
            'learning_rate': float(payload.get('learning_rate', 0.1)),
            'subsample': float(payload.get('subsample', 0.8)),
            'colsample_bytree': float(payload.get('colsample_bytree', 0.8)),
            'min_child_weight': int(payload.get('min_child_weight', 1)),
            'gamma': float(payload.get('gamma', 0)),
            'reg_alpha': float(payload.get('reg_alpha', 0)),
            'reg_lambda': float(payload.get('reg_lambda', 1)),
            'random_state': random_state
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if task_type == 'classification' else None
        )

        if task_type == 'classification':
            result = train_xgboost_classifier(X_train, X_test, y_train, y_test, params)
        else:
            result = train_xgboost_regressor(X_train, X_test, y_train, y_test, params)

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

        shap_result = compute_shap(model, X_test.values, feature_cols)

        pdp_plot = compute_pdp(model, X_train.values, feature_cols, feature_importance, top_n=6)

        class_names = result.get('class_labels') if task_type == 'classification' else None
        tree_rules = extract_tree_rules(model, feature_cols, task_type, class_names)

        importance_plot = generate_feature_importance_plot(feature_importance)
        learning_plot = generate_learning_curve_plot(result['train_history'], task_type)

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            regression_plot = None
        else:
            cm_plot = None
            roc_plot = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])

        interpretation = generate_interpretation(result, task_type, feature_importance, language=language)

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
            'parameters': params,
            'metrics': result['metrics'],
            'feature_importance': feature_importance,
            'perm_importance': perm_importance,
            'cv_results': cv_result,
            'importance_plot': importance_plot,
            'learning_plot': learning_plot,
            'shap_plot': shap_result.get('shap_plot'),
            'shap_importance': shap_result.get('shap_importance'),
            'shap_error': shap_result.get('error'),
            'pdp_plot': pdp_plot,
            'tree_rules': tree_rules,
            'interpretation': interpretation,
            'bootstrap_ci': result.get('bootstrap_ci'),
        }

        if task_type == 'classification':
            response['per_class_metrics'] = result['per_class_metrics']
            response['confusion_matrix'] = result['confusion_matrix']
            response['class_labels'] = result['class_labels']
            response['cm_plot'] = cm_plot
            response['roc_plot'] = roc_plot
            response['calibration'] = result.get('calibration')
            response['pr_curve'] = result.get('pr_curve')
            response['error_examples'] = result.get('error_examples')
        else:
            response['regression_plot'] = regression_plot

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
