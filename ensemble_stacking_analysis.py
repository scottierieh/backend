"""
Voting & Stacking Ensemble — CLI script
Combine multiple base learners via hard/soft voting or a meta-learner stack.
Ported from scottierieh/backend's api/ensemble_voting_stacking.py (FastAPI router) to the
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
from cv_strategy import run_cv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
)

# Optional boosting base learners — so an ensemble can blend the actual Auto Compare
# winners (often XGBoost/LightGBM/GBM), not just the 5 sklearn defaults. Guarded so the
# script still imports if a lib is absent; unavailable models are simply skipped by
# build_estimators' registry filter.
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
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


BASE_ESTIMATORS = {
    'classification': {
        'logistic_regression': lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
        'decision_tree': lambda rs: DecisionTreeClassifier(max_depth=5, random_state=rs),
        'random_forest': lambda rs: RandomForestClassifier(n_estimators=100, random_state=rs),
        'gbm': lambda rs: GradientBoostingClassifier(random_state=rs),
        'svm': lambda rs: SVC(probability=True, random_state=rs),
        'knn': lambda rs: KNeighborsClassifier(n_neighbors=5),
    },
    'regression': {
        'ridge': lambda rs: Ridge(random_state=rs),
        'decision_tree': lambda rs: DecisionTreeRegressor(max_depth=5, random_state=rs),
        'random_forest': lambda rs: RandomForestRegressor(n_estimators=100, random_state=rs),
        'gbm': lambda rs: GradientBoostingRegressor(random_state=rs),
        'svm': lambda rs: SVR(),
        'knn': lambda rs: KNeighborsRegressor(n_neighbors=5),
    }
}

# Boosting learners registered only if their lib imported — keeps blends able to
# include the common Auto Compare winners without making them a hard dependency.
if _HAS_XGB:
    BASE_ESTIMATORS['classification']['xgboost'] = lambda rs: XGBClassifier(random_state=rs, n_jobs=-1, verbosity=0, eval_metric='logloss')
    BASE_ESTIMATORS['regression']['xgboost'] = lambda rs: XGBRegressor(random_state=rs, n_jobs=-1, verbosity=0)
if _HAS_LGBM:
    BASE_ESTIMATORS['classification']['lightgbm'] = lambda rs: LGBMClassifier(random_state=rs, n_jobs=-1, verbose=-1)
    BASE_ESTIMATORS['regression']['lightgbm'] = lambda rs: LGBMRegressor(random_state=rs, n_jobs=-1, verbose=-1)

DEFAULT_ESTIMATORS = {
    'classification': ['logistic_regression', 'decision_tree', 'random_forest'],
    'regression': ['ridge', 'decision_tree', 'random_forest']
}


def build_estimators(task_type: str, names: Optional[List[str]], random_state: int):
    registry = BASE_ESTIMATORS[task_type]
    chosen = [n for n in (names or DEFAULT_ESTIMATORS[task_type]) if n in registry]
    if not chosen:
        chosen = DEFAULT_ESTIMATORS[task_type]
    return [(name, registry[name](random_state)) for name in chosen]


def _to_native_scalar_list(arr):
    return [_to_native_type(x) for x in arr]


def train_ensemble(X_train, X_test, y_train, y_test, task_type: str, params: dict, feature_names: List[str]) -> Dict[str, Any]:
    estimators = build_estimators(task_type, params['base_estimators'], params['random_state'])

    individual_scores = {}
    for name, est in estimators:
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        if task_type == 'classification':
            individual_scores[name] = _to_native_type(accuracy_score(y_test, pred))
        else:
            individual_scores[name] = _to_native_type(r2_score(y_test, pred))

    if params['ensemble_method'] == 'stacking':
        final_est_name = params['final_estimator']
        registry = BASE_ESTIMATORS[task_type]
        final_estimator = registry.get(final_est_name, registry[DEFAULT_ESTIMATORS[task_type][0]])(params['random_state'])
        if task_type == 'classification':
            model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5, n_jobs=-1)
        else:
            model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5, n_jobs=-1)
        model_label = f"Stacking Ensemble (meta={final_est_name})"
    else:
        if task_type == 'classification':
            model = VotingClassifier(estimators=estimators, voting=params['voting_type'])
        else:
            model = VotingRegressor(estimators=estimators)
        model_label = f"Voting Ensemble ({params['voting_type']})" if task_type == 'classification' else "Voting Ensemble (average)"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    individual_scores[model_label] = _to_native_type(
        accuracy_score(y_test, y_pred) if task_type == 'classification' else r2_score(y_test, y_pred)
    )

    result = {'model': model, 'model_label': model_label, 'individual_scores': individual_scores, 'base_estimator_names': [n for n, _ in estimators]}

    if task_type == 'classification':
        le = LabelEncoder().fit(y_train)
        y_test_encoded = le.transform(y_test)
        y_pred_encoded = le.transform(y_pred)

        metrics = {
            'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred_encoded)),
            'precision_macro': _to_native_type(precision_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)),
            'recall_macro': _to_native_type(recall_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)),
            'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred_encoded, average='macro', zero_division=0)),
        }
        class_report = classification_report(y_test_encoded, y_pred_encoded, target_names=[str(c) for c in le.classes_], output_dict=True)
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
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)

        roc_data = {}
        n_classes = len(le.classes_)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_data['binary'] = {'fpr': _to_native_scalar_list(fpr), 'tpr': _to_native_scalar_list(tpr), 'auc': _to_native_type(roc_auc)}
                metrics['auc'] = _to_native_type(roc_auc)
            else:
                for i, cls in enumerate(le.classes_):
                    y_binary = (y_test_encoded == i).astype(int)
                    fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_data[str(cls)] = {'fpr': _to_native_scalar_list(fpr), 'tpr': _to_native_scalar_list(tpr), 'auc': _to_native_type(roc_auc)}
                macro_auc = _compute_multiclass_auc(y_test_encoded, y_pred_proba)
                if macro_auc is not None:
                    metrics['auc'] = macro_auc

        result.update({
            'metrics': metrics, 'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(), 'class_labels': [str(c) for c in le.classes_], 'roc_data': roc_data,
            'label_encoder': le, 'y_test_encoded': y_test_encoded, 'y_pred': y_pred_encoded, 'y_pred_proba': y_pred_proba
        })
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'mse': _to_native_type(mse), 'rmse': _to_native_type(np.sqrt(mse)),
            'mae': _to_native_type(mean_absolute_error(y_test, y_pred)), 'r2': _to_native_type(r2_score(y_test, y_pred)),
        }
        result.update({'metrics': metrics, 'y_test': y_test.values if hasattr(y_test, 'values') else y_test, 'y_pred': y_pred})

    return result


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


def _perm_to_feature_importance(perm_importance):
    """Reshape perm_importance's {feature, importance_mean, importance_std} into the
    standard {feature, importance, importance_pct} shape used by tree/boosting models,
    so this model also shows up in Model Lab's cross-model importance comparison table."""
    if not perm_importance:
        return []
    pos = [max(p.get('importance_mean', 0.0), 0.0) for p in perm_importance]
    total = sum(pos) or 1.0
    return [
        {
            'feature': p['feature'],
            'importance': p.get('importance_mean', 0.0),
            'importance_pct': _to_native_type(v / total * 100),
        }
        for p, v in zip(perm_importance, pos)
    ]


def compute_shap(model, X_test: np.ndarray, feature_names: List[str], task_type: str,
                  max_background: int = 50, max_samples: int = 100) -> Dict:
    """Model-agnostic SHAP (Permutation explainer) — a voting/stacking ensemble mixes
    heterogeneous base learners, so there's no single TreeExplainer that applies."""
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

        predict_fn = model.predict_proba if (task_type == 'classification' and hasattr(model, 'predict_proba')) else model.predict
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


def perform_cross_validation(X, y, task_type: str, params: dict, cv_folds: int) -> Dict[str, Any]:
    estimators = build_estimators(task_type, params['base_estimators'], params['random_state'])
    if params['ensemble_method'] == 'stacking':
        registry = BASE_ESTIMATORS[task_type]
        final_estimator = registry.get(params['final_estimator'], registry[DEFAULT_ESTIMATORS[task_type][0]])(params['random_state'])
        model = (StackingClassifier if task_type == 'classification' else StackingRegressor)(estimators=estimators, final_estimator=final_estimator, cv=5)
    else:
        if task_type == 'classification':
            model = VotingClassifier(estimators=estimators, voting=params['voting_type'])
        else:
            model = VotingRegressor(estimators=estimators)

    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        cv_target, cv_task = y_encoded, 'classification'
    else:
        cv_target, cv_task = y, 'regression'

    return run_cv(model, X, cv_target, cv_task, cv_folds, params['random_state'])


def generate_comparison_plot(individual_scores: Dict[str, float], model_label: str, task_type: str) -> str:
    fig, ax = plt.subplots(figsize=(10, max(4, len(individual_scores) * 0.6)))
    names = list(individual_scores.keys())
    scores = [individual_scores[n] for n in names]
    order = np.argsort(scores)
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]
    colors = ['#f59e0b' if n == model_label else '#3b82f6' for n in names]
    ax.barh(names, scores, color=colors, edgecolor='black', alpha=0.85)
    for i, s in enumerate(scores):
        ax.text(s, i, f' {s:.3f}', va='center', fontsize=10)
    ax.set_xlabel('Accuracy' if task_type == 'classification' else 'R²', fontsize=11)
    ax.set_title('Base Models vs. Ensemble Performance', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
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


def generate_interpretation(result: Dict, task_type: str, params: dict) -> Dict[str, Any]:
    key_insights = []
    model_label = result['model_label']
    individual_scores = result['individual_scores']
    ensemble_score = individual_scores[model_label]

    base_only = {k: v for k, v in individual_scores.items() if k != model_label}
    best_base_name = max(base_only, key=base_only.get) if base_only else None
    best_base_score = base_only.get(best_base_name) if best_base_name else None

    metric_name = 'Accuracy' if task_type == 'classification' else 'R²'
    status = 'positive' if ensemble_score >= 0.8 else 'neutral' if ensemble_score >= 0.6 else 'warning'
    key_insights.append({
        'title': f'{model_label} Performance',
        'description': f'Ensemble {metric_name}: {ensemble_score:.3f}',
        'status': status
    })

    if best_base_name:
        diff = ensemble_score - best_base_score
        if diff > 0.01:
            key_insights.append({
                'title': 'Ensemble Improves on Best Base Model',
                'description': f'The ensemble outperforms the best individual model ({best_base_name}: {best_base_score:.3f}) by {diff:.3f} — the base learners are contributing complementary information.',
                'status': 'positive'
            })
        elif diff < -0.01:
            key_insights.append({
                'title': 'Ensemble Underperforms Best Base Model',
                'description': f'The ensemble underperforms the best individual model ({best_base_name}: {best_base_score:.3f}) by {abs(diff):.3f}. Consider dropping weaker base learners or using weighted voting.',
                'status': 'warning'
            })
        else:
            key_insights.append({
                'title': 'Comparable to Best Base Model',
                'description': 'The ensemble performs about the same as the best individual model, which can indicate correlated base learners.',
                'status': 'neutral'
            })

    key_insights.append({
        'title': 'Base Estimators',
        'description': f"Combined via {params['ensemble_method']}: {', '.join(result['base_estimator_names'])}",
        'status': 'neutral'
    })

    if params['ensemble_method'] == 'voting' and task_type == 'classification' and params['voting_type'] == 'hard':
        key_insights.append({
            'title': 'Hard vs. Soft Voting',
            'description': 'Hard voting only counts predicted labels. Soft voting (averaging predicted probabilities) usually performs better when base models are well-calibrated.',
            'status': 'neutral'
        })

    recommendation = (
        'Stacking with a well-chosen meta-learner and diverse base models typically outperforms voting, '
        'at the cost of longer training (nested cross-validation).'
        if params['ensemble_method'] == 'voting' else
        'Ensure the meta-learner\'s CV folds are large enough to avoid leakage. Try a simpler meta-learner '
        'if the stack overfits, or a more flexible one if it underfits.'
    )

    return {'key_insights': key_insights, 'recommendation': recommendation}


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
            y_pred_proba = result.get('y_pred_proba')
            n = min(n_examples, len(y_test_enc))
            idx = rng.choice(len(y_test_enc), size=n, replace=False)
            examples = []
            for i in idx:
                actual_label = str(le.inverse_transform([y_test_enc[i]])[0])
                predicted_label = str(le.inverse_transform([y_pred_enc[i]])[0])
                confidence = _to_native_type(float(y_pred_proba[i].max())) if y_pred_proba is not None else None
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
        ensemble_method = payload.get('ensemble_method', 'voting')
        voting_type = payload.get('voting_type', 'soft')
        base_estimators = payload.get('base_estimators')
        final_estimator = payload.get('final_estimator', 'logistic_regression')
        cv_folds = int(payload.get('cv_folds', 5))
        random_state = int(payload.get('random_state', 42))
        scale_features = bool(payload.get('scale_features', True))

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

        if task_type == 'auto':
            task_type = detect_task_type(y)
        if ensemble_method not in ('voting', 'stacking'):
            raise ValueError("ensemble_method must be 'voting' or 'stacking'.")

        X_array = X.values
        if scale_features:
            X_array = StandardScaler().fit_transform(X_array)

        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y, test_size=test_size, random_state=random_state,
            stratify=y if task_type == 'classification' else None
        )

        params = {
            'ensemble_method': ensemble_method,
            'voting_type': voting_type,
            'base_estimators': base_estimators,
            'final_estimator': final_estimator,
            'random_state': random_state,
        }

        result = train_ensemble(X_train, X_test, y_train, y_test, task_type, params, feature_cols)
        perm_importance = compute_permutation_importance(result['model'], X_test, y_test, feature_cols)
        shap_result = compute_shap(result['model'], X_test, feature_cols, task_type)
        cv_result = perform_cross_validation(X_array, y, task_type, params, cv_folds)

        comparison_plot = generate_comparison_plot(result['individual_scores'], result['model_label'], task_type)

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data'])
            regression_plot = None
        else:
            cm_plot = None
            roc_plot = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])

        interpretation = generate_interpretation(result, task_type, params)
        prediction_examples = generate_prediction_examples(result, task_type)

        try:
            from guardrails import compute_guardrails
            guardrails = compute_guardrails(X, y, feature_cols, task_type, result['metrics'])
        except Exception:
            guardrails = []

        response = {
            'guardrails': guardrails,
            'task_type': task_type,
            'ensemble_method': ensemble_method,
            'model_label': result['model_label'],
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'base_estimators': result['base_estimator_names'],
            'individual_scores': result['individual_scores'],
            'metrics': result['metrics'],
            'perm_importance': perm_importance,
            'feature_importance': _perm_to_feature_importance(perm_importance),
            'shap_importance': shap_result.get('shap_importance'),
            'shap_plot': shap_result.get('shap_plot'),
            'shap_error': shap_result.get('error'),
            'cv_results': cv_result,
            'comparison_plot': comparison_plot,
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
