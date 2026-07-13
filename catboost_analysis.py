"""
CatBoost Classification and Regression — CLI script
Ordered boosting with native categorical feature support (no manual encoding
needed). Ported from scottierieh/backend's api/catboost_analysis.py (FastAPI
router) to the stdin/stdout CLI contract used by src/backend/main.py's
generic script runner.
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
from cv_strategy import make_cv_splitter
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
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
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    elif len(y.unique()) <= 10 or unique_ratio < 0.05:
        return 'classification'
    else:
        return 'regression'


def _common_params(params: dict) -> dict:
    return dict(
        iterations=params['iterations'],
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_seed=params['random_state'],
        verbose=False,
        allow_writing_files=False
    )


def train_catboost_classifier(X_train, X_test, y_train, y_test, params: dict, cat_features: List[int]) -> Dict[str, Any]:
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    n_classes = len(le.classes_)

    model = CatBoostClassifier(**_common_params(params))

    train_pool = Pool(X_train, y_train_encoded, cat_features=cat_features)
    test_pool = Pool(X_test, y_test_encoded, cat_features=cat_features)

    fit_kwargs = dict(eval_set=test_pool, use_best_model=bool(params.get('early_stopping_rounds')))
    if params.get('early_stopping_rounds'):
        fit_kwargs['early_stopping_rounds'] = params['early_stopping_rounds']
    model.fit(train_pool, **fit_kwargs)

    y_pred = model.predict(test_pool).ravel().astype(int)
    y_pred_proba = model.predict_proba(test_pool)

    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred)),
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
        macro_auc = _compute_multiclass_auc(y_test_encoded, y_pred_proba)
        if macro_auc is not None:
            metrics['auc'] = macro_auc

    evals_result = model.get_evals_result()
    metric_name = list(evals_result.get('learn', {}).keys())[0] if evals_result.get('learn') else None
    train_history = None
    if metric_name and 'validation' in evals_result:
        valid_key = list(evals_result['validation'].keys())[0]
        train_history = {'train': evals_result['learn'][metric_name], 'test': evals_result['validation'][valid_key]}

    return {
        'model': model, 'metrics': metrics, 'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(), 'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data, 'train_history': train_history, 'eval_metric': metric_name,
        'label_encoder': le, 'train_pool': train_pool, 'test_pool': test_pool,
        'best_iteration': int(model.get_best_iteration() or params['iterations']),
        'y_test_encoded': y_test_encoded, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba
    }


def train_catboost_regressor(X_train, X_test, y_train, y_test, params: dict, cat_features: List[int]) -> Dict[str, Any]:
    model = CatBoostRegressor(**_common_params(params), loss_function='RMSE')

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    fit_kwargs = dict(eval_set=test_pool, use_best_model=bool(params.get('early_stopping_rounds')))
    if params.get('early_stopping_rounds'):
        fit_kwargs['early_stopping_rounds'] = params['early_stopping_rounds']
    model.fit(train_pool, **fit_kwargs)

    y_pred = model.predict(test_pool)
    y_train_pred = model.predict(train_pool)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        'mse': _to_native_type(mse),
        'rmse': _to_native_type(np.sqrt(mse)),
        'mae': _to_native_type(mean_absolute_error(y_test, y_pred)),
        'r2': _to_native_type(r2_score(y_test, y_pred)),
        'train_r2': _to_native_type(r2_score(y_train, y_train_pred)),
    }

    evals_result = model.get_evals_result()
    metric_name = list(evals_result.get('learn', {}).keys())[0] if evals_result.get('learn') else None
    train_history = None
    if metric_name and 'validation' in evals_result:
        valid_key = list(evals_result['validation'].keys())[0]
        train_history = {'train': evals_result['learn'][metric_name], 'test': evals_result['validation'][valid_key]}

    return {
        'model': model, 'metrics': metrics,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred, 'train_history': train_history, 'eval_metric': metric_name,
        'train_pool': train_pool, 'test_pool': test_pool,
        'best_iteration': int(model.get_best_iteration() or params['iterations'])
    }


def get_feature_importance(model, feature_names: List[str], pool) -> List[Dict[str, Any]]:
    importance = model.get_feature_importance(pool)
    total = importance.sum() if importance.sum() > 0 else 1.0
    max_imp = importance.max() if importance.max() > 0 else 1.0
    importance_data = []
    for name, imp in zip(feature_names, importance):
        importance_data.append({
            'feature': name, 'importance': _to_native_type(imp),
            'importance_pct': _to_native_type(imp / total * 100),
            'normalized_importance': _to_native_type(imp / max_imp),
        })
    importance_data.sort(key=lambda x: x['importance'], reverse=True)
    for i, item in enumerate(importance_data):
        item['rank'] = i + 1
    return importance_data


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


def compute_shap(model, test_pool, feature_names: List[str]) -> Dict:
    """CatBoost computes exact SHAP values natively (no `shap` package needed)."""
    try:
        raw = model.get_feature_importance(test_pool, type='ShapValues')
        arr = np.array(raw)
        # Binary/regression: (n_samples, n_features+1); multiclass: (n_samples, n_classes, n_features+1).
        # Last column is the expected-value bias term — drop it before averaging.
        if arr.ndim == 3:
            mean_shap = np.abs(arr[:, :, :-1]).mean(axis=(0, 1))
        else:
            mean_shap = np.abs(arr[:, :-1]).mean(axis=0)

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


def perform_cross_validation(X, y, params: dict, task_type: str, cv_folds: int, cat_features: List[int]) -> Dict[str, Any]:
    cv_params = _common_params(params)
    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        cv_splitter = make_cv_splitter('classification', cv_folds, params['random_state'])
        scores = []
        for train_idx, test_idx in cv_splitter.split(X, y_encoded):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_encoded[train_idx], y_encoded[test_idx]
            m = CatBoostClassifier(**cv_params)
            m.fit(Pool(X_tr, y_tr, cat_features=cat_features), verbose=False)
            scores.append(accuracy_score(y_te, m.predict(Pool(X_te, cat_features=cat_features)).ravel().astype(int)))
    else:
        scores = []
        kf = make_cv_splitter('regression', cv_folds, params['random_state'])
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            m = CatBoostRegressor(**cv_params, loss_function='RMSE')
            m.fit(Pool(X_tr, y_tr, cat_features=cat_features), verbose=False)
            scores.append(r2_score(y_te, m.predict(Pool(X_te, cat_features=cat_features))))

    scores = np.array(scores)
    return {
        'cv_scores': [_to_native_type(s) for s in scores],
        'cv_mean': _to_native_type(np.mean(scores)),
        'cv_std': _to_native_type(np.std(scores)),
        'cv_folds': cv_folds
    }


def generate_feature_importance_plot(importance_data: List[Dict], top_n: int = 20) -> str:
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_data[:top_n]) * 0.4)))
    top_features = importance_data[:top_n]
    features = [d['feature'] for d in top_features][::-1]
    importances = [d['importance'] for d in top_features][::-1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, importances, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Feature Importance (PredictionValuesChange)', fontsize=11)
    ax.set_title('CatBoost Feature Importance', fontsize=13, fontweight='bold')
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


def generate_roc_plot(roc_data: Dict) -> str:
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


def generate_learning_curve_plot(train_history: Dict, eval_metric: str, best_iteration: int) -> Optional[str]:
    if not train_history:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    rounds = range(1, len(train_history['train']) + 1)
    ax.plot(rounds, train_history['train'], 'b-', linewidth=2, label='Train')
    ax.plot(rounds, train_history['test'], 'r-', linewidth=2, label='Test')
    ax.axvline(x=best_iteration, color='gray', linestyle=':', linewidth=1.5, label=f'Best iteration ({best_iteration})')
    ax.set_xlabel('Boosting Iteration', fontsize=11)
    ax.set_ylabel(eval_metric or 'Loss', fontsize=11)
    ax.set_title('Training History', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.3)
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


def generate_interpretation(result: Dict, task_type: str, feature_importance: List[Dict],
                             n_cat_features: int) -> Dict[str, Any]:
    key_insights = []

    if task_type == 'classification':
        accuracy = result['metrics']['accuracy']
        f1 = result['metrics']['f1_macro']
        status, perf_desc = ('positive', 'Excellent classification performance') if accuracy >= 0.9 else \
                             ('neutral', 'Good classification performance') if accuracy >= 0.7 else \
                             ('warning', 'Model may need improvement')
        key_insights.append({'title': 'Classification Performance', 'description': f'{perf_desc}. Accuracy: {accuracy:.1%}, F1-macro: {f1:.3f}', 'status': status})
        if 'auc' in result['metrics']:
            auc_val = result['metrics']['auc']
            key_insights.append({'title': 'AUC Score', 'description': f'Area Under ROC Curve: {auc_val:.3f}.', 'status': 'positive' if auc_val > 0.8 else 'neutral'})
    else:
        r2 = result['metrics']['r2']; rmse = result['metrics']['rmse']
        status, perf_desc = ('positive', 'Excellent fit') if r2 >= 0.8 else ('neutral', 'Moderate fit') if r2 >= 0.5 else ('warning', 'Weak fit')
        key_insights.append({'title': 'Regression Performance', 'description': f'{perf_desc}. R² = {r2:.3f}, RMSE = {rmse:.4f}', 'status': status})

    if n_cat_features > 0:
        key_insights.append({
            'title': 'Native Categorical Handling',
            'description': f'{n_cat_features} categorical feature(s) were passed directly to CatBoost without one-hot encoding, using ordered target statistics.',
            'status': 'positive'
        })

    if result.get('train_history'):
        n_rounds = len(result['train_history']['train'])
        best_iter = result['best_iteration']
        if best_iter < n_rounds * 0.6:
            key_insights.append({
                'title': 'Early Stopping Triggered',
                'description': f'Best iteration ({best_iter}) reached well before the max of {n_rounds} iterations.',
                'status': 'positive'
            })

    top_features = feature_importance[:3]
    feature_str = ', '.join([f"{f['feature']} ({f['importance']:.1f})" for f in top_features])
    key_insights.append({'title': 'Key Predictors', 'description': f'Top features: {feature_str}', 'status': 'neutral'})

    return {
        'key_insights': key_insights,
        'recommendation': (
            'CatBoost model trained successfully. For further improvement, tune depth and l2_leaf_reg '
            'jointly — deeper trees with weak regularization are the most common overfitting source.'
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

        iterations = int(payload.get('iterations', 300))
        depth = int(payload.get('depth', 6))
        learning_rate = float(payload.get('learning_rate', 0.1))
        l2_leaf_reg = float(payload.get('l2_leaf_reg', 3.0))
        random_state = int(payload.get('random_state', 42))
        early_stopping_rounds = payload.get('early_stopping_rounds', 20)
        early_stopping_rounds = int(early_stopping_rounds) if early_stopping_rounds not in (None, '', 0, False) else None
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
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna('missing')
        for c in [col for col in X.columns if col not in cat_cols]:
            X[c] = pd.to_numeric(X[c], errors='coerce')

        valid_mask = ~(X[[c for c in X.columns if c not in cat_cols]].isna().any(axis=1) | y.isna())
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

        if len(X) < 30:
            raise ValueError("At least 30 valid samples required.")

        if task_type == 'auto':
            task_type = detect_task_type(y)

        cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

        params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'random_state': random_state,
            'early_stopping_rounds': early_stopping_rounds
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
        X_train = X_train.reset_index(drop=True); X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True); y_test = y_test.reset_index(drop=True)

        if task_type == 'classification':
            result = train_catboost_classifier(X_train, X_test, y_train, y_test, params, cat_feature_indices)
        else:
            result = train_catboost_regressor(X_train, X_test, y_train, y_test, params, cat_feature_indices)

        model = result['model']
        feature_importance = get_feature_importance(model, feature_cols, result['train_pool'])

        y_test_for_perm = result['label_encoder'].transform(y_test) if task_type == 'classification' else y_test
        perm_importance = compute_permutation_importance(model, X_test, y_test_for_perm, feature_cols)
        shap_result = compute_shap(model, result['test_pool'], feature_cols)

        cv_result = perform_cross_validation(X, y, params, task_type, cv_folds, cat_feature_indices)

        importance_plot = generate_feature_importance_plot(feature_importance)
        learning_plot = generate_learning_curve_plot(result['train_history'], result['eval_metric'], result['best_iteration'])

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            regression_plot = None
        else:
            cm_plot = None
            roc_plot = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])

        interpretation = generate_interpretation(result, task_type, feature_importance, len(cat_cols))
        prediction_examples = generate_prediction_examples(result, task_type)

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
            'n_categorical_features': len(cat_cols),
            'categorical_features': cat_cols,
            'parameters': params,
            'metrics': result['metrics'],
            'feature_importance': feature_importance,
            'perm_importance': perm_importance,
            'shap_importance': shap_result.get('shap_importance'),
            'shap_plot': shap_result.get('shap_plot'),
            'shap_error': shap_result.get('error'),
            'cv_results': cv_result,
            'best_iteration': result['best_iteration'],
            'importance_plot': importance_plot,
            'learning_plot': learning_plot,
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
