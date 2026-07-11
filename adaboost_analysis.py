"""
AdaBoost Classification and Regression — CLI script
Adaptive boosting over shallow decision trees ("stumps"). Ported from
scottierieh/backend's api/adaboost_analysis.py (FastAPI router) to the
stdin/stdout CLI contract used by src/backend/main.py's generic script runner.
"""

import sys
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.inspection import permutation_importance
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


def train_adaboost_classifier(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    n_classes = len(le.classes_)

    base_estimator = DecisionTreeClassifier(max_depth=params['base_max_depth'], random_state=params['random_state'])
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        random_state=params['random_state']
    )
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

    staged_train = [_to_native_type(s) for s in model.staged_score(X_train, y_train_encoded)]
    staged_test = [_to_native_type(s) for s in model.staged_score(X_test, y_test_encoded)]

    return {
        'model': model,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'staged_train_scores': staged_train,
        'staged_test_scores': staged_test,
        'estimator_weights': [_to_native_type(w) for w in model.estimator_weights_],
        'estimator_errors': [_to_native_type(e) for e in model.estimator_errors_],
        'label_encoder': le,
        'y_test_encoded': y_test_encoded,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }


def train_adaboost_regressor(X_train, X_test, y_train, y_test, params: dict) -> Dict[str, Any]:
    base_estimator = DecisionTreeRegressor(max_depth=params['base_max_depth'], random_state=params['random_state'])
    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        random_state=params['random_state']
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
        'train_r2': _to_native_type(r2_score(y_train, y_train_pred)),
    }

    staged_train = [_to_native_type(s) for s in model.staged_score(X_train, y_train)]
    staged_test = [_to_native_type(s) for s in model.staged_score(X_test, y_test)]

    return {
        'model': model,
        'metrics': metrics,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred,
        'staged_train_scores': staged_train,
        'staged_test_scores': staged_test,
        'estimator_weights': [_to_native_type(w) for w in model.estimator_weights_],
        'estimator_errors': [_to_native_type(e) for e in model.estimator_errors_],
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


def compute_permutation_importance(model, X_test, y_test, feature_names: List[str],
                                    n_repeats: int = 10, random_state: int = 42) -> List[Dict[str, Any]]:
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                                       random_state=random_state, n_jobs=-1)
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
    """Model-agnostic SHAP (Permutation explainer) — AdaBoost's tree ensemble isn't
    reliably supported by shap.TreeExplainer, so we explain predict_proba/predict directly."""
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


def perform_cross_validation(X, y, params: dict, task_type: str, cv_folds: int) -> Dict[str, Any]:
    base_est_cls = DecisionTreeClassifier if task_type == 'classification' else DecisionTreeRegressor
    base_estimator = base_est_cls(max_depth=params['base_max_depth'], random_state=params['random_state'])

    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model = AdaBoostClassifier(estimator=base_estimator, n_estimators=params['n_estimators'],
                                    learning_rate=params['learning_rate'], random_state=params['random_state'])
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=params['random_state'])
        scores = cross_val_score(model, X, y_encoded, cv=cv_splitter, scoring='accuracy')
    else:
        model = AdaBoostRegressor(estimator=base_estimator, n_estimators=params['n_estimators'],
                                   learning_rate=params['learning_rate'], random_state=params['random_state'])
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')

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
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('AdaBoost Feature Importance', fontsize=13, fontweight='bold')
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
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
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
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_staged_accuracy_plot(staged_train: List[float], staged_test: List[float], task_type: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    rounds = range(1, len(staged_train) + 1)
    ax.plot(rounds, staged_train, 'b-', linewidth=2, label='Train')
    ax.plot(rounds, staged_test, 'r-', linewidth=2, label='Test')
    ax.set_xlabel('Boosting Round (n_estimators)', fontsize=11)
    ax.set_ylabel('Accuracy' if task_type == 'classification' else 'R²', fontsize=11)
    ax.set_title('Staged Performance vs. Boosting Rounds', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
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


def generate_interpretation(result: Dict, task_type: str, feature_importance: List[Dict], params: dict) -> Dict[str, Any]:
    key_insights = []

    if task_type == 'classification':
        accuracy = result['metrics']['accuracy']
        train_accuracy = result['metrics'].get('train_accuracy')
        f1 = result['metrics']['f1_macro']
        gap = (train_accuracy - accuracy) if train_accuracy is not None else 0

        if accuracy >= 0.9:
            status, perf_desc = 'positive', 'Excellent classification performance'
        elif accuracy >= 0.75:
            status, perf_desc = 'neutral', 'Good classification performance'
        else:
            status, perf_desc = 'warning', 'Moderate performance — consider more estimators or a deeper base tree'

        key_insights.append({
            'title': 'Classification Performance',
            'description': f'{perf_desc}. Test Accuracy: {accuracy:.1%}, Train: {train_accuracy:.1%}, F1-macro: {f1:.3f}',
            'status': status
        })

        if gap > 0.15:
            key_insights.append({
                'title': 'Overfitting Warning',
                'description': f'Train accuracy exceeds test accuracy by {gap:.1%}. Reduce n_estimators/learning_rate or shrink the base tree depth.',
                'status': 'warning'
            })
    else:
        r2 = result['metrics']['r2']
        rmse = result['metrics']['rmse']
        status, perf_desc = ('positive', 'Excellent fit') if r2 >= 0.8 else ('neutral', 'Moderate fit') if r2 >= 0.5 else ('warning', 'Weak fit')
        key_insights.append({
            'title': 'Regression Performance',
            'description': f'{perf_desc}. R² = {r2:.3f}, RMSE = {rmse:.4f}',
            'status': status
        })

    staged_test = result['staged_test_scores']
    best_idx = int(np.argmax(staged_test))
    if best_idx < len(staged_test) * 0.6:
        key_insights.append({
            'title': 'Early Convergence',
            'description': f'Test performance peaked at round {best_idx + 1} of {len(staged_test)}. Additional estimators beyond that point add little value.',
            'status': 'neutral'
        })

    mean_error = float(np.mean(result['estimator_errors']))
    key_insights.append({
        'title': 'Weak Learner Error',
        'description': f'Mean weighted error of individual weak learners: {mean_error:.3f} (lower means each stump is more informative).',
        'status': 'neutral'
    })

    top_features = feature_importance[:3]
    feature_str = ', '.join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
    key_insights.append({
        'title': 'Key Predictors',
        'description': f'Top features: {feature_str}',
        'status': 'neutral'
    })

    return {
        'key_insights': key_insights,
        'recommendation': (
            'AdaBoost trained successfully. For further improvement: increase n_estimators if the '
            'staged performance curve is still rising, or reduce it (with early stopping in mind) if '
            'test performance already peaked. base_max_depth=1 (decision stumps) is the classic '
            'AdaBoost setup; slightly deeper trees can help on more complex data at the cost of overfitting risk.'
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

        n_estimators = int(payload.get('n_estimators', 100))
        learning_rate = float(payload.get('learning_rate', 1.0))
        base_max_depth = int(payload.get('base_max_depth', payload.get('max_depth', 1)))
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

        if task_type == 'auto':
            task_type = detect_task_type(y)

        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'base_max_depth': base_max_depth,
            'random_state': random_state
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
            result = train_adaboost_classifier(X_train, X_test, y_train, y_test, params)
        else:
            result = train_adaboost_regressor(X_train, X_test, y_train, y_test, params)

        model = result['model']
        feature_importance = get_feature_importance(model, feature_cols)

        y_test_for_perm = result['label_encoder'].transform(y_test) if task_type == 'classification' else (y_test.values if hasattr(y_test, 'values') else y_test)
        X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
        perm_importance = compute_permutation_importance(model, X_test_arr, y_test_for_perm, feature_cols)
        shap_result = compute_shap(model, X_test_arr, feature_cols, task_type)

        cv_result = perform_cross_validation(X, y, params, task_type, cv_folds)

        importance_plot = generate_feature_importance_plot(feature_importance)
        staged_plot = generate_staged_accuracy_plot(result['staged_train_scores'], result['staged_test_scores'], task_type)

        if task_type == 'classification':
            cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
            roc_plot = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
            regression_plot = None
        else:
            cm_plot = None
            roc_plot = None
            regression_plot = generate_regression_plot(result['y_test'], result['y_pred'])

        interpretation = generate_interpretation(result, task_type, feature_importance, params)
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
            'parameters': params,
            'metrics': result['metrics'],
            'feature_importance': feature_importance,
            'perm_importance': perm_importance,
            'shap_importance': shap_result.get('shap_importance'),
            'shap_plot': shap_result.get('shap_plot'),
            'shap_error': shap_result.get('error'),
            'cv_results': cv_result,
            'staged_train_scores': result['staged_train_scores'],
            'staged_test_scores': result['staged_test_scores'],
            'estimator_weights': result['estimator_weights'],
            'estimator_errors': result['estimator_errors'],
            'importance_plot': importance_plot,
            'staged_plot': staged_plot,
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
