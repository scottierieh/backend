"""
Naive Bayes Classification — CLI script
Probabilistic classifier based on Bayes' theorem. Ported from
scottierieh/backend's api/naive_bayes_analysis.py (FastAPI router) to the
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

VALID_NB_TYPES = {'gaussian', 'multinomial', 'bernoulli'}


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


def train_naive_bayes(X_train, X_test, y_train, y_test, params: dict, feature_names: List[str]) -> Dict[str, Any]:
    """Train Naive Bayes classifier with proper nb_type-specific preprocessing."""
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    n_classes = len(le.classes_)

    nb_type = params['nb_type']

    # ── nb_type-specific input preparation ─────────────────────────
    if nb_type == 'gaussian':
        # GaussianNB: continuous features — use as-is
        X_train_nb = X_train.astype(float)
        X_test_nb  = X_test.astype(float)
        model = GaussianNB(var_smoothing=params['var_smoothing'])

    elif nb_type == 'multinomial':
        # MultinomialNB: requires non-negative counts/frequencies.
        # Shift to non-negative using training min (applied to test consistently).
        train_min = X_train.min(axis=0)
        neg_mask  = train_min < 0
        X_train_nb = X_train.copy().astype(float)
        X_test_nb  = X_test.copy().astype(float)
        if neg_mask.any():
            shift = np.where(neg_mask, -train_min + 1e-6, 0.0)
            X_train_nb += shift
            X_test_nb  += shift
        # Ensure no zeros that would cause numerical issues
        X_train_nb = np.maximum(X_train_nb, 1e-10)
        X_test_nb  = np.maximum(X_test_nb,  1e-10)
        model = MultinomialNB(alpha=params['alpha'], fit_prior=params['fit_prior'])

    elif nb_type == 'bernoulli':
        # BernoulliNB: binarize using threshold (default median of training data)
        threshold = params.get('binarize_threshold')
        if threshold is None:
            threshold = float(np.median(X_train))
        X_train_nb = (X_train >= threshold).astype(float)
        X_test_nb  = (X_test  >= threshold).astype(float)
        model = BernoulliNB(alpha=params['alpha'], fit_prior=params['fit_prior'])

    else:
        X_train_nb = X_train.astype(float)
        X_test_nb  = X_test.astype(float)
        model = GaussianNB(var_smoothing=params['var_smoothing'])

    # ── Fit & predict ───────────────────────────────────────────────
    model.fit(X_train_nb, y_train_encoded)
    y_pred        = model.predict(X_test_nb)
    y_pred_proba  = model.predict_proba(X_test_nb)
    y_train_pred  = model.predict(X_train_nb)

    # ── Metrics ─────────────────────────────────────────────────────
    metrics = {
        'accuracy':         _to_native_type(accuracy_score(y_test_encoded, y_pred)),
        'train_accuracy':   _to_native_type(accuracy_score(y_train_encoded, y_train_pred)),
        'precision_macro':  _to_native_type(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'recall_macro':     _to_native_type(recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'f1_macro':         _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0))
    }

    # ── Per-class metrics ───────────────────────────────────────────
    class_report = classification_report(
        y_test_encoded, y_pred,
        target_names=[str(c) for c in le.classes_],
        output_dict=True
    )
    per_class_metrics = []
    for cls in le.classes_:
        cls_str = str(cls)
        if cls_str in class_report:
            per_class_metrics.append({
                'class':     cls_str,
                'precision': _to_native_type(class_report[cls_str]['precision']),
                'recall':    _to_native_type(class_report[cls_str]['recall']),
                'f1_score':  _to_native_type(class_report[cls_str]['f1-score']),
                'support':   int(class_report[cls_str]['support'])
            })

    # ── Confusion matrix ────────────────────────────────────────────
    cm = confusion_matrix(y_test_encoded, y_pred)

    # ── ROC curves + macro AUC ──────────────────────────────────────
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
        auc_values = []
        for i, cls in enumerate(le.classes_):
            y_binary = (y_test_encoded == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_values.append(roc_auc)
            roc_data[str(cls)] = {
                'fpr': [_to_native_type(x) for x in fpr],
                'tpr': [_to_native_type(x) for x in tpr],
                'auc': _to_native_type(roc_auc)
            }
        # macro AUC summary for multiclass
        metrics['auc_macro'] = _to_native_type(float(np.mean(auc_values)))

    # ── Class priors ────────────────────────────────────────────────
    class_priors = {
        str(cls): _to_native_type(model.class_prior_[i])
        for i, cls in enumerate(le.classes_)
    }

    return {
        'model':             model,
        'metrics':           metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix':  cm.tolist(),
        'class_labels':      [str(c) for c in le.classes_],
        'roc_data':          roc_data,
        'label_encoder':     le,
        'class_priors':      class_priors,
        'n_classes':         n_classes,
        # For prediction examples
        'y_test_encoded':    y_test_encoded,
        'y_pred':            y_pred,
        'y_pred_proba':      y_pred_proba,
        # Store preprocessed arrays for CV consistency
        'X_train_nb':        X_train_nb,
        'X_test_nb':         X_test_nb,
    }


def get_feature_importance_nb(model, feature_names: List[str], nb_type: str) -> List[Dict[str, Any]]:
    """Extract feature importance for Naive Bayes (based on variance or log probability)"""
    importance_data = []

    if nb_type == 'gaussian':
        # For Gaussian NB, use inverse of variance as importance (higher variance = more discriminative)
        # Average variance across classes
        avg_var = np.mean(model.var_, axis=0)
        # Use 1/var as importance (features with lower variance within classes are more useful)
        importance = 1.0 / (avg_var + 1e-10)
        importance = importance / importance.sum()  # Normalize
    elif nb_type in ['multinomial', 'bernoulli']:
        # For Multinomial/Bernoulli NB, use feature log probabilities
        # Higher absolute difference in log probabilities across classes = more important
        log_probs = model.feature_log_prob_
        importance = np.std(log_probs, axis=0)  # Variance across classes
        importance = importance / importance.sum()  # Normalize
    else:
        importance = np.ones(len(feature_names)) / len(feature_names)

    for name, imp in zip(feature_names, importance):
        importance_data.append({
            'feature': name,
            'importance': _to_native_type(imp)
        })

    # Sort by importance
    importance_data.sort(key=lambda x: x['importance'], reverse=True)
    return importance_data


def perform_cross_validation(X, y, params: dict, cv_folds: int) -> Dict[str, Any]:
    """Perform cross-validation with StratifiedKFold."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    nb_type = params['nb_type']
    if nb_type == 'gaussian':
        model = GaussianNB(var_smoothing=params['var_smoothing'])
    elif nb_type == 'multinomial':
        train_min = X.min(axis=0)
        neg_mask  = train_min < 0
        if neg_mask.any():
            shift = np.where(neg_mask, -train_min + 1e-6, 0.0)
            X = X + shift
        X = np.maximum(X, 1e-10)
        model = MultinomialNB(alpha=params['alpha'], fit_prior=params['fit_prior'])
    elif nb_type == 'bernoulli':
        threshold = params.get('binarize_threshold')
        if threshold is None:
            threshold = float(np.median(X))
        X = (X >= threshold).astype(float)
        model = BernoulliNB(alpha=params['alpha'], fit_prior=params['fit_prior'])
    else:
        model = GaussianNB(var_smoothing=params['var_smoothing'])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')

    return {
        'cv_scores': [_to_native_type(s) for s in scores],
        'cv_mean':   _to_native_type(np.mean(scores)),
        'cv_std':    _to_native_type(np.std(scores)),
        'cv_folds':  cv_folds,
        'cv_metric': 'accuracy'
    }


def generate_feature_importance_plot(importance_data: List[Dict], top_n: int = 20) -> str:
    """Generate feature importance plot"""
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_data[:top_n]) * 0.4)))

    top_features = importance_data[:top_n]
    features = [d['feature'] for d in top_features][::-1]
    importances = [d['importance'] for d in top_features][::-1]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Naive Bayes Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    # Add value labels
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9)

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


def generate_class_prior_plot(class_priors: Dict) -> str:
    """Generate class prior probabilities plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    classes = list(class_priors.keys())
    priors = list(class_priors.values())

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    bars = ax.bar(classes, priors, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Prior Probability', fontsize=11)
    ax.set_title('Class Prior Probabilities', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add value labels
    for bar, prior in zip(bars, priors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prior:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_probability_distribution_plot(model, feature_names: List[str], nb_type: str, class_labels: List[str]) -> str:
    """Generate probability distribution plot for features"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    n_features_to_show = min(4, len(feature_names))

    if nb_type == 'gaussian':
        # Show Gaussian distributions for top 4 features
        means = model.theta_
        vars_ = model.var_

        for idx in range(n_features_to_show):
            ax = axes[idx]
            x_range = np.linspace(
                means[:, idx].min() - 3 * np.sqrt(vars_[:, idx].max()),
                means[:, idx].max() + 3 * np.sqrt(vars_[:, idx].max()),
                200
            )

            for i, cls in enumerate(class_labels):
                mean = means[i, idx]
                var = vars_[i, idx]
                std = np.sqrt(var)
                pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
                ax.plot(x_range, pdf, linewidth=2, label=f'{cls}')
                ax.fill_between(x_range, pdf, alpha=0.3)

            ax.set_xlabel(feature_names[idx], fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Distribution: {feature_names[idx]}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
    else:
        # For multinomial/bernoulli, show feature log probabilities
        log_probs = model.feature_log_prob_

        for idx in range(n_features_to_show):
            ax = axes[idx]
            probs = np.exp(log_probs[:, idx])

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_labels)))
            bars = ax.bar(class_labels, probs, color=colors, edgecolor='black', alpha=0.8)

            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title(f'P({feature_names[idx]} | Class)', fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_features_to_show, 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_prediction_examples(result: Dict, n_examples: int = 15) -> List[Dict]:
    """Generate sample prediction examples for the report."""
    examples = []
    try:
        y_test_enc   = result.get('y_test_encoded')
        y_pred       = result.get('y_pred')
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
                'actual':     class_labels[int(y_test_enc[i])] if class_labels else int(y_test_enc[i]),
                'predicted':  class_labels[int(y_pred[i])]     if class_labels else int(y_pred[i]),
                'correct':    bool(y_test_enc[i] == y_pred[i]),
                'confidence': round(conf, 4) if conf is not None else None
            })
    except Exception:
        pass
    return examples


def generate_interpretation(result: Dict, feature_importance: List[Dict], params: dict) -> Dict[str, Any]:
    """Generate interpretation of Naive Bayes results."""
    key_insights = []

    accuracy  = result['metrics']['accuracy']
    train_acc = result['metrics'].get('train_accuracy')
    f1        = result['metrics']['f1_macro']
    nb_type   = params['nb_type']

    # ── 1. Performance ──────────────────────────────────────────────
    if accuracy >= 0.90:
        status    = 'positive'
        perf_desc = 'Excellent classification performance'
    elif accuracy >= 0.75:
        status    = 'neutral'
        perf_desc = 'Good classification performance'
    elif accuracy >= 0.60:
        status    = 'warning'
        perf_desc = 'Moderate performance — consider feature engineering'
    else:
        status    = 'warning'
        perf_desc = 'Low accuracy — model assumptions may not fit the data'

    train_str = f', Train: {train_acc:.1%}' if train_acc is not None else ''
    key_insights.append({
        'title':       'Classification Performance',
        'description': f'{perf_desc}. Test Accuracy: {accuracy:.1%}{train_str}, F1-macro: {f1:.3f}',
        'status':      status
    })

    # ── 2. Train-test gap ──────────────────────────────────────────
    if train_acc is not None:
        gap = train_acc - accuracy
        if gap > 0.15:
            gap_desc   = (f'Train accuracy ({train_acc:.1%}) is much higher than test ({accuracy:.1%}). '
                          'Possible data distribution mismatch or preprocessing inconsistency.')
            gap_status = 'warning'
        elif gap > 0.07:
            gap_desc   = (f'Moderate train-test gap ({gap:.1%}). '
                          'Some overfitting — monitor on new data.')
            gap_status = 'neutral'
        else:
            gap_desc   = (f'Train ({train_acc:.1%}) and test ({accuracy:.1%}) accuracy are close — '
                          'good generalization.')
            gap_status = 'positive'
        key_insights.append({
            'title':       'Train vs Test Gap',
            'description': gap_desc,
            'status':      gap_status
        })

    # ── 3. AUC ─────────────────────────────────────────────────────
    if 'auc' in result['metrics']:
        auc_val = result['metrics']['auc']
        key_insights.append({
            'title':       'AUC Score (Binary)',
            'description': (f'ROC-AUC: {auc_val:.3f}. '
                            f'{"Excellent" if auc_val > 0.9 else "Good" if auc_val > 0.7 else "Fair"} discrimination.'),
            'status': 'positive' if auc_val > 0.8 else 'neutral'
        })
    elif 'auc_macro' in result['metrics']:
        auc_val = result['metrics']['auc_macro']
        key_insights.append({
            'title':       'Macro AUC (One-vs-Rest)',
            'description': (f'Mean one-vs-rest ROC-AUC: {auc_val:.3f}. '
                            f'{"Excellent" if auc_val > 0.9 else "Good" if auc_val > 0.7 else "Fair"} overall discrimination.'),
            'status': 'positive' if auc_val > 0.8 else 'neutral'
        })

    # ── 4. NB type ─────────────────────────────────────────────────
    type_info = {
        'gaussian':    ('GaussianNB: assumes each feature follows a normal distribution per class. '
                        'Best suited for continuous numeric features.'),
        'multinomial': ('MultinomialNB: models feature counts/frequencies. '
                        'Non-negative values required; features shifted automatically if needed.'),
        'bernoulli':   ('BernoulliNB: designed for binary (0/1) features. '
                        'Input was binarized at the median threshold before training.')
    }
    key_insights.append({
        'title':       f'Model Type: {nb_type.capitalize()}NB',
        'description': type_info.get(nb_type, 'Naive Bayes classifier'),
        'status':      'neutral'
    })

    # ── 5. Approximate feature importance ──────────────────────────
    top_features = feature_importance[:3]
    feature_str = ', '.join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
    key_insights.append({
        'title':       'Key Predictors (Approximate Signal)',
        'description': (f'Top features: {feature_str}. '
                        'Importance is model-based heuristic (inverse variance for Gaussian, '
                        'log-prob spread for Multinomial/Bernoulli) — treat as indicative.'),
        'status': 'neutral'
    })

    # ── 6. Class imbalance ─────────────────────────────────────────
    priors    = result['class_priors']
    max_prior = max(priors.values())
    min_prior = min(priors.values())
    if min_prior > 0 and max_prior / min_prior > 3:
        key_insights.append({
            'title':       'Class Imbalance',
            'description': (f'Classes are imbalanced (ratio {max_prior/min_prior:.1f}:1). '
                            'Consider fit_prior=False to use uniform priors, or resampling.'),
            'status': 'warning'
        })

    rec_map = {
        'gaussian':    'GaussianNB trained on continuous features. If features are heavily skewed, consider log-transforming first.',
        'multinomial': 'MultinomialNB trained on non-negative data. Works best with true count or frequency features.',
        'bernoulli':   'BernoulliNB trained on binarized features. Verify the binarization threshold is meaningful for your data.'
    }
    return {
        'key_insights':   key_insights,
        'recommendation': rec_map.get(nb_type, 'Naive Bayes model trained successfully.')
    }


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        target_col = payload.get('target_col') or payload.get('target')
        feature_cols = payload.get('feature_cols') or payload.get('features')
        test_size = float(payload.get('test_size', 0.2))

        # Naive Bayes parameters
        nb_type = payload.get('nb_type', 'gaussian')
        var_smoothing = float(payload.get('var_smoothing', 1e-9))
        alpha = float(payload.get('alpha', 1.0))
        fit_prior = bool(payload.get('fit_prior', True))
        binarize_threshold = payload.get('binarize_threshold', None)
        random_state = int(payload.get('random_state', 42))
        cv_folds = int(payload.get('cv_folds', 5))

        if not data:
            raise ValueError("Data not provided.")
        if not target_col or not feature_cols:
            raise ValueError("Missing data, features, or target")

        df = pd.DataFrame(data)

        # ── Input validation (mirrors NaiveBayesRequest.validate_inputs) ──
        if nb_type not in VALID_NB_TYPES:
            raise ValueError(
                f"Invalid nb_type '{nb_type}'. "
                f"Supported: {sorted(VALID_NB_TYPES)}"
            )
        if cv_folds < 2:
            raise ValueError("cv_folds must be ≥ 2.")
        if not (0.05 <= test_size <= 0.5):
            raise ValueError("test_size must be between 0.05 and 0.50.")

        # Validate columns
        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {', '.join(missing)}")

        # ── Prepare features ────────────────────────────────────────
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Categorical → one-hot (safer than LabelEncoder for NB)
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
        feature_cols = list(X.columns)  # update after one-hot expansion

        if len(X) < 50:
            raise ValueError("At least 50 valid samples required.")

        # ── Parameters ──────────────────────────────────────────────
        params = {
            'nb_type':             nb_type,
            'var_smoothing':       var_smoothing,
            'alpha':               alpha,
            'fit_prior':           fit_prior,
            'binarize_threshold':  binarize_threshold,
            'random_state':        random_state
        }

        # ── Split data ──────────────────────────────────────────────
        X_array = X.values
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y, test_size=test_size,
            random_state=random_state, stratify=y
        )

        # ── Train model ─────────────────────────────────────────────
        result = train_naive_bayes(X_train, X_test, y_train, y_test, params, feature_cols)
        model  = result['model']

        # ── Feature importance ──────────────────────────────────────
        feature_importance = get_feature_importance_nb(model, feature_cols, nb_type)

        # ── Cross-validation ────────────────────────────────────────
        cv_result = perform_cross_validation(X_array, y, params, cv_folds)

        # ── Prediction examples ─────────────────────────────────────
        prediction_examples = generate_prediction_examples(result, n_examples=15)

        # ── Visualizations ──────────────────────────────────────────
        importance_plot  = generate_feature_importance_plot(feature_importance)
        cm_plot          = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
        roc_plot         = generate_roc_plot(result['roc_data']) if result['roc_data'] else None
        prior_plot       = generate_class_prior_plot(result['class_priors'])
        prob_dist_plot   = generate_probability_distribution_plot(
            model, feature_cols, nb_type, result['class_labels']
        )

        # ── Interpretation ──────────────────────────────────────────
        interpretation = generate_interpretation(result, feature_importance, params)

        # ── Response ────────────────────────────────────────────────
        response = {
            'task_type':           'classification',
            'n_samples':           len(X),
            'n_features':          len(feature_cols),
            'n_train':             len(X_train),
            'n_test':              len(X_test),
            'n_classes':           result['n_classes'],
            'parameters':          params,
            'metrics':             result['metrics'],
            'feature_importance':  feature_importance,
            'cv_results':          cv_result,
            'class_priors':        result['class_priors'],
            'per_class_metrics':   result['per_class_metrics'],
            'confusion_matrix':    result['confusion_matrix'],
            'class_labels':        result['class_labels'],
            'importance_plot':     importance_plot,
            'cm_plot':             cm_plot,
            'roc_plot':            roc_plot,
            'prior_plot':          prior_plot,
            'prob_dist_plot':      prob_dist_plot,
            'interpretation':      interpretation,
            'prediction_examples': prediction_examples
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
