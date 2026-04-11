"""
Discriminant Analysis Router for FastAPI
Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) for classification
SPSS-level statistical output included.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class DiscriminantAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_col: str
    feature_cols: List[str]
    method: str = "lda"
    test_size: float = 0.2
    solver: str = "svd"
    shrinkage: Optional[str] = None
    n_components: Optional[int] = None
    reg_param: float = 0.0
    priors: Optional[List[float]] = None
    random_state: int = 42
    cv_folds: int = 5


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def _to_native_type(obj):
    """
    Convert numpy/pandas types to JSON-serializable Python types.
    [FIX] bool 체크를 int/float보다 먼저 (bool은 int 서브클래스)
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _parse_shrinkage(shrinkage_val):
    """
    [FIX] shrinkage 파라미터 타입 변환
    None / 'None' → None / 'auto' → 'auto' / '0.5' → float(0.5)
    """
    if shrinkage_val is None or shrinkage_val == 'None':
        return None
    if shrinkage_val == 'auto':
        return 'auto'
    try:
        return float(shrinkage_val)
    except (ValueError, TypeError):
        return None


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def _normalize_importance(scores: np.ndarray) -> np.ndarray:
    """
    [FIX] feature importance 합이 정확히 1.0이 되도록 보정
    """
    total = scores.sum()
    if total == 0:
        return np.ones(len(scores)) / len(scores)
    normalized = scores / total
    normalized[-1] = 1.0 - normalized[:-1].sum()
    return normalized


def _compute_multiclass_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Optional[float]:
    """
    [FIX] 멀티클래스 macro-average AUC
    이진: binary AUC / 멀티클래스: OvR macro AUC
    """
    try:
        n_classes = y_pred_proba.shape[1]
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'))
    except Exception:
        return None


# ──────────────────────────────────────────────
# SPSS-level LDA statistical output
# ──────────────────────────────────────────────

def compute_lda_statistics(X_train: np.ndarray, y_train_encoded: np.ndarray,
                            model: LinearDiscriminantAnalysis,
                            feature_names: List[str],
                            class_labels: List[str]) -> Dict[str, Any]:
    """
    SPSS / SAS 수준의 LDA 통계 출력
    - Wilks' Lambda
    - Eigenvalues & Canonical Correlation
    - Structure Matrix (pooled within-group correlations)
    - ANOVA F-statistics per feature
    """
    stats_output = {}
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train_encoded))

    # ── 1. Eigenvalues & Canonical Correlation ──────────────
    # Between-class scatter matrix (S_B)
    overall_mean = X_train.mean(axis=0)
    S_B = np.zeros((n_features, n_features))
    S_W = np.zeros((n_features, n_features))

    class_info = []
    for i in range(n_classes):
        mask = y_train_encoded == i
        n_k = mask.sum()
        mean_k = X_train[mask].mean(axis=0)
        diff = (mean_k - overall_mean).reshape(-1, 1)
        S_B += n_k * (diff @ diff.T)

        centered = X_train[mask] - mean_k
        S_W += centered.T @ centered

        class_info.append({
            'class': class_labels[i],
            'n': int(n_k),
            'mean': mean_k.tolist()
        })

    # Eigenvalues from S_W^-1 * S_B
    try:
        S_W_inv = np.linalg.pinv(S_W)
        eigenvalues_raw, eigenvectors = np.linalg.eig(S_W_inv @ S_B)
        eigenvalues_raw = np.real(eigenvalues_raw)
        # 유효한 양수 eigenvalue만 선택
        valid_idx = np.argsort(eigenvalues_raw)[::-1]
        eigenvalues = eigenvalues_raw[valid_idx]
        eigenvalues = eigenvalues[eigenvalues > 1e-10][:n_classes - 1]

        # Canonical Correlation: r = sqrt(λ / (1 + λ))
        canonical_correlations = np.sqrt(eigenvalues / (1 + eigenvalues))

        # % of variance explained by each discriminant function
        total_eigen = eigenvalues.sum()
        variance_explained = (eigenvalues / total_eigen * 100) if total_eigen > 0 else eigenvalues * 0

        stats_output['eigenvalues'] = [
            {
                'function': f'LD{i+1}',
                'eigenvalue': _to_native_type(float(ev)),
                'variance_explained_pct': _to_native_type(float(ve)),
                'cumulative_pct': _to_native_type(float(variance_explained[:i+1].sum())),
                'canonical_correlation': _to_native_type(float(cc))
            }
            for i, (ev, ve, cc) in enumerate(zip(eigenvalues, variance_explained, canonical_correlations))
        ]
    except Exception:
        stats_output['eigenvalues'] = []

    # ── 2. Wilks' Lambda ────────────────────────────────────
    # Wilks' Λ = ∏ 1/(1+λᵢ)
    try:
        wilks_lambda = float(np.prod([1 / (1 + ev) for ev in eigenvalues]))

        # Chi-square approximation: χ² = -(n - 1 - (p + g)/2) * ln(Λ)
        p = n_features
        g = n_classes
        chi2_stat = -(n_samples - 1 - (p + g) / 2) * np.log(wilks_lambda + 1e-10)
        df = p * (g - 1)
        p_value = float(1 - stats.chi2.cdf(chi2_stat, df))

        stats_output['wilks_lambda'] = {
            'lambda': _to_native_type(wilks_lambda),
            'chi2': _to_native_type(float(chi2_stat)),
            'df': int(df),
            'p_value': _to_native_type(p_value),
            'significant': p_value < 0.05
        }
    except Exception:
        stats_output['wilks_lambda'] = None

    # ── 3. Structure Matrix ──────────────────────────────────
    # pooled within-group correlation between each feature and each discriminant function
    try:
        if hasattr(model, 'scalings_') and model.scalings_ is not None:
            scalings = model.scalings_  # (n_features, n_components)

            # Pooled within-group covariance
            S_W_pooled = S_W / (n_samples - n_classes)
            std_diag = np.sqrt(np.diag(S_W_pooled))
            std_diag[std_diag == 0] = 1e-10

            structure_matrix = []
            for feat_idx, feat_name in enumerate(feature_names):
                row = {'feature': feat_name}
                for comp_idx in range(scalings.shape[1]):
                    # correlation between feature and discriminant function score
                    ldf_scores = X_train @ scalings[:, comp_idx]
                    feat_vals = X_train[:, feat_idx]
                    corr = np.corrcoef(feat_vals, ldf_scores)[0, 1]
                    row[f'LD{comp_idx+1}'] = _to_native_type(float(corr))
                structure_matrix.append(row)

            stats_output['structure_matrix'] = structure_matrix
    except Exception:
        stats_output['structure_matrix'] = []

    # ── 4. ANOVA F-statistics per feature ───────────────────
    try:
        anova_results = []
        for feat_idx, feat_name in enumerate(feature_names):
            groups = [X_train[y_train_encoded == i, feat_idx] for i in range(n_classes)]
            f_stat, p_val = stats.f_oneway(*groups)
            anova_results.append({
                'feature': feat_name,
                'f_statistic': _to_native_type(float(f_stat)),
                'p_value': _to_native_type(float(p_val)),
                'significant': bool(p_val < 0.05)
            })
        # F값 기준 내림차순 정렬
        anova_results.sort(key=lambda x: x['f_statistic'] or 0, reverse=True)
        stats_output['anova_f_statistics'] = anova_results
    except Exception:
        stats_output['anova_f_statistics'] = []

    # ── 5. Class centroids (group means on discriminant functions) ──
    try:
        if hasattr(model, 'scalings_') and model.scalings_ is not None:
            centroids = []
            for i, label in enumerate(class_labels):
                mask = y_train_encoded == i
                class_scores = X_train[mask] @ model.scalings_
                centroid = {'class': label}
                for comp_idx in range(class_scores.shape[1]):
                    centroid[f'LD{comp_idx+1}'] = _to_native_type(float(class_scores[:, comp_idx].mean()))
                centroids.append(centroid)
            stats_output['group_centroids'] = centroids
    except Exception:
        stats_output['group_centroids'] = []

    stats_output['class_info'] = class_info

    return stats_output


# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────

def train_lda(X_train, X_test, y_train, y_test, params: dict, feature_names: List[str]) -> Dict[str, Any]:
    """Train Linear Discriminant Analysis model"""
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(le.classes_)
    n_features = X_train.shape[1]

    max_components = min(n_features, n_classes - 1)
    n_components = params.get('n_components')
    if n_components is None or n_components > max_components:
        n_components = max_components if max_components > 0 else None

    shrinkage = _parse_shrinkage(params.get('shrinkage'))
    solver = params.get('solver', 'svd')
    if shrinkage is not None and solver == 'svd':
        solver = 'lsqr'
    solver_used = solver

    model = LinearDiscriminantAnalysis(
        solver=solver,
        shrinkage=shrinkage,
        n_components=n_components,
        priors=params.get('priors')
    )
    model.fit(X_train, y_train_encoded)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred)),
        'precision_macro': _to_native_type(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'recall_macro': _to_native_type(recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0))
    }

    y_train_pred = model.predict(X_train)
    metrics['train_accuracy'] = _to_native_type(accuracy_score(y_train_encoded, y_train_pred))

    macro_auc = _compute_multiclass_auc(y_test_encoded, y_pred_proba)
    if macro_auc is not None:
        metrics['auc'] = macro_auc

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
                'class': cls_str,
                'precision': _to_native_type(class_report[cls_str]['precision']),
                'recall': _to_native_type(class_report[cls_str]['recall']),
                'f1_score': _to_native_type(class_report[cls_str]['f1-score']),
                'support': int(class_report[cls_str]['support'])
            })

    cm = confusion_matrix(y_test_encoded, y_pred)

    # ROC curves
    roc_data = {}
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba[:, 1])
        roc_auc_val = auc(fpr, tpr)
        roc_data['binary'] = {
            'fpr': [_to_native_type(x) for x in fpr],
            'tpr': [_to_native_type(x) for x in tpr],
            'auc': _to_native_type(roc_auc_val)
        }
    else:
        for i, cls in enumerate(le.classes_):
            y_binary = (y_test_encoded == i).astype(int)
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                continue
            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            roc_data[str(cls)] = {
                'fpr': [_to_native_type(x) for x in fpr],
                'tpr': [_to_native_type(x) for x in tpr],
                'auc': _to_native_type(roc_auc_val)
            }
        if macro_auc is not None:
            roc_data['__macro_auc__'] = macro_auc

    # LDA info
    lda_info = {}
    lda_info['solver_used'] = solver_used

    # ── [FIX 1] Feature Importance — standardized coefficient 방식 ──
    # scalings 방식 대신 coef_ * std 사용 (scale 문제 해결)
    X_train_arr = np.array(X_train)
    std = X_train_arr.std(axis=0)
    std[std == 0] = 1e-10  # zero std 방어

    if hasattr(model, 'coef_') and model.coef_ is not None:
        coef = np.abs(model.coef_)  # (n_classes-1, n_features) or (1, n_features)
        # 멀티클래스는 mean, 이진은 squeeze
        coef_mean = coef.mean(axis=0) if coef.ndim > 1 else coef.ravel()
        importance_scores = coef_mean * std
        importance_scores = _normalize_importance(importance_scores)
    elif hasattr(model, 'scalings_') and model.scalings_ is not None:
        scalings = model.scalings_
        importance_scores = np.abs(scalings).mean(axis=1) if scalings.ndim > 1 else np.abs(scalings.ravel())
        importance_scores = _normalize_importance(importance_scores)
    else:
        importance_scores = np.ones(n_features) / n_features

    feature_importance = []
    for name, imp in zip(feature_names, importance_scores):
        feature_importance.append({
            'feature': name,
            'importance': _to_native_type(float(imp))
        })
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)

    # scalings 원본 저장
    if hasattr(model, 'scalings_') and model.scalings_ is not None:
        lda_info['scalings'] = model.scalings_.tolist()

    if hasattr(model, 'explained_variance_ratio_'):
        lda_info['explained_variance_ratio'] = [_to_native_type(x) for x in model.explained_variance_ratio_]

    if hasattr(model, 'means_'):
        lda_info['class_means'] = model.means_.tolist()
    if hasattr(model, 'priors_'):
        lda_info['priors'] = [_to_native_type(x) for x in model.priors_]

    # LDA transform for projection plot
    lda_transform = None
    if n_components is not None and n_components >= 1:
        X_combined = np.vstack([X_train_arr, np.array(X_test)])
        y_combined = np.hstack([y_train_encoded, y_test_encoded])
        X_transformed = model.transform(X_combined)
        lda_transform = {
            'X': X_transformed.tolist(),
            'y': y_combined.tolist(),
            'class_labels': [str(c) for c in le.classes_],
            'n_components': X_transformed.shape[1]
        }

    # SPSS-level statistics
    lda_statistics = compute_lda_statistics(
        X_train_arr, y_train_encoded, model,
        feature_names, [str(c) for c in le.classes_]
    )

    return {
        'model': model,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'label_encoder': le,
        'feature_importance': feature_importance,
        'lda_info': lda_info,
        'lda_statistics': lda_statistics,
        'lda_transform': lda_transform
    }


def train_qda(X_train, X_test, y_train, y_test, params: dict, feature_names: List[str]) -> Dict[str, Any]:
    """Train Quadratic Discriminant Analysis model"""
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    n_classes = len(le.classes_)

    # [FIX] singular matrix 방어
    reg_param = params.get('reg_param', 0.0)
    min_samples_per_class = min(np.sum(y_train_encoded == i) for i in range(n_classes))
    n_features = X_train.shape[1]
    if reg_param == 0.0 and min_samples_per_class <= n_features:
        reg_param = 0.01

    model = QuadraticDiscriminantAnalysis(
        reg_param=reg_param,
        priors=params.get('priors')
    )
    try:
        model.fit(X_train, y_train_encoded)
    except Exception:
        model = QuadraticDiscriminantAnalysis(
            reg_param=max(reg_param, 0.1),
            priors=params.get('priors')
        )
        model.fit(X_train, y_train_encoded)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test_encoded, y_pred)),
        'precision_macro': _to_native_type(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'recall_macro': _to_native_type(recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)),
        'f1_macro': _to_native_type(f1_score(y_test_encoded, y_pred, average='macro', zero_division=0))
    }

    y_train_pred = model.predict(X_train)
    metrics['train_accuracy'] = _to_native_type(accuracy_score(y_train_encoded, y_train_pred))

    macro_auc = _compute_multiclass_auc(y_test_encoded, y_pred_proba)
    if macro_auc is not None:
        metrics['auc'] = macro_auc

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
        roc_auc_val = auc(fpr, tpr)
        roc_data['binary'] = {
            'fpr': [_to_native_type(x) for x in fpr],
            'tpr': [_to_native_type(x) for x in tpr],
            'auc': _to_native_type(roc_auc_val)
        }
    else:
        for i, cls in enumerate(le.classes_):
            y_binary = (y_test_encoded == i).astype(int)
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                continue
            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            roc_data[str(cls)] = {
                'fpr': [_to_native_type(x) for x in fpr],
                'tpr': [_to_native_type(x) for x in tpr],
                'auc': _to_native_type(roc_auc_val)
            }
        if macro_auc is not None:
            roc_data['__macro_auc__'] = macro_auc

    qda_info = {}
    if hasattr(model, 'means_'):
        qda_info['class_means'] = model.means_.tolist()
    if hasattr(model, 'priors_'):
        qda_info['priors'] = [_to_native_type(x) for x in model.priors_]
    qda_info['reg_param_used'] = float(reg_param)

    # ── [FIX 4] QDA Feature Importance — ANOVA F-statistic 방식 ──
    feature_importance = []
    X_train_arr = np.array(X_train)
    try:
        f_stats = []
        for feat_idx in range(n_features):
            groups = [X_train_arr[y_train_encoded == i, feat_idx] for i in range(n_classes)]
            f_val, _ = stats.f_oneway(*groups)
            f_stats.append(max(float(f_val), 0.0) if not np.isnan(f_val) else 0.0)
        f_arr = np.array(f_stats)
        importance_scores = _normalize_importance(f_arr) if f_arr.sum() > 0 else np.ones(n_features) / n_features
    except Exception:
        # fallback: class means variance
        if hasattr(model, 'means_'):
            mean_variance = np.var(model.means_, axis=0)
            importance_scores = _normalize_importance(mean_variance) if mean_variance.sum() > 0 else np.ones(n_features) / n_features
        else:
            importance_scores = np.ones(n_features) / n_features

    for name, imp in zip(feature_names, importance_scores):
        feature_importance.append({
            'feature': name,
            'importance': _to_native_type(float(imp))
        })
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)

    return {
        'model': model,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'class_labels': [str(c) for c in le.classes_],
        'roc_data': roc_data,
        'label_encoder': le,
        'feature_importance': feature_importance,
        'qda_info': qda_info
    }


# ──────────────────────────────────────────────
# Cross Validation — Pipeline으로 leakage 완전 차단
# ──────────────────────────────────────────────

def perform_cross_validation(X_raw, y, params: dict, method: str, cv_folds: int) -> Dict[str, Any]:
    """
    [FIX 3] Pipeline으로 CV 내부에서 scaler fit → data leakage 완전 차단
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if method == 'lda':
        n_classes = len(le.classes_)
        n_features = X_raw.shape[1]
        max_components = min(n_features, n_classes - 1)
        n_components = params.get('n_components')
        if n_components is None or n_components > max_components:
            n_components = max_components if max_components > 0 else None

        shrinkage = _parse_shrinkage(params.get('shrinkage'))
        solver = params.get('solver', 'svd')
        if shrinkage is not None and solver == 'svd':
            solver = 'lsqr'

        base_model = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            n_components=n_components,
            priors=params.get('priors')
        )
    else:
        base_model = QuadraticDiscriminantAnalysis(
            reg_param=params.get('reg_param', 0.0),
            priors=params.get('priors')
        )

    # [FIX] Pipeline: scaler는 각 fold의 train set에서만 fit
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])

    scores = cross_val_score(pipeline, X_raw, y_encoded, cv=cv_folds, scoring='accuracy')

    return {
        'cv_scores': [_to_native_type(s) for s in scores],
        'cv_mean': _to_native_type(np.mean(scores)),
        'cv_std': _to_native_type(np.std(scores)),
        'cv_folds': cv_folds
    }


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def generate_feature_importance_plot(importance_data: List[Dict], top_n: int = 20) -> str:
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_data[:top_n]) * 0.4)))
    top_features = importance_data[:top_n]
    features = [d['feature'] for d in top_features][::-1]
    importances = [d['importance'] for d in top_features][::-1]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Discriminant Analysis Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{imp:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_confusion_matrix_plot(cm: List[List[int]], class_labels: List[str]) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_array = np.array(cm)
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_roc_plot(roc_data: Dict) -> Optional[str]:
    plot_data = {k: v for k, v in roc_data.items() if k != '__macro_auc__'}
    if not plot_data:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_data)))
    for (label, data), color in zip(plot_data.items(), colors):
        ax.plot(data['fpr'], data['tpr'], color=color, linewidth=2,
                label=f'{label} (AUC = {data["auc"]:.3f})')
    if '__macro_auc__' in roc_data:
        ax.plot([], [], ' ', label=f'Macro AUC = {roc_data["__macro_auc__"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.15)
    return _fig_to_base64(fig)


def generate_lda_projection_plot(lda_transform: Dict, random_state: int = 42) -> str:
    X = np.array(lda_transform['X'])
    y = np.array(lda_transform['y'])
    class_labels = lda_transform['class_labels']
    n_components = lda_transform['n_components']
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_labels)))
    rng = np.random.RandomState(random_state)
    for i, (label, color) in enumerate(zip(class_labels, colors)):
        mask = y == i
        if n_components >= 2:
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], label=label,
                       alpha=0.7, s=50, edgecolors='white')
        else:
            jitter = rng.randn(mask.sum()) * 0.1
            ax.scatter(X[mask, 0], np.zeros(mask.sum()) + jitter,
                       c=[color], label=label, alpha=0.7, s=50, edgecolors='white')
    ax.set_xlabel('LD1', fontsize=11)
    if n_components >= 2:
        ax.set_ylabel('LD2', fontsize=11)
    else:
        ax.set_ylabel('', fontsize=11)
        ax.set_yticks([])
    ax.set_title('LDA Projection', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_class_separation_plot(info: Dict, class_labels: List[str], feature_names: List[str]) -> Optional[str]:
    if 'class_means' not in info:
        return None
    means = np.array(info['class_means'])
    n_classes, n_features = means.shape
    if n_features > 8:
        variance = np.var(means, axis=0)
        top_indices = np.argsort(variance)[-8:]
        means = means[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        n_features = 8
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_features)
    width = 0.8 / n_classes
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (label, color) in enumerate(zip(class_labels, colors)):
        offset = (i - n_classes / 2 + 0.5) * width
        ax.bar(x + offset, means[i], width, label=label, color=color, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Features', fontsize=11)
    ax.set_ylabel('Mean Value (Standardized)', fontsize=11)
    ax.set_title('Class Means by Feature', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, method: str, feature_importance: List[Dict]) -> Dict[str, Any]:
    key_insights = []
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

    method_name = "Linear Discriminant Analysis (LDA)" if method == 'lda' else "Quadratic Discriminant Analysis (QDA)"

    key_insights.append({
        'title': 'Classification Performance',
        'description': f'{perf_desc}. Accuracy: {accuracy:.1%}, F1-macro: {f1:.3f}',
        'status': status
    })

    if 'auc' in result['metrics']:
        auc_val = result['metrics']['auc']
        key_insights.append({
            'title': 'AUC Score (Macro)',
            'description': f'Macro-average ROC-AUC: {auc_val:.3f}. {"Excellent" if auc_val > 0.9 else "Good" if auc_val > 0.7 else "Fair"} discrimination.',
            'status': 'positive' if auc_val > 0.8 else 'neutral'
        })

    if method == 'lda' and 'lda_statistics' in result:
        lda_stats = result['lda_statistics']
        # Wilks Lambda insight
        if lda_stats.get('wilks_lambda'):
            wl = lda_stats['wilks_lambda']
            p_str = '< 0.001' if wl['p_value'] < 0.001 else f"= {wl['p_value']:.3f}"
            sig_str = 'Discriminant functions are significant.' if wl['significant'] else 'Not statistically significant.'
            key_insights.append({
                'title': "Wilks' Lambda",
                'description': f"Λ = {wl['lambda']:.4f}, χ²({wl['df']}) = {wl['chi2']:.3f}, p {p_str}. {sig_str}",
                'status': 'positive' if wl['significant'] else 'warning'
            })
        # Eigenvalues insight
        if lda_stats.get('eigenvalues'):
            ev = lda_stats['eigenvalues'][0]
            key_insights.append({
                'title': 'Canonical Correlation',
                'description': f"LD1 explains {ev['variance_explained_pct']:.1f}% of between-class variance (canonical r = {ev['canonical_correlation']:.3f}).",
                'status': 'positive' if ev['canonical_correlation'] > 0.7 else 'neutral'
            })

    if method == 'lda' and 'lda_info' in result:
        lda_info = result['lda_info']
        if 'explained_variance_ratio' in lda_info:
            evr = lda_info['explained_variance_ratio']
            total_var = sum(evr)
            key_insights.append({
                'title': 'Explained Variance',
                'description': f'First {len(evr)} discriminant(s) explain {total_var:.1%} of between-class variance.',
                'status': 'positive' if total_var > 0.9 else 'neutral'
            })
        if lda_info.get('solver_used') and lda_info['solver_used'] != result.get('params_requested_solver', lda_info['solver_used']):
            key_insights.append({
                'title': 'Solver Auto-adjusted',
                'description': f'Solver changed to "{lda_info["solver_used"]}" because shrinkage requires lsqr or eigen.',
                'status': 'neutral'
            })

    if method == 'qda' and 'qda_info' in result:
        qda_info = result['qda_info']
        if qda_info.get('reg_param_used', 0) > 0:
            key_insights.append({
                'title': 'Regularization Applied',
                'description': f'reg_param auto-set to {qda_info["reg_param_used"]:.3f} to prevent singular matrix.',
                'status': 'neutral'
            })

    top_features = feature_importance[:min(3, len(feature_importance))]
    feature_str = ', '.join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
    key_insights.append({
        'title': 'Key Discriminators',
        'description': f'Most discriminative features: {feature_str}',
        'status': 'neutral'
    })

    return {
        'key_insights': key_insights,
        'recommendation': (
            f'{method_name} model trained successfully. '
            f'{"Consider QDA if class boundaries are non-linear." if method == "lda" else "Consider LDA if you have limited data or want simpler boundaries."}'
        )
    }


# ──────────────────────────────────────────────
# Main endpoint
# ──────────────────────────────────────────────

@router.post("/lda")
async def run_discriminant_analysis(request: DiscriminantAnalysisRequest) -> Dict[str, Any]:
    """Train Discriminant Analysis model for classification."""
    try:
        data = request.data
        target_col = request.target_col
        feature_cols = request.feature_cols
        method = request.method.lower()

        if method not in ['lda', 'qda']:
            raise HTTPException(status_code=400, detail="Method must be 'lda' or 'qda'")
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")

        df = pd.DataFrame(data)
        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        categorical_features = []
        for col in X.columns:
            if X[col].dtype == 'object':
                unique_count = X[col].nunique()
                categorical_features.append({
                    'feature': col,
                    'unique_values': int(unique_count),
                    'note': 'Label encoded — ordinal assumption applied. Consider One-Hot Encoding for nominal variables.'
                })
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 50:
            raise HTTPException(status_code=400, detail="At least 50 valid samples required.")
        n_unique = y.nunique()
        if n_unique < 2:
            raise HTTPException(status_code=400, detail="Target must have at least 2 classes.")
        if n_unique > 50:
            raise HTTPException(status_code=400, detail="Target has too many unique values.")

        # [FIX] Data leakage 수정 — split 먼저, scaler는 train에만 fit
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=request.test_size,
            random_state=request.random_state,
            stratify=y
        )

        # [FIX 2] StandardScaler — train에만 fit, test는 transform만
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            columns=X.columns, index=X_train_raw.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test_raw),
            columns=X.columns, index=X_test_raw.index
        )

        parsed_shrinkage = _parse_shrinkage(request.shrinkage)
        params = {
            'solver': request.solver,
            'shrinkage': parsed_shrinkage,
            'n_components': request.n_components,
            'reg_param': request.reg_param,
            'priors': request.priors,
            'random_state': request.random_state
        }

        if method == 'lda':
            result = train_lda(X_train, X_test, y_train, y_test, params, feature_cols)
        else:
            result = train_qda(X_train, X_test, y_train, y_test, params, feature_cols)

        # [FIX 3] CV — 원본 X_raw + Pipeline으로 leakage 차단
        cv_result = perform_cross_validation(np.array(X), y, params, method, request.cv_folds)

        importance_plot = generate_feature_importance_plot(result['feature_importance'])
        cm_plot = generate_confusion_matrix_plot(result['confusion_matrix'], result['class_labels'])
        roc_plot = generate_roc_plot(result['roc_data']) if result['roc_data'] else None

        lda_projection_plot = None
        class_separation_plot = None
        if method == 'lda':
            if result.get('lda_transform'):
                lda_projection_plot = generate_lda_projection_plot(
                    result['lda_transform'], random_state=request.random_state
                )
            if result.get('lda_info'):
                class_separation_plot = generate_class_separation_plot(
                    result['lda_info'], result['class_labels'], feature_cols
                )
        elif method == 'qda':
            if result.get('qda_info'):
                class_separation_plot = generate_class_separation_plot(
                    result['qda_info'], result['class_labels'], feature_cols
                )

        interpretation = generate_interpretation(result, method, result['feature_importance'])

        actual_solver = 'N/A'
        if method == 'lda':
            actual_solver = result.get('lda_info', {}).get('solver_used', params['solver'])

        response = {
            'method': method.upper(),
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'n_classes': len(result['class_labels']),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'parameters': {
                'method': method.upper(),
                'solver': actual_solver if method == 'lda' else 'N/A',
                'shrinkage': str(parsed_shrinkage) if method == 'lda' else 'N/A',
                'n_components': params['n_components'] if method == 'lda' else 'N/A',
                'reg_param': result.get('qda_info', {}).get('reg_param_used', params['reg_param']) if method == 'qda' else 'N/A',
                'test_size': request.test_size,
                'cv_folds': request.cv_folds,
            },
            'metrics': result['metrics'],
            'feature_importance': result['feature_importance'],
            'cv_results': cv_result,
            'importance_plot': importance_plot,
            'cm_plot': cm_plot,
            'roc_plot': roc_plot,
            'lda_projection_plot': lda_projection_plot,
            'class_separation_plot': class_separation_plot,
            'per_class_metrics': result['per_class_metrics'],
            'confusion_matrix': result['confusion_matrix'],
            'class_labels': result['class_labels'],
            'interpretation': interpretation
        }

        if method == 'lda' and 'lda_info' in result:
            response['lda_info'] = result['lda_info']
        if method == 'lda' and 'lda_statistics' in result:
            response['lda_statistics'] = result['lda_statistics']
        elif method == 'qda' and 'qda_info' in result:
            response['qda_info'] = result['qda_info']

        response['data_warnings'] = {
            'categorical_features': categorical_features,
            'has_categorical': len(categorical_features) > 0
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discriminant Analysis failed: {str(e)}")
