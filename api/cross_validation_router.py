"""
Cross-Validation Analysis Router for FastAPI
Model evaluation with multiple CV strategies
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
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    LeaveOneOut, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CVRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_col: str
    feature_cols: List[str]
    task_type: str = "auto"  # auto, classification, regression
    cv_strategy: str = "kfold"  # kfold, stratified, loo
    n_splits: int = 5
    model_type: str = "rf"  # rf, lr, svm
    random_state: int = 42


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
    if isinstance(obj, pd.Series):
        return obj.to_list()
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


def generate_cv_plot(cv_scores: List[float], cv_mean: float, cv_std: float) -> str:
    """Generate CV scores plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = range(1, len(cv_scores) + 1)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(cv_scores)))
    
    bars = ax.bar(folds, cv_scores, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    ax.axhline(y=cv_mean, color='#dc2626', linestyle='--', linewidth=2.5, label=f'Mean: {cv_mean:.3f}')
    ax.axhline(y=cv_mean + cv_std, color='#9ca3af', linestyle=':', linewidth=2, label=f'±1 Std: {cv_std:.3f}')
    ax.axhline(y=cv_mean - cv_std, color='#9ca3af', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Scores by Fold', fontsize=13, fontweight='bold')
    ax.set_xticks(folds)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.legend(loc='best')
    ax.set_ylim([min(cv_scores) - 0.1, max(cv_scores) + 0.1])
    
    # Add value labels on bars
    for bar, score in zip(bars, cv_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_distribution_plot(cv_scores: List[float]) -> str:
    """Generate CV scores distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(cv_scores, bins=max(3, len(cv_scores)//2), color='#3b82f6', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(cv_scores), color='#dc2626', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cv_scores):.3f}')
    ax1.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of CV Scores', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Box plot
    bp = ax2.boxplot(cv_scores, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3b82f6')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot of CV Scores', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def get_model(model_type: str, task_type: str, random_state: int):
    """Get model based on type"""
    if task_type == 'classification':
        if model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        elif model_type == 'lr':
            return LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == 'svm':
            return SVC(kernel='rbf', random_state=random_state)
    else:
        if model_type == 'rf':
            return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        elif model_type == 'lr':
            return LinearRegression()
        elif model_type == 'svm':
            return SVR(kernel='rbf')
    
    return None


def generate_interpretation(cv_scores: List[float], cv_mean: float, cv_std: float, model_type: str) -> Dict[str, Any]:
    """Generate interpretation of CV results"""
    key_insights = []
    
    # Model stability
    cv_coefficient_variation = cv_std / cv_mean if cv_mean > 0 else 0
    
    if cv_coefficient_variation < 0.05:
        stability = 'Excellent'
        status = 'positive'
    elif cv_coefficient_variation < 0.10:
        stability = 'Good'
        status = 'positive'
    elif cv_coefficient_variation < 0.15:
        stability = 'Fair'
        status = 'neutral'
    else:
        stability = 'Poor'
        status = 'warning'
    
    key_insights.append({
        'title': 'Model Stability',
        'description': f'CV Std/Mean ratio: {cv_coefficient_variation:.3f} — {stability} stability across folds',
        'status': status
    })
    
    # Performance quality
    if cv_mean >= 0.9:
        perf_status = 'positive'
        perf_desc = 'Excellent'
    elif cv_mean >= 0.8:
        perf_status = 'positive'
        perf_desc = 'Good'
    elif cv_mean >= 0.7:
        perf_status = 'neutral'
        perf_desc = 'Fair'
    else:
        perf_status = 'warning'
        perf_desc = 'Needs improvement'
    
    key_insights.append({
        'title': 'Model Performance',
        'description': f'Mean CV score: {cv_mean:.3f} — {perf_desc} predictive ability',
        'status': perf_status
    })
    
    # Variance analysis
    min_score = min(cv_scores)
    max_score = max(cv_scores)
    score_range = max_score - min_score
    
    key_insights.append({
        'title': 'Score Range',
        'description': f'Min: {min_score:.3f}, Max: {max_score:.3f}, Range: {score_range:.3f}',
        'status': 'neutral'
    })
    
    # Model type info
    model_names = {'rf': 'Random Forest', 'lr': 'Logistic Regression' if 'clf' in str(type(None)).lower() else 'Linear Regression', 'svm': 'Support Vector Machine'}
    key_insights.append({
        'title': 'Model Type',
        'description': f'{model_names.get(model_type, model_type)} with {len(cv_scores)}-fold cross-validation',
        'status': 'neutral'
    })
    
    return {
        'key_insights': key_insights,
        'recommendation': 'Review fold-wise performance. High variance indicates need for more data or feature engineering.'
    }


@router.post("/cross-validation")
async def run_cross_validation_analysis(request: CVRequest) -> Dict[str, Any]:
    """
    Perform Cross-Validation analysis.
    
    Supports:
    - K-Fold CV
    - Stratified K-Fold
    - Leave-One-Out
    - Multiple model types
    """
    try:
        data = request.data
        target_col = request.target_col
        feature_cols = request.feature_cols
        task_type = request.task_type
        cv_strategy = request.cv_strategy
        n_splits = request.n_splits
        model_type = request.model_type
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Drop NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            raise HTTPException(status_code=400, detail="At least 10 samples required.")
        
        # Auto-detect task type
        if task_type == 'auto':
            task_type = detect_task_type(y)
        
        # Encode y for classification
        if task_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get model
        model = get_model(model_type, task_type, request.random_state)
        if model is None:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Select CV strategy
        if cv_strategy == 'stratified':
            if task_type == 'classification':
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=request.random_state)
            else:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=request.random_state)
        elif cv_strategy == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=request.random_state)
        
        # Determine scoring metric
        if task_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        # Perform CV
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring=scoring)
        
        # Generate visualizations
        cv_plot = generate_cv_plot(cv_scores, np.mean(cv_scores), np.std(cv_scores))
        dist_plot = generate_distribution_plot(cv_scores)
        
        # Interpretation
        interpretation = generate_interpretation(cv_scores, np.mean(cv_scores), np.std(cv_scores), model_type)
        
        # Fold details
        fold_details = []
        for i, score in enumerate(cv_scores, 1):
            fold_details.append({
                'fold': i,
                'score': _to_native_type(score),
                'status': 'good' if score >= np.mean(cv_scores) else 'below_avg'
            })
        
        # Response
        response = {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'task_type': task_type,
            'model_type': model_type,
            'cv_strategy': cv_strategy,
            'n_splits': len(cv_scores),
            'cv_scores': [_to_native_type(s) for s in cv_scores],
            'cv_mean': _to_native_type(np.mean(cv_scores)),
            'cv_std': _to_native_type(np.std(cv_scores)),
            'cv_min': _to_native_type(np.min(cv_scores)),
            'cv_max': _to_native_type(np.max(cv_scores)),
            'cv_median': _to_native_type(np.median(cv_scores)),
            'fold_details': fold_details,
            'cv_plot': cv_plot,
            'distribution_plot': dist_plot,
            'interpretation': interpretation
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-validation failed: {str(e)}")
