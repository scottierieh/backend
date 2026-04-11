"""
Cross-Validation Analysis Router for FastAPI
Comprehensive k-fold cross-validation for model evaluation
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
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold, LeaveOneOut, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CrossValidationRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_col: str
    feature_cols: List[str]
    task_type: str = "auto"  # auto, classification, regression
    # Cross-validation parameters
    cv_method: str = "kfold"  # kfold, stratified, loocv, repeated_kfold, repeated_stratified
    n_folds: int = 5
    n_repeats: int = 3  # For repeated CV
    shuffle: bool = True
    # Model selection
    model_type: str = "auto"  # auto, logistic, decision_tree, random_forest, gradient_boosting, svm, knn, naive_bayes, ridge, lasso, elasticnet
    # Scoring
    scoring: str = "auto"  # auto, accuracy, f1, precision, recall, r2, rmse, mae
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


def get_cv_splitter(cv_method: str, n_folds: int, n_repeats: int, shuffle: bool, random_state: int, task_type: str):
    """Get cross-validation splitter based on method"""
    if cv_method == "loocv":
        return LeaveOneOut()
    elif cv_method == "stratified" and task_type == "classification":
        return StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
    elif cv_method == "repeated_kfold":
        return RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
    elif cv_method == "repeated_stratified" and task_type == "classification":
        return RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
    else:  # Default kfold
        return KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state if shuffle else None)


def get_model(model_type: str, task_type: str, random_state: int):
    """Get model based on type and task"""
    if task_type == "classification":
        models = {
            "logistic": LogisticRegression(max_iter=1000, random_state=random_state),
            "decision_tree": DecisionTreeClassifier(random_state=random_state),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
            "svm": SVC(kernel='rbf', probability=True, random_state=random_state),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "naive_bayes": GaussianNB()
        }
        default = "random_forest"
    else:
        models = {
            "ridge": Ridge(random_state=random_state),
            "lasso": Lasso(random_state=random_state),
            "elasticnet": ElasticNet(random_state=random_state),
            "decision_tree": DecisionTreeRegressor(random_state=random_state),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
            "svm": SVR(kernel='rbf'),
            "knn": KNeighborsRegressor(n_neighbors=5)
        }
        default = "random_forest"
    
    if model_type == "auto":
        return models[default], default
    elif model_type in models:
        return models[model_type], model_type
    else:
        return models[default], default


def get_scoring_metric(scoring: str, task_type: str) -> str:
    """Get sklearn scoring string"""
    if scoring == "auto":
        return "accuracy" if task_type == "classification" else "r2"
    
    scoring_map = {
        "accuracy": "accuracy",
        "f1": "f1_macro",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error"
    }
    return scoring_map.get(scoring, "accuracy" if task_type == "classification" else "r2")


def perform_cross_validation(X, y, model, cv_splitter, scoring: str, task_type: str) -> Dict[str, Any]:
    """Perform cross-validation and return detailed results"""
    # Get scores
    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
    
    # For negative metrics (RMSE, MAE), convert back to positive
    if scoring.startswith("neg_"):
        scores = -scores
    
    # Get predictions for each fold
    try:
        y_pred = cross_val_predict(model, X, y, cv=cv_splitter)
    except:
        y_pred = None
    
    # Calculate per-fold details
    fold_details = []
    fold_idx = 0
    for train_idx, test_idx in cv_splitter.split(X, y):
        fold_details.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': _to_native_type(scores[fold_idx]) if fold_idx < len(scores) else None
        })
        fold_idx += 1
        if fold_idx >= len(scores):
            break
    
    return {
        'scores': [_to_native_type(s) for s in scores],
        'mean': _to_native_type(np.mean(scores)),
        'std': _to_native_type(np.std(scores)),
        'min': _to_native_type(np.min(scores)),
        'max': _to_native_type(np.max(scores)),
        'median': _to_native_type(np.median(scores)),
        'n_folds': len(scores),
        'fold_details': fold_details,
        'y_pred': y_pred.tolist() if y_pred is not None else None
    }


def calculate_additional_metrics(y_true, y_pred, task_type: str) -> Dict[str, Any]:
    """Calculate additional metrics from cross-validated predictions"""
    if y_pred is None:
        return {}
    
    y_pred = np.array(y_pred)
    
    if task_type == "classification":
        return {
            'accuracy': _to_native_type(accuracy_score(y_true, y_pred)),
            'precision_macro': _to_native_type(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': _to_native_type(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': _to_native_type(f1_score(y_true, y_pred, average='macro', zero_division=0))
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mse': _to_native_type(mse),
            'rmse': _to_native_type(np.sqrt(mse)),
            'mae': _to_native_type(mean_absolute_error(y_true, y_pred)),
            'r2': _to_native_type(r2_score(y_true, y_pred))
        }


def generate_cv_scores_plot(cv_results: Dict) -> str:
    """Generate CV scores distribution plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scores = cv_results['scores']
    
    # Bar plot of fold scores
    ax1 = axes[0]
    folds = list(range(1, len(scores) + 1))
    colors = ['#22c55e' if s >= cv_results['mean'] else '#f59e0b' for s in scores]
    bars = ax1.bar(folds, scores, color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(y=cv_results['mean'], color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: {cv_results["mean"]:.4f}')
    ax1.fill_between([0, len(scores) + 1], 
                     cv_results['mean'] - cv_results['std'], 
                     cv_results['mean'] + cv_results['std'], 
                     color='#ef4444', alpha=0.1, label=f'±1 Std: {cv_results["std"]:.4f}')
    ax1.set_xlabel('Fold', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Cross-Validation Scores by Fold', fontsize=13, fontweight='bold')
    ax1.set_xticks(folds)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(scores, patch_artist=True)
    bp['boxes'][0].set_facecolor('#22c55e')
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color('#ef4444')
    bp['medians'][0].set_linewidth(2)
    
    # Add individual points
    ax2.scatter([1] * len(scores), scores, color='#166534', alpha=0.6, s=50, zorder=3)
    
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(['CV Scores'])
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Mean: {cv_results["mean"]:.4f}\nStd: {cv_results["std"]:.4f}\nMin: {cv_results["min"]:.4f}\nMax: {cv_results["max"]:.4f}'
    ax2.text(1.3, cv_results['mean'], stats_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_learning_stability_plot(cv_results: Dict) -> str:
    """Generate learning stability visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores = cv_results['scores']
    n_folds = len(scores)
    
    # Cumulative mean
    cumulative_means = [np.mean(scores[:i+1]) for i in range(n_folds)]
    cumulative_stds = [np.std(scores[:i+1]) if i > 0 else 0 for i in range(n_folds)]
    
    folds = list(range(1, n_folds + 1))
    
    ax.plot(folds, cumulative_means, 'o-', color='#22c55e', linewidth=2, markersize=8, label='Cumulative Mean')
    ax.fill_between(folds,
                    [m - s for m, s in zip(cumulative_means, cumulative_stds)],
                    [m + s for m, s in zip(cumulative_means, cumulative_stds)],
                    color='#22c55e', alpha=0.2, label='±1 Std')
    
    ax.axhline(y=cv_results['mean'], color='#ef4444', linestyle='--', linewidth=2, label=f'Final Mean: {cv_results["mean"]:.4f}')
    
    ax.set_xlabel('Number of Folds', fontsize=11)
    ax.set_ylabel('Cumulative Mean Score', fontsize=11)
    ax.set_title('Learning Stability Across Folds', fontsize=13, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_confusion_matrix_plot(y_true, y_pred, class_labels: List[str]) -> str:
    """Generate confusion matrix from CV predictions"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix (Cross-Validated)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_regression_plot(y_true, y_pred) -> str:
    """Generate actual vs predicted plot for regression"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, color='#22c55e', s=30)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted (Cross-Validated)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Residuals
    ax2 = axes[1]
    residuals = np.array(y_true) - np.array(y_pred)
    ax2.scatter(y_pred, residuals, alpha=0.5, color='#16a34a', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(cv_results: Dict, task_type: str, additional_metrics: Dict, model_name: str, cv_method: str) -> Dict[str, Any]:
    """Generate interpretation of Cross-Validation results"""
    key_insights = []
    
    mean_score = cv_results['mean']
    std_score = cv_results['std']
    
    # Model performance
    if task_type == 'classification':
        if mean_score >= 0.9:
            status = 'positive'
            perf_desc = 'Excellent classification performance'
        elif mean_score >= 0.7:
            status = 'neutral'
            perf_desc = 'Good classification performance'
        else:
            status = 'warning'
            perf_desc = 'Model may need improvement'
        
        key_insights.append({
            'title': 'CV Performance',
            'description': f'{perf_desc}. Mean CV Score: {mean_score:.1%} ± {std_score:.1%}',
            'status': status
        })
    else:
        if mean_score >= 0.8:
            status = 'positive'
            perf_desc = 'Excellent fit'
        elif mean_score >= 0.5:
            status = 'neutral'
            perf_desc = 'Moderate fit'
        else:
            status = 'warning'
            perf_desc = 'Weak fit'
        
        key_insights.append({
            'title': 'CV Performance',
            'description': f'{perf_desc}. Mean R² = {mean_score:.3f} ± {std_score:.3f}',
            'status': status
        })
    
    # Model stability
    cv_coefficient = std_score / mean_score if mean_score != 0 else float('inf')
    if cv_coefficient < 0.1:
        stability_status = 'positive'
        stability_desc = 'Very stable'
    elif cv_coefficient < 0.2:
        stability_status = 'neutral'
        stability_desc = 'Stable'
    else:
        stability_status = 'warning'
        stability_desc = 'High variance'
    
    key_insights.append({
        'title': 'Model Stability',
        'description': f'{stability_desc}. Coefficient of variation: {cv_coefficient:.2%}. Std/Mean ratio indicates {"consistent" if cv_coefficient < 0.15 else "variable"} performance across folds.',
        'status': stability_status
    })
    
    # Score range
    score_range = cv_results['max'] - cv_results['min']
    key_insights.append({
        'title': 'Score Range',
        'description': f'Scores range from {cv_results["min"]:.3f} to {cv_results["max"]:.3f} (range: {score_range:.3f}). Median: {cv_results["median"]:.3f}.',
        'status': 'neutral'
    })
    
    # CV method info
    method_names = {
        'kfold': 'K-Fold',
        'stratified': 'Stratified K-Fold',
        'loocv': 'Leave-One-Out',
        'repeated_kfold': 'Repeated K-Fold',
        'repeated_stratified': 'Repeated Stratified K-Fold'
    }
    key_insights.append({
        'title': 'Validation Method',
        'description': f'Used {method_names.get(cv_method, cv_method)} with {cv_results["n_folds"]} iterations. Model: {model_name.replace("_", " ").title()}.',
        'status': 'neutral'
    })
    
    # Recommendation
    if mean_score >= 0.8 and cv_coefficient < 0.15:
        recommendation = 'Cross-validation shows strong, stable performance. The model generalizes well and is ready for deployment.'
    elif mean_score >= 0.7:
        recommendation = 'Performance is acceptable. Consider hyperparameter tuning or feature engineering to improve results.'
    else:
        recommendation = 'Performance is below expectations. Try different models, add more features, or collect more data.'
    
    return {
        'key_insights': key_insights,
        'recommendation': recommendation
    }


@router.post("/cross-validation")
async def run_cross_validation_analysis(request: CrossValidationRequest) -> Dict[str, Any]:
    """
    Perform comprehensive cross-validation analysis.
    
    Supports:
    - Multiple CV methods (K-Fold, Stratified, LOOCV, Repeated)
    - Multiple model types
    - Classification and regression
    - Detailed fold-by-fold analysis
    """
    try:
        data = request.data
        target_col = request.target_col
        feature_cols = request.feature_cols
        task_type = request.task_type
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        all_cols = [target_col] + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing)}")
        
        # Prepare features
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical features
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Drop rows with NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 50:
            raise HTTPException(status_code=400, detail="At least 50 valid samples required.")
        
        # Auto-detect task type
        if task_type == 'auto':
            task_type = detect_task_type(y)
        
        # Encode target for classification
        target_encoder = None
        class_labels = None
        if task_type == 'classification':
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
            class_labels = [str(c) for c in target_encoder.classes_]
        else:
            y_encoded = y.values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get CV splitter
        cv_splitter = get_cv_splitter(
            request.cv_method, 
            request.n_folds, 
            request.n_repeats,
            request.shuffle, 
            request.random_state, 
            task_type
        )
        
        # Get model
        model, model_name = get_model(request.model_type, task_type, request.random_state)
        
        # Get scoring metric
        scoring = get_scoring_metric(request.scoring, task_type)
        
        # Perform cross-validation
        cv_results = perform_cross_validation(X_scaled, y_encoded, model, cv_splitter, scoring, task_type)
        
        # Calculate additional metrics from predictions
        additional_metrics = {}
        if cv_results['y_pred'] is not None:
            additional_metrics = calculate_additional_metrics(y_encoded, cv_results['y_pred'], task_type)
        
        # Generate visualizations
        scores_plot = generate_cv_scores_plot(cv_results)
        stability_plot = generate_learning_stability_plot(cv_results)
        
        if task_type == 'classification' and cv_results['y_pred'] is not None:
            cm_plot = generate_confusion_matrix_plot(y_encoded, cv_results['y_pred'], class_labels)
            regression_plot = None
        elif task_type == 'regression' and cv_results['y_pred'] is not None:
            cm_plot = None
            regression_plot = generate_regression_plot(y_encoded, cv_results['y_pred'])
        else:
            cm_plot = None
            regression_plot = None
        
        # Generate interpretation
        interpretation = generate_interpretation(cv_results, task_type, additional_metrics, model_name, request.cv_method)
        
        # Prepare parameters
        parameters = {
            'cv_method': request.cv_method,
            'n_folds': request.n_folds,
            'n_repeats': request.n_repeats if request.cv_method in ['repeated_kfold', 'repeated_stratified'] else None,
            'shuffle': request.shuffle,
            'model_type': model_name,
            'scoring': scoring,
            'random_state': request.random_state
        }
        
        # Prepare response
        response = {
            'task_type': task_type,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'parameters': parameters,
            'cv_results': cv_results,
            'additional_metrics': additional_metrics,
            'scores_plot': scores_plot,
            'stability_plot': stability_plot,
            'interpretation': interpretation
        }
        
        if task_type == 'classification':
            response['class_labels'] = class_labels
            response['cm_plot'] = cm_plot
        else:
            response['regression_plot'] = regression_plot
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-validation analysis failed: {str(e)}")
