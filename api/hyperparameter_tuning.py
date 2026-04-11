"""
Hyperparameter Tuning FastAPI Endpoint
Automatic optimization of ML model hyperparameters
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

router = APIRouter()


class HyperparameterTuningRequest(BaseModel):
    """Request model for hyperparameter tuning"""
    target_column: str = Field(..., description="Name of target column")
    model_type: str = Field(..., pattern="^(random_forest|xgboost|svm|logistic|ridge|lasso|mlp)$")
    task_type: str = Field(..., pattern="^(classification|regression)$")
    search_method: str = Field(..., pattern="^(grid|random|optuna)$")
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter: int = Field(default=20, ge=5, le=100, description="Number of iterations for random/optuna search")
    param_grid: Optional[Dict[str, Any]] = Field(default=None, description="Custom parameter grid")


class HyperparameterTuningResponse(BaseModel):
    """Response model for hyperparameter tuning"""
    success: bool
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    dataset_info: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]


def get_default_param_grid(model_type: str, task_type: str, search_method: str) -> Dict[str, Any]:
    """Get default parameter grid for model"""
    
    if model_type == "random_forest":
        if search_method == "grid":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        else:  # random or optuna
            return {
                "n_estimators": (50, 500),
                "max_depth": (3, 30),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            }
    
    elif model_type == "xgboost":
        if search_method == "grid":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 0.8, 1.0]
            }
        else:
            return {
                "n_estimators": (50, 500),
                "max_depth": (3, 10),
                "learning_rate": (0.001, 0.3),
                "subsample": (0.5, 1.0)
            }
    
    elif model_type == "svm":
        if search_method == "grid":
            return {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        else:
            return {
                "C": (0.01, 100),
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
    
    elif model_type == "logistic":
        if search_method == "grid":
            return {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            }
        else:
            return {
                "C": (0.001, 100),
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            }
    
    elif model_type in ["ridge", "lasso"]:
        if search_method == "grid":
            return {
                "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
            }
        else:
            return {
                "alpha": (0.001, 1000)
            }
    
    elif model_type == "mlp":
        if search_method == "grid":
            return {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"]
            }
        else:
            return {
                "hidden_layer_sizes": [(10, 200)],  # Range for layer size
                "activation": ["relu", "tanh", "logistic"],
                "alpha": (0.0001, 0.1),
                "learning_rate": ["constant", "adaptive"]
            }


def get_model(model_type: str, task_type: str, **params):
    """Get model instance with parameters"""
    
    if model_type == "random_forest":
        if task_type == "classification":
            return RandomForestClassifier(**params, random_state=42)
        else:
            return RandomForestRegressor(**params, random_state=42)
    
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not installed")
        if task_type == "classification":
            return xgb.XGBClassifier(**params, random_state=42)
        else:
            return xgb.XGBRegressor(**params, random_state=42)
    
    elif model_type == "svm":
        if task_type == "classification":
            return SVC(**params, random_state=42)
        else:
            return SVR(**params)
    
    elif model_type == "logistic":
        return LogisticRegression(**params, random_state=42, max_iter=1000)
    
    elif model_type == "ridge":
        return Ridge(**params, random_state=42)
    
    elif model_type == "lasso":
        return Lasso(**params, random_state=42)
    
    elif model_type == "mlp":
        if task_type == "classification":
            return MLPClassifier(**params, random_state=42, max_iter=1000)
        else:
            return MLPRegressor(**params, random_state=42, max_iter=1000)


def run_grid_search(X, y, model_type: str, task_type: str, param_grid: Dict, cv_folds: int):
    """Run grid search"""
    base_model = get_model(model_type, task_type)
    
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X, y)
    
    results = []
    for i in range(len(grid_search.cv_results_["params"])):
        results.append({
            "params": grid_search.cv_results_["params"][i],
            "mean_score": grid_search.cv_results_["mean_test_score"][i],
            "std_score": grid_search.cv_results_["std_test_score"][i]
        })
    
    return grid_search.best_params_, grid_search.best_score_, results


def run_random_search(X, y, model_type: str, task_type: str, param_distributions: Dict, cv_folds: int, n_iter: int):
    """Run random search"""
    base_model = get_model(model_type, task_type)
    
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
    
    # Convert range tuples to scipy distributions
    from scipy.stats import uniform, randint
    
    processed_distributions = {}
    for key, value in param_distributions.items():
        if isinstance(value, tuple) and len(value) == 2:
            if isinstance(value[0], int):
                processed_distributions[key] = randint(value[0], value[1])
            else:
                processed_distributions[key] = uniform(value[0], value[1] - value[0])
        else:
            processed_distributions[key] = value
    
    random_search = RandomizedSearchCV(
        base_model,
        processed_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X, y)
    
    results = []
    for i in range(len(random_search.cv_results_["params"])):
        results.append({
            "params": random_search.cv_results_["params"][i],
            "mean_score": random_search.cv_results_["mean_test_score"][i],
            "std_score": random_search.cv_results_["std_test_score"][i]
        })
    
    return random_search.best_params_, random_search.best_score_, results


def run_optuna_search(X, y, model_type: str, task_type: str, param_space: Dict, cv_folds: int, n_iter: int):
    """Run Optuna Bayesian optimization"""
    
    if not OPTUNA_AVAILABLE:
        raise ValueError("Optuna not installed")
    
    def objective(trial):
        params = {}
        
        for param_name, param_range in param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        
        model = get_model(model_type, task_type, **params)
        
        scores = cross_val_score(model, X, y, cv=cv_folds, 
                                scoring="accuracy" if task_type == "classification" else "neg_mean_squared_error")
        
        return scores.mean()
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_iter, show_progress_bar=False)
    
    results = []
    for trial in study.trials:
        results.append({
            "params": trial.params,
            "mean_score": trial.value,
            "std_score": 0.0  # Optuna doesn't provide std by default
        })
    
    return study.best_params, study.best_value, results


def create_param_importance_plot(results: List[Dict], task_type: str) -> str:
    """Create parameter importance visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract scores
    scores = [r['mean_score'] for r in results]
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['mean_score'], reverse=True)
    top_10 = sorted_results[:min(10, len(sorted_results))]
    
    # Create bar plot
    param_labels = [f"Config {i+1}" for i in range(len(top_10))]
    scores_top = [r['mean_score'] for r in top_10]
    
    bars = ax.barh(param_labels, scores_top, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Highlight best
    bars[0].set_color('green')
    bars[0].set_alpha(0.8)
    
    for i, (bar, score) in enumerate(zip(bars, scores_top)):
        ax.text(score, bar.get_y() + bar.get_height()/2, 
               f'{score:.4f}',
               va='center', ha='left', fontsize=10, weight='bold')
    
    ax.set_xlabel('Score', fontsize=12, weight='bold')
    ax.set_title('Top 10 Parameter Configurations', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_convergence_plot(results: List[Dict]) -> str:
    """Create convergence plot showing optimization progress"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scores = [r['mean_score'] for r in results]
    iterations = list(range(1, len(scores) + 1))
    
    # Running best
    running_best = []
    current_best = float('-inf')
    for score in scores:
        current_best = max(current_best, score)
        running_best.append(current_best)
    
    ax.plot(iterations, scores, 'o-', alpha=0.6, label='Trial Score', linewidth=1.5)
    ax.plot(iterations, running_best, 'r-', linewidth=2.5, label='Best Score', alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Score', fontsize=12, weight='bold')
    ax.set_title('Optimization Convergence', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_learning_curve(X, y, model, task_type: str) -> str:
    """Create learning curve for best model"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy" if task_type == "classification" else "neg_mean_squared_error"
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    
    ax.set_xlabel('Training Examples', fontsize=12, weight='bold')
    ax.set_ylabel('Score', fontsize=12, weight='bold')
    ax.set_title('Learning Curve', fontsize=13, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def generate_interpretation(
    best_params: Dict,
    best_score: float,
    all_results: List[Dict],
    model_type: str,
    search_method: str,
    task_type: str
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Best score insight
    score_type = "Accuracy" if task_type == "classification" else "R² Score"
    key_insights.append({
        "title": f"Best {score_type}: {best_score:.4f}",
        "description": f"Optimal hyperparameters found using {search_method} search.",
        "status": "positive"
    })
    
    # Search efficiency
    n_trials = len(all_results)
    score_improvement = best_score - min(r['mean_score'] for r in all_results)
    
    key_insights.append({
        "title": f"Search Efficiency",
        "description": f"Explored {n_trials} configurations. Score improved by {score_improvement:.4f} from worst to best.",
        "status": "neutral"
    })
    
    # Model-specific insights
    if model_type == "random_forest":
        if 'n_estimators' in best_params:
            key_insights.append({
                "title": f"Optimal Trees: {best_params['n_estimators']}",
                "description": "Number of trees in the forest. More trees generally improve performance but increase computation.",
                "status": "neutral"
            })
    
    elif model_type == "xgboost":
        if 'learning_rate' in best_params:
            lr = best_params['learning_rate']
            if lr < 0.1:
                recommendations.append("Low learning rate found - model learns slowly but carefully")
            elif lr > 0.3:
                recommendations.append("High learning rate - model learns quickly but may miss optimal solution")
    
    # Recommendations
    recommendations.append(f"Best model: {model_type} with {search_method} search")
    recommendations.append(f"Tested {n_trials} different configurations")
    
    if search_method == "grid":
        recommendations.append("Grid search tested all combinations - results are exhaustive")
    elif search_method == "random":
        recommendations.append("Random search sampled parameter space - consider increasing n_iter for better results")
    else:  # optuna
        recommendations.append("Bayesian optimization efficiently explored parameter space")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/hyperparameter-tuning")
async def tune_hyperparameters(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    model_type: str = Form(...),
    task_type: str = Form(...),
    search_method: str = Form(...),
    cv_folds: int = Form(5),
    n_iter: int = Form(20),
    param_grid: Optional[str] = Form(None)
):
    """
    Tune ML model hyperparameters using Grid Search, Random Search, or Bayesian Optimization
    
    Upload CSV file and specify tuning parameters
    """
    try:
        # Build request object
        request = HyperparameterTuningRequest(
            target_column=target_column,
            model_type=model_type,
            task_type=task_type,
            search_method=search_method,
            cv_folds=cv_folds,
            n_iter=n_iter,
            param_grid=json.loads(param_grid) if param_grid else None
        )
        
        # Read CSV
        df = pd.read_csv(file.file)
        
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
        
        # Prepare data
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]
        
        # Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target for classification
        if request.task_type == "classification":
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
        
        # Get parameter grid
        param_grid = request.param_grid or get_default_param_grid(
            request.model_type, 
            request.task_type, 
            request.search_method
        )
        
        # Run search
        if request.search_method == "grid":
            best_params, best_score, results = run_grid_search(
                X, y, request.model_type, request.task_type, param_grid, request.cv_folds
            )
        elif request.search_method == "random":
            best_params, best_score, results = run_random_search(
                X, y, request.model_type, request.task_type, param_grid, 
                request.cv_folds, request.n_iter
            )
        else:  # optuna
            best_params, best_score, results = run_optuna_search(
                X, y, request.model_type, request.task_type, param_grid,
                request.cv_folds, request.n_iter
            )
        
        # Train best model for learning curve
        best_model = get_model(request.model_type, request.task_type, **best_params)
        
        # Generate plots
        plots = {
            "param_importance": create_param_importance_plot(results, request.task_type),
            "convergence": create_convergence_plot(results),
            "learning_curve": create_learning_curve(X, y, best_model, request.task_type)
        }
        
        # Generate interpretation
        interpretation = generate_interpretation(
            best_params, best_score, results,
            request.model_type, request.search_method, request.task_type
        )
        
        # Sort results by score (highest first)
        sorted_results = sorted(results, key=lambda x: x['mean_score'], reverse=True)
        
        return HyperparameterTuningResponse(
            success=True,
            best_params=best_params,
            best_score=float(best_score),
            all_results=sorted_results[:50],  # Top 50 sorted results
            model_info={
                "model_type": request.model_type,
                "task_type": request.task_type,
                "search_method": request.search_method,
                "n_trials": len(results),
                "cv_folds": request.cv_folds
            },
            dataset_info={
                "n_samples": len(df),
                "n_features": X.shape[1],
                "feature_names": list(X.columns),
                "target_name": request.target_column
            },
            plots=plots,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tuning error: {str(e)}")
