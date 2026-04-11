
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.inspection import permutation_importance
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()

class FeatureImportanceRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    dependent: str = Field(...)
    independents: List[str] = Field(...)
    task_type: str = Field(default="regression")
    model_type: str = Field(default="random_forest")
    n_repeats: int = Field(default=10)
    test_size: float = Field(default=0.2)
    standardize: bool = Field(default=False)

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj

def get_model(model_type, task_type):
    if task_type == 'regression':
        models = {
            'linear': LinearRegression(), 
            'ridge': Ridge(), 
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42), 
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeRegressor(random_state=42)
        }
    else:
        models = {
            'logistic': LogisticRegression(max_iter=1000), 
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42), 
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
    return models.get(model_type, list(models.values())[0])

@router.post("/feature-importance")
def feature_importance(req: FeatureImportanceRequest):
    try:
        df = pd.DataFrame(req.data)
        dep = req.dependent
        indeps = req.independents
        task = req.task_type
        
        df_clean = df[[dep] + indeps].dropna()
        X = df_clean[indeps].values.astype(float)
        y = df_clean[dep].values
        
        classes = None
        if task == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
            classes = le.classes_.tolist()
        else:
            y = y.astype(float)
        
        n_observations = len(X)
        
        # Standardize if requested
        scaler = None
        if req.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=42, 
            stratify=y if task == 'classification' else None
        )
        
        n_train = len(X_train)
        n_test = len(X_test)
        
        model = get_model(req.model_type, task)
        model.fit(X_train, y_train)
        
        # Calculate scores
        if task == 'regression':
            train_score = r2_score(y_train, model.predict(X_train))
            test_score = r2_score(y_test, model.predict(X_test))
            scoring = 'r2'
            metric = 'R²'
        else:
            train_score = accuracy_score(y_train, model.predict(X_train))
            test_score = accuracy_score(y_test, model.predict(X_test))
            scoring = 'accuracy'
            metric = 'Accuracy'
        
        # Permutation importance
        result = permutation_importance(
            model, X_test, y_test, 
            n_repeats=req.n_repeats, 
            random_state=42, 
            scoring=scoring
        )
        
        # Build permutation_importance list
        permutation_importance_list = []
        for i in range(len(indeps)):
            permutation_importance_list.append({
                'feature': indeps[i],
                'importance_mean': float(result.importances_mean[i]),
                'importance_std': float(result.importances_std[i]),
                'importance_values': [float(v) for v in result.importances[i]]
            })
        permutation_importance_list.sort(key=lambda x: x['importance_mean'], reverse=True)
        
        # Build feature_ranking
        total_importance = sum(max(0, d['importance_mean']) for d in permutation_importance_list)
        feature_ranking = []
        for rank, item in enumerate(permutation_importance_list, 1):
            relative = (max(0, item['importance_mean']) / total_importance * 100) if total_importance > 0 else 0
            feature_ranking.append({
                'rank': rank,
                'feature': item['feature'],
                'importance': item['importance_mean'],
                'std': item['importance_std'],
                'relative_importance': relative
            })
        
        # Baseline importance
        baseline_importance = []
        if hasattr(model, 'feature_importances_'):
            for i, f in enumerate(indeps):
                baseline_importance.append({
                    'feature': f, 
                    'importance': float(model.feature_importances_[i]),
                    'type': 'tree_importance'
                })
            baseline_importance.sort(key=lambda x: x['importance'], reverse=True)
        elif hasattr(model, 'coef_'):
            coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            for i, f in enumerate(indeps):
                baseline_importance.append({
                    'feature': f, 
                    'importance': float(abs(coefs[i])), 
                    'coefficient': float(coefs[i]),
                    'type': 'coefficient'
                })
            baseline_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Generate insights
        insights = []
        top_feature = feature_ranking[0] if feature_ranking else None
        
        if top_feature:
            insights.append({
                'type': 'info',
                'title': f"Top Feature: {top_feature['feature']}",
                'description': f"This feature has the highest importance ({top_feature['importance']:.4f}), contributing {top_feature['relative_importance']:.1f}% of the total importance."
            })
        
        # Check for negative importance
        negative_features = [f for f in feature_ranking if f['importance'] < 0]
        if negative_features:
            insights.append({
                'type': 'warning',
                'title': 'Negative Importance Detected',
                'description': f"{len(negative_features)} feature(s) have negative importance, suggesting they may be adding noise to the model."
            })
        
        # Check model performance
        if task == 'regression':
            if test_score < 0.3:
                insights.append({
                    'type': 'warning',
                    'title': 'Low Model Performance',
                    'description': f"R² of {test_score:.3f} suggests the model explains limited variance. Feature importance may be less reliable."
                })
            elif test_score > 0.7:
                insights.append({
                    'type': 'info',
                    'title': 'Good Model Fit',
                    'description': f"R² of {test_score:.3f} indicates a good fit. Feature importance estimates are likely reliable."
                })
        else:
            if test_score < 0.6:
                insights.append({
                    'type': 'warning',
                    'title': 'Low Accuracy',
                    'description': f"Accuracy of {test_score:.1%} is relatively low. Consider feature engineering or different model."
                })
            elif test_score > 0.85:
                insights.append({
                    'type': 'info',
                    'title': 'High Accuracy',
                    'description': f"Accuracy of {test_score:.1%} indicates strong predictive performance."
                })
        
        # Check overfitting
        if train_score - test_score > 0.15:
            insights.append({
                'type': 'warning',
                'title': 'Potential Overfitting',
                'description': f"Train score ({train_score:.3f}) is notably higher than test score ({test_score:.3f}). Consider regularization or more data."
            })
        
        # Concentration of importance
        if len(feature_ranking) >= 3:
            top3_importance = sum(f['relative_importance'] for f in feature_ranking[:3])
            if top3_importance > 80:
                insights.append({
                    'type': 'info',
                    'title': 'Concentrated Importance',
                    'description': f"Top 3 features account for {top3_importance:.1f}% of importance. Consider focusing on these key predictors."
                })
        
        # Generate recommendations
        recommendations = []
        if top_feature:
            recommendations.append(f"Focus on '{top_feature['feature']}' as it has the highest predictive importance.")
        
        if negative_features:
            neg_names = [f['feature'] for f in negative_features[:3]]
            recommendations.append(f"Consider removing low-value features: {', '.join(neg_names)}")
        
        if len(feature_ranking) > 10:
            top_features = [f['feature'] for f in feature_ranking[:5]]
            recommendations.append(f"With {len(feature_ranking)} features, consider focusing on the top 5: {', '.join(top_features)}")
        
        if train_score - test_score > 0.1:
            recommendations.append("Consider using regularization or cross-validation to reduce overfitting.")
        
        recommendations.append(f"The permutation importance was computed with {req.n_repeats} repeats for stability.")
        
        # ===== PLOTS =====
        plots = {}
        
        # 1. Importance Bar Chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(indeps) * 0.4)))
        features = [d['feature'] for d in feature_ranking]
        means = [d['importance'] for d in feature_ranking]
        stds = [d['std'] for d in feature_ranking]
        colors = ['#55A868' if m > 0 else '#C44E52' for m in means]
        y_pos = np.arange(len(features))
        ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Importance (decrease in score when shuffled)')
        ax.set_title('Permutation Feature Importance', fontweight='bold')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        plots['importance_bar'] = base64.b64encode(buf.read()).decode('utf-8')
        
        # 2. Box Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(indeps) * 0.4)))
        box_data = [item['importance_values'] for item in sorted(permutation_importance_list, key=lambda x: x['importance_mean'], reverse=True)]
        box_labels = [item['feature'] for item in sorted(permutation_importance_list, key=lambda x: x['importance_mean'], reverse=True)]
        bp = ax.boxplot(box_data, vert=False, labels=box_labels, patch_artist=True)
        for patch, mean in zip(bp['boxes'], [item['importance_mean'] for item in sorted(permutation_importance_list, key=lambda x: x['importance_mean'], reverse=True)]):
            patch.set_facecolor('#55A868' if mean > 0 else '#C44E52')
            patch.set_alpha(0.7)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Importance')
        ax.set_title('Permutation Importance Distribution', fontweight='bold')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        plots['importance_box'] = base64.b64encode(buf.read()).decode('utf-8')
        
        # 3. Cumulative Importance
        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative = np.cumsum([f['relative_importance'] for f in feature_ranking])
        x_labels = [f['feature'] for f in feature_ranking]
        ax.bar(range(len(x_labels)), [f['relative_importance'] for f in feature_ranking], alpha=0.7, color='#4C72B0', label='Individual')
        ax.plot(range(len(x_labels)), cumulative, 'ro-', linewidth=2, markersize=6, label='Cumulative')
        ax.axhline(y=80, color='orange', linestyle='--', label='80% threshold')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Features')
        ax.set_ylabel('Relative Importance (%)')
        ax.set_title('Cumulative Feature Importance', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        plots['cumulative'] = base64.b64encode(buf.read()).decode('utf-8')
        
        # 4. Comparison Plot (if baseline available)
        if baseline_importance:
            fig, ax = plt.subplots(figsize=(10, max(6, len(indeps) * 0.4)))
            
            # Normalize both for comparison
            perm_dict = {f['feature']: f['importance'] for f in feature_ranking}
            base_dict = {f['feature']: f['importance'] for f in baseline_importance}
            
            all_features = list(perm_dict.keys())
            perm_vals = [perm_dict.get(f, 0) for f in all_features]
            base_vals = [base_dict.get(f, 0) for f in all_features]
            
            # Normalize to 0-1 range for comparison
            perm_max = max(abs(v) for v in perm_vals) if perm_vals else 1
            base_max = max(abs(v) for v in base_vals) if base_vals else 1
            perm_norm = [v / perm_max for v in perm_vals]
            base_norm = [v / base_max for v in base_vals]
            
            y_pos = np.arange(len(all_features))
            width = 0.35
            ax.barh(y_pos - width/2, perm_norm, width, label='Permutation', alpha=0.7, color='#4C72B0')
            ax.barh(y_pos + width/2, base_norm, width, label='Built-in', alpha=0.7, color='#55A868')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(all_features)
            ax.invert_yaxis()
            ax.set_xlabel('Normalized Importance')
            ax.set_title('Permutation vs Built-in Importance', fontweight='bold')
            ax.legend()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            plots['comparison'] = base64.b64encode(buf.read()).decode('utf-8')
        
        # 5. Feature Correlation Heatmap
        if len(indeps) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df_clean[indeps].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, ax=ax, square=True, linewidths=0.5)
            ax.set_title('Feature Correlation Matrix', fontweight='bold')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            plots['correlation'] = base64.b64encode(buf.read()).decode('utf-8')
        
        return _to_native({
            'task_type': task,
            'model_type': req.model_type,
            'n_observations': n_observations,
            'n_train': n_train,
            'n_test': n_test,
            'n_features': len(indeps),
            'n_repeats': req.n_repeats,
            'feature_names': indeps,
            'dependent_var': dep,
            'classes': classes,
            'model_performance': {
                'train_score': train_score,
                'test_score': test_score,
                'metric': metric
            },
            'permutation_importance': permutation_importance_list,
            'baseline_importance': baseline_importance,
            'feature_ranking': feature_ranking,
            'insights': insights,
            'recommendations': recommendations,
            'plots': plots
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
