"""
Aha-Moment Discovery FastAPI Endpoint
Identify critical user actions that drive retention and loyalty
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
from io import BytesIO
import base64
import warnings

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class AhaMomentRequest(BaseModel):
    """Request model for Aha-Moment Discovery"""
    data: List[Dict[str, Any]]
    customer_id_col: str
    feature_cols: List[str]  # Binary feature usage columns (0/1)
    is_retained_col: str      # Target: 1 = retained/loyal, 0 = churned


class KeyInsight(BaseModel):
    """Key insight"""
    title: str
    description: str
    status: str


class AhaMomentResponse(BaseModel):
    """Response model for Aha-Moment Discovery"""
    success: bool
    results: Dict[str, Any]
    visualizations: Dict[str, Optional[str]]
    key_insights: List[KeyInsight]
    summary: Dict[str, Any]


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@router.post("/aha-moment")
async def discover_aha_moment(request: AhaMomentRequest):
    """
    Aha-Moment Discovery
    
    Identify critical user actions that convert new users to loyal customers
    using Random Forest feature importance analysis
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 20:
            raise HTTPException(400, "Insufficient data (need at least 20 customers)")
        
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.customer_id_col not in df.columns:
            raise HTTPException(400, f"Customer ID column '{request.customer_id_col}' not found")
        if request.is_retained_col not in df.columns:
            raise HTTPException(400, f"Retention column '{request.is_retained_col}' not found")
        
        missing_features = [col for col in request.feature_cols if col not in df.columns]
        if missing_features:
            raise HTTPException(400, f"Feature columns not found: {missing_features}")
        
        # Prepare data
        X = df[request.feature_cols].copy()
        y = df[request.is_retained_col].copy()
        
        # Ensure binary
        for col in request.feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
            X[col] = X[col].apply(lambda x: 1 if x > 0 else 0)
        
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        y = y.apply(lambda x: 1 if x > 0 else 0)
        
        if X.isnull().any().any():
            raise HTTPException(400, "Feature columns contain invalid data")
        
        # Calculate basic metrics
        total_customers = len(df)
        retained_customers = int(y.sum())
        churned_customers = total_customers - retained_customers
        retention_rate = (retained_customers / total_customers) * 100
        
        # Feature usage rates
        feature_usage = {}
        for feature in request.feature_cols:
            usage_count = int(X[feature].sum())
            usage_rate = (usage_count / total_customers) * 100
            
            # Retention rate for users who used this feature
            used_feature = df[X[feature] == 1]
            if len(used_feature) > 0:
                retention_with_feature = used_feature[request.is_retained_col].mean() * 100
            else:
                retention_with_feature = 0
            
            # Retention rate for users who didn't use this feature
            no_feature = df[X[feature] == 0]
            if len(no_feature) > 0:
                retention_without_feature = no_feature[request.is_retained_col].mean() * 100
            else:
                retention_without_feature = 0
            
            # Calculate lift
            if retention_without_feature > 0:
                lift = ((retention_with_feature - retention_without_feature) / retention_without_feature) * 100
            else:
                lift = 0 if retention_with_feature == 0 else 100
            
            feature_usage[feature] = {
                'usage_count': usage_count,
                'usage_rate': float(usage_rate),
                'retention_with': float(retention_with_feature),
                'retention_without': float(retention_without_feature),
                'lift': float(lift)
            }
        
        # Random Forest Feature Importance
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        
        # Feature importance
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': request.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Model performance
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        # Identify Aha-Moment (top feature)
        top_feature = feature_importance.iloc[0]['feature']
        top_importance = feature_importance.iloc[0]['importance']
        
        # Aha-Moment metrics
        aha_usage_count = feature_usage[top_feature]['usage_count']
        aha_retention_with = feature_usage[top_feature]['retention_with']
        aha_retention_without = feature_usage[top_feature]['retention_without']
        aha_lift = feature_usage[top_feature]['lift']
        
        # Determine impact level
        if aha_lift >= 50:
            impact_level = 'Critical'
        elif aha_lift >= 25:
            impact_level = 'High'
        elif aha_lift >= 10:
            impact_level = 'Moderate'
        else:
            impact_level = 'Low'
        
        # Multi-feature analysis (combinations)
        combination_analysis = []
        if len(request.feature_cols) >= 2:
            top_3_features = feature_importance.head(3)['feature'].tolist()
            
            for i, feat1 in enumerate(top_3_features):
                for feat2 in top_3_features[i+1:]:
                    combo_mask = (X[feat1] == 1) & (X[feat2] == 1)
                    combo_users = df[combo_mask]
                    
                    if len(combo_users) > 0:
                        combo_retention = combo_users[request.is_retained_col].mean() * 100
                        combo_count = len(combo_users)
                        
                        combination_analysis.append({
                            'features': f"{feat1} + {feat2}",
                            'user_count': int(combo_count),
                            'retention_rate': float(combo_retention),
                            'lift_vs_baseline': float(combo_retention - retention_rate)
                        })
        
        combination_analysis = sorted(combination_analysis, key=lambda x: x['retention_rate'], reverse=True)
        
        metrics = {
            'total_customers': total_customers,
            'retained_customers': retained_customers,
            'churned_customers': churned_customers,
            'retention_rate': float(retention_rate),
            'aha_moment_feature': top_feature,
            'aha_moment_importance': float(top_importance),
            'aha_moment_lift': float(aha_lift),
            'impact_level': impact_level,
            'model_accuracy': float(test_score),
            'model_auc': float(auc)
        }
        
        model_performance = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test))
        }
        
        # Customer analysis
        customer_analysis = []
        for idx, row in df.iterrows():
            features_used = [feat for feat in request.feature_cols if row[feat] > 0]
            feature_count = len(features_used)
            is_retained = int(row[request.is_retained_col])
            used_aha_moment = 1 if row[top_feature] > 0 else 0
            
            customer_analysis.append({
                request.customer_id_col: row[request.customer_id_col],
                'features_used': features_used,
                'feature_count': feature_count,
                'used_aha_moment': used_aha_moment,
                'is_retained': is_retained
            })
        
        # Visualizations
        visualizations = {}
        
        # 1. Feature Importance (Top 10)
        top_n = min(10, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2C3E50' if i == 0 else '#5A6C7D' for i in range(len(top_features))]
        ax.barh(top_features['feature'], top_features['importance'], color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title('Feature Importance for Retention', fontsize=13, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (feat, imp) in enumerate(zip(top_features['feature'], top_features['importance'])):
            ax.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        visualizations['feature_importance'] = fig_to_base64(fig)
        
        # 2. Retention Lift by Feature
        fig, ax = plt.subplots(figsize=(12, 6))
        
        lift_data = pd.DataFrame([
            {'feature': feat, 'lift': stats['lift']}
            for feat, stats in feature_usage.items()
        ]).sort_values('lift', ascending=False).head(top_n)
        
        colors_lift = ['#2C3E50' if lift >= 50 else '#5A6C7D' if lift >= 25 else '#8A9CAD' 
                       for lift in lift_data['lift']]
        
        ax.barh(lift_data['feature'], lift_data['lift'], color=colors_lift, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Retention Lift (%)', fontsize=11, fontweight='bold')
        ax.set_title('Retention Lift by Feature Usage', fontsize=13, fontweight='bold', pad=15)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (feat, lift) in enumerate(zip(lift_data['feature'], lift_data['lift'])):
            ax.text(lift + 2, i, f'{lift:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        visualizations['retention_lift'] = fig_to_base64(fig)
        
        # 3. Retention Rate Comparison (With vs Without)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_5_features = feature_importance.head(5)['feature'].tolist()
        with_rates = [feature_usage[f]['retention_with'] for f in top_5_features]
        without_rates = [feature_usage[f]['retention_without'] for f in top_5_features]
        
        x = np.arange(len(top_5_features))
        width = 0.35
        
        ax.bar(x - width/2, with_rates, width, label='Used Feature', color='#2C3E50', edgecolor='black', alpha=0.8)
        ax.bar(x + width/2, without_rates, width, label='Did Not Use', color='#8A9CAD', edgecolor='black', alpha=0.8)
        
        ax.set_xlabel('Feature', fontsize=11, fontweight='bold')
        ax.set_ylabel('Retention Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Retention Rate: Feature Usage Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(top_5_features, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        visualizations['retention_comparison'] = fig_to_base64(fig)
        
        # 4. Feature Usage Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        usage_data = pd.DataFrame([
            {'feature': feat, 'usage_rate': stats['usage_rate']}
            for feat, stats in feature_usage.items()
        ]).sort_values('usage_rate', ascending=False).head(top_n)
        
        ax.barh(usage_data['feature'], usage_data['usage_rate'], color='#5A6C7D', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Usage Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Feature Adoption Rate', fontsize=13, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (feat, rate) in enumerate(zip(usage_data['feature'], usage_data['usage_rate'])):
            ax.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        visualizations['usage_distribution'] = fig_to_base64(fig)
        
        # 5. Model Performance Metrics
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Accuracy comparison
        metrics_data = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [train_score, test_score, precision, recall, f1]
        
        axes[0].barh(metrics_data, metrics_values, color='#5A6C7D', edgecolor='black', alpha=0.8)
        axes[0].set_xlabel('Score', fontsize=10, fontweight='bold')
        axes[0].set_title('Model Performance Metrics', fontsize=12, fontweight='bold', pad=15)
        axes[0].set_xlim(0, 1)
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].invert_yaxis()
        
        for i, (metric, value) in enumerate(zip(metrics_data, metrics_values)):
            axes[0].text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Right: Feature count vs Retention
        feature_count_groups = df.groupby(df[request.feature_cols].sum(axis=1))[request.is_retained_col].agg(['mean', 'count'])
        feature_count_groups = feature_count_groups[feature_count_groups['count'] >= 3]  # Min 3 users per group
        
        if len(feature_count_groups) > 0:
            axes[1].plot(feature_count_groups.index, feature_count_groups['mean'] * 100, 
                        marker='o', linewidth=2, markersize=8, color='#2C3E50')
            axes[1].set_xlabel('Number of Features Used', fontsize=10, fontweight='bold')
            axes[1].set_ylabel('Retention Rate (%)', fontsize=10, fontweight='bold')
            axes[1].set_title('Retention vs Feature Adoption', fontsize=12, fontweight='bold', pad=15)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        visualizations['model_performance'] = fig_to_base64(fig)
        
        # Key Insights
        insights = []
        
        insights.append({
            'title': f'Aha-Moment: {top_feature}',
            'description': f"Users who engage with '{top_feature}' show {aha_lift:.1f}% higher retention ({aha_retention_with:.1f}% vs {aha_retention_without:.1f}%). This is the critical action for converting new users to loyal customers.",
            'status': 'positive' if aha_lift >= 25 else 'neutral'
        })
        
        insights.append({
            'title': f'Impact Level: {impact_level}',
            'description': f"{'Critical impact' if impact_level == 'Critical' else 'Significant impact' if impact_level == 'High' else 'Moderate impact' if impact_level == 'Moderate' else 'Low impact'} with {aha_lift:.1f}% retention lift. {aha_usage_count} customers ({(aha_usage_count/total_customers)*100:.1f}%) have adopted this feature.",
            'status': 'positive' if impact_level in ['Critical', 'High'] else 'neutral'
        })
        
        if len(combination_analysis) > 0:
            best_combo = combination_analysis[0]
            insights.append({
                'title': f"Best Combination: {best_combo['features']}",
                'description': f"{best_combo['user_count']} users adopted both features, achieving {best_combo['retention_rate']:.1f}% retention (+{best_combo['lift_vs_baseline']:.1f}pp vs baseline).",
                'status': 'positive'
            })
        
        if model_performance['auc'] >= 0.75:
            insights.append({
                'title': 'Strong Predictive Model',
                'description': f"Model AUC of {model_performance['auc']:.3f} indicates reliable feature importance rankings. Test accuracy: {model_performance['test_accuracy']:.1%}.",
                'status': 'positive'
            })
        elif model_performance['auc'] >= 0.6:
            insights.append({
                'title': 'Moderate Model Performance',
                'description': f"Model AUC of {model_performance['auc']:.3f} provides reasonable guidance. Consider collecting more data or additional features for improved accuracy.",
                'status': 'neutral'
            })
        
        summary = {
            'analysis_type': 'Aha-Moment Discovery',
            'total_customers': total_customers,
            'retention_rate': float(retention_rate),
            'aha_moment_feature': top_feature,
            'aha_moment_lift': float(aha_lift)
        }
        
        return AhaMomentResponse(
            success=True,
            results={
                'metrics': metrics,
                'model_performance': model_performance,
                'feature_importance': feature_importance.to_dict('records'),
                'feature_usage': feature_usage,
                'combination_analysis': combination_analysis,
                'customer_analysis': customer_analysis
            },
            visualizations=visualizations,
            key_insights=insights,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
