"""
Churn Prediction FastAPI Endpoint
Predict customer churn risk using machine learning
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
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


class ChurnRequest(BaseModel):
    """Request model for Churn Prediction"""
    data: List[Dict[str, Any]]
    customer_id_col: str
    last_activity_col: str
    reference_date: Optional[str] = None  # YYYY-MM-DD, defaults to today


class KeyInsight(BaseModel):
    """Key insight"""
    title: str
    description: str
    status: str


class ChurnResponse(BaseModel):
    """Response model for Churn Prediction"""
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


def calculate_engagement_features(df, last_activity_col, reference_date):
    """Calculate engagement features from last activity date"""
    # Parse dates
    df['last_activity_date'] = pd.to_datetime(df[last_activity_col], errors='coerce')
    
    # Calculate days since last activity
    df['days_since_activity'] = (reference_date - df['last_activity_date']).dt.days
    
    # Create engagement bins
    df['engagement_level'] = pd.cut(
        df['days_since_activity'],
        bins=[-1, 7, 30, 90, 180, float('inf')],
        labels=['Very Active', 'Active', 'Moderate', 'Low', 'Inactive']
    )
    
    # Binary churn indicator (inactive for >90 days)
    df['is_churned'] = (df['days_since_activity'] > 90).astype(int)
    
    return df


def assign_risk_tier(probability):
    """Assign risk tier based on churn probability"""
    if probability >= 0.75:
        return 'Critical'
    elif probability >= 0.50:
        return 'High'
    elif probability >= 0.25:
        return 'Medium'
    else:
        return 'Low'


@router.post("/churn-prediction")
async def predict_churn(request: ChurnRequest):
    """
    Churn Prediction Analysis
    
    Predict customer churn risk using engagement patterns
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 20:
            raise HTTPException(400, "Insufficient data (need at least 20 customers)")
        
        df = pd.DataFrame(request.data)
        
        # Set reference date
        if request.reference_date:
            reference_date = pd.to_datetime(request.reference_date)
        else:
            reference_date = pd.Timestamp.now()
        
        # Calculate features
        df = calculate_engagement_features(df, request.last_activity_col, reference_date)
        
        # Remove invalid records
        df = df.dropna(subset=['days_since_activity'])
        
        if len(df) < 20:
            raise HTTPException(400, "Insufficient valid data after processing")
        
        # Check if we have variation in churn status
        if df['is_churned'].nunique() < 2:
            raise HTTPException(400, "All customers have same churn status - need variation")
        
        # Prepare features for modeling
        X = df[['days_since_activity']].values
        y = df['is_churned'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Model metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Score all customers
        churn_probabilities = model.predict_proba(X)[:, 1]
        df['churn_probability'] = churn_probabilities
        df['churn_probability_pct'] = churn_probabilities * 100
        df['risk_tier'] = df['churn_probability'].apply(assign_risk_tier)
        
        # Sort by probability
        df_sorted = df.sort_values('churn_probability', ascending=False)
        
        # Risk tier summary
        risk_summary = df.groupby('risk_tier').agg({
            request.customer_id_col: 'count',
            'churn_probability': 'mean',
            'days_since_activity': 'mean'
        }).reset_index()
        risk_summary.columns = ['risk_tier', 'customer_count', 'avg_probability', 'avg_days_inactive']
        risk_summary['avg_probability_pct'] = risk_summary['avg_probability'] * 100
        
        # Ensure proper order
        tier_order = ['Critical', 'High', 'Medium', 'Low']
        risk_summary['risk_tier'] = pd.Categorical(risk_summary['risk_tier'], categories=tier_order, ordered=True)
        risk_summary = risk_summary.sort_values('risk_tier')
        
        # Engagement level summary
        engagement_summary = df.groupby('engagement_level').agg({
            request.customer_id_col: 'count',
            'churn_probability': 'mean',
            'is_churned': 'mean'
        }).reset_index()
        engagement_summary.columns = ['engagement_level', 'customer_count', 'avg_churn_prob', 'actual_churn_rate']
        engagement_summary['avg_churn_prob_pct'] = engagement_summary['avg_churn_prob'] * 100
        engagement_summary['actual_churn_rate_pct'] = engagement_summary['actual_churn_rate'] * 100
        
        # Overall metrics
        total_customers = len(df)
        critical_risk = len(df[df['risk_tier'] == 'Critical'])
        high_risk = len(df[df['risk_tier'] == 'High'])
        at_risk_total = critical_risk + high_risk
        avg_churn_prob = df['churn_probability'].mean() * 100
        avg_days_inactive = df['days_since_activity'].mean()
        
        # Current churned count
        currently_churned = df['is_churned'].sum()
        churn_rate = (currently_churned / total_customers) * 100
        
        metrics = {
            'total_customers': int(total_customers),
            'critical_risk_count': int(critical_risk),
            'high_risk_count': int(high_risk),
            'at_risk_total': int(at_risk_total),
            'avg_churn_probability': float(avg_churn_prob),
            'avg_days_inactive': float(avg_days_inactive),
            'current_churn_count': int(currently_churned),
            'current_churn_rate': float(churn_rate),
            'model_auc': float(auc_score)
        }
        
        # Visualizations
        visualizations = {}
        
        # 1. Risk Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.pie(risk_summary['customer_count'], labels=risk_summary['risk_tier'],
               autopct='%1.1f%%', startangle=90,
               colors=['#E74C3C', '#F39C12', '#F1C40F', '#95A5A6'])
        ax1.set_title('Customer Distribution by Risk Tier', fontsize=12, fontweight='bold')
        
        bars = ax2.barh(risk_summary['risk_tier'], risk_summary['customer_count'],
                       color='#2C3E50', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
        ax2.set_title('Customer Count by Risk Tier', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(risk_summary['customer_count']):
            ax2.text(v, i, f' {int(v)}', va='center', fontweight='bold')
        plt.tight_layout()
        visualizations['risk_distribution'] = fig_to_base64(fig)
        
        # 2. Churn Probability Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(df['churn_probability'], bins=30, color='#2C3E50', edgecolor='black', alpha=0.7)
        ax.axvline(df['churn_probability'].mean(), color='#3498DB', linestyle='--',
                  linewidth=2, label=f'Mean: {df["churn_probability"].mean():.2%}')
        ax.axvline(0.5, color='#E74C3C', linestyle='--',
                  linewidth=2, label='High Risk Threshold (50%)')
        ax.set_xlabel('Churn Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
        ax.set_title('Churn Probability Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        visualizations['probability_distribution'] = fig_to_base64(fig)
        
        # 3. Days Inactive vs Churn Probability
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(df['days_since_activity'], df['churn_probability'],
                           alpha=0.5, c=df['churn_probability'], cmap='YlOrRd',
                           edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Days Since Last Activity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Churn Probability', fontsize=11, fontweight='bold')
        ax.set_title('Engagement vs Churn Risk', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Churn Probability', ax=ax)
        plt.tight_layout()
        visualizations['engagement_vs_churn'] = fig_to_base64(fig)
        
        # 4. ROC Curve
        fig, ax = plt.subplots(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax.plot(fpr, tpr, linewidth=3, color='#3498DB',
               label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title('ROC Curve - Model Performance', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        visualizations['roc_curve'] = fig_to_base64(fig)
        
        # 5. Risk Tier Metrics
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].barh(risk_summary['risk_tier'], risk_summary['avg_probability_pct'],
                    color='#3498DB', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Avg Churn Probability (%)', fontsize=10, fontweight='bold')
        axes[0].set_title('Average Churn Probability by Risk Tier', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(risk_summary['avg_probability_pct']):
            axes[0].text(v, i, f' {v:.1f}%', va='center', fontweight='bold')
        
        axes[1].barh(risk_summary['risk_tier'], risk_summary['avg_days_inactive'],
                    color='#95A5A6', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Avg Days Inactive', fontsize=10, fontweight='bold')
        axes[1].set_title('Average Inactivity by Risk Tier', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(risk_summary['avg_days_inactive']):
            axes[1].text(v, i, f' {v:.0f}', va='center', fontweight='bold')
        plt.tight_layout()
        visualizations['tier_metrics'] = fig_to_base64(fig)
        
        # Insights
        insights = []
        
        if auc_score >= 0.8:
            insights.append({
                'title': 'Excellent Model Performance',
                'description': f"Model AUC of {auc_score:.1%} indicates strong predictive accuracy. Churn risk scores are highly reliable for prioritization.",
                'status': 'positive'
            })
        elif auc_score >= 0.7:
            insights.append({
                'title': 'Good Model Performance',
                'description': f"Model AUC of {auc_score:.1%} provides solid predictions. Scores effectively distinguish high from low-risk customers.",
                'status': 'positive'
            })
        else:
            insights.append({
                'title': 'Moderate Model Performance',
                'description': f"Model AUC of {auc_score:.1%}. Consider enriching data with additional engagement metrics for better predictions.",
                'status': 'warning'
            })
        
        if at_risk_total > 0:
            at_risk_pct = (at_risk_total / total_customers) * 100
            insights.append({
                'title': f'High-Risk Customers: {at_risk_total}',
                'description': f"{at_risk_pct:.1f}% of customers are at high/critical risk. Launch immediate retention campaigns to prevent churn.",
                'status': 'warning'
            })
        
        if critical_risk > 0:
            critical_pct = (critical_risk / total_customers) * 100
            insights.append({
                'title': f'Critical Risk Segment: {critical_risk} customers',
                'description': f"{critical_pct:.1f}% of customers are in critical risk tier (>75% churn probability). Urgent intervention required.",
                'status': 'warning'
            })
        
        return ChurnResponse(
            success=True,
            results={
                'metrics': metrics,
                'customer_predictions': df_sorted[[
                    request.customer_id_col, request.last_activity_col,
                    'days_since_activity', 'churn_probability', 'churn_probability_pct',
                    'risk_tier', 'engagement_level', 'is_churned'
                ]].to_dict('records'),
                'risk_summary': risk_summary.to_dict('records'),
                'engagement_summary': engagement_summary.to_dict('records'),
                'model_performance': {
                    'auc_score': float(auc_score),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
            },
            visualizations=visualizations,
            key_insights=insights,
            summary={
                'analysis_type': 'churn_prediction',
                'total_customers': metrics['total_customers'],
                'at_risk_count': metrics['at_risk_total'],
                'model_auc': metrics['model_auc']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
