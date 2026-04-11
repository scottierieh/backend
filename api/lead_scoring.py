"""
Lead Scoring FastAPI Endpoint
Predict conversion probability for leads using machine learning
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


class LeadScoringRequest(BaseModel):
    """Request model for Lead Scoring"""
    data: List[Dict[str, Any]]
    lead_id_col: str
    source_col: str
    converted_col: str


class KeyInsight(BaseModel):
    """Key insight"""
    title: str
    description: str
    status: str


class LeadScoringResponse(BaseModel):
    """Response model for Lead Scoring"""
    success: bool
    results: Dict[str, Any]
    visualizations: Dict[str, Optional[str]]
    key_insights: List[KeyInsight]
    summary: Dict[str, Any]


def normalize_conversion(val):
    """Convert various formats to 0/1"""
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        val_lower = val.lower()
        if val_lower in ['yes', 'true', '1', 'converted']:
            return 1
        return 0
    return 0


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_visualizations(scored_df, tier_summary, source_analysis, y_test, y_pred_proba, converted_col):
    """Generate all visualizations"""
    visualizations = {}
    
    # 1. Score Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scored_df['conversion_probability'], bins=30, color='#2C3E50', edgecolor='black', alpha=0.7)
    ax.axvline(scored_df['conversion_probability'].mean(), color='#3498DB', linestyle='--', 
              linewidth=2, label=f'Mean: {scored_df["conversion_probability"].mean():.2%}')
    ax.set_xlabel('Conversion Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Leads', fontsize=11, fontweight='bold')
    ax.set_title('Lead Score Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['score_distribution'] = fig_to_base64(fig)
    
    # 2. Tier Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    tier_order = ['Low', 'Medium', 'High', 'Very High']
    tier_summary_sorted = tier_summary.set_index('tier').reindex(tier_order).reset_index()
    
    bars1 = ax1.bar(tier_summary_sorted['tier'], tier_summary_sorted['lead_count'],
                   color='#2C3E50', edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Number of Leads', fontsize=11, fontweight='bold')
    ax1.set_title('Leads by Score Tier', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    bars2 = ax2.bar(tier_summary_sorted['tier'], tier_summary_sorted['conversion_rate'],
                   color='#3498DB', edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Conversion Rate by Tier', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    visualizations['tier_performance'] = fig_to_base64(fig)
    
    # 3. Source Performance
    fig, ax = plt.subplots(figsize=(12, 8))
    top_10 = source_analysis.head(10)
    y_pos = np.arange(len(top_10))
    ax.barh(y_pos, top_10['conversion_rate_pct'], color='#2C3E50', edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10['source'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Traffic Sources by Conversion Rate', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(top_10['conversion_rate_pct']):
        ax.text(v, i, f' {v:.1f}%', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    visualizations['source_performance'] = fig_to_base64(fig)
    
    # 4. ROC Curve
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, linewidth=3, color='#3498DB', label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('ROC Curve - Model Performance', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['roc_curve'] = fig_to_base64(fig)
    
    # 5. Calibration
    fig, ax = plt.subplots(figsize=(10, 6))
    scored_df['prob_bin'] = pd.cut(scored_df['conversion_probability'], bins=10, labels=False)
    calibration = scored_df.groupby('prob_bin').agg({
        'conversion_probability': 'mean',
        converted_col: 'mean'
    }).reset_index()
    ax.scatter(calibration['conversion_probability'], calibration[converted_col],
              s=100, color='#2C3E50', alpha=0.7, edgecolors='black', linewidths=2)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Conversion Rate', fontsize=11, fontweight='bold')
    ax.set_title('Probability Calibration', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['probability_calibration'] = fig_to_base64(fig)
    
    return visualizations


@router.post("/lead-scoring")
async def score_leads(request: LeadScoringRequest):
    """
    Lead Scoring Analysis
    
    Predict conversion probability for leads using machine learning
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 20:
            raise HTTPException(400, "Insufficient data (need at least 20 leads)")
        
        df = pd.DataFrame(request.data)
        df[request.converted_col] = df[request.converted_col].apply(normalize_conversion)
        
        # Check conversion balance
        conversion_rate = df[request.converted_col].mean()
        if conversion_rate == 0 or conversion_rate == 1:
            raise HTTPException(400, "All leads have same conversion status - need variation")
        
        # Encode features
        label_encoder = LabelEncoder()
        df['source_encoded'] = label_encoder.fit_transform(df[request.source_col])
        
        X = df['source_encoded'].values.reshape(-1, 1)
        y = df[request.converted_col].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                      random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Score all leads
        probabilities = model.predict_proba(X)[:, 1]
        df['conversion_probability'] = probabilities
        df['probability_pct'] = probabilities * 100
        df['score_tier'] = pd.cut(probabilities, bins=[0, 0.25, 0.50, 0.75, 1.0],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
        
        scored_df = df.sort_values('conversion_probability', ascending=False)
        
        # Tier summary
        tier_summary = scored_df.groupby('score_tier').agg({
            request.lead_id_col: 'count',
            'conversion_probability': 'mean',
            request.converted_col: 'sum'
        }).reset_index()
        tier_summary.columns = ['tier', 'lead_count', 'avg_probability', 'actual_conversions']
        tier_summary['avg_probability_pct'] = tier_summary['avg_probability'] * 100
        tier_summary['conversion_rate'] = (tier_summary['actual_conversions'] / tier_summary['lead_count']) * 100
        
        # Source analysis
        source_analysis = df.groupby(request.source_col).agg({
            request.lead_id_col: 'count',
            request.converted_col: ['sum', 'mean']
        }).reset_index()
        source_analysis.columns = ['source', 'total_leads', 'conversions', 'conversion_rate']
        source_analysis['conversion_rate_pct'] = source_analysis['conversion_rate'] * 100
        source_analysis = source_analysis.sort_values('conversion_rate', ascending=False)
        
        # Metrics
        total_leads = len(df)
        total_conversions = df[request.converted_col].sum()
        overall_conversion_rate = (total_conversions / total_leads) * 100
        avg_score = scored_df['conversion_probability'].mean() * 100
        top_source = source_analysis.iloc[0]['source']
        top_source_rate = source_analysis.iloc[0]['conversion_rate_pct']
        
        metrics = {
            'total_leads': int(total_leads),
            'total_conversions': int(total_conversions),
            'overall_conversion_rate': float(overall_conversion_rate),
            'unique_sources': int(df[request.source_col].nunique()),
            'avg_lead_score': float(avg_score),
            'top_source': str(top_source),
            'top_source_conversion_rate': float(top_source_rate),
            'model_auc': float(auc_score)
        }
        
        # Visualizations
        visualizations = generate_visualizations(
            scored_df, tier_summary, source_analysis, y_test, y_pred_proba, request.converted_col
        )
        
        # Insights
        insights = []
        if auc_score >= 0.8:
            insights.append({
                'title': 'Excellent Model Performance',
                'description': f"Model AUC of {auc_score:.1%} indicates strong predictive accuracy. Lead scores are highly reliable for prioritization.",
                'status': 'positive'
            })
        elif auc_score >= 0.7:
            insights.append({
                'title': 'Good Model Performance',
                'description': f"Model AUC of {auc_score:.1%} provides solid predictions. Scores effectively distinguish high from low-potential leads.",
                'status': 'positive'
            })
        else:
            insights.append({
                'title': 'Moderate Model Performance',
                'description': f"Model AUC of {auc_score:.1%}. Consider enriching data with additional lead attributes for better predictions.",
                'status': 'warning'
            })
        
        insights.append({
            'title': f'Best Performing Source: {top_source}',
            'description': f"{top_source} delivers {top_source_rate:.1f}% conversion rate. Allocate more resources to this channel.",
            'status': 'positive'
        })
        
        very_high_tier = tier_summary[tier_summary['tier'] == 'Very High']
        if len(very_high_tier) > 0:
            vh_count = very_high_tier['lead_count'].iloc[0]
            vh_conv_rate = very_high_tier['conversion_rate'].iloc[0]
            insights.append({
                'title': 'High-Value Lead Segment Identified',
                'description': f"{vh_count} leads in 'Very High' tier with {vh_conv_rate:.1f}% conversion rate. Prioritize immediate outreach to these leads.",
                'status': 'positive'
            })
        
        return LeadScoringResponse(
            success=True,
            results={
                'metrics': metrics,
                'scored_leads': scored_df[[
                    request.lead_id_col, request.source_col, request.converted_col,
                    'conversion_probability', 'probability_pct', 'score_tier'
                ]].to_dict('records'),
                'tier_summary': tier_summary.to_dict('records'),
                'source_performance': source_analysis.to_dict('records'),
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
                'analysis_type': 'lead_scoring',
                'total_leads': metrics['total_leads'],
                'model_auc': metrics['model_auc'],
                'avg_lead_score': metrics['avg_lead_score']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")
