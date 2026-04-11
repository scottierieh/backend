"""
Employee Attrition Prediction Router for FastAPI
Turnover Risk Analysis, Attrition Drivers, Retention Recommendations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class AttritionRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_col: str  # Attrition column (0/1 or Yes/No)
    feature_cols: List[str]  # Features for prediction
    employee_id_col: Optional[str] = None
    model_type: str = "random_forest"  # random_forest or gradient_boosting
    test_size: float = 0.2
    n_estimators: int = 100


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


def prepare_features(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> tuple:
    """Prepare features for modeling"""
    df_model = df.copy()
    
    # Encode target if needed
    if df_model[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df_model[target_col] = le_target.fit_transform(df_model[target_col])
    
    # Encode categorical features
    label_encoders = {}
    for col in feature_cols:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
    
    X = df_model[feature_cols].values
    y = df_model[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, label_encoders, scaler


def train_model(X: np.ndarray, y: np.ndarray, model_type: str, 
                n_estimators: int, test_size: float) -> Dict[str, Any]:
    """Train attrition prediction model"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Select model
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    
    # Metrics
    metrics = {
        'accuracy': _to_native_type(accuracy_score(y_test, y_pred)),
        'precision': _to_native_type(precision_score(y_test, y_pred)),
        'recall': _to_native_type(recall_score(y_test, y_pred)),
        'f1_score': _to_native_type(f1_score(y_test, y_pred)),
        'roc_auc': _to_native_type(roc_auc_score(y_test, y_prob)),
        'cv_mean': _to_native_type(cv_scores.mean()),
        'cv_std': _to_native_type(cv_scores.std())
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    return {
        'model': model,
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': [_to_native_type(f) for f in fpr],
            'tpr': [_to_native_type(t) for t in tpr]
        },
        'test_size': len(y_test),
        'train_size': len(y_train)
    }


def calculate_feature_importance(model, feature_cols: List[str]) -> List[Dict[str, Any]]:
    """Calculate and rank feature importance"""
    importances = model.feature_importances_
    
    feature_importance = []
    for feat, imp in zip(feature_cols, importances):
        feature_importance.append({
            'feature': feat,
            'importance': _to_native_type(imp),
            'importance_pct': _to_native_type(imp * 100)
        })
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return feature_importance


def calculate_risk_scores(model, X: np.ndarray, df: pd.DataFrame, 
                          employee_id_col: Optional[str]) -> List[Dict[str, Any]]:
    """Calculate attrition risk scores for all employees"""
    probabilities = model.predict_proba(X)[:, 1]
    
    risk_scores = []
    for i, prob in enumerate(probabilities):
        risk_level = 'High' if prob >= 0.7 else ('Medium' if prob >= 0.4 else 'Low')
        
        record = {
            'index': i,
            'risk_score': _to_native_type(prob),
            'risk_pct': _to_native_type(prob * 100),
            'risk_level': risk_level
        }
        
        if employee_id_col and employee_id_col in df.columns:
            record['employee_id'] = str(df.iloc[i][employee_id_col])
        
        risk_scores.append(record)
    
    # Sort by risk score descending
    risk_scores.sort(key=lambda x: x['risk_score'], reverse=True)
    
    return risk_scores


def analyze_attrition_by_segment(df: pd.DataFrame, target_col: str, 
                                  feature_cols: List[str]) -> Dict[str, Any]:
    """Analyze attrition rates by different segments"""
    
    # Convert target to binary if needed
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0})
    
    overall_rate = df[target_col].mean()
    
    segment_analysis = []
    for col in feature_cols:
        if df[col].dtype == 'object' or df[col].nunique() <= 10:
            # Categorical or low cardinality
            segment_rates = df.groupby(col)[target_col].agg(['mean', 'count']).reset_index()
            segment_rates.columns = [col, 'attrition_rate', 'count']
            
            for _, row in segment_rates.iterrows():
                segment_analysis.append({
                    'feature': col,
                    'value': str(row[col]),
                    'attrition_rate': _to_native_type(row['attrition_rate']),
                    'attrition_pct': _to_native_type(row['attrition_rate'] * 100),
                    'count': int(row['count']),
                    'vs_overall': _to_native_type((row['attrition_rate'] - overall_rate) / overall_rate * 100) if overall_rate > 0 else 0
                })
    
    return {
        'overall_rate': _to_native_type(overall_rate),
        'overall_pct': _to_native_type(overall_rate * 100),
        'segments': segment_analysis
    }


def calculate_risk_distribution(risk_scores: List[Dict]) -> Dict[str, Any]:
    """Calculate distribution of risk levels"""
    high = sum(1 for r in risk_scores if r['risk_level'] == 'High')
    medium = sum(1 for r in risk_scores if r['risk_level'] == 'Medium')
    low = sum(1 for r in risk_scores if r['risk_level'] == 'Low')
    total = len(risk_scores)
    
    return {
        'high_risk': {'count': high, 'pct': _to_native_type(high / total * 100)},
        'medium_risk': {'count': medium, 'pct': _to_native_type(medium / total * 100)},
        'low_risk': {'count': low, 'pct': _to_native_type(low / total * 100)},
        'total': total
    }


def create_feature_importance_chart(feature_importance: List[Dict]) -> str:
    """Create feature importance visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Top 10 features
    top_features = feature_importance[:10]
    features = [f['feature'] for f in top_features]
    importances = [f['importance_pct'] for f in top_features]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
    
    bars = ax.barh(features[::-1], importances[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Importance (%)')
    ax.set_title('Top Attrition Drivers', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, imp in zip(bars, importances[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{imp:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_risk_distribution_chart(risk_dist: Dict) -> str:
    """Create risk distribution visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    labels = ['High Risk', 'Medium Risk', 'Low Risk']
    sizes = [risk_dist['high_risk']['count'], risk_dist['medium_risk']['count'], risk_dist['low_risk']['count']]
    colors = ['#ef4444', '#f59e0b', '#22c55e']
    explode = (0.05, 0.02, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title('Risk Level Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart
    categories = ['High\nRisk', 'Medium\nRisk', 'Low\nRisk']
    counts = [risk_dist['high_risk']['count'], risk_dist['medium_risk']['count'], risk_dist['low_risk']['count']]
    
    bars = ax2.bar(categories, counts, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Number of Employees')
    ax2.set_title('Employee Count by Risk Level', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.02,
                f'{count:,}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_model_performance_chart(model_result: Dict) -> str:
    """Create model performance visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr = model_result['roc_curve']['fpr']
    tpr = model_result['roc_curve']['tpr']
    auc = model_result['metrics']['roc_auc']
    
    ax1.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC Curve (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.3, color='blue')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Metrics bar chart
    metrics = model_result['metrics']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['f1_score'], metrics['roc_auc']]
    
    colors = ['#3b82f6' if v >= 0.7 else '#f59e0b' if v >= 0.5 else '#ef4444' for v in metric_values]
    bars = ax2.bar(metric_names, metric_values, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_analysis_chart(segment_data: Dict) -> str:
    """Create segment analysis visualization"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    segments = segment_data['segments']
    overall_rate = segment_data['overall_pct']
    
    # Group by feature, show top segments
    if len(segments) > 15:
        # Sort by deviation from overall and take top/bottom
        segments.sort(key=lambda x: abs(x['vs_overall']), reverse=True)
        segments = segments[:15]
    
    labels = [f"{s['feature']}: {s['value']}" for s in segments]
    rates = [s['attrition_pct'] for s in segments]
    colors = ['#ef4444' if r > overall_rate else '#22c55e' for r in rates]
    
    bars = ax.barh(labels[::-1], rates[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    ax.axvline(x=overall_rate, color='blue', linestyle='--', linewidth=2, label=f'Overall: {overall_rate:.1f}%')
    
    ax.set_xlabel('Attrition Rate (%)')
    ax.set_title('Attrition Rate by Segment', fontsize=14, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, rate in zip(bars, rates[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{rate:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(model_result: Dict, feature_importance: List[Dict],
                      risk_dist: Dict, segment_data: Dict) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    # Model performance
    auc = model_result['metrics']['roc_auc']
    if auc >= 0.8:
        insights.append({
            'title': f'Strong Predictive Model (AUC: {auc:.3f})',
            'description': 'Model has excellent ability to identify at-risk employees.',
            'status': 'positive'
        })
    elif auc >= 0.7:
        insights.append({
            'title': f'Good Predictive Model (AUC: {auc:.3f})',
            'description': 'Model performs well for attrition prediction.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Model Needs Improvement (AUC: {auc:.3f})',
            'description': 'Consider adding more features or data.',
            'status': 'warning'
        })
    
    # Top driver
    top_driver = feature_importance[0]
    insights.append({
        'title': f'Top Attrition Driver: {top_driver["feature"]}',
        'description': f'Accounts for {top_driver["importance_pct"]:.1f}% of prediction importance.',
        'status': 'neutral'
    })
    
    # High risk count
    high_risk_pct = risk_dist['high_risk']['pct']
    if high_risk_pct >= 20:
        insights.append({
            'title': f'High Risk Alert: {high_risk_pct:.1f}% of employees',
            'description': f'{risk_dist["high_risk"]["count"]} employees at high attrition risk.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': f'Manageable Risk: {high_risk_pct:.1f}% high risk',
            'description': f'{risk_dist["high_risk"]["count"]} employees need attention.',
            'status': 'positive'
        })
    
    # Highest risk segment
    if segment_data['segments']:
        highest_segment = max(segment_data['segments'], key=lambda x: x['attrition_pct'])
        insights.append({
            'title': f'Highest Risk Segment: {highest_segment["feature"]} = {highest_segment["value"]}',
            'description': f'{highest_segment["attrition_pct"]:.1f}% attrition rate ({highest_segment["vs_overall"]:+.1f}% vs overall).',
            'status': 'warning' if highest_segment['vs_overall'] > 20 else 'neutral'
        })
    
    return insights


@router.post("/attrition")
async def run_attrition_analysis(request: AttritionRequest) -> Dict[str, Any]:
    """
    Perform Employee Attrition Prediction Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Need at least 50 records for reliable prediction")
        
        if request.target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_col}' not found")
        
        for col in request.feature_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature column '{col}' not found")
        
        # Prepare features
        X, y, label_encoders, scaler = prepare_features(df, request.feature_cols, request.target_col)
        
        # Train model
        model_result = train_model(X, y, request.model_type, request.n_estimators, request.test_size)
        
        # Feature importance
        feature_importance = calculate_feature_importance(model_result['model'], request.feature_cols)
        
        # Risk scores for all employees
        risk_scores = calculate_risk_scores(model_result['model'], X, df, request.employee_id_col)
        
        # Risk distribution
        risk_distribution = calculate_risk_distribution(risk_scores)
        
        # Segment analysis
        segment_analysis = analyze_attrition_by_segment(df, request.target_col, request.feature_cols)
        
        # Visualizations
        visualizations = {
            'feature_importance_chart': create_feature_importance_chart(feature_importance),
            'risk_distribution_chart': create_risk_distribution_chart(risk_distribution),
            'model_performance_chart': create_model_performance_chart(model_result),
            'segment_analysis_chart': create_segment_analysis_chart(segment_analysis)
        }
        
        # Insights
        insights = generate_insights(model_result, feature_importance, risk_distribution, segment_analysis)
        
        # Summary
        summary = {
            'total_employees': len(df),
            'overall_attrition_rate': segment_analysis['overall_pct'],
            'high_risk_count': risk_distribution['high_risk']['count'],
            'high_risk_pct': risk_distribution['high_risk']['pct'],
            'model_auc': model_result['metrics']['roc_auc'],
            'top_driver': feature_importance[0]['feature'],
            'top_driver_importance': feature_importance[0]['importance_pct']
        }
        
        return {
            'success': True,
            'results': {
                'model_metrics': model_result['metrics'],
                'confusion_matrix': model_result['confusion_matrix'],
                'feature_importance': feature_importance,
                'risk_scores': risk_scores[:50],  # Top 50 highest risk
                'risk_distribution': risk_distribution,
                'segment_analysis': segment_analysis
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attrition analysis failed: {str(e)}")
