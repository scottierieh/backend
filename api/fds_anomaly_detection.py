"""
FDS Anomaly Detection Router for FastAPI (using PyOD)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import base64
import time
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class FDSRequest(BaseModel):
    data: List[Dict[str, Any]]
    id_col: str
    feature_cols: List[str]
    method: Literal["iforest", "lof", "knn", "ocsvm", "ecod", "copod"] = "iforest"
    contamination: float = 0.05
    threshold: Optional[float] = None


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


RISK_COLORS = {
    'critical': '#dc2626',
    'high': '#f97316',
    'medium': '#f59e0b',
    'low': '#22c55e',
    'normal': '#3b82f6',
}


def get_detector(method: str, contamination: float):
    """Get PyOD detector based on method"""
    try:
        if method == "iforest":
            from pyod.models.iforest import IForest
            return IForest(contamination=contamination, random_state=42, n_estimators=100)
        elif method == "lof":
            from pyod.models.lof import LOF
            return LOF(contamination=contamination, n_neighbors=20)
        elif method == "knn":
            from pyod.models.knn import KNN
            return KNN(contamination=contamination, n_neighbors=10)
        elif method == "ocsvm":
            from pyod.models.ocsvm import OCSVM
            return OCSVM(contamination=contamination)
        elif method == "ecod":
            from pyod.models.ecod import ECOD
            return ECOD(contamination=contamination)
        elif method == "copod":
            from pyod.models.copod import COPOD
            return COPOD(contamination=contamination)
        else:
            from pyod.models.iforest import IForest
            return IForest(contamination=contamination, random_state=42)
    except ImportError:
        # Fallback to simple statistical method if PyOD not available
        return None


def statistical_anomaly_detection(X: np.ndarray, contamination: float) -> tuple:
    """Fallback statistical anomaly detection using Z-scores"""
    # Calculate Z-scores for each feature
    z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10))
    
    # Average Z-score across features
    avg_z = z_scores.mean(axis=1)
    
    # Normalize to 0-1 range
    scores = (avg_z - avg_z.min()) / (avg_z.max() - avg_z.min() + 1e-10)
    
    # Determine threshold based on contamination
    threshold = np.percentile(scores, (1 - contamination) * 100)
    labels = (scores >= threshold).astype(int)
    
    return scores, labels, threshold


def calculate_feature_importance(X: np.ndarray, scores: np.ndarray, feature_names: List[str]) -> List[Dict]:
    """Calculate feature importance based on correlation with anomaly scores"""
    importance = []
    
    for i, name in enumerate(feature_names):
        # Correlation between feature values and anomaly scores
        corr = np.corrcoef(X[:, i], scores)[0, 1]
        corr = 0 if np.isnan(corr) else abs(corr)
        
        # Mean values for normal vs anomaly
        threshold = np.percentile(scores, 95)
        normal_mask = scores < threshold
        anomaly_mask = scores >= threshold
        
        mean_normal = X[normal_mask, i].mean() if normal_mask.sum() > 0 else 0
        mean_anomaly = X[anomaly_mask, i].mean() if anomaly_mask.sum() > 0 else 0
        
        importance.append({
            'feature': name,
            'importance': corr,
            'mean_normal': mean_normal,
            'mean_anomaly': mean_anomaly,
        })
    
    # Normalize importance
    total = sum(f['importance'] for f in importance) or 1
    for f in importance:
        f['importance'] = f['importance'] / total
    
    # Sort by importance
    importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return importance


def get_top_contributing_features(x: np.ndarray, mean_normal: np.ndarray, std_normal: np.ndarray, 
                                   feature_names: List[str], top_n: int = 3) -> List[str]:
    """Get top features contributing to anomaly for a single record"""
    # Z-scores for this record
    z_scores = np.abs((x - mean_normal) / (std_normal + 1e-10))
    
    # Get top features
    top_indices = z_scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]


def classify_risk_level(score: float, threshold: float) -> str:
    """Classify anomaly into risk levels"""
    if score >= 0.9:
        return 'critical'
    elif score >= 0.7:
        return 'high'
    elif score >= 0.5:
        return 'medium'
    elif score >= threshold:
        return 'low'
    return 'normal'


def create_score_distribution_chart(scores: np.ndarray, threshold: float, labels: np.ndarray) -> str:
    """Create anomaly score distribution chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    bins = np.linspace(0, 1, 50)
    ax.hist(normal_scores, bins=bins, alpha=0.7, color=RISK_COLORS['normal'], label=f'Normal ({len(normal_scores)})', edgecolor='white')
    ax.hist(anomaly_scores, bins=bins, alpha=0.7, color=RISK_COLORS['critical'], label=f'Anomaly ({len(anomaly_scores)})', edgecolor='white')
    
    # Threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    ax.set_xlabel('Anomaly Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_feature_importance_chart(feature_importance: List[Dict]) -> str:
    """Create feature importance chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = [f['feature'] for f in feature_importance[:10]]
    importances = [f['importance'] * 100 for f in feature_importance[:10]]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]
    
    bars = ax.barh(features[::-1], importances[::-1], color=colors, edgecolor='white')
    
    for bar, imp in zip(bars, importances[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{imp:.1f}%', va='center', fontsize=9)
    
    ax.set_xlabel('Importance (%)', fontsize=11)
    ax.set_title('Feature Importance for Anomaly Detection', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_scatter_plot(X: np.ndarray, scores: np.ndarray, labels: np.ndarray, 
                        feature_names: List[str]) -> str:
    """Create scatter plot of top 2 features"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use first 2 features or PCA if more
    if X.shape[1] >= 2:
        x_plot = X[:, 0]
        y_plot = X[:, 1]
        xlabel = feature_names[0]
        ylabel = feature_names[1]
    else:
        x_plot = X[:, 0]
        y_plot = scores
        xlabel = feature_names[0]
        ylabel = 'Anomaly Score'
    
    # Plot normal points
    normal_mask = labels == 0
    ax.scatter(x_plot[normal_mask], y_plot[normal_mask], 
               c=RISK_COLORS['normal'], alpha=0.5, s=30, label='Normal')
    
    # Plot anomalies with color by score
    anomaly_mask = labels == 1
    scatter = ax.scatter(x_plot[anomaly_mask], y_plot[anomaly_mask],
                         c=scores[anomaly_mask], cmap='Reds', alpha=0.8, s=60,
                         edgecolors='black', linewidths=0.5, label='Anomaly')
    
    plt.colorbar(scatter, ax=ax, label='Anomaly Score')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('Anomaly Detection Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(summary: Dict, feature_importance: List[Dict]) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Anomaly rate assessment
    if summary['anomaly_rate'] <= 0.02:
        insights.append({
            'title': f'Low Anomaly Rate: {summary["anomaly_rate"]*100:.2f}%',
            'description': 'Very few anomalies detected. Data appears mostly normal.',
            'status': 'positive'
        })
    elif summary['anomaly_rate'] <= 0.10:
        insights.append({
            'title': f'Moderate Anomaly Rate: {summary["anomaly_rate"]*100:.2f}%',
            'description': 'Anomaly rate within typical range. Review flagged records.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'High Anomaly Rate: {summary["anomaly_rate"]*100:.2f}%',
            'description': 'Higher than expected anomaly rate. May need model tuning or data review.',
            'status': 'warning'
        })
    
    # Critical alerts
    if summary['critical_count'] > 0:
        insights.append({
            'title': f'{summary["critical_count"]} Critical Risk Anomalies',
            'description': 'Immediate review recommended for critical risk records.',
            'status': 'warning'
        })
    
    # Top feature
    if feature_importance:
        top_feat = feature_importance[0]
        insights.append({
            'title': f'Top Indicator: {top_feat["feature"]}',
            'description': f'This feature contributes {top_feat["importance"]*100:.1f}% to anomaly detection.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/fds")
async def run_fds_detection(request: FDSRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate
        if request.id_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"ID column '{request.id_col}' not found")
        
        for col in request.feature_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature column '{col}' not found")
        
        # Prepare features
        X = df[request.feature_cols].copy()
        
        # Handle non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        # Fill missing values
        X = X.fillna(X.median())
        X = X.values.astype(float)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get detector
        detector = get_detector(request.method, request.contamination)
        
        if detector is not None:
            # Fit and predict
            detector.fit(X_scaled)
            scores = detector.decision_scores_
            
            # Normalize scores to 0-1
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            # Get labels
            labels = detector.labels_
            
            # Threshold
            threshold = request.threshold if request.threshold else np.percentile(scores, (1 - request.contamination) * 100)
        else:
            # Fallback to statistical method
            scores, labels, threshold = statistical_anomaly_detection(X_scaled, request.contamination)
        
        # Override with custom threshold if provided
        if request.threshold:
            threshold = request.threshold
            labels = (scores >= threshold).astype(int)
        
        # Calculate statistics for feature contribution
        mean_normal = X_scaled[labels == 0].mean(axis=0) if (labels == 0).sum() > 0 else X_scaled.mean(axis=0)
        std_normal = X_scaled[labels == 0].std(axis=0) if (labels == 0).sum() > 0 else X_scaled.std(axis=0)
        
        # Calculate feature importance
        feature_importance = calculate_feature_importance(X_scaled, scores, request.feature_cols)
        
        # Build anomaly records
        ids = df[request.id_col].values
        anomalies = []
        all_scores = []
        
        for i in range(len(df)):
            score = scores[i]
            is_anomaly = labels[i] == 1
            risk_level = classify_risk_level(score, threshold)
            
            all_scores.append({
                'id': ids[i],
                'score': score,
                'is_anomaly': is_anomaly,
            })
            
            if is_anomaly:
                top_features = get_top_contributing_features(
                    X_scaled[i], mean_normal, std_normal, request.feature_cols
                )
                anomalies.append({
                    'id': ids[i],
                    'anomaly_score': score,
                    'is_anomaly': True,
                    'risk_level': risk_level,
                    'top_contributing_features': top_features,
                })
        
        # Sort anomalies by score
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        # Risk distribution
        risk_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for a in anomalies:
            risk_counts[a['risk_level']] += 1
        
        total_anomalies = len(anomalies)
        risk_distribution = [
            {'level': level, 'count': count, 'percent': (count / total_anomalies * 100) if total_anomalies > 0 else 0}
            for level, count in risk_counts.items()
        ]
        
        # Summary
        summary = {
            'total_records': len(df),
            'anomaly_count': total_anomalies,
            'anomaly_rate': total_anomalies / len(df) if len(df) > 0 else 0,
            'critical_count': risk_counts['critical'],
            'high_count': risk_counts['high'],
            'medium_count': risk_counts['medium'],
            'low_count': risk_counts['low'],
            'avg_anomaly_score': float(np.mean([a['anomaly_score'] for a in anomalies])) if anomalies else 0,
            'threshold': threshold,
        }
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {
            'score_distribution': create_score_distribution_chart(scores, threshold, labels),
            'feature_importance': create_feature_importance_chart(feature_importance),
            'scatter_plot': create_scatter_plot(X_scaled, scores, labels, request.feature_cols),
        }
        
        # Key insights
        key_insights = generate_key_insights(summary, feature_importance)
        
        results = {
            'summary': {k: _to_native_type(v) for k, v in summary.items()},
            'anomalies': [{k: _to_native_type(v) if not isinstance(v, list) else v for k, v in a.items()} for a in anomalies],
            'all_scores': [{k: _to_native_type(v) for k, v in s.items()} for s in all_scores],
            'feature_importance': [{k: _to_native_type(v) for k, v in f.items()} for f in feature_importance],
            'risk_distribution': [{k: _to_native_type(v) for k, v in r.items()} for r in risk_distribution],
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'method': request.method,
                'anomaly_rate': summary['anomaly_rate'],
                'critical_count': summary['critical_count'],
                'top_feature': feature_importance[0]['feature'] if feature_importance else 'N/A',
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FDS detection failed: {str(e)}")
