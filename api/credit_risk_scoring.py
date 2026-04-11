"""
Credit Risk Scoring Router for FastAPI
Implements credit scoring models with AUC-ROC, Gini, KS statistics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import time
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CreditRiskRequest(BaseModel):
    data: List[Dict[str, Any]]
    target_col: str
    feature_cols: List[str]
    model_type: Literal["logistic", "xgboost", "random_forest", "lightgbm"] = "logistic"
    test_size: float = 0.2


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


def probability_to_score(prob: np.ndarray, base_score: int = 600, pdo: int = 20, base_odds: float = 50) -> np.ndarray:
    """Convert default probability to credit score (300-850 scale)"""
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    odds = (1 - prob) / prob
    scores = base_score + pdo * np.log(odds / base_odds) / np.log(2)
    return np.clip(scores, 300, 850)


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str):
    """Train credit risk model"""
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    model.fit(X_train, y_train)
    return model


def calculate_model_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate model performance metrics"""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
    
    auc_roc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc_roc - 1
    ks_stat = calculate_ks_statistic(y_true, y_prob)
    
    return {
        'auc_roc': auc_roc,
        'gini': gini,
        'ks_statistic': ks_stat,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'log_loss': log_loss(y_true, y_prob),
    }


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic"""
    sorted_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_idx]
    
    n_bad = y_true.sum()
    n_good = len(y_true) - n_bad
    
    if n_bad == 0 or n_good == 0:
        return 0.0
    
    cum_bad = np.cumsum(y_true_sorted) / n_bad
    cum_good = np.cumsum(1 - y_true_sorted) / n_good
    
    return float(np.max(np.abs(cum_bad - cum_good)))


def get_feature_importance(model, feature_names: List[str], model_type: str) -> List[Dict]:
    """Extract feature importance from model"""
    if model_type == "logistic":
        coefs = model.coef_[0]
        importance = np.abs(coefs)
        importance = importance / (importance.sum() + 1e-10)
        
        results = []
        for name, imp, coef in zip(feature_names, importance, coefs):
            results.append({
                'feature': name,
                'importance': float(imp),
                'coefficient': float(coef),
                'direction': 'positive' if coef > 0 else 'negative'
            })
    else:
        importance = model.feature_importances_
        importance = importance / (importance.sum() + 1e-10)
        
        results = []
        for name, imp in zip(feature_names, importance):
            results.append({
                'feature': name,
                'importance': float(imp),
                'coefficient': 0.0,
                'direction': 'unknown'
            })
    
    results.sort(key=lambda x: x['importance'], reverse=True)
    return results


def create_risk_segments(scores: np.ndarray, y_true: np.ndarray) -> List[Dict]:
    """Create risk segments based on score ranges"""
    segments = [
        ('Very Low', 750, 850),
        ('Low', 700, 749),
        ('Medium', 650, 699),
        ('High', 550, 649),
        ('Very High', 300, 549),
    ]
    
    recommendations = {
        'Very Low': 'Auto-approve with best rates',
        'Low': 'Approve with standard rates',
        'Medium': 'Manual review, consider higher rates',
        'High': 'Decline or require collateral',
        'Very High': 'Decline application',
    }
    
    total = len(scores)
    results = []
    
    for name, low, high in segments:
        mask = (scores >= low) & (scores <= high)
        count = mask.sum()
        
        if count > 0:
            default_rate = y_true[mask].sum() / count
            avg_score = scores[mask].mean()
        else:
            default_rate = 0
            avg_score = (low + high) / 2
        
        results.append({
            'segment': name,
            'score_range': f'{low}-{high}',
            'count': int(count),
            'pct': count / total if total > 0 else 0,
            'default_rate': float(default_rate),
            'avg_score': float(avg_score),
            'recommendation': recommendations[name],
        })
    
    return results


def create_score_distribution(scores: np.ndarray, y_true: np.ndarray) -> List[Dict]:
    """Create score distribution analysis"""
    bins = np.arange(300, 875, 50)
    bin_labels = [f'{b}-{b+49}' for b in bins[:-1]]
    
    digitized = np.digitize(scores, bins) - 1
    digitized = np.clip(digitized, 0, len(bin_labels) - 1)
    
    results = []
    cumulative_defaults = 0
    cumulative_total = 0
    
    for i, label in enumerate(bin_labels):
        mask = digitized == i
        count = mask.sum()
        defaults = y_true[mask].sum() if count > 0 else 0
        
        cumulative_defaults += defaults
        cumulative_total += count
        
        results.append({
            'bin': label,
            'count': int(count),
            'default_rate': float(defaults / count) if count > 0 else 0,
            'cumulative_default_rate': float(cumulative_defaults / cumulative_total) if cumulative_total > 0 else 0,
        })
    
    return results


def find_optimal_cutoff(scores: np.ndarray, y_true: np.ndarray) -> int:
    """Find optimal approval cutoff"""
    cutoffs = np.arange(500, 750, 10)
    best_cutoff = 600
    best_score = -np.inf
    
    for cutoff in cutoffs:
        approved = scores >= cutoff
        approval_rate = approved.sum() / len(scores)
        
        if approved.sum() > 0:
            default_rate = y_true[approved].sum() / approved.sum()
        else:
            default_rate = 1.0
        
        score = approval_rate * (1 - default_rate * 5)
        
        if score > best_score:
            best_score = score
            best_cutoff = cutoff
    
    return int(best_cutoff)


def create_cutoff_analysis(scores: np.ndarray, y_true: np.ndarray) -> List[Dict]:
    """Analyze different cutoff scenarios"""
    cutoffs = [500, 550, 600, 650, 700, 750]
    results = []
    base_default_rate = y_true.mean()
    
    for cutoff in cutoffs:
        approved = scores >= cutoff
        approval_rate = approved.sum() / len(scores)
        default_rate = y_true[approved].sum() / approved.sum() if approved.sum() > 0 else 0
        loss_reduction = (base_default_rate - default_rate * approval_rate) / base_default_rate if base_default_rate > 0 else 0
        
        results.append({
            'cutoff': cutoff,
            'approval_rate': float(approval_rate),
            'default_rate_if_approved': float(default_rate),
            'expected_loss_reduction': float(loss_reduction),
        })
    
    return results


# ============ VISUALIZATION ============
def create_score_distribution_chart(scores: np.ndarray, y_true: np.ndarray) -> str:
    """Create score distribution histogram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scores_good = scores[y_true == 0]
    scores_bad = scores[y_true == 1]
    bins = np.arange(300, 875, 25)
    
    ax.hist(scores_good, bins=bins, alpha=0.6, label='Non-Default', color='#22c55e', edgecolor='white')
    ax.hist(scores_bad, bins=bins, alpha=0.6, label='Default', color='#ef4444', edgecolor='white')
    
    ax.set_xlabel('Credit Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Score Distribution by Default Status', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc: float) -> str:
    """Create ROC curve"""
    from sklearn.metrics import roc_curve
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    ax.plot(fpr, tpr, color='#3b82f6', linewidth=2, label=f'Model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#3b82f6')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_feature_importance_chart(feature_importance: List[Dict]) -> str:
    """Create feature importance bar chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = [f['feature'] for f in feature_importance[:12]]
    importances = [f['importance'] for f in feature_importance[:12]]
    directions = [f['direction'] for f in feature_importance[:12]]
    
    colors = ['#ef4444' if d == 'positive' else '#22c55e' if d == 'negative' else '#3b82f6' for d in directions]
    
    y_pos = range(len(features))
    bars = ax.barh(y_pos, importances, color=colors, edgecolor='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Feature Importance (Red = Risk, Green = Protective)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{imp:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_ks_chart(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """Create KS chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_idx]
    
    n_bad = y_true.sum()
    n_good = len(y_true) - n_bad
    
    cum_bad = np.cumsum(y_true_sorted) / n_bad
    cum_good = np.cumsum(1 - y_true_sorted) / n_good
    x = np.arange(len(y_true)) / len(y_true)
    
    ax.plot(x, cum_bad, color='#ef4444', linewidth=2, label='Cumulative Bad')
    ax.plot(x, cum_good, color='#22c55e', linewidth=2, label='Cumulative Good')
    
    ks_idx = np.argmax(np.abs(cum_bad - cum_good))
    ks_stat = np.abs(cum_bad - cum_good)[ks_idx]
    
    ax.axvline(x=x[ks_idx], color='#3b82f6', linestyle='--', alpha=0.7)
    ax.annotate(f'KS = {ks_stat:.3f}', xy=(x[ks_idx], (cum_bad[ks_idx] + cum_good[ks_idx])/2),
                xytext=(x[ks_idx] + 0.1, (cum_bad[ks_idx] + cum_good[ks_idx])/2), fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Population Proportion', fontsize=11)
    ax.set_ylabel('Cumulative Proportion', fontsize=11)
    ax.set_title('KS Chart', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_risk_segments_chart(risk_segments: List[Dict]) -> str:
    """Create risk segments chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    segments = [s['segment'] for s in risk_segments]
    counts = [s['count'] for s in risk_segments]
    default_rates = [s['default_rate'] * 100 for s in risk_segments]
    colors = ['#22c55e', '#84cc16', '#f59e0b', '#f97316', '#ef4444']
    
    ax1.bar(segments, counts, color=colors, edgecolor='white')
    ax1.set_xlabel('Risk Segment', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Population by Risk Segment', fontsize=12, fontweight='bold')
    
    for i, cnt in enumerate(counts):
        ax1.text(i, cnt + max(counts)*0.02, f'{cnt:,}', ha='center', fontsize=9)
    
    ax2.bar(segments, default_rates, color=colors, edgecolor='white')
    ax2.set_xlabel('Risk Segment', fontsize=11)
    ax2.set_ylabel('Default Rate (%)', fontsize=11)
    ax2.set_title('Default Rate by Risk Segment', fontsize=12, fontweight='bold')
    
    for i, rate in enumerate(default_rates):
        ax2.text(i, rate + max(default_rates)*0.02, f'{rate:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(metrics: Dict, risk_segments: List[Dict], feature_importance: List[Dict]) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    if metrics['auc_roc'] >= 0.75:
        insights.append({
            'title': f"Strong Model Performance (AUC = {metrics['auc_roc']:.3f})",
            'description': "Model shows excellent discrimination between defaulters and non-defaulters.",
            'status': 'positive'
        })
    elif metrics['auc_roc'] >= 0.7:
        insights.append({
            'title': f"Good Model Performance (AUC = {metrics['auc_roc']:.3f})",
            'description': "Model shows good discrimination. Consider adding more features.",
            'status': 'positive'
        })
    else:
        insights.append({
            'title': f"Model Needs Improvement (AUC = {metrics['auc_roc']:.3f})",
            'description': "Consider adding more predictive features or different model types.",
            'status': 'warning'
        })
    
    high_risk = [s for s in risk_segments if s['segment'] in ['High', 'Very High']]
    high_risk_pct = sum(s['pct'] for s in high_risk)
    if high_risk_pct > 0.3:
        insights.append({
            'title': f"High Risk Population: {high_risk_pct*100:.1f}%",
            'description': "Significant portion in high risk categories. Consider stricter criteria.",
            'status': 'warning'
        })
    
    if feature_importance:
        top = feature_importance[0]
        insights.append({
            'title': f"Top Risk Factor: {top['feature']}",
            'description': f"Accounts for {top['importance']*100:.1f}% of predictive power.",
            'status': 'neutral'
        })
    
    if metrics['ks_statistic'] >= 0.4:
        insights.append({
            'title': f"Excellent Separation (KS = {metrics['ks_statistic']:.3f})",
            'description': "Strong separation between good and bad customers.",
            'status': 'positive'
        })
    
    return insights


@router.post("/credit-risk")
async def run_credit_risk_analysis(request: CreditRiskRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        all_cols = [request.target_col] + request.feature_cols
        for col in all_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Prepare data
        y = pd.to_numeric(df[request.target_col], errors='coerce').values.astype(int)
        
        X_list = []
        feature_names = []
        for col in request.feature_cols:
            if df[col].dtype == 'object':
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                X_list.append(dummies.values)
                feature_names.extend(dummies.columns.tolist())
            else:
                vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values.reshape(-1, 1)
                X_list.append(vals)
                feature_names.append(col)
        
        X = np.hstack(X_list)
        X = np.nan_to_num(X, nan=0)
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=42, stratify=y)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = train_model(X_train_scaled, y_train, request.model_type)
        
        # Predictions
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        
        # Full dataset predictions
        X_scaled = scaler.transform(X)
        y_prob_full = model.predict_proba(X_scaled)[:, 1]
        scores = probability_to_score(y_prob_full)
        
        # Calculate metrics
        metrics = calculate_model_metrics(y_test, y_prob_test, y_pred_test)
        feature_importance = get_feature_importance(model, feature_names, request.model_type)
        risk_segments = create_risk_segments(scores, y)
        score_distribution = create_score_distribution(scores, y)
        optimal_cutoff = find_optimal_cutoff(scores, y)
        cutoff_analysis = create_cutoff_analysis(scores, y)
        
        summary_data = {
            'total_records': len(y),
            'default_rate': float(y.mean()),
            'avg_score': float(scores.mean()),
            'median_score': float(np.median(scores)),
            'score_std': float(scores.std()),
            'approved_rate': float((scores >= optimal_cutoff).sum() / len(scores)),
            'rejected_rate': float((scores < optimal_cutoff).sum() / len(scores)),
        }
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        visualizations = {
            'score_distribution': create_score_distribution_chart(scores, y),
            'roc_curve': create_roc_curve(y_test, y_prob_test, metrics['auc_roc']),
            'feature_importance': create_feature_importance_chart(feature_importance),
            'ks_chart': create_ks_chart(y_test, y_prob_test),
            'risk_segments': create_risk_segments_chart(risk_segments),
        }
        
        key_insights = generate_key_insights(metrics, risk_segments, feature_importance)
        
        results = {
            'summary': {k: _to_native_type(v) for k, v in summary_data.items()},
            'model_metrics': {k: _to_native_type(v) for k, v in metrics.items()},
            'feature_importance': [{k: _to_native_type(v) for k, v in f.items()} for f in feature_importance],
            'score_distribution': [{k: _to_native_type(v) for k, v in s.items()} for s in score_distribution],
            'risk_segments': [{k: _to_native_type(v) for k, v in s.items()} for s in risk_segments],
            'cutoff_analysis': [{k: _to_native_type(v) for k, v in c.items()} for c in cutoff_analysis],
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'model_type': request.model_type,
                'auc_roc': metrics['auc_roc'],
                'gini': metrics['gini'],
                'optimal_cutoff': optimal_cutoff,
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Credit risk analysis failed: {str(e)}")
