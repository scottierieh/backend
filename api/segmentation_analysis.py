"""
Customer Segmentation Strategy API
5-step framework for comprehensive customer segmentation analysis
1. Usage Behavior Statistics
2. Channel Characteristics Comparison
3. Preference-Purchase Relationship
4. Cluster Segmentation
5. Target Response Prediction
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class SegmentationRequest(BaseModel):
    data: List[Dict[str, Any]]
    behavior_cols: List[str]  # Usage behavior metrics
    channel_col: Optional[str] = None  # Channel column
    preference_cols: Optional[List[str]] = None  # Preference scores
    purchase_col: Optional[str] = None  # Purchase amount/frequency
    target_col: Optional[str] = None  # Response target (0/1)
    n_clusters: Optional[int] = None  # Number of clusters (auto if None)


def _to_native(obj):
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# =============================================================================
# Step 1: Usage Behavior Statistics
# =============================================================================
def analyze_usage_behavior(df: pd.DataFrame, behavior_cols: List[str]) -> Dict:
    stats_list = []
    
    for col in behavior_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) == 0:
            continue
        
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        stats_list.append({
            'metric': col,
            'n': len(values),
            'mean': _to_native(values.mean()),
            'std': _to_native(values.std()),
            'min': _to_native(values.min()),
            'q1': _to_native(q1),
            'median': _to_native(values.median()),
            'q3': _to_native(q3),
            'max': _to_native(values.max()),
            'skewness': _to_native(values.skew()),
            'cv': _to_native(values.std() / values.mean() * 100) if values.mean() != 0 else None
        })
    
    # Identify high variability metrics
    high_var_metrics = [s for s in stats_list if s.get('cv') and s['cv'] > 50]
    
    # User segments by behavior intensity
    behavior_scores = df[behavior_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    q33, q66 = behavior_scores.quantile([0.33, 0.66])
    
    segments = {
        'low': len(behavior_scores[behavior_scores <= q33]),
        'medium': len(behavior_scores[(behavior_scores > q33) & (behavior_scores <= q66)]),
        'high': len(behavior_scores[behavior_scores > q66])
    }
    
    return {
        'statistics': stats_list,
        'n_metrics': len(stats_list),
        'n_customers': len(df),
        'high_variability_metrics': [m['metric'] for m in high_var_metrics],
        'behavior_segments': segments,
        'segment_pct': {
            'low': _to_native(segments['low'] / len(df) * 100),
            'medium': _to_native(segments['medium'] / len(df) * 100),
            'high': _to_native(segments['high'] / len(df) * 100)
        }
    }


def create_usage_chart(usage_data: Dict, df: pd.DataFrame, behavior_cols: List[str]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Metric means comparison
    ax1 = axes[0]
    stats = usage_data.get('statistics', [])
    if stats:
        metrics = [s['metric'][:12] for s in stats]
        means = [s['mean'] for s in stats]
        ax1.barh(metrics, means, color='#3b82f6', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Mean Value')
        ax1.set_title('Behavior Metrics Overview', fontsize=11, fontweight='bold')
    
    # Chart 2: Coefficient of Variation
    ax2 = axes[1]
    if stats:
        cvs = [s['cv'] or 0 for s in stats]
        colors = ['#ef4444' if cv > 50 else '#10b981' for cv in cvs]
        ax2.barh(metrics, cvs, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='High variability')
        ax2.set_xlabel('CV (%)')
        ax2.set_title('Metric Variability', fontsize=11, fontweight='bold')
    
    # Chart 3: Behavior segments pie
    ax3 = axes[2]
    segments = usage_data.get('behavior_segments', {})
    if segments:
        sizes = [segments['low'], segments['medium'], segments['high']]
        labels = ['Low', 'Medium', 'High']
        colors = ['#ef4444', '#f59e0b', '#10b981']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Behavior Intensity Segments', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Channel Characteristics Comparison
# =============================================================================
def analyze_channel_comparison(df: pd.DataFrame, behavior_cols: List[str], channel_col: str) -> Dict:
    channel_stats = []
    
    for channel in df[channel_col].unique():
        channel_data = df[df[channel_col] == channel]
        behavior_means = channel_data[behavior_cols].apply(pd.to_numeric, errors='coerce').mean()
        
        channel_stats.append({
            'channel': _to_native(channel),
            'n_customers': len(channel_data),
            'pct': _to_native(len(channel_data) / len(df) * 100),
            'behavior_means': {col: _to_native(behavior_means[col]) for col in behavior_cols}
        })
    
    channel_stats = sorted(channel_stats, key=lambda x: x['n_customers'], reverse=True)
    
    # ANOVA for each behavior metric
    metric_differences = []
    for col in behavior_cols:
        groups = [df[df[channel_col] == ch][col].dropna().values for ch in df[channel_col].unique()]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                significant = bool(p_value < 0.05)
            except:
                f_stat, p_value, significant = None, None, False
        else:
            f_stat, p_value, significant = None, None, False
        
        metric_differences.append({
            'metric': col,
            'f_statistic': _to_native(f_stat),
            'p_value': _to_native(p_value),
            'significant': significant
        })
    
    # Best channel for each metric
    best_channels = {}
    for col in behavior_cols:
        channel_means = df.groupby(channel_col)[col].mean()
        best_channels[col] = _to_native(channel_means.idxmax())
    
    return {
        'channel_stats': channel_stats,
        'n_channels': len(channel_stats),
        'metric_differences': metric_differences,
        'n_significant': sum(1 for m in metric_differences if m['significant']),
        'best_channels': best_channels,
        'largest_channel': channel_stats[0] if channel_stats else None
    }


def create_channel_chart(channel_data: Dict, df: pd.DataFrame, behavior_cols: List[str], channel_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Channel distribution
    ax1 = axes[0]
    channels = [c['channel'] for c in channel_data['channel_stats']]
    counts = [c['n_customers'] for c in channel_data['channel_stats']]
    colors = plt.cm.Set3(np.linspace(0, 1, len(channels)))
    ax1.bar(channels, counts, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Customers')
    ax1.set_title('Customer Distribution by Channel', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Chart 2: Behavior comparison by channel
    ax2 = axes[1]
    channel_means = df.groupby(channel_col)[behavior_cols].mean()
    channel_means.plot(kind='bar', ax=ax2, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Mean Value')
    ax2.set_title('Behavior Metrics by Channel', fontsize=11, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Preference-Purchase Relationship
# =============================================================================
def analyze_preference_purchase(df: pd.DataFrame, preference_cols: List[str], purchase_col: str) -> Dict:
    df_clean = df[preference_cols + [purchase_col]].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Insufficient data'}
    
    purchase_values = df_clean[purchase_col]
    
    # Correlations
    correlations = []
    for pref in preference_cols:
        corr, p_value = stats.pearsonr(df_clean[pref], purchase_values)
        correlations.append({
            'preference': pref,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_value),
            'significant': bool(p_value < 0.05),
            'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
        })
    
    correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    # Preference groups and purchase analysis
    pref_score = df_clean[preference_cols].mean(axis=1)
    q33, q66 = pref_score.quantile([0.33, 0.66])
    
    groups = {
        'low_preference': purchase_values[pref_score <= q33].mean(),
        'medium_preference': purchase_values[(pref_score > q33) & (pref_score <= q66)].mean(),
        'high_preference': purchase_values[pref_score > q66].mean()
    }
    
    # Lift calculation
    baseline = purchase_values.mean()
    lift = {
        'low': _to_native((groups['low_preference'] / baseline - 1) * 100) if baseline > 0 else 0,
        'medium': _to_native((groups['medium_preference'] / baseline - 1) * 100) if baseline > 0 else 0,
        'high': _to_native((groups['high_preference'] / baseline - 1) * 100) if baseline > 0 else 0
    }
    
    return {
        'correlations': correlations,
        'top_predictor': correlations[0] if correlations else None,
        'purchase_by_preference': {
            'low': _to_native(groups['low_preference']),
            'medium': _to_native(groups['medium_preference']),
            'high': _to_native(groups['high_preference'])
        },
        'lift': lift,
        'n_observations': len(df_clean),
        'n_significant': sum(1 for c in correlations if c['significant'])
    }


def create_preference_chart(pref_data: Dict, df: pd.DataFrame, preference_cols: List[str], purchase_col: str) -> str:
    if pref_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Correlation bars
    ax1 = axes[0]
    corrs = pref_data.get('correlations', [])
    if corrs:
        prefs = [c['preference'][:12] for c in corrs]
        vals = [c['correlation'] for c in corrs]
        colors = ['#10b981' if v > 0 else '#ef4444' for v in vals]
        ax1.barh(prefs, vals, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Correlation with Purchase')
        ax1.set_title('Preference-Purchase Correlation', fontsize=11, fontweight='bold')
    
    # Chart 2: Purchase by preference level
    ax2 = axes[1]
    purch = pref_data.get('purchase_by_preference', {})
    if purch:
        levels = ['Low', 'Medium', 'High']
        values = [purch['low'], purch['medium'], purch['high']]
        colors = ['#ef4444', '#f59e0b', '#10b981']
        ax2.bar(levels, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Average Purchase')
        ax2.set_title('Purchase by Preference Level', fontsize=11, fontweight='bold')
    
    # Chart 3: Lift
    ax3 = axes[2]
    lift = pref_data.get('lift', {})
    if lift:
        levels = ['Low', 'Medium', 'High']
        values = [lift['low'], lift['medium'], lift['high']]
        colors = ['#ef4444' if v < 0 else '#10b981' for v in values]
        ax3.bar(levels, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Lift vs Baseline (%)')
        ax3.set_title('Purchase Lift by Preference', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Cluster Segmentation
# =============================================================================
def analyze_clusters(df: pd.DataFrame, behavior_cols: List[str], n_clusters: Optional[int] = None) -> Dict:
    df_clean = df[behavior_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Insufficient data for clustering'}
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Determine optimal clusters if not specified
    if n_clusters is None:
        silhouette_scores = []
        for k in range(2, min(8, len(df_clean) // 5)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append({'k': k, 'score': score})
        
        best_k = max(silhouette_scores, key=lambda x: x['score'])
        n_clusters = best_k['k']
    else:
        silhouette_scores = []
    
    # Final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    df_clustered = df_clean.copy()
    df_clustered['cluster'] = labels
    
    # Cluster profiles
    cluster_profiles = []
    for c in range(n_clusters):
        cluster_data = df_clustered[df_clustered['cluster'] == c]
        profile = {
            'cluster': c,
            'n_customers': len(cluster_data),
            'pct': _to_native(len(cluster_data) / len(df_clustered) * 100),
            'metrics': {}
        }
        
        for col in behavior_cols:
            profile['metrics'][col] = {
                'mean': _to_native(cluster_data[col].mean()),
                'std': _to_native(cluster_data[col].std())
            }
        
        # Cluster naming based on behavior
        overall_means = df_clean[behavior_cols].mean()
        cluster_means = cluster_data[behavior_cols].mean()
        relative = (cluster_means / overall_means).mean()
        
        if relative > 1.3:
            profile['label'] = 'High Value'
        elif relative > 1.1:
            profile['label'] = 'Above Average'
        elif relative > 0.9:
            profile['label'] = 'Average'
        elif relative > 0.7:
            profile['label'] = 'Below Average'
        else:
            profile['label'] = 'Low Engagement'
        
        cluster_profiles.append(profile)
    
    # Sort by size
    cluster_profiles = sorted(cluster_profiles, key=lambda x: x['n_customers'], reverse=True)
    
    # Silhouette score
    final_silhouette = silhouette_score(X_scaled, labels)
    
    return {
        'n_clusters': n_clusters,
        'cluster_profiles': cluster_profiles,
        'silhouette_score': _to_native(final_silhouette),
        'quality': 'good' if final_silhouette > 0.5 else 'moderate' if final_silhouette > 0.3 else 'weak',
        'silhouette_analysis': silhouette_scores,
        'cluster_labels': labels.tolist()
    }


def create_cluster_chart(cluster_data: Dict, df: pd.DataFrame, behavior_cols: List[str]) -> str:
    if cluster_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    profiles = cluster_data.get('cluster_profiles', [])
    
    # Chart 1: Cluster sizes
    ax1 = axes[0]
    if profiles:
        names = [f"C{p['cluster']}: {p['label']}" for p in profiles]
        sizes = [p['n_customers'] for p in profiles]
        colors = plt.cm.Set2(np.linspace(0, 1, len(profiles)))
        ax1.pie(sizes, labels=names, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Cluster Distribution', fontsize=11, fontweight='bold')
    
    # Chart 2: Cluster profiles radar-like bar chart
    ax2 = axes[1]
    if profiles and behavior_cols:
        x = np.arange(len(behavior_cols))
        width = 0.8 / len(profiles)
        
        for i, profile in enumerate(profiles):
            means = [profile['metrics'][col]['mean'] for col in behavior_cols]
            ax2.bar(x + i * width, means, width, label=f"C{profile['cluster']}", alpha=0.8)
        
        ax2.set_xticks(x + width * (len(profiles) - 1) / 2)
        ax2.set_xticklabels([c[:10] for c in behavior_cols], rotation=45, ha='right')
        ax2.set_ylabel('Mean Value')
        ax2.set_title('Cluster Behavior Profiles', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
    
    # Chart 3: Silhouette analysis
    ax3 = axes[2]
    sil_analysis = cluster_data.get('silhouette_analysis', [])
    if sil_analysis:
        ks = [s['k'] for s in sil_analysis]
        scores = [s['score'] for s in sil_analysis]
        ax3.plot(ks, scores, 'bo-', markersize=8)
        ax3.axhline(y=cluster_data['silhouette_score'], color='red', linestyle='--', 
                   label=f"Selected: {cluster_data['silhouette_score']:.3f}")
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Optimal Cluster Selection', fontsize=11, fontweight='bold')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, f"Silhouette: {cluster_data['silhouette_score']:.3f}\n({cluster_data['quality']})",
                ha='center', va='center', fontsize=14)
        ax3.set_title('Cluster Quality', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Target Response Prediction
# =============================================================================
def analyze_target_prediction(df: pd.DataFrame, behavior_cols: List[str], target_col: str, 
                             cluster_labels: Optional[List[int]] = None) -> Dict:
    df_clean = df[behavior_cols + [target_col]].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) < 20:
        return {'error': 'Insufficient data for prediction'}
    
    X = df_clean[behavior_cols]
    y = df_clean[target_col]
    
    # Check if binary
    unique_vals = y.unique()
    if len(unique_vals) > 2:
        # Convert to binary (above/below median)
        y = (y > y.median()).astype(int)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    # Fit final model
    model.fit(X_scaled, y)
    
    # Feature importance
    importances = []
    for col, coef in zip(behavior_cols, model.coef_[0]):
        importances.append({
            'feature': col,
            'coefficient': _to_native(coef),
            'importance': _to_native(abs(coef)),
            'direction': 'positive' if coef > 0 else 'negative'
        })
    
    importances = sorted(importances, key=lambda x: x['importance'], reverse=True)
    
    # Overall response rate
    response_rate = y.mean()
    
    # Response by cluster if available
    cluster_response = None
    if cluster_labels is not None and len(cluster_labels) == len(df_clean):
        df_clean['cluster'] = cluster_labels[:len(df_clean)]
        cluster_response = []
        for c in df_clean['cluster'].unique():
            cluster_data = df_clean[df_clean['cluster'] == c]
            rate = cluster_data[target_col].mean() if target_col in cluster_data else cluster_data.iloc[:, -2].mean()
            cluster_response.append({
                'cluster': _to_native(c),
                'response_rate': _to_native(rate * 100),
                'n': len(cluster_data),
                'index': _to_native(rate / response_rate * 100) if response_rate > 0 else 100
            })
        cluster_response = sorted(cluster_response, key=lambda x: x['response_rate'], reverse=True)
    
    # Decile analysis
    proba = model.predict_proba(X_scaled)[:, 1]
    df_clean['proba'] = proba
    df_clean['decile'] = pd.qcut(proba, 10, labels=False, duplicates='drop')
    
    decile_analysis = []
    for d in sorted(df_clean['decile'].unique(), reverse=True):
        decile_data = df_clean[df_clean['decile'] == d]
        rate = decile_data[target_col].mean() if target_col in decile_data.columns else y[df_clean['decile'] == d].mean()
        decile_analysis.append({
            'decile': _to_native(d + 1),
            'response_rate': _to_native(rate * 100),
            'n': len(decile_data),
            'cumulative_pct': _to_native((d + 1) * 10)
        })
    
    return {
        'model_accuracy': _to_native(cv_scores.mean()),
        'accuracy_std': _to_native(cv_scores.std()),
        'feature_importance': importances,
        'top_predictor': importances[0] if importances else None,
        'overall_response_rate': _to_native(response_rate * 100),
        'cluster_response': cluster_response,
        'decile_analysis': decile_analysis,
        'n_observations': len(df_clean)
    }


def create_prediction_chart(pred_data: Dict) -> str:
    if pred_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Feature importance
    ax1 = axes[0]
    importance = pred_data.get('feature_importance', [])
    if importance:
        features = [f['feature'][:12] for f in importance]
        values = [f['coefficient'] for f in importance]
        colors = ['#10b981' if v > 0 else '#ef4444' for v in values]
        ax1.barh(features, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Coefficient')
        ax1.set_title('Feature Importance', fontsize=11, fontweight='bold')
    
    # Chart 2: Cluster response (or decile)
    ax2 = axes[1]
    cluster_resp = pred_data.get('cluster_response')
    if cluster_resp:
        clusters = [f"C{c['cluster']}" for c in cluster_resp]
        rates = [c['response_rate'] for c in cluster_resp]
        baseline = pred_data['overall_response_rate']
        colors = ['#10b981' if r > baseline else '#ef4444' for r in rates]
        ax2.bar(clusters, rates, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline: {baseline:.1f}%')
        ax2.set_ylabel('Response Rate (%)')
        ax2.set_title('Response by Cluster', fontsize=11, fontweight='bold')
        ax2.legend()
    else:
        decile = pred_data.get('decile_analysis', [])
        if decile:
            deciles = [f"D{d['decile']}" for d in decile]
            rates = [d['response_rate'] for d in decile]
            ax2.bar(deciles, rates, color='#3b82f6', alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Response Rate (%)')
            ax2.set_title('Response by Decile', fontsize=11, fontweight='bold')
    
    # Chart 3: Model performance
    ax3 = axes[2]
    accuracy = pred_data.get('model_accuracy', 0)
    ax3.bar(['Accuracy'], [accuracy * 100], color='#3b82f6', alpha=0.7, edgecolor='black')
    ax3.axhline(y=50, color='gray', linestyle='--', label='Random')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(0, 100)
    ax3.set_title(f"Model Accuracy: {accuracy*100:.1f}%", fontsize=11, fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(usage: Dict, channel: Optional[Dict], pref: Optional[Dict], 
                   cluster: Dict, prediction: Optional[Dict]) -> Dict:
    report = {}
    
    report['step1_usage'] = {
        'title': '1. Usage Behavior Statistics',
        'question': 'What are the key usage behavior patterns?',
        'finding': f"{usage['n_customers']} customers, {usage['n_metrics']} metrics analyzed",
        'detail': f"Analysis of {usage['n_metrics']} behavior metrics shows {len(usage['high_variability_metrics'])} high-variability metrics. "
                 f"Customer segments: High engagement {usage['segment_pct']['high']:.1f}%, "
                 f"Medium {usage['segment_pct']['medium']:.1f}%, Low {usage['segment_pct']['low']:.1f}%."
    }
    
    if channel and not channel.get('error'):
        report['step2_channel'] = {
            'title': '2. Channel Characteristics',
            'question': 'How do channels differ in customer behavior?',
            'finding': f"{channel['n_channels']} channels, {channel['n_significant']} significant differences",
            'detail': f"Comparison across {channel['n_channels']} channels shows {channel['n_significant']} metrics with significant differences. "
                     f"Largest channel: {channel['largest_channel']['channel']} ({channel['largest_channel']['pct']:.1f}% of customers)."
        }
    else:
        report['step2_channel'] = {
            'title': '2. Channel Characteristics',
            'question': 'How do channels differ in customer behavior?',
            'finding': 'Channel analysis not performed',
            'detail': 'No channel column specified.'
        }
    
    if pref and not pref.get('error'):
        top = pref.get('top_predictor', {})
        report['step3_preference'] = {
            'title': '3. Preference-Purchase Relationship',
            'question': 'How do preferences relate to purchase behavior?',
            'finding': f"Top predictor: {top.get('preference', 'N/A')} (r={top.get('correlation', 0):.3f})",
            'detail': f"Analysis shows {pref['n_significant']} preferences significantly correlated with purchase. "
                     f"High-preference customers show {pref['lift']['high']:.1f}% lift vs baseline."
        }
    else:
        report['step3_preference'] = {
            'title': '3. Preference-Purchase Relationship',
            'question': 'How do preferences relate to purchase behavior?',
            'finding': 'Preference analysis not performed',
            'detail': pref.get('error', 'No preference/purchase columns specified.')
        }
    
    if cluster and not cluster.get('error'):
        report['step4_cluster'] = {
            'title': '4. Cluster Segmentation',
            'question': 'What distinct customer segments exist?',
            'finding': f"{cluster['n_clusters']} clusters identified (silhouette: {cluster['silhouette_score']:.3f})",
            'detail': f"K-Means clustering identified {cluster['n_clusters']} distinct segments. "
                     f"Quality: {cluster['quality']} (silhouette={cluster['silhouette_score']:.3f}). "
                     + ' '.join([f"{p['label']}: {p['pct']:.1f}%." for p in cluster['cluster_profiles'][:3]])
        }
    else:
        report['step4_cluster'] = {
            'title': '4. Cluster Segmentation',
            'question': 'What distinct customer segments exist?',
            'finding': 'Clustering not performed',
            'detail': cluster.get('error', 'Insufficient data for clustering.')
        }
    
    if prediction and not prediction.get('error'):
        report['step5_prediction'] = {
            'title': '5. Target Response Prediction',
            'question': 'Which customers are likely to respond?',
            'finding': f"Model accuracy: {prediction['model_accuracy']*100:.1f}%, Top predictor: {prediction['top_predictor']['feature']}",
            'detail': f"Logistic regression achieves {prediction['model_accuracy']*100:.1f}% accuracy (±{prediction['accuracy_std']*100:.1f}%). "
                     f"Overall response rate: {prediction['overall_response_rate']:.1f}%. "
                     f"Top predictor: {prediction['top_predictor']['feature']} ({prediction['top_predictor']['direction']})."
        }
    else:
        report['step5_prediction'] = {
            'title': '5. Target Response Prediction',
            'question': 'Which customers are likely to respond?',
            'finding': 'Prediction not performed',
            'detail': prediction.get('error', 'No target column specified.')
        }
    
    return report


def generate_insights(usage: Dict, channel: Optional[Dict], pref: Optional[Dict],
                     cluster: Dict, prediction: Optional[Dict]) -> List[Dict]:
    insights = []
    
    # Usage insights
    if usage['segment_pct']['high'] < 20:
        insights.append({
            'title': 'Low High-Engagement Rate',
            'description': f"Only {usage['segment_pct']['high']:.1f}% of customers show high engagement.",
            'status': 'warning'
        })
    
    # Cluster insights
    if cluster and not cluster.get('error'):
        high_value = [p for p in cluster['cluster_profiles'] if 'High' in p.get('label', '')]
        if high_value:
            insights.append({
                'title': 'High Value Segment Identified',
                'description': f"High Value segment contains {high_value[0]['pct']:.1f}% of customers.",
                'status': 'positive'
            })
    
    # Prediction insights
    if prediction and not prediction.get('error'):
        if prediction['model_accuracy'] > 0.7:
            insights.append({
                'title': 'Strong Prediction Model',
                'description': f"Model achieves {prediction['model_accuracy']*100:.1f}% accuracy for targeting.",
                'status': 'positive'
            })
        
        if prediction.get('cluster_response'):
            best = prediction['cluster_response'][0]
            if best['index'] > 150:
                insights.append({
                    'title': 'High Response Cluster Found',
                    'description': f"Cluster {best['cluster']} shows {best['index']:.0f}% response index vs baseline.",
                    'status': 'positive'
                })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/segmentation-analysis")
async def analyze_segmentation(request: SegmentationRequest):
    try:
        df = pd.DataFrame(request.data)
        behavior_cols = request.behavior_cols
        channel_col = request.channel_col
        preference_cols = request.preference_cols or []
        purchase_col = request.purchase_col
        target_col = request.target_col
        n_clusters = request.n_clusters
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points")
        
        # Convert behavior columns to numeric
        for col in behavior_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Usage Behavior
        usage = analyze_usage_behavior(df, behavior_cols)
        results['usage'] = usage
        visualizations['usage_chart'] = create_usage_chart(usage, df, behavior_cols)
        
        # Step 2: Channel Comparison
        channel = None
        if channel_col and channel_col in df.columns:
            channel = analyze_channel_comparison(df, behavior_cols, channel_col)
            results['channel'] = channel
            visualizations['channel_chart'] = create_channel_chart(channel, df, behavior_cols, channel_col)
        
        # Step 3: Preference-Purchase
        pref = None
        if preference_cols and purchase_col and purchase_col in df.columns:
            valid_prefs = [c for c in preference_cols if c in df.columns]
            if valid_prefs:
                pref = analyze_preference_purchase(df, valid_prefs, purchase_col)
                results['preference'] = pref
                if not pref.get('error'):
                    visualizations['preference_chart'] = create_preference_chart(pref, df, valid_prefs, purchase_col)
        
        # Step 4: Clustering
        cluster = analyze_clusters(df, behavior_cols, n_clusters)
        results['cluster'] = cluster
        if not cluster.get('error'):
            visualizations['cluster_chart'] = create_cluster_chart(cluster, df, behavior_cols)
        
        # Step 5: Prediction
        prediction = None
        if target_col and target_col in df.columns:
            cluster_labels = cluster.get('cluster_labels') if not cluster.get('error') else None
            prediction = analyze_target_prediction(df, behavior_cols, target_col, cluster_labels)
            results['prediction'] = prediction
            if not prediction.get('error'):
                visualizations['prediction_chart'] = create_prediction_chart(prediction)
        
        report = generate_report(usage, channel, pref, cluster, prediction)
        insights = generate_insights(usage, channel, pref, cluster, prediction)
        
        summary = {
            'n_customers': usage['n_customers'],
            'n_clusters': cluster.get('n_clusters', 0) if not cluster.get('error') else 0,
            'model_accuracy': prediction.get('model_accuracy') if prediction and not prediction.get('error') else None,
            'high_engagement_pct': usage['segment_pct']['high']
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'report': report,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
