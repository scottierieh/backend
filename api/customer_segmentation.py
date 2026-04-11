"""
Customer Segmentation Router for FastAPI
RFM Analysis, K-Means Clustering, Customer Value Segmentation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class SegmentationRequest(BaseModel):
    data: List[Dict[str, Any]]
    customer_col: str  # Customer identifier
    date_col: Optional[str] = None  # Transaction date (for RFM)
    value_col: Optional[str] = None  # Transaction value/revenue
    quantity_col: Optional[str] = None  # Transaction quantity
    feature_cols: Optional[List[str]] = None  # Custom features for clustering
    analysis_type: Literal["rfm", "kmeans", "value_based"] = "rfm"
    n_segments: int = 4  # Number of segments for K-means
    reference_date: Optional[str] = None  # Reference date for recency calculation


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
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_rfm(df: pd.DataFrame, customer_col: str, date_col: str,
                  value_col: str, reference_date: Optional[str] = None) -> pd.DataFrame:
    """Calculate RFM metrics for each customer"""
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set reference date
    if reference_date:
        ref_date = pd.to_datetime(reference_date)
    else:
        ref_date = df[date_col].max() + timedelta(days=1)
    
    # Calculate RFM
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (ref_date - x.max()).days,  # Recency
        customer_col: 'count',  # Frequency (using customer_col as proxy for transaction count)
        value_col: 'sum'  # Monetary
    })
    
    # Rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    
    return rfm


def assign_rfm_scores(rfm: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Assign RFM scores (1-5) to each customer"""
    
    # Recency: lower is better (more recent)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=n_bins, labels=range(n_bins, 0, -1), duplicates='drop')
    
    # Frequency: higher is better
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=n_bins, labels=range(1, n_bins + 1), duplicates='drop')
    
    # Monetary: higher is better
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=n_bins, labels=range(1, n_bins + 1), duplicates='drop')
    
    # Convert to int
    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)
    
    # Combined RFM Score
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    return rfm


def assign_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign customer segments based on RFM scores"""
    
    def get_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Champions: Best customers
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        # Loyal: High frequency and monetary
        elif f >= 4 and m >= 4:
            return 'Loyal Customers'
        # Potential Loyalists: Recent with good frequency
        elif r >= 4 and f >= 3:
            return 'Potential Loyalists'
        # New Customers: Very recent, low frequency
        elif r >= 4 and f <= 2:
            return 'New Customers'
        # Promising: Recent, moderate activity
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Promising'
        # Need Attention: Above average but slipping
        elif r >= 3 and f >= 3:
            return 'Need Attention'
        # About to Sleep: Below average recency and frequency
        elif r >= 2 and f >= 2:
            return 'About to Sleep'
        # At Risk: High value but haven't purchased recently
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        # Can't Lose Them: Made big purchases but long time ago
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        # Hibernating: Low on all metrics
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        # Lost: Lowest engagement
        else:
            return 'Lost'
    
    rfm['Segment'] = rfm.apply(get_segment, axis=1)
    
    return rfm


def perform_kmeans_segmentation(df: pd.DataFrame, feature_cols: List[str],
                                 n_clusters: int, customer_col: str) -> Dict[str, Any]:
    """Perform K-means clustering on customer features"""
    
    # Prepare features
    features = df[feature_cols].copy()
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Find optimal k if needed
    if n_clusters <= 0:
        n_clusters = find_optimal_k(features_scaled, max_k=10)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    # Calculate silhouette score
    sil_score = silhouette_score(features_scaled, labels) if n_clusters > 1 else 0
    sil_samples = silhouette_samples(features_scaled, labels) if n_clusters > 1 else np.zeros(len(labels))
    
    # Create result dataframe
    result_df = df[[customer_col]].copy()
    result_df['Segment'] = labels
    result_df['Segment_Name'] = result_df['Segment'].apply(lambda x: f'Segment {x + 1}')
    
    for i, col in enumerate(feature_cols):
        result_df[col] = features.iloc[:, i].values
    
    # Cluster centers (inverse transform to original scale)
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # ========== 수정된 부분: Monetary 컬럼 확인하고 revenue 계산 ==========
    # Check if Monetary column exists (for revenue calculation)
    has_monetary = 'Monetary' in feature_cols
    total_monetary = features['Monetary'].sum() if has_monetary else None
    
    # Cluster statistics
    cluster_stats = []
    for i in range(n_clusters):
        mask = labels == i
        cluster_data = features[mask]
        stats = {
            'segment': i,
            'segment_name': f'Segment {i + 1}',
            'size': int(np.sum(mask)),
            'percentage': _to_native_type(np.sum(mask) / len(labels) * 100),
            'center': {col: _to_native_type(centers_original[i, j]) for j, col in enumerate(feature_cols)},
            'avg_silhouette': _to_native_type(np.mean(sil_samples[mask]))
        }
        
        # Add monetary metrics if available
        if has_monetary:
            cluster_monetary = cluster_data['Monetary']
            stats['total_monetary'] = _to_native_type(cluster_monetary.sum())
            stats['avg_monetary'] = _to_native_type(cluster_monetary.mean())
            stats['revenue_share'] = _to_native_type(cluster_monetary.sum() / total_monetary * 100) if total_monetary > 0 else 0
        
        for j, col in enumerate(feature_cols):
            stats[f'{col}_mean'] = _to_native_type(cluster_data.iloc[:, j].mean())
            stats[f'{col}_std'] = _to_native_type(cluster_data.iloc[:, j].std())
        cluster_stats.append(stats)
    # ========== 수정된 부분 끝 ==========
    
    return {
        'customer_segments': result_df,
        'cluster_stats': cluster_stats,
        'silhouette_score': _to_native_type(sil_score),
        'n_clusters': n_clusters,
        'feature_cols': feature_cols,
        'inertia': _to_native_type(kmeans.inertia_),
        'total_monetary': _to_native_type(total_monetary) if has_monetary else None  # 추가
    }


def find_optimal_k(X: np.ndarray, max_k: int = 10) -> int:
    """Find optimal number of clusters using elbow method and silhouette"""
    
    max_k = min(max_k, len(X) - 1)
    
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    # Find best k based on silhouette
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    return best_k


def calculate_value_segments(df: pd.DataFrame, customer_col: str,
                             value_col: str) -> pd.DataFrame:
    """Segment customers by value (CLV-based)"""
    
    # Aggregate by customer
    value_df = df.groupby(customer_col).agg({
        value_col: 'sum'
    }).reset_index()
    value_df.columns = [customer_col, 'Total_Value']
    
    # Calculate percentiles
    value_df['Percentile'] = value_df['Total_Value'].rank(pct=True) * 100
    
    # Assign segments based on percentiles
    def assign_segment(pct):
        if pct >= 90:
            return 'Platinum'
        elif pct >= 75:
            return 'Gold'
        elif pct >= 50:
            return 'Silver'
        elif pct >= 25:
            return 'Bronze'
        else:
            return 'Standard'
    
    value_df['Segment'] = value_df['Percentile'].apply(assign_segment)
    
    return value_df


# ============ Visualization Functions ============
def create_rfm_distribution_chart(rfm: pd.DataFrame) -> str:
    """Create RFM score distribution charts"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for idx, (col, title, color) in enumerate([
        ('R_Score', 'Recency Score', '#3b82f6'),
        ('F_Score', 'Frequency Score', '#22c55e'),
        ('M_Score', 'Monetary Score', '#f59e0b')
    ]):
        ax = axes[idx]
        counts = rfm[col].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color, alpha=0.8, edgecolor='white')
        ax.set_xlabel('Score')
        ax.set_ylabel('Customers')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_distribution_chart(rfm: pd.DataFrame) -> str:
    """Create segment distribution pie/bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    segment_counts = rfm['Segment'].value_counts()
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
    axes[0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[0].set_title('Segment Distribution', fontweight='bold')
    
    # Bar chart
    axes[1].barh(segment_counts.index, segment_counts.values, color=colors, edgecolor='white')
    axes[1].set_xlabel('Number of Customers')
    axes[1].set_title('Customers by Segment', fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_rfm_heatmap(rfm: pd.DataFrame) -> str:
    """Create RFM heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pivot table for R vs F
    pivot = rfm.groupby(['R_Score', 'F_Score']).size().unstack(fill_value=0)
    
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Customer Count'})
    ax.set_xlabel('Frequency Score')
    ax.set_ylabel('Recency Score')
    ax.set_title('RFM Heatmap (R vs F)', fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_profile_chart(rfm: pd.DataFrame) -> str:
    """Create segment profile radar/comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mean RFM scores per segment
    segment_means = rfm.groupby('Segment')[['R_Score', 'F_Score', 'M_Score']].mean()
    
    x = np.arange(len(segment_means))
    width = 0.25
    
    ax.bar(x - width, segment_means['R_Score'], width, label='Recency', color='#3b82f6', alpha=0.8)
    ax.bar(x, segment_means['F_Score'], width, label='Frequency', color='#22c55e', alpha=0.8)
    ax.bar(x + width, segment_means['M_Score'], width, label='Monetary', color='#f59e0b', alpha=0.8)
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Average Score')
    ax.set_title('RFM Scores by Segment', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(segment_means.index, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 5.5)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cluster_scatter(df: pd.DataFrame, feature_cols: List[str],
                           labels: np.ndarray) -> str:
    """Create cluster scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use first two features for 2D visualization
    x_col = feature_cols[0]
    y_col = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    
    scatter = ax.scatter(df[x_col], df[y_col], c=labels, cmap='Set2',
                         alpha=0.6, s=50, edgecolors='white')
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title('Customer Clusters', fontweight='bold')
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title='Cluster')
    ax.add_artist(legend)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_comparison_chart(cluster_stats: List[Dict], feature_cols: List[str]) -> str:
    """Create segment comparison bar chart"""
    fig, axes = plt.subplots(1, min(len(feature_cols), 3), figsize=(14, 5))
    
    if len(feature_cols) == 1:
        axes = [axes]
    
    segments = [s['segment_name'] for s in cluster_stats]
    colors = plt.cm.Set2(np.linspace(0, 1, len(segments)))
    
    for idx, col in enumerate(feature_cols[:3]):
        ax = axes[idx] if len(feature_cols) > 1 else axes[0]
        means = [s[f'{col}_mean'] for s in cluster_stats]
        
        ax.barh(segments, means, color=colors, edgecolor='white')
        ax.set_xlabel(f'Mean {col}')
        ax.set_title(f'{col} by Segment', fontweight='bold')
        ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_value_segment_chart(value_df: pd.DataFrame) -> str:
    """Create value segment visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count by segment
    segment_order = ['Platinum', 'Gold', 'Silver', 'Bronze', 'Standard']
    segment_counts = value_df['Segment'].value_counts().reindex(segment_order).fillna(0)
    
    colors = ['#E5E4E2', '#FFD700', '#C0C0C0', '#CD7F32', '#808080']
    
    # Customer count
    axes[0].bar(segment_counts.index, segment_counts.values, color=colors, edgecolor='white')
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('Customers')
    axes[0].set_title('Customer Count by Value Tier', fontweight='bold')
    
    # Revenue contribution
    segment_revenue = value_df.groupby('Segment')['Total_Value'].sum().reindex(segment_order).fillna(0)
    total_rev = segment_revenue.sum()
    revenue_pct = (segment_revenue / total_rev * 100) if total_rev > 0 else segment_revenue
    
    axes[1].pie(revenue_pct.values, labels=revenue_pct.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Revenue Share by Tier', fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(results: Dict, analysis_type: str) -> List[Dict]:
    """Generate key insights based on analysis results"""
    insights = []
    
    segment_dist = results.get('segment_distribution', {})
    
    if analysis_type == 'rfm':
        # Find champions
        champions = segment_dist.get('Champions', {})
        if champions:
            insights.append({
                'title': 'Champion Customers',
                'description': f'{champions.get("count", 0)} customers ({champions.get("percentage", 0):.1f}%) are Champions - your best customers.',
                'status': 'positive' if champions.get('percentage', 0) >= 10 else 'neutral'
            })
        
        # At risk customers
        at_risk = segment_dist.get('At Risk', {})
        cant_lose = segment_dist.get("Can't Lose Them", {})
        risk_count = at_risk.get('count', 0) + cant_lose.get('count', 0)
        risk_pct = at_risk.get('percentage', 0) + cant_lose.get('percentage', 0)
        
        if risk_count > 0:
            insights.append({
                'title': 'At Risk Alert',
                'description': f'{risk_count} customers ({risk_pct:.1f}%) need immediate attention to prevent churn.',
                'status': 'warning' if risk_pct > 15 else 'neutral'
            })
        
        # Revenue concentration
        top_segments = ['Champions', 'Loyal Customers']
        top_revenue = sum(segment_dist.get(s, {}).get('revenue_share', 0) for s in top_segments)
        
        insights.append({
            'title': 'Revenue Concentration',
            'description': f'Top segments (Champions + Loyal) contribute {top_revenue:.1f}% of total revenue.',
            'status': 'positive' if top_revenue >= 50 else 'neutral'
        })
        
    elif analysis_type == 'kmeans':
        # Silhouette interpretation
        sil_score = results.get('silhouette_score', 0)
        if sil_score:
            if sil_score >= 0.5:
                quality = 'Good'
                status = 'positive'
            elif sil_score >= 0.25:
                quality = 'Moderate'
                status = 'neutral'
            else:
                quality = 'Weak'
                status = 'warning'
            
            insights.append({
                'title': f'{quality} Segment Separation',
                'description': f'Silhouette score: {sil_score:.3f}. {"Clusters are well-defined." if sil_score >= 0.5 else "Some overlap between segments."}',
                'status': status
            })
        
        # Largest segment
        largest = max(segment_dist.items(), key=lambda x: x[1].get('count', 0))
        insights.append({
            'title': f'Largest Segment: {largest[0]}',
            'description': f'{largest[1].get("percentage", 0):.1f}% of customers ({largest[1].get("count", 0)} customers).',
            'status': 'neutral'
        })
        
        # ========== 추가: Revenue concentration for K-Means ==========
        # Check if revenue_share is available
        has_revenue = any(s.get('revenue_share') for s in segment_dist.values())
        if has_revenue:
            sorted_segments = sorted(segment_dist.items(), key=lambda x: x[1].get('revenue_share', 0), reverse=True)
            top_segment = sorted_segments[0]
            insights.append({
                'title': 'Top Revenue Segment',
                'description': f'{top_segment[0]} contributes {top_segment[1].get("revenue_share", 0):.1f}% of total revenue.',
                'status': 'positive' if top_segment[1].get('revenue_share', 0) >= 30 else 'neutral'
            })
        # ========== 추가 끝 ==========
        
    elif analysis_type == 'value_based':
        # Pareto check
        platinum_rev = segment_dist.get('Platinum', {}).get('revenue_share', 0)
        
        insights.append({
            'title': 'Pareto Principle Check',
            'description': f'Top 10% (Platinum) contribute {platinum_rev:.1f}% of revenue.',
            'status': 'positive' if platinum_rev >= 40 else 'neutral'
        })
    
    return insights


@router.post("/segmentation")
async def run_customer_segmentation(request: SegmentationRequest) -> Dict[str, Any]:
    """
    Perform Customer Segmentation analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        if request.customer_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Customer column '{request.customer_col}' not found")
        
        visualizations = {}
        results = {}
        
        if request.analysis_type == 'rfm':
            # Validate RFM requirements
            if not request.date_col or request.date_col not in df.columns:
                raise HTTPException(status_code=400, detail="Date column required for RFM analysis")
            if not request.value_col or request.value_col not in df.columns:
                raise HTTPException(status_code=400, detail="Value column required for RFM analysis")
            
            # Calculate RFM
            rfm = calculate_rfm(df, request.customer_col, request.date_col,
                               request.value_col, request.reference_date)
            rfm = assign_rfm_scores(rfm)
            rfm = assign_rfm_segments(rfm)
            
            # Create visualizations
            visualizations['rfm_distribution'] = create_rfm_distribution_chart(rfm)
            visualizations['segment_distribution'] = create_segment_distribution_chart(rfm)
            visualizations['rfm_heatmap'] = create_rfm_heatmap(rfm)
            visualizations['segment_profile'] = create_segment_profile_chart(rfm)
            
            # Segment statistics
            segment_stats = rfm.groupby('Segment').agg({
                request.customer_col: 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'sum'],
                'RFM_Total': 'mean'
            }).reset_index()
            segment_stats.columns = ['Segment', 'Count', 'Avg_Recency', 'Avg_Frequency', 
                                     'Avg_Monetary', 'Total_Monetary', 'Avg_RFM_Score']
            
            segment_distribution = {}
            total_customers = len(rfm)
            total_revenue = rfm['Monetary'].sum()
            
            for _, row in segment_stats.iterrows():
                segment_distribution[row['Segment']] = {
                    'count': int(row['Count']),
                    'percentage': _to_native_type(row['Count'] / total_customers * 100),
                    'avg_recency': _to_native_type(row['Avg_Recency']),
                    'avg_frequency': _to_native_type(row['Avg_Frequency']),
                    'avg_monetary': _to_native_type(row['Avg_Monetary']),
                    'total_monetary': _to_native_type(row['Total_Monetary']),
                    'revenue_share': _to_native_type(row['Total_Monetary'] / total_revenue * 100)
                }
            
            # Customer list
            customer_segments = rfm[[request.customer_col, 'Recency', 'Frequency', 'Monetary',
                                     'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Segment']].to_dict('records')
            
            results = {
                'segment_distribution': segment_distribution,
                'customer_segments': customer_segments,
                'metrics': {
                    'total_customers': total_customers,
                    'total_revenue': _to_native_type(total_revenue),
                    'avg_recency': _to_native_type(rfm['Recency'].mean()),
                    'avg_frequency': _to_native_type(rfm['Frequency'].mean()),
                    'avg_monetary': _to_native_type(rfm['Monetary'].mean())
                }
            }
            
        elif request.analysis_type == 'kmeans':
            # Determine feature columns
            if request.feature_cols:
                feature_cols = request.feature_cols
            elif request.value_col and request.date_col:
                # Use RFM as features
                rfm = calculate_rfm(df, request.customer_col, request.date_col,
                                   request.value_col, request.reference_date)
                feature_cols = ['Recency', 'Frequency', 'Monetary']
                df_features = rfm
            else:
                raise HTTPException(status_code=400, detail="Feature columns or value/date columns required")
            
            if 'df_features' not in locals():
                df_features = df.groupby(request.customer_col).first().reset_index()
            
            # Perform K-means
            kmeans_results = perform_kmeans_segmentation(
                df_features, feature_cols, request.n_segments, request.customer_col
            )
            
            # Create visualizations
            visualizations['cluster_scatter'] = create_cluster_scatter(
                kmeans_results['customer_segments'], feature_cols,
                kmeans_results['customer_segments']['Segment'].values
            )
            visualizations['segment_comparison'] = create_segment_comparison_chart(
                kmeans_results['cluster_stats'], feature_cols
            )
            
            # ========== 수정된 부분: segment_distribution에 monetary 정보 추가 ==========
            segment_distribution = {}
            for stats in kmeans_results['cluster_stats']:
                seg_data = {
                    'count': stats['size'],
                    'percentage': stats['percentage'],
                    'avg_silhouette': stats['avg_silhouette']
                }
                # Add monetary metrics if available
                if 'avg_monetary' in stats:
                    seg_data['avg_monetary'] = stats['avg_monetary']
                if 'total_monetary' in stats:
                    seg_data['total_monetary'] = stats['total_monetary']
                if 'revenue_share' in stats:
                    seg_data['revenue_share'] = stats['revenue_share']
                
                segment_distribution[stats['segment_name']] = seg_data
            # ========== 수정된 부분 끝 ==========
            
            # ========== 추가: metrics에 total_revenue 추가 ==========
            metrics = {
                'total_customers': len(kmeans_results['customer_segments']),
            }
            if kmeans_results.get('total_monetary'):
                metrics['total_revenue'] = kmeans_results['total_monetary']
            # ========== 추가 끝 ==========
            
            results = {
                'segment_distribution': segment_distribution,
                'cluster_stats': kmeans_results['cluster_stats'],
                'customer_segments': kmeans_results['customer_segments'].to_dict('records'),
                'silhouette_score': kmeans_results['silhouette_score'],
                'n_segments': kmeans_results['n_clusters'],
                'feature_cols': feature_cols,
                'metrics': metrics  # 추가
            }
            
        elif request.analysis_type == 'value_based':
            if not request.value_col or request.value_col not in df.columns:
                raise HTTPException(status_code=400, detail="Value column required for value-based segmentation")
            
            # Calculate value segments
            value_df = calculate_value_segments(df, request.customer_col, request.value_col)
            
            # Create visualizations
            visualizations['value_segments'] = create_value_segment_chart(value_df)
            
            # Segment distribution
            segment_distribution = {}
            total_customers = len(value_df)
            total_revenue = value_df['Total_Value'].sum()
            
            for segment in ['Platinum', 'Gold', 'Silver', 'Bronze', 'Standard']:
                seg_data = value_df[value_df['Segment'] == segment]
                if len(seg_data) > 0:
                    segment_distribution[segment] = {
                        'count': len(seg_data),
                        'percentage': _to_native_type(len(seg_data) / total_customers * 100),
                        'total_value': _to_native_type(seg_data['Total_Value'].sum()),
                        'avg_value': _to_native_type(seg_data['Total_Value'].mean()),
                        'revenue_share': _to_native_type(seg_data['Total_Value'].sum() / total_revenue * 100)
                    }
            
            results = {
                'segment_distribution': segment_distribution,
                'customer_segments': value_df.to_dict('records'),
                'metrics': {
                    'total_customers': total_customers,
                    'total_revenue': _to_native_type(total_revenue),
                    'avg_customer_value': _to_native_type(value_df['Total_Value'].mean())
                }
            }
        
        # Generate insights
        insights = generate_key_insights(results, request.analysis_type)
        
        # Summary
        summary = {
            'analysis_type': request.analysis_type,
            'total_customers': len(df[request.customer_col].unique()),
            'n_segments': len(results.get('segment_distribution', {})),
            'total_transactions': len(df)
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation analysis failed: {str(e)}")
