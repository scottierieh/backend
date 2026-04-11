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
        for j, col in enumerate(feature_cols):
            stats[f'{col}_mean'] = _to_native_type(cluster_data.iloc[:, j].mean())
            stats[f'{col}_std'] = _to_native_type(cluster_data.iloc[:, j].std())
        cluster_stats.append(stats)
    
    return {
        'customer_segments': result_df,
        'cluster_stats': cluster_stats,
        'silhouette_score': _to_native_type(sil_score),
        'n_clusters': n_clusters,
        'feature_cols': feature_cols,
        'inertia': _to_native_type(kmeans.inertia_)
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
    customer_value = df.groupby(customer_col).agg({
        value_col: ['sum', 'mean', 'count']
    }).reset_index()
    customer_value.columns = [customer_col, 'Total_Value', 'Avg_Value', 'Transaction_Count']
    
    # Calculate percentiles
    customer_value['Value_Percentile'] = customer_value['Total_Value'].rank(pct=True) * 100
    
    # Assign segments
    def assign_value_segment(percentile):
        if percentile >= 90:
            return 'Platinum'
        elif percentile >= 75:
            return 'Gold'
        elif percentile >= 50:
            return 'Silver'
        elif percentile >= 25:
            return 'Bronze'
        else:
            return 'Standard'
    
    customer_value['Segment'] = customer_value['Value_Percentile'].apply(assign_value_segment)
    
    return customer_value


def create_rfm_distribution_chart(rfm: pd.DataFrame) -> str:
    """Create RFM score distribution charts"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['#3b82f6', '#22c55e', '#f59e0b']
    titles = ['Recency Distribution', 'Frequency Distribution', 'Monetary Distribution']
    cols = ['Recency', 'Frequency', 'Monetary']
    
    for ax, col, color, title in zip(axes, cols, colors, titles):
        ax.hist(rfm[col], bins=30, color=color, edgecolor='white', alpha=0.7)
        ax.axvline(rfm[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rfm[col].mean():.1f}')
        ax.axvline(rfm[col].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {rfm[col].median():.1f}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_distribution_chart(rfm: pd.DataFrame) -> str:
    """Create segment distribution pie and bar charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    segment_counts = rfm['Segment'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(segment_counts, labels=segment_counts.index,
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Segment Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.barh(segment_counts.index, segment_counts.values, color=colors)
    ax2.set_xlabel('Number of Customers')
    ax2.set_title('Customers by Segment', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, segment_counts.values):
        ax2.text(bar.get_width() + max(segment_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_rfm_heatmap(rfm: pd.DataFrame) -> str:
    """Create RFM segment heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pivot table
    heatmap_data = rfm.groupby(['R_Score', 'F_Score']).size().unstack(fill_value=0)
    
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
                cbar_kws={'label': 'Customer Count'})
    ax.set_xlabel('Frequency Score', fontsize=12)
    ax.set_ylabel('Recency Score', fontsize=12)
    ax.set_title('RFM Segment Heatmap (R vs F)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_profile_chart(rfm: pd.DataFrame) -> str:
    """Create segment profile radar/bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate segment averages
    segment_profile = rfm.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'RFM_Total': 'mean'
    }).reset_index()
    
    # Normalize for comparison
    segment_profile['R_norm'] = (segment_profile['Recency'].max() - segment_profile['Recency']) / segment_profile['Recency'].max() * 100
    segment_profile['F_norm'] = segment_profile['Frequency'] / segment_profile['Frequency'].max() * 100
    segment_profile['M_norm'] = segment_profile['Monetary'] / segment_profile['Monetary'].max() * 100
    
    x = np.arange(len(segment_profile))
    width = 0.25
    
    bars1 = ax.bar(x - width, segment_profile['R_norm'], width, label='Recency (inverted)', color='#3b82f6')
    bars2 = ax.bar(x, segment_profile['F_norm'], width, label='Frequency', color='#22c55e')
    bars3 = ax.bar(x + width, segment_profile['M_norm'], width, label='Monetary', color='#f59e0b')
    
    ax.set_ylabel('Normalized Score (%)')
    ax.set_title('Segment Profile Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(segment_profile['Segment'], rotation=45, ha='right')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cluster_scatter(df: pd.DataFrame, feature_cols: List[str], 
                           labels: np.ndarray) -> str:
    """Create cluster scatter plot"""
    n_clusters = len(np.unique(labels))
    
    if len(feature_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = labels == i
            ax.scatter(df[feature_cols[0]][mask], df[feature_cols[1]][mask],
                      c=[colors[i]], alpha=0.6, s=50, label=f'Segment {i+1}')
        
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_title('Customer Segments', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = labels == i
            ax.hist(df[feature_cols[0]][mask], bins=20, alpha=0.6, 
                   color=colors[i], label=f'Segment {i+1}')
        
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel('Count')
        ax.set_title('Customer Segments Distribution', fontsize=14, fontweight='bold')
        ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_comparison_chart(cluster_stats: List[Dict], feature_cols: List[str]) -> str:
    """Create segment comparison chart"""
    n_segments = len(cluster_stats)
    n_features = min(len(feature_cols), 4)
    
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 5))
    if n_features == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_segments))
    
    for i, (ax, col) in enumerate(zip(axes, feature_cols[:n_features])):
        means = [s[f'{col}_mean'] for s in cluster_stats]
        segment_names = [s['segment_name'] for s in cluster_stats]
        
        bars = ax.bar(segment_names, means, color=colors)
        ax.set_ylabel(col)
        ax.set_title(f'{col} by Segment', fontsize=11, fontweight='bold')
        ax.set_xticklabels(segment_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_value_segment_chart(value_df: pd.DataFrame) -> str:
    """Create value-based segment chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    segment_order = ['Platinum', 'Gold', 'Silver', 'Bronze', 'Standard']
    segment_colors = {'Platinum': '#1e3a5f', 'Gold': '#ffd700', 'Silver': '#c0c0c0', 
                      'Bronze': '#cd7f32', 'Standard': '#808080'}
    
    # Count by segment
    segment_counts = value_df['Segment'].value_counts().reindex(segment_order).fillna(0)
    colors = [segment_colors[s] for s in segment_counts.index]
    
    ax1.bar(segment_counts.index, segment_counts.values, color=colors, edgecolor='white')
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Customers by Value Segment', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Revenue by segment
    segment_revenue = value_df.groupby('Segment')['Total_Value'].sum().reindex(segment_order).fillna(0)
    colors = [segment_colors[s] for s in segment_revenue.index]
    
    ax2.bar(segment_revenue.index, segment_revenue.values, color=colors, edgecolor='white')
    ax2.set_ylabel('Total Revenue')
    ax2.set_title('Revenue by Value Segment', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Format y-axis
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K' if x >= 1000 else f'${x:.0f}'))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(results: Dict, analysis_type: str) -> List[Dict[str, Any]]:
    """Generate key insights based on analysis type"""
    insights = []
    
    if analysis_type == 'rfm':
        segment_dist = results.get('segment_distribution', {})
        metrics = results.get('metrics', {})
        
        # Champion customers
        champions_pct = segment_dist.get('Champions', {}).get('percentage', 0)
        if champions_pct >= 10:
            insights.append({
                'title': 'Strong Champion Base',
                'description': f'{champions_pct:.1f}% of customers are Champions. Focus on retention and referral programs.',
                'status': 'positive'
            })
        elif champions_pct >= 5:
            insights.append({
                'title': 'Growing Champion Segment',
                'description': f'{champions_pct:.1f}% Champions. Opportunity to grow through loyalty programs.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Low Champion Rate',
                'description': f'Only {champions_pct:.1f}% Champions. Focus on converting Potential Loyalists.',
                'status': 'warning'
            })
        
        # At Risk customers
        at_risk_pct = segment_dist.get('At Risk', {}).get('percentage', 0) + segment_dist.get("Can't Lose Them", {}).get('percentage', 0)
        if at_risk_pct > 15:
            insights.append({
                'title': 'High At-Risk Population',
                'description': f'{at_risk_pct:.1f}% customers at risk of churning. Urgent re-engagement needed.',
                'status': 'warning'
            })
        elif at_risk_pct > 5:
            insights.append({
                'title': 'Moderate Churn Risk',
                'description': f'{at_risk_pct:.1f}% at risk. Implement win-back campaigns.',
                'status': 'neutral'
            })
        
        # New customers
        new_pct = segment_dist.get('New Customers', {}).get('percentage', 0)
        if new_pct >= 15:
            insights.append({
                'title': 'Strong New Customer Acquisition',
                'description': f'{new_pct:.1f}% are new customers. Focus on onboarding to build loyalty.',
                'status': 'positive'
            })
        
    elif analysis_type == 'kmeans':
        cluster_stats = results.get('cluster_stats', [])
        sil_score = results.get('silhouette_score', 0)
        
        if sil_score >= 0.5:
            insights.append({
                'title': 'Well-Defined Segments',
                'description': f'Silhouette score: {sil_score:.3f}. Clear customer segment boundaries.',
                'status': 'positive'
            })
        elif sil_score >= 0.25:
            insights.append({
                'title': 'Moderate Segment Separation',
                'description': f'Silhouette score: {sil_score:.3f}. Some overlap between segments.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Overlapping Segments',
                'description': f'Silhouette score: {sil_score:.3f}. Consider fewer segments.',
                'status': 'warning'
            })
        
        # Largest segment
        if cluster_stats:
            largest = max(cluster_stats, key=lambda x: x['size'])
            insights.append({
                'title': f'Largest Segment: {largest["segment_name"]}',
                'description': f'{largest["percentage"]:.1f}% of customers ({largest["size"]:,} customers).',
                'status': 'neutral'
            })
    
    elif analysis_type == 'value_based':
        segment_dist = results.get('segment_distribution', {})
        
        platinum_pct = segment_dist.get('Platinum', {}).get('percentage', 0)
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
            
            # Segment distribution
            segment_distribution = {}
            for stats in kmeans_results['cluster_stats']:
                segment_distribution[stats['segment_name']] = {
                    'count': stats['size'],
                    'percentage': stats['percentage'],
                    'avg_silhouette': stats['avg_silhouette']
                }
            
            results = {
                'segment_distribution': segment_distribution,
                'cluster_stats': kmeans_results['cluster_stats'],
                'customer_segments': kmeans_results['customer_segments'].to_dict('records'),
                'silhouette_score': kmeans_results['silhouette_score'],
                'n_segments': kmeans_results['n_clusters'],
                'feature_cols': feature_cols
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
