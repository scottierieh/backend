"""
RFM Segmentation FastAPI Endpoint
Customer segmentation based on Recency, Frequency, Monetary value
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


class RFMRequest(BaseModel):
    """Request model for RFM Segmentation"""
    data: List[Dict[str, Any]]
    customer_id_col: str
    recency_col: str
    frequency_col: str
    monetary_col: str


class KeyInsight(BaseModel):
    """Key insight"""
    title: str
    description: str
    status: str


class RFMResponse(BaseModel):
    """Response model for RFM Segmentation"""
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


def assign_segment(r, f, m):
    """Assign customer segment based on RFM scores"""
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif f >= 4 and m >= 4:
        return 'Loyal Customers'
    elif r >= 4 and f >= 3 and m >= 3:
        return 'Potential Loyalists'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r <= 2 and f >= 4 and m >= 4:
        return "Can't Lose Them"
    elif r <= 2 and f >= 2 and m >= 2:
        return 'Hibernating'
    elif r <= 2 and f <= 2:
        return 'Lost'
    elif r >= 3 and f <= 2 and m <= 2:
        return 'Promising'
    else:
        return 'Need Attention'


@router.post("/rfm-segmentation")
async def segment_customers(request: RFMRequest):
    """
    RFM Customer Segmentation
    
    Segment customers based on Recency, Frequency, and Monetary value
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 10:
            raise HTTPException(400, "Insufficient data (need at least 10 customers)")
        
        df = pd.DataFrame(request.data)
        
        # Ensure numeric
        df[request.recency_col] = pd.to_numeric(df[request.recency_col], errors='coerce')
        df[request.frequency_col] = pd.to_numeric(df[request.frequency_col], errors='coerce')
        df[request.monetary_col] = pd.to_numeric(df[request.monetary_col], errors='coerce')
        df = df.dropna(subset=[request.recency_col, request.frequency_col, request.monetary_col])
        
        # Calculate RFM scores (1-5)
        df['R_Score'] = pd.qcut(df[request.recency_col], q=5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
        df['F_Score'] = pd.qcut(df[request.frequency_col], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
        df['M_Score'] = pd.qcut(df[request.monetary_col], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
        
        df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
        df['Total_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']
        
        # Assign segments
        df['Segment'] = df.apply(lambda row: assign_segment(row['R_Score'], row['F_Score'], row['M_Score']), axis=1)
        
        # Segment stats
        segment_stats = df.groupby('Segment').agg({
            request.customer_id_col: 'count',
            request.recency_col: 'mean',
            request.frequency_col: 'mean',
            request.monetary_col: ['mean', 'sum'],
            'Total_Score': 'mean'
        }).reset_index()
        
        segment_stats.columns = ['segment', 'customer_count', 'avg_recency', 'avg_frequency', 
                                 'avg_monetary', 'total_revenue', 'avg_score']
        
        total_customers = segment_stats['customer_count'].sum()
        total_revenue = segment_stats['total_revenue'].sum()
        segment_stats['customer_pct'] = (segment_stats['customer_count'] / total_customers) * 100
        segment_stats['revenue_pct'] = (segment_stats['total_revenue'] / total_revenue) * 100
        segment_stats = segment_stats.sort_values('total_revenue', ascending=False)
        
        # Metrics
        champions = len(df[df['Segment'] == 'Champions'])
        loyal = len(df[df['Segment'] == 'Loyal Customers'])
        at_risk = len(df[df['Segment'] == 'At Risk'])
        lost = len(df[df['Segment'] == 'Lost'])
        
        top_segment = segment_stats.iloc[0]['segment']
        top_segment_revenue = segment_stats.iloc[0]['total_revenue']
        top_segment_pct = segment_stats.iloc[0]['revenue_pct']
        
        metrics = {
            'total_customers': int(len(df)),
            'total_revenue': float(df[request.monetary_col].sum()),
            'avg_recency_days': float(df[request.recency_col].mean()),
            'avg_frequency': float(df[request.frequency_col].mean()),
            'avg_monetary': float(df[request.monetary_col].mean()),
            'champions_count': int(champions),
            'loyal_count': int(loyal),
            'at_risk_count': int(at_risk),
            'lost_count': int(lost),
            'top_segment': str(top_segment),
            'top_segment_revenue': float(top_segment_revenue),
            'top_segment_revenue_pct': float(top_segment_pct)
        }
        
        # Visualizations
        visualizations = {}
        
        # 1. Segment Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.pie(segment_stats['customer_count'], labels=segment_stats['segment'], autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette('Blues_r', len(segment_stats)))
        ax1.set_title('Customer Distribution by Segment', fontsize=12, fontweight='bold')
        
        ax2.barh(segment_stats['segment'], segment_stats['customer_count'], 
                color='#2C3E50', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
        ax2.set_title('Customer Count by Segment', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(segment_stats['customer_count']):
            ax2.text(v, i, f' {int(v)}', va='center', fontweight='bold')
        plt.tight_layout()
        visualizations['segment_distribution'] = fig_to_base64(fig)
        
        # 2. Revenue by Segment
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(segment_stats))
        ax.barh(y_pos, segment_stats['total_revenue'], color='#3498DB', edgecolor='black', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(segment_stats['segment'])
        ax.invert_yaxis()
        ax.set_xlabel('Total Revenue ($)', fontsize=11, fontweight='bold')
        ax.set_title('Total Revenue by Customer Segment', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(segment_stats['total_revenue']):
            ax.text(v, i, f' ${v:,.0f} ({segment_stats.iloc[i]["revenue_pct"]:.1f}%)', 
                   va='center', fontweight='bold', fontsize=9)
        plt.tight_layout()
        visualizations['revenue_by_segment'] = fig_to_base64(fig)
        
        # 3. RFM Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        pivot = df.groupby(['R_Score', 'F_Score']).size().unstack(fill_value=0)
        sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Customer Count'},
                   ax=ax, linewidths=0.5, linecolor='gray')
        ax.set_xlabel('Frequency Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Recency Score', fontsize=11, fontweight='bold')
        ax.set_title('RFM Heatmap: Recency vs Frequency', fontsize=13, fontweight='bold')
        plt.tight_layout()
        visualizations['rfm_heatmap'] = fig_to_base64(fig)
        
        # 4. RFM Distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(df['R_Score'], bins=5, color='#2C3E50', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Recency Score', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Customer Count', fontsize=10, fontweight='bold')
        axes[0].set_title('Recency Distribution', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].hist(df['F_Score'], bins=5, color='#3498DB', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Frequency Score', fontsize=10, fontweight='bold')
        axes[1].set_ylabel('Customer Count', fontsize=10, fontweight='bold')
        axes[1].set_title('Frequency Distribution', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].hist(df['M_Score'], bins=5, color='#95A5A6', edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Monetary Score', fontsize=10, fontweight='bold')
        axes[2].set_ylabel('Customer Count', fontsize=10, fontweight='bold')
        axes[2].set_title('Monetary Distribution', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        visualizations['rfm_distribution'] = fig_to_base64(fig)
        
        # 5. Segment Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].barh(segment_stats['segment'], segment_stats['avg_recency'],
                       color='#2C3E50', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Avg Days Since Last Purchase', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('Average Recency by Segment', fontsize=11, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        axes[0, 1].barh(segment_stats['segment'], segment_stats['avg_frequency'],
                       color='#3498DB', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Avg Number of Purchases', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('Average Frequency by Segment', fontsize=11, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        axes[1, 0].barh(segment_stats['segment'], segment_stats['avg_monetary'],
                       color='#95A5A6', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Avg Spending ($)', fontsize=10, fontweight='bold')
        axes[1, 0].set_title('Average Monetary Value by Segment', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        axes[1, 1].barh(segment_stats['segment'], segment_stats['avg_score'],
                       color='#E74C3C', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Avg RFM Score', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('Average Total Score by Segment', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        visualizations['segment_comparison'] = fig_to_base64(fig)
        
        # Insights
        insights = []
        insights.append({
            'title': f'Top Revenue Segment: {top_segment}',
            'description': f"{top_segment} generates ${top_segment_revenue:,.0f} ({top_segment_pct:.1f}% of total revenue). Focus retention efforts here.",
            'status': 'positive'
        })
        
        if champions > 0:
            champions_pct = (champions / len(df)) * 100
            insights.append({
                'title': f'Champions: {champions} customers',
                'description': f"{champions_pct:.1f}% of customers are Champions (high R/F/M). Reward loyalty with VIP programs and exclusive offers.",
                'status': 'positive'
            })
        
        if at_risk > 0:
            at_risk_pct = (at_risk / len(df)) * 100
            insights.append({
                'title': f'At Risk: {at_risk} customers',
                'description': f"{at_risk_pct:.1f}% of customers are At Risk. Launch win-back campaigns with personalized offers before they're lost.",
                'status': 'warning'
            })
        
        return RFMResponse(
            success=True,
            results={
                'metrics': metrics,
                'segment_summary': segment_stats.to_dict('records'),
                'customer_segments': df[[
                    request.customer_id_col, request.recency_col, request.frequency_col, 
                    request.monetary_col, 'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 
                    'Total_Score', 'Segment'
                ]].sort_values('Total_Score', ascending=False).to_dict('records'),
                'rfm_statistics': {
                    'recency': {
                        'mean': float(df[request.recency_col].mean()),
                        'median': float(df[request.recency_col].median()),
                        'min': float(df[request.recency_col].min()),
                        'max': float(df[request.recency_col].max())
                    },
                    'frequency': {
                        'mean': float(df[request.frequency_col].mean()),
                        'median': float(df[request.frequency_col].median()),
                        'min': float(df[request.frequency_col].min()),
                        'max': float(df[request.frequency_col].max())
                    },
                    'monetary': {
                        'mean': float(df[request.monetary_col].mean()),
                        'median': float(df[request.monetary_col].median()),
                        'min': float(df[request.monetary_col].min()),
                        'max': float(df[request.monetary_col].max())
                    }
                }
            },
            visualizations=visualizations,
            key_insights=insights,
            summary={
                'analysis_type': 'rfm_segmentation',
                'total_customers': metrics['total_customers'],
                'total_segments': len(segment_stats),
                'top_segment': metrics['top_segment']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation error: {str(e)}")
