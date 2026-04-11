"""
Customer Lifetime Value (CLV) Analysis Router for FastAPI
Calculate CLV, RFM segmentation, cohort analysis
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
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CLVRequest(BaseModel):
    data: List[Dict[str, Any]]
    customer_id_col: str
    order_date_col: str
    order_value_col: str
    order_id_col: Optional[str] = None
    clv_model: str = "historical"  # historical, simple, cohort
    projection_months: int = 12


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
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    return obj


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_rfm_scores(customer_df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    """Calculate RFM scores for each customer"""
    # Recency: days since last purchase
    customer_df['recency'] = (analysis_date - customer_df['last_purchase']).dt.days
    
    # Score R, F, M (1-5 quintiles)
    customer_df['R'] = pd.qcut(customer_df['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    customer_df['F'] = pd.qcut(customer_df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    customer_df['M'] = pd.qcut(customer_df['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    customer_df['R'] = customer_df['R'].astype(int)
    customer_df['F'] = customer_df['F'].astype(int)
    customer_df['M'] = customer_df['M'].astype(int)
    customer_df['RFM_Score'] = customer_df['R'] * 100 + customer_df['F'] * 10 + customer_df['M']
    
    return customer_df


def assign_rfm_segment(row) -> str:
    """Assign RFM segment based on scores"""
    r, f, m = row['R'], row['F'], row['M']
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal'
    elif r >= 4 and f <= 2:
        return 'New'
    elif r >= 3 and f >= 2:
        return 'Potential'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Hibernating'


def calculate_cohort_clv(df: pd.DataFrame, customer_id_col: str, order_date_col: str, order_value_col: str) -> List[Dict]:
    """Calculate CLV by acquisition cohort"""
    df['order_month'] = df[order_date_col].dt.to_period('M')
    
    # Find first purchase month for each customer
    first_purchase = df.groupby(customer_id_col)[order_date_col].min().reset_index()
    first_purchase.columns = [customer_id_col, 'first_purchase']
    first_purchase['cohort'] = first_purchase['first_purchase'].dt.to_period('M')
    
    df = df.merge(first_purchase[[customer_id_col, 'cohort']], on=customer_id_col)
    df['months_since_first'] = (df['order_month'] - df['cohort']).apply(lambda x: x.n if hasattr(x, 'n') else 0)
    
    # Calculate cumulative CLV by cohort
    cohort_data = []
    for cohort in df['cohort'].unique():
        cohort_df = df[df['cohort'] == cohort]
        customers = cohort_df[customer_id_col].nunique()
        
        m0 = cohort_df[cohort_df['months_since_first'] == 0][order_value_col].sum() / customers if customers > 0 else 0
        m3 = cohort_df[cohort_df['months_since_first'] <= 3][order_value_col].sum() / customers if customers > 0 else 0
        m6 = cohort_df[cohort_df['months_since_first'] <= 6][order_value_col].sum() / customers if customers > 0 else 0
        m12 = cohort_df[cohort_df['months_since_first'] <= 12][order_value_col].sum() / customers if customers > 0 else 0
        
        # Project CLV (simple linear projection)
        total_clv = cohort_df.groupby(customer_id_col)[order_value_col].sum().mean()
        projected = total_clv * 1.2 if m12 > 0 else m6 * 2
        
        cohort_data.append({
            'cohort': str(cohort),
            'customers': int(customers),
            'month_0': float(m0),
            'month_3': float(m3),
            'month_6': float(m6),
            'month_12': float(m12),
            'projected_clv': float(projected),
        })
    
    return sorted(cohort_data, key=lambda x: x['cohort'], reverse=True)[:12]


# ============ VISUALIZATION ============
def create_clv_distribution_chart(clv_distribution: List[Dict]) -> str:
    """Create CLV distribution histogram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    buckets = [d['bucket'] for d in clv_distribution]
    counts = [d['count'] for d in clv_distribution]
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(buckets)))
    bars = ax.bar(buckets, counts, color=colors, edgecolor='white')
    
    ax.set_xlabel('CLV Range', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_title('Customer Lifetime Value Distribution', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_comparison_chart(segments: List[Dict]) -> str:
    """Create segment comparison bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    names = [s['segment'] for s in segments]
    customers = [s['customer_count'] for s in segments]
    avg_clvs = [s['avg_clv'] for s in segments]
    
    colors = {
        'Champions': '#22c55e', 'Loyal': '#3b82f6', 'Potential': '#8b5cf6',
        'New': '#06b6d4', 'At Risk': '#f59e0b', 'Hibernating': '#ef4444'
    }
    bar_colors = [colors.get(n, '#64748b') for n in names]
    
    # Customer count
    ax1 = axes[0]
    ax1.barh(names, customers, color=bar_colors, alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Number of Customers', fontsize=11)
    ax1.set_title('Customers by Segment', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Average CLV
    ax2 = axes[1]
    ax2.barh(names, avg_clvs, color=bar_colors, alpha=0.8, edgecolor='white')
    ax2.set_xlabel('Average CLV ($)', fontsize=11)
    ax2.set_title('Average CLV by Segment', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_pareto_chart(clv_distribution: List[Dict]) -> str:
    """Create Pareto chart showing revenue concentration"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    buckets = [d['bucket'] for d in clv_distribution]
    pcts = [d['pct'] for d in clv_distribution]
    cum_rev = [d['cumulative_revenue_pct'] for d in clv_distribution]
    
    ax1.bar(buckets, pcts, color='#3b82f6', alpha=0.7, edgecolor='white', label='% Customers')
    ax1.set_xlabel('CLV Range', fontsize=11)
    ax1.set_ylabel('% of Customers', fontsize=11, color='#3b82f6')
    
    ax2 = ax1.twinx()
    ax2.plot(buckets, cum_rev, color='#ef4444', linewidth=2, marker='o', label='Cumulative Revenue %')
    ax2.set_ylabel('Cumulative Revenue %', fontsize=11, color='#ef4444')
    ax2.axhline(y=80, color='#22c55e', linestyle='--', alpha=0.7, label='80% line')
    
    ax1.set_title('Pareto Analysis: Customer Value Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cohort_heatmap(cohort_clv: List[Dict]) -> str:
    """Create cohort CLV heatmap"""
    if not cohort_clv:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cohorts = [c['cohort'] for c in cohort_clv[:8]]
    data = [[c['month_0'], c['month_3'], c['month_6'], c['month_12']] for c in cohort_clv[:8]]
    
    im = ax.imshow(data, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Month 0', 'Month 3', 'Month 6', 'Month 12'])
    ax.set_yticks(range(len(cohorts)))
    ax.set_yticklabels(cohorts)
    
    # Add value annotations
    for i in range(len(cohorts)):
        for j in range(4):
            text = ax.text(j, i, f'${data[i][j]:.0f}', ha='center', va='center', fontsize=9)
    
    ax.set_title('Cohort CLV Progression', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='CLV ($)')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(summary: Dict, segments: List[Dict]) -> List[Dict]:
    """Generate key CLV insights"""
    insights = []
    
    # Pareto insight
    top_20_rev = summary['top_20_pct_revenue']
    if top_20_rev >= 80:
        insights.append({
            'title': f"Strong Pareto Effect ({top_20_rev:.0f}% from top 20%)",
            'description': "Focus retention efforts on top customers for maximum impact.",
            'status': 'positive'
        })
    elif top_20_rev >= 60:
        insights.append({
            'title': f"Moderate Concentration ({top_20_rev:.0f}% from top 20%)",
            'description': "Revenue is moderately concentrated among top customers.",
            'status': 'neutral'
        })
    
    # CLV insight
    avg_clv = summary['avg_clv']
    aov = summary['avg_order_value']
    orders = summary['avg_orders_per_customer']
    
    insights.append({
        'title': f"Average CLV: ${avg_clv:,.0f}",
        'description': f"Based on {orders:.1f} orders avg at ${aov:.0f} AOV.",
        'status': 'neutral'
    })
    
    # Retention insight
    retention = summary['retention_rate']
    if retention < 20:
        insights.append({
            'title': f"Low Retention ({retention:.1f}%)",
            'description': "Most customers make only 1-2 purchases. Focus on repeat purchase programs.",
            'status': 'warning'
        })
    elif retention >= 40:
        insights.append({
            'title': f"Strong Retention ({retention:.1f}%)",
            'description': "Good repeat purchase rate. Continue current engagement strategies.",
            'status': 'positive'
        })
    
    # Segment insight
    champions = next((s for s in segments if s['segment'] == 'Champions'), None)
    at_risk = next((s for s in segments if s['segment'] == 'At Risk'), None)
    
    if champions:
        insights.append({
            'title': f"{champions['customer_count']} Champion Customers",
            'description': f"Avg CLV: ${champions['avg_clv']:,.0f}. These are your most valuable customers.",
            'status': 'positive'
        })
    
    if at_risk and at_risk['customer_count'] > 0:
        insights.append({
            'title': f"{at_risk['customer_count']} At-Risk Customers",
            'description': "Previously active customers who haven't purchased recently. Consider win-back campaigns.",
            'status': 'warning'
        })
    
    return insights


@router.post("/clv")
async def run_clv_analysis(request: CLVRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        for col in [request.customer_id_col, request.order_date_col, request.order_value_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Convert columns
        df[request.order_date_col] = pd.to_datetime(df[request.order_date_col], errors='coerce')
        df[request.order_value_col] = pd.to_numeric(df[request.order_value_col], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=[request.customer_id_col, request.order_date_col, request.order_value_col])
        
        # Analysis date
        analysis_date = df[request.order_date_col].max()
        min_date = df[request.order_date_col].min()
        
        # Customer-level aggregation
        customer_df = df.groupby(request.customer_id_col).agg({
            request.order_value_col: ['sum', 'mean', 'count'],
            request.order_date_col: ['min', 'max']
        }).reset_index()
        
        customer_df.columns = [request.customer_id_col, 'monetary', 'avg_order_value', 'frequency', 'first_purchase', 'last_purchase']
        customer_df['clv'] = customer_df['monetary']  # Historical CLV
        customer_df['lifespan_days'] = (customer_df['last_purchase'] - customer_df['first_purchase']).dt.days
        
        # RFM Analysis
        customer_df = calculate_rfm_scores(customer_df, analysis_date)
        customer_df['segment'] = customer_df.apply(assign_rfm_segment, axis=1)
        
        # Summary stats
        total_customers = len(customer_df)
        total_revenue = customer_df['monetary'].sum()
        avg_clv = customer_df['clv'].mean()
        median_clv = customer_df['clv'].median()
        avg_orders = customer_df['frequency'].mean()
        avg_aov = customer_df['avg_order_value'].mean()
        avg_lifespan = customer_df['lifespan_days'].mean()
        
        # Retention: customers with >1 order
        retention_rate = (customer_df['frequency'] > 1).sum() / total_customers * 100 if total_customers > 0 else 0
        
        # Pareto: top 20% revenue contribution
        customer_df_sorted = customer_df.sort_values('clv', ascending=False)
        top_20_count = int(total_customers * 0.2)
        top_20_revenue = customer_df_sorted.head(top_20_count)['monetary'].sum()
        top_20_pct = (top_20_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        # CLV Distribution
        clv_bins = [0, 50, 100, 200, 500, 1000, 2000, float('inf')]
        clv_labels = ['$0-50', '$50-100', '$100-200', '$200-500', '$500-1K', '$1K-2K', '$2K+']
        customer_df['clv_bucket'] = pd.cut(customer_df['clv'], bins=clv_bins, labels=clv_labels)
        
        clv_dist = customer_df.groupby('clv_bucket', observed=True).agg({
            request.customer_id_col: 'count',
            'monetary': 'sum'
        }).reset_index()
        clv_dist.columns = ['bucket', 'count', 'revenue']
        
        total_dist_revenue = clv_dist['revenue'].sum()
        clv_dist['pct'] = clv_dist['count'] / total_customers * 100
        clv_dist['revenue_pct'] = clv_dist['revenue'] / total_dist_revenue * 100 if total_dist_revenue > 0 else 0
        
        # Cumulative revenue (sorted by CLV descending)
        clv_dist_sorted = clv_dist.sort_values('bucket', ascending=False)
        clv_dist_sorted['cumulative_revenue_pct'] = clv_dist_sorted['revenue_pct'].cumsum()
        clv_dist = clv_dist_sorted.sort_values('bucket')
        
        clv_distribution = [
            {
                'bucket': str(row['bucket']),
                'count': int(row['count']),
                'pct': float(row['pct']),
                'cumulative_revenue_pct': float(row['cumulative_revenue_pct']),
            }
            for _, row in clv_dist.iterrows()
        ]
        
        # Customer segments
        segment_stats = customer_df.groupby('segment').agg({
            request.customer_id_col: 'count',
            'monetary': 'sum',
            'clv': 'mean',
            'frequency': 'mean',
            'avg_order_value': 'mean',
        }).reset_index()
        
        customer_segments = []
        for _, row in segment_stats.iterrows():
            seg_retention = (customer_df[customer_df['segment'] == row['segment']]['frequency'] > 1).sum()
            seg_total = row[request.customer_id_col]
            customer_segments.append({
                'segment': row['segment'],
                'customer_count': int(row[request.customer_id_col]),
                'pct_of_customers': float(row[request.customer_id_col] / total_customers * 100),
                'total_revenue': float(row['monetary']),
                'avg_clv': float(row['clv']),
                'avg_orders': float(row['frequency']),
                'avg_aov': float(row['avg_order_value']),
                'retention_rate': float(seg_retention / seg_total * 100) if seg_total > 0 else 0,
            })
        
        customer_segments.sort(key=lambda x: x['avg_clv'], reverse=True)
        
        # RFM Segments with actions
        rfm_segments = []
        segment_actions = {
            'Champions': 'Reward with exclusive offers and VIP treatment',
            'Loyal': 'Upsell higher value products, referral programs',
            'Potential': 'Nurture with personalized recommendations',
            'New': 'Onboard with welcome series and education',
            'At Risk': 'Launch win-back campaigns immediately',
            'Hibernating': 'Attempt reactivation with deep discounts',
        }
        segment_descriptions = {
            'Champions': 'Best customers: recent, frequent, high spenders',
            'Loyal': 'Regular customers with good spending',
            'Potential': 'Recent customers with growth potential',
            'New': 'Recently acquired, low frequency so far',
            'At Risk': 'Were good customers, declining engagement',
            'Hibernating': 'Inactive for extended period',
        }
        
        for seg in customer_segments:
            rfm_segments.append({
                'segment': seg['segment'],
                'description': segment_descriptions.get(seg['segment'], ''),
                'count': seg['customer_count'],
                'avg_clv': seg['avg_clv'],
                'action': segment_actions.get(seg['segment'], 'Monitor and engage'),
            })
        
        # Top customers
        top_customers = []
        for _, row in customer_df.nlargest(20, 'clv').iterrows():
            top_customers.append({
                'customer_id': str(row[request.customer_id_col]),
                'clv': float(row['clv']),
                'orders': int(row['frequency']),
                'first_purchase': row['first_purchase'].strftime('%Y-%m-%d'),
                'last_purchase': row['last_purchase'].strftime('%Y-%m-%d'),
                'segment': row['segment'],
            })
        
        # Cohort CLV
        cohort_clv = calculate_cohort_clv(df, request.customer_id_col, request.order_date_col, request.order_value_col)
        
        # Summary
        summary_data = {
            'total_customers': int(total_customers),
            'total_revenue': float(total_revenue),
            'avg_clv': float(avg_clv),
            'median_clv': float(median_clv),
            'avg_orders_per_customer': float(avg_orders),
            'avg_order_value': float(avg_aov),
            'avg_customer_lifespan': float(avg_lifespan),
            'retention_rate': float(retention_rate),
            'top_20_pct_revenue': float(top_20_pct),
        }
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {
            'clv_distribution': create_clv_distribution_chart(clv_distribution),
            'segment_comparison': create_segment_comparison_chart(customer_segments),
            'pareto_chart': create_pareto_chart(clv_distribution),
            'cohort_heatmap': create_cohort_heatmap(cohort_clv) if cohort_clv else None,
        }
        
        # Key insights
        key_insights = generate_key_insights(summary_data, customer_segments)
        
        return {
            'success': True,
            'results': {
                'summary': {k: _to_native_type(v) for k, v in summary_data.items()},
                'customer_segments': customer_segments,
                'cohort_clv': cohort_clv,
                'clv_distribution': clv_distribution,
                'rfm_segments': rfm_segments,
                'top_customers': top_customers,
            },
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'time_period': f"{min_date.strftime('%Y-%m-%d')} to {analysis_date.strftime('%Y-%m-%d')}",
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CLV analysis failed: {str(e)}")
