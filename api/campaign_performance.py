"""
Campaign Performance Evaluation Router for FastAPI
Analyzes marketing campaign metrics: ROI, ROAS, CTR, CVR, CPA
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
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CampaignRequest(BaseModel):
    data: List[Dict[str, Any]]
    campaign_id_col: str
    campaign_name_col: Optional[str] = None
    channel_col: Optional[str] = None
    impressions_col: str
    clicks_col: str
    conversions_col: Optional[str] = None
    spend_col: str
    revenue_col: Optional[str] = None
    date_col: Optional[str] = None


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


def calculate_campaign_metrics(row: pd.Series) -> Dict:
    """Calculate all campaign performance metrics"""
    impressions = max(1, row['_impressions'])
    clicks = max(0, row['_clicks'])
    conversions = max(0, row['_conversions'])
    spend = max(0.01, row['_spend'])
    revenue = max(0, row['_revenue'])
    
    ctr = (clicks / impressions) * 100 if impressions > 0 else 0
    cvr = (conversions / clicks) * 100 if clicks > 0 else 0
    cpc = spend / clicks if clicks > 0 else 0
    cpa = spend / conversions if conversions > 0 else float('inf')
    roas = revenue / spend if spend > 0 else 0
    roi = ((revenue - spend) / spend) * 100 if spend > 0 else 0
    
    # Performance score (weighted combination)
    score = 0
    if roas >= 4:
        score += 40
    elif roas >= 3:
        score += 30
    elif roas >= 2:
        score += 20
    elif roas >= 1:
        score += 10
    
    if ctr >= 5:
        score += 20
    elif ctr >= 3:
        score += 15
    elif ctr >= 2:
        score += 10
    elif ctr >= 1:
        score += 5
    
    if cvr >= 5:
        score += 20
    elif cvr >= 3:
        score += 15
    elif cvr >= 2:
        score += 10
    elif cvr >= 1:
        score += 5
    
    if roi >= 200:
        score += 20
    elif roi >= 100:
        score += 15
    elif roi >= 50:
        score += 10
    elif roi >= 0:
        score += 5
    
    # Performance tier
    if score >= 80:
        tier = 'Excellent'
    elif score >= 60:
        tier = 'Good'
    elif score >= 40:
        tier = 'Average'
    elif score >= 20:
        tier = 'Poor'
    else:
        tier = 'Critical'
    
    return {
        'ctr': float(ctr),
        'cvr': float(cvr),
        'cpc': float(cpc),
        'cpa': float(min(cpa, 9999999)),
        'roas': float(roas),
        'roi': float(roi),
        'performance_score': float(score),
        'performance_tier': tier,
    }


def calculate_channel_summary(df: pd.DataFrame, channel_col: str) -> List[Dict]:
    """Calculate summary metrics by channel"""
    if channel_col not in df.columns:
        return []
    
    summaries = []
    for channel in df[channel_col].unique():
        ch_df = df[df[channel_col] == channel]
        
        total_spend = ch_df['_spend'].sum()
        total_revenue = ch_df['_revenue'].sum()
        total_impressions = ch_df['_impressions'].sum()
        total_clicks = ch_df['_clicks'].sum()
        total_conversions = ch_df['_conversions'].sum()
        
        summaries.append({
            'channel': str(channel),
            'campaigns': len(ch_df),
            'total_spend': float(total_spend),
            'total_revenue': float(total_revenue),
            'total_impressions': int(total_impressions),
            'total_clicks': int(total_clicks),
            'total_conversions': int(total_conversions),
            'avg_ctr': float((total_clicks / total_impressions * 100) if total_impressions > 0 else 0),
            'avg_cvr': float((total_conversions / total_clicks * 100) if total_clicks > 0 else 0),
            'avg_cpa': float(total_spend / total_conversions if total_conversions > 0 else 0),
            'roas': float(total_revenue / total_spend if total_spend > 0 else 0),
            'roi': float((total_revenue - total_spend) / total_spend * 100 if total_spend > 0 else 0),
        })
    
    # Sort by ROAS descending
    summaries.sort(key=lambda x: x['roas'], reverse=True)
    return summaries


# ============ VISUALIZATION ============
def create_channel_performance_chart(channel_summary: List[Dict]) -> str:
    """Create channel ROAS comparison chart"""
    if not channel_summary:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    channels = [c['channel'] for c in channel_summary]
    roas_values = [c['roas'] for c in channel_summary]
    
    colors = ['#22c55e' if r >= 3 else '#3b82f6' if r >= 2 else '#f59e0b' if r >= 1 else '#ef4444' for r in roas_values]
    
    bars = ax.bar(channels, roas_values, color=colors, edgecolor='white')
    
    ax.axhline(y=3, color='#22c55e', linestyle='--', alpha=0.7, label='Target ROAS (3x)')
    ax.axhline(y=1, color='#ef4444', linestyle='--', alpha=0.7, label='Break-even (1x)')
    
    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel('ROAS', fontsize=11)
    ax.set_title('Channel Performance (ROAS)', fontsize=14, fontweight='bold')
    ax.legend()
    
    for bar, roas in zip(bars, roas_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{roas:.2f}x', ha='center', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_spend_vs_revenue_chart(channel_summary: List[Dict]) -> str:
    """Create spend vs revenue comparison"""
    if not channel_summary:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    channels = [c['channel'] for c in channel_summary]
    spends = [c['total_spend'] / 1000 for c in channel_summary]
    revenues = [c['total_revenue'] / 1000 for c in channel_summary]
    
    x = np.arange(len(channels))
    width = 0.35
    
    ax.bar(x - width/2, spends, width, label='Spend', color='#ef4444', alpha=0.8)
    ax.bar(x + width/2, revenues, width, label='Revenue', color='#22c55e', alpha=0.8)
    
    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel('Amount ($K)', fontsize=11)
    ax.set_title('Spend vs Revenue by Channel', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_conversion_funnel_chart(summary: Dict) -> str:
    """Create conversion funnel visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = ['Impressions', 'Clicks', 'Conversions']
    values = [summary['total_impressions'], summary['total_clicks'], summary['total_conversions']]
    
    # Normalize for visualization
    max_val = max(values)
    widths = [v / max_val for v in values]
    
    colors = ['#3b82f6', '#8b5cf6', '#22c55e']
    
    for i, (stage, width, value, color) in enumerate(zip(stages, widths, values, colors)):
        left = (1 - width) / 2
        ax.barh(i, width, left=left, height=0.6, color=color, alpha=0.8)
        ax.text(0.5, i, f'{stage}\n{value:,.0f}', ha='center', va='center', fontsize=11, fontweight='bold')
        
        if i > 0:
            rate = (values[i] / values[i-1] * 100) if values[i-1] > 0 else 0
            ax.text(0.95, i - 0.5, f'{rate:.1f}%', ha='right', va='center', fontsize=9, color='#64748b')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_title('Conversion Funnel', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_roas_comparison_chart(campaign_metrics: List[Dict]) -> str:
    """Create ROAS comparison for top campaigns"""
    if not campaign_metrics:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Top 15 campaigns by ROAS
    top_campaigns = sorted(campaign_metrics, key=lambda x: x['roas'], reverse=True)[:15]
    
    names = [c['campaign_name'][:20] for c in top_campaigns]
    roas_values = [c['roas'] for c in top_campaigns]
    
    colors = ['#22c55e' if r >= 3 else '#3b82f6' if r >= 2 else '#f59e0b' if r >= 1 else '#ef4444' for r in roas_values]
    
    bars = ax.barh(names, roas_values, color=colors, edgecolor='white')
    
    ax.axvline(x=3, color='#22c55e', linestyle='--', alpha=0.7)
    ax.axvline(x=1, color='#ef4444', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('ROAS', fontsize=11)
    ax.set_title('Top Campaigns by ROAS', fontsize=14, fontweight='bold')
    
    ax.invert_yaxis()
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_recommendations(summary: Dict, channel_summary: List[Dict], campaign_metrics: List[Dict]) -> List[Dict]:
    """Generate optimization recommendations"""
    recommendations = []
    
    # Overall ROAS check
    if summary['overall_roas'] < 1:
        recommendations.append({
            'priority': 'High', 'category': 'Overall Performance',
            'recommendation': 'Overall ROAS is below break-even. Immediately pause underperforming campaigns.',
            'impact': 'Stop revenue loss',
        })
    elif summary['overall_roas'] < 2:
        recommendations.append({
            'priority': 'Medium', 'category': 'Overall Performance',
            'recommendation': 'ROAS is below target (3x). Review and optimize ad creatives and targeting.',
            'impact': 'Improve profitability',
        })
    
    # Channel recommendations
    if channel_summary:
        best_channel = channel_summary[0]
        worst_channel = channel_summary[-1] if len(channel_summary) > 1 else None
        
        if best_channel['roas'] >= 3:
            recommendations.append({
                'priority': 'Medium', 'category': 'Budget Allocation',
                'recommendation': f"Increase budget for {best_channel['channel']} (ROAS: {best_channel['roas']:.2f}x)",
                'impact': f"Scale successful channel",
            })
        
        if worst_channel and worst_channel['roas'] < 1:
            recommendations.append({
                'priority': 'High', 'category': 'Budget Allocation',
                'recommendation': f"Reduce/pause spend on {worst_channel['channel']} (ROAS: {worst_channel['roas']:.2f}x)",
                'impact': 'Eliminate unprofitable spend',
            })
    
    # Campaign-level recommendations
    poor_campaigns = [c for c in campaign_metrics if c['performance_tier'] in ['Poor', 'Critical']]
    if len(poor_campaigns) > 5:
        recommendations.append({
            'priority': 'High', 'category': 'Campaign Optimization',
            'recommendation': f"Review {len(poor_campaigns)} underperforming campaigns for optimization or termination",
            'impact': 'Improve overall portfolio performance',
        })
    
    # CTR check
    if summary['overall_ctr'] < 1:
        recommendations.append({
            'priority': 'Medium', 'category': 'Creative Optimization',
            'recommendation': 'CTR is below 1%. Test new ad creatives and headlines.',
            'impact': 'Improve engagement',
        })
    
    # CVR check
    if summary['overall_cvr'] < 2:
        recommendations.append({
            'priority': 'Medium', 'category': 'Landing Page',
            'recommendation': 'CVR is below 2%. Optimize landing pages and conversion flow.',
            'impact': 'Improve conversion efficiency',
        })
    
    return recommendations


def generate_key_insights(summary: Dict, channel_summary: List[Dict]) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # ROAS insight
    roas = summary['overall_roas']
    if roas >= 4:
        insights.append({'title': f"Excellent ROAS ({roas:.2f}x)", 'description': "Campaign portfolio is highly profitable.", 'status': 'positive'})
    elif roas >= 3:
        insights.append({'title': f"Strong ROAS ({roas:.2f}x)", 'description': "Meeting target return on ad spend.", 'status': 'positive'})
    elif roas >= 1:
        insights.append({'title': f"Below Target ROAS ({roas:.2f}x)", 'description': "Profitable but below 3x target. Optimization needed.", 'status': 'neutral'})
    else:
        insights.append({'title': f"Negative ROAS ({roas:.2f}x)", 'description': "Campaigns are losing money. Immediate action required.", 'status': 'warning'})
    
    # ROI insight
    roi = summary['overall_roi']
    insights.append({
        'title': f"ROI: {roi:.0f}%",
        'description': f"${summary['total_spend']:,.0f} invested returned ${summary['total_revenue']:,.0f}",
        'status': 'positive' if roi >= 100 else 'neutral' if roi >= 0 else 'warning'
    })
    
    # Channel insight
    if channel_summary:
        best = channel_summary[0]
        insights.append({
            'title': f"Best Channel: {best['channel']}",
            'description': f"ROAS: {best['roas']:.2f}x with {best['campaigns']} campaigns",
            'status': 'positive'
        })
    
    # Conversion insight
    ctr = summary['overall_ctr']
    cvr = summary['overall_cvr']
    insights.append({
        'title': f"Funnel: {ctr:.2f}% CTR → {cvr:.2f}% CVR",
        'description': f"{summary['total_conversions']:,} conversions from {summary['total_clicks']:,} clicks",
        'status': 'positive' if cvr >= 3 else 'neutral'
    })
    
    return insights


@router.post("/campaign-performance")
async def run_campaign_analysis(request: CampaignRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        for col in [request.campaign_id_col, request.impressions_col, request.clicks_col, request.spend_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Prepare data
        df['_impressions'] = pd.to_numeric(df[request.impressions_col], errors='coerce').fillna(0)
        df['_clicks'] = pd.to_numeric(df[request.clicks_col], errors='coerce').fillna(0)
        df['_spend'] = pd.to_numeric(df[request.spend_col], errors='coerce').fillna(0)
        
        df['_conversions'] = pd.to_numeric(df[request.conversions_col], errors='coerce').fillna(0) if request.conversions_col and request.conversions_col in df.columns else 0
        df['_revenue'] = pd.to_numeric(df[request.revenue_col], errors='coerce').fillna(0) if request.revenue_col and request.revenue_col in df.columns else 0
        
        df['_name'] = df[request.campaign_name_col].astype(str) if request.campaign_name_col and request.campaign_name_col in df.columns else df[request.campaign_id_col].astype(str)
        df['_channel'] = df[request.channel_col].astype(str) if request.channel_col and request.channel_col in df.columns else 'Unknown'
        
        # Calculate metrics for each campaign
        campaign_metrics = []
        for _, row in df.iterrows():
            metrics = calculate_campaign_metrics(row)
            campaign_metrics.append({
                'campaign_id': str(row[request.campaign_id_col]),
                'campaign_name': str(row['_name']),
                'channel': str(row['_channel']),
                'impressions': int(row['_impressions']),
                'clicks': int(row['_clicks']),
                'conversions': int(row['_conversions']),
                'spend': float(row['_spend']),
                'revenue': float(row['_revenue']),
                **metrics,
            })
        
        # Sort by performance score
        campaign_metrics.sort(key=lambda x: x['performance_score'], reverse=True)
        
        # Overall summary
        total_impressions = df['_impressions'].sum()
        total_clicks = df['_clicks'].sum()
        total_conversions = df['_conversions'].sum()
        total_spend = df['_spend'].sum()
        total_revenue = df['_revenue'].sum()
        
        summary_data = {
            'total_campaigns': len(df),
            'total_spend': float(total_spend),
            'total_revenue': float(total_revenue),
            'total_impressions': int(total_impressions),
            'total_clicks': int(total_clicks),
            'total_conversions': int(total_conversions),
            'overall_ctr': float((total_clicks / total_impressions * 100) if total_impressions > 0 else 0),
            'overall_cvr': float((total_conversions / total_clicks * 100) if total_clicks > 0 else 0),
            'overall_cpa': float(total_spend / total_conversions if total_conversions > 0 else 0),
            'overall_roas': float(total_revenue / total_spend if total_spend > 0 else 0),
            'overall_roi': float((total_revenue - total_spend) / total_spend * 100 if total_spend > 0 else 0),
            'top_performer': campaign_metrics[0]['campaign_name'] if campaign_metrics else 'N/A',
            'worst_performer': campaign_metrics[-1]['campaign_name'] if campaign_metrics else 'N/A',
        }
        
        # Channel summary
        channel_summary = calculate_channel_summary(df, '_channel')
        
        # Performance distribution
        tiers = ['Excellent', 'Good', 'Average', 'Poor', 'Critical']
        distribution = []
        for tier in tiers:
            count = len([c for c in campaign_metrics if c['performance_tier'] == tier])
            distribution.append({
                'tier': tier,
                'count': count,
                'pct': count / len(campaign_metrics) if campaign_metrics else 0,
            })
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {
            'channel_performance': create_channel_performance_chart(channel_summary),
            'spend_vs_revenue': create_spend_vs_revenue_chart(channel_summary),
            'conversion_funnel': create_conversion_funnel_chart(summary_data),
            'roas_comparison': create_roas_comparison_chart(campaign_metrics),
        }
        
        # Recommendations and insights
        recommendations = generate_recommendations(summary_data, channel_summary, campaign_metrics)
        key_insights = generate_key_insights(summary_data, channel_summary)
        
        # Best/worst channel
        best_channel = channel_summary[0]['channel'] if channel_summary else 'N/A'
        worst_channel = channel_summary[-1]['channel'] if channel_summary else 'N/A'
        
        return {
            'success': True,
            'results': {
                'summary': {k: _to_native_type(v) for k, v in summary_data.items()},
                'campaign_metrics': campaign_metrics,
                'channel_summary': channel_summary,
                'time_trends': [],
                'performance_distribution': distribution,
                'recommendations': recommendations,
            },
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'best_channel': best_channel,
                'worst_channel': worst_channel,
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Campaign analysis failed: {str(e)}")
