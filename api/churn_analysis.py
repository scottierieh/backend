"""
Churn & Drop-off Analysis Router for FastAPI
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
import time
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class ChurnRequest(BaseModel):
    data: List[Dict[str, Any]]
    user_id_col: str
    churn_col: str
    segment_cols: List[str]
    tenure_col: Optional[str] = None
    value_col: Optional[str] = None
    stage_col: Optional[str] = None
    date_col: Optional[str] = None
    analysis_type: Literal["churn", "drop_off", "cohort", "risk"] = "churn"


def _to_native_type(obj):
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


SEGMENT_COLORS = [
    '#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def calculate_segment_churn(df: pd.DataFrame, churn_col: str, segment_cols: List[str],
                            tenure_col: Optional[str], value_col: Optional[str]) -> List[Dict]:
    segments = []
    
    for col in segment_cols:
        for value in df[col].unique():
            segment_df = df[df[col] == value]
            total = len(segment_df)
            churned = segment_df[churn_col].sum()
            
            # Ensure value is converted to string properly
            value_str = str(value) if value is not None else "Unknown"
            
            segment_data = {
                'segment': f"{col}: {value_str}",
                'total_users': total,
                'churned_users': int(churned),
                'churn_rate': churned / total if total > 0 else 0,
                'avg_tenure': segment_df[tenure_col].mean() if tenure_col and tenure_col in df.columns else 0,
                'avg_value': segment_df[value_col].mean() if value_col and value_col in df.columns else 0
            }
            segments.append(segment_data)
    
    return segments


def calculate_drop_off(df: pd.DataFrame, stage_col: str, churn_col: str) -> List[Dict]:
    if stage_col not in df.columns:
        return []
    
    stages = df[stage_col].unique()
    stage_order = ['Signup', 'Onboarding', 'First_Value', 'Habit', 'Loyal']
    ordered_stages = [s for s in stage_order if s in stages]
    ordered_stages.extend([s for s in stages if s not in ordered_stages])
    
    drop_off_points = []
    total_users = len(df)
    cumulative = total_users
    
    for i, stage in enumerate(ordered_stages):
        stage_users = len(df[df[stage_col] == stage])
        entered = total_users if i == 0 else cumulative
        stage_churned = df[(df[stage_col] == stage) & (df[churn_col] == 1)]
        exited = len(stage_churned)
        cumulative = entered - exited
        
        drop_off_points.append({
            'stage': str(stage) if stage is not None else "Unknown",
            'entered': entered,
            'exited': exited,
            'drop_off_rate': exited / entered if entered > 0 else 0,
            'cumulative_retention': cumulative / total_users if total_users > 0 else 0
        })
    
    return drop_off_points


def calculate_risk_factors(df: pd.DataFrame, churn_col: str, 
                           segment_cols: List[str], tenure_col: Optional[str],
                           value_col: Optional[str]) -> List[Dict]:
    risk_factors = []
    overall_churn = df[churn_col].mean()
    
    for col in segment_cols:
        for value in df[col].unique():
            segment_churn = df[df[col] == value][churn_col].mean()
            lift = (segment_churn - overall_churn) / overall_churn if overall_churn > 0 else 0
            
            # Ensure value is converted to string properly
            value_str = str(value) if value is not None else "Unknown"
            
            if abs(lift) > 0.1:
                importance = min(abs(lift), 1.0)
                risk_factors.append({
                    'factor': f"{col} = {value_str}",
                    'importance': importance,
                    'direction': 'increases' if lift > 0 else 'decreases',
                    'description': f"{'Increases' if lift > 0 else 'Decreases'} churn by {abs(lift)*100:.0f}%"
                })
    
    if tenure_col and tenure_col in df.columns:
        low_tenure = df[df[tenure_col] <= df[tenure_col].quantile(0.25)]
        high_tenure = df[df[tenure_col] >= df[tenure_col].quantile(0.75)]
        low_churn = low_tenure[churn_col].mean()
        high_churn = high_tenure[churn_col].mean()
        
        if low_churn > high_churn:
            importance = min((low_churn - high_churn) / overall_churn, 1.0) if overall_churn > 0 else 0
            risk_factors.append({
                'factor': 'Low Tenure (<3 months)',
                'importance': abs(importance),
                'direction': 'increases',
                'description': f"New users have {((low_churn - overall_churn) / overall_churn * 100):.0f}% higher churn"
            })
    
    risk_factors.sort(key=lambda x: -x['importance'])
    return risk_factors[:10]


def create_churn_by_segment_chart(segments: List[Dict]) -> str:
    if not segments:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    sorted_segments = sorted(segments, key=lambda x: -x['churn_rate'])[:12]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [s['segment'] for s in sorted_segments]
    rates = [s['churn_rate'] * 100 for s in sorted_segments]
    colors = ['#ef4444' if r > 20 else '#f59e0b' if r > 15 else '#3b82f6' for r in rates]
    
    bars = ax.barh(labels, rates, color=colors, edgecolor='white', linewidth=1)
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%', ha='left', va='center', fontsize=9)
    
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.5, label='15% threshold')
    ax.set_xlabel('Churn Rate (%)', fontsize=11)
    ax.set_title('Churn Rate by Segment', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_drop_off_funnel(drop_off_points: List[Dict]) -> str:
    if not drop_off_points:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stages = [p['stage'] for p in drop_off_points]
    retentions = [p['cumulative_retention'] * 100 for p in drop_off_points]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
    
    bars = ax.bar(stages, retentions, color=colors, edgecolor='white', linewidth=2)
    
    for i, (bar, ret, point) in enumerate(zip(bars, retentions, drop_off_points)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ret:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if i > 0:
            drop = point['drop_off_rate'] * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'-{drop:.0f}%', ha='center', va='center', fontsize=9, color='red')
    
    ax.set_ylabel('Retention (%)', fontsize=11)
    ax.set_xlabel('Stage', fontsize=11)
    ax.set_title('User Retention Funnel', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_risk_factors_chart(risk_factors: List[Dict]) -> str:
    if not risk_factors:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No significant risk factors', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    factors = [f['factor'] for f in risk_factors[:8]]
    importances = [f['importance'] * 100 for f in risk_factors[:8]]
    colors = ['#ef4444' if f['direction'] == 'increases' else '#22c55e' for f in risk_factors[:8]]
    
    bars = ax.barh(factors, importances, color=colors, edgecolor='white', linewidth=1)
    
    for bar, imp, factor in zip(bars, importances, risk_factors[:8]):
        label = '↑' if factor['direction'] == 'increases' else '↓'
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{label} {imp:.0f}%', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Impact on Churn (%)', fontsize=11)
    ax.set_title('Risk Factors (Red=Increases, Green=Decreases Churn)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_tenure_distribution(df: pd.DataFrame, tenure_col: str, churn_col: str) -> str:
    if tenure_col not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No tenure data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    churned = df[df[churn_col] == 1][tenure_col]
    retained = df[df[churn_col] == 0][tenure_col]
    
    ax.hist(retained, bins=20, alpha=0.7, label='Retained', color='#22c55e', edgecolor='white')
    ax.hist(churned, bins=20, alpha=0.7, label='Churned', color='#ef4444', edgecolor='white')
    
    ax.axvline(x=churned.mean(), color='#ef4444', linestyle='--', linewidth=2, 
               label=f'Churned Avg: {churned.mean():.1f}')
    ax.axvline(x=retained.mean(), color='#22c55e', linestyle='--', linewidth=2,
               label=f'Retained Avg: {retained.mean():.1f}')
    
    ax.set_xlabel('Tenure (months)', fontsize=11)
    ax.set_ylabel('Number of Users', fontsize=11)
    ax.set_title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cohort_heatmap(df: pd.DataFrame, date_col: str, churn_col: str) -> str:
    if date_col not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No date data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    df['cohort'] = pd.to_datetime(df[date_col]).dt.to_period('M')
    
    cohort_data = df.groupby('cohort').agg({
        churn_col: ['count', 'sum']
    }).reset_index()
    cohort_data.columns = ['cohort', 'total', 'churned']
    cohort_data['retention'] = 1 - (cohort_data['churned'] / cohort_data['total'])
    
    if len(cohort_data) < 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Insufficient cohort data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cohorts = cohort_data['cohort'].astype(str).tolist()
    retentions = cohort_data['retention'].values * 100
    colors = plt.cm.RdYlGn(cohort_data['retention'].values)
    
    bars = ax.bar(cohorts, retentions, color=colors, edgecolor='white', linewidth=1)
    
    for bar, ret in zip(bars, retentions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ret:.0f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Retention Rate (%)', fontsize=11)
    ax.set_xlabel('Cohort', fontsize=11)
    ax.set_title('Retention by Cohort', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(overall_churn: float, segments: List[Dict], 
                          risk_factors: List[Dict], metrics: Dict) -> List[Dict]:
    insights = []
    
    if overall_churn > 0.20:
        insights.append({
            'title': f'Critical Churn Rate: {overall_churn*100:.1f}%',
            'description': 'Churn rate exceeds 20%. Immediate intervention required.',
            'status': 'warning'
        })
    elif overall_churn > 0.10:
        insights.append({
            'title': f'Elevated Churn Rate: {overall_churn*100:.1f}%',
            'description': 'Churn rate above 10%. Focus on retention strategies.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': f'Healthy Churn Rate: {overall_churn*100:.1f}%',
            'description': 'Churn rate below 10%. Continue monitoring.',
            'status': 'positive'
        })
    
    high_risk = [s for s in segments if s['churn_rate'] > 0.20]
    if high_risk:
        insights.append({
            'title': f'{len(high_risk)} High-Risk Segments Identified',
            'description': f"Highest: {high_risk[0]['segment']} at {high_risk[0]['churn_rate']*100:.0f}% churn.",
            'status': 'warning'
        })
    
    if metrics.get('avg_tenure_churned') and metrics.get('avg_tenure_retained'):
        if metrics['avg_tenure_churned'] < metrics['avg_tenure_retained'] * 0.5:
            insights.append({
                'title': 'Early Churn Problem',
                'description': f"Churned users average {metrics['avg_tenure_churned']:.1f} months vs {metrics['avg_tenure_retained']:.1f} for retained.",
                'status': 'warning'
            })
    
    if risk_factors:
        top = risk_factors[0]
        insights.append({
            'title': f"Top Risk Factor: {top['factor']}",
            'description': top['description'],
            'status': 'warning' if top['direction'] == 'increases' else 'neutral'
        })
    
    return insights


@router.post("/churn")
async def run_churn_analysis(request: ChurnRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        if request.user_id_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"User ID column '{request.user_id_col}' not found")
        if request.churn_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Churn column '{request.churn_col}' not found")
        
        df[request.churn_col] = pd.to_numeric(df[request.churn_col], errors='coerce').fillna(0).astype(int)
        
        total_users = len(df)
        churned_users = int(df[request.churn_col].sum())
        retained_users = total_users - churned_users
        overall_churn_rate = churned_users / total_users if total_users > 0 else 0
        
        segments = calculate_segment_churn(
            df, request.churn_col, request.segment_cols,
            request.tenure_col, request.value_col
        )
        
        drop_off_points = []
        if request.stage_col and request.stage_col in df.columns:
            drop_off_points = calculate_drop_off(df, request.stage_col, request.churn_col)
        
        risk_factors = calculate_risk_factors(
            df, request.churn_col, request.segment_cols,
            request.tenure_col, request.value_col
        )
        
        metrics = {
            'avg_tenure_churned': 0,
            'avg_tenure_retained': 0,
            'avg_value_churned': 0,
            'avg_value_retained': 0,
            'churn_cost': 0
        }
        
        if request.tenure_col and request.tenure_col in df.columns:
            churned_df = df[df[request.churn_col] == 1]
            retained_df = df[df[request.churn_col] == 0]
            metrics['avg_tenure_churned'] = churned_df[request.tenure_col].mean() if len(churned_df) > 0 else 0
            metrics['avg_tenure_retained'] = retained_df[request.tenure_col].mean() if len(retained_df) > 0 else 0
        
        if request.value_col and request.value_col in df.columns:
            churned_df = df[df[request.churn_col] == 1]
            retained_df = df[df[request.churn_col] == 0]
            metrics['avg_value_churned'] = churned_df[request.value_col].mean() if len(churned_df) > 0 else 0
            metrics['avg_value_retained'] = retained_df[request.value_col].mean() if len(retained_df) > 0 else 0
            metrics['churn_cost'] = metrics['avg_value_churned'] * churned_users
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        high_risk_segments = sum(1 for s in segments if s['churn_rate'] > 0.20)
        
        visualizations = {
            'churn_by_segment': create_churn_by_segment_chart(segments),
            'risk_factors_chart': create_risk_factors_chart(risk_factors),
        }
        
        if drop_off_points:
            visualizations['drop_off_funnel'] = create_drop_off_funnel(drop_off_points)
        
        if request.tenure_col and request.tenure_col in df.columns:
            visualizations['tenure_distribution'] = create_tenure_distribution(
                df, request.tenure_col, request.churn_col
            )
        
        if request.date_col and request.date_col in df.columns:
            visualizations['cohort_heatmap'] = create_cohort_heatmap(
                df, request.date_col, request.churn_col
            )
        
        key_insights = generate_key_insights(overall_churn_rate, segments, risk_factors, metrics)
        
        results = {
            'overall_churn_rate': overall_churn_rate,
            'total_users': total_users,
            'churned_users': churned_users,
            'retained_users': retained_users,
            'segments': [{k: _to_native_type(v) for k, v in s.items()} for s in segments],
            'drop_off_points': [{k: _to_native_type(v) for k, v in p.items()} for p in drop_off_points] if drop_off_points else None,
            'risk_factors': [{k: _to_native_type(v) for k, v in f.items()} for f in risk_factors],
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'analysis_type': request.analysis_type,
            'churn_rate': overall_churn_rate,
            'high_risk_segments': high_risk_segments,
            'solve_time_ms': solve_time_ms
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Churn analysis failed: {str(e)}")
