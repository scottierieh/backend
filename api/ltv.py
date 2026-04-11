"""
LTV (Customer Lifetime Value) Prediction Router for FastAPI
BG/NBD + Gamma-Gamma Model based LTV prediction
- Modified to match frontend expected structure
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
import io
import base64
from datetime import datetime, timedelta
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Try to import lifetimes library
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False

router = APIRouter()


# ============ Request Model (프론트엔드에 맞춤) ============
class LTVRequest(BaseModel):
    data: List[Dict[str, Any]]
    customer_id_col: str
    date_col: str  # 프론트엔드: date_col
    amount_col: str  # 프론트엔드: amount_col
    model: str = "bgf_ggf"  # 프론트엔드: model (bgf_ggf, simple, rfm)
    prediction_days: int = 365  # 프론트엔드: prediction_days (일 단위)
    discount_rate: float = 0.1  # 프론트엔드: discount_rate (연간)


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


def calculate_rfm_data(df: pd.DataFrame, customer_id_col: str, 
                       datetime_col: str, monetary_col: str) -> tuple:
    """Calculate RFM summary data from transaction data"""
    
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df[monetary_col] = pd.to_numeric(df[monetary_col], errors='coerce')
    
    # Remove invalid data
    df = df.dropna(subset=[customer_id_col, datetime_col, monetary_col])
    df = df[df[monetary_col] > 0]
    
    if len(df) == 0:
        raise ValueError("No valid transaction data after cleaning")
    
    # Get observation period end
    snapshot_date = df[datetime_col].max() + timedelta(days=1)
    
    # Use lifetimes utility if available
    if LIFETIMES_AVAILABLE:
        rfm_summary = summary_data_from_transaction_data(
            df,
            customer_id_col=customer_id_col,
            datetime_col=datetime_col,
            monetary_value_col=monetary_col,
            observation_period_end=snapshot_date
        )
        rfm_summary = rfm_summary.reset_index()
        rfm_summary = rfm_summary.rename(columns={rfm_summary.columns[0]: 'customer_id'})
    else:
        # Manual RFM calculation
        customer_data = df.groupby(customer_id_col).agg({
            datetime_col: ['min', 'max', 'count'],
            monetary_col: ['sum', 'mean']
        }).reset_index()
        customer_data.columns = ['customer_id', 'first_purchase', 'last_purchase', 
                                  'frequency_total', 'total_monetary', 'monetary_value']
        
        customer_data['recency'] = (customer_data['last_purchase'] - customer_data['first_purchase']).dt.days
        customer_data['T'] = (snapshot_date - customer_data['first_purchase']).dt.days
        customer_data['frequency'] = customer_data['frequency_total'] - 1  # Repeat purchases only
        
        rfm_summary = customer_data[['customer_id', 'frequency', 'recency', 'T', 'monetary_value', 'total_monetary']]
    
    # 총 구매액 계산 (historical_revenue 용)
    total_by_customer = df.groupby(customer_id_col)[monetary_col].sum().reset_index()
    total_by_customer.columns = ['customer_id', 'historical_revenue']
    rfm_summary = rfm_summary.merge(total_by_customer, on='customer_id', how='left')
    
    return rfm_summary, snapshot_date


def predict_ltv_bgf_ggf(rfm_df: pd.DataFrame, prediction_days: int, 
                        discount_rate: float) -> pd.DataFrame:
    """Predict LTV using BG/NBD and Gamma-Gamma models"""
    
    all_customers = rfm_df.copy()
    repeat_customers = rfm_df[rfm_df['frequency'] > 0].copy()
    
    # 기본값 설정
    all_customers['predicted_purchases'] = 0.0
    all_customers['probability_alive'] = 1.0
    all_customers['expected_avg_value'] = all_customers['monetary_value'].fillna(0)
    all_customers['predicted_future_revenue'] = 0.0
    
    model_params = None
    
    if len(repeat_customers) < 5 or not LIFETIMES_AVAILABLE:
        # Fallback to simple calculation
        return predict_ltv_simple(rfm_df, prediction_days, discount_rate), None
    
    try:
        # Fit BG/NBD model
        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(
            repeat_customers['frequency'],
            repeat_customers['recency'],
            repeat_customers['T']
        )
        
        # Predict future transactions for ALL customers
        all_customers['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
            prediction_days,
            all_customers['frequency'],
            all_customers['recency'],
            all_customers['T']
        )
        
        # Probability alive for all customers
        all_customers['probability_alive'] = bgf.conditional_probability_alive(
            all_customers['frequency'],
            all_customers['recency'],
            all_customers['T']
        )
        
        # Model parameters
        model_params = {
            'r': _to_native_type(bgf.params_['r']),
            'alpha': _to_native_type(bgf.params_['alpha']),
            'a': _to_native_type(bgf.params_['a']),
            'b': _to_native_type(bgf.params_['b'])
        }
        
        # Fit Gamma-Gamma model
        gg_data = repeat_customers[repeat_customers['monetary_value'] > 0].copy()
        
        if len(gg_data) >= 5:
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(
                gg_data['frequency'],
                gg_data['monetary_value']
            )
            
            # Predict average monetary value for customers with frequency > 0
            mask_repeat = all_customers['frequency'] > 0
            all_customers.loc[mask_repeat, 'expected_avg_value'] = ggf.conditional_expected_average_profit(
                all_customers.loc[mask_repeat, 'frequency'],
                all_customers.loc[mask_repeat, 'monetary_value']
            )
            
            # For customers with frequency = 0, use overall average
            avg_monetary = all_customers.loc[mask_repeat, 'monetary_value'].mean()
            all_customers.loc[~mask_repeat, 'expected_avg_value'] = avg_monetary
        
        # Calculate predicted future revenue (할인 적용)
        monthly_discount = discount_rate / 12
        discount_factor = (1 / (1 + monthly_discount)) ** (prediction_days / 30)
        
        all_customers['predicted_future_revenue'] = (
            all_customers['predicted_purchases'] * 
            all_customers['expected_avg_value'] * 
            discount_factor
        )
        
    except Exception as e:
        # Fallback to simple calculation
        return predict_ltv_simple(rfm_df, prediction_days, discount_rate), None
    
    return all_customers, model_params


def predict_ltv_simple(rfm_df: pd.DataFrame, prediction_days: int, 
                       discount_rate: float) -> pd.DataFrame:
    """Calculate simple LTV without probabilistic models"""
    
    df = rfm_df.copy()
    
    avg_monetary = df['monetary_value'].mean()
    
    # Estimate future purchases based on historical frequency
    df['avg_purchase_interval'] = df.apply(
        lambda x: x['T'] / (x['frequency'] + 1) if x['T'] > 0 else 30, axis=1
    )
    df['predicted_purchases'] = prediction_days / df['avg_purchase_interval']
    df['predicted_purchases'] = df['predicted_purchases'].clip(upper=prediction_days / 7)  # Max weekly
    
    df['probability_alive'] = 1.0  # Assume all alive for simple model
    df['expected_avg_value'] = df['monetary_value'].fillna(avg_monetary)
    df['predicted_future_revenue'] = df['predicted_purchases'] * df['expected_avg_value']
    
    return df


def predict_ltv_rfm(rfm_df: pd.DataFrame, prediction_days: int, 
                    discount_rate: float) -> pd.DataFrame:
    """RFM-based LTV calculation with segment multipliers"""
    
    df = rfm_df.copy()
    
    # RFM scoring (1-5)
    df['R_score'] = pd.qcut(df['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_score'] = pd.qcut(df['monetary_value'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    df['RFM_score'] = df['R_score'] + df['F_score'] + df['M_score']
    
    # Segment multipliers based on RFM score
    def get_multiplier(score):
        if score >= 12:
            return 2.5  # Champions
        elif score >= 9:
            return 1.8  # Loyal
        elif score >= 6:
            return 1.2  # Potential
        elif score >= 4:
            return 0.6  # At Risk
        else:
            return 0.3  # Dormant
    
    df['multiplier'] = df['RFM_score'].apply(get_multiplier)
    
    # Simple base prediction
    avg_monetary = df['monetary_value'].mean()
    df['avg_purchase_interval'] = df.apply(
        lambda x: x['T'] / (x['frequency'] + 1) if x['T'] > 0 else 30, axis=1
    )
    df['predicted_purchases'] = (prediction_days / df['avg_purchase_interval']) * df['multiplier']
    df['predicted_purchases'] = df['predicted_purchases'].clip(upper=prediction_days / 7)
    
    df['probability_alive'] = df['multiplier'] / 2.5  # Normalize to 0-1
    df['expected_avg_value'] = df['monetary_value'].fillna(avg_monetary)
    df['predicted_future_revenue'] = df['predicted_purchases'] * df['expected_avg_value']
    
    return df


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign customer segments based on RFM and probability alive"""
    
    df = df.copy()
    
    if len(df) < 5:
        df['segment'] = 'Unknown'
        return df
    
    # Calculate percentiles for LTV
    ltv_percentiles = df['predicted_ltv'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    
    def assign_segment(row):
        ltv = row['predicted_ltv']
        prob_alive = row.get('probability_alive', 1.0)
        recency = row.get('recency', 0)
        frequency = row.get('frequency', 0)
        
        # High LTV customers
        if ltv >= ltv_percentiles[0.8]:
            if prob_alive >= 0.7 and frequency >= 3:
                return 'Champions'
            else:
                return 'Loyal'
        elif ltv >= ltv_percentiles[0.6]:
            if prob_alive >= 0.5:
                return 'Loyal'
            else:
                return 'At Risk'
        elif ltv >= ltv_percentiles[0.4]:
            if frequency <= 2:
                return 'New'
            else:
                return 'Potential'
        elif ltv >= ltv_percentiles[0.2]:
            if prob_alive < 0.3:
                return 'Dormant'
            else:
                return 'At Risk'
        else:
            if prob_alive < 0.3:
                return 'Dormant'
            else:
                return 'At Risk'
    
    df['segment'] = df.apply(assign_segment, axis=1)
    
    return df


def calculate_ltv_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate LTV percentile for each customer"""
    df = df.copy()
    df['ltv_percentile'] = df['predicted_ltv'].rank(pct=True) * 100
    return df


# ============ Visualization Functions ============

def create_ltv_distribution_chart(df: pd.DataFrame) -> str:
    """Create LTV distribution histogram"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ltv_values = df['predicted_ltv'].dropna()
    q99 = ltv_values.quantile(0.99)
    ltv_clipped = ltv_values[ltv_values <= q99]
    
    sns.histplot(ltv_clipped, bins=50, kde=True, color='#2563eb', alpha=0.7, ax=ax)
    
    # Percentile lines
    percentiles = [0.25, 0.50, 0.75, 0.90]
    colors = ['#22c55e', '#eab308', '#f97316', '#ef4444']
    labels = ['25th', 'Median', '75th', '90th']
    
    for p, c, l in zip(percentiles, colors, labels):
        val = ltv_values.quantile(p)
        ax.axvline(val, color=c, linestyle='--', linewidth=2, label=f'{l}: ${val:,.0f}')
    
    ax.set_xlabel('Predicted LTV ($)', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_title('LTV Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_comparison_chart(segments: List[Dict]) -> str:
    """Create segment comparison bar chart"""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sns.set_style("whitegrid")
    
    seg_names = [s['segment'] for s in segments]
    colors_map = {
        'Champions': '#22c55e',
        'Loyal': '#84cc16',
        'Potential': '#f59e0b',
        'New': '#3b82f6',
        'At Risk': '#f97316',
        'Dormant': '#ef4444'
    }
    colors = [colors_map.get(s, '#6b7280') for s in seg_names]
    
    # Chart 1: Customer Count
    counts = [s['customer_count'] for s in segments]
    bars1 = axes[0].bar(seg_names, counts, color=colors, alpha=0.8)
    axes[0].set_title('Customers by Segment', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Chart 2: Average LTV
    avg_ltvs = [s['avg_ltv'] for s in segments]
    bars2 = axes[1].bar(seg_names, avg_ltvs, color=colors, alpha=0.8)
    axes[1].set_title('Average LTV by Segment', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Avg LTV ($)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Chart 3: Total LTV
    total_ltvs = [s['total_ltv'] for s in segments]
    bars3 = axes[2].bar(seg_names, total_ltvs, color=colors, alpha=0.8)
    axes[2].set_title('Total LTV by Segment', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Total LTV ($)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_rfm_scatter_chart(df: pd.DataFrame) -> str:
    """Create RFM scatter plot"""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    plot_df = df.dropna(subset=['frequency', 'monetary_value', 'predicted_ltv']).copy()
    
    # Clip outliers
    for col in ['frequency', 'monetary_value']:
        q99 = plot_df[col].quantile(0.99)
        plot_df = plot_df[plot_df[col] <= q99]
    
    scatter = ax.scatter(
        plot_df['frequency'],
        plot_df['monetary_value'],
        c=plot_df['predicted_ltv'],
        cmap='RdYlGn',
        alpha=0.6,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Predicted LTV ($)', fontsize=10)
    
    ax.set_xlabel('Purchase Frequency', fontsize=11)
    ax.set_ylabel('Average Monetary Value ($)', fontsize=11)
    ax.set_title('RFM Distribution by LTV', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_top_customers_chart(customers: List[Dict], top_n: int = 10) -> str:
    """Create top customers horizontal bar chart"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    top = customers[:top_n]
    names = [c['customer_id'][:15] for c in top]
    ltvs = [c['predicted_ltv'] for c in top]
    
    colors_map = {
        'Champions': '#22c55e',
        'Loyal': '#84cc16',
        'Potential': '#f59e0b',
        'New': '#3b82f6',
        'At Risk': '#f97316',
        'Dormant': '#ef4444'
    }
    colors = [colors_map.get(c.get('segment', ''), '#6b7280') for c in top]
    
    bars = ax.barh(names[::-1], ltvs[::-1], color=colors[::-1], alpha=0.8)
    
    ax.set_xlabel('Predicted LTV ($)', fontsize=11)
    ax.set_title(f'Top {top_n} Customers by LTV', fontsize=13, fontweight='bold')
    
    for bar, ltv in zip(bars, ltvs[::-1]):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                f'${ltv:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(summary: Dict, segments: List[Dict]) -> List[Dict]:
    """Generate key insights from LTV analysis"""
    
    insights = []
    
    total_customers = summary.get('total_customers', 0)
    avg_ltv = summary.get('avg_ltv', 0)
    median_ltv = summary.get('median_ltv', 0)
    high_value = summary.get('high_value_customers', 0)
    at_risk = summary.get('at_risk_customers', 0)
    
    # Insight 1: Overall health
    if high_value / total_customers > 0.15:
        insights.append({
            'title': 'Strong High-Value Base',
            'description': f'{high_value} customers ({high_value/total_customers*100:.1f}%) are high-value. This is a healthy customer base.',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': 'Growth Opportunity',
            'description': f'Only {high_value} customers ({high_value/total_customers*100:.1f}%) are high-value. Focus on increasing customer value.',
            'status': 'neutral'
        })
    
    # Insight 2: At-risk alert
    if at_risk / total_customers > 0.25:
        insights.append({
            'title': 'At-Risk Alert',
            'description': f'{at_risk} customers ({at_risk/total_customers*100:.1f}%) are at risk of churning. Implement retention campaigns.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Low Churn Risk',
            'description': f'Only {at_risk/total_customers*100:.1f}% customers are at risk. Retention is healthy.',
            'status': 'positive'
        })
    
    # Insight 3: Value distribution
    if avg_ltv > median_ltv * 1.5:
        insights.append({
            'title': 'Value Concentration',
            'description': f'Top customers drive significant value. Average LTV (${avg_ltv:,.0f}) is much higher than median (${median_ltv:,.0f}).',
            'status': 'neutral'
        })
    
    # Insight 4: CAC recommendation
    recommended_cac = avg_ltv / 3
    insights.append({
        'title': 'Acquisition Budget',
        'description': f'Based on ${avg_ltv:,.0f} average LTV, keep CAC under ${recommended_cac:,.0f} for healthy 3:1 LTV:CAC ratio.',
        'status': 'neutral'
    })
    
    return insights


# ============ Main API Endpoint (프론트엔드 구조에 맞춤) ============

@router.post("/ltv")
async def run_ltv_prediction(request: LTVRequest) -> Dict[str, Any]:
    """
    Perform LTV (Customer Lifetime Value) Prediction analysis.
    Response structure matches frontend expectations.
    """
    start_time = time.time()
    
    try:
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        required_cols = [request.customer_id_col, request.date_col, request.amount_col]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in data")
        
        # Validate data size
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="At least 10 transactions required for LTV analysis")
        
        # Calculate RFM data
        rfm_df, snapshot_date = calculate_rfm_data(
            df, request.customer_id_col, request.date_col, request.amount_col
        )
        
        total_customers = len(rfm_df)
        
        if total_customers < 5:
            raise HTTPException(status_code=400, detail=f"Not enough unique customers ({total_customers}). Need at least 5.")
        
        # Run prediction based on selected model
        model_params = None
        if request.model == "bgf_ggf":
            customer_df, model_params = predict_ltv_bgf_ggf(rfm_df, request.prediction_days, request.discount_rate)
        elif request.model == "simple":
            customer_df = predict_ltv_simple(rfm_df, request.prediction_days, request.discount_rate)
        elif request.model == "rfm":
            customer_df = predict_ltv_rfm(rfm_df, request.prediction_days, request.discount_rate)
        else:
            customer_df, model_params = predict_ltv_bgf_ggf(rfm_df, request.prediction_days, request.discount_rate)
        
        # Calculate total LTV (historical + future)
        customer_df['predicted_ltv'] = (
            customer_df['historical_revenue'].fillna(0) + 
            customer_df['predicted_future_revenue'].fillna(0)
        )
        
        # Assign segments
        customer_df = assign_segments(customer_df)
        
        # Calculate percentiles
        customer_df = calculate_ltv_percentile(customer_df)
        
        # Sort by LTV descending
        customer_df = customer_df.sort_values('predicted_ltv', ascending=False)
        
        # ============ Build Response (프론트엔드 구조) ============
        
        ltv_values = customer_df['predicted_ltv'].dropna()
        
        # Summary stats
        high_value_customers = len(customer_df[customer_df['segment'].isin(['Champions', 'Loyal'])])
        at_risk_customers = len(customer_df[customer_df['segment'].isin(['At Risk', 'Dormant'])])
        
        # Build customers list (프론트엔드 CustomerLTV 인터페이스에 맞춤)
        customers_list = []
        for _, row in customer_df.iterrows():
            customers_list.append({
                'customer_id': str(row['customer_id']),
                'predicted_ltv': _to_native_type(row['predicted_ltv']),
                'historical_revenue': _to_native_type(row.get('historical_revenue', 0)),
                'predicted_future_revenue': _to_native_type(row.get('predicted_future_revenue', 0)),
                'expected_purchases': _to_native_type(row.get('predicted_purchases', 0)),
                'probability_alive': _to_native_type(row.get('probability_alive', 1.0)),
                'segment': row.get('segment', 'Unknown'),
                'ltv_percentile': _to_native_type(row.get('ltv_percentile', 50))
            })
        
        # Build segments list (프론트엔드 SegmentSummary 인터페이스에 맞춤)
        segment_order = ['Champions', 'Loyal', 'Potential', 'New', 'At Risk', 'Dormant']
        segments_list = []
        for seg_name in segment_order:
            seg_data = customer_df[customer_df['segment'] == seg_name]
            if len(seg_data) > 0:
                segments_list.append({
                    'segment': seg_name,
                    'customer_count': int(len(seg_data)),
                    'avg_ltv': _to_native_type(seg_data['predicted_ltv'].mean()),
                    'total_ltv': _to_native_type(seg_data['predicted_ltv'].sum()),
                    'avg_frequency': _to_native_type(seg_data['frequency'].mean()) if 'frequency' in seg_data.columns else 0,
                    'avg_recency': _to_native_type(seg_data['recency'].mean()) if 'recency' in seg_data.columns else 0,
                    'avg_monetary': _to_native_type(seg_data['monetary_value'].mean()) if 'monetary_value' in seg_data.columns else 0
                })
        
        # RFM summary
        rfm_summary = {
            'avg_recency': _to_native_type(customer_df['recency'].mean()) if 'recency' in customer_df.columns else 0,
            'avg_frequency': _to_native_type(customer_df['frequency'].mean()) if 'frequency' in customer_df.columns else 0,
            'avg_monetary': _to_native_type(customer_df['monetary_value'].mean()) if 'monetary_value' in customer_df.columns else 0
        }
        
        # Create visualizations
        visualizations = {}
        try:
            visualizations['ltv_distribution'] = create_ltv_distribution_chart(customer_df)
            visualizations['segment_comparison'] = create_segment_comparison_chart(segments_list)
            visualizations['rfm_scatter'] = create_rfm_scatter_chart(customer_df)
            visualizations['top_customers'] = create_top_customers_chart(customers_list)
        except Exception as viz_error:
            pass  # Continue without visualizations
        
        # Calculate solve time
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Summary for results.summary (프론트엔드 구조)
        results_summary = {
            'total_customers': total_customers,
            'total_predicted_ltv': _to_native_type(ltv_values.sum()),
            'avg_ltv': _to_native_type(ltv_values.mean()),
            'median_ltv': _to_native_type(ltv_values.median()),
            'top_10_percent_ltv': _to_native_type(ltv_values.quantile(0.9)),
            'bottom_10_percent_ltv': _to_native_type(ltv_values.quantile(0.1)),
            'high_value_customers': high_value_customers,
            'at_risk_customers': at_risk_customers
        }
        
        # Key insights
        key_insights = generate_key_insights(results_summary, segments_list)
        
        # Top-level summary (프론트엔드 summary 구조)
        top_summary = {
            'model': request.model,
            'total_ltv': _to_native_type(ltv_values.sum()),
            'avg_ltv': _to_native_type(ltv_values.mean()),
            'high_value_count': high_value_customers,
            'solve_time_ms': solve_time_ms
        }
        
        # Final response (프론트엔드 LTVResult 인터페이스에 정확히 맞춤)
        return {
            'success': True,
            'results': {
                'summary': results_summary,
                'customers': customers_list,
                'segments': segments_list,
                'rfm_summary': rfm_summary,
                'model_params': model_params
            },
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': top_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LTV analysis failed: {str(e)}")
