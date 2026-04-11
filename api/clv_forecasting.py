"""
Customer Lifetime Value Forecasting - Scikit-learn Version
Uses Random Forest regression instead of lifetimes library
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from io import BytesIO
import base64
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class CLVRequest(BaseModel):
    """Request model for CLV Forecasting"""
    data: List[Dict[str, Any]]
    customer_col: str
    date_col: str
    revenue_col: str
    forecast_months: int = Field(default=12, ge=1, le=36)


def calculate_rfm_features(df: pd.DataFrame, customer_col: str, date_col: str, revenue_col: str):
    """Calculate RFM features for each customer"""
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    
    # Group by customer
    customer_features = df.groupby(customer_col).agg({
        date_col: ['min', 'max', 'count'],
        revenue_col: ['sum', 'mean', 'std', 'min', 'max']
    }).reset_index()
    
    customer_features.columns = [
        'customer_id', 
        'first_purchase_date', 'last_purchase_date', 'frequency',
        'total_revenue', 'avg_order_value', 'std_order_value', 'min_order_value', 'max_order_value'
    ]
    
    # Calculate additional features
    customer_features['recency_days'] = (max_date - customer_features['last_purchase_date']).dt.days
    customer_features['customer_age_days'] = (max_date - customer_features['first_purchase_date']).dt.days
    customer_features['avg_days_between_purchases'] = customer_features['customer_age_days'] / customer_features['frequency'].clip(lower=1)
    customer_features['revenue_per_day'] = customer_features['total_revenue'] / customer_features['customer_age_days'].clip(lower=1)
    
    # Fill NaN in std with 0
    customer_features['std_order_value'] = customer_features['std_order_value'].fillna(0)
    
    return customer_features


def train_clv_model(features_df: pd.DataFrame, forecast_months: int):
    """Train Random Forest model to predict CLV"""
    # Feature columns
    feature_cols = [
        'frequency', 'avg_order_value', 'std_order_value',
        'recency_days', 'customer_age_days', 'avg_days_between_purchases',
        'revenue_per_day', 'min_order_value', 'max_order_value'
    ]
    
    # Target: total revenue (as proxy for CLV)
    X = features_df[feature_cols].values
    y = features_df['total_revenue'].values
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Predict future CLV (scaled by forecast period)
    predicted_historical = model.predict(X)
    
    # Scale predictions based on forecast period
    # Estimate: CLV = historical_pattern × (forecast_months / historical_months)
    historical_months = features_df['customer_age_days'] / 30
    scale_factor = forecast_months / historical_months.clip(lower=1)
    predicted_clv = predicted_historical * scale_factor
    
    return model, predicted_clv, feature_cols


def segment_customers(features_df: pd.DataFrame, predicted_clv: np.ndarray):
    """Segment customers based on predicted CLV and activity"""
    features_df['predicted_clv'] = predicted_clv
    
    # Calculate activity score (0-1)
    max_recency = features_df['recency_days'].max()
    features_df['activity_score'] = 1 - (features_df['recency_days'] / max_recency)
    
    # Segmentation
    clv_75th = np.percentile(predicted_clv, 75)
    clv_50th = np.percentile(predicted_clv, 50)
    
    def assign_segment(row):
        if row['predicted_clv'] > clv_75th and row['activity_score'] > 0.5:
            return "High Value"
        elif row['predicted_clv'] > clv_50th and row['activity_score'] > 0.5:
            return "Medium Value"
        elif row['activity_score'] < 0.2:
            return "At Risk"
        elif row['frequency'] == 1:
            return "New Customer"
        else:
            return "Low Value"
    
    features_df['customer_segment'] = features_df.apply(assign_segment, axis=1)
    
    return features_df


def generate_visualizations(results_df: pd.DataFrame, forecast_months: int):
    """Generate visualizations"""
    visualizations = {}
    
    # 1. CLV Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results_df['predicted_clv'], bins=50, color='#4A90E2', edgecolor='black', alpha=0.7)
    ax.axvline(results_df['predicted_clv'].mean(), color='#E74C3C', linestyle='--', linewidth=2, 
              label=f'Mean: ${results_df["predicted_clv"].mean():,.2f}')
    ax.axvline(results_df['predicted_clv'].median(), color='#27AE60', linestyle='--', linewidth=2, 
              label=f'Median: ${results_df["predicted_clv"].median():,.2f}')
    ax.set_xlabel(f'Predicted CLV ({forecast_months} months)', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_title('Customer Lifetime Value Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    visualizations['clv_distribution'] = fig_to_base64(fig)
    
    # 2. Frequency vs CLV
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(results_df['frequency'], results_df['avg_order_value'],
                        c=results_df['predicted_clv'], cmap='YlOrRd', s=80, alpha=0.6, edgecolor='black')
    ax.set_xlabel('Purchase Frequency', fontsize=11)
    ax.set_ylabel('Average Order Value ($)', fontsize=11)
    ax.set_title('Customer Frequency vs Order Value', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Predicted CLV ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['frequency_value_scatter'] = fig_to_base64(fig)
    
    # 3. Customer Segmentation
    fig, ax = plt.subplots(figsize=(10, 6))
    segment_counts = results_df['customer_segment'].value_counts()
    colors_map = {'High Value': '#27AE60', 'Medium Value': '#4A90E2',
                  'Low Value': '#F39C12', 'At Risk': '#E74C3C', 'New Customer': '#95A5A6'}
    bar_colors = [colors_map.get(seg, '#9E9E9E') for seg in segment_counts.index]
    bars = ax.bar(range(len(segment_counts)), segment_counts.values,
                  color=bar_colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(segment_counts)))
    ax.set_xticklabels(segment_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_title('Customer Segmentation', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, count) in enumerate(zip(bars, segment_counts.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, 
               f'{int(count)}\n({count/len(results_df)*100:.1f}%)',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    visualizations['customer_segmentation'] = fig_to_base64(fig)
    
    # 4. Top Customers
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = results_df.nlargest(20, 'predicted_clv')
    y_pos = np.arange(len(top_20))
    ax.barh(y_pos, top_20['predicted_clv'], color='#4A90E2', edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(c)[:20] for c in top_20['customer_id'].values], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f'Predicted CLV ({forecast_months} months) ($)', fontsize=11)
    ax.set_title('Top 20 Customers by Predicted CLV', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax.text(row['predicted_clv'], i, f" ${row['predicted_clv']:,.0f}",
               va='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    visualizations['top_customers'] = fig_to_base64(fig)
    
    # 5. Activity Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results_df['activity_score'], bins=30, color='#27AE60', edgecolor='black', alpha=0.7)
    ax.axvline(results_df['activity_score'].mean(), color='#E74C3C', linestyle='--', linewidth=2,
              label=f'Mean: {results_df["activity_score"].mean():.2f}')
    ax.set_xlabel('Activity Score (0=Inactive, 1=Active)', fontsize=11)
    ax.set_ylabel('Number of Customers', fontsize=11)
    ax.set_title('Customer Activity Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    visualizations['activity_distribution'] = fig_to_base64(fig)
    
    # 6. CLV by Segment
    fig, ax = plt.subplots(figsize=(10, 6))
    segment_clv = results_df.groupby('customer_segment').agg({
        'predicted_clv': 'sum',
        'customer_id': 'count'
    }).sort_values('predicted_clv', ascending=False)
    
    x_pos = np.arange(len(segment_clv))
    bars = ax.bar(x_pos, segment_clv['predicted_clv'],
                  color=[colors_map.get(seg, '#9E9E9E') for seg in segment_clv.index],
                  edgecolor='black', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(segment_clv.index, rotation=45, ha='right')
    ax.set_ylabel('Total Predicted CLV ($)', fontsize=11)
    ax.set_title('Total CLV by Customer Segment', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, (seg, row)) in enumerate(zip(bars, segment_clv.iterrows())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:,.0f}\n({int(row["customer_id"])} customers)',
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.tight_layout()
    visualizations['clv_by_segment'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(results_df: pd.DataFrame, metrics: dict, forecast_months: int):
    """Generate key insights"""
    insights = []
    
    high_value_pct = (metrics['high_value_customers'] / metrics['total_customers']) * 100
    insights.append({
        'title': 'High-Value Customer Base',
        'description': f"{metrics['high_value_customers']} customers ({high_value_pct:.1f}%) are classified as high-value. Focus retention efforts on this segment.",
        'status': 'positive' if high_value_pct > 20 else 'neutral'
    })
    
    at_risk_pct = (metrics['at_risk_customers'] / metrics['total_customers']) * 100
    if at_risk_pct > 15:
        insights.append({
            'title': 'Customer Churn Risk',
            'description': f"{metrics['at_risk_customers']} customers ({at_risk_pct:.1f}%) are at risk. Implement re-engagement campaigns.",
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Low Churn Risk',
            'description': f"Only {metrics['at_risk_customers']} customers ({at_risk_pct:.1f}%) are at risk.",
            'status': 'positive'
        })
    
    top_20_pct = results_df.nlargest(int(len(results_df) * 0.2), 'predicted_clv')['predicted_clv'].sum()
    revenue_concentration = (top_20_pct / metrics['total_predicted_clv']) * 100
    insights.append({
        'title': 'Revenue Concentration',
        'description': f"Top 20% of customers account for {revenue_concentration:.1f}% of predicted CLV.",
        'status': 'warning' if revenue_concentration > 70 else 'positive'
    })
    
    insights.append({
        'title': 'Model Type',
        'description': f"Machine learning model (Random Forest) used for prediction. Model trained on {len(results_df)} customers with historical behavior patterns.",
        'status': 'neutral'
    })
    
    insights.append({
        'title': 'CLV Forecast Summary',
        'description': f"Over {forecast_months} months, average predicted CLV is ${metrics['avg_clv']:,.2f} per customer with total predicted revenue of ${metrics['total_predicted_clv']:,.2f}.",
        'status': 'neutral'
    })
    
    return insights


@router.post("/clv-forecast")
async def forecast_clv(request: CLVRequest):
    """
    Customer Lifetime Value Forecasting - Scikit-learn Version
    
    Uses Random Forest machine learning instead of probabilistic models
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 20:
            raise HTTPException(400, "Insufficient data (need at least 20 transactions)")
        
        df = pd.DataFrame(request.data)
        
        required_cols = [request.customer_col, request.date_col, request.revenue_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns: {missing}")
        
        df[request.revenue_col] = pd.to_numeric(df[request.revenue_col])
        df = df[df[request.revenue_col] > 0]
        
        # Calculate RFM features
        features_df = calculate_rfm_features(df, request.customer_col, request.date_col, request.revenue_col)
        
        if len(features_df) < 5:
            raise HTTPException(400, "Insufficient customers for modeling (need at least 5)")
        
        # Train model and predict CLV
        model, predicted_clv, feature_cols = train_clv_model(features_df, request.forecast_months)
        
        # Segment customers
        results_df = segment_customers(features_df, predicted_clv)
        
        # Calculate metrics
        total_customers = len(results_df)
        high_value = int((results_df['customer_segment'] == 'High Value').sum())
        at_risk = int((results_df['customer_segment'] == 'At Risk').sum())
        
        metrics = {
            'total_customers': int(total_customers),
            'total_revenue': float(df[request.revenue_col].sum()),
            'avg_clv': float(results_df['predicted_clv'].mean()),
            'median_clv': float(results_df['predicted_clv'].median()),
            'total_predicted_clv': float(results_df['predicted_clv'].sum()),
            'avg_frequency': float(results_df['frequency'].mean()),
            'avg_monetary_value': float(results_df['avg_order_value'].mean()),
            'model_quality_score': float(model.score(
                results_df[feature_cols].values, 
                results_df['total_revenue'].values
            )),
            'high_value_customers': high_value,
            'at_risk_customers': at_risk
        }
        
        # Generate visualizations
        visualizations = generate_visualizations(results_df, request.forecast_months)
        
        # Generate insights
        insights = generate_insights(results_df, metrics, request.forecast_months)
        
        # Prepare output - select key columns
        output_cols = [
            'customer_id', 'frequency', 'recency_days', 'customer_age_days',
            'avg_order_value', 'total_revenue', 'predicted_clv',
            'activity_score', 'customer_segment'
        ]
        customer_data = results_df[output_cols].to_dict('records')
        
        # Convert to JSON using pandas (handles numpy types)
        import json
        customer_clv_json = json.loads(pd.DataFrame(customer_data).to_json(orient='records'))
        
        response_data = {
            'success': True,
            'results': {
                'customer_clv': customer_clv_json,
                'metrics': metrics
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'clv_forecast_ml',
                'total_customers': int(total_customers),
                'avg_clv': float(results_df['predicted_clv'].mean()),
                'total_predicted_clv': float(results_df['predicted_clv'].sum()),
                'forecast_period_months': int(request.forecast_months)
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")
