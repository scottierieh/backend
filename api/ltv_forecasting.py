"""
LTV Forecasting API Router
FastAPI endpoints for customer lifetime value prediction
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifetimes.utils import summary_data_from_transaction_data
import warnings
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# Pydantic models for request/response
class TransactionRecord(BaseModel):
    customer_id: str
    transaction_date: str
    amount: float

class LTVAnalysisRequest(BaseModel):
    data: List[dict]
    customer_col: str = Field(default="customer_id", description="Column name for customer ID")
    date_col: str = Field(default="transaction_date", description="Column name for transaction date")
    value_col: str = Field(default="amount", description="Column name for transaction amount")
    forecast_months: int = Field(default=12, ge=1, le=60, description="Forecast horizon in months")


class LTVForecaster:
    """
    Customer Lifetime Value Forecasting using BG/NBD + Gamma-Gamma models
    """
    
    def __init__(self, data, customer_col, date_col, value_col):
        """
        Initialize with transaction data
        
        Args:
            data: List of transaction records
            customer_col: Column name for customer ID
            date_col: Column name for transaction date
            value_col: Column name for transaction value/amount
        """
        self.customer_col = customer_col
        self.date_col = date_col
        self.value_col = value_col
        
        # Convert to DataFrame
        self.df = pd.DataFrame(data)
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df[value_col] = pd.to_numeric(self.df[value_col])
        
        # Analysis date (most recent transaction date)
        self.analysis_date = self.df[date_col].max()
        
        # Initialize models
        self.bgf = None
        self.ggf = None
        self.summary = None
        
    def prepare_data(self):
        """Prepare RFM summary data for lifetimes models"""
        self.summary = summary_data_from_transaction_data(
            self.df,
            customer_id_col=self.customer_col,
            datetime_col=self.date_col,
            monetary_value_col=self.value_col,
            observation_period_end=self.analysis_date
        )
        
        # Remove customers with zero frequency for Gamma-Gamma
        self.summary_with_value = self.summary[self.summary['frequency'] > 0].copy()
        
        return self.summary
    
    def fit_models(self):
        """Fit BG/NBD and Gamma-Gamma models"""
        if self.summary is None:
            self.prepare_data()
        
        # BG/NBD model for predicting purchase frequency
        self.bgf = BetaGeoFitter(penalizer_coef=0.1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.bgf.fit(
                    self.summary['frequency'],
                    self.summary['recency'],
                    self.summary['T']
                )
        except Exception as e:
            self.bgf = BetaGeoFitter(penalizer_coef=1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.bgf.fit(
                    self.summary['frequency'],
                    self.summary['recency'],
                    self.summary['T']
                )
        
        # Gamma-Gamma model for predicting average order value
        self.ggf = GammaGammaFitter(penalizer_coef=0.1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.ggf.fit(
                    self.summary_with_value['frequency'],
                    self.summary_with_value['monetary_value']
                )
        except Exception as e:
            self.ggf = GammaGammaFitter(penalizer_coef=1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.ggf.fit(
                    self.summary_with_value['frequency'],
                    self.summary_with_value['monetary_value']
                )
        
        return self.bgf, self.ggf
    
    def predict_ltv(self, time_months=12):
        """
        Predict customer lifetime value
        
        Args:
            time_months: Prediction horizon in months (default: 12)
        
        Returns:
            DataFrame with LTV predictions and related metrics
        """
        if self.bgf is None or self.ggf is None:
            self.fit_models()
        
        time_days = time_months * 30  # Approximate days
        
        # Predict expected purchases
        self.summary['predicted_purchases'] = self.bgf.predict(
            time_days,
            self.summary['frequency'],
            self.summary['recency'],
            self.summary['T']
        )
        
        # Predict probability alive
        self.summary['prob_alive'] = self.bgf.conditional_probability_alive(
            self.summary['frequency'],
            self.summary['recency'],
            self.summary['T']
        )
        
        # Predict average transaction value (only for repeat customers)
        self.summary['predicted_avg_value'] = 0
        self.summary.loc[self.summary['frequency'] > 0, 'predicted_avg_value'] = \
            self.ggf.conditional_expected_average_profit(
                self.summary_with_value['frequency'],
                self.summary_with_value['monetary_value']
            )
        
        # Calculate LTV
        self.summary['predicted_ltv'] = (
            self.summary['predicted_purchases'] * 
            self.summary['predicted_avg_value']
        )
        
        # Customer value tier based on LTV
        self.summary['value_tier'] = pd.qcut(
            self.summary['predicted_ltv'],
            q=4,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
            duplicates='drop'
        )
        
        return self.summary
    
    def get_metrics(self):
        """Calculate overall metrics"""
        if self.summary is None:
            return {}
        
        total_customers = len(self.summary)
        total_historical_value = self.df[self.value_col].sum()
        total_predicted_ltv = self.summary['predicted_ltv'].sum()
        avg_ltv = self.summary['predicted_ltv'].mean()
        median_ltv = self.summary['predicted_ltv'].median()
        
        # Model fit metrics
        bgf_score = None
        if self.bgf is not None:
            try:
                if hasattr(self.bgf, 'params_'):
                    bgf_score = sum(self.bgf.params_.values()) / len(self.bgf.params_)
            except:
                bgf_score = None
        
        return {
            'total_customers': int(total_customers),
            'total_historical_value': float(total_historical_value),
            'total_predicted_ltv': float(total_predicted_ltv),
            'avg_ltv': float(avg_ltv),
            'median_ltv': float(median_ltv),
            'avg_historical_value': float(total_historical_value / total_customers),
            'model_quality_score': float(bgf_score) if bgf_score else None
        }
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        visualizations = {}
        
        visualizations['ltv_distribution'] = self._plot_ltv_distribution()
        visualizations['value_tiers'] = self._plot_value_tiers()
        visualizations['probability_matrix'] = self._plot_probability_matrix()
        visualizations['top_customers'] = self._plot_top_customers()
        
        return visualizations
    
    def _plot_ltv_distribution(self):
        """Plot LTV distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.summary['predicted_ltv'], bins=50, color='#4A90E2', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Predicted LTV ($)', fontsize=11)
        ax.set_ylabel('Number of Customers', fontsize=11)
        ax.set_title('Distribution of Predicted Customer Lifetime Value', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_value_tiers(self):
        """Plot customer value tiers"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tier_counts = self.summary['value_tier'].value_counts().sort_index()
        colors = ['#CD7F32', '#C0C0C0', '#FFD700', '#E5E4E2']
        
        ax.bar(tier_counts.index, tier_counts.values, color=colors, edgecolor='black')
        ax.set_ylabel('Number of Customers', fontsize=11)
        ax.set_title('Customer Distribution by Value Tier', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(tier_counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_probability_matrix(self):
        """Plot probability alive matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        plot_probability_alive_matrix(self.bgf, ax=ax)
        ax.set_title('Customer Probability of Being Alive', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_top_customers(self):
        """Plot top 20 customers by LTV"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_20 = self.summary.nlargest(20, 'predicted_ltv')
        
        y_pos = np.arange(len(top_20))
        ax.barh(y_pos, top_20['predicted_ltv'].values, color='#4A90E2', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(x)[:20] for x in top_20.index], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Predicted LTV ($)', fontsize=11)
        ax.set_title('Top 20 Customers by Predicted LTV', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(top_20['predicted_ltv'].values):
            ax.text(v, i, f' ${v:.2f}', va='center', fontsize=8)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def get_key_insights(self):
        """Generate key insights from the analysis"""
        insights = []
        
        metrics = self.get_metrics()
        
        insights.append({
            'title': 'Total Predicted Customer Lifetime Value',
            'description': f"The total predicted LTV across all {metrics['total_customers']} customers is ${metrics['total_predicted_ltv']:,.2f} over the forecast period. This represents the expected future revenue from your current customer base.",
            'status': 'positive'
        })
        
        avg_ltv = metrics['avg_ltv']
        median_ltv = metrics['median_ltv']
        if avg_ltv > median_ltv * 1.5:
            insights.append({
                'title': 'High-Value Customer Concentration',
                'description': f"Average LTV (${avg_ltv:.2f}) significantly exceeds median LTV (${median_ltv:.2f}), indicating a small group of high-value customers drives most revenue. Focus retention efforts on top customers.",
                'status': 'warning'
            })
        else:
            insights.append({
                'title': 'Balanced Customer Value Distribution',
                'description': f"Average LTV (${avg_ltv:.2f}) is close to median LTV (${median_ltv:.2f}), indicating a healthy, balanced customer base without excessive concentration risk.",
                'status': 'positive'
            })
        
        tier_counts = self.summary['value_tier'].value_counts()
        platinum_count = tier_counts.get('Platinum', 0)
        platinum_pct = (platinum_count / len(self.summary)) * 100
        
        if platinum_pct < 20:
            insights.append({
                'title': 'Limited High-Value Customer Base',
                'description': f"Only {platinum_count} customers ({platinum_pct:.1f}%) are in the Platinum tier. Consider strategies to upgrade Gold and Silver customers to maximize LTV.",
                'status': 'warning'
            })
        else:
            insights.append({
                'title': 'Strong High-Value Customer Base',
                'description': f"{platinum_count} customers ({platinum_pct:.1f}%) are in the Platinum tier, representing a healthy proportion of high-value customers.",
                'status': 'positive'
            })
        
        low_prob = (self.summary['prob_alive'] < 0.3).sum()
        if low_prob > len(self.summary) * 0.2:
            insights.append({
                'title': 'Significant Customer Churn Risk',
                'description': f"{low_prob} customers ({low_prob/len(self.summary)*100:.1f}%) have less than 30% probability of being active. Immediate re-engagement campaigns are recommended.",
                'status': 'warning'
            })
        
        return insights


@router.post("/ltv")
async def forecast_ltv(request: LTVAnalysisRequest):
    """
    Perform LTV forecasting analysis on customer transaction data
    
    This endpoint uses BG/NBD and Gamma-Gamma models to predict:
    - Customer lifetime value over a specified forecast period
    - Purchase probability and frequency
    - Average transaction values
    - Customer value tiers
    """
    try:
        # Validate data
        if not request.data or len(request.data) == 0:
            raise HTTPException(status_code=400, detail="No transaction data provided")
        
        # Initialize forecaster
        forecaster = LTVForecaster(
            request.data,
            request.customer_col,
            request.date_col,
            request.value_col
        )
        
        # Prepare data
        forecaster.prepare_data()
        
        # Fit models
        forecaster.fit_models()
        
        # Predict LTV
        ltv_results = forecaster.predict_ltv(time_months=request.forecast_months)
        
        # Get metrics
        metrics = forecaster.get_metrics()
        
        # Generate visualizations
        visualizations = forecaster.generate_visualizations()
        
        # Get insights
        insights = forecaster.get_key_insights()
        
        # Prepare results
        customer_predictions = ltv_results.reset_index()
        customer_predictions.columns = [request.customer_col if c == 'index' else c for c in customer_predictions.columns]
        
        # Convert float columns to integers (round first)
        float_cols = customer_predictions.select_dtypes(include=['float64', 'float32']).columns
        for col in float_cols:
            if col != 'prob_alive':  # Keep probability as float
                customer_predictions[col] = customer_predictions[col].round(0).astype(int)
        
        predictions_records = customer_predictions.to_dict('records')
        
        # Value tier summary
        tier_summary = ltv_results.groupby('value_tier').agg({
            'predicted_ltv': ['count', 'sum', 'mean'],
            'prob_alive': 'mean',
            'predicted_purchases': 'mean'
        })
        
        tier_summary.columns = ['_'.join(col).strip() for col in tier_summary.columns.values]
        tier_summary = tier_summary.reset_index()
        
        # Round integer columns
        tier_summary['predicted_ltv_count'] = tier_summary['predicted_ltv_count'].astype(int)
        tier_summary['predicted_ltv_sum'] = tier_summary['predicted_ltv_sum'].round(0).astype(int)
        tier_summary['predicted_ltv_mean'] = tier_summary['predicted_ltv_mean'].round(0).astype(int)
        tier_summary['predicted_purchases_mean'] = tier_summary['predicted_purchases_mean'].round(2)
        tier_summary['prob_alive_mean'] = tier_summary['prob_alive_mean'].round(2)
        
        tier_summary['ltv_percentage'] = (
            tier_summary['predicted_ltv_sum'] / tier_summary['predicted_ltv_sum'].sum() * 100
        ).round(1)
        
        tier_records = tier_summary.to_dict('records')
        
        results = {
            'success': True,
            'results': {
                'customer_predictions': predictions_records,
                'tier_summary': tier_records,
                'metrics': metrics
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'ltv_forecasting',
                'total_customers': metrics['total_customers'],
                'forecast_months': request.forecast_months,
                'total_predicted_ltv': metrics['total_predicted_ltv'],
                'avg_ltv': metrics['avg_ltv']
            }
        }
        
        # Use custom JSON response to handle numpy types
        return JSONResponse(
            content=json.loads(json.dumps(results, cls=NumpyEncoder))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LTV analysis failed: {str(e)}")


@router.get("/ltv/health")
async def health_check():
    """Health check endpoint for LTV forecasting service"""
    return {
        "status": "healthy",
        "service": "ltv_forecasting",
        "models": ["BG/NBD", "Gamma-Gamma"]
    }
