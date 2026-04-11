"""
Demand Elasticity Analysis FastAPI Endpoint
Price elasticity calculation and revenue optimization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar
from io import BytesIO
import base64
import warnings

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class ElasticityRequest(BaseModel):
    """Request model for Elasticity Analysis"""
    data: List[Dict[str, Any]]
    product_col: str
    price_col: str
    quantity_col: str
    optimize: bool = Field(default=True)


class ElasticityByProduct(BaseModel):
    """Elasticity results for a single product"""
    product_id: str
    elasticity: float
    r_squared: float
    demand_type: str
    interpretation: str
    avg_price: float
    avg_quantity: float
    total_revenue: float
    avg_revenue: float
    min_price: float
    max_price: float
    price_range: float
    observations: int


class OptimizationResult(BaseModel):
    """Price optimization result for a single product"""
    product_id: str
    current_price: float
    optimal_price: float
    price_change: float
    price_change_pct: float
    current_quantity: float
    optimal_quantity: float
    current_revenue: float
    optimal_revenue: float
    revenue_change: float
    revenue_change_pct: float
    recommendation: str
    action: str
    elasticity: float


class Metrics(BaseModel):
    """Overall metrics"""
    total_products: int
    elastic_products: int
    inelastic_products: int
    avg_elasticity: float
    median_elasticity: float
    total_revenue: float
    avg_revenue_per_product: float
    avg_r_squared: float


class KeyInsight(BaseModel):
    """Key insight information"""
    title: str
    description: str
    status: str  # "positive" | "neutral" | "warning"


class Summary(BaseModel):
    """Analysis summary"""
    analysis_type: str
    total_products: int
    avg_elasticity: float
    total_revenue: float


class ElasticityResponse(BaseModel):
    """Response model for Elasticity Analysis"""
    success: bool
    results: Dict[str, Any]
    visualizations: Dict[str, Optional[str]]
    key_insights: List[KeyInsight]
    summary: Summary


def calculate_elasticity(df: pd.DataFrame, product_col: str, price_col: str, quantity_col: str):
    """Calculate price elasticity for each product"""
    results = []
    product_elasticities = {}
    
    for product_id, group in df.groupby(product_col):
        if len(group) < 5:
            continue
        
        prices = group[price_col].values
        quantities = group[quantity_col].values
        revenue = prices * quantities
        
        log_price = np.log(prices + 1)
        log_quantity = np.log(quantities + 1)
        
        model = LinearRegression()
        model.fit(log_price.reshape(-1, 1), log_quantity)
        
        elasticity = model.coef_[0]
        r_squared = model.score(log_price.reshape(-1, 1), log_quantity)
        
        avg_price = prices.mean()
        avg_quantity = quantities.mean()
        total_revenue = revenue.sum()
        avg_revenue = revenue.mean()
        
        if elasticity < -1:
            demand_type = "Elastic"
            interpretation = "Demand is sensitive to price changes"
        elif elasticity > -1 and elasticity < 0:
            demand_type = "Inelastic"
            interpretation = "Demand is relatively insensitive to price changes"
        else:
            demand_type = "Unusual"
            interpretation = "Positive or zero elasticity (check data quality)"
        
        results.append({
            'product_id': str(product_id),
            'elasticity': float(elasticity),
            'r_squared': float(r_squared),
            'demand_type': demand_type,
            'interpretation': interpretation,
            'avg_price': float(avg_price),
            'avg_quantity': float(avg_quantity),
            'total_revenue': float(total_revenue),
            'avg_revenue': float(avg_revenue),
            'min_price': float(prices.min()),
            'max_price': float(prices.max()),
            'price_range': float(prices.max() - prices.min()),
            'observations': int(len(group))
        })
        
        product_elasticities[str(product_id)] = {
            'model': model,
            'elasticity': elasticity,
            'avg_price': avg_price,
            'avg_quantity': avg_quantity
        }
    
    return pd.DataFrame(results), product_elasticities


def optimize_prices(elasticity_df: pd.DataFrame, product_elasticities: dict, price_range_pct: float = 0.3):
    """Find optimal prices to maximize revenue"""
    optimization_results = []
    
    for _, row in elasticity_df.iterrows():
        product_id = row['product_id']
        if product_id not in product_elasticities:
            continue
        
        info = product_elasticities[product_id]
        current_price = info['avg_price']
        current_quantity = info['avg_quantity']
        elasticity = info['elasticity']
        
        def revenue_function(price):
            price_ratio = price / current_price
            quantity_ratio = price_ratio ** elasticity
            predicted_quantity = current_quantity * quantity_ratio
            return -(price * predicted_quantity)
        
        min_price = current_price * (1 - price_range_pct)
        max_price = current_price * (1 + price_range_pct)
        
        result = minimize_scalar(revenue_function, bounds=(min_price, max_price), method='bounded')
        
        optimal_price = result.x
        optimal_revenue = -result.fun
        
        price_ratio = optimal_price / current_price
        quantity_ratio = price_ratio ** elasticity
        optimal_quantity = current_quantity * quantity_ratio
        
        current_revenue = current_price * current_quantity
        revenue_change = optimal_revenue - current_revenue
        revenue_change_pct = (revenue_change / current_revenue) * 100
        price_change = optimal_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        if abs(price_change_pct) < 2:
            recommendation = "Keep current price"
            action = "No change needed"
        elif price_change_pct > 0:
            recommendation = f"Increase price by {price_change_pct:.1f}%"
            action = "Price increase"
        else:
            recommendation = f"Decrease price by {abs(price_change_pct):.1f}%"
            action = "Price decrease"
        
        optimization_results.append({
            'product_id': product_id,
            'current_price': float(current_price),
            'optimal_price': float(optimal_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'current_quantity': float(current_quantity),
            'optimal_quantity': float(optimal_quantity),
            'current_revenue': float(current_revenue),
            'optimal_revenue': float(optimal_revenue),
            'revenue_change': float(revenue_change),
            'revenue_change_pct': float(revenue_change_pct),
            'recommendation': recommendation,
            'action': action,
            'elasticity': float(elasticity)
        })
    
    return pd.DataFrame(optimization_results)


def generate_visualizations(elasticity_df: pd.DataFrame, optimization_df: Optional[pd.DataFrame], df: pd.DataFrame, 
                           product_col: str, price_col: str, quantity_col: str):
    """Generate all visualizations"""
    visualizations = {}
    
    # 1. Elasticity Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    elasticities = elasticity_df['elasticity']
    ax.hist(elasticities, bins=30, edgecolor='black', alpha=0.7, color='#4A90E2')
    ax.axvline(elasticities.mean(), color='red', linestyle='--', label=f'Mean: {elasticities.mean():.2f}')
    ax.axvline(-1, color='orange', linestyle=':', label='Elastic/Inelastic threshold (-1)')
    ax.set_xlabel('Price Elasticity', fontsize=11)
    ax.set_ylabel('Number of Products', fontsize=11)
    ax.set_title('Distribution of Price Elasticities', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['elasticity_distribution'] = fig_to_base64(fig)
    
    # 2. Price vs Quantity Scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    top_products = elasticity_df.nlargest(10, 'total_revenue')['product_id']
    for product_id in top_products:
        product_data = df[df[product_col] == product_id]
        ax.scatter(product_data[price_col], product_data[quantity_col], alpha=0.6, s=50, label=str(product_id)[:15])
    ax.set_xlabel('Price ($)', fontsize=11)
    ax.set_ylabel('Quantity Sold', fontsize=11)
    ax.set_title('Price vs Quantity (Top 10 Products by Revenue)', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['price_quantity_scatter'] = fig_to_base64(fig)
    
    # 3. Revenue by Product
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = elasticity_df.nlargest(20, 'total_revenue')
    y_pos = np.arange(len(top_20))
    ax.barh(y_pos, top_20['total_revenue'].values, color='#4A90E2', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(x)[:20] for x in top_20['product_id'].values], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Total Revenue ($)', fontsize=11)
    ax.set_title('Top 20 Products by Revenue', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(top_20['total_revenue'].values):
        ax.text(v, i, f' ${v:,.0f}', va='center', fontsize=8)
    plt.tight_layout()
    visualizations['revenue_by_product'] = fig_to_base64(fig)
    
    # 4. Demand Type Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    demand_counts = elasticity_df['demand_type'].value_counts()
    demand_revenue = elasticity_df.groupby('demand_type')['total_revenue'].sum()
    colors = ['#4A90E2', '#E57373', '#9E9E9E']
    ax1.pie(demand_counts.values, labels=demand_counts.index, autopct='%1.1f%%',
           colors=colors[:len(demand_counts)], startangle=90)
    ax1.set_title('Products by Demand Type', fontsize=12, fontweight='bold')
    ax2.bar(demand_revenue.index, demand_revenue.values, color=colors[:len(demand_revenue)], edgecolor='black')
    ax2.set_ylabel('Total Revenue ($)', fontsize=11)
    ax2.set_title('Revenue by Demand Type', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(demand_revenue.values):
        ax2.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    visualizations['demand_type_distribution'] = fig_to_base64(fig)
    
    # 5. Optimization Comparison
    if optimization_df is not None and len(optimization_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_20 = optimization_df.nlargest(20, 'revenue_change')
        x = np.arange(len(top_20))
        width = 0.35
        ax.barh(x - width/2, top_20['current_price'], width, label='Current Price', color='#9E9E9E', edgecolor='black')
        ax.barh(x + width/2, top_20['optimal_price'], width, label='Optimal Price', color='#4A90E2', edgecolor='black')
        ax.set_yticks(x)
        ax.set_yticklabels([str(p)[:20] for p in top_20['product_id'].values], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Price ($)', fontsize=11)
        ax.set_title('Current vs Optimal Price (Top 20 by Revenue Gain)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        visualizations['optimization_comparison'] = fig_to_base64(fig)
        
        # 6. Price Recommendations
        fig, ax = plt.subplots(figsize=(10, 6))
        increase = optimization_df[optimization_df['action'] == 'Price increase']
        decrease = optimization_df[optimization_df['action'] == 'Price decrease']
        no_change = optimization_df[optimization_df['action'] == 'No change needed']
        categories = ['Increase\nPrice', 'Decrease\nPrice', 'Keep\nCurrent']
        counts = [len(increase), len(decrease), len(no_change)]
        colors_map = ['#4CAF50', '#FF5252', '#9E9E9E']
        bars = ax.bar(categories, counts, color=colors_map, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Number of Products', fontsize=11)
        ax.set_title('Price Optimization Recommendations', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        rev_increase = increase['revenue_change'].sum() if len(increase) > 0 else 0
        rev_decrease = decrease['revenue_change'].sum() if len(decrease) > 0 else 0
        total_gain = rev_increase + rev_decrease
        ax.text(0.5, 0.95, f'Total Potential Revenue Gain: ${total_gain:,.2f}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10, fontweight='bold')
        plt.tight_layout()
        visualizations['price_recommendations'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(elasticity_df: pd.DataFrame, optimization_df: Optional[pd.DataFrame], metrics: dict):
    """Generate key insights"""
    insights = []
    
    avg_elasticity = metrics['avg_elasticity']
    if avg_elasticity < -1:
        insights.append({
            'title': 'Overall Elastic Demand',
            'description': f"Average price elasticity is {avg_elasticity:.2f}, indicating demand is generally sensitive to price changes. Price reductions could significantly increase sales volume.",
            'status': 'positive'
        })
    else:
        insights.append({
            'title': 'Overall Inelastic Demand',
            'description': f"Average price elasticity is {avg_elasticity:.2f}, indicating demand is relatively insensitive to price changes. Price increases may not significantly hurt sales volume.",
            'status': 'positive'
        })
    
    elastic_pct = (metrics['elastic_products'] / metrics['total_products']) * 100
    insights.append({
        'title': 'Product Portfolio Composition',
        'description': f"{metrics['elastic_products']} products ({elastic_pct:.1f}%) have elastic demand, while {metrics['inelastic_products']} products have inelastic demand. Different pricing strategies should be applied to each category.",
        'status': 'neutral'
    })
    
    if optimization_df is not None and len(optimization_df) > 0:
        total_gain = optimization_df['revenue_change'].sum()
        products_to_increase = (optimization_df['action'] == 'Price increase').sum()
        products_to_decrease = (optimization_df['action'] == 'Price decrease').sum()
        
        if total_gain > 0:
            insights.append({
                'title': 'Significant Revenue Opportunity',
                'description': f"Price optimization could increase total revenue by ${total_gain:,.2f}. {products_to_increase} products should increase price, {products_to_decrease} should decrease price.",
                'status': 'positive'
            })
        else:
            insights.append({
                'title': 'Well-Optimized Pricing',
                'description': f"Current pricing is near-optimal. Only minor adjustments recommended with estimated revenue change of ${total_gain:,.2f}.",
                'status': 'positive'
            })
    
    avg_r_squared = metrics['avg_r_squared']
    if avg_r_squared > 0.7:
        insights.append({
            'title': 'High Model Reliability',
            'description': f"Average R² of {avg_r_squared:.2f} indicates strong predictive power. Elasticity estimates are reliable for decision-making.",
            'status': 'positive'
        })
    elif avg_r_squared > 0.4:
        insights.append({
            'title': 'Moderate Model Reliability',
            'description': f"Average R² of {avg_r_squared:.2f} indicates moderate fit. Recommendations should be validated with A/B testing.",
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Low Model Reliability',
            'description': f"Average R² of {avg_r_squared:.2f} indicates weak relationship between price and quantity. Other factors may be more important than price.",
            'status': 'warning'
        })
    
    return insights


@router.post("/demand-elasticity")
async def analyze_elasticity(request: ElasticityRequest):
    """
    Demand Elasticity Analysis & Price Optimization
    
    Calculates price elasticity and recommends optimal prices
    """
    try:
        # Validate input
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 10:
            raise HTTPException(400, "Insufficient data (need at least 10 observations)")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        df[request.price_col] = pd.to_numeric(df[request.price_col])
        df[request.quantity_col] = pd.to_numeric(df[request.quantity_col])
        
        # Calculate elasticity
        elasticity_df, product_elasticities = calculate_elasticity(
            df, request.product_col, request.price_col, request.quantity_col
        )
        
        if len(elasticity_df) == 0:
            raise HTTPException(400, "Insufficient data for elasticity calculation")
        
        # Optimize prices
        optimization_df = None
        if request.optimize:
            optimization_df = optimize_prices(elasticity_df, product_elasticities)
        
        # Calculate metrics
        total_products = len(elasticity_df)
        elastic_count = (elasticity_df['elasticity'] < -1).sum()
        inelastic_count = ((elasticity_df['elasticity'] >= -1) & (elasticity_df['elasticity'] < 0)).sum()
        
        metrics = {
            'total_products': int(total_products),
            'elastic_products': int(elastic_count),
            'inelastic_products': int(inelastic_count),
            'avg_elasticity': float(elasticity_df['elasticity'].mean()),
            'median_elasticity': float(elasticity_df['elasticity'].median()),
            'total_revenue': float(elasticity_df['total_revenue'].sum()),
            'avg_revenue_per_product': float(elasticity_df['avg_revenue'].mean()),
            'avg_r_squared': float(elasticity_df['r_squared'].mean())
        }
        
        # Generate visualizations
        visualizations = generate_visualizations(
            elasticity_df, optimization_df, df, 
            request.product_col, request.price_col, request.quantity_col
        )
        
        # Generate insights
        insights = generate_insights(elasticity_df, optimization_df, metrics)
        
        # Prepare response
        return ElasticityResponse(
            success=True,
            results={
                'elasticity_by_product': elasticity_df.to_dict('records'),
                'optimization_results': optimization_df.to_dict('records') if optimization_df is not None else [],
                'metrics': metrics
            },
            visualizations=visualizations,
            key_insights=insights,
            summary=Summary(
                analysis_type='demand_elasticity',
                total_products=metrics['total_products'],
                avg_elasticity=metrics['avg_elasticity'],
                total_revenue=metrics['total_revenue']
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
