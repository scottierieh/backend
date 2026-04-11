"""
Promotion Optimization FastAPI Endpoint
Uplift modeling and ROI optimization for marketing campaigns
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from io import BytesIO
import base64
import warnings
import json

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class PromotionRequest(BaseModel):
    """Request model for Promotion Optimization"""
    data: List[Dict[str, Any]]
    promotion_col: str
    conversion_col: str
    revenue_col: Optional[str] = None
    cost_per_promotion: float = Field(default=1.0, gt=0)


def calculate_promotion_metrics(df: pd.DataFrame, promotion_col: str, conversion_col: str, revenue_col: Optional[str] = None):
    """Calculate basic promotion performance metrics"""
    
    # Group by promotion status
    promo_group = df[df[promotion_col] == 1]
    control_group = df[df[promotion_col] == 0]
    
    # Conversion rates
    promo_conv_rate = promo_group[conversion_col].mean()
    control_conv_rate = control_group[conversion_col].mean()
    
    # Uplift (incremental conversion)
    uplift = promo_conv_rate - control_conv_rate
    uplift_pct = (uplift / control_conv_rate * 100) if control_conv_rate > 0 else 0
    
    # Statistical significance test
    from scipy.stats import chi2_contingency
    contingency_table = pd.crosstab(df[promotion_col], df[conversion_col])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    is_significant = p_value < 0.05
    
    metrics = {
        'total_customers': len(df),
        'promoted_customers': len(promo_group),
        'control_customers': len(control_group),
        'promo_conversion_rate': float(promo_conv_rate),
        'control_conversion_rate': float(control_conv_rate),
        'uplift': float(uplift),
        'uplift_percentage': float(uplift_pct),
        'p_value': float(p_value),
        'is_significant': bool(is_significant)
    }
    
    # Revenue metrics if available
    if revenue_col:
        promo_revenue = promo_group[revenue_col].sum()
        control_revenue = control_group[revenue_col].sum()
        promo_avg_revenue = promo_group[revenue_col].mean()
        control_avg_revenue = control_group[revenue_col].mean()
        
        metrics.update({
            'promo_total_revenue': float(promo_revenue),
            'control_total_revenue': float(control_revenue),
            'promo_avg_revenue': float(promo_avg_revenue),
            'control_avg_revenue': float(control_avg_revenue),
            'revenue_uplift': float(promo_avg_revenue - control_avg_revenue)
        })
    
    return metrics


def train_uplift_model(df: pd.DataFrame, promotion_col: str, conversion_col: str):
    """Train uplift model using Two-Model approach"""
    
    # Prepare features (excluding promotion and conversion columns)
    feature_cols = [col for col in df.columns 
                   if col not in [promotion_col, conversion_col] 
                   and df[col].dtype in ['int64', 'float64']]
    
    if len(feature_cols) == 0:
        # No features available - return None
        return None, None, []
    
    X = df[feature_cols].fillna(0)
    y = df[conversion_col]
    treatment = df[promotion_col]
    
    # Train separate models for treatment and control
    # Model 1: Treatment group
    X_treatment = X[treatment == 1]
    y_treatment = y[treatment == 1]
    
    # Model 2: Control group
    X_control = X[treatment == 0]
    y_control = y[treatment == 0]
    
    if len(X_treatment) < 10 or len(X_control) < 10:
        return None, None, []
    
    # Train models
    model_treatment = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_control = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    model_treatment.fit(X_treatment, y_treatment)
    model_control.fit(X_control, y_control)
    
    # Predict on full dataset
    prob_treatment = model_treatment.predict_proba(X)[:, 1]
    prob_control = model_control.predict_proba(X)[:, 1]
    
    # Uplift = P(conversion | treatment) - P(conversion | control)
    uplift_scores = prob_treatment - prob_control
    
    return uplift_scores, feature_cols, [model_treatment, model_control]


def segment_customers(df: pd.DataFrame, uplift_scores: np.ndarray):
    """Segment customers by uplift score"""
    
    df['uplift_score'] = uplift_scores
    
    # Define segments based on uplift quartiles
    q25 = np.percentile(uplift_scores, 25)
    q50 = np.percentile(uplift_scores, 50)
    q75 = np.percentile(uplift_scores, 75)
    
    def assign_segment(score):
        if score > q75:
            return "Sure Things"  # High uplift - definitely target
        elif score > q50:
            return "Persuadables"  # Medium uplift - good targets
        elif score > q25:
            return "Sleeping Dogs"  # Low uplift - maybe skip
        else:
            return "Lost Causes"  # Negative uplift - don't target
    
    df['segment'] = df['uplift_score'].apply(assign_segment)
    
    return df


def optimize_promotion_strategy(df: pd.DataFrame, promotion_col: str, conversion_col: str, 
                                revenue_col: Optional[str], cost_per_promotion: float):
    """Calculate optimal promotion strategy and ROI"""
    
    strategies = []
    
    # Current strategy (baseline)
    current_promoted = df[df[promotion_col] == 1]
    current_cost = len(current_promoted) * cost_per_promotion
    current_conversions = current_promoted[conversion_col].sum()
    current_revenue = current_promoted[revenue_col].sum() if revenue_col else current_conversions * 100
    current_roi = (current_revenue - current_cost) / current_cost if current_cost > 0 else 0
    
    strategies.append({
        'strategy': 'Current (Baseline)',
        'customers_targeted': int(len(current_promoted)),
        'cost': float(current_cost),
        'conversions': int(current_conversions),
        'revenue': float(current_revenue),
        'roi': float(current_roi),
        'profit': float(current_revenue - current_cost)
    })
    
    # Check if uplift scores are available
    if 'uplift_score' not in df.columns:
        return strategies
    
    # Strategy 1: Target top 50% by uplift
    top_50 = df.nlargest(int(len(df) * 0.5), 'uplift_score')
    cost_50 = len(top_50) * cost_per_promotion
    conv_50 = top_50[conversion_col].sum()
    rev_50 = top_50[revenue_col].sum() if revenue_col else conv_50 * 100
    roi_50 = (rev_50 - cost_50) / cost_50 if cost_50 > 0 else 0
    
    strategies.append({
        'strategy': 'Target Top 50% Uplift',
        'customers_targeted': int(len(top_50)),
        'cost': float(cost_50),
        'conversions': int(conv_50),
        'revenue': float(rev_50),
        'roi': float(roi_50),
        'profit': float(rev_50 - cost_50)
    })
    
    # Strategy 2: Target top 25% by uplift
    top_25 = df.nlargest(int(len(df) * 0.25), 'uplift_score')
    cost_25 = len(top_25) * cost_per_promotion
    conv_25 = top_25[conversion_col].sum()
    rev_25 = top_25[revenue_col].sum() if revenue_col else conv_25 * 100
    roi_25 = (rev_25 - cost_25) / cost_25 if cost_25 > 0 else 0
    
    strategies.append({
        'strategy': 'Target Top 25% Uplift',
        'customers_targeted': int(len(top_25)),
        'cost': float(cost_25),
        'conversions': int(conv_25),
        'revenue': float(rev_25),
        'roi': float(roi_25),
        'profit': float(rev_25 - cost_25)
    })
    
    # Strategy 3: Target only positive uplift
    positive_uplift = df[df['uplift_score'] > 0]
    cost_pos = len(positive_uplift) * cost_per_promotion
    conv_pos = positive_uplift[conversion_col].sum()
    rev_pos = positive_uplift[revenue_col].sum() if revenue_col else conv_pos * 100
    roi_pos = (rev_pos - cost_pos) / cost_pos if cost_pos > 0 else 0
    
    strategies.append({
        'strategy': 'Target Only Positive Uplift',
        'customers_targeted': int(len(positive_uplift)),
        'cost': float(cost_pos),
        'conversions': int(conv_pos),
        'revenue': float(rev_pos),
        'roi': float(roi_pos),
        'profit': float(rev_pos - cost_pos)
    })
    
    return strategies


def generate_visualizations(df: pd.DataFrame, metrics: dict, strategies: list, 
                           promotion_col: str, conversion_col: str):
    """Generate visualizations"""
    visualizations = {}
    
    # 1. Conversion Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Promoted', 'Control']
    rates = [metrics['promo_conversion_rate'], metrics['control_conversion_rate']]
    colors = ['#4A90E2', '#95A5A6']
    bars = ax.bar(categories, rates, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Conversion Rate', fontsize=11)
    ax.set_title('Conversion Rate: Promoted vs Control', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add uplift annotation
    uplift_pct = metrics['uplift_percentage']
    if uplift_pct > 0:
        ax.annotate(f'Uplift: +{uplift_pct:.1f}%',
                   xy=(0.5, max(rates)), xytext=(0.5, max(rates) * 1.15),
                   ha='center', fontsize=11, fontweight='bold', color='#27AE60',
                   bbox=dict(boxstyle='round', facecolor='#D5F4E6', alpha=0.8))
    
    plt.tight_layout()
    visualizations['conversion_comparison'] = fig_to_base64(fig)
    
    # 2. Uplift Distribution (if available)
    if 'uplift_score' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['uplift_score'], bins=50, color='#4A90E2', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='#E74C3C', linestyle='--', linewidth=2, label='Zero Uplift')
        ax.axvline(df['uplift_score'].mean(), color='#27AE60', linestyle='--', linewidth=2,
                  label=f'Mean: {df["uplift_score"].mean():.3f}')
        ax.set_xlabel('Uplift Score', fontsize=11)
        ax.set_ylabel('Number of Customers', fontsize=11)
        ax.set_title('Distribution of Uplift Scores', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        visualizations['uplift_distribution'] = fig_to_base64(fig)
    
    # 3. Customer Segmentation (if available)
    if 'segment' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        segment_counts = df['segment'].value_counts()
        segment_order = ['Sure Things', 'Persuadables', 'Sleeping Dogs', 'Lost Causes']
        segment_counts = segment_counts.reindex([s for s in segment_order if s in segment_counts.index])
        
        colors_map = {
            'Sure Things': '#27AE60',
            'Persuadables': '#4A90E2',
            'Sleeping Dogs': '#F39C12',
            'Lost Causes': '#E74C3C'
        }
        bar_colors = [colors_map.get(seg, '#9E9E9E') for seg in segment_counts.index]
        
        bars = ax.bar(range(len(segment_counts)), segment_counts.values,
                     color=bar_colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(segment_counts)))
        ax.set_xticklabels(segment_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Customers', fontsize=11)
        ax.set_title('Customer Segmentation by Uplift', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, count) in enumerate(zip(bars, segment_counts.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/len(df)*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        plt.tight_layout()
        visualizations['customer_segmentation'] = fig_to_base64(fig)
    
    # 4. Strategy Comparison
    if len(strategies) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        strategy_names = [s['strategy'] for s in strategies]
        rois = [s['roi'] for s in strategies]
        profits = [s['profit'] for s in strategies]
        
        # ROI comparison
        bars1 = ax1.bar(range(len(strategy_names)), rois, 
                       color='#4A90E2', edgecolor='black', alpha=0.8)
        ax1.set_xticks(range(len(strategy_names)))
        ax1.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('ROI', fontsize=11)
        ax1.set_title('ROI by Strategy', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='black', linewidth=0.8)
        
        for bar, roi in zip(bars1, rois):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{roi:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontweight='bold', fontsize=9)
        
        # Profit comparison
        bars2 = ax2.bar(range(len(strategy_names)), profits,
                       color='#27AE60', edgecolor='black', alpha=0.8)
        ax2.set_xticks(range(len(strategy_names)))
        ax2.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Profit ($)', fontsize=11)
        ax2.set_title('Profit by Strategy', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(0, color='black', linewidth=0.8)
        
        for bar, profit in zip(bars2, profits):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${profit:,.0f}', ha='center',
                    va='bottom' if height > 0 else 'top',
                    fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        visualizations['strategy_comparison'] = fig_to_base64(fig)
    
    # 5. Statistical Significance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create visualization of statistical test
    categories = ['Promoted', 'Control']
    conv_rates = [metrics['promo_conversion_rate'], metrics['control_conversion_rate']]
    
    ax.bar(categories, conv_rates, color=['#4A90E2', '#95A5A6'], 
          edgecolor='black', alpha=0.8, width=0.6)
    
    # Add error bars (approximate)
    n_promo = metrics['promoted_customers']
    n_control = metrics['control_customers']
    se_promo = np.sqrt(conv_rates[0] * (1 - conv_rates[0]) / n_promo)
    se_control = np.sqrt(conv_rates[1] * (1 - conv_rates[1]) / n_control)
    
    ax.errorbar(range(2), conv_rates, yerr=[se_promo * 1.96, se_control * 1.96],
               fmt='none', color='black', capsize=5, capthick=2)
    
    ax.set_ylabel('Conversion Rate', fontsize=11)
    ax.set_title('Statistical Significance Test', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance annotation
    p_val = metrics['p_value']
    is_sig = metrics['is_significant']
    
    sig_text = f"p-value: {p_val:.4f}\n"
    sig_text += f"{'✓ Statistically Significant' if is_sig else '✗ Not Significant'} (α=0.05)"
    
    ax.text(0.5, max(conv_rates) * 1.15, sig_text,
           ha='center', fontsize=11, fontweight='bold',
           color='#27AE60' if is_sig else '#E74C3C',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    visualizations['statistical_significance'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(metrics: dict, strategies: list):
    """Generate key insights"""
    insights = []
    
    # Uplift insight
    uplift_pct = metrics['uplift_percentage']
    if metrics['is_significant']:
        if uplift_pct > 10:
            insights.append({
                'title': 'Strong Positive Uplift',
                'description': f"Promotion increases conversion by {uplift_pct:.1f}% (statistically significant, p={metrics['p_value']:.4f}). Strong evidence that promotions drive incremental conversions.",
                'status': 'positive'
            })
        elif uplift_pct > 0:
            insights.append({
                'title': 'Modest Positive Uplift',
                'description': f"Promotion increases conversion by {uplift_pct:.1f}% (statistically significant). Consider targeting high-uplift segments to improve ROI.",
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': 'Negative Uplift Detected',
                'description': f"Promotion decreases conversion by {abs(uplift_pct):.1f}% (statistically significant). Promotions may be cannibalizing organic conversions.",
                'status': 'warning'
            })
    else:
        insights.append({
            'title': 'Inconclusive Results',
            'description': f"Uplift of {uplift_pct:.1f}% is not statistically significant (p={metrics['p_value']:.4f}). Need more data or larger sample size.",
            'status': 'warning'
        })
    
    # Strategy recommendation
    if len(strategies) > 1:
        best_strategy = max(strategies[1:], key=lambda x: x['roi'])  # Exclude baseline
        
        if best_strategy['roi'] > strategies[0]['roi']:
            improvement = ((best_strategy['roi'] - strategies[0]['roi']) / abs(strategies[0]['roi']) * 100) if strategies[0]['roi'] != 0 else 0
            insights.append({
                'title': 'Optimization Opportunity',
                'description': f"'{best_strategy['strategy']}' strategy improves ROI by {improvement:.1f}% compared to current approach. Target {best_strategy['customers_targeted']:,} customers for ${best_strategy['profit']:,.0f} profit.",
                'status': 'positive'
            })
        else:
            insights.append({
                'title': 'Current Strategy Near-Optimal',
                'description': f"Current promotion strategy is already performing well. Optimization strategies show minimal improvement opportunity.",
                'status': 'neutral'
            })
    
    # ROI insight
    current_roi = strategies[0]['roi'] if strategies else 0
    if current_roi > 2:
        insights.append({
            'title': 'Excellent ROI',
            'description': f"Current campaign ROI of {current_roi:.2f} indicates highly effective promotions. Every $1 spent generates ${current_roi:.2f} in return.",
            'status': 'positive'
        })
    elif current_roi > 1:
        insights.append({
            'title': 'Positive ROI',
            'description': f"Campaign ROI of {current_roi:.2f} is profitable. Consider targeting optimization to improve returns further.",
            'status': 'neutral'
        })
    elif current_roi > 0:
        insights.append({
            'title': 'Low ROI',
            'description': f"Campaign ROI of {current_roi:.2f} is barely profitable. Strong need for targeting optimization or promotion redesign.",
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Negative ROI',
            'description': f"Campaign is losing money (ROI: {current_roi:.2f}). Immediate optimization or campaign suspension recommended.",
            'status': 'warning'
        })
    
    # Segmentation insight (if available)
    if len(strategies) > 2:
        insights.append({
            'title': 'Targeting Recommendations',
            'description': f"Focus promotions on 'Sure Things' and 'Persuadables' segments. Avoid 'Lost Causes' to reduce wasted spend and prevent negative reactions.",
            'status': 'neutral'
        })
    
    return insights


@router.post("/promotion-optimization")
async def optimize_promotions(request: PromotionRequest):
    """
    Promotion Optimization Analysis
    
    Uses uplift modeling to identify optimal promotion targeting
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 50:
            raise HTTPException(400, "Insufficient data (need at least 50 observations)")
        
        df = pd.DataFrame(request.data)
        
        required_cols = [request.promotion_col, request.conversion_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns: {missing}")
        
        # Validate binary columns
        if not df[request.promotion_col].isin([0, 1]).all():
            raise HTTPException(400, f"{request.promotion_col} must be binary (0/1)")
        if not df[request.conversion_col].isin([0, 1]).all():
            raise HTTPException(400, f"{request.conversion_col} must be binary (0/1)")
        
        # Check if we have both treatment and control
        if df[request.promotion_col].nunique() < 2:
            raise HTTPException(400, "Need both promoted (1) and control (0) groups")
        
        # Calculate basic metrics
        metrics = calculate_promotion_metrics(df, request.promotion_col, request.conversion_col, request.revenue_col)
        
        # Train uplift model
        uplift_scores, feature_cols, models = train_uplift_model(df, request.promotion_col, request.conversion_col)
        
        # Segment customers (if uplift model trained)
        if uplift_scores is not None:
            df = segment_customers(df, uplift_scores)
        
        # Optimize strategy
        strategies = optimize_promotion_strategy(df, request.promotion_col, request.conversion_col,
                                                request.revenue_col, request.cost_per_promotion)
        
        # Generate visualizations
        visualizations = generate_visualizations(df, metrics, strategies, 
                                                request.promotion_col, request.conversion_col)
        
        # Generate insights
        insights = generate_insights(metrics, strategies)
        
        # Prepare output
        customer_data = []
        output_cols = [request.promotion_col, request.conversion_col]
        if request.revenue_col:
            output_cols.append(request.revenue_col)
        if 'uplift_score' in df.columns:
            output_cols.extend(['uplift_score', 'segment'])
        
        customer_data_json = json.loads(df[output_cols].head(1000).to_json(orient='records'))
        
        response_data = {
            'success': True,
            'results': {
                'metrics': metrics,
                'strategies': strategies,
                'customer_sample': customer_data_json,
                'feature_importance': feature_cols if feature_cols else []
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'promotion_optimization',
                'total_customers': int(metrics['total_customers']),
                'uplift_percentage': float(metrics['uplift_percentage']),
                'is_significant': bool(metrics['is_significant']),
                'best_strategy': max(strategies, key=lambda x: x['roi'])['strategy'] if strategies else 'N/A'
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
