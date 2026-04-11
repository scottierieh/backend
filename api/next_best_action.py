"""
Next Best Action (NBA) FastAPI Endpoint
Personalized product recommendations using association rules mining
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
from collections import Counter

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Primary color scheme
PRIMARY_COLOR = '#3b82f6'  # Blue primary
PRIMARY_DARK = '#2563eb'
PRIMARY_LIGHT = '#60a5fa'

router = APIRouter()


class NBARequest(BaseModel):
    """Request model for Next Best Action"""
    data: List[Dict[str, Any]]
    customer_id_col: str
    product_cols: List[str]  # Binary columns (0/1) for each product purchased


class KeyInsight(BaseModel):
    """Key insight"""
    title: str
    description: str
    status: str


class NBAResponse(BaseModel):
    """Response model for Next Best Action"""
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


@router.post("/next-best-action")
async def analyze_next_best_action(request: NBARequest):
    """
    Next Best Action Analysis
    
    Generate personalized product recommendations using association rules
    (Market Basket Analysis with Apriori algorithm)
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        if len(request.data) < 10:
            raise HTTPException(400, "Insufficient data (need at least 10 customers)")
        
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.customer_id_col not in df.columns:
            raise HTTPException(400, f"Customer ID column '{request.customer_id_col}' not found")
        
        missing_products = [col for col in request.product_cols if col not in df.columns]
        if missing_products:
            raise HTTPException(400, f"Product columns not found: {missing_products}")
        
        if len(request.product_cols) < 3:
            raise HTTPException(400, "Need at least 3 product columns")
        
        # Prepare transaction data
        X = df[request.product_cols].copy()
        
        # Ensure binary
        for col in request.product_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)
            X[col] = X[col].apply(lambda x: 1 if x > 0 else 0)
        
        # Calculate basic metrics
        total_customers = len(df)
        
        # Product purchase rates
        product_stats = {}
        for product in request.product_cols:
            purchase_count = int(X[product].sum())
            purchase_rate = (purchase_count / total_customers) * 100
            
            product_stats[product] = {
                'purchase_count': purchase_count,
                'purchase_rate': float(purchase_rate),
                'customers': int(purchase_count)
            }
        
        # Most popular products
        popularity_rank = sorted(
            [(p, stats['purchase_count']) for p, stats in product_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Association Rules Mining using mlxtend
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        
        # Convert to transaction format
        transactions = []
        for idx, row in X.iterrows():
            transaction = [product for product in request.product_cols if row[product] == 1]
            if len(transaction) > 0:
                transactions.append(transaction)
        
        if len(transactions) < 5:
            raise HTTPException(400, "Insufficient transactions (need customers with at least 1 purchase)")
        
        # Calculate basket size distribution
        basket_sizes = [len(t) for t in transactions]
        avg_basket_size = np.mean(basket_sizes)
        
        # One-hot encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        transactions_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Apply Apriori algorithm
        min_support = max(0.05, 2 / len(transactions))  # At least 5% or 2 transactions
        frequent_itemsets = apriori(transactions_df, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            raise HTTPException(400, "No frequent itemsets found. Data may be too sparse.")
        
        # Generate association rules
        try:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
            rules = rules.sort_values('lift', ascending=False)
        except:
            # Fallback if no rules found
            rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
        
        # Top rules
        top_rules = []
        if len(rules) > 0:
            for idx, rule in rules.head(10).iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                top_rules.append({
                    'if_bought': antecedents,
                    'then_recommend': consequents,
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'support': float(rule['support'])
                })
        
        # Customer recommendations
        customer_recommendations = []
        for idx, row in df.iterrows():
            customer_id = row[request.customer_id_col]
            purchased_products = [p for p in request.product_cols if row[p] > 0]
            
            # Find applicable rules
            recommendations = []
            if len(rules) > 0 and len(purchased_products) > 0:
                for _, rule in rules.iterrows():
                    antecedents = set(rule['antecedents'])
                    consequents = set(rule['consequents'])
                    
                    # Check if customer bought antecedents and hasn't bought consequents
                    if antecedents.issubset(purchased_products):
                        for rec in consequents:
                            if rec not in purchased_products:
                                recommendations.append({
                                    'product': rec,
                                    'confidence': float(rule['confidence']),
                                    'lift': float(rule['lift'])
                                })
            
            # Remove duplicates, keep highest confidence
            unique_recs = {}
            for rec in recommendations:
                prod = rec['product']
                if prod not in unique_recs or rec['confidence'] > unique_recs[prod]['confidence']:
                    unique_recs[prod] = rec
            
            final_recs = sorted(unique_recs.values(), key=lambda x: x['confidence'], reverse=True)[:3]
            
            customer_recommendations.append({
                request.customer_id_col: customer_id,
                'purchased_products': purchased_products,
                'basket_size': len(purchased_products),
                'recommendations': final_recs,
                'recommendation_count': len(final_recs)
            })
        
        # Calculate coverage
        customers_with_recs = sum(1 for c in customer_recommendations if c['recommendation_count'] > 0)
        coverage_rate = (customers_with_recs / total_customers) * 100
        
        # Top recommended products
        all_recommendations = []
        for customer in customer_recommendations:
            for rec in customer['recommendations']:
                all_recommendations.append(rec['product'])
        
        recommendation_counts = Counter(all_recommendations)
        top_recommendations = recommendation_counts.most_common(5)
        
        metrics = {
            'total_customers': total_customers,
            'total_products': len(request.product_cols),
            'total_transactions': len(transactions),
            'avg_basket_size': float(avg_basket_size),
            'total_rules': len(rules),
            'coverage_rate': float(coverage_rate),
            'customers_with_recommendations': int(customers_with_recs)
        }
        
        # Visualizations
        visualizations = {}
        
        # 1. Product Purchase Rates
        fig, ax = plt.subplots(figsize=(12, 6))
        
        products = [p[0] for p in popularity_rank[:10]]
        counts = [p[1] for p in popularity_rank[:10]]
        
        ax.barh(products, counts, color=PRIMARY_COLOR, edgecolor='black', alpha=0.8)
        ax.set_xlabel('Number of Customers', fontsize=11, fontweight='bold')
        ax.set_title('Product Popularity', fontsize=13, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (prod, count) in enumerate(zip(products, counts)):
            ax.text(count + 1, i, str(count), va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        visualizations['product_popularity'] = fig_to_base64(fig)
        
        # 2. Basket Size Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        
        basket_counts = Counter(basket_sizes)
        sizes = sorted(basket_counts.keys())
        counts = [basket_counts[s] for s in sizes]
        
        ax.bar(sizes, counts, color=PRIMARY_COLOR, edgecolor='black', alpha=0.8, width=0.6)
        ax.set_xlabel('Basket Size (# of Products)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
        ax.set_title('Purchase Basket Size Distribution', fontsize=13, fontweight='bold', pad=15)
        ax.axvline(avg_basket_size, color=PRIMARY_DARK, linestyle='--', linewidth=2, 
                  label=f'Average: {avg_basket_size:.1f}')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        visualizations['basket_distribution'] = fig_to_base64(fig)
        
        # 3. Association Rules (Top 10)
        if len(rules) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            top_10_rules = rules.head(10).copy()
            rule_labels = []
            for _, rule in top_10_rules.iterrows():
                ant = ', '.join(list(rule['antecedents'])[:2])
                cons = ', '.join(list(rule['consequents'])[:2])
                rule_labels.append(f"{ant} → {cons}")
            
            lifts = top_10_rules['lift'].values
            
            colors = [PRIMARY_DARK if lift >= 2.0 else PRIMARY_COLOR if lift >= 1.5 else PRIMARY_LIGHT 
                     for lift in lifts]
            
            ax.barh(rule_labels, lifts, color=colors, edgecolor='black', alpha=0.8)
            ax.set_xlabel('Lift', fontsize=11, fontweight='bold')
            ax.set_title('Top Association Rules by Lift', fontsize=13, fontweight='bold', pad=15)
            ax.axvline(1.0, color='black', linestyle=':', linewidth=1, alpha=0.5)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, (label, lift) in enumerate(zip(rule_labels, lifts)):
                ax.text(lift + 0.1, i, f'{lift:.2f}', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            visualizations['association_rules'] = fig_to_base64(fig)
        
        # 4. Recommendation Coverage
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Coverage pie
        coverage_data = [customers_with_recs, total_customers - customers_with_recs]
        labels = ['With Recommendations', 'No Recommendations']
        colors = [PRIMARY_COLOR, PRIMARY_LIGHT]
        
        axes[0].pie(coverage_data, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=colors, textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[0].set_title('Recommendation Coverage', fontsize=12, fontweight='bold', pad=15)
        
        # Right: Top recommended products
        if top_recommendations:
            products_rec = [p[0] for p in top_recommendations]
            counts_rec = [p[1] for p in top_recommendations]
            
            axes[1].barh(products_rec, counts_rec, color=PRIMARY_COLOR, edgecolor='black', alpha=0.8)
            axes[1].set_xlabel('Recommendation Frequency', fontsize=10, fontweight='bold')
            axes[1].set_title('Most Recommended Products', fontsize=12, fontweight='bold', pad=15)
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3, axis='x')
            
            for i, (prod, count) in enumerate(zip(products_rec, counts_rec)):
                axes[1].text(count + 1, i, str(count), va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        visualizations['coverage_and_recommendations'] = fig_to_base64(fig)
        
        # 5. Confidence vs Lift Scatter
        if len(rules) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.scatter(rules['confidence'], rules['lift'], alpha=0.6, s=100, 
                      color=PRIMARY_COLOR, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
            ax.set_ylabel('Lift', fontsize=11, fontweight='bold')
            ax.set_title('Association Rules: Confidence vs Lift', fontsize=13, fontweight='bold', pad=15)
            ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (Lift=1)')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Annotate top rules
            for idx, rule in rules.head(5).iterrows():
                ant = list(rule['antecedents'])[0] if len(rule['antecedents']) > 0 else ''
                cons = list(rule['consequents'])[0] if len(rule['consequents']) > 0 else ''
                ax.annotate(f"{ant}→{cons}", 
                          xy=(rule['confidence'], rule['lift']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            visualizations['confidence_lift_scatter'] = fig_to_base64(fig)
        
        # Key Insights
        insights = []
        
        insights.append({
            'title': f'{coverage_rate:.1f}% Recommendation Coverage',
            'description': f"{customers_with_recs} out of {total_customers} customers received personalized recommendations based on their purchase history. {total_customers - customers_with_recs} customers need broader product exploration.",
            'status': 'positive' if coverage_rate >= 70 else 'neutral'
        })
        
        insights.append({
            'title': f'{len(rules)} Association Rules Discovered',
            'description': f"Market basket analysis identified {len(rules)} cross-sell patterns. Average basket size is {avg_basket_size:.1f} products per customer, indicating {'strong' if avg_basket_size >= 3 else 'moderate' if avg_basket_size >= 2 else 'limited'} product bundling opportunity.",
            'status': 'positive' if len(rules) >= 10 else 'neutral'
        })
        
        if top_rules:
            best_rule = top_rules[0]
            insights.append({
                'title': f"Top Rule: {best_rule['if_bought'][0]} → {best_rule['then_recommend'][0]}",
                'description': f"Customers who bought {best_rule['if_bought'][0]} have {best_rule['confidence']*100:.1f}% probability of buying {best_rule['then_recommend'][0]} (Lift: {best_rule['lift']:.2f}x). This is your strongest cross-sell opportunity.",
                'status': 'positive'
            })
        
        if avg_basket_size < 1.5:
            insights.append({
                'title': 'Low Product Bundling',
                'description': f"Average basket size of {avg_basket_size:.1f} suggests customers primarily buy single products. Consider bundle promotions and product education to increase cross-sell.",
                'status': 'warning'
            })
        
        summary = {
            'analysis_type': 'Next Best Action',
            'total_customers': total_customers,
            'coverage_rate': float(coverage_rate),
            'total_rules': len(rules)
        }
        
        return NBAResponse(
            success=True,
            results={
                'metrics': metrics,
                'product_stats': product_stats,
                'top_rules': top_rules,
                'customer_recommendations': customer_recommendations,
                'top_recommendations': [{'product': p[0], 'frequency': p[1]} for p in top_recommendations]
            },
            visualizations=visualizations,
            key_insights=insights,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
