from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import io
import base64
from collections import Counter

router = APIRouter()

class AssociationRuleRequest(BaseModel):
    data: List[Dict[str, Any]]
    item_cols: List[str]
    min_support: float = 0.01
    metric: str = "confidence"
    min_threshold: float = 0.5

class AssociationRuleResponse(BaseModel):
    frequent_itemsets: List[Dict[str, Any]]
    association_rules: List[Dict[str, Any]]
    scatter_plot: Optional[str] = None
    network_graph: Optional[str] = None
    heatmap: Optional[str] = None
    item_frequency_chart: Optional[str] = None
    lift_distribution: Optional[str] = None
    itemset_size_chart: Optional[str] = None
    interpretation: Dict[str, Any]

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def create_scatter_plot(rules_df: pd.DataFrame) -> str:
    """Create support vs confidence scatter plot colored by lift"""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        rules_df['support'], 
        rules_df['confidence'],
        c=rules_df['lift'],
        cmap='RdYlGn',
        s=60,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lift', fontsize=10)
    ax.set_xlabel('Support', fontsize=11)
    ax.set_ylabel('Confidence', fontsize=11)
    ax.set_title('Support vs Confidence (colored by Lift)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Confidence = 0.5')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig_to_base64(fig)

def create_network_graph(rules_df: pd.DataFrame, top_n: int = 15) -> str:
    """Create network graph of top association rules"""
    fig, ax = plt.subplots(figsize=(12, 10))
    G = nx.DiGraph()
    
    top_rules = rules_df.nlargest(top_n, 'lift')
    
    for _, row in top_rules.iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        G.add_edge(antecedents, consequents, weight=row['lift'], confidence=row['confidence'])
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'No rules to display', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig_to_base64(fig)
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges with varying width based on lift
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_weight) * 4 for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6, 
                           width=edge_widths, arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")
    
    # Draw nodes
    node_colors = ['#4CAF50' if G.out_degree(n) > 0 else '#2196F3' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=2000, alpha=0.9)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
    
    ax.set_title(f'Association Network (Top {len(top_rules)} Rules by Lift)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Antecedent (If)'),
        Patch(facecolor='#2196F3', label='Consequent (Then)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    fig.tight_layout()
    return fig_to_base64(fig)

def create_heatmap(rules_df: pd.DataFrame, top_n: int = 15) -> str:
    """Create lift heatmap for top item pairs"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique antecedents and consequents
    top_rules = rules_df.nlargest(top_n, 'lift').copy()
    top_rules['ant_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['con_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Create pivot table
    pivot_data = top_rules.pivot_table(
        values='lift', 
        index='ant_str', 
        columns='con_str', 
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        ax.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig_to_base64(fig)
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=1, ax=ax, cbar_kws={'label': 'Lift'})
    ax.set_xlabel('Consequent (Then)', fontsize=11)
    ax.set_ylabel('Antecedent (If)', fontsize=11)
    ax.set_title('Lift Heatmap', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig_to_base64(fig)

def create_item_frequency_chart(df: pd.DataFrame, item_cols: List[str], top_n: int = 15) -> str:
    """Create item frequency bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate item frequencies
    item_counts = df[item_cols].sum().sort_values(ascending=True)
    if len(item_counts) > top_n:
        item_counts = item_counts.tail(top_n)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(item_counts)))
    bars = ax.barh(item_counts.index, item_counts.values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, item_counts.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{int(val)}', va='center', fontsize=9)
    
    ax.set_xlabel('Frequency (Number of Transactions)', fontsize=11)
    ax.set_ylabel('Item', fontsize=11)
    ax.set_title(f'Top {len(item_counts)} Items by Frequency', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)

def create_lift_distribution(rules_df: pd.DataFrame) -> str:
    """Create lift and confidence distribution plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Lift distribution
    axes[0].hist(rules_df['lift'], bins=20, color='#4CAF50', alpha=0.7, edgecolor='white')
    axes[0].axvline(x=1, color='red', linestyle='--', label='Lift = 1 (No association)')
    axes[0].axvline(x=2, color='orange', linestyle='--', label='Lift = 2 (Strong)')
    axes[0].set_xlabel('Lift', fontsize=11)
    axes[0].set_ylabel('Number of Rules', fontsize=11)
    axes[0].set_title('Lift Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Confidence distribution
    axes[1].hist(rules_df['confidence'], bins=20, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0.5, color='red', linestyle='--', label='Confidence = 50%')
    axes[1].axvline(x=0.7, color='orange', linestyle='--', label='Confidence = 70%')
    axes[1].set_xlabel('Confidence', fontsize=11)
    axes[1].set_ylabel('Number of Rules', fontsize=11)
    axes[1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    return fig_to_base64(fig)

def create_itemset_size_chart(itemsets_df: pd.DataFrame) -> str:
    """Create itemset size distribution chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    size_counts = itemsets_df['itemsets'].apply(len).value_counts().sort_index()
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(size_counts)))
    bars = ax.bar(size_counts.index, size_counts.values, color=colors, edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, size_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(val)), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Itemset Size', fontsize=11)
    ax.set_ylabel('Number of Itemsets', fontsize=11)
    ax.set_title('Distribution of Itemset Sizes', fontsize=12, fontweight='bold')
    ax.set_xticks(size_counts.index)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)

def generate_interpretation(
    rules_df: pd.DataFrame, 
    itemsets_df: pd.DataFrame, 
    df: pd.DataFrame, 
    item_cols: List[str]
) -> Dict[str, Any]:
    """Generate comprehensive interpretation of results"""
    
    n_rules = len(rules_df)
    n_itemsets = len(itemsets_df)
    n_transactions = len(df)
    n_items = len(item_cols)
    
    # Basic statistics
    avg_confidence = rules_df['confidence'].mean() if n_rules > 0 else 0
    avg_lift = rules_df['lift'].mean() if n_rules > 0 else 0
    max_lift = rules_df['lift'].max() if n_rules > 0 else 0
    min_lift = rules_df['lift'].min() if n_rules > 0 else 0
    avg_support = rules_df['support'].mean() if n_rules > 0 else 0
    
    # Count rules by lift threshold
    rules_lift_gt_2 = len(rules_df[rules_df['lift'] > 2]) if n_rules > 0 else 0
    rules_confidence_gt_70 = len(rules_df[rules_df['confidence'] > 0.7]) if n_rules > 0 else 0
    
    # Categorize rules by lift
    rule_categories = {
        'very_strong': len(rules_df[rules_df['lift'] > 3]) if n_rules > 0 else 0,
        'strong': len(rules_df[(rules_df['lift'] > 2) & (rules_df['lift'] <= 3)]) if n_rules > 0 else 0,
        'moderate': len(rules_df[(rules_df['lift'] > 1.5) & (rules_df['lift'] <= 2)]) if n_rules > 0 else 0,
        'weak': len(rules_df[(rules_df['lift'] > 1) & (rules_df['lift'] <= 1.5)]) if n_rules > 0 else 0,
        'negative': len(rules_df[rules_df['lift'] <= 1]) if n_rules > 0 else 0,
    }
    
    # Top rules with interpretations
    top_rules = []
    if n_rules > 0:
        for _, row in rules_df.nlargest(5, 'lift').iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            lift = row['lift']
            conf = row['confidence']
            sup = row['support']
            
            interp = f"Customers who buy {ant} are {lift:.1f}x more likely to also buy {con}. "
            interp += f"This happens in {sup*100:.1f}% of all transactions with {conf*100:.1f}% confidence."
            
            top_rules.append({
                'rule': f"{ant} → {con}",
                'lift': lift,
                'confidence': conf,
                'support': sup,
                'interpretation': interp
            })
    
    # Key insights
    key_insights = []
    
    if n_rules > 0:
        if rules_lift_gt_2 > 0:
            key_insights.append({
                'title': 'Strong Associations Found',
                'description': f'{rules_lift_gt_2} rules have lift > 2, indicating strong positive associations suitable for cross-selling.'
            })
        
        if avg_lift > 1.5:
            key_insights.append({
                'title': 'Above Average Association Strength',
                'description': f'Average lift of {avg_lift:.2f} suggests meaningful patterns in purchasing behavior.'
            })
        
        if rules_confidence_gt_70 > 0:
            key_insights.append({
                'title': 'High Confidence Rules',
                'description': f'{rules_confidence_gt_70} rules have confidence > 70%, meaning these patterns are highly reliable.'
            })
    
    if n_itemsets > 0:
        max_itemset_size = itemsets_df['itemsets'].apply(len).max()
        key_insights.append({
            'title': 'Itemset Complexity',
            'description': f'Found itemsets up to size {max_itemset_size}, revealing multi-item purchasing patterns.'
        })
    
    # Item statistics
    item_stats = []
    item_frequencies = df[item_cols].sum().sort_values(ascending=False)
    
    for item in item_frequencies.head(10).index:
        freq = int(item_frequencies[item])
        freq_pct = round(freq / n_transactions * 100, 1)
        
        # Count rules containing this item
        rules_as_ant = 0
        rules_as_con = 0
        if n_rules > 0:
            for _, row in rules_df.iterrows():
                if item in row['antecedents']:
                    rules_as_ant += 1
                if item in row['consequents']:
                    rules_as_con += 1
        
        item_stats.append({
            'item': item,
            'frequency': freq,
            'frequency_pct': freq_pct,
            'rules_as_antecedent': rules_as_ant,
            'rules_as_consequent': rules_as_con,
            'total_rules': rules_as_ant + rules_as_con
        })
    
    return {
        'summary': {
            'total_rules': n_rules,
            'total_itemsets': n_itemsets,
            'total_transactions': n_transactions,
            'total_items': n_items,
            'avg_confidence': round(avg_confidence, 4),
            'avg_lift': round(avg_lift, 4),
            'max_lift': round(max_lift, 4),
            'min_lift': round(min_lift, 4),
            'avg_support': round(avg_support, 4),
            'rules_lift_gt_2': rules_lift_gt_2,
            'rules_confidence_gt_70': rules_confidence_gt_70
        },
        'top_rules': top_rules,
        'key_insights': key_insights,
        'item_stats': item_stats,
        'rule_categories': rule_categories
    }


@router.post("/association-rule", response_model=AssociationRuleResponse)
async def analyze_association_rules(request: AssociationRuleRequest):
    """
    Perform Association Rule Mining using Apriori algorithm.
    
    Parameters:
    - data: List of transaction records (one-hot encoded)
    - item_cols: List of column names representing items
    - min_support: Minimum support threshold (default: 0.01)
    - metric: Metric for rule evaluation ('confidence' or 'lift')
    - min_threshold: Minimum threshold for the metric
    
    Returns:
    - frequent_itemsets: List of frequent itemsets with support
    - association_rules: List of rules with metrics
    - Various visualization plots
    - interpretation: Detailed analysis and insights
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate item columns exist
        missing_cols = [col for col in request.item_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
        
        # Ensure binary encoding
        item_df = df[request.item_cols].copy()
        
        # Convert to boolean for mlxtend
        for col in request.item_cols:
            item_df[col] = item_df[col].astype(bool)
        
        # Check if data is valid
        if item_df.sum().sum() == 0:
            raise HTTPException(status_code=400, detail="No items found in data. Ensure columns contain 1s for purchased items.")
        
        # Run Apriori algorithm
        try:
            frequent_itemsets = apriori(
                item_df, 
                min_support=request.min_support, 
                use_colnames=True,
                max_len=None
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Apriori algorithm failed: {str(e)}. Try lowering min_support.")
        
        if frequent_itemsets.empty:
            raise HTTPException(
                status_code=400, 
                detail=f"No frequent itemsets found with min_support={request.min_support}. Try lowering the threshold."
            )
        
        # Generate association rules
        try:
            rules = association_rules(
                frequent_itemsets, 
                metric=request.metric, 
                min_threshold=request.min_threshold,
                num_itemsets=len(frequent_itemsets)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Rule generation failed: {str(e)}")
        
        # Sort rules by lift
        if not rules.empty:
            rules = rules.sort_values('lift', ascending=False)
        
        # Generate plots
        scatter_plot = None
        network_graph = None
        heatmap = None
        item_frequency_chart = None
        lift_distribution = None
        itemset_size_chart = None
        
        try:
            if not rules.empty:
                scatter_plot = create_scatter_plot(rules)
                network_graph = create_network_graph(rules)
                heatmap = create_heatmap(rules)
                lift_distribution = create_lift_distribution(rules)
            
            item_frequency_chart = create_item_frequency_chart(df, request.item_cols)
            
            if not frequent_itemsets.empty:
                itemset_size_chart = create_itemset_size_chart(frequent_itemsets)
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")
        
        # Generate interpretation
        interpretation = generate_interpretation(rules, frequent_itemsets, df, request.item_cols)
        
        # Format output
        itemsets_output = []
        for _, row in frequent_itemsets.iterrows():
            itemsets_output.append({
                'support': float(row['support']),
                'itemsets': list(row['itemsets'])
            })
        
        rules_output = []
        for _, row in rules.iterrows():
            rules_output.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift']),
                'leverage': float(row['leverage']) if 'leverage' in row else 0.0,
                'conviction': float(row['conviction']) if 'conviction' in row and not np.isinf(row['conviction']) else 999.0
            })
        
        return AssociationRuleResponse(
            frequent_itemsets=itemsets_output,
            association_rules=rules_output,
            scatter_plot=scatter_plot,
            network_graph=network_graph,
            heatmap=heatmap,
            item_frequency_chart=item_frequency_chart,
            lift_distribution=lift_distribution,
            itemset_size_chart=itemset_size_chart,
            interpretation=interpretation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
