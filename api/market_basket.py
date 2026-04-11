"""
Market Basket Analysis (Association Rule Mining) Router for FastAPI
Discover product associations using Apriori algorithm
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
import networkx as nx
import io
import base64
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class MarketBasketRequest(BaseModel):
    data: List[Dict[str, Any]]
    item_cols: List[str]
    min_support: float = 0.01
    metric: str = "confidence"
    min_threshold: float = 0.5


class RuleCategories(BaseModel):
    very_strong: int
    strong: int
    moderate: int
    weak: int
    negative: int


class Summary(BaseModel):
    total_rules: int
    total_itemsets: int
    total_transactions: int
    total_items: int
    avg_confidence: float
    avg_lift: float
    max_lift: float
    min_lift: float
    avg_support: float
    rules_lift_gt_2: int
    rules_confidence_gt_70: int


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (frozenset, set)):
        return sorted(list(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _convert_dict(d: dict) -> dict:
    """Recursively convert numpy types in dictionary"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_dict(v)
        elif isinstance(v, list):
            result[k] = [_convert_dict(i) if isinstance(i, dict) else _to_native_type(i) for i in v]
        else:
            result[k] = _to_native_type(v)
    return result


def _dataframe_to_json(df) -> List[Dict]:
    """Convert DataFrame to JSON-serializable list of dicts"""
    return [
        {key: _to_native_type(value) for key, value in row.items()}
        for _, row in df.iterrows()
    ]


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def generate_scatter_plot(rules: pd.DataFrame) -> Optional[str]:
    """Generate scatter plot: Support vs Confidence, colored by Lift"""
    if rules.empty:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            rules['support'], 
            rules['confidence'],
            c=rules['lift'],
            s=rules['lift'] * 50,
            cmap='viridis',
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Lift', fontsize=11)
        
        ax.set_title('Association Rules: Support vs Confidence', fontsize=14, fontweight='bold')
        ax.set_xlabel('Support', fontsize=11)
        ax.set_ylabel('Confidence', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Confidence = 0.5')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='Confidence = 0.7')
        
        plt.tight_layout()
        return _fig_to_base64(fig)
        
    except Exception:
        return None


def generate_network_graph(rules: pd.DataFrame, max_rules: int = 30) -> Optional[str]:
    """Generate network graph showing item relationships"""
    if rules.empty:
        return None
    
    try:
        # Limit rules for readability
        top_rules = rules.nlargest(min(max_rules, len(rules)), 'lift')
        
        G = nx.DiGraph()
        
        # Add edges with weights
        for _, rule in top_rules.iterrows():
            for ant in rule['antecedents']:
                for cons in rule['consequents']:
                    G.add_edge(ant, cons, weight=rule['lift'], confidence=rule['confidence'])
        
        if len(G.nodes()) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node sizes based on degree
        node_sizes = [300 + G.degree(node) * 200 for node in G.nodes()]
        
        # Edge weights for width
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [1 + (w / max_weight) * 3 for w in edge_weights]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                               edgecolors='darkblue', linewidths=2, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                               edge_color='gray', arrows=True, 
                               arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax)
        
        ax.set_title('Item Association Network\n(Arrow: If → Then, Width: Lift strength)', 
                     fontsize=13, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return _fig_to_base64(fig)
        
    except Exception:
        return None


def generate_heatmap(rules: pd.DataFrame, max_items: int = 15) -> Optional[str]:
    """Generate heatmap showing lift between item pairs"""
    if rules.empty:
        return None
    
    try:
        # Get unique items
        all_items = set()
        for _, rule in rules.iterrows():
            all_items.update(rule['antecedents'])
            all_items.update(rule['consequents'])
        
        items = sorted(list(all_items))[:max_items]
        
        if len(items) < 2:
            return None
        
        # Create lift matrix
        lift_matrix = pd.DataFrame(0.0, index=items, columns=items)
        
        for _, rule in rules.iterrows():
            for ant in rule['antecedents']:
                for cons in rule['consequents']:
                    if ant in items and cons in items:
                        lift_matrix.loc[ant, cons] = max(lift_matrix.loc[ant, cons], rule['lift'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = lift_matrix == 0
        sns.heatmap(lift_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                    mask=mask, ax=ax, cbar_kws={'label': 'Lift'},
                    linewidths=0.5, square=True)
        
        ax.set_title('Item Pair Lift Heatmap\n(Row → Column)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Consequent (Then)', fontsize=11)
        ax.set_ylabel('Antecedent (If)', fontsize=11)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return _fig_to_base64(fig)
        
    except Exception:
        return None


def generate_item_frequency_chart(df_encoded: pd.DataFrame, top_n: int = 15) -> Optional[str]:
    """Generate bar chart of item frequencies"""
    try:
        item_freq = df_encoded.sum().sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(item_freq)))
        bars = ax.barh(range(len(item_freq)), item_freq.values, color=colors)
        
        ax.set_yticks(range(len(item_freq)))
        ax.set_yticklabels(item_freq.index)
        ax.invert_yaxis()
        
        ax.set_xlabel('Frequency (Number of Transactions)', fontsize=11)
        ax.set_title(f'Top {top_n} Most Frequent Items', fontsize=13, fontweight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, item_freq.values)):
            ax.text(val + max(item_freq.values) * 0.01, i, f'{int(val)}', 
                    va='center', fontsize=9)
        
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return _fig_to_base64(fig)
        
    except Exception:
        return None


def generate_lift_distribution(rules: pd.DataFrame) -> Optional[str]:
    """Generate histogram of lift and confidence distribution"""
    if rules.empty:
        return None
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Lift distribution
        ax1 = axes[0]
        ax1.hist(rules['lift'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax1.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Independent)')
        ax1.axvline(x=rules['lift'].mean(), color='orange', linestyle='--', 
                    label=f'Mean = {rules["lift"].mean():.2f}')
        ax1.set_xlabel('Lift', fontsize=11)
        ax1.set_ylabel('Number of Rules', fontsize=11)
        ax1.set_title('Lift Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Confidence distribution
        ax2 = axes[1]
        ax2.hist(rules['confidence'], bins=20, color='seagreen', edgecolor='white', alpha=0.8)
        ax2.axvline(x=0.5, color='red', linestyle='--', label='Confidence = 0.5')
        ax2.axvline(x=rules['confidence'].mean(), color='orange', linestyle='--',
                    label=f'Mean = {rules["confidence"].mean():.2f}')
        ax2.set_xlabel('Confidence', fontsize=11)
        ax2.set_ylabel('Number of Rules', fontsize=11)
        ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return _fig_to_base64(fig)
        
    except Exception:
        return None


def generate_itemset_size_chart(frequent_itemsets: pd.DataFrame) -> Optional[str]:
    """Generate chart showing itemset sizes"""
    if frequent_itemsets.empty:
        return None
    
    try:
        # Calculate itemset sizes
        sizes = frequent_itemsets['itemsets'].apply(len)
        size_counts = sizes.value_counts().sort_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of itemset sizes
        ax1 = axes[0]
        colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(size_counts)))
        bars = ax1.bar(size_counts.index, size_counts.values, color=colors, edgecolor='white')
        ax1.set_xlabel('Itemset Size', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Itemset Size Distribution', fontsize=12, fontweight='bold')
        ax1.set_xticks(size_counts.index)
        
        for bar, val in zip(bars, size_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, str(val), 
                    ha='center', fontsize=10, fontweight='bold')
        
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Support by itemset size (box plot)
        ax2 = axes[1]
        size_support_data = [frequent_itemsets[sizes == s]['support'].values 
                            for s in sorted(sizes.unique())]
        bp = ax2.boxplot(size_support_data, labels=sorted(sizes.unique()), patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_xlabel('Itemset Size', fontsize=11)
        ax2.set_ylabel('Support', fontsize=11)
        ax2.set_title('Support by Itemset Size', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return _fig_to_base64(fig)
        
    except Exception:
        return None


def generate_interpretation(rules: pd.DataFrame, frequent_itemsets: pd.DataFrame, df_encoded: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive interpretation of the results"""
    num_rules = len(rules)
    num_transactions = len(df_encoded)
    
    # Summary statistics
    summary = {
        'total_rules': num_rules,
        'total_itemsets': len(frequent_itemsets),
        'total_transactions': num_transactions,
        'total_items': len(df_encoded.columns),
        'avg_confidence': float(rules['confidence'].mean()) if num_rules > 0 else 0,
        'avg_lift': float(rules['lift'].mean()) if num_rules > 0 else 0,
        'max_lift': float(rules['lift'].max()) if num_rules > 0 else 0,
        'min_lift': float(rules['lift'].min()) if num_rules > 0 else 0,
        'avg_support': float(rules['support'].mean()) if num_rules > 0 else 0,
        'rules_lift_gt_2': int((rules['lift'] > 2).sum()) if num_rules > 0 else 0,
        'rules_confidence_gt_70': int((rules['confidence'] > 0.7).sum()) if num_rules > 0 else 0,
    }
    
    # Top rules by lift
    top_rules = []
    if num_rules > 0:
        for _, rule in rules.nlargest(5, 'lift').iterrows():
            ant_str = ', '.join(rule['antecedents'])
            cons_str = ', '.join(rule['consequents'])
            top_rules.append({
                'rule': f"{ant_str} → {cons_str}",
                'lift': float(rule['lift']),
                'confidence': float(rule['confidence']),
                'support': float(rule['support']),
                'interpretation': f"Customers who buy {ant_str} are {rule['lift']:.2f}x more likely to also buy {cons_str} (occurs in {rule['support']*100:.1f}% of transactions)"
            })
    
    # Item statistics
    item_stats = []
    item_freq = df_encoded.sum().sort_values(ascending=False)
    for item in item_freq.head(10).index:
        freq = int(item_freq[item])
        pct = freq / num_transactions * 100
        
        # Count rules involving this item
        rules_as_ant = sum(1 for _, r in rules.iterrows() if item in r['antecedents'])
        rules_as_cons = sum(1 for _, r in rules.iterrows() if item in r['consequents'])
        
        item_stats.append({
            'item': item,
            'frequency': freq,
            'frequency_pct': round(pct, 1),
            'rules_as_antecedent': rules_as_ant,
            'rules_as_consequent': rules_as_cons,
            'total_rules': rules_as_ant + rules_as_cons
        })
    
    # Key insights
    key_insights = []
    if num_rules > 0:
        strong_rules = rules[rules['lift'] > 1.5]
        high_conf_rules = rules[rules['confidence'] > 0.7]
        
        key_insights.append({
            'title': 'Strong Associations',
            'description': f"Found {len(strong_rules)} rules with lift > 1.5, indicating strong positive associations."
        })
        key_insights.append({
            'title': 'Reliable Rules',
            'description': f"Found {len(high_conf_rules)} rules with confidence > 70%, meaning these patterns occur reliably."
        })
        
        # Most connected items
        all_items = [
            item 
            for _, rule in rules.iterrows() 
            for item in list(rule['antecedents']) + list(rule['consequents'])
        ]
        
        if all_items:
            most_common = Counter(all_items).most_common(5)
            items_str = ', '.join([f"{item} ({count})" for item, count in most_common])
            key_insights.append({
                'title': 'Most Connected Items',
                'description': f"Items appearing most frequently in rules: {items_str}"
            })
        
        # Actionable insight
        if summary['rules_lift_gt_2'] > 0:
            best_rule = rules.nlargest(1, 'lift').iloc[0]
            key_insights.append({
                'title': 'Top Recommendation',
                'description': f"Strongest opportunity: Bundle {', '.join(best_rule['antecedents'])} with {', '.join(best_rule['consequents'])} (lift: {best_rule['lift']:.2f})"
            })
    
    # Rule categories
    rule_categories = {
        'very_strong': int((rules['lift'] > 3).sum()) if num_rules > 0 else 0,
        'strong': int(((rules['lift'] > 2) & (rules['lift'] <= 3)).sum()) if num_rules > 0 else 0,
        'moderate': int(((rules['lift'] > 1.5) & (rules['lift'] <= 2)).sum()) if num_rules > 0 else 0,
        'weak': int(((rules['lift'] > 1) & (rules['lift'] <= 1.5)).sum()) if num_rules > 0 else 0,
        'negative': int((rules['lift'] <= 1).sum()) if num_rules > 0 else 0,
    }
    
    return {
        'summary': summary,
        'top_rules': top_rules,
        'key_insights': key_insights,
        'item_stats': item_stats,
        'rule_categories': rule_categories
    }


@router.post("/association-rule")
async def run_market_basket_analysis(request: MarketBasketRequest) -> Dict[str, Any]:
    """
    Perform Market Basket Analysis (Association Rule Mining) using Apriori algorithm.
    
    This endpoint:
    1. Finds frequent itemsets meeting minimum support threshold
    2. Generates association rules from itemsets
    3. Calculates Support, Confidence, Lift metrics
    4. Creates visualizations for pattern discovery
    """
    try:
        # Import mlxtend here to handle potential import errors gracefully
        from mlxtend.frequent_patterns import apriori, association_rules
        
        data = request.data
        item_cols = request.item_cols
        min_support = request.min_support
        metric = request.metric
        min_threshold = request.min_threshold

        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        if not item_cols or len(item_cols) < 2:
            raise HTTPException(status_code=400, detail="At least 2 item columns required.")

        # Prepare data - convert to boolean DataFrame
        df = pd.DataFrame(data)
        
        # Validate columns exist
        missing_cols = [col for col in item_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_cols)}")
        
        df_encoded = df[item_cols].astype(bool)
        
        if len(df_encoded) < 10:
            raise HTTPException(status_code=400, detail="At least 10 transactions required for meaningful analysis.")

        # Run Apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            raise HTTPException(
                status_code=400, 
                detail=f"No itemsets found with minimum support of {min_support}. Try lowering the value (e.g., 0.01)."
            )

        # Generate association rules
        rules = association_rules(
            frequent_itemsets, 
            metric=metric, 
            min_threshold=min_threshold, 
            num_itemsets=len(frequent_itemsets)
        )
        
        if rules.empty:
            raise HTTPException(
                status_code=400, 
                detail=f"No rules found with {metric} >= {min_threshold}. Try lowering the threshold."
            )
        
        # Sort by lift and confidence
        rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
        
        # Convert frozensets to sorted lists
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: sorted(list(x)))
        rules['antecedents'] = rules['antecedents'].apply(lambda x: sorted(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: sorted(list(x)))

        # Generate visualizations
        scatter_plot = generate_scatter_plot(rules)
        network_graph = generate_network_graph(rules)
        heatmap = generate_heatmap(rules)
        item_frequency_chart = generate_item_frequency_chart(df_encoded)
        lift_distribution = generate_lift_distribution(rules)
        itemset_size_chart = generate_itemset_size_chart(frequent_itemsets)
        
        # Generate interpretation
        interpretation = generate_interpretation(rules, frequent_itemsets, df_encoded)

        # Prepare response
        response = {
            'frequent_itemsets': _dataframe_to_json(frequent_itemsets),
            'association_rules': _dataframe_to_json(rules),
            'scatter_plot': scatter_plot,
            'network_graph': network_graph,
            'heatmap': heatmap,
            'item_frequency_chart': item_frequency_chart,
            'lift_distribution': lift_distribution,
            'itemset_size_chart': itemset_size_chart,
            'interpretation': _convert_dict(interpretation)
        }

        return response

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Required library not installed: {str(e)}. Please install mlxtend: pip install mlxtend"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market basket analysis failed: {str(e)}")
