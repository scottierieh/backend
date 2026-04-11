from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

router = APIRouter()


class AssociationRulesRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    item_cols: List[str] = Field(...)
    min_support: float = Field(default=0.05)
    metric: str = Field(default="confidence")
    min_threshold: float = Field(default=0.7)


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (frozenset, set)):
        return sorted(list(obj))
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img


def scatter_plot(rules):
    if rules.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(rules['support'], rules['confidence'], c=rules['lift'], s=rules['lift']*50, cmap='viridis', alpha=0.7, edgecolor='white')
    plt.colorbar(scatter, ax=ax, label='Lift')
    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_title('Association Rules: Support vs Confidence', fontweight='bold')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def network_graph(rules, max_rules=30):
    if rules.empty:
        return None
    top_rules = rules.nlargest(min(max_rules, len(rules)), 'lift')
    G = nx.DiGraph()
    for _, rule in top_rules.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                G.add_edge(ant, cons, weight=rule['lift'])
    if len(G.nodes()) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    node_sizes = [300 + G.degree(node)*200 for node in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w/max_w)*3 for w in edge_weights]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', edgecolors='darkblue', linewidths=2, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', arrows=True, arrowsize=15, ax=ax)
    ax.set_title('Item Association Network', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig_to_base64(fig)


def heatmap(rules, max_items=15):
    if rules.empty:
        return None
    all_items = set()
    for _, rule in rules.iterrows():
        all_items.update(rule['antecedents'])
        all_items.update(rule['consequents'])
    items = sorted(list(all_items))[:max_items]
    if len(items) < 2:
        return None
    
    lift_matrix = pd.DataFrame(0.0, index=items, columns=items)
    for _, rule in rules.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                if ant in items and cons in items:
                    lift_matrix.loc[ant, cons] = max(lift_matrix.loc[ant, cons], rule['lift'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = lift_matrix == 0
    sns.heatmap(lift_matrix, annot=True, fmt='.2f', cmap='YlOrRd', mask=mask, ax=ax, linewidths=0.5, square=True)
    ax.set_title('Item Pair Lift Heatmap', fontweight='bold')
    ax.set_xlabel('Consequent')
    ax.set_ylabel('Antecedent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)


def frequency_chart(df_encoded, top_n=15):
    item_freq = df_encoded.sum().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(item_freq)))
    bars = ax.barh(range(len(item_freq)), item_freq.values, color=colors)
    ax.set_yticks(range(len(item_freq)))
    ax.set_yticklabels(item_freq.index)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Most Frequent Items', fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, item_freq.values)):
        ax.text(val + max(item_freq.values)*0.01, i, f'{int(val)}', va='center')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)


def lift_distribution(rules):
    if rules.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(rules['lift'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(1, color='red', linestyle='--', label='Lift=1')
    axes[0].axvline(rules['lift'].mean(), color='orange', linestyle='--', label=f"Mean={rules['lift'].mean():.2f}")
    axes[0].set_xlabel('Lift')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Lift Distribution', fontweight='bold')
    axes[0].legend()
    
    axes[1].hist(rules['confidence'], bins=20, color='seagreen', edgecolor='white', alpha=0.8)
    axes[1].axvline(0.5, color='red', linestyle='--', label='Conf=0.5')
    axes[1].axvline(rules['confidence'].mean(), color='orange', linestyle='--', label=f"Mean={rules['confidence'].mean():.2f}")
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution', fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    return fig_to_base64(fig)


def generate_interpretation(rules, frequent_itemsets, df_encoded):
    num_rules = len(rules)
    num_transactions = len(df_encoded)
    
    summary = {
        'total_rules': num_rules,
        'total_itemsets': len(frequent_itemsets),
        'total_transactions': num_transactions,
        'total_items': len(df_encoded.columns),
        'avg_confidence': float(rules['confidence'].mean()) if num_rules > 0 else 0,
        'avg_lift': float(rules['lift'].mean()) if num_rules > 0 else 0,
        'max_lift': float(rules['lift'].max()) if num_rules > 0 else 0,
        'rules_lift_gt_2': int((rules['lift'] > 2).sum()) if num_rules > 0 else 0,
        'rules_confidence_gt_70': int((rules['confidence'] > 0.7).sum()) if num_rules > 0 else 0,
    }
    
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
                'interpretation': f"Customers who buy {ant_str} are {rule['lift']:.2f}x more likely to buy {cons_str}"
            })
    
    key_insights = []
    if num_rules > 0:
        strong_rules = rules[rules['lift'] > 1.5]
        high_conf_rules = rules[rules['confidence'] > 0.7]
        key_insights.append({'title': 'Strong Associations', 'description': f"Found {len(strong_rules)} rules with lift > 1.5"})
        key_insights.append({'title': 'Reliable Rules', 'description': f"Found {len(high_conf_rules)} rules with confidence > 70%"})
        
        if summary['rules_lift_gt_2'] > 0:
            best = rules.nlargest(1, 'lift').iloc[0]
            key_insights.append({'title': 'Top Recommendation', 'description': f"Bundle {', '.join(best['antecedents'])} with {', '.join(best['consequents'])} (lift: {best['lift']:.2f})"})
    
    return {'summary': summary, 'top_rules': top_rules, 'key_insights': key_insights}


@router.post("/association-rules")
def association_rules_analysis(req: AssociationRulesRequest):
    try:
        df = pd.DataFrame(req.data)[req.item_cols].astype(bool)
        
        frequent_itemsets = apriori(df, min_support=req.min_support, use_colnames=True)
        if frequent_itemsets.empty:
            raise ValueError(f"No itemsets found with min_support={req.min_support}. Try lowering it.")
        
        rules = association_rules(frequent_itemsets, metric=req.metric, min_threshold=req.min_threshold, num_itemsets=len(frequent_itemsets))
        if rules.empty:
            raise ValueError(f"No rules found with {req.metric} >= {req.min_threshold}. Try lowering threshold.")
        
        rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
        
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: sorted(list(x)))
        rules['antecedents'] = rules['antecedents'].apply(lambda x: sorted(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: sorted(list(x)))
        
        freq_json = [{k: _to_native(v) for k, v in row.items()} for _, row in frequent_itemsets.iterrows()]
        rules_json = [{k: _to_native(v) for k, v in row.items()} for _, row in rules.iterrows()]
        
        return _to_native({
            'frequent_itemsets': freq_json,
            'association_rules': rules_json,
            'scatter_plot': scatter_plot(rules),
            'network_graph': network_graph(rules),
            'heatmap': heatmap(rules),
            'item_frequency_chart': frequency_chart(df),
            'lift_distribution': lift_distribution(rules),
            'interpretation': generate_interpretation(rules, frequent_itemsets, df)
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
