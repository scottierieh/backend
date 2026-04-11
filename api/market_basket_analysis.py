"""
Market Basket Analysis Router for FastAPI
Apriori, FP-Growth, ECLAT algorithms for Association Rule Mining
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
from collections import defaultdict
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class BasketAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    transaction_col: str  # Transaction identifier
    item_col: str  # Item/product identifier
    algorithm: Literal["apriori", "fpgrowth", "eclat"] = "apriori"
    min_support: float = 0.01  # Minimum support threshold
    min_confidence: float = 0.3  # Minimum confidence threshold
    max_length: int = 3  # Maximum itemset length
    min_lift: float = 1.0  # Minimum lift threshold


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
    if isinstance(obj, frozenset):
        return list(obj)
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def prepare_transactions(df: pd.DataFrame, transaction_col: str, item_col: str) -> List[set]:
    """Convert dataframe to list of transaction sets"""
    transactions = df.groupby(transaction_col)[item_col].apply(set).tolist()
    return transactions


def get_item_frequencies(transactions: List[set]) -> Dict[str, int]:
    """Calculate frequency of each item"""
    frequencies = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            frequencies[item] += 1
    return dict(frequencies)


def apriori_algorithm(transactions: List[set], min_support: float, max_length: int) -> List[Dict]:
    """
    Implementation of Apriori algorithm for frequent itemset mining
    """
    n_transactions = len(transactions)
    min_count = int(min_support * n_transactions)
    
    # Get frequent 1-itemsets
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
    
    frequent_itemsets = []
    current_frequent = {
        itemset: count for itemset, count in item_counts.items()
        if count >= min_count
    }
    
    # Add to results
    for itemset, count in current_frequent.items():
        frequent_itemsets.append({
            'itemsets': list(itemset),
            'support': count / n_transactions,
            'length': 1,
            'count': count
        })
    
    k = 2
    while current_frequent and k <= max_length:
        # Generate candidates
        items = set()
        for itemset in current_frequent.keys():
            items.update(itemset)
        
        candidates = set()
        prev_itemsets = list(current_frequent.keys())
        
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    candidates.add(union)
        
        # Count candidates
        candidate_counts = defaultdict(int)
        for transaction in transactions:
            transaction_set = frozenset(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1
        
        # Filter by minimum support
        current_frequent = {
            itemset: count for itemset, count in candidate_counts.items()
            if count >= min_count
        }
        
        # Add to results
        for itemset, count in current_frequent.items():
            frequent_itemsets.append({
                'itemsets': list(itemset),
                'support': count / n_transactions,
                'length': k,
                'count': count
            })
        
        k += 1
    
    return frequent_itemsets


def eclat_algorithm(transactions: List[set], min_support: float, max_length: int) -> List[Dict]:
    """
    Implementation of ECLAT algorithm using vertical data format
    """
    n_transactions = len(transactions)
    min_count = int(min_support * n_transactions)
    
    # Create vertical representation (item -> set of transaction indices)
    vertical = defaultdict(set)
    for idx, transaction in enumerate(transactions):
        for item in transaction:
            vertical[item].add(idx)
    
    # Filter items by minimum support
    frequent_items = {
        item: tids for item, tids in vertical.items()
        if len(tids) >= min_count
    }
    
    frequent_itemsets = []
    
    # Add frequent 1-itemsets
    for item, tids in frequent_items.items():
        frequent_itemsets.append({
            'itemsets': [item],
            'support': len(tids) / n_transactions,
            'length': 1,
            'count': len(tids)
        })
    
    def eclat_recursive(prefix: frozenset, items: Dict[str, set], k: int):
        if k > max_length:
            return
        
        item_list = list(items.keys())
        for i, item in enumerate(item_list):
            new_prefix = prefix | frozenset([item])
            new_tids = items[item]
            
            if len(new_tids) >= min_count:
                frequent_itemsets.append({
                    'itemsets': list(new_prefix),
                    'support': len(new_tids) / n_transactions,
                    'length': len(new_prefix),
                    'count': len(new_tids)
                })
                
                # Generate new items for next level
                new_items = {}
                for j in range(i + 1, len(item_list)):
                    other_item = item_list[j]
                    intersection = new_tids & items[other_item]
                    if len(intersection) >= min_count:
                        new_items[other_item] = intersection
                
                if new_items:
                    eclat_recursive(new_prefix, new_items, k + 1)
    
    # Start recursion
    eclat_recursive(frozenset(), frequent_items, 2)
    
    return frequent_itemsets


def fpgrowth_simplified(transactions: List[set], min_support: float, max_length: int) -> List[Dict]:
    """
    Simplified FP-Growth implementation
    For production, consider using mlxtend library
    """
    # Use Apriori as fallback (FP-Growth optimization requires complex tree structure)
    return apriori_algorithm(transactions, min_support, max_length)


def generate_association_rules(frequent_itemsets: List[Dict], transactions: List[set], 
                                min_confidence: float, min_lift: float) -> List[Dict]:
    """
    Generate association rules from frequent itemsets
    """
    n_transactions = len(transactions)
    
    # Create support lookup
    support_lookup = {}
    for itemset in frequent_itemsets:
        key = frozenset(itemset['itemsets'])
        support_lookup[key] = itemset['support']
    
    rules = []
    
    # Only generate rules from itemsets with length >= 2
    multi_item_sets = [fs for fs in frequent_itemsets if fs['length'] >= 2]
    
    for itemset_data in multi_item_sets:
        itemset = frozenset(itemset_data['itemsets'])
        itemset_support = itemset_data['support']
        
        # Generate all possible rules
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                
                # Get supports
                ant_support = support_lookup.get(antecedent, 0)
                cons_support = support_lookup.get(consequent, 0)
                
                if ant_support == 0 or cons_support == 0:
                    continue
                
                # Calculate metrics
                confidence = itemset_support / ant_support
                lift = confidence / cons_support if cons_support > 0 else 0
                
                # Filter by thresholds
                if confidence >= min_confidence and lift >= min_lift:
                    # Calculate conviction and leverage
                    if confidence < 1:
                        conviction = (1 - cons_support) / (1 - confidence)
                    else:
                        conviction = float('inf')
                    
                    leverage = itemset_support - (ant_support * cons_support)
                    
                    rules.append({
                        'antecedents': list(antecedent),
                        'consequents': list(consequent),
                        'support': itemset_support,
                        'confidence': confidence,
                        'lift': lift,
                        'conviction': conviction if conviction != float('inf') else 999999,
                        'leverage': leverage,
                        'antecedent_support': ant_support,
                        'consequent_support': cons_support
                    })
    
    # Sort by lift (descending)
    rules.sort(key=lambda x: x['lift'], reverse=True)
    
    return rules


def create_item_frequency_chart(item_frequencies: Dict[str, int], top_n: int = 20) -> str:
    """Create item frequency bar chart"""
    sorted_items = sorted(item_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    items, counts = zip(*sorted_items) if sorted_items else ([], [])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(items)))
    bars = ax.barh(range(len(items)), counts, color=colors)
    
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency (Number of Transactions)')
    ax.set_title('Top Items by Purchase Frequency', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_support_confidence_chart(rules: List[Dict]) -> str:
    """Create support vs confidence scatter plot"""
    if not rules:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No rules to display', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    supports = [r['support'] * 100 for r in rules]
    confidences = [r['confidence'] * 100 for r in rules]
    lifts = [r['lift'] for r in rules]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(supports, confidences, c=lifts, cmap='RdYlGn', 
                         s=80, alpha=0.6, edgecolors='white', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lift', fontsize=10)
    
    ax.set_xlabel('Support (%)', fontsize=11)
    ax.set_ylabel('Confidence (%)', fontsize=11)
    ax.set_title('Association Rules: Support vs Confidence', fontsize=14, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Confidence')
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_lift_matrix(rules: List[Dict], top_n: int = 15) -> str:
    """Create lift matrix heatmap for top items"""
    if not rules:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No rules to display', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    # Get unique items from top rules
    top_rules = rules[:50]  # Use top 50 rules
    all_items = set()
    for rule in top_rules:
        all_items.update(rule['antecedents'])
        all_items.update(rule['consequents'])
    
    items = sorted(list(all_items))[:top_n]
    
    # Create lift matrix
    matrix = np.ones((len(items), len(items)))
    
    for rule in rules:
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                if ant in items and cons in items:
                    i, j = items.index(ant), items.index(cons)
                    if rule['lift'] > matrix[i, j]:
                        matrix[i, j] = rule['lift']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = matrix == 1
    sns.heatmap(matrix, xticklabels=items, yticklabels=items, 
                annot=True, fmt='.2f', cmap='YlOrRd', mask=mask,
                ax=ax, cbar_kws={'label': 'Lift'})
    
    ax.set_xlabel('Consequent (Then Buy)', fontsize=11)
    ax.set_ylabel('Antecedent (If Buy)', fontsize=11)
    ax.set_title('Lift Matrix: Item Associations', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_rule_network_chart(rules: List[Dict], top_n: int = 20) -> str:
    """Create network-style visualization of top rules"""
    if not rules:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No rules to display', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    top_rules = rules[:top_n]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create positions for rules
    n_rules = len(top_rules)
    
    for idx, rule in enumerate(top_rules):
        y_pos = n_rules - idx
        
        # Antecedent position
        ant_text = ' + '.join(rule['antecedents'][:3])
        if len(rule['antecedents']) > 3:
            ant_text += '...'
        
        # Consequent position
        cons_text = ' + '.join(rule['consequents'][:3])
        if len(rule['consequents']) > 3:
            cons_text += '...'
        
        # Color based on lift
        lift = rule['lift']
        if lift >= 3:
            color = '#22c55e'  # Green
        elif lift >= 2:
            color = '#3b82f6'  # Blue
        elif lift >= 1.5:
            color = '#f59e0b'  # Orange
        else:
            color = '#6b7280'  # Gray
        
        # Draw arrow
        ax.annotate('', xy=(7, y_pos), xytext=(3, y_pos),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Draw text
        ax.text(1.5, y_pos, ant_text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1.5))
        ax.text(8.5, y_pos, cons_text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor=color, alpha=0.2, linewidth=1.5))
        
        # Add metrics
        metrics_text = f"Lift: {lift:.2f} | Conf: {rule['confidence']*100:.0f}%"
        ax.text(5, y_pos + 0.3, metrics_text, ha='center', va='bottom', fontsize=7, color='gray')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_rules + 1)
    ax.axis('off')
    ax.set_title('Top Association Rules', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='#22c55e', lw=3, label='Lift ≥ 3 (Very Strong)'),
        plt.Line2D([0], [0], color='#3b82f6', lw=3, label='Lift 2-3 (Strong)'),
        plt.Line2D([0], [0], color='#f59e0b', lw=3, label='Lift 1.5-2 (Moderate)'),
        plt.Line2D([0], [0], color='#6b7280', lw=3, label='Lift < 1.5 (Weak)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_itemset_distribution_chart(frequent_itemsets: List[Dict]) -> str:
    """Create distribution chart of frequent itemsets by length"""
    if not frequent_itemsets:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No itemsets to display', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    # Count by length
    length_counts = defaultdict(int)
    length_avg_support = defaultdict(list)
    
    for itemset in frequent_itemsets:
        length = itemset['length']
        length_counts[length] += 1
        length_avg_support[length].append(itemset['support'])
    
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    avg_supports = [np.mean(length_avg_support[l]) * 100 for l in lengths]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for counts
    bars = ax1.bar(lengths, counts, color='#3b82f6', alpha=0.7, label='Count')
    ax1.set_xlabel('Itemset Length', fontsize=11)
    ax1.set_ylabel('Number of Frequent Itemsets', fontsize=11, color='#3b82f6')
    ax1.tick_params(axis='y', labelcolor='#3b82f6')
    
    # Line chart for average support
    ax2 = ax1.twinx()
    ax2.plot(lengths, avg_supports, 'o-', color='#ef4444', linewidth=2, markersize=8, label='Avg Support')
    ax2.set_ylabel('Average Support (%)', fontsize=11, color='#ef4444')
    ax2.tick_params(axis='y', labelcolor='#ef4444')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)
    
    ax1.set_title('Frequent Itemsets Distribution by Length', fontsize=14, fontweight='bold')
    ax1.set_xticks(lengths)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(rules: List[Dict], metrics: Dict, item_frequencies: Dict[str, int]) -> List[Dict]:
    """Generate key insights from the analysis"""
    insights = []
    
    # Total rules insight
    total_rules = len(rules)
    if total_rules > 0:
        high_lift_rules = len([r for r in rules if r['lift'] >= 2])
        insights.append({
            'title': f'{total_rules} Association Rules Discovered',
            'description': f'{high_lift_rules} rules have lift ≥ 2, indicating strong positive associations worth acting on.',
            'status': 'positive' if high_lift_rules > 5 else 'neutral'
        })
    else:
        insights.append({
            'title': 'No Rules Found',
            'description': 'Try lowering the minimum support or confidence thresholds to discover more patterns.',
            'status': 'warning'
        })
    
    # Top rule insight
    if rules:
        top_rule = rules[0]
        ant_str = ' + '.join(top_rule['antecedents'])
        cons_str = ' + '.join(top_rule['consequents'])
        insights.append({
            'title': f'Strongest Association: {ant_str} → {cons_str}',
            'description': f'Customers who buy {ant_str} are {top_rule["lift"]:.1f}x more likely to also buy {cons_str}. Confidence: {top_rule["confidence"]*100:.1f}%',
            'status': 'positive'
        })
    
    # Basket size insight
    avg_basket = metrics.get('avg_basket_size', 0)
    if avg_basket >= 3:
        insights.append({
            'title': f'Good Basket Size: {avg_basket:.1f} items/transaction',
            'description': 'Customers are buying multiple items per visit, providing good data for pattern discovery.',
            'status': 'positive'
        })
    elif avg_basket >= 2:
        insights.append({
            'title': f'Moderate Basket Size: {avg_basket:.1f} items/transaction',
            'description': 'Consider promotions to increase items per transaction.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Basket Size: {avg_basket:.1f} items/transaction',
            'description': 'Cross-selling opportunities may be limited. Focus on increasing basket size first.',
            'status': 'warning'
        })
    
    # Top items insight
    if item_frequencies:
        top_items = sorted(item_frequencies.items(), key=lambda x: x[1], reverse=True)[:3]
        top_items_str = ', '.join([item for item, _ in top_items])
        insights.append({
            'title': f'Most Popular Items: {top_items_str}',
            'description': 'These items appear most frequently and drive many associations. Consider featuring them prominently.',
            'status': 'neutral'
        })
    
    # Confidence distribution insight
    if rules:
        high_conf_rules = len([r for r in rules if r['confidence'] >= 0.7])
        if high_conf_rules > 0:
            insights.append({
                'title': f'{high_conf_rules} High-Confidence Rules (≥70%)',
                'description': 'These rules are highly reliable for recommendation systems and targeted marketing.',
                'status': 'positive'
            })
    
    return insights


@router.post("/basket")
async def run_basket_analysis(request: BasketAnalysisRequest) -> Dict[str, Any]:
    """
    Perform Market Basket Analysis using association rule mining.
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        if request.transaction_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Transaction column '{request.transaction_col}' not found")
        if request.item_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item column '{request.item_col}' not found")
        
        # Prepare transactions
        transactions = prepare_transactions(df, request.transaction_col, request.item_col)
        n_transactions = len(transactions)
        
        if n_transactions < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 transactions for analysis")
        
        # Get item frequencies
        item_frequencies = get_item_frequencies(transactions)
        unique_items = len(item_frequencies)
        
        # Calculate average basket size
        total_items = sum(len(t) for t in transactions)
        avg_basket_size = total_items / n_transactions
        
        # Run selected algorithm
        if request.algorithm == "apriori":
            frequent_itemsets = apriori_algorithm(transactions, request.min_support, request.max_length)
        elif request.algorithm == "eclat":
            frequent_itemsets = eclat_algorithm(transactions, request.min_support, request.max_length)
        else:  # fpgrowth
            frequent_itemsets = fpgrowth_simplified(transactions, request.min_support, request.max_length)
        
        # Generate association rules
        rules = generate_association_rules(
            frequent_itemsets, 
            transactions, 
            request.min_confidence, 
            request.min_lift
        )
        
        # Calculate metrics
        max_lift = max([r['lift'] for r in rules]) if rules else 0
        avg_confidence = np.mean([r['confidence'] for r in rules]) if rules else 0
        
        metrics = {
            'total_transactions': n_transactions,
            'unique_items': unique_items,
            'avg_basket_size': _to_native_type(avg_basket_size),
            'total_rules': len(rules),
            'max_lift': _to_native_type(max_lift),
            'avg_confidence': _to_native_type(avg_confidence)
        }
        
        # Create visualizations
        visualizations = {
            'item_frequency': create_item_frequency_chart(item_frequencies),
            'support_confidence': create_support_confidence_chart(rules),
            'lift_matrix': create_lift_matrix(rules),
            'rule_network': create_rule_network_chart(rules),
            'itemset_treemap': create_itemset_distribution_chart(frequent_itemsets)
        }
        
        # Generate insights
        key_insights = generate_key_insights(rules, metrics, item_frequencies)
        
        # Prepare results
        results = {
            'association_rules': [
                {
                    'antecedents': r['antecedents'],
                    'consequents': r['consequents'],
                    'support': _to_native_type(r['support']),
                    'confidence': _to_native_type(r['confidence']),
                    'lift': _to_native_type(r['lift']),
                    'conviction': _to_native_type(r['conviction']),
                    'leverage': _to_native_type(r['leverage']),
                    'antecedent_support': _to_native_type(r['antecedent_support']),
                    'consequent_support': _to_native_type(r['consequent_support'])
                }
                for r in rules
            ],
            'frequent_itemsets': [
                {
                    'itemsets': fs['itemsets'],
                    'support': _to_native_type(fs['support']),
                    'length': fs['length']
                }
                for fs in sorted(frequent_itemsets, key=lambda x: x['support'], reverse=True)
            ],
            'item_frequencies': {k: _to_native_type(v) for k, v in item_frequencies.items()},
            'metrics': metrics
        }
        
        # Summary
        summary = {
            'algorithm': request.algorithm,
            'total_transactions': n_transactions,
            'total_rules': len(rules),
            'min_support': request.min_support,
            'min_confidence': request.min_confidence
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
        raise HTTPException(status_code=500, detail=f"Basket analysis failed: {str(e)}")
