from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class BasketRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nTransactions: int = 500
    seed: Optional[int] = None
    # Column mapping
    colTransactionId: Optional[str] = None
    colItem: Optional[str] = None
    colQuantity: Optional[str] = None
    # Config
    minSupport: float = 0.02
    minConfidence: float = 0.2
    minLift: float = 1.0
    algorithm: str = 'fpgrowth'  # apriori | fpgrowth
    maxRules: int = 50


def _to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_native(x) for x in obj]
    elif isinstance(obj, frozenset): return list(obj)
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception: return default


# ══════════════════════════════════════════════════════════════
# Data Generation
# ══════════════════════════════════════════════════════════════

PRODUCTS = {
    'Bread': {'category': 'Bakery', 'price': 3.5, 'freq': 0.35},
    'Milk': {'category': 'Dairy', 'price': 4.0, 'freq': 0.30},
    'Eggs': {'category': 'Dairy', 'price': 5.0, 'freq': 0.25},
    'Butter': {'category': 'Dairy', 'price': 3.0, 'freq': 0.18},
    'Cheese': {'category': 'Dairy', 'price': 6.0, 'freq': 0.15},
    'Chicken': {'category': 'Meat', 'price': 8.0, 'freq': 0.20},
    'Beef': {'category': 'Meat', 'price': 12.0, 'freq': 0.12},
    'Rice': {'category': 'Grains', 'price': 4.5, 'freq': 0.22},
    'Pasta': {'category': 'Grains', 'price': 2.5, 'freq': 0.18},
    'Tomato Sauce': {'category': 'Condiments', 'price': 3.0, 'freq': 0.15},
    'Onions': {'category': 'Vegetables', 'price': 2.0, 'freq': 0.25},
    'Potatoes': {'category': 'Vegetables', 'price': 3.0, 'freq': 0.20},
    'Bananas': {'category': 'Fruits', 'price': 2.5, 'freq': 0.28},
    'Apples': {'category': 'Fruits', 'price': 4.0, 'freq': 0.18},
    'Orange Juice': {'category': 'Beverages', 'price': 5.0, 'freq': 0.15},
    'Coffee': {'category': 'Beverages', 'price': 8.0, 'freq': 0.20},
    'Tea': {'category': 'Beverages', 'price': 4.0, 'freq': 0.12},
    'Chips': {'category': 'Snacks', 'price': 3.5, 'freq': 0.16},
    'Cookies': {'category': 'Snacks', 'price': 4.0, 'freq': 0.14},
    'Yogurt': {'category': 'Dairy', 'price': 3.5, 'freq': 0.17},
}

# Co-purchase affinities (products that tend to be bought together)
AFFINITIES = [
    (['Bread', 'Butter', 'Milk'], 0.25),
    (['Pasta', 'Tomato Sauce', 'Cheese'], 0.20),
    (['Chicken', 'Rice', 'Onions'], 0.15),
    (['Coffee', 'Milk', 'Cookies'], 0.18),
    (['Eggs', 'Bread', 'Butter'], 0.15),
    (['Bananas', 'Yogurt', 'Orange Juice'], 0.12),
    (['Beef', 'Potatoes', 'Onions'], 0.10),
    (['Tea', 'Cookies', 'Milk'], 0.10),
    (['Chips', 'Cheese', 'Bread'], 0.08),
]


def generate_transactions(n: int, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    products = list(PRODUCTS.keys())
    base_probs = np.array([PRODUCTS[p]['freq'] for p in products])
    rows = []

    for tid in range(n):
        basket = set()

        # Apply affinity bundles
        for bundle, prob in AFFINITIES:
            if rng.random() < prob:
                basket.update(bundle)

        # Add random items
        n_extra = rng.integers(1, 5)
        for _ in range(n_extra):
            if rng.random() < 0.7:
                idx = rng.choice(len(products), p=base_probs / base_probs.sum())
                basket.add(products[idx])

        for item in basket:
            qty = int(rng.integers(1, 4))
            rows.append({
                'transaction_id': f'T{tid+1:05d}',
                'item': item,
                'quantity': qty,
                'price': PRODUCTS[item]['price'],
                'category': PRODUCTS[item]['category'],
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/basket-analysis")
async def basket_analysis(request: BasketRequest):
    try:
        from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_transactions(request.nTransactions, request.seed)
            col_tid = 'transaction_id'
            col_item = 'item'
            col_qty = 'quantity'
        else:
            df = pd.DataFrame(request.data)
            col_tid = request.colTransactionId or next((c for c in df.columns if 'transaction' in c.lower() or 'order' in c.lower() or 'basket' in c.lower() or 'invoice' in c.lower()), None)
            col_item = request.colItem or next((c for c in df.columns if 'item' in c.lower() or 'product' in c.lower() or 'sku' in c.lower() or 'name' in c.lower()), None)
            col_qty = request.colQuantity or next((c for c in df.columns if 'qty' in c.lower() or 'quantity' in c.lower()), None)

            if not col_tid or not col_item:
                raise HTTPException(status_code=400, detail=f"Cannot find transaction_id or item column. Found: tid={col_tid}, item={col_item}")

        n_rows = len(df)
        n_transactions = df[col_tid].nunique()
        items = sorted(df[col_item].unique().tolist())
        n_items = len(items)

        if n_transactions < 10:
            raise HTTPException(status_code=400, detail=f"Need >=10 transactions. Got {n_transactions}.")

        # ── 2. Build Transaction Matrix ──
        transactions = df.groupby(col_tid)[col_item].apply(list).tolist()

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_array, columns=te.columns_)

        # ── 3. Frequent Itemsets ──
        min_sup = max(request.minSupport, 2 / n_transactions)  # at least 2 transactions

        if request.algorithm == 'apriori':
            freq_items = apriori(basket_df, min_support=min_sup, use_colnames=True)
        else:
            freq_items = fpgrowth(basket_df, min_support=min_sup, use_colnames=True)

        if len(freq_items) == 0:
            raise HTTPException(status_code=400, detail=f"No frequent itemsets found at min_support={min_sup:.3f}. Try lowering it.")

        # ── 4. Association Rules ──
        rules = association_rules(freq_items, metric='lift', min_threshold=request.minLift)
        rules = rules[rules['confidence'] >= request.minConfidence]
        rules = rules.sort_values('lift', ascending=False).head(request.maxRules)

        # Format rules
        rules_list = []
        for _, row in rules.iterrows():
            rules_list.append({
                'antecedents': sorted(list(row['antecedents'])),
                'consequents': sorted(list(row['consequents'])),
                'support': safe_float(row['support']),
                'confidence': safe_float(row['confidence']),
                'lift': safe_float(row['lift']),
                'leverage': safe_float(row.get('leverage', 0)),
                'conviction': safe_float(row.get('conviction', 0)),
                'antecedent_support': safe_float(row['antecedent support']),
                'consequent_support': safe_float(row['consequent support']),
                'rule_str': f"{', '.join(sorted(row['antecedents']))} → {', '.join(sorted(row['consequents']))}",
            })

        # ── 5. Frequent Itemsets Summary ──
        freq_list = []
        for _, row in freq_items.sort_values('support', ascending=False).head(30).iterrows():
            freq_list.append({
                'itemset': sorted(list(row['itemsets'])),
                'support': safe_float(row['support']),
                'count': int(row['support'] * n_transactions),
                'size': len(row['itemsets']),
            })

        # ── 6. Item Frequency ──
        item_freq = df[col_item].value_counts()
        item_freq_chart = [{'item': item, 'count': int(count), 'pct': safe_float(count / n_transactions * 100)}
                           for item, count in item_freq.head(20).items()]

        # ── 7. Co-occurrence Matrix (top items) ──
        top_items = item_freq.head(12).index.tolist()
        cooccurrence = []
        for i, item_a in enumerate(top_items):
            for j, item_b in enumerate(top_items):
                if i <= j:
                    count = sum(1 for t in transactions if item_a in t and item_b in t)
                    if count > 0:
                        cooccurrence.append({
                            'item_a': item_a,
                            'item_b': item_b,
                            'count': count,
                            'pct': safe_float(count / n_transactions * 100),
                        })

        # Heatmap data
        heatmap = []
        for item_a in top_items:
            for item_b in top_items:
                if item_a == item_b:
                    count = sum(1 for t in transactions if item_a in t)
                else:
                    count = sum(1 for t in transactions if item_a in t and item_b in t)
                heatmap.append({
                    'row': item_a,
                    'col': item_b,
                    'value': safe_float(count / n_transactions * 100),
                })

        # ── 8. Basket Size Distribution ──
        basket_sizes = [len(t) for t in transactions]
        size_dist = []
        for s in range(1, max(basket_sizes) + 1):
            c = basket_sizes.count(s)
            if c > 0:
                size_dist.append({'size': s, 'count': c, 'pct': safe_float(c / n_transactions * 100)})

        # ── 9. Category Analysis ──
        category_chart = []
        if 'category' in df.columns:
            cat_counts = df.groupby('category')[col_tid].nunique()
            for cat, count in cat_counts.sort_values(ascending=False).items():
                category_chart.append({'category': str(cat), 'transactions': int(count), 'pct': safe_float(count / n_transactions * 100)})

        # ── 10. Network Data (for visualization) ──
        network_nodes = set()
        network_edges = []
        for r in rules_list[:30]:
            for a in r['antecedents']:
                network_nodes.add(a)
            for c in r['consequents']:
                network_nodes.add(c)
            for a in r['antecedents']:
                for c in r['consequents']:
                    network_edges.append({
                        'source': a,
                        'target': c,
                        'lift': r['lift'],
                        'confidence': r['confidence'],
                    })

        network = {
            'nodes': [{'id': n, 'frequency': int(item_freq.get(n, 0))} for n in network_nodes],
            'edges': network_edges,
        }

        # ── 11. Lift Chart (top rules) ──
        lift_chart = [{'rule': r['rule_str'][:60], 'lift': r['lift'], 'confidence': r['confidence'], 'support': r['support']}
                      for r in rules_list[:20]]

        # ── Response ──
        results = {
            'n_transactions': n_transactions,
            'n_items': n_items,
            'n_rows': n_rows,
            'n_frequent_itemsets': len(freq_items),
            'n_rules': len(rules_list),
            'algorithm': request.algorithm,
            'config': {
                'min_support': request.minSupport,
                'min_confidence': request.minConfidence,
                'min_lift': request.minLift,
            },
            'avg_basket_size': safe_float(np.mean(basket_sizes)),
            'median_basket_size': safe_float(np.median(basket_sizes)),
            'columns_used': {'transaction_id': col_tid, 'item': col_item, 'quantity': col_qty},
            'rules': rules_list,
            'frequent_itemsets': freq_list,
            'charts': {
                'item_frequency': item_freq_chart,
                'lift': lift_chart,
                'basket_size': size_dist,
                'heatmap': heatmap,
                'heatmap_items': top_items,
                'category': category_chart,
            },
            'network': network,
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
