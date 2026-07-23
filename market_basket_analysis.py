import sys
import json
import io
import base64
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

NETWORK_CAP = 18  # top-N products by frequency shown in the network / heatmap


def _native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, frozenset):
        return sorted(list(obj))
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def _rule_row(row):
    return {
        "antecedents": sorted(list(row["antecedents"])),
        "consequents": sorted(list(row["consequents"])),
        "support": round(float(row["support"]), 5),
        "confidence": round(float(row["confidence"]), 5),
        "lift": round(float(row["lift"]), 5),
    }


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get("data")
        tx_col = payload.get("transaction_col")
        item_col = payload.get("item_col")
        price_col = payload.get("price_col") or None
        segment_col = payload.get("segment_col") or None
        period_col = payload.get("period_col") or None
        min_support = payload.get("min_support")
        min_confidence = payload.get("min_confidence")
        focus_item = payload.get("focus_item") or None

        min_support = float(min_support) if min_support not in (None, "") else 0.02
        min_confidence = float(min_confidence) if min_confidence not in (None, "") else 0.25
        min_support = min(max(min_support, 0.001), 0.5)
        min_confidence = min(max(min_confidence, 0.01), 1.0)

        if not data or not tx_col or not item_col:
            raise ValueError("Provide 'data', 'transaction_col', and 'item_col'")

        df = pd.DataFrame(data)
        if tx_col not in df.columns or item_col not in df.columns:
            raise ValueError("transaction_col or item_col not found in data")

        df[tx_col] = df[tx_col].astype(str)
        df[item_col] = df[item_col].astype(str).str.strip()
        df = df.dropna(subset=[tx_col, item_col])
        df = df[df[item_col] != ""]

        n_transactions = df[tx_col].nunique()
        if n_transactions < 20:
            raise ValueError("Need at least 20 distinct transactions")

        has_price = bool(price_col and price_col in df.columns)
        if has_price:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

        # ---- basket-level rollups ----
        basket_sizes = df.groupby(tx_col)[item_col].nunique()
        basket_items = df.groupby(tx_col)[item_col].apply(lambda s: sorted(set(s)))
        transactions_list = basket_items.tolist()

        n_items = df[item_col].nunique()

        # ---- ② Basket size distribution ----
        size_bins = []
        for lo, hi, label in [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, None, "5+")]:
            if hi is None:
                mask = basket_sizes >= lo
            else:
                mask = (basket_sizes >= lo) & (basket_sizes <= hi)
            size_bins.append({"n_items": label, "n_transactions": int(mask.sum()), "pct": round(float(mask.mean() * 100), 2)})

        avg_basket_size = float(basket_sizes.mean())

        # ---- ③ Product frequency / support ----
        item_counts = df.groupby(item_col)[tx_col].nunique().sort_values(ascending=False)
        product_frequency = [
            {"item": item, "n_transactions": int(cnt), "support": round(float(cnt / n_transactions), 5)}
            for item, cnt in item_counts.items()
        ]

        # ---- basket value (if price available) ----
        avg_basket_value = None
        if has_price:
            basket_value = df.groupby(tx_col)[price_col].sum()
            avg_basket_value = round(float(basket_value.mean()), 2)

        # ---- ④ Apriori: frequent itemsets + rules ----
        te = TransactionEncoder()
        te_ary = te.fit(transactions_list).transform(transactions_list)
        onehot = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True, max_len=4)
        n_itemsets = int(len(frequent_itemsets))

        rules_df = pd.DataFrame()
        if not frequent_itemsets.empty:
            rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            if not rules_df.empty:
                rules_df = rules_df.sort_values("lift", ascending=False)

        all_rules = [_rule_row(row) for _, row in rules_df.iterrows()] if not rules_df.empty else []
        n_rules = len(all_rules)

        # cap rows returned to the frontend to keep payload sane
        RULES_CAP = 200
        association_rules_table = all_rules[:RULES_CAP]

        unique_pairs = len({tuple(sorted(r["antecedents"] + r["consequents"])) for r in all_rules if len(r["antecedents"]) + len(r["consequents"]) == 2})

        top_rule = all_rules[0] if all_rules else None

        # ---- ⑦ Product pair analysis (2-item rules only, by count + lift) ----
        pair_rules = [r for r in all_rules if len(r["antecedents"]) == 1 and len(r["consequents"]) == 1]
        # count co-occurrences directly from baskets for every pair appearing in pair_rules
        pair_counts = {}
        for basket in transactions_list:
            for a, b in combinations(sorted(set(basket)), 2):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
        pair_analysis = []
        seen_pairs = set()
        for r in pair_rules:
            a, b = r["antecedents"][0], r["consequents"][0]
            key = tuple(sorted([a, b]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pair_analysis.append({
                "item_a": key[0], "item_b": key[1],
                "co_purchase_count": int(pair_counts.get(key, 0)),
                "support": r["support"], "confidence": r["confidence"], "lift": r["lift"],
            })
        pair_analysis.sort(key=lambda x: (-x["lift"], -x["co_purchase_count"]))

        # ---- ⑧ Product bundles: itemsets of size >= 3 (from rules where combined len >= 3) ----
        bundle_rules = [r for r in all_rules if (len(r["antecedents"]) + len(r["consequents"])) >= 3]
        bundles = []
        seen_bundles = set()
        for r in bundle_rules:
            items = tuple(sorted(set(r["antecedents"] + r["consequents"])))
            if items in seen_bundles:
                continue
            seen_bundles.add(items)
            bundles.append({
                "itemset": list(items), "support": r["support"], "confidence": r["confidence"], "lift": r["lift"],
            })
        bundles.sort(key=lambda x: -x["lift"])
        bundles = bundles[:100]

        # ---- ⑤ Rule explorer data: for every item, its rules-as-antecedent sorted by lift ----
        rules_by_antecedent = {}
        for r in all_rules:
            if len(r["antecedents"]) == 1:
                key = r["antecedents"][0]
                rules_by_antecedent.setdefault(key, []).append({
                    "consequent": ", ".join(r["consequents"]), "support": r["support"],
                    "confidence": r["confidence"], "lift": r["lift"],
                })
        for k in rules_by_antecedent:
            rules_by_antecedent[k].sort(key=lambda x: -x["lift"])
            rules_by_antecedent[k] = rules_by_antecedent[k][:15]

        explorer_items = sorted(rules_by_antecedent.keys())
        chosen_focus = focus_item if focus_item in rules_by_antecedent else (explorer_items[0] if explorer_items else None)
        focus_recommendations = rules_by_antecedent.get(chosen_focus, []) if chosen_focus else []
        # Full map returned so the frontend Rule Explorer dropdown can switch
        # items client-side without another backend round trip.
        rules_by_antecedent_full = rules_by_antecedent

        # ---- ⑥/⑩ Top-N products for network + lift matrix ----
        top_products = item_counts.head(NETWORK_CAP).index.tolist()
        network_capped = n_items > NETWORK_CAP
        network_note = (
            f"Capped to the top {NETWORK_CAP} products by frequency (out of {n_items} total) to keep the network/heatmap legible."
            if network_capped else
            f"All {n_items} products shown (catalog is within the {NETWORK_CAP}-product cap)."
        )

        # pairwise lift among top products, computed directly (not only from mined rules,
        # so the matrix is dense even at pairs below the confidence threshold)
        top_set = set(top_products)
        item_support = {item: cnt / n_transactions for item, cnt in item_counts.items()}
        pairwise_lift = {}
        for basket in transactions_list:
            b = set(basket) & top_set
            for a, c in combinations(sorted(b), 2):
                pairwise_lift.setdefault((a, c), 0)
                pairwise_lift[(a, c)] += 1
        lift_matrix = []
        for a in top_products:
            row = []
            for b in top_products:
                if a == b:
                    row.append(1.0)
                    continue
                key = tuple(sorted([a, b]))
                co = pairwise_lift.get(key, 0)
                sup_ab = co / n_transactions
                denom = item_support[a] * item_support[b]
                lift_val = round(sup_ab / denom, 3) if denom > 0 and sup_ab > 0 else 0.0
                row.append(lift_val)
            lift_matrix.append(row)

        network_edges = []
        for i, a in enumerate(top_products):
            for j, b in enumerate(top_products):
                if j <= i:
                    continue
                lv = lift_matrix[i][j]
                if lv > 1.0:
                    network_edges.append({"source": a, "target": b, "lift": lv, "weight": round(min(lv, 6.0), 2)})
        network_edges.sort(key=lambda x: -x["lift"])
        network_edges = network_edges[:60]
        network_nodes = [{"item": p, "support": round(item_support[p], 4), "count": int(item_counts[p])} for p in top_products]

        # ---- ⑨ Cross-sell opportunities: support x lift with tiering ----
        if pair_analysis:
            supports = [p["support"] for p in pair_analysis]
            lifts = [p["lift"] for p in pair_analysis]
            sup_med = float(np.median(supports))
            lift_med = float(np.median(lifts))
        else:
            sup_med, lift_med = 0.0, 1.0
        cross_sell = []
        for p in pair_analysis:
            high_sup = p["support"] >= sup_med
            high_lift = p["lift"] >= max(lift_med, 1.2)
            if high_sup and high_lift:
                tier = "High"
            elif high_lift or high_sup:
                tier = "Medium"
            else:
                tier = "Low"
            cross_sell.append({**p, "opportunity_tier": tier})
        cross_sell.sort(key=lambda x: (-{"High": 2, "Medium": 1, "Low": 0}[x["opportunity_tier"]], -x["lift"]))

        # ---- ⑪ Segment-level basket analysis (conditional) ----
        segment_analysis = None
        segment_note = None
        if segment_col and segment_col in df.columns:
            segment_analysis = []
            for seg_val, sub in df.groupby(segment_col):
                seg_baskets = sub.groupby(tx_col)[item_col].apply(lambda s: sorted(set(s))).tolist()
                if len(seg_baskets) < 10:
                    continue
                te2 = TransactionEncoder()
                arr2 = te2.fit(seg_baskets).transform(seg_baskets)
                oh2 = pd.DataFrame(arr2, columns=te2.columns_)
                fi2 = apriori(oh2, min_support=max(min_support, 0.03), use_colnames=True, max_len=3)
                top_rule_seg = None
                if not fi2.empty:
                    r2 = association_rules(fi2, metric="confidence", min_threshold=min_confidence)
                    if not r2.empty:
                        r2 = r2.sort_values("lift", ascending=False)
                        row = r2.iloc[0]
                        top_rule_seg = _rule_row(row)
                segment_analysis.append({
                    "segment": str(seg_val), "n_transactions": len(seg_baskets),
                    "top_rule": (f"{{{', '.join(top_rule_seg['antecedents'])}}} -> {{{', '.join(top_rule_seg['consequents'])}}}" if top_rule_seg else None),
                    "top_rule_lift": top_rule_seg["lift"] if top_rule_seg else None,
                })
        else:
            segment_note = "No customer-segment column provided — segment-level basket analysis skipped (no new segments created)."

        # ---- ⑫ Time-based basket analysis (conditional) ----
        time_analysis = None
        time_note = None
        if period_col and period_col in df.columns:
            time_analysis = []
            for period_val, sub in df.groupby(period_col):
                period_baskets = sub.groupby(tx_col)[item_col].apply(lambda s: sorted(set(s))).tolist()
                if len(period_baskets) < 10:
                    continue
                te3 = TransactionEncoder()
                arr3 = te3.fit(period_baskets).transform(period_baskets)
                oh3 = pd.DataFrame(arr3, columns=te3.columns_)
                fi3 = apriori(oh3, min_support=max(min_support, 0.03), use_colnames=True, max_len=3)
                top_rule_period = None
                if not fi3.empty:
                    r3 = association_rules(fi3, metric="confidence", min_threshold=min_confidence)
                    if not r3.empty:
                        r3 = r3.sort_values("lift", ascending=False)
                        row = r3.iloc[0]
                        top_rule_period = _rule_row(row)
                time_analysis.append({
                    "period": str(period_val), "n_transactions": len(period_baskets),
                    "top_rule": (f"{{{', '.join(top_rule_period['antecedents'])}}} -> {{{', '.join(top_rule_period['consequents'])}}}" if top_rule_period else None),
                    "top_rule_lift": top_rule_period["lift"] if top_rule_period else None,
                })
            time_analysis.sort(key=lambda x: x["period"])
        else:
            time_note = "No date/period column provided — time-based basket analysis skipped."

        # ---- ① Overview ----
        overview = {
            "n_transactions": n_transactions,
            "n_distinct_products": n_items,
            "avg_basket_size": round(avg_basket_size, 3),
            "avg_basket_value": avg_basket_value,
            "unique_product_pairs": unique_pairs,
            "top_rule": (f"{{{', '.join(top_rule['antecedents'])}}} -> {{{', '.join(top_rule['consequents'])}}}" if top_rule else None),
            "top_rule_lift": top_rule["lift"] if top_rule else None,
            "n_rules": n_rules,
            "n_itemsets": n_itemsets,
        }

        setup = {
            "transaction_col": tx_col, "item_col": item_col, "price_col": price_col,
            "segment_col": segment_col, "period_col": period_col,
            "min_support": min_support, "min_confidence": min_confidence,
        }

        # ---- Charts ----
        charts = {}

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar([b["n_items"] for b in size_bins], [b["n_transactions"] for b in size_bins], color="#2563eb")
        ax.set_xlabel("Items per basket")
        ax.set_ylabel("Transactions")
        ax.set_title("Basket Size Distribution")
        charts["basket_size_distribution"] = _png(fig)

        top_items_chart = product_frequency[:15]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([r["item"] for r in top_items_chart][::-1], [r["support"] for r in top_items_chart][::-1], color="#0891b2")
        ax.set_xlabel("Support (fraction of transactions)")
        ax.set_title("Top Product Frequency (Support)")
        charts["product_frequency"] = _png(fig)

        if pair_analysis:
            top_pairs_chart = pair_analysis[:15]
            fig, ax = plt.subplots(figsize=(7, 5))
            labels = [f"{p['item_a']} + {p['item_b']}" for p in top_pairs_chart][::-1]
            ax.barh(labels, [p["co_purchase_count"] for p in top_pairs_chart][::-1], color="#16a34a")
            ax.set_xlabel("Co-purchase count")
            ax.set_title("Top Product Pairs")
            charts["product_pairs"] = _png(fig)

        # Association network (simple circular layout, no networkx dependency needed)
        if network_nodes:
            fig, ax = plt.subplots(figsize=(8, 8))
            n = len(network_nodes)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos = {node["item"]: (np.cos(a), np.sin(a)) for node, a in zip(network_nodes, angles)}
            for e in network_edges:
                x1, y1 = pos[e["source"]]
                x2, y2 = pos[e["target"]]
                lift_norm = min(e["lift"] / 4.0, 1.0)
                ax.plot([x1, x2], [y1, y2], color=plt.cm.YlOrRd(0.3 + 0.7 * lift_norm), linewidth=0.5 + 2.5 * lift_norm, alpha=0.8, zorder=1)
            sizes = np.array([node["support"] for node in network_nodes])
            sizes = 300 + 2500 * (sizes / (sizes.max() or 1))
            ax.scatter([pos[node["item"]][0] for node in network_nodes], [pos[node["item"]][1] for node in network_nodes], s=sizes, color="#2563eb", edgecolor="white", zorder=2)
            for node in network_nodes:
                x, y = pos[node["item"]]
                ax.annotate(node["item"], (x, y), fontsize=8, ha="center", va="center", zorder=3)
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.axis("off")
            ax.set_title(f"Association Network (top {len(network_nodes)} products)")
            charts["association_network"] = _png(fig)

        # Lift heatmap
        if len(top_products) >= 2:
            fig, ax = plt.subplots(figsize=(8, 7))
            mat = np.array(lift_matrix)
            im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=max(3.0, mat.max()))
            ax.set_xticks(range(len(top_products)))
            ax.set_xticklabels(top_products, rotation=90, fontsize=7)
            ax.set_yticks(range(len(top_products)))
            ax.set_yticklabels(top_products, fontsize=7)
            ax.set_title("Product Association Matrix (Lift)")
            fig.colorbar(im, ax=ax, shrink=0.8, label="Lift")
            charts["association_matrix"] = _png(fig)

        if cross_sell:
            fig, ax = plt.subplots(figsize=(7, 6))
            colors_map = {"High": "#16a34a", "Medium": "#f59e0b", "Low": "#94a3b8"}
            for tier in ["Low", "Medium", "High"]:
                pts = [p for p in cross_sell if p["opportunity_tier"] == tier]
                if pts:
                    ax.scatter([p["support"] for p in pts], [p["lift"] for p in pts], color=colors_map[tier], label=tier, alpha=0.8, s=50)
            ax.axhline(lift_med, color="gray", linestyle="--", linewidth=0.8)
            ax.axvline(sup_med, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Support")
            ax.set_ylabel("Lift")
            ax.set_title("Cross-Sell Opportunity Matrix")
            ax.legend(title="Opportunity")
            charts["cross_sell_matrix"] = _png(fig)

        results = {
            "overview": overview,
            "setup": setup,
            "basket_size_distribution": size_bins,
            "product_frequency": product_frequency[:60],
            "association_rules": association_rules_table,
            "explorer_items": explorer_items,
            "chosen_focus_item": chosen_focus,
            "focus_recommendations": focus_recommendations,
            "rules_by_antecedent": rules_by_antecedent_full,
            "network_nodes": network_nodes,
            "network_edges": network_edges,
            "network_note": network_note,
            "network_cap": NETWORK_CAP,
            "product_pairs": pair_analysis[:60],
            "product_bundles": bundles,
            "cross_sell_opportunities": cross_sell[:60],
            "lift_matrix": {"products": top_products, "matrix": lift_matrix},
            "segment_analysis": segment_analysis,
            "segment_note": segment_note,
            "time_analysis": time_analysis,
            "time_note": time_note,
            "charts": charts,
        }

        print(json.dumps({"results": results, "plot": charts.get("association_network")}, default=_native))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
