#!/usr/bin/env python3
"""Market Basket Analysis — association rules via Apriori. mlxtend.

Finds which items are bought together: frequent itemsets and association rules
with support, confidence, lift, leverage and conviction.

Input (from market-basket-page.tsx):
    data           : list[dict]   one row per transaction/basket
    item_cols      : string[]     columns holding 0/1 (item present in basket)
    min_support    : float        (default 0.05)
    min_confidence : float        (default 0.3)
    max_len        : int          (default 3) max itemset size
Output: { results: {rules[], itemsets[], item_freq[]}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _to_bool(s):
    v = pd.to_numeric(s, errors="coerce")
    return (v > 0).fillna(False)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        cols = [c for c in (p.get("item_cols") or []) if c in df.columns]
        if len(cols) < 2:
            raise ValueError("Select at least two item columns (0/1 per basket).")
        min_support = float(p.get("min_support") or 0.05)
        min_conf = float(p.get("min_confidence") or 0.3)
        max_len = int(p.get("max_len") or 3)
        max_len = max(2, min(max_len, 4))

        B = pd.DataFrame({c: _to_bool(df[c]) for c in cols})
        n_tx = len(B)
        if n_tx < 5:
            raise ValueError("Need at least 5 transactions.")

        item_freq = [{"item": c, "count": int(B[c].sum()), "support": _fin(float(B[c].mean()), 4)}
                     for c in cols]
        item_freq.sort(key=lambda z: -z["count"])

        fi = apriori(B, min_support=min_support, use_colnames=True, max_len=max_len)
        if fi.empty:
            raise ValueError(f"No itemsets meet min support {min_support:.0%}. Lower the support threshold.")
        fi = fi.sort_values("support", ascending=False)

        def _names(fs):
            return sorted(list(fs))

        itemsets = [{"items": _names(r["itemsets"]), "size": len(r["itemsets"]),
                     "support": _fin(float(r["support"]), 4)} for _, r in fi.head(40).iterrows()]

        rules_out = []
        try:
            rules = association_rules(fi, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= 1.0].sort_values(["lift", "confidence"], ascending=False)
            for _, r in rules.head(60).iterrows():
                rules_out.append({
                    "antecedents": _names(r["antecedents"]), "consequents": _names(r["consequents"]),
                    "support": _fin(float(r["support"]), 4), "confidence": _fin(float(r["confidence"]), 4),
                    "lift": _fin(float(r["lift"]), 4), "leverage": _fin(float(r["leverage"]), 5),
                    "conviction": _fin(float(r["conviction"]), 4) if np.isfinite(r["conviction"]) else None,
                })
        except Exception:
            rules_out = []

        n_rules = len(rules_out)
        best = rules_out[0] if rules_out else None

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=118,
                                           gridspec_kw={"width_ratios": [1, 1.2]})
            top = item_freq[:min(12, len(item_freq))]
            ax1.barh([x["item"] for x in top][::-1], [x["support"] * 100 for x in top][::-1], color="#2563eb")
            ax1.set_xlabel("Support (%)"); ax1.set_title("Item frequency")
            if rules_out:
                sup = [r["support"] for r in rules_out]
                conf = [r["confidence"] for r in rules_out]
                lift = [r["lift"] for r in rules_out]
                sc = ax2.scatter(sup, conf, c=lift, cmap="viridis", s=60, edgecolor="white")
                fig.colorbar(sc, ax=ax2, label="Lift")
                ax2.set_xlabel("Support"); ax2.set_ylabel("Confidence")
                ax2.set_title(f"Association rules ({n_rules})")
            else:
                ax2.text(0.5, 0.5, "No rules at these thresholds", ha="center", va="center")
                ax2.set_axis_off()
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Across {n_tx} baskets and {len(cols)} items, Apriori found {len(fi)} frequent itemsets and "
            f"{n_rules} association rules (support ≥ {min_support:.0%}, confidence ≥ {min_conf:.0%}). "
            + (f"The strongest rule is {{{', '.join(best['antecedents'])}}} → {{{', '.join(best['consequents'])}}} "
               f"with a lift of {best['lift']:.2f}: baskets containing the first are {best['lift']:.1f}× more likely "
               f"to also contain the second than chance would predict, and this holds in {best['confidence']:.0%} of "
               f"such baskets. " if best else "No rules cleared the thresholds — try lowering support or confidence. ")
            + "Lift above 1 means a genuine positive association (cross-sell opportunity); lift near 1 means the items "
            "are independent. Use high-lift, reasonable-support rules for product placement, bundling and recommendations."
        )

        results = {
            "status": "ok", "n_transactions": n_tx, "n_items": len(cols),
            "min_support": _fin(min_support, 4), "min_confidence": _fin(min_conf, 4),
            "n_itemsets": int(len(fi)), "n_rules": n_rules,
            "item_freq": item_freq, "itemsets": itemsets, "rules": rules_out,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
