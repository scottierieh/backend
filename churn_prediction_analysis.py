#!/usr/bin/env python3
"""Churn Prediction — logistic regression with churn probabilities, risk
segments, ROC/PR curves, feature importance and revenue-at-risk. scikit-learn
(+ statsmodels for coefficient p-values).

Input (from churn-prediction-page.tsx):
    data           : list[dict]
    dependentVar   : str        binary churn column (1 = churned)
    independentVars: string[]   numeric feature columns
    standardize    : bool
    value_col      : str        (optional) revenue / CLV per customer, for revenue-at-risk
Output: { results: {metrics, roc_data, pr_curve, feature_importance, risk_segments, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings("ignore")


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        dep = p.get("dependentVar")
        feats = p.get("independentVars") or []
        standardize = bool(p.get("standardize", False))
        value_col = p.get("value_col")
        if not rows or not dep or len(feats) < 1:
            raise ValueError("Provide data, a churn column and at least one feature.")
        df = pd.DataFrame(rows)
        feats = [c for c in feats if c in df.columns]
        if dep not in df.columns or len(feats) < 1:
            raise ValueError("Churn column or features not found in the data.")

        y_raw = df[dep]
        # map churn column to 0/1
        classes = sorted(pd.Series(y_raw.astype(str)).unique().tolist())
        yv = pd.to_numeric(y_raw, errors="coerce")
        if yv.notna().all() and set(np.unique(yv.dropna())) <= {0, 1}:
            y = yv.astype(int).to_numpy()
            dependent_classes = ["0", "1"]
        else:
            pos = None
            for cand in ("1", "yes", "true", "churn", "churned", "y"):
                for cl in classes:
                    if str(cl).lower() == cand:
                        pos = cl; break
                if pos: break
            pos = pos if pos is not None else classes[-1]
            y = (y_raw.astype(str) == str(pos)).astype(int).to_numpy()
            dependent_classes = [str(c) for c in classes if str(c) != str(pos)][:1] + [str(pos)]

        X = df[feats].apply(pd.to_numeric, errors="coerce")
        keep = X.notna().all(axis=1) & pd.Series(np.isfinite(y), index=df.index)
        X = X[keep].to_numpy(dtype=float); y = y[keep.to_numpy()]
        n = len(y)
        if n < len(feats) + 10 or y.sum() == 0 or y.sum() == n:
            raise ValueError("Need enough rows and both churned and retained customers.")

        Xs = StandardScaler().fit_transform(X) if standardize else X
        model = LogisticRegression(max_iter=1000, class_weight=None)
        model.fit(Xs, y)
        # out-of-sample probabilities via cross-val for honest metrics
        try:
            proba = cross_val_predict(LogisticRegression(max_iter=1000), Xs, y, cv=min(5, int(y.sum()), n // 10 or 2), method="predict_proba")[:, 1]
        except Exception:
            proba = model.predict_proba(Xs)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y, proba); roc_auc = float(auc(fpr, tpr))
        prec, rec, _ = precision_recall_curve(y, proba); ap = float(average_precision_score(y, proba))
        cm = confusion_matrix(y, pred).tolist()
        metrics = {
            "accuracy": _fin(accuracy_score(y, pred), 4),
            "balanced_accuracy": _fin(balanced_accuracy_score(y, pred), 4),
            "sensitivity": _fin(recall_score(y, pred, zero_division=0), 4),
            "specificity": _fin(recall_score(1 - y, 1 - pred, zero_division=0), 4),
            "precision": _fin(precision_score(y, pred, zero_division=0), 4),
            "f1": _fin(f1_score(y, pred, zero_division=0), 4),
            "confusion_matrix": cm,
        }

        # coefficient odds ratios + p-values (statsmodels)
        odds_ratios, p_values, feat_imp = {}, {}, []
        try:
            import statsmodels.api as sm
            Xc = sm.add_constant(Xs)
            res = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
            for i, f in enumerate(feats):
                coef = float(res.params[i + 1]); pv = float(res.pvalues[i + 1])
                odds_ratios[f] = _fin(np.exp(coef), 4); p_values[f] = _fin(pv, 6)
                feat_imp.append({"feature": f, "coefficient": _fin(coef, 4), "odds_ratio": _fin(np.exp(coef), 4),
                                 "p_value": _fin(pv, 6), "significant": bool(pv < 0.05), "abs_coef": abs(coef)})
        except Exception:
            for i, f in enumerate(feats):
                coef = float(model.coef_[0][i])
                odds_ratios[f] = _fin(np.exp(coef), 4); p_values[f] = None
                feat_imp.append({"feature": f, "coefficient": _fin(coef, 4), "odds_ratio": _fin(np.exp(coef), 4),
                                 "p_value": None, "significant": None, "abs_coef": abs(coef)})
        feat_imp.sort(key=lambda z: -z["abs_coef"])
        for fi in feat_imp:
            fi.pop("abs_coef", None)

        # risk segments
        def band(pr):
            return "High" if pr >= 0.7 else "Medium" if pr >= 0.4 else "Low"
        bands = np.array([band(pr) for pr in proba])
        risk_segments = []
        val = pd.to_numeric(df[value_col], errors="coerce").to_numpy()[keep.to_numpy()] if (value_col and value_col in df.columns) else None
        for b in ["High", "Medium", "Low"]:
            m = bands == b
            seg = {"segment": b, "count": int(m.sum()), "pct": _fin(m.mean(), 4),
                   "avg_prob": _fin(float(proba[m].mean()) if m.any() else 0, 4),
                   "actual_churn_rate": _fin(float(y[m].mean()) if m.any() else 0, 4)}
            if val is not None:
                seg["revenue_at_risk"] = _fin(float(np.nansum(proba[m] * np.nan_to_num(val[m]))), 2)
            risk_segments.append(seg)

        expected_churn = float(proba.sum())
        churn_rate = float(y.mean())
        revenue_at_risk_total = _fin(float(np.nansum(proba * np.nan_to_num(val))), 2) if val is not None else None

        # probability histogram
        hist_counts, hist_edges = np.histogram(proba, bins=20, range=(0, 1))
        prob_distribution = {"counts": [int(c) for c in hist_counts],
                             "edges": [_fin(float(e), 3) for e in hist_edges]}

        # top-risk customers
        order = np.argsort(-proba)[:20]
        idx_labels = df.index[keep].to_numpy()
        top_risk = []
        for oi in order:
            rec_ = {"row": int(idx_labels[oi]), "churn_probability": _fin(float(proba[oi]), 4),
                    "risk": band(proba[oi]), "actual": int(y[oi])}
            if val is not None:
                rec_["value"] = _fin(float(val[oi]), 2)
                rec_["revenue_at_risk"] = _fin(float(proba[oi] * (val[oi] if np.isfinite(val[oi]) else 0)), 2)
            top_risk.append(rec_)

        # ---- 4-panel plot ----
        plot = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=110)
            ax = axes[0, 0]
            ax.hist(proba[y == 0], bins=20, range=(0, 1), alpha=0.6, color="#16a34a", label="Retained")
            ax.hist(proba[y == 1], bins=20, range=(0, 1), alpha=0.6, color="#dc2626", label="Churned")
            ax.axvline(0.5, color="#111827", ls="--", lw=1); ax.set_xlabel("Predicted churn probability")
            ax.set_ylabel("Customers"); ax.set_title("1. Churn probability distribution"); ax.legend(fontsize=8)
            ax = axes[0, 1]
            ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("2. ROC curve"); ax.legend(fontsize=8)
            ax = axes[1, 0]
            ax.plot(rec, prec, color="#9333ea", lw=2, label=f"PR (AP={ap:.3f})")
            ax.axhline(churn_rate, color="#94a3b8", ls="--", lw=1, label=f"baseline {churn_rate:.2f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("3. Precision-Recall curve"); ax.legend(fontsize=8)
            ax = axes[1, 1]
            fnames = [f["feature"] for f in feat_imp][::-1]
            fcoef = [f["coefficient"] for f in feat_imp][::-1]
            cols = ["#dc2626" if c > 0 else "#2563eb" for c in fcoef]
            ax.barh(fnames, fcoef, color=cols)
            ax.axvline(0, color="#111827", lw=0.8)
            ax.set_xlabel("Logit coefficient (→ more churn)"); ax.set_title("4. Feature importance")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        top_feat = feat_imp[0] if feat_imp else None
        interpretation = (
            f"The model separates churners from non-churners with an AUC of {roc_auc:.3f} "
            f"({'excellent' if roc_auc > 0.85 else 'good' if roc_auc > 0.75 else 'moderate' if roc_auc > 0.65 else 'weak'}). "
            f"Of {n} customers, {int((bands=='High').sum())} are High risk (≥70% churn probability) and "
            f"the model expects about {expected_churn:.0f} total churners versus {int(y.sum())} observed. "
            + (f"The strongest driver is {top_feat['feature']} (odds ratio {top_feat['odds_ratio']:.2f}): each unit "
               f"{'raises' if top_feat['coefficient']>0 else 'lowers'} the odds of churn. " if top_feat else "")
            + (f"Total revenue at risk (probability-weighted) is {revenue_at_risk_total:,.0f}. " if revenue_at_risk_total is not None else "")
            + "Target retention at the High-risk segment, where intervention prevents the most expected churn."
        )

        results = {
            "status": "ok", "n_total": n, "n_features": len(feats), "churn_rate": _fin(churn_rate, 4),
            "expected_churn": _fin(expected_churn, 2), "dependent_classes": dependent_classes,
            "metrics": metrics,
            "roc_data": {"fpr": [_fin(float(x), 4) for x in fpr], "tpr": [_fin(float(x), 4) for x in tpr], "auc": _fin(roc_auc, 4)},
            "pr_curve": {"precision": [_fin(float(x), 4) for x in prec], "recall": [_fin(float(x), 4) for x in rec], "average_precision": _fin(ap, 4)},
            "odds_ratios": odds_ratios, "p_values": p_values, "feature_importance": feat_imp,
            "risk_segments": risk_segments, "prob_distribution": prob_distribution,
            "revenue_at_risk": revenue_at_risk_total, "top_risk": top_risk,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
