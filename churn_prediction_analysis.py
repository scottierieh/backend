#!/usr/bin/env python3
"""Churn Prediction Analysis — trains and compares 3 classifiers to predict
customer churn, then reports risk scores, drivers, explainability, threshold
analysis and a forecast.

Sections produced (see churn-prediction-page.tsx):
  1. Churn Overview (KPIs)
  2. Churn Probability Distribution (+ risk-tier table)
  3. Customer Churn Risk (per-customer table)
  4. Churn Model Comparison (>=3 models, same split)
  5. Model Performance (ROC/PR for recommended model)
  6. Confusion Matrix (recommended model @0.5)
  7. Churn Drivers (feature importance, recommended model)
  8. Explainable Churn Prediction (SHAP or documented approximation)
  9. Churn Risk Trend (optional, needs a date/month column)
  10. Churn by Customer Characteristics (existing categorical columns)
  11. Churn Prediction Threshold (precision/recall/FPR/FNR grid)
  12. Churn Forecast (30/60/90-day projection)

Input (stdin JSON):
    data           : list[dict]
    churn_col      : str            -- binary target column
    feature_cols   : list[str]      -- numeric + categorical predictors
    value_col      : str | None     -- optional revenue/value column (revenue at risk)
    time_col       : str | None     -- optional date/month column for section 9
                                        (auto-detected if omitted and a
                                        plausible column exists)

Output: { "results": {...} }
"""
import sys
import json
import warnings
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, confusion_matrix,
)

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#9333ea"
TEAL = "#0d9488"
GRAY = "#64748b"


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches='tight')
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def risk_tier(p):
    if p < 0.20:
        return "Low"
    if p < 0.50:
        return "Medium"
    if p < 0.80:
        return "High"
    return "Critical"


TIER_ORDER = ["Low", "Medium", "High", "Critical"]
TIER_DECAY_90D = {"Low": 0.05, "Medium": 0.15, "High": 0.35, "Critical": 0.60}  # documented assumption


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get("data")
        churn_col = payload.get("churn_col") or payload.get("dependentVar")
        feature_cols = payload.get("feature_cols") or payload.get("independentVars") or []
        value_col = payload.get("value_col")
        time_col = payload.get("time_col")

        if not data or not churn_col or not feature_cols:
            print(json.dumps({"error": "data, churn_col and feature_cols are required"}), file=sys.stderr)
            sys.exit(1)

        df = pd.DataFrame(data)
        if churn_col not in df.columns:
            print(json.dumps({"error": f"churn_col '{churn_col}' not found"}), file=sys.stderr)
            sys.exit(1)
        feature_cols = [c for c in feature_cols if c in df.columns and c != churn_col]
        if not feature_cols:
            print(json.dumps({"error": "no valid feature_cols found in data"}), file=sys.stderr)
            sys.exit(1)

        # target -> binary 0/1
        y_raw = df[churn_col]
        if y_raw.dtype == object:
            uniques = sorted(y_raw.dropna().unique().tolist(), key=str)
            pos_candidates = [u for u in uniques if str(u).lower() in ('1', 'yes', 'true', 'churn', 'churned')]
            pos_label = pos_candidates[0] if pos_candidates else uniques[-1]
            y = (y_raw == pos_label).astype(int)
        else:
            y = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(int)
            y = (y > 0).astype(int)

        X_raw = df[feature_cols].copy()
        numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_raw[c])]
        categorical_feats = [c for c in feature_cols if c not in numeric_feats]

        for c in numeric_feats:
            X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')
        X_raw[numeric_feats] = X_raw[numeric_feats].fillna(X_raw[numeric_feats].mean())

        X_enc = pd.get_dummies(X_raw, columns=categorical_feats, drop_first=False)
        X_enc = X_enc.fillna(0)
        feature_names = list(X_enc.columns)

        n_total = len(df)
        if n_total < 30:
            print(json.dumps({"error": f"Need at least 30 rows, got {n_total}"}), file=sys.stderr)
            sys.exit(1)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_enc), columns=feature_names, index=X_enc.index)

        idx_all = np.arange(n_total)
        strat = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        idx_train, idx_test = train_test_split(idx_all, test_size=0.25, random_state=42, stratify=strat)

        X_train, X_test = X_scaled.iloc[idx_train], X_scaled.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        models = {}
        models["logistic_regression"] = LogisticRegression(max_iter=1000, random_state=42)
        models["random_forest"] = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        if HAS_XGB:
            models["gradient_boosting"] = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
                use_label_encoder=False, eval_metric='logloss', verbosity=0,
            )
            gb_engine = "xgboost"
        else:
            models["gradient_boosting"] = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
            gb_engine = "sklearn.GradientBoostingClassifier (xgboost not available)"

        MODEL_LABELS = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting" if HAS_XGB else "Gradient Boosting (sklearn)",
        }

        model_comparison = []
        fitted = {}
        test_probs = {}
        for key, model in models.items():
            model.fit(X_train, y_train)
            fitted[key] = model
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            test_probs[key] = proba
            model_comparison.append({
                "model": MODEL_LABELS[key],
                "model_key": key,
                "accuracy": _fin(accuracy_score(y_test, pred), 4),
                "precision": _fin(precision_score(y_test, pred, zero_division=0), 4),
                "recall": _fin(recall_score(y_test, pred, zero_division=0), 4),
                "f1": _fin(f1_score(y_test, pred, zero_division=0), 4),
                "auc": _fin(roc_auc_score(y_test, proba), 4),
            })

        best = max(model_comparison, key=lambda r: r["auc"] if r["auc"] is not None else -1)
        best_key = best["model_key"]
        best_model = fitted[best_key]
        recommended_model_label = MODEL_LABELS[best_key]

        # refit recommended model on ALL data for final scoring / drivers / SHAP
        final_model = type(best_model)(**best_model.get_params())
        final_model.fit(X_scaled, y)
        proba_all = final_model.predict_proba(X_scaled)[:, 1]

        # --- Section 1: Overview ---
        churn_rate = float(y.mean())
        risk_tiers_all = [risk_tier(p) for p in proba_all]
        high_risk_count = sum(1 for t in risk_tiers_all if t in ("High", "Critical"))
        overview = {
            "total_customers": n_total,
            "predicted_churn_count": int((proba_all >= 0.5).sum()),
            "avg_churn_probability": _fin(float(np.mean(proba_all)), 4),
            "high_risk_customer_count": high_risk_count,
            "churn_rate": _fin(churn_rate, 4),
            "model_auc": best["auc"],
            "recommended_model": recommended_model_label,
        }

        # --- Section 2: Probability distribution + risk tier table ---
        hist_counts, hist_edges = np.histogram(proba_all, bins=10, range=(0, 1))
        prob_distribution = {
            "bin_edges": [_fin(e, 3) for e in hist_edges],
            "counts": [int(c) for c in hist_counts],
        }
        tier_counts = {t: 0 for t in TIER_ORDER}
        for t in risk_tiers_all:
            tier_counts[t] += 1
        risk_tier_table = [
            {"tier": t, "range": {"Low": "0-20%", "Medium": "20-50%", "High": "50-80%", "Critical": "80-100%"}[t],
             "count": tier_counts[t], "pct": _fin(tier_counts[t] / n_total, 4)}
            for t in TIER_ORDER
        ]

        # --- Section 3: per-customer table ---
        value_series = pd.to_numeric(df[value_col], errors='coerce').fillna(0) if value_col and value_col in df.columns else None
        customer_table = []
        for i in range(n_total):
            row = {
                "id": i,
                "churn_probability": _fin(float(proba_all[i]), 4),
                "risk_level": risk_tiers_all[i],
                "predicted_churn": bool(proba_all[i] >= 0.5),
                "actual_churn": int(y.iloc[i]),
            }
            if value_series is not None:
                row["value"] = _fin(float(value_series.iloc[i]), 2)
                row["revenue_at_risk"] = _fin(float(value_series.iloc[i]) * float(proba_all[i]), 2)
            customer_table.append(row)

        revenue_at_risk_total = _fin(float(np.sum([r.get("revenue_at_risk", 0) for r in customer_table])), 2) if value_series is not None else None

        # --- Section 5: ROC / PR for recommended model (on held-out test set) ---
        best_test_proba = test_probs[best_key]
        fpr, tpr, _ = roc_curve(y_test, best_test_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, best_test_proba)
        avg_precision = _fin(average_precision_score(y_test, best_test_proba), 4)

        best_test_pred = (best_test_proba >= 0.5).astype(int)
        perf_metrics = {
            "accuracy": _fin(accuracy_score(y_test, best_test_pred), 4),
            "precision": _fin(precision_score(y_test, best_test_pred, zero_division=0), 4),
            "recall": _fin(recall_score(y_test, best_test_pred, zero_division=0), 4),
            "f1": _fin(f1_score(y_test, best_test_pred, zero_division=0), 4),
            "roc_auc": best["auc"],
            "pr_auc": avg_precision,
        }

        # --- Section 6: confusion matrix @0.5 ---
        cm = confusion_matrix(y_test, best_test_pred, labels=[0, 1]).tolist()
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        confusion = {"matrix": cm, "labels": ["Retained (0)", "Churned (1)"], "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        # --- Section 7: feature importance (recommended model, fit on full data) ---
        if best_key == "logistic_regression":
            importances = np.abs(final_model.coef_[0])
            raw_coef = final_model.coef_[0]
            importance_kind = "coefficient magnitude (|coef|, standardized features)"
        else:
            importances = final_model.feature_importances_
            raw_coef = importances
            importance_kind = "impurity/gain-based feature importance"
        imp_order = np.argsort(-importances)[:10]
        feature_importance = [
            {"feature": feature_names[i], "importance": _fin(float(importances[i]), 5),
             "direction": "increases" if raw_coef[i] > 0 else "decreases"}
            for i in imp_order
        ]

        # --- Section 8: explainability for top-N highest risk customers ---
        top_risk_idx = np.argsort(-proba_all)[:5]
        explanations = []
        shap_used = False
        if HAS_SHAP and best_key in ("random_forest", "gradient_boosting"):
            try:
                explainer = shap.TreeExplainer(final_model)
                shap_vals = explainer.shap_values(X_scaled.iloc[top_risk_idx])
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_used = True
                for j, ci in enumerate(top_risk_idx):
                    contribs = shap_vals[j]
                    order = np.argsort(-np.abs(contribs))[:4]
                    explanations.append({
                        "customer_id": int(ci),
                        "churn_probability": _fin(float(proba_all[ci]), 4),
                        "method": "shap",
                        "top_factors": [
                            {"feature": feature_names[k], "contribution": _fin(float(contribs[k]), 5),
                             "effect": "increases risk" if contribs[k] > 0 else "decreases risk"}
                            for k in order
                        ],
                    })
            except Exception:
                shap_used = False
        if not shap_used:
            # documented linear approximation: coefficient (or importance-signed) x
            # standardized deviation from the mean, NOT true SHAP values.
            means = X_scaled.mean(axis=0).values
            for ci in top_risk_idx:
                row_vals = X_scaled.iloc[ci].values
                deviation = row_vals - means
                contribs = raw_coef * deviation
                order = np.argsort(-np.abs(contribs))[:4]
                explanations.append({
                    "customer_id": int(ci),
                    "churn_probability": _fin(float(proba_all[ci]), 4),
                    "method": "approximation",
                    "top_factors": [
                        {"feature": feature_names[k], "contribution": _fin(float(contribs[k]), 5),
                         "effect": "increases risk" if contribs[k] > 0 else "decreases risk"}
                        for k in order
                    ],
                })
        explainability_note = (
            "Real SHAP TreeExplainer values from the recommended tree-based model."
            if shap_used else
            "Approximation (NOT SHAP): per-feature contribution = coefficient/importance x "
            "standardized deviation from the feature's mean. Used because the recommended "
            "model is linear or shap was unavailable/failed for this run."
        )

        # --- Section 9: trend (optional) ---
        trend_table = None
        trend_note = None
        if not time_col:
            for cand in ("signup_month", "month", "date", "observation_month", "acquisition_month"):
                if cand in df.columns:
                    time_col = cand
                    break
        if time_col and time_col in df.columns:
            tmp = df[[time_col]].copy()
            tmp["_prob"] = proba_all
            tmp["_actual"] = y.values
            tmp["_tier"] = risk_tiers_all
            grouped = tmp.groupby(time_col)
            trend_rows = []
            for period, g in grouped:
                trend_rows.append({
                    "period": str(period),
                    "customers": int(len(g)),
                    "churn_rate": _fin(float(g["_actual"].mean()), 4),
                    "avg_churn_probability": _fin(float(g["_prob"].mean()), 4),
                    "high_risk_count": int((g["_tier"].isin(["High", "Critical"])).sum()),
                })
            trend_rows.sort(key=lambda r: r["period"])
            trend_table = trend_rows
            trend_note = f"Computed from '{time_col}' (customer cohort/acquisition period), grouping churn rate and high-risk count by period."
        else:
            trend_note = "No usable date/month column was selected or found; Churn Risk Trend is gracefully skipped."

        # --- Section 10: churn by characteristics (existing categorical columns) ---
        characteristics = {}
        for c in categorical_feats:
            g = df.groupby(c).apply(lambda gg: float(y.loc[gg.index].mean()))
            characteristics[c] = [
                {"category": str(k), "churn_rate": _fin(v, 4), "count": int((df[c] == k).sum())}
                for k, v in g.items()
            ]

        # --- Section 11: threshold analysis ---
        thresholds = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)
        threshold_table = []
        for th in thresholds:
            pred_th = (best_test_proba >= th).astype(int)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, pred_th, labels=[0, 1]).ravel()
            precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else None
            recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else None
            fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else None
            fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else None
            threshold_table.append({
                "threshold": _fin(th, 2),
                "precision": _fin(precision_t, 4) if precision_t is not None else None,
                "recall": _fin(recall_t, 4) if recall_t is not None else None,
                "fpr": _fin(fpr_t, 4) if fpr_t is not None else None,
                "fnr": _fin(fnr_t, 4) if fnr_t is not None else None,
            })

        # --- Section 12: forecast ---
        if trend_table and len(trend_table) >= 3:
            rates = [r["churn_rate"] for r in trend_table if r["churn_rate"] is not None]
            xs = np.arange(len(rates))
            if len(rates) >= 2:
                slope, intercept = np.polyfit(xs, rates, 1)
            else:
                slope, intercept = 0.0, rates[0] if rates else churn_rate
            last_x = xs[-1] if len(xs) else 0
            forecast_method = "trend"
            formula_note = (
                "Linear extrapolation of the per-period churn rate observed in the Churn Risk "
                "Trend section (section 9), projected forward by 1/2/3 additional periods "
                "(treated as ~30/60/90 days) and applied to current customer count."
            )
            horizons = {}
            for label, steps in (("30d", 1), ("60d", 2), ("90d", 3)):
                rate_proj = max(0.0, min(1.0, float(slope * (last_x + steps) + intercept)))
                horizons[label] = {"expected_churn_rate": _fin(rate_proj, 4), "expected_churn_count": int(round(rate_proj * n_total))}
        else:
            forecast_method = "risk_tier_decay"
            formula_note = (
                "No real time series was available (Churn Risk Trend was skipped), so this is "
                "probability-distribution-based, not time-series-based: for each risk tier, an "
                "assumed 90-day cumulative churn probability (Low 5%, Medium 15%, High 35%, "
                "Critical 60%) is scaled to 30/60/90 days via a simple exponential-decay share "
                "(30d = 1-e^(-1/3 * lambda), 60d = 1-e^(-2/3*lambda), 90d = full assumed value) "
                "and summed across each tier's customer count."
            )
            horizons = {}
            for label, frac in (("30d", 1 / 3), ("60d", 2 / 3), ("90d", 1.0)):
                expected = 0.0
                for t in TIER_ORDER:
                    p90 = TIER_DECAY_90D[t]
                    lam = -np.log(1 - p90) if p90 < 1 else 5.0
                    p_h = 1 - np.exp(-lam * frac)
                    expected += tier_counts[t] * p_h
                horizons[label] = {"expected_churn_rate": _fin(expected / n_total, 4), "expected_churn_count": int(round(expected))}
        forecast = {"method": forecast_method, "formula_note": formula_note, "horizons": horizons}

        # --- Charts ---
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(proba_all, bins=20, color=BLUE, edgecolor='white')
        ax.set_xlabel('Predicted churn probability'); ax.set_ylabel('Customers')
        ax.set_title('Churn Probability Distribution')
        chart_prob_dist = _png(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {best['auc']:.3f}")
        ax.plot([0, 1], [0, 1], color=GRAY, ls='--', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {recommended_model_label}'); ax.legend(loc='lower right')
        chart_roc = _png(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec_curve, prec_curve, color=GREEN, lw=2, label=f"AP = {avg_precision:.3f}")
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve — {recommended_model_label}'); ax.legend(loc='lower left')
        chart_pr = _png(fig)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha='center', va='center', color='black', fontsize=14)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Retained', 'Churned']); ax.set_yticklabels(['Retained', 'Churned'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix (threshold 0.5)')
        chart_confusion = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        top10 = feature_importance[:10][::-1]
        colors = [RED if f["direction"] == "increases" else BLUE for f in top10]
        ax.barh([f["feature"] for f in top10], [f["importance"] for f in top10], color=colors)
        ax.set_xlabel('Importance'); ax.set_title(f'Churn Drivers — {recommended_model_label} ({importance_kind})')
        chart_drivers = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        model_names = [m["model"] for m in model_comparison]
        aucs = [m["auc"] for m in model_comparison]
        ax.bar(model_names, aucs, color=[BLUE, GREEN, AMBER][:len(model_names)])
        ax.set_ylabel('AUC'); ax.set_ylim(0, 1); ax.set_title('Model Comparison (AUC)')
        chart_model_comparison = _png(fig)

        chart_trend = None
        if trend_table:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            periods = [r["period"] for r in trend_table]
            ax.plot(periods, [r["churn_rate"] for r in trend_table], marker='o', color=RED, label='Churn rate')
            ax.set_ylabel('Churn rate'); ax.set_xlabel('Period'); plt.xticks(rotation=45, ha='right')
            ax.set_title('Churn Risk Trend'); ax.legend()
            chart_trend = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([t["threshold"] for t in threshold_table], [t["precision"] for t in threshold_table], marker='o', color=BLUE, label='Precision')
        ax.plot([t["threshold"] for t in threshold_table], [t["recall"] for t in threshold_table], marker='o', color=GREEN, label='Recall')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Score'); ax.set_title('Precision / Recall vs Threshold')
        ax.legend()
        chart_threshold = _png(fig)

        charts = {
            "prob_distribution": chart_prob_dist,
            "roc": chart_roc,
            "pr": chart_pr,
            "confusion": chart_confusion,
            "drivers": chart_drivers,
            "model_comparison": chart_model_comparison,
            "trend": chart_trend,
            "threshold": chart_threshold,
        }

        results = {
            "status": "ok",
            "n_total": n_total,
            "gb_engine": gb_engine,
            "overview": overview,
            "prob_distribution": prob_distribution,
            "risk_tier_table": risk_tier_table,
            "customer_table": customer_table,
            "revenue_at_risk": revenue_at_risk_total,
            "model_comparison": model_comparison,
            "recommended_model": recommended_model_label,
            "recommended_model_key": best_key,
            "perf_metrics": perf_metrics,
            "roc_curve": {"fpr": [_fin(v, 4) for v in fpr], "tpr": [_fin(v, 4) for v in tpr]},
            "pr_curve": {"precision": [_fin(v, 4) for v in prec_curve], "recall": [_fin(v, 4) for v in rec_curve], "average_precision": avg_precision},
            "confusion_matrix": confusion,
            "feature_importance": feature_importance,
            "explainability": {"used_shap": shap_used, "note": explainability_note, "customers": explanations},
            "trend_table": trend_table,
            "trend_note": trend_note,
            "time_col_used": time_col,
            "characteristics": characteristics,
            "threshold_table": [t for t in threshold_table],
            "forecast": forecast,
            "charts": charts,
        }

        print(json.dumps({"results": results}))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
