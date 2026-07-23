import sys
import json
import io
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def cumulative_gain_curve(uplift_scores, outcome, treatment, n_points=20):
    """Standard Qini-style cumulative gain: sort by predicted uplift desc, at each
    fraction f of population compute g(f) = sum(treated conversions in top-f) -
    sum(control conversions in top-f) * (n_treated_top_f / n_control_top_f).
    Returns (pop_fraction[], gain[], random_gain[])."""
    order = np.argsort(-uplift_scores)
    outcome = np.asarray(outcome)[order]
    treatment = np.asarray(treatment)[order]
    n = len(outcome)
    fracs = np.linspace(1.0 / n_points, 1.0, n_points)
    gains = []
    for f in fracs:
        k = max(1, int(round(f * n)))
        t_mask = treatment[:k] == 1
        c_mask = treatment[:k] == 0
        n_t, n_c = t_mask.sum(), c_mask.sum()
        conv_t = outcome[:k][t_mask].sum()
        conv_c = outcome[:k][c_mask].sum()
        if n_c > 0:
            g = conv_t - conv_c * (n_t / n_c)
        else:
            g = conv_t
        gains.append(float(g))
    total_gain = gains[-1]
    random_gains = [total_gain * f for f in fracs]
    return fracs.tolist(), gains, random_gains


def qini_and_auuc(uplift_scores, outcome, treatment, n_points=50):
    fracs, gains, random_gains = cumulative_gain_curve(uplift_scores, outcome, treatment, n_points)
    # Qini coefficient: area between model curve and random-targeting diagonal
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    area_model = _trapz(gains, fracs)
    area_random = _trapz(random_gains, fracs)
    qini_coef = float(area_model - area_random)
    # AUUC: area under the (model) uplift/gain curve itself
    auuc = float(area_model)
    return qini_coef, auuc, fracs, gains, random_gains


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get("data")
        treatment_col = payload.get("treatment_col")
        treatment_value = payload.get("treatment_value")
        control_value = payload.get("control_value")
        outcome_col = payload.get("outcome_col")
        feature_cols = payload.get("feature_cols") or []
        segment_col = payload.get("segment_col") or None
        cost_per_treatment = payload.get("cost_per_treatment")
        revenue_per_conversion = payload.get("revenue_per_conversion")
        cost_per_treatment = float(cost_per_treatment) if cost_per_treatment not in (None, "") else None
        revenue_per_conversion = float(revenue_per_conversion) if revenue_per_conversion not in (None, "") else None

        if not data or not treatment_col or not outcome_col:
            raise ValueError("Provide 'data', 'treatment_col', and 'outcome_col'")

        df = pd.DataFrame(data)
        if treatment_col not in df.columns or outcome_col not in df.columns:
            raise ValueError("treatment_col or outcome_col not found in data")

        df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
        df = df.dropna(subset=[outcome_col])
        df[treatment_col] = df[treatment_col].astype(str)

        levels = df[treatment_col].unique().tolist()
        if control_value is None or treatment_value is None:
            treatment_value = str(treatment_value) if treatment_value is not None else next(
                (l for l in levels if str(l).lower() in ("1", "true", "yes", "treatment", "treated")), None
            )
            if treatment_value is None:
                levels_sorted = sorted(levels)
                treatment_value = levels_sorted[-1]
            control_value = next((l for l in levels if l != treatment_value), None)
        else:
            treatment_value, control_value = str(treatment_value), str(control_value)

        if treatment_value not in levels or control_value not in levels:
            raise ValueError("treatment_value/control_value not found among treatment_col's values")

        df = df[df[treatment_col].isin([treatment_value, control_value])].copy()
        df["_T"] = (df[treatment_col] == treatment_value).astype(int)

        if not feature_cols:
            feature_cols = [
                c for c in df.columns
                if c not in (treatment_col, outcome_col, "_T", segment_col) and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
            ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            raise ValueError("No usable numeric feature columns found for uplift modeling")

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=feature_cols)

        n_total = len(df)
        n_treat = int(df["_T"].sum())
        n_ctrl = int(n_total - n_treat)
        if n_treat < 10 or n_ctrl < 10:
            raise ValueError("Need at least 10 rows in both treatment and control groups")

        X = df[feature_cols].values
        y = df[outcome_col].astype(int).values
        T = df["_T"].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # ---- T-Learner: two separate models, one per arm ----
        model_t1 = LogisticRegression(max_iter=1000)
        model_t0 = LogisticRegression(max_iter=1000)
        model_t1.fit(Xs[T == 1], y[T == 1])
        model_t0.fit(Xs[T == 0], y[T == 0])
        p1_t = model_t1.predict_proba(Xs)[:, 1]
        p0_t = model_t0.predict_proba(Xs)[:, 1]
        uplift_t = p1_t - p0_t

        # ---- S-Learner: single model with treatment as a feature ----
        Xs_with_t1 = np.hstack([Xs, np.ones((n_total, 1))])
        Xs_with_t0 = np.hstack([Xs, np.zeros((n_total, 1))])
        Xs_with_t_actual = np.hstack([Xs, T.reshape(-1, 1).astype(float)])
        model_s = LogisticRegression(max_iter=1000)
        model_s.fit(Xs_with_t_actual, y)
        p1_s = model_s.predict_proba(Xs_with_t1)[:, 1]
        p0_s = model_s.predict_proba(Xs_with_t0)[:, 1]
        uplift_s = p1_s - p0_s

        # ---- Model comparison: Qini / AUUC for S vs T learner ----
        qini_t, auuc_t, fracs_t, gains_t, rand_t = qini_and_auuc(uplift_t, y, T)
        qini_s, auuc_s, fracs_s, gains_s, rand_s = qini_and_auuc(uplift_s, y, T)

        model_comparison = [
            {"model": "T-Learner", "qini_coefficient": round(qini_t, 4), "auuc": round(auuc_t, 4)},
            {"model": "S-Learner", "qini_coefficient": round(qini_s, 4), "auuc": round(auuc_s, 4)},
        ]
        primary = "T-Learner"

        # Primary per-customer scores use T-Learner (standard, avoids S-learner's
        # tendency to shrink the treatment coefficient toward 0 for weak-signal features).
        uplift = uplift_t
        p1, p0 = p1_t, p0_t

        # ---- ④ Uplift Segments (2x2 framework) ----
        overall_rate = float(y.mean())
        thr = overall_rate  # split point: overall observed conversion rate
        high1, high0 = p1 >= thr, p0 >= thr

        def classify(h1, h0):
            if (not h0) and h1:
                return "Persuadables"
            if h0 and h1:
                return "Sure Things"
            if (not h0) and (not h1):
                return "Lost Causes"
            return "Sleeping Dogs"  # h0 and not h1

        segment_labels = np.array([classify(h1, h0) for h1, h0 in zip(high1, high0)])
        strategy_map = {"Persuadables": "Target", "Sure Things": "Avoid", "Lost Causes": "Avoid", "Sleeping Dogs": "Suppress"}

        seg_summary = []
        for seg in ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]:
            mask = segment_labels == seg
            cnt = int(mask.sum())
            avg_up = float(uplift[mask].mean()) if cnt else 0.0
            seg_summary.append({
                "segment": seg, "count": cnt,
                "pct_of_total": round(cnt / n_total * 100, 2) if n_total else 0.0,
                "avg_uplift": round(avg_up, 4),
                "recommended_strategy": strategy_map[seg],
            })

        n_targetable = int((segment_labels == "Persuadables").sum())

        # ---- ⑤ Per-customer table (cap rows for payload size) ----
        id_col = next((c for c in df.columns if c.lower() in ("customer_id", "id", "customerid")), None)
        cust_ids = df[id_col].astype(str).tolist() if id_col else [f"CUST_{i:05d}" for i in range(n_total)]
        per_customer_full = pd.DataFrame({
            "customer_id": cust_ids,
            "predicted_uplift": np.round(uplift, 4),
            "treatment_prob": np.round(p1, 4),
            "control_prob": np.round(p0, 4),
            "segment": segment_labels,
        })
        CAP = 300
        per_customer_table = per_customer_full.sort_values("predicted_uplift", ascending=False).head(CAP).to_dict("records")

        # ---- ⑦/⑧ curves (already computed above with 50 points) ----
        uplift_curve = {"pop_fraction": [round(f, 4) for f in fracs_t], "gain": [round(g, 3) for g in gains_t], "random_gain": [round(g, 3) for g in rand_t]}
        qini_curve = {"pop_fraction": [round(f, 4) for f in fracs_t], "cumulative_incremental_uplift": [round(g, 3) for g in gains_t], "random_baseline": [round(g, 3) for g in rand_t]}

        # ---- ⑨ Targeting policy ----
        order = np.argsort(-uplift)
        y_sorted, T_sorted, uplift_sorted = y[order], T[order], uplift[order]
        tiers = [0.10, 0.20, 0.30, 0.50, 1.00]
        targeting_policy = []
        for tier in tiers:
            k = max(1, int(round(tier * n_total)))
            expected_incremental = float(uplift_sorted[:k].sum())
            row = {
                "tier_pct": int(tier * 100), "n_targeted": k,
                "expected_incremental_conversions": round(expected_incremental, 2),
            }
            if cost_per_treatment is not None:
                row["total_cost"] = round(k * cost_per_treatment, 2)
            targeting_policy.append(row)
        targeting_note = None if cost_per_treatment is not None else "No cost-per-treatment provided — cost column omitted from the targeting policy table."

        # ---- ⑩ Incremental ROI (conditional) ----
        incremental_roi = None
        roi_note = None
        if cost_per_treatment is not None and revenue_per_conversion is not None:
            incremental_roi = []
            for tier in tiers:
                k = max(1, int(round(tier * n_total)))
                expected_incremental = float(uplift_sorted[:k].sum())
                incr_revenue = expected_incremental * revenue_per_conversion
                cost = k * cost_per_treatment
                roi = ((incr_revenue - cost) / cost * 100) if cost > 0 else None
                incremental_roi.append({
                    "tier_pct": int(tier * 100), "n_targeted": k,
                    "incremental_revenue": round(incr_revenue, 2), "cost": round(cost, 2),
                    "incremental_roi_pct": round(roi, 2) if roi is not None else None,
                })
        else:
            roi_note = "Provide cost_per_treatment and revenue_per_conversion to compute incremental ROI — skipped (no $ inputs given)."

        # ---- ⑪ Treatment effect by segment (conditional, uses GIVEN column only) ----
        segment_effect = None
        segment_effect_note = None
        if segment_col and segment_col in df.columns:
            rows = []
            for seg_val, idx in df.groupby(segment_col).groups.items():
                mask = df.index.isin(idx)
                rows.append({"group": str(seg_val), "n": int(mask.sum()), "avg_uplift": round(float(uplift[mask.values if hasattr(mask, 'values') else mask].mean()) if mask.sum() else 0.0, 4)})
            segment_effect = rows
        else:
            segment_effect_note = "No existing segment column provided — treatment-effect-by-segment breakdown skipped (no new clusters created here)."

        # ---- ⑫ Feature importance / uplift drivers ----
        # Heterogeneity proxy: |coef in treatment-arm model - coef in control-arm model|,
        # per standardized feature. Larger gap = feature more strongly modulates the
        # treatment effect (NOT how strongly it predicts the outcome itself).
        coef_gap = np.abs(model_t1.coef_[0] - model_t0.coef_[0])
        importance_rows = sorted(
            [{"feature": f, "heterogeneity_score": round(float(g), 4)} for f, g in zip(feature_cols, coef_gap)],
            key=lambda r: -r["heterogeneity_score"],
        )

        # ---- ⑬ deferred by explicit scope decision ----
        individual_and_policy_sim_note = "Deferred: prioritized sections ①-⑫ per explicit user scope guidance; individual treatment-effect explanations and the target-all/random/no-campaign policy simulation were cut from this build."

        # ---- ① Overview KPIs ----
        qini_coefficient = qini_t
        auuc = auuc_t
        overview = {
            "n_total": n_total, "n_treatment": n_treat, "n_control": n_ctrl,
            "average_uplift": round(float(uplift.mean()), 4),
            "n_targetable_persuadables": n_targetable,
            "qini_coefficient": round(qini_coefficient, 4),
            "auuc": round(auuc, 4),
        }

        # ---- ② basic conversion-rate table ----
        conv_table = [
            {"group": "Treatment", "n": n_treat, "conversions": int(y[T == 1].sum()), "conversion_rate": round(float(y[T == 1].mean()) * 100, 3)},
            {"group": "Control", "n": n_ctrl, "conversions": int(y[T == 0].sum()), "conversion_rate": round(float(y[T == 0].mean()) * 100, 3)},
        ]

        setup = {
            "treatment_col": treatment_col, "treatment_value": treatment_value, "control_value": control_value,
            "outcome_col": outcome_col, "feature_cols": feature_cols, "segment_col": segment_col,
        }

        # ---- Charts ----
        charts = {}

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(uplift, bins=40, color="#2563eb", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Predicted individual uplift")
        ax.set_ylabel("Customers")
        ax.set_title("Uplift Score Distribution")
        charts["uplift_distribution"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        segs = [r["segment"] for r in seg_summary]
        counts = [r["count"] for r in seg_summary]
        colors = {"Persuadables": "#16a34a", "Sure Things": "#94a3b8", "Lost Causes": "#64748b", "Sleeping Dogs": "#dc2626"}
        ax.bar(segs, counts, color=[colors[s] for s in segs])
        ax.set_ylabel("Customers")
        ax.set_title("Uplift Segments")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        charts["segments"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(uplift_curve["pop_fraction"], uplift_curve["gain"], color="#2563eb", label="Model (T-Learner)")
        ax.plot(uplift_curve["pop_fraction"], uplift_curve["random_gain"], color="#94a3b8", linestyle="--", label="Random targeting")
        ax.set_xlabel("% of population targeted")
        ax.set_ylabel("Cumulative incremental conversions")
        ax.set_title("Uplift Curve")
        ax.legend()
        charts["uplift_curve"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(qini_curve["pop_fraction"], qini_curve["cumulative_incremental_uplift"], color="#7c3aed", label="Model (T-Learner)")
        ax.plot(qini_curve["pop_fraction"], qini_curve["random_baseline"], color="#94a3b8", linestyle="--", label="Random targeting")
        ax.fill_between(qini_curve["pop_fraction"], qini_curve["cumulative_incremental_uplift"], qini_curve["random_baseline"], alpha=0.15, color="#7c3aed")
        ax.set_xlabel("% of population targeted")
        ax.set_ylabel("Cumulative incremental uplift")
        ax.set_title(f"Qini Curve (Qini={qini_coefficient:.3f})")
        ax.legend()
        charts["qini_curve"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        top_imp = importance_rows[:15]
        ax.barh([r["feature"] for r in top_imp][::-1], [r["heterogeneity_score"] for r in top_imp][::-1], color="#0891b2")
        ax.set_xlabel("Heterogeneity score |coef diff|")
        ax.set_title("Uplift Drivers (Effect Heterogeneity)")
        charts["feature_importance"] = _png(fig)

        results = {
            "overview": overview,
            "setup": setup,
            "conversion_table": conv_table,
            "uplift_distribution_stats": {
                "min": round(float(uplift.min()), 4), "max": round(float(uplift.max()), 4),
                "mean": round(float(uplift.mean()), 4), "std": round(float(uplift.std()), 4),
                "pct_negative": round(float((uplift < 0).mean() * 100), 2),
            },
            "segments": seg_summary,
            "per_customer": per_customer_table,
            "per_customer_note": f"Showing top {min(CAP, n_total)} of {n_total} customers by predicted uplift.",
            "model_comparison": model_comparison,
            "primary_model": primary,
            "uplift_curve": uplift_curve,
            "qini_curve": qini_curve,
            "targeting_policy": targeting_policy,
            "targeting_note": targeting_note,
            "incremental_roi": incremental_roi,
            "roi_note": roi_note,
            "segment_effect": segment_effect,
            "segment_effect_note": segment_effect_note,
            "feature_importance": importance_rows,
            "individual_and_policy_sim_note": individual_and_policy_sim_note,
            "charts": charts,
        }

        print(json.dumps({"results": results, "plot": charts.get("qini_curve")}, default=_native))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
