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

from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SEGMENT_DETAIL_CAP = 50


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


def two_prop_z(n1, x1, n2, x2):
    """n1/x1 = control, n2/x2 = treatment. Returns z, p (two-sided)."""
    count = np.array([x2, x1])
    nobs = np.array([n2, n1])
    z, p = proportions_ztest(count, nobs)
    return float(z), float(p)


def lift_ci(n1, p1, n2, p2, alpha=0.05):
    """95% CI for absolute lift (p2 - p1), Wald normal approx."""
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    return float(diff - z_crit * se), float(diff + z_crit * se), float(se)


def required_n_per_group(p1, mde, alpha=0.05, power=0.8):
    p2 = min(0.999, p1 + mde)
    effect_size = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    if effect_size == 0:
        return None
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0, alternative="two-sided")
    return float(np.ceil(n))


def achieved_power(n1, p1, n2, p2, alpha=0.05):
    effect_size = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    if effect_size == 0:
        return 0.0
    analysis = NormalIndPower()
    pw = analysis.power(effect_size=abs(effect_size), nobs1=n1, ratio=n2 / n1, alpha=alpha, alternative="two-sided")
    return float(np.clip(pw, 0, 1))


def beta_binomial_bayes(x1, n1, x2, n2, prior_a=1.0, prior_b=1.0, n_sim=200000, seed=42):
    """Beta(1,1) uniform prior (documented). Monte-Carlo posterior comparison."""
    rng = np.random.default_rng(seed)
    post_control = rng.beta(prior_a + x1, prior_b + (n1 - x1), n_sim)
    post_treat = rng.beta(prior_a + x2, prior_b + (n2 - x2), n_sim)
    prob_treat_better = float(np.mean(post_treat > post_control))
    lift_samples = post_treat - post_control
    ci_low, ci_high = np.percentile(lift_samples, [2.5, 97.5])
    return prob_treat_better, float(np.mean(lift_samples)), float(ci_low), float(ci_high)


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get("data")
        group_col = payload.get("group_col")
        control_value = payload.get("control_value")
        treatment_value = payload.get("treatment_value")
        outcome_col = payload.get("outcome_col")
        secondary_cols = payload.get("secondary_metric_cols") or []
        segment_col = payload.get("segment_col") or None
        time_col = payload.get("time_col") or None
        alpha = float(payload.get("alpha") or 0.05)
        power_target = float(payload.get("power_target") or 0.8)
        mde_list = payload.get("mde_list") or [0.01, 0.02, 0.03, 0.05]

        if not data or not group_col or not outcome_col or control_value is None or treatment_value is None:
            raise ValueError("Provide 'data', 'group_col', 'control_value', 'treatment_value' and 'outcome_col'")

        df = pd.DataFrame(data)
        if group_col not in df.columns or outcome_col not in df.columns:
            raise ValueError("group_col or outcome_col not found in data")

        df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
        df = df.dropna(subset=[group_col, outcome_col])
        df[group_col] = df[group_col].astype(str)
        control_value = str(control_value)
        treatment_value = str(treatment_value)

        ctrl = df[df[group_col] == control_value]
        trt = df[df[group_col] == treatment_value]
        if len(ctrl) < 2 or len(trt) < 2:
            raise ValueError("Need at least 2 rows in each of the control and treatment groups")

        n1, n2 = int(len(ctrl)), int(len(trt))
        x1, x2 = int(ctrl[outcome_col].sum()), int(trt[outcome_col].sum())
        p1, p2 = x1 / n1, x2 / n2

        abs_lift = p2 - p1
        rel_lift = (abs_lift / p1 * 100) if p1 > 0 else None

        z_stat, p_value = two_prop_z(n1, x1, n2, x2)
        significant = bool(p_value < alpha)
        ci_low, ci_high, se = lift_ci(n1, p1, n2, p2, alpha)

        # ---- ① Overview handled by fields above ----

        # ---- ③ Primary metric table ----
        primary_rows = [
            {"group": "Control", "users": n1, "conversions": x1, "conversion_rate": round(p1 * 100, 3)},
            {"group": "Treatment", "users": n2, "conversions": x2, "conversion_rate": round(p2 * 100, 3)},
        ]

        # ---- ④ Lift analysis ----
        incremental_conversions = float(x2 - round(n2 * p1))
        lift_rows = [{
            "metric": "Conversion rate",
            "absolute_lift_pct_pt": round(abs_lift * 100, 3),
            "relative_lift_pct": round(rel_lift, 2) if rel_lift is not None else None,
            "estimated_incremental_conversions": incremental_conversions,
        }]

        # ---- ⑤ Significance ----
        significance = {
            "test": "two-proportion z-test",
            "z_statistic": z_stat,
            "p_value": p_value,
            "alpha": alpha,
            "significant": significant,
        }

        # ---- ⑥ Confidence interval ----
        ci_result = {
            "point_estimate_pct_pt": round(abs_lift * 100, 3),
            "ci_lower_pct_pt": round(ci_low * 100, 3),
            "ci_upper_pct_pt": round(ci_high * 100, 3),
            "alpha": alpha,
        }

        # ---- ⑦/⑧ Trend + cumulative (conditional on time_col) ----
        trend = None
        cumulative = None
        if time_col and time_col in df.columns:
            tdf = df.copy()
            tdf[time_col] = pd.to_datetime(tdf[time_col], errors="coerce")
            tdf = tdf.dropna(subset=[time_col])
            if len(tdf) > 0 and tdf[time_col].nunique() > 1:
                tdf["_date"] = tdf[time_col].dt.date
                daily = (
                    tdf.groupby(["_date", group_col])[outcome_col]
                    .agg(["sum", "count"])
                    .reset_index()
                )
                daily["rate"] = daily["sum"] / daily["count"]
                dates = sorted(daily["_date"].unique())
                ctrl_daily = daily[daily[group_col] == control_value].set_index("_date")
                trt_daily = daily[daily[group_col] == treatment_value].set_index("_date")
                trend = {
                    "dates": [str(d) for d in dates],
                    "control_rate": [float(ctrl_daily["rate"].get(d, np.nan)) * 100 if d in ctrl_daily.index else None for d in dates],
                    "treatment_rate": [float(trt_daily["rate"].get(d, np.nan)) * 100 if d in trt_daily.index else None for d in dates],
                }

                # cumulative
                cum_ctrl_n, cum_ctrl_x, cum_trt_n, cum_trt_x = 0, 0, 0, 0
                cum_lift = []
                for d in dates:
                    if d in ctrl_daily.index:
                        cum_ctrl_n += int(ctrl_daily.loc[d, "count"])
                        cum_ctrl_x += int(ctrl_daily.loc[d, "sum"])
                    if d in trt_daily.index:
                        cum_trt_n += int(trt_daily.loc[d, "count"])
                        cum_trt_x += int(trt_daily.loc[d, "sum"])
                    cp1 = cum_ctrl_x / cum_ctrl_n if cum_ctrl_n else None
                    cp2 = cum_trt_x / cum_trt_n if cum_trt_n else None
                    cum_lift.append(((cp2 - cp1) * 100) if (cp1 is not None and cp2 is not None) else None)
                cumulative = {"dates": [str(d) for d in dates], "cumulative_lift_pct_pt": cum_lift}
        time_note = None if trend else "No time/date column provided (or fewer than 2 distinct dates) — trend and cumulative analysis skipped."

        # ---- ⑨ Sample size & power analysis ----
        mde_rows = []
        for mde in mde_list:
            n_req = required_n_per_group(p1, mde, alpha, power_target)
            mde_rows.append({"mde_pct_pt": round(mde * 100, 2), "required_n_per_group": n_req})
        power_analysis = {
            "baseline_rate": round(p1 * 100, 3),
            "alpha": alpha,
            "target_power": power_target,
            "mde_table": mde_rows,
        }

        # ---- ⑩ Achieved (actual) power ----
        ach_power = achieved_power(n1, p1, n2, p2, alpha)
        power_verdict = "sufficient" if ach_power >= 0.8 else "underpowered"
        power_result = {
            "achieved_power": round(ach_power, 4),
            "n_control": n1,
            "n_treatment": n2,
            "observed_effect_abs": round(abs_lift, 5),
            "alpha": alpha,
            "verdict": power_verdict,
        }

        # ---- ⑪ Segment-level results (conditional) ----
        segment_results = None
        segment_note = None
        if segment_col and segment_col in df.columns:
            seg_rows = []
            for seg_val, seg_df in df.groupby(segment_col):
                sc = seg_df[seg_df[group_col] == control_value]
                st = seg_df[seg_df[group_col] == treatment_value]
                if len(sc) < 2 or len(st) < 2:
                    continue
                sn1, sn2 = len(sc), len(st)
                sx1, sx2 = int(sc[outcome_col].sum()), int(st[outcome_col].sum())
                sp1, sp2 = sx1 / sn1, sx2 / sn2
                try:
                    sz, sp = two_prop_z(sn1, sx1, sn2, sx2)
                except Exception:
                    sz, sp = None, None
                seg_rows.append({
                    "segment": str(seg_val),
                    "control_n": sn1, "treatment_n": sn2,
                    "control_rate": round(sp1 * 100, 3), "treatment_rate": round(sp2 * 100, 3),
                    "lift_pct_pt": round((sp2 - sp1) * 100, 3),
                    "p_value": sp,
                })
            if seg_rows:
                segment_results = seg_rows
            else:
                segment_note = "Segment column provided but no segment had at least 2 rows per group — segment analysis skipped."
        else:
            segment_note = "No segment column provided — segment-level analysis skipped."

        # ---- ⑫ Secondary metrics (conditional) ----
        secondary_results = None
        if secondary_cols:
            sec_rows = []
            for col in secondary_cols:
                if col not in df.columns:
                    continue
                sc = pd.to_numeric(ctrl[col], errors="coerce").dropna()
                st = pd.to_numeric(trt[col], errors="coerce").dropna()
                if len(sc) < 2 or len(st) < 2:
                    continue
                m1, m2 = float(sc.mean()), float(st.mean())
                tstat, tp = stats.ttest_ind(st, sc, equal_var=False)
                lift_pct = ((m2 - m1) / m1 * 100) if m1 != 0 else None
                # heuristic: "refund"/"cost"/"churn"/"complaint" style columns -> lower is better
                undesirable_up = bool(__import__("re").search(r"refund|cost|churn|complaint|cancel", col, __import__("re").IGNORECASE))
                flag = "watch" if (undesirable_up and m2 > m1) else "ok"
                sec_rows.append({
                    "metric": col, "control_mean": round(m1, 4), "treatment_mean": round(m2, 4),
                    "lift_pct": round(lift_pct, 2) if lift_pct is not None else None,
                    "t_statistic": float(tstat), "p_value": float(tp),
                    "significant": bool(tp < alpha), "flag": flag,
                })
            secondary_results = sec_rows if sec_rows else None

        # ---- ⑬ Bayesian A/B test ----
        prob_treat_better, bayes_mean_lift, bayes_ci_low, bayes_ci_high = beta_binomial_bayes(x1, n1, x2, n2)
        bayesian = {
            "prior": "Beta(1,1) uniform prior",
            "prob_treatment_better": round(prob_treat_better, 4),
            "expected_lift_abs": round(bayes_mean_lift, 5),
            "credible_interval_95": [round(bayes_ci_low, 5), round(bayes_ci_high, 5)],
        }

        # ---- ⑭ Multi-variant test (conditional on 3+ groups) ----
        multi_variant = None
        multi_variant_note = None
        all_groups = df[group_col].unique().tolist()
        if len(all_groups) >= 3:
            others = [g for g in all_groups if g != control_value]
            variant_rows = []
            raw_pvals = []
            for g in [control_value] + others:
                gd = df[df[group_col] == g]
                gn, gx = len(gd), int(gd[outcome_col].sum())
                gp = gx / gn if gn else 0
                if g == control_value:
                    variant_rows.append({"variant": g, "n": gn, "conversions": gx, "conversion_rate": round(gp * 100, 3), "lift_vs_control_pct_pt": 0.0, "p_value_raw": None})
                else:
                    try:
                        _, gpval = two_prop_z(n1, x1, gn, gx)
                    except Exception:
                        gpval = 1.0
                    raw_pvals.append(gpval)
                    variant_rows.append({"variant": g, "n": gn, "conversions": gx, "conversion_rate": round(gp * 100, 3), "lift_vs_control_pct_pt": round((gp - p1) * 100, 3), "p_value_raw": gpval})
            if raw_pvals:
                _, bonf, _, _ = multipletests(raw_pvals, alpha=alpha, method="bonferroni")
                _, holm, _, _ = multipletests(raw_pvals, alpha=alpha, method="holm")
                idx = 0
                for row in variant_rows:
                    if row["variant"] != control_value:
                        row["p_value_bonferroni"] = float(bonf[idx])
                        row["p_value_holm"] = float(holm[idx])
                        row["significant_bonferroni"] = bool(bonf[idx] < alpha)
                        row["significant_holm"] = bool(holm[idx] < alpha)
                        idx += 1
                    else:
                        row["p_value_bonferroni"] = None
                        row["p_value_holm"] = None
                        row["significant_bonferroni"] = None
                        row["significant_holm"] = None
            multi_variant = variant_rows
        else:
            multi_variant_note = "Group column has only 2 distinct values — multi-variant test skipped (would fabricate variants otherwise)."

        # ---- ⑮ Conclusion (interpretation string) ----
        practical_sig = abs(rel_lift) >= 5 if rel_lift is not None else False
        side_effect_flags = [r["metric"] for r in (secondary_results or []) if r["flag"] == "watch"]
        interpretation = (
            f"The treatment group converted at {p2*100:.2f}% vs {p1*100:.2f}% for control, an absolute lift of "
            f"{abs_lift*100:.2f} percentage points ({rel_lift:.1f}% relative lift). This result is "
            f"{'statistically significant' if significant else 'not statistically significant'} "
            f"(two-proportion z-test, p = {p_value:.4f}, alpha = {alpha}), and the achieved power of this experiment "
            f"is {ach_power*100:.1f}% ({power_verdict}). "
            f"The Bayesian analysis estimates a {prob_treat_better*100:.1f}% probability that treatment truly beats control. "
            + (f"The lift also appears practically meaningful (≥5% relative). " if practical_sig else "The lift is small in practical terms even if statistically detectable. ")
            + (f"Caution: {', '.join(side_effect_flags)} moved in an undesirable direction alongside the primary metric — review before rolling out. " if side_effect_flags else "")
            + ("Recommend shipping the treatment." if (significant and practical_sig and not side_effect_flags) else "Recommend further validation before shipping.")
        )

        # ---- Charts ----
        charts = {}

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["Control", "Treatment"], [p1 * 100, p2 * 100], color=["#94a3b8", "#2563eb"])
        ax.set_ylabel("Conversion rate (%)")
        ax.set_title("Primary Metric: Conversion Rate")
        for i, v in enumerate([p1 * 100, p2 * 100]):
            ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
        charts["primary_metric"] = _png(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Absolute lift (pp)", "Relative lift (%)"], [abs_lift * 100, rel_lift or 0], color=["#16a34a" if abs_lift >= 0 else "#dc2626", "#2563eb"])
        ax.set_title("Lift Analysis")
        ax.axhline(0, color="black", linewidth=0.8)
        charts["lift"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.errorbar([abs_lift * 100], [0], xerr=[[abs_lift * 100 - ci_low * 100], [ci_high * 100 - abs_lift * 100]], fmt="o", color="#2563eb", capsize=6, markersize=8)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_yticks([])
        ax.set_xlabel("Lift (percentage points)")
        ax.set_title(f"95% CI for Lift [{ci_low*100:.2f}, {ci_high*100:.2f}]")
        charts["confidence_interval"] = _png(fig)

        if trend:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(trend["dates"], trend["control_rate"], marker="o", label="Control", color="#94a3b8")
            ax.plot(trend["dates"], trend["treatment_rate"], marker="o", label="Treatment", color="#2563eb")
            ax.set_ylabel("Conversion rate (%)")
            ax.set_title("Conversion Trend")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend()
            charts["trend"] = _png(fig)

        if cumulative:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(cumulative["dates"], cumulative["cumulative_lift_pct_pt"], marker="o", color="#7c3aed")
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_ylabel("Cumulative lift (pp)")
            ax.set_title("Cumulative Lift Over Time")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            charts["cumulative"] = _png(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        mdes = [r["mde_pct_pt"] for r in mde_rows]
        ns = [r["required_n_per_group"] or 0 for r in mde_rows]
        ax.bar([f"{m}%" for m in mdes], ns, color="#0891b2")
        ax.set_xlabel("Minimum Detectable Effect (pp)")
        ax.set_ylabel("Required n per group")
        ax.set_title(f"Sample Size vs MDE (power={power_target})")
        charts["sample_size_mde"] = _png(fig)

        if segment_results:
            fig, ax = plt.subplots(figsize=(7, 5))
            segs = [r["segment"] for r in segment_results]
            lifts = [r["lift_pct_pt"] for r in segment_results]
            colors = ["#16a34a" if l >= 0 else "#dc2626" for l in lifts]
            ax.barh(segs, lifts, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Lift (pp)")
            ax.set_title("Segment-Level Lift")
            charts["segment"] = _png(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        rng = np.random.default_rng(1)
        pc = rng.beta(1 + x1, 1 + (n1 - x1), 20000)
        pt = rng.beta(1 + x2, 1 + (n2 - x2), 20000)
        ax.hist(pc, bins=60, alpha=0.6, label="Control posterior", color="#94a3b8", density=True)
        ax.hist(pt, bins=60, alpha=0.6, label="Treatment posterior", color="#2563eb", density=True)
        ax.set_title(f"Bayesian Posteriors (P(T>C)={prob_treat_better*100:.1f}%)")
        ax.legend()
        charts["bayesian"] = _png(fig)

        if multi_variant:
            fig, ax = plt.subplots(figsize=(7, 5))
            names = [r["variant"] for r in multi_variant]
            rates = [r["conversion_rate"] for r in multi_variant]
            ax.bar(names, rates, color="#0891b2")
            ax.set_ylabel("Conversion rate (%)")
            ax.set_title("Multi-Variant Conversion Rates")
            charts["multi_variant"] = _png(fig)

        results = {
            "overview": {
                "control_n": n1, "treatment_n": n2,
                "control_rate": round(p1 * 100, 3), "treatment_rate": round(p2 * 100, 3),
                "absolute_lift_pct_pt": round(abs_lift * 100, 3),
                "relative_lift_pct": round(rel_lift, 2) if rel_lift is not None else None,
                "significant": significant, "p_value": p_value,
                "ci_lower_pct_pt": round(ci_low * 100, 3), "ci_upper_pct_pt": round(ci_high * 100, 3),
            },
            "setup": {
                "group_col": group_col, "control_value": control_value, "treatment_value": treatment_value,
                "outcome_col": outcome_col, "secondary_metric_cols": secondary_cols,
                "segment_col": segment_col, "time_col": time_col,
            },
            "primary_metric": primary_rows,
            "lift": lift_rows,
            "significance": significance,
            "confidence_interval": ci_result,
            "trend": trend,
            "cumulative": cumulative,
            "time_note": time_note,
            "power_analysis": power_analysis,
            "achieved_power": power_result,
            "segment_results": segment_results,
            "segment_note": segment_note,
            "secondary_results": secondary_results,
            "bayesian": bayesian,
            "multi_variant": multi_variant,
            "multi_variant_note": multi_variant_note,
            "interpretation": interpretation,
            "charts": charts,
        }

        print(json.dumps({"results": results, "plot": charts.get("primary_metric")}, default=_native))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
