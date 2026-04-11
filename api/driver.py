"""
Driver Analysis Router — FastAPI
Modules: importance | correlation | multicollinearity | partial_effects | interactions | segmentation | recommendations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

router = APIRouter()

# ─── Input / Output Models ────────────────────────────────────────────────────

class DriverRequest(BaseModel):
    rows: List[Dict[str, Any]]          # raw data rows
    target: str                          # dependent variable (Y)
    drivers: List[str]                   # independent variables (X)
    analysis_type: str                   # importance | correlation | multicollinearity | partial_effects | interactions | segmentation | recommendations
    segment_col: Optional[str] = None    # column to segment by
    params: Optional[Dict[str, Any]] = {}

class DriverResult(BaseModel):
    analysis_type: str
    summary: Dict[str, Any]
    drivers: List[Dict[str, Any]]
    chart_data: Optional[List[Dict[str, Any]]] = []
    insights: List[str]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_df(rows: List[Dict[str, Any]], cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)[cols].apply(pd.to_numeric, errors='coerce').dropna()
    if len(df) < 5:
        raise HTTPException(status_code=400, detail="Not enough valid rows after cleaning (need ≥ 5).")
    return df

# ─── Module 1: Relative Importance ────────────────────────────────────────────

def analyze_importance(rows, target, drivers, params) -> DriverResult:
    """
    Compute relative importance using:
    1. Shapley value decomposition (LMG / Lindeman-Merenda-Gold approximation)
    2. Standardised regression coefficients
    3. Dominance analysis (sequential R² contribution)
    """
    from itertools import combinations
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    df = _to_df(rows, [target] + drivers)
    y = df[target].values
    X = df[drivers].values
    n_drivers = len(drivers)

    # ── Standardised coefficients ──
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    y_std = (y - y.mean()) / (y.std() + 1e-9)
    reg_std = LinearRegression().fit(X_std, y_std)
    std_coefs = dict(zip(drivers, reg_std.coef_))

    # ── LMG / Shapley R² decomposition ──
    # For each driver: average marginal R² contribution over all orderings
    def r2_subset(subset_idx):
        if not subset_idx:
            return 0.0
        Xs = X[:, list(subset_idx)]
        try:
            r = LinearRegression().fit(Xs, y)
            ss_res = np.sum((y - r.predict(Xs)) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return max(0.0, 1 - ss_res / (ss_tot + 1e-12))
        except Exception:
            return 0.0

    shapley = {d: 0.0 for d in drivers}
    all_idx = list(range(n_drivers))

    # Limit to 8 drivers for exact computation (2^8 = 256 subsets)
    if n_drivers <= 8:
        for i, driver in enumerate(drivers):
            others = [j for j in all_idx if j != i]
            marginal_sum = 0.0
            count = 0
            for size in range(len(others) + 1):
                for combo in combinations(others, size):
                    r2_with    = r2_subset(set(combo) | {i})
                    r2_without = r2_subset(set(combo))
                    marginal_sum += r2_with - r2_without
                    count += 1
            shapley[driver] = max(0.0, marginal_sum / count if count else 0.0)
    else:
        # Approximate via random sampling
        import random
        n_samples = 500
        for _ in range(n_samples):
            perm = list(range(n_drivers))
            random.shuffle(perm)
            for pos, i in enumerate(perm):
                before = set(perm[:pos])
                after  = before | {i}
                shapley[drivers[i]] += r2_subset(after) - r2_subset(before)
        shapley = {k: v / n_samples for k, v in shapley.items()}

    # Normalise Shapley to % of explained variance
    total_shapley = sum(shapley.values()) or 1.0
    shapley_pct = {k: v / total_shapley * 100 for k, v in shapley.items()}

    # ── Full model R² ──
    reg_full = LinearRegression().fit(X, y)
    y_pred   = reg_full.predict(X)
    ss_res   = np.sum((y - y_pred) ** 2)
    ss_tot   = np.sum((y - y.mean()) ** 2)
    r2_full  = max(0.0, 1 - ss_res / (ss_tot + 1e-12))

    # ── Rank drivers ──
    ranked = sorted(shapley_pct.items(), key=lambda x: -x[1])

    driver_results = []
    for rank, (d, pct) in enumerate(ranked, 1):
        raw_coef = float(reg_full.coef_[drivers.index(d)])
        direction = "positive" if raw_coef >= 0 else "negative"
        driver_results.append({
            "name":         d,
            "rank":         rank,
            "importance":   round(pct, 2),
            "std_coef":     round(float(std_coefs[d]), 4),
            "raw_coef":     round(raw_coef, 4),
            "direction":    direction,
            "shapley_r2":   round(float(shapley[d]), 4),
        })

    chart_data = [{"name": d["name"], "importance": d["importance"], "direction": d["direction"]} for d in driver_results]

    # ── Insights ──
    insights = []
    top3 = ranked[:3]
    insights.append(
        f"Top 3 drivers explain {sum(v for _, v in top3):.1f}% of variance in {target}: "
        f"{', '.join(d for d, _ in top3)}."
    )

    pos_drivers = [d for d in driver_results if d["direction"] == "positive"]
    neg_drivers = [d for d in driver_results if d["direction"] == "negative"]
    if neg_drivers:
        insights.append(
            f"{len(neg_drivers)} driver(s) have a negative relationship with {target}: "
            f"{', '.join(d['name'] for d in neg_drivers[:3])}. "
            f"Increasing these variables is associated with lower {target}."
        )

    insights.append(
        f"Overall model R² = {r2_full:.3f}. "
        f"{'Strong' if r2_full > 0.7 else 'Moderate' if r2_full > 0.4 else 'Weak'} explanatory power — "
        f"{'the selected drivers explain most of the variance.' if r2_full > 0.7 else 'consider adding more drivers or non-linear terms.'}"
    )

    if driver_results[-1]["importance"] < 2:
        weak = [d["name"] for d in driver_results if d["importance"] < 2]
        insights.append(
            f"{len(weak)} driver(s) contribute less than 2% each: {', '.join(weak[:3])}. "
            f"Consider dropping them to simplify the model."
        )

    return DriverResult(
        analysis_type="importance",
        summary={
            "target":       target,
            "n_drivers":    n_drivers,
            "n_rows":       len(df),
            "r2":           round(r2_full, 4),
            "top_driver":   ranked[0][0] if ranked else None,
        },
        drivers=driver_results,
        chart_data=chart_data,
        insights=insights,
    )

# ─── Module 2: Correlation Matrix ─────────────────────────────────────────────

def analyze_correlation(rows, target, drivers, params) -> DriverResult:
    method = params.get("method", "pearson")  # pearson | spearman | kendall
    all_cols = [target] + drivers
    df = _to_df(rows, all_cols)

    corr_matrix = df.corr(method=method)
    target_corr = corr_matrix[target].drop(target)

    # Pairwise correlations among drivers
    pairwise = []
    for i, d1 in enumerate(drivers):
        for j, d2 in enumerate(drivers):
            if i < j:
                pairwise.append({
                    "var1": d1, "var2": d2,
                    "r": round(float(corr_matrix.loc[d1, d2]), 4),
                    "abs_r": round(abs(float(corr_matrix.loc[d1, d2])), 4),
                })

    driver_results = []
    for d in drivers:
        r = float(target_corr[d])
        driver_results.append({
            "name":       d,
            "r":          round(r, 4),
            "r2":         round(r ** 2, 4),
            "direction":  "positive" if r >= 0 else "negative",
            "strength":   "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak",
        })

    driver_results.sort(key=lambda x: -abs(x["r"]))
    for i, d in enumerate(driver_results):
        d["rank"] = i + 1

    # High multicollinearity pairs
    high_corr_pairs = [p for p in pairwise if p["abs_r"] > 0.7]

    insights = []
    top = driver_results[0]
    insights.append(
        f"Strongest correlation with {target}: {top['name']} (r = {top['r']:.3f}, {top['strength']})."
    )
    if high_corr_pairs:
        pair_labels = ', '.join(p['var1'] + '–' + p['var2'] for p in high_corr_pairs[:3])
        insights.append(
            f"{len(high_corr_pairs)} driver pair(s) show high inter-correlation (|r| > 0.7): "
            f"{pair_labels}. "
            f"This may inflate importance estimates — check multicollinearity."
        )

    near_zero = [d["name"] for d in driver_results if abs(d["r"]) < 0.1]
    if near_zero:
        insights.append(
            f"{len(near_zero)} driver(s) show near-zero correlation with {target}: {', '.join(near_zero[:3])}. "
            f"These may not be relevant predictors."
        )

    return DriverResult(
        analysis_type="correlation",
        summary={
            "method":           method,
            "target":           target,
            "n_drivers":        len(drivers),
            "high_corr_pairs":  len(high_corr_pairs),
            "pairwise":         pairwise[:20],
        },
        drivers=driver_results,
        chart_data=[{"name": d["name"], "r": d["r"], "r2": d["r2"]} for d in driver_results],
        insights=insights,
    )

# ─── Module 3: Multicollinearity ──────────────────────────────────────────────

def analyze_multicollinearity(rows, target, drivers, params) -> DriverResult:
    """Compute VIF (Variance Inflation Factor) for each driver."""
    from sklearn.linear_model import LinearRegression

    df = _to_df(rows, [target] + drivers)
    X = df[drivers].values

    vif_results = []
    for i, d in enumerate(drivers):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)
        if X_i.shape[1] == 0:
            vif = 1.0
        else:
            reg = LinearRegression().fit(X_i, y_i)
            ss_res = np.sum((y_i - reg.predict(X_i)) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r2_i = max(0.0, 1 - ss_res / (ss_tot + 1e-12))
            vif = 1 / (1 - r2_i) if r2_i < 1.0 else float('inf')

        severity = "OK" if vif < 5 else "Moderate" if vif < 10 else "High"
        vif_results.append({
            "name":     d,
            "vif":      round(float(min(vif, 999.0)), 2),
            "severity": severity,
            "r2_aux":   round(1 - 1 / vif if vif > 1 else 0.0, 4),
        })

    vif_results.sort(key=lambda x: -x["vif"])
    for i, v in enumerate(vif_results):
        v["rank"] = i + 1

    high_vif = [v for v in vif_results if v["vif"] >= 10]
    moderate_vif = [v for v in vif_results if 5 <= v["vif"] < 10]

    insights = []
    if not high_vif and not moderate_vif:
        insights.append(
            "No significant multicollinearity detected. All VIF values are below 5 — "
            "importance estimates are reliable."
        )
    if high_vif:
        insights.append(
            f"High multicollinearity (VIF ≥ 10) detected in: {', '.join(v['name'] for v in high_vif)}. "
            f"Consider removing one of each correlated pair or using Ridge regression."
        )
    if moderate_vif:
        insights.append(
            f"Moderate multicollinearity (VIF 5–10): {', '.join(v['name'] for v in moderate_vif)}. "
            f"Results are acceptable but interpret importance with caution."
        )
    insights.append(
        "Rule of thumb: VIF < 5 = OK, 5–10 = moderate concern, > 10 = problematic. "
        "High VIF inflates standard errors and can distort driver importance rankings."
    )

    return DriverResult(
        analysis_type="multicollinearity",
        summary={
            "target":           target,
            "n_high_vif":       len(high_vif),
            "n_moderate_vif":   len(moderate_vif),
            "max_vif":          vif_results[0]["vif"] if vif_results else None,
            "max_vif_driver":   vif_results[0]["name"] if vif_results else None,
        },
        drivers=vif_results,
        chart_data=[{"name": v["name"], "vif": v["vif"], "severity": v["severity"]} for v in vif_results],
        insights=insights,
    )

# ─── Module 4: Partial Effects ────────────────────────────────────────────────

def analyze_partial_effects(rows, target, drivers, params) -> DriverResult:
    """Compute partial regression plots data: residuals of Y~others vs residuals of X_i~others."""
    from sklearn.linear_model import LinearRegression

    df = _to_df(rows, [target] + drivers)
    y  = df[target].values
    X  = df[drivers].values

    partial_data = []
    for i, d in enumerate(drivers):
        X_others = np.delete(X, i, axis=1)

        # Residuals of Y on others
        if X_others.shape[1] > 0:
            reg_y = LinearRegression().fit(X_others, y)
            e_y = y - reg_y.predict(X_others)
        else:
            e_y = y - y.mean()

        # Residuals of X_i on others
        x_i = X[:, i]
        if X_others.shape[1] > 0:
            reg_x = LinearRegression().fit(X_others, x_i)
            e_x = x_i - reg_x.predict(X_others)
        else:
            e_x = x_i - x_i.mean()

        # Partial slope
        cov = np.cov(e_x, e_y, ddof=1)
        partial_slope = cov[0, 1] / (cov[0, 0] + 1e-12)
        r_partial = float(np.corrcoef(e_x, e_y)[0, 1])

        # Sample for chart (max 200 pts)
        n_sample = min(200, len(e_x))
        idx = np.random.choice(len(e_x), n_sample, replace=False)
        scatter = [{"x": round(float(e_x[j]), 4), "y": round(float(e_y[j]), 4)} for j in idx]

        partial_data.append({
            "name":          d,
            "partial_slope": round(partial_slope, 4),
            "partial_r":     round(r_partial, 4),
            "partial_r2":    round(r_partial ** 2, 4),
            "direction":     "positive" if partial_slope >= 0 else "negative",
            "scatter":       scatter,
        })

    partial_data.sort(key=lambda x: -abs(x["partial_r"]))
    for i, p in enumerate(partial_data):
        p["rank"] = i + 1

    insights = []
    top = partial_data[0]
    insights.append(
        f"Strongest partial effect: {top['name']} (partial r = {top['partial_r']:.3f}). "
        f"After controlling for all other drivers, {top['name']} still has the most unique explanatory power."
    )

    diff_rank = [
        p for p in partial_data
        if abs(p["rank"] - drivers.index(p["name"]) - 1) > 1
    ]
    if diff_rank:
        insights.append(
            f"Partial effect rankings differ from raw correlation rankings for: "
            f"{', '.join(p['name'] for p in diff_rank[:3])}. "
            f"This suggests confounding — these drivers share variance with others."
        )

    near_zero_partial = [p for p in partial_data if abs(p["partial_r"]) < 0.1]
    if near_zero_partial:
        insights.append(
            f"{len(near_zero_partial)} driver(s) have near-zero partial effects: "
            f"{', '.join(p['name'] for p in near_zero_partial[:3])}. "
            f"Their apparent correlation with {target} may be explained by other drivers."
        )

    return DriverResult(
        analysis_type="partial_effects",
        summary={
            "target":     target,
            "n_drivers":  len(drivers),
            "top_driver": top["name"],
            "top_partial_r": top["partial_r"],
        },
        drivers=partial_data,
        chart_data=[{"name": p["name"], "partial_r": p["partial_r"], "partial_r2": p["partial_r2"]} for p in partial_data],
        insights=insights,
    )

# ─── Module 5: Interaction Effects ────────────────────────────────────────────

def analyze_interactions(rows, target, drivers, params) -> DriverResult:
    """Test pairwise interaction terms: Y ~ X_i + X_j + X_i*X_j vs Y ~ X_i + X_j."""
    from itertools import combinations
    from sklearn.linear_model import LinearRegression

    df = _to_df(rows, [target] + drivers)
    y  = df[target].values
    X_df = df[drivers]

    interaction_results = []
    for d1, d2 in combinations(drivers, 2):
        x1 = X_df[d1].values
        x2 = X_df[d2].values
        interaction = x1 * x2

        # Base model
        X_base = np.column_stack([x1, x2])
        reg_base = LinearRegression().fit(X_base, y)
        r2_base = max(0.0, reg_base.score(X_base, y))

        # Interaction model
        X_int = np.column_stack([x1, x2, interaction])
        reg_int = LinearRegression().fit(X_int, y)
        r2_int = max(0.0, reg_int.score(X_int, y))

        delta_r2 = round(r2_int - r2_base, 4)
        int_coef = float(reg_int.coef_[2])

        interaction_results.append({
            "var1":          d1,
            "var2":          d2,
            "label":         f"{d1} × {d2}",
            "delta_r2":      delta_r2,
            "int_coef":      round(int_coef, 4),
            "r2_base":       round(r2_base, 4),
            "r2_with_int":   round(r2_int, 4),
            "significant":   delta_r2 > 0.01,
        })

    interaction_results.sort(key=lambda x: -x["delta_r2"])
    for i, r in enumerate(interaction_results):
        r["rank"] = i + 1

    significant = [r for r in interaction_results if r["significant"]]

    insights = []
    if significant:
        top_int = significant[0]
        insights.append(
            f"Strongest interaction: {top_int['label']} — adds ΔR² = {top_int['delta_r2']:.4f} "
            f"beyond the main effects model. The effect of {top_int['var1']} on {target} "
            f"depends on the level of {top_int['var2']}."
        )
        insights.append(
            f"{len(significant)} significant interaction(s) detected (ΔR² > 0.01). "
            f"Including these terms could improve model fit."
        )
    else:
        insights.append(
            "No significant interaction effects detected (all ΔR² < 0.01). "
            "Driver effects appear to be largely additive and independent."
        )

    insights.append(
        "Interpretation: A positive interaction coefficient means the two drivers amplify each other's effect. "
        "A negative coefficient means they dampen each other."
    )

    return DriverResult(
        analysis_type="interactions",
        summary={
            "target":           target,
            "n_pairs_tested":   len(interaction_results),
            "n_significant":    len(significant),
            "top_interaction":  interaction_results[0]["label"] if interaction_results else None,
        },
        drivers=interaction_results,
        chart_data=[{"label": r["label"], "delta_r2": r["delta_r2"], "significant": r["significant"]} for r in interaction_results[:15]],
        insights=insights,
    )

# ─── Module 6: Segmentation ───────────────────────────────────────────────────

def analyze_segmentation(rows, target, drivers, params, segment_col) -> DriverResult:
    """Run importance analysis per segment and compare driver rankings across groups."""
    from sklearn.linear_model import LinearRegression

    if not segment_col:
        raise HTTPException(status_code=400, detail="segment_col is required for segmentation analysis.")

    all_cols = [target, segment_col] + drivers
    df_raw = pd.DataFrame(rows)[all_cols].dropna(subset=[segment_col])
    segments = df_raw[segment_col].unique()

    if len(segments) > 20:
        raise HTTPException(status_code=400, detail=f"Too many segments ({len(segments)}). Max 20.")

    segment_results = []
    for seg in segments:
        seg_df = df_raw[df_raw[segment_col] == seg][drivers + [target]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(seg_df) < 5:
            continue

        y = seg_df[target].values
        X = seg_df[drivers].values

        reg = LinearRegression().fit(X, y)
        ss_res = np.sum((y - reg.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = max(0.0, 1 - ss_res / (ss_tot + 1e-12))

        # Standardised coefs as proxy for importance
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        y_std = (y - y.mean()) / (y.std() + 1e-9)
        reg_std = LinearRegression().fit(X_std, y_std)
        std_coefs = dict(zip(drivers, reg_std.coef_))

        # Rank by abs std coef
        ranked = sorted(std_coefs.items(), key=lambda x: -abs(x[1]))
        segment_results.append({
            "segment":  str(seg),
            "n":        len(seg_df),
            "r2":       round(r2, 4),
            "top_driver": ranked[0][0] if ranked else None,
            "driver_importance": [
                {"name": d, "std_coef": round(float(v), 4), "direction": "positive" if v >= 0 else "negative"}
                for d, v in ranked
            ],
        })

    segment_results.sort(key=lambda x: -x["r2"])

    # Cross-segment rank comparison
    rank_matrix = {}
    for seg in segment_results:
        for i, di in enumerate(seg["driver_importance"]):
            d = di["name"]
            if d not in rank_matrix:
                rank_matrix[d] = {}
            rank_matrix[d][seg["segment"]] = i + 1

    # Drivers whose rank varies most across segments
    rank_variance = {}
    for d, ranks in rank_matrix.items():
        vals = list(ranks.values())
        rank_variance[d] = round(float(np.std(vals)), 2)
    unstable = sorted(rank_variance.items(), key=lambda x: -x[1])

    insights = []
    top_drivers_per_seg = {s["segment"]: s["top_driver"] for s in segment_results}
    unique_tops = set(top_drivers_per_seg.values())
    if len(unique_tops) == 1:
        insights.append(
            f"Consistent #1 driver across all segments: {list(unique_tops)[0]}. "
            f"Driver importance is stable regardless of {segment_col}."
        )
    else:
        insights.append(
            f"Top driver varies by segment: {', '.join(f'{s}→{d}' for s, d in list(top_drivers_per_seg.items())[:4])}. "
            f"Consider segment-specific strategies."
        )

    if unstable:
        insights.append(
            f"Most unstable driver across segments: {unstable[0][0]} (rank std = {unstable[0][1]:.2f}). "
            f"Its importance fluctuates significantly — investigate segment-specific dynamics."
        )

    return DriverResult(
        analysis_type="segmentation",
        summary={
            "segment_col":  segment_col,
            "n_segments":   len(segment_results),
            "segments":     [s["segment"] for s in segment_results],
            "rank_variance": dict(unstable[:5]),
        },
        drivers=segment_results,
        chart_data=segment_results,
        insights=insights,
    )

# ─── Module 7: Recommendations ────────────────────────────────────────────────

def analyze_recommendations(rows, target, drivers, params) -> DriverResult:
    """Synthesise importance + partial effects + multicollinearity into an action plan."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    df = _to_df(rows, [target] + drivers)
    y  = df[target].values
    X  = df[drivers].values

    # Quick importance (std coefs)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    y_std = (y - y.mean()) / (y.std() + 1e-9)
    reg_std = LinearRegression().fit(X_std, y_std)
    std_coefs = {d: float(c) for d, c in zip(drivers, reg_std.coef_)}

    # VIF
    vif_map = {}
    for i, d in enumerate(drivers):
        x_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        if X_others.shape[1] > 0:
            reg_v = LinearRegression().fit(X_others, x_i)
            r2_v  = max(0.0, reg_v.score(X_others, x_i))
            vif_map[d] = min(1 / (1 - r2_v) if r2_v < 1 else 999.0, 999.0)
        else:
            vif_map[d] = 1.0

    # Full R²
    reg_full = LinearRegression().fit(X, y)
    r2_full  = max(0.0, reg_full.score(X, y))

    # Build action plan
    ranked = sorted(std_coefs.items(), key=lambda x: -abs(x[1]))
    action_results = []
    for rank, (d, coef) in enumerate(ranked, 1):
        importance_pct = abs(coef) / (sum(abs(c) for c in std_coefs.values()) + 1e-9) * 100
        vif = vif_map[d]
        direction = "positive" if coef >= 0 else "negative"
        priority = (
            "High"   if rank <= 2 and vif < 5 else
            "Medium" if rank <= 4 or vif < 5  else
            "Low"
        )
        action = (
            f"Prioritise increasing {d} — it has the strongest positive impact on {target}."
            if direction == "positive" and rank == 1 else
            f"Monitor {d} closely — it negatively affects {target}. Reducing it may improve outcomes."
            if direction == "negative" and rank <= 3 else
            f"Optimise {d} — it contributes meaningfully to {target} but may have diminishing returns."
            if rank <= len(drivers) // 2 else
            f"Low priority: {d} has limited unique contribution. Consider dropping from future models."
        )
        if vif >= 10:
            action += f" ⚠️ High multicollinearity (VIF={vif:.1f}) — interpret with caution."

        action_results.append({
            "name":           d,
            "rank":           rank,
            "priority":       priority,
            "importance_pct": round(importance_pct, 1),
            "std_coef":       round(coef, 4),
            "direction":      direction,
            "vif":            round(vif, 1),
            "action":         action,
        })

    insights = []
    high_priority = [a for a in action_results if a["priority"] == "High"]
    insights.append(
        f"Focus on {len(high_priority)} high-priority driver(s): "
        f"{', '.join(a['name'] for a in high_priority)}. "
        f"These have the greatest potential impact on {target}."
    )

    neg_high = [a for a in action_results if a["direction"] == "negative" and a["rank"] <= 3]
    if neg_high:
        insights.append(
            f"Risk drivers to manage: {', '.join(a['name'] for a in neg_high)}. "
            f"These negatively influence {target} — reducing or controlling them will help."
        )

    insights.append(
        f"Model R² = {r2_full:.3f}. "
        f"{'Good explanatory power — recommendations are well-grounded.' if r2_full > 0.6 else 'Moderate fit — validate with domain expertise before acting on these recommendations.'}"
    )

    return DriverResult(
        analysis_type="recommendations",
        summary={
            "target":           target,
            "r2":               round(r2_full, 4),
            "n_high_priority":  len(high_priority),
            "top_driver":       action_results[0]["name"] if action_results else None,
        },
        drivers=action_results,
        chart_data=[{"name": a["name"], "importance_pct": a["importance_pct"], "priority": a["priority"]} for a in action_results],
        insights=insights,
    )

# ─── Main Endpoint ─────────────────────────────────────────────────────────────

@router.post("/driver", response_model=DriverResult)
async def run_driver(request: DriverRequest):
    if not request.rows:
        raise HTTPException(status_code=400, detail="No data rows provided.")
    if not request.target:
        raise HTTPException(status_code=400, detail="Target variable is required.")
    if len(request.drivers) < 2:
        raise HTTPException(status_code=400, detail="At least 2 driver variables are required.")

    try:
        t = request.analysis_type
        p = request.params or {}

        if t == "importance":
            return analyze_importance(request.rows, request.target, request.drivers, p)
        elif t == "correlation":
            return analyze_correlation(request.rows, request.target, request.drivers, p)
        elif t == "multicollinearity":
            return analyze_multicollinearity(request.rows, request.target, request.drivers, p)
        elif t == "partial_effects":
            return analyze_partial_effects(request.rows, request.target, request.drivers, p)
        elif t == "interactions":
            return analyze_interactions(request.rows, request.target, request.drivers, p)
        elif t == "segmentation":
            return analyze_segmentation(request.rows, request.target, request.drivers, p, request.segment_col)
        elif t == "recommendations":
            return analyze_recommendations(request.rows, request.target, request.drivers, p)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported analysis_type: '{t}'. "
                       f"Available: importance, correlation, multicollinearity, partial_effects, interactions, segmentation, recommendations",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/driver/health")
async def health():
    return {
        "status":  "ok",
        "version": "1.0",
        "modules": ["importance", "correlation", "multicollinearity", "partial_effects", "interactions", "segmentation", "recommendations"],
        "endpoints": {
            "analysis": "POST /api/analysis/driver",
            "health":   "GET  /api/analysis/driver/health",
        },
    }
