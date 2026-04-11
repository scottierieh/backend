"""
Synthetic Control Method (SCM) Router — FastAPI
5 modules: data_validate | optimize | results | placebo | leave_one_out
Endpoint: POST /api/analysis/scm
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from scipy.optimize import minimize
except ImportError:
    raise ImportError("pip install scipy")

router = APIRouter()

# ─── Input / Output Models ─────────────────────────────────────────────────────

class DataRow(BaseModel):
    id:        str            # unit identifier (region, company, country…)
    time:      str            # time period (sortable string or number as string)
    outcome:   float          # Y value
    treatment: Optional[int] = None   # 0 / 1 — optional (Level 1 only)

class ScmRequest(BaseModel):
    rows:           List[DataRow]
    analysis_type:  str            # validate | optimize | results | placebo | leave_one_out
    treated_unit:   Optional[str]  = None
    treatment_date: Optional[str]  = None   # first period where treatment == 1
    covariates:     Optional[List[str]] = []
    donor_pool:     Optional[List[str]] = None   # None = use all non-treated units
    params:         Optional[Dict[str, Any]] = {}

class ScmResult(BaseModel):
    analysis_type: str
    summary:       Dict[str, Any]
    series:        Optional[List[Dict[str, Any]]] = []   # time-series output
    units:         Optional[List[Dict[str, Any]]] = []   # per-unit output
    insights:      List[str]

# ─── Data Builder ──────────────────────────────────────────────────────────────

def build_panel(rows: List[DataRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.dict() for r in rows])
    df["time_sort"] = df["time"]   # keep original for display
    try:
        df["time_sort"] = pd.to_numeric(df["time"])
    except Exception:
        pass
    df = df.sort_values(["id", "time_sort"]).reset_index(drop=True)
    return df


def identify_treated(df: pd.DataFrame) -> tuple[str, str]:
    """Return (treated_unit, treatment_date) from binary treatment column."""
    treated_rows = df[df["treatment"] == 1]
    if treated_rows.empty:
        raise ValueError("No rows with treatment == 1 found.")
    treated_unit  = treated_rows["id"].value_counts().idxmax()
    treatment_date = treated_rows[treated_rows["id"] == treated_unit]["time_sort"].min()
    return str(treated_unit), str(treatment_date)


def split_pre_post(df: pd.DataFrame, treated_unit: str, treatment_date: str):
    treated_df = df[df["id"] == treated_unit].copy()
    donor_df   = df[df["id"] != treated_unit].copy()

    try:
        td = float(treatment_date)
        pre_treated  = treated_df[treated_df["time_sort"] <  td]
        post_treated = treated_df[treated_df["time_sort"] >= td]
        pre_donor    = donor_df[donor_df["time_sort"] < td]
    except (ValueError, TypeError):
        pre_treated  = treated_df[treated_df["time_sort"] <  treatment_date]
        post_treated = treated_df[treated_df["time_sort"] >= treatment_date]
        pre_donor    = donor_df[donor_df["time_sort"] < treatment_date]

    return pre_treated, post_treated, pre_donor, donor_df


# ─── Module 1: Data Validation ────────────────────────────────────────────────

def analyze_validate(df: pd.DataFrame, treated_unit: str, treatment_date: str, donor_pool: List[str]) -> ScmResult:
    all_units    = df["id"].unique().tolist()
    all_times    = sorted(df["time_sort"].unique().tolist())
    n_units      = len(all_units)
    n_times      = len(all_times)
    n_donors     = len(donor_pool)

    # Pre/post split
    pre_treated, post_treated, pre_donor, donor_df = split_pre_post(df, treated_unit, treatment_date)
    n_pre  = len(pre_treated)
    n_post = len(post_treated)

    # Missing value check
    missing = df["outcome"].isna().sum()

    # Balance check: does every unit have every time period?
    expected_cells = n_units * n_times
    actual_cells   = len(df)
    is_balanced    = (actual_cells == expected_cells)

    # Pre-treatment mean of treated
    treated_pre_mean   = pre_treated["outcome"].mean()
    treated_pre_std    = pre_treated["outcome"].std()

    # Donor coverage: how many donors are within 2x range of treated unit?
    donor_pre_means = donor_df[donor_df["time_sort"] < treatment_date].groupby("id")["outcome"].mean()
    similar_donors  = donor_pre_means[
        (donor_pre_means >= treated_pre_mean * 0.3) &
        (donor_pre_means <= treated_pre_mean * 3.0)
    ].index.tolist()

    quality_score = min(100, int(
        (min(n_pre, 20) / 20) * 40 +       # pre-period length (max 20 = 40pts)
        (min(n_donors, 10) / 10) * 30 +     # donor count (max 10 = 30pts)
        (len(similar_donors) / max(n_donors, 1)) * 30  # donor similarity (30pts)
    ))

    issues: List[str] = []
    if n_pre < 10:
        issues.append(f"Short pre-treatment period ({n_pre} periods). Recommend ≥10 for reliable weights.")
    if n_donors < 3:
        issues.append(f"Only {n_donors} donor units. Recommend ≥3 for a robust synthetic control.")
    if not similar_donors:
        issues.append("No donors have similar outcome levels to the treated unit. Results may be unreliable.")
    if missing > 0:
        issues.append(f"{missing} missing outcome values detected. Fill or drop before analysis.")
    if not is_balanced:
        issues.append(f"Unbalanced panel: {actual_cells} rows found, expected {expected_cells}. Some unit-time cells are missing.")

    insights: List[str] = []
    insights.append(
        f"Panel: {n_units} units × {n_times} time periods. "
        f"Treated unit: '{treated_unit}', treatment from '{treatment_date}'."
    )
    insights.append(
        f"Pre-treatment periods: {n_pre}, post-treatment periods: {n_post}. "
        f"{'✓ Sufficient pre-period.' if n_pre >= 10 else '⚠ Consider collecting more pre-treatment data.'}"
    )
    insights.append(
        f"Donor pool: {n_donors} units. {len(similar_donors)} are within a comparable outcome range."
    )
    if quality_score >= 75:
        insights.append(f"Data quality score: {quality_score}/100 — Good. Proceed to Donor Pool setup.")
    elif quality_score >= 50:
        insights.append(f"Data quality score: {quality_score}/100 — Fair. Review issues before proceeding.")
    else:
        insights.append(f"Data quality score: {quality_score}/100 — Poor. Address issues before running SCM.")

    return ScmResult(
        analysis_type="validate",
        summary={
            "n_units":         n_units,
            "n_times":         n_times,
            "n_pre":           n_pre,
            "n_post":          n_post,
            "n_donors":        n_donors,
            "treated_unit":    treated_unit,
            "treatment_date":  treatment_date,
            "is_balanced":     is_balanced,
            "missing_values":  int(missing),
            "quality_score":   quality_score,
            "similar_donors":  similar_donors[:10],
            "issues":          issues,
            "all_units":       all_units,
            "all_times":       [str(t) for t in all_times],
        },
        insights=insights,
    )


# ─── Module 2: Optimization ───────────────────────────────────────────────────

def _build_matrices(df: pd.DataFrame, treated_unit: str, treatment_date: str, donor_pool: List[str]):
    """Build Y0 (donor matrix) and y1 (treated vector) for pre-treatment period."""
    try:
        td = float(treatment_date)
        pre = df[df["time_sort"] < td]
    except (ValueError, TypeError):
        pre = df[df["time_sort"] < treatment_date]

    pre_times = sorted(pre["time_sort"].unique())

    # Treated vector
    treated_pre = pre[pre["id"] == treated_unit].set_index("time_sort")["outcome"]
    y1 = np.array([treated_pre.get(t, np.nan) for t in pre_times])

    # Donor matrix (T_pre × n_donors)
    Y0_cols = []
    valid_donors = []
    for d in donor_pool:
        donor_pre = pre[pre["id"] == d].set_index("time_sort")["outcome"]
        col = np.array([donor_pre.get(t, np.nan) for t in pre_times])
        if not np.any(np.isnan(col)):
            Y0_cols.append(col)
            valid_donors.append(d)

    if not Y0_cols:
        raise ValueError("No donors have complete pre-treatment data.")

    Y0 = np.column_stack(Y0_cols)   # shape: (T_pre, n_donors)
    return Y0, y1, valid_donors, pre_times


def optimize_weights(Y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """
    Minimize ||y1 - Y0 @ w||^2  subject to  w >= 0, sum(w) == 1.
    Uses scipy SLSQP — no cvxpy dependency needed.
    """
    n_donors = Y0.shape[1]
    w0 = np.ones(n_donors) / n_donors

    def loss(w):
        return np.sum((y1 - Y0 @ w) ** 2)

    def grad(w):
        return -2 * Y0.T @ (y1 - Y0 @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n_donors

    res = minimize(loss, w0, jac=grad, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-10, "maxiter": 1000})

    weights = np.clip(res.x, 0, 1)
    weights /= weights.sum()   # renormalize
    return weights


def analyze_optimize(df: pd.DataFrame, treated_unit: str, treatment_date: str, donor_pool: List[str]) -> ScmResult:
    Y0, y1, valid_donors, pre_times = _build_matrices(df, treated_unit, treatment_date, donor_pool)

    weights = optimize_weights(Y0, y1)

    # Pre-fit quality
    synthetic_pre = Y0 @ weights
    rmse = float(np.sqrt(np.mean((y1 - synthetic_pre) ** 2)))
    rmspe = float(rmse / (np.mean(np.abs(y1)) + 1e-9))

    # Build full synthetic series (pre + post)
    all_times = sorted(df["time_sort"].unique())
    weight_map = dict(zip(valid_donors, weights.tolist()))

    synthetic_series = []
    actual_series    = []

    try:
        td = float(treatment_date)
    except (ValueError, TypeError):
        td = treatment_date

    for t in all_times:
        # Synthetic: weighted sum of donor outcomes at time t
        synth_val = 0.0
        for d, w in weight_map.items():
            row = df[(df["id"] == d) & (df["time_sort"] == t)]
            if not row.empty:
                synth_val += w * float(row["outcome"].iloc[0])

        # Actual treated
        actual_row = df[(df["id"] == treated_unit) & (df["time_sort"] == t)]
        actual_val = float(actual_row["outcome"].iloc[0]) if not actual_row.empty else None

        synthetic_series.append({"time": str(t), "synthetic": round(synth_val, 4)})
        if actual_val is not None:
            actual_series.append({"time": str(t), "actual": round(actual_val, 4)})

    # Active donors (weight > 1%)
    active_donors = {d: round(float(w), 4) for d, w in weight_map.items() if w > 0.01}
    active_donors = dict(sorted(active_donors.items(), key=lambda x: -x[1]))

    insights: List[str] = []
    insights.append(
        f"Optimal weights computed across {len(valid_donors)} donor units using SLSQP optimization."
    )
    insights.append(
        f"Pre-treatment RMSPE: {rmspe:.4f}. "
        f"{'Excellent fit (< 0.05).' if rmspe < 0.05 else 'Good fit (< 0.15).' if rmspe < 0.15 else 'Poor fit — synthetic control may not match treated unit well.'}"
    )
    if active_donors:
        top = list(active_donors.items())[:3]
        insights.append(
            f"Top donors by weight: {', '.join(f'{d} ({w:.1%})' for d, w in top)}. "
            f"Synthetic control is constructed from {len(active_donors)} active donors."
        )
    zero_donors = [d for d, w in weight_map.items() if w <= 0.01]
    if zero_donors:
        insights.append(
            f"{len(zero_donors)} donor(s) received near-zero weight and do not contribute to the synthetic control."
        )

    return ScmResult(
        analysis_type="optimize",
        summary={
            "weights":        weight_map,
            "active_donors":  active_donors,
            "rmse":           round(rmse, 4),
            "rmspe":          round(rmspe, 4),
            "n_donors_used":  len(valid_donors),
            "treated_unit":   treated_unit,
            "treatment_date": treatment_date,
        },
        series=synthetic_series,
        units=[{"id": d, "weight": round(float(w), 4)} for d, w in weight_map.items()],
        insights=insights,
    )


# ─── Module 3: Results ────────────────────────────────────────────────────────

def analyze_results(df: pd.DataFrame, treated_unit: str, treatment_date: str,
                    donor_pool: List[str], params: dict) -> ScmResult:
    # Re-run optimization to get weights
    Y0, y1, valid_donors, _ = _build_matrices(df, treated_unit, treatment_date, donor_pool)
    weights    = optimize_weights(Y0, y1)
    weight_map = dict(zip(valid_donors, weights.tolist()))

    all_times = sorted(df["time_sort"].unique())

    try:
        td = float(treatment_date)
        is_post = lambda t: float(t) >= td
    except (ValueError, TypeError):
        is_post = lambda t: str(t) >= str(treatment_date)

    series_out = []
    att_vals   = []

    for t in all_times:
        synth_val = sum(
            weight_map.get(d, 0) * float(df[(df["id"] == d) & (df["time_sort"] == t)]["outcome"].iloc[0])
            for d in valid_donors
            if not df[(df["id"] == d) & (df["time_sort"] == t)].empty
        )
        actual_row = df[(df["id"] == treated_unit) & (df["time_sort"] == t)]
        actual_val = float(actual_row["outcome"].iloc[0]) if not actual_row.empty else None

        gap = round(actual_val - synth_val, 4) if actual_val is not None else None
        if is_post(t) and gap is not None:
            att_vals.append(gap)

        series_out.append({
            "time":      str(t),
            "actual":    round(actual_val, 4) if actual_val is not None else None,
            "synthetic": round(synth_val, 4),
            "gap":       gap,
            "is_post":   is_post(t),
        })

    att       = float(np.mean(att_vals)) if att_vals else 0.0
    pre_mean  = float(df[(df["id"] == treated_unit)]["outcome"].mean())
    att_pct   = att / (pre_mean + 1e-9) * 100

    insights: List[str] = []
    direction = "increase" if att > 0 else "decrease"
    insights.append(
        f"Average Treatment Effect on the Treated (ATT): {att:+.3f} "
        f"({att_pct:+.1f}% vs pre-treatment mean)."
    )
    if abs(att_pct) < 3:
        insights.append("The effect size is small (< 3%). The intervention may not have had a meaningful impact.")
    elif abs(att_pct) < 10:
        insights.append(f"Moderate {direction} of ~{abs(att_pct):.1f}% detected post-intervention.")
    else:
        insights.append(f"Large {direction} of ~{abs(att_pct):.1f}% detected. The intervention had a substantial causal effect.")

    max_gap = max(att_vals, key=abs) if att_vals else 0
    insights.append(
        f"Largest single-period gap: {max_gap:+.3f}. "
        f"Donor composition: {', '.join(f'{d} ({w:.1%})' for d, w in sorted(weight_map.items(), key=lambda x: -x[1])[:3])}."
    )

    return ScmResult(
        analysis_type="results",
        summary={
            "att":            round(att, 4),
            "att_pct":        round(att_pct, 2),
            "treated_unit":   treated_unit,
            "treatment_date": treatment_date,
            "n_post_periods": len(att_vals),
            "weights":        {d: round(float(w), 4) for d, w in weight_map.items()},
        },
        series=series_out,
        insights=insights,
    )


# ─── Module 4: Placebo Test ───────────────────────────────────────────────────

def analyze_placebo(df: pd.DataFrame, treated_unit: str, treatment_date: str,
                    donor_pool: List[str]) -> ScmResult:
    """
    Apply SCM to each donor as if it were the treated unit.
    Compare effect magnitudes to assess significance.
    """
    placebo_effects: List[Dict[str, Any]] = []

    try:
        td = float(treatment_date)
        is_post = lambda t: float(t) >= td
    except (ValueError, TypeError):
        is_post = lambda t: str(t) >= str(treatment_date)

    # Real treated effect
    real_donors = [d for d in donor_pool if d != treated_unit]
    try:
        Y0r, y1r, vd_r, _ = _build_matrices(df, treated_unit, treatment_date, real_donors)
        wr = optimize_weights(Y0r, y1r)
        wmap_r = dict(zip(vd_r, wr.tolist()))
        all_times = sorted(df["time_sort"].unique())

        real_gaps = []
        for t in all_times:
            if not is_post(t): continue
            synth = sum(wmap_r.get(d, 0) * float(df[(df["id"] == d) & (df["time_sort"] == t)]["outcome"].iloc[0])
                        for d in vd_r if not df[(df["id"] == d) & (df["time_sort"] == t)].empty)
            actual_row = df[(df["id"] == treated_unit) & (df["time_sort"] == t)]
            if not actual_row.empty:
                real_gaps.append(float(actual_row["outcome"].iloc[0]) - synth)
        real_att = float(np.mean(real_gaps)) if real_gaps else 0.0
    except Exception:
        real_att = 0.0

    placebo_effects.append({"unit": treated_unit, "att": round(real_att, 4), "is_treated": True})

    # Placebo for each donor
    for placebo_unit in donor_pool[:15]:   # cap at 15 for speed
        try:
            placebo_donors = [d for d in donor_pool if d != placebo_unit and d != treated_unit]
            if len(placebo_donors) < 2:
                continue
            Y0p, y1p, vdp, _ = _build_matrices(df, placebo_unit, treatment_date, placebo_donors)
            wp   = optimize_weights(Y0p, y1p)
            wmap = dict(zip(vdp, wp.tolist()))

            gaps = []
            for t in all_times:
                if not is_post(t): continue
                synth = sum(wmap.get(d, 0) * float(df[(df["id"] == d) & (df["time_sort"] == t)]["outcome"].iloc[0])
                            for d in vdp if not df[(df["id"] == d) & (df["time_sort"] == t)].empty)
                actual_row = df[(df["id"] == placebo_unit) & (df["time_sort"] == t)]
                if not actual_row.empty:
                    gaps.append(float(actual_row["outcome"].iloc[0]) - synth)

            placebo_att = float(np.mean(gaps)) if gaps else 0.0
            placebo_effects.append({"unit": placebo_unit, "att": round(placebo_att, 4), "is_treated": False})
        except Exception:
            continue

    # p-value: fraction of placebos with |ATT| >= |real ATT|
    placebo_atts = [p["att"] for p in placebo_effects if not p["is_treated"]]
    p_value = sum(abs(a) >= abs(real_att) for a in placebo_atts) / max(len(placebo_atts), 1)

    insights: List[str] = []
    insights.append(
        f"Placebo test: applied SCM to {len(placebo_atts)} donor units as pseudo-treated."
    )
    insights.append(
        f"Real ATT: {real_att:+.4f}. "
        f"Fraction of placebos with |ATT| ≥ |real|: {p_value:.3f} "
        f"({'statistically significant (p < 0.10)' if p_value < 0.10 else 'not statistically significant'})."
    )
    if p_value < 0.05:
        insights.append("Strong evidence the observed effect is not due to chance (p < 0.05).")
    elif p_value < 0.10:
        insights.append("Moderate evidence of a real causal effect (p < 0.10).")
    else:
        insights.append("Weak evidence — the effect may not be distinguishable from noise.")

    return ScmResult(
        analysis_type="placebo",
        summary={
            "real_att":        round(real_att, 4),
            "p_value":         round(p_value, 4),
            "n_placebos":      len(placebo_atts),
            "treated_unit":    treated_unit,
            "treatment_date":  treatment_date,
        },
        units=placebo_effects,
        insights=insights,
    )


# ─── Module 5: Leave-one-out ──────────────────────────────────────────────────

def analyze_leave_one_out(df: pd.DataFrame, treated_unit: str, treatment_date: str,
                          donor_pool: List[str]) -> ScmResult:
    """
    Re-run SCM leaving out one donor at a time.
    Shows how sensitive results are to each donor's inclusion.
    """
    try:
        td = float(treatment_date)
        is_post = lambda t: float(t) >= td
    except (ValueError, TypeError):
        is_post = lambda t: str(t) >= str(treatment_date)

    all_times = sorted(df["time_sort"].unique())

    def compute_att(donors: List[str]) -> float:
        Y0, y1, vd, _ = _build_matrices(df, treated_unit, treatment_date, donors)
        w    = optimize_weights(Y0, y1)
        wmap = dict(zip(vd, w.tolist()))
        gaps = []
        for t in all_times:
            if not is_post(t): continue
            synth = sum(wmap.get(d, 0) * float(df[(df["id"] == d) & (df["time_sort"] == t)]["outcome"].iloc[0])
                        for d in vd if not df[(df["id"] == d) & (df["time_sort"] == t)].empty)
            actual_row = df[(df["id"] == treated_unit) & (df["time_sort"] == t)]
            if not actual_row.empty:
                gaps.append(float(actual_row["outcome"].iloc[0]) - synth)
        return float(np.mean(gaps)) if gaps else 0.0

    # Baseline
    baseline_att = compute_att(donor_pool)

    loo_results: List[Dict[str, Any]] = []
    for donor in donor_pool[:12]:  # cap for speed
        try:
            reduced = [d for d in donor_pool if d != donor]
            if len(reduced) < 2:
                continue
            att = compute_att(reduced)
            loo_results.append({
                "removed":         donor,
                "att":             round(att, 4),
                "att_change":      round(att - baseline_att, 4),
                "pct_change":      round((att - baseline_att) / (abs(baseline_att) + 1e-9) * 100, 1),
            })
        except Exception:
            continue

    # Sort by sensitivity (largest ATT change)
    loo_results.sort(key=lambda x: abs(x["att_change"]), reverse=True)

    # Sensitivity range
    atts = [r["att"] for r in loo_results] + [baseline_att]
    att_range = max(atts) - min(atts)
    is_robust = att_range / (abs(baseline_att) + 1e-9) < 0.20

    insights: List[str] = []
    insights.append(
        f"Leave-one-out analysis across {len(loo_results)} donors. Baseline ATT: {baseline_att:+.4f}."
    )
    insights.append(
        f"ATT range when removing single donors: [{min(atts):+.4f}, {max(atts):+.4f}]. "
        f"Spread: {att_range:.4f} ({att_range / (abs(baseline_att) + 1e-9):.1%} of baseline)."
    )
    if is_robust:
        insights.append("Results are robust — removing any single donor changes the ATT by < 20%.")
    else:
        most_influential = loo_results[0]["removed"] if loo_results else "N/A"
        insights.append(
            f"Results are sensitive to the inclusion of '{most_influential}' "
            f"(ATT change: {loo_results[0]['att_change']:+.4f}). "
            f"Verify this donor is an appropriate control unit."
        )

    return ScmResult(
        analysis_type="leave_one_out",
        summary={
            "baseline_att":    round(baseline_att, 4),
            "att_range":       round(att_range, 4),
            "is_robust":       is_robust,
            "treated_unit":    treated_unit,
            "treatment_date":  treatment_date,
            "n_donors":        len(donor_pool),
        },
        units=loo_results,
        insights=insights,
    )


# ─── Main Endpoint ─────────────────────────────────────────────────────────────

@router.post("/scm", response_model=ScmResult)
async def run_scm(request: ScmRequest):
    try:
        if not request.rows:
            raise HTTPException(status_code=400, detail="No data rows provided.")

        df = build_panel(request.rows)

        if df.empty or df["outcome"].isna().all():
            raise HTTPException(status_code=400, detail="Panel is empty or all outcome values are missing.")

        # Resolve treated unit and treatment date
        treated_unit   = request.treated_unit
        treatment_date = request.treatment_date

        if not treated_unit or not treatment_date:
            # Auto-identify from treatment column
            if "treatment" not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail="'treated_unit' and 'treatment_date' are required, or provide a 'treatment' column for auto-detection."
                )
            treated_unit, treatment_date = identify_treated(df)

        # Build donor pool
        all_units  = df["id"].unique().tolist()
        donor_pool = request.donor_pool or [u for u in all_units if u != treated_unit]

        if len(donor_pool) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 donor units to construct a synthetic control.")

        t = request.analysis_type
        p = request.params or {}

        if t == "validate":
            return analyze_validate(df, treated_unit, treatment_date, donor_pool)
        elif t == "optimize":
            return analyze_optimize(df, treated_unit, treatment_date, donor_pool)
        elif t == "results":
            return analyze_results(df, treated_unit, treatment_date, donor_pool, p)
        elif t == "placebo":
            return analyze_placebo(df, treated_unit, treatment_date, donor_pool)
        elif t == "leave_one_out":
            return analyze_leave_one_out(df, treated_unit, treatment_date, donor_pool)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown analysis_type: '{t}'. Available: validate | optimize | results | placebo | leave_one_out"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SCM error: {str(e)}")


# ─── Health Check ──────────────────────────────────────────────────────────────

@router.get("/scm/health")
async def health():
    return {
        "status":   "ok",
        "version":  "1.0",
        "modules":  ["validate", "optimize", "results", "placebo", "leave_one_out"],
        "endpoint": "POST /api/analysis/scm",
        "solver":   "scipy SLSQP (no cvxpy required)",
    }
