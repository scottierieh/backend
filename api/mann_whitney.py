from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
import io
import base64
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class MannWhitneyRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    group_col: str = Field(..., description="Column name for grouping variable")
    value_col: str = Field(..., description="Column name for numeric value")
    groups: Optional[List[str]] = Field(None, description="Optional: specific groups to compare")
    alternative: str = Field(default='two-sided', description="Alternative hypothesis")
    alpha: float = Field(default=0.05, description="Significance level")


def _to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def get_effect_size_r_interpretation(r: float) -> dict:
    """Interpret effect size r — returns {text, magnitude}"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return {"magnitude": "Negligible", "text": "Negligible effect (r < 0.1)"}
    elif abs_r < 0.3:
        return {"magnitude": "Small", "text": "Small effect (0.1 ≤ r < 0.3)"}
    elif abs_r < 0.5:
        return {"magnitude": "Medium", "text": "Medium effect (0.3 ≤ r < 0.5)"}
    else:
        return {"magnitude": "Large", "text": "Large effect (r ≥ 0.5)"}


def get_cliffs_delta_interpretation(delta: float) -> dict:
    """Interpret Cliff's delta — returns {text, magnitude}"""
    abs_d = abs(delta)
    if abs_d < 0.147:
        return {"magnitude": "Negligible", "text": "Negligible dominance (|δ| < 0.147)"}
    elif abs_d < 0.33:
        return {"magnitude": "Small", "text": "Small dominance (0.147 ≤ |δ| < 0.33)"}
    elif abs_d < 0.474:
        return {"magnitude": "Medium", "text": "Medium dominance (0.33 ≤ |δ| < 0.474)"}
    else:
        return {"magnitude": "Large", "text": "Large dominance (|δ| ≥ 0.474)"}


def compute_cliffs_delta(g1: np.ndarray, g2: np.ndarray) -> float:
    """Compute Cliff's delta (non-parametric effect size)."""
    n1, n2 = len(g1), len(g2)
    dominance = sum(1 if x > y else (-1 if x < y else 0) for x in g1 for y in g2)
    return dominance / (n1 * n2)


def detect_ties(combined: np.ndarray) -> bool:
    return len(combined) != len(set(combined))


def same_shape_test(g1: np.ndarray, g2: np.ndarray) -> bool:
    """Levene test for equal variance as a shape proxy."""
    if len(g1) < 2 or len(g2) < 2:
        return True
    _, p = stats.levene(g1, g2)
    return p > 0.05


def generate_interpretations(
    group1_name: str, group2_name: str,
    g1: np.ndarray, g2: np.ndarray,
    statistic: float, p_value: float,
    effect_size_r: float, cliffs_delta: float,
    is_significant: bool, alpha: float,
    z_score: float,
) -> dict:
    """Generate frontend interpretations dict with overall_analysis, statistical_insights, recommendations."""
    abs_r = abs(effect_size_r)
    abs_d = abs(cliffs_delta)
    r_interp = get_effect_size_r_interpretation(effect_size_r)["magnitude"].lower()
    d_interp = get_cliffs_delta_interpretation(cliffs_delta)["magnitude"].lower()

    pf = "p < .001" if p_value < 0.001 else f"p = {p_value:.4f}"
    med1, med2 = float(np.median(g1)), float(np.median(g2))
    higher = group1_name if med1 >= med2 else group2_name
    lower  = group2_name if med1 >= med2 else group1_name

    if is_significant:
        overall = (
            f"A Mann-Whitney U test indicated a statistically significant difference between "
            f"<strong>{group1_name}</strong> (Mdn = {med1:.2f}) and <strong>{group2_name}</strong> "
            f"(Mdn = {med2:.2f}), <em>U</em> = {statistic:.1f}, {pf}. "
            f"<strong>{higher}</strong> tended to score higher than <strong>{lower}</strong>. "
            f"The effect size was {r_interp} (<em>r</em> = {effect_size_r:.3f}, "
            f"Cliff's δ = {cliffs_delta:.3f})."
        )
    else:
        overall = (
            f"A Mann-Whitney U test found no statistically significant difference between "
            f"<strong>{group1_name}</strong> (Mdn = {med1:.2f}) and <strong>{group2_name}</strong> "
            f"(Mdn = {med2:.2f}), <em>U</em> = {statistic:.1f}, {pf}. "
            f"The effect size was {r_interp} (<em>r</em> = {effect_size_r:.3f})."
        )

    ptext = "< .001" if p_value < 0.001 else f"= {p_value:.4f}"
    insights = [
        f"<strong>Test Statistic:</strong> <em>U</em> = {statistic:.1f}, <em>Z</em> = {z_score:.3f}.",
        f"<strong>Significance:</strong> <em>p</em> {ptext} — {'statistically significant.' if is_significant else 'not statistically significant.'}",
        f"<strong>Effect Size r:</strong> {effect_size_r:.3f} ({r_interp}) — computed as Z / √N.",
        f"<strong>Cliff's δ:</strong> {cliffs_delta:.3f} ({d_interp}) — probability of dominance.",
        f"<strong>Medians:</strong> {group1_name} = {med1:.2f}, {group2_name} = {med2:.2f}.",
    ]

    if not is_significant:
        recommendations = (
            "The null hypothesis cannot be rejected. Both groups appear to come from similar distributions. "
            "Consider increasing sample size or investigating whether a meaningful difference is theoretically expected."
        )
    elif abs_r < 0.1:
        recommendations = (
            "Despite statistical significance, the effect size is negligible. "
            "With large samples, even tiny differences can reach significance. Evaluate practical importance carefully."
        )
    elif abs_r >= 0.5:
        recommendations = (
            f"A large effect size (r = {effect_size_r:.3f}) indicates a substantial, practically meaningful difference. "
            f"<strong>{higher}</strong> consistently outranks <strong>{lower}</strong>. Consider tailored strategies per group."
        )
    else:
        recommendations = (
            f"A {r_interp} effect size (r = {effect_size_r:.3f}) suggests a real but moderate difference. "
            "Consider whether this difference has practical implications in your domain."
        )

    return {
        "overall_analysis": overall,
        "statistical_insights": insights,
        "recommendations": recommendations,
    }


@router.post("/mann-whitney")
def mann_whitney_test(req: MannWhitneyRequest):
    try:
        df = pd.DataFrame(req.data)
        group_col = req.group_col
        value_col = req.value_col
        alternative = req.alternative
        alpha = req.alpha

        # Get unique groups
        groups = df[group_col].dropna().unique().tolist()
        if len(groups) != 2:
            raise ValueError(f"Mann-Whitney U test requires exactly 2 groups, found {len(groups)}")

        g1_data = df[df[group_col] == groups[0]][value_col].dropna().values.astype(float)
        g2_data = df[df[group_col] == groups[1]][value_col].dropna().values.astype(float)
        n1, n2 = len(g1_data), len(g2_data)

        if n1 == 0 or n2 == 0:
            raise ValueError("One or both groups have no valid data")

        # ── Core test ────────────────────────────────────────────────────────
        statistic, p_value = stats.mannwhitneyu(g1_data, g2_data, alternative=alternative)

        # ── U1 / U2 ─────────────────────────────────────────────────────────
        combined = np.concatenate([g1_data, g2_data])
        ranks = rankdata(combined)
        R1 = float(np.sum(ranks[:n1]))
        R2 = float(np.sum(ranks[n1:]))
        U1 = R1 - n1 * (n1 + 1) / 2
        U2 = R2 - n2 * (n2 + 1) / 2
        U_min = min(U1, U2)

        # ── Ties ────────────────────────────────────────────────────────────
        has_ties = detect_ties(combined)
        method = "asymptotic"   # scipy uses asymptotic by default for large n

        # ── Z-score & effect size r ──────────────────────────────────────────
        mean_u = n1 * n2 / 2
        # Tie-corrected std
        _, tie_counts = np.unique(combined, return_counts=True)
        tie_correction = np.sum(tie_counts ** 3 - tie_counts) / 12 if has_ties else 0.0
        N = n1 + n2
        std_u = np.sqrt((n1 * n2 / 12) * ((N + 1) - tie_correction / (N * (N - 1)))) if N > 1 else 1.0
        z_score = float((U_min - mean_u) / std_u) if std_u > 0 else 0.0
        effect_size_r = float(abs(z_score) / np.sqrt(N))

        # ── Cliff's delta ────────────────────────────────────────────────────
        cliffs_delta = compute_cliffs_delta(g1_data, g2_data)

        # ── Shape check ──────────────────────────────────────────────────────
        same_shape = same_shape_test(g1_data, g2_data)
        if same_shape:
            assumption_note = (
                "The two groups appear to have similar distribution shapes (Levene p > .05). "
                "The Mann-Whitney U test can be interpreted as a test of median differences."
            )
        else:
            assumption_note = (
                "The two groups appear to have different distribution shapes (Levene p ≤ .05). "
                "Interpret results as stochastic dominance rather than a median comparison. "
                "Cliff's δ is the recommended effect size in this case."
            )

        # ── Group statistics ─────────────────────────────────────────────────
        group_stats = {
            str(groups[0]): {
                "count": int(n1),
                "mean":      float(np.mean(g1_data)),
                "median":    float(np.median(g1_data)),
                "std":       float(np.std(g1_data, ddof=1)) if n1 > 1 else 0.0,
                "min":       float(np.min(g1_data)),
                "max":       float(np.max(g1_data)),
                "mean_rank": float(R1 / n1),
            },
            str(groups[1]): {
                "count": int(n2),
                "mean":      float(np.mean(g2_data)),
                "median":    float(np.median(g2_data)),
                "std":       float(np.std(g2_data, ddof=1)) if n2 > 1 else 0.0,
                "min":       float(np.min(g2_data)),
                "max":       float(np.max(g2_data)),
                "mean_rank": float(R2 / n2),
            },
        }

        # ── Significance & text ──────────────────────────────────────────────
        is_significant = bool(p_value < alpha)
        pf = "p < .001" if p_value < 0.001 else f"p = {p_value:.4f}"
        r_interp = get_effect_size_r_interpretation(effect_size_r)
        d_interp = get_cliffs_delta_interpretation(cliffs_delta)

        if is_significant:
            decision   = "Reject null hypothesis"
            conclusion = (
                f"Mann-Whitney U = {statistic:.1f}, {pf}. "
                f"There is a statistically significant difference between the two groups. "
                f"Effect size r = {effect_size_r:.3f} ({r_interp['magnitude'].lower()}), "
                f"Cliff's δ = {cliffs_delta:.3f} ({d_interp['magnitude'].lower()})."
            )
        else:
            decision   = "Fail to reject null hypothesis"
            conclusion = (
                f"Mann-Whitney U = {statistic:.1f}, {pf}. "
                f"There is no statistically significant difference between the two groups. "
                f"Effect size r = {effect_size_r:.3f} ({r_interp['magnitude'].lower()})."
            )

        # ── Frontend interpretations ─────────────────────────────────────────
        interpretations = generate_interpretations(
            group1_name=str(groups[0]), group2_name=str(groups[1]),
            g1=g1_data, g2=g2_data,
            statistic=float(statistic), p_value=float(p_value),
            effect_size_r=effect_size_r, cliffs_delta=cliffs_delta,
            is_significant=is_significant, alpha=alpha,
            z_score=z_score,
        )

        # ── Visualization ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sns.boxplot(x=group_col, y=value_col, hue=group_col, data=df, palette='crest', legend=False, ax=axes[0])
        axes[0].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')

        sns.violinplot(x=group_col, y=value_col, hue=group_col, data=df, palette='crest', legend=False, ax=axes[1])
        axes[1].set_title('Violin Plot Comparison', fontsize=12, fontweight='bold')

        for g in groups:
            gd = df[df[group_col] == g][value_col].dropna()
            axes[2].hist(gd, alpha=0.6, label=str(g), bins=20)
        axes[2].legend()
        axes[2].set_title('Distribution Histograms', fontsize=12, fontweight='bold')
        axes[2].set_xlabel(value_col)
        axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        # ── Response ─────────────────────────────────────────────────────────
        result = {
            "test_type":   "Mann-Whitney U Test",

            # core stats
            "statistic":   float(statistic),
            "U":           float(U_min),
            "U1":          float(U1),
            "U2":          float(U2),
            "z_score":     z_score,
            "p_value":     float(p_value),
            "method":      method,
            "has_ties":    has_ties,

            # effect sizes — both naming conventions
            "effect_size":                  effect_size_r,   # legacy
            "effect_size_r":                effect_size_r,   # frontend primary
            "effect_size_r_interpretation": r_interp,        # {text, magnitude}
            "effect_size_interpretation":   r_interp,        # legacy alias

            "cliffs_delta":                 cliffs_delta,
            "cliffs_delta_interpretation":  d_interp,

            # significance
            "is_significant": is_significant,

            # interpretation object (frontend reads .decision, .conclusion, .assumption_note, .same_shape)
            "interpretation": {
                "decision":        decision,
                "conclusion":      conclusion,
                "assumption_note": assumption_note,
                "same_shape":      same_shape,
            },

            # group stats
            "group_stats": group_stats,
            "groups":      [str(g) for g in groups],
            "n1":          int(n1),
            "n2":          int(n2),
        }

        return _to_native({
            "results":        result,
            "plot":           plot_base64,
            "interpretations": interpretations,   # frontend Step 5 reads this
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
