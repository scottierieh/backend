from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import base64
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

_TOP_N = 20   # max categories shown in display table


# ─────────────────────────────────────────────
# Missing value handling — same rules as descriptive.py
# ─────────────────────────────────────────────
_MISSING_PATTERNS = {
    "", " ", "  ", "na", "n/a", "na/", "nan", "null",
    "none", "nil", ".", "-", "--", "?", "unknown", "missing",
}

def _clean_series(series: pd.Series) -> pd.Series:
    """Replace all real-world missing patterns with NaN (vectorized)."""
    s        = series.copy()
    s        = s.where(s.notna(), other=np.nan)
    str_mask = s.astype(str).str.strip().str.lower().isin(_MISSING_PATTERNS)
    s[str_mask] = np.nan
    return s

def _missing_count(series: pd.Series) -> int:
    null_mask = series.isna()
    str_mask  = series.astype(str).str.strip().str.lower().isin(_MISSING_PATTERNS)
    return int((null_mask | str_mask).sum())


# ─────────────────────────────────────────────
# Variable type detection
# ─────────────────────────────────────────────
_CONTINUOUS_UNIQUE_RATIO = 0.20
_CONTINUOUS_MIN_UNIQUE   = 30

def _detect_var_type(series: pd.Series) -> dict:
    cleaned       = _clean_series(series).dropna()
    n             = len(cleaned)
    if n == 0:
        return {"type": "categorical", "message": None, "unique_count": 0, "numeric_ratio": 0.0}

    num_conv      = pd.to_numeric(cleaned, errors="coerce")
    numeric_ratio = float(num_conv.notna().sum() / n)
    n_unique      = int(cleaned.nunique())
    unique_ratio  = n_unique / n

    if n_unique == 2:
        return {"type": "binary", "message": None, "unique_count": n_unique, "numeric_ratio": numeric_ratio}

    if numeric_ratio >= 0.8 and n_unique >= _CONTINUOUS_MIN_UNIQUE and unique_ratio >= _CONTINUOUS_UNIQUE_RATIO:
        msg = (
            f"This variable appears to be continuous ({n_unique} unique values, "
            f"unique ratio: {unique_ratio:.1%}). Descriptive statistics are recommended over frequency analysis."
        )
        return {"type": "continuous_warning", "message": msg, "unique_count": n_unique, "numeric_ratio": numeric_ratio}

    return {"type": "categorical", "message": None, "unique_count": n_unique, "numeric_ratio": numeric_ratio}


# ─────────────────────────────────────────────
# Near-zero variance threshold (sample-size adjusted)
# ─────────────────────────────────────────────
def _nzv_threshold(n: int) -> float:
    """
    Freq-ratio threshold scales with sample size:
      n < 100  → 9x  (relaxed for small samples)
      n < 1000 → 19x (standard)
      n ≥ 1000 → 29x (tightened for large samples)
    """
    if n < 100:  return 9.0
    if n < 1000: return 19.0
    return 29.0


# ─────────────────────────────────────────────
# Serialization helper
# ─────────────────────────────────────────────
def _to_native(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    if isinstance(obj, np.ndarray):  return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, dict):        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [_to_native(x) for x in obj]
    return obj


# ─────────────────────────────────────────────
# Core analysis function
# ─────────────────────────────────────────────
def analyze_single_variable(
    df: pd.DataFrame,
    variable: str,
    value_order: Optional[List[str]] = None,
    rare_threshold: float = 0.0,
) -> dict:
    """
    Frequency analysis for a single variable.

    Processing order (prevents Rare+Others collision and cumulative drift):
      1. Drop missing values
      2. Merge Rare   — always before Others, so they never coexist in the table
      3. Apply value_order — after Rare removal to preserve ordinal structure
      4. Split top-N + Others
      5. Build cumulative based solely on final display order

    value_order    : explicit category order for ordinal variables
    rare_threshold : if 0 < x < 1, categories whose share < x are merged into Rare first
    """
    try:
        raw_series = df[variable]
        missing_n  = _missing_count(raw_series)

        # value_counts computed once and reused throughout
        series = _clean_series(raw_series).dropna()
        n      = len(series)
        if n == 0:
            return {"error": f"No valid data for variable '{variable}'"}

        type_info     = _detect_var_type(series)
        all_counts    = series.value_counts()      # original counts — used for stats only
        total         = int(all_counts.sum())
        n_unique_orig = int(len(all_counts))

        # ── Statistical metrics (based on original counts) ───────────────
        probs              = all_counts.values / total
        entropy            = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy        = float(np.log2(n_unique_orig)) if n_unique_orig > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        freq_ratio         = (
            float(all_counts.values[0] / all_counts.values[1])
            if n_unique_orig >= 2 and all_counts.values[1] > 0 else None
        )
        mode        = str(all_counts.index[0])
        small_cells = int((all_counts < 5).sum())

        # Top-N box — key survey metrics (frequency-desc original order)
        mode_pct = round(float(all_counts.values[0] / total * 100), 2)
        top2_pct = round(float(all_counts.values[:2].sum() / total * 100), 2) if n_unique_orig >= 2 else mode_pct
        top3_pct = round(float(all_counts.values[:3].sum() / total * 100), 2) if n_unique_orig >= 3 else top2_pct

        # ── STEP 1: Merge Rare (always before Others) ────────────────────
        rare_merged_count = 0
        rare_merged_n     = 0
        work_counts       = all_counts.copy()

        if 0 < rare_threshold < 1:
            rare_mask         = (work_counts / total) < rare_threshold
            rare_merged_n     = int(work_counts[rare_mask].sum())
            rare_merged_count = int(rare_mask.sum())
            work_counts       = work_counts[~rare_mask]
            # Categories removed as Rare are fully excluded from display.
            # If they appeared in value_order, they are silently dropped — no ordinal conflict.

        # ── STEP 2: Apply value_order (after Rare removal) ───────────────
        if value_order:
            ordered_vals   = [v for v in value_order if v in work_counts.index]
            trailing_vals  = [v for v in work_counts.index if v not in value_order]
            display_counts = work_counts.reindex(ordered_vals + trailing_vals).dropna()
            cumulative_basis = "value_order"
        else:
            display_counts   = work_counts
            cumulative_basis = "frequency_desc"

        # ── STEP 3: top-N + Others (from remaining categories after Rare) ─
        has_others   = len(display_counts) > _TOP_N
        table_counts = display_counts.iloc[:_TOP_N]
        others_count = int(display_counts.iloc[_TOP_N:].sum()) if has_others else 0

        # ── STEP 4: Build frequency table ────────────────────────────────
        # Cumulative runs continuously: main rows → Others → Rare (at the bottom)
        table   = []
        running = 0.0

        for val, cnt in table_counts.items():
            pct      = cnt / total * 100
            running += pct
            table.append({
                "Value":             str(val),
                "Frequency":         int(cnt),
                "Percentage":        round(float(pct), 2),
                "CumulativePercent": round(float(running), 2),
            })

        if has_others:
            op       = others_count / total * 100
            running += op
            table.append({
                "Value":             f"Others ({len(display_counts) - _TOP_N} categories)",
                "Frequency":         others_count,
                "Percentage":        round(op, 2),
                "CumulativePercent": round(running, 2),
            })

        if rare_merged_count > 0:
            rp       = rare_merged_n / total * 100
            running += rp
            table.append({
                "Value":             f"Rare (<{rare_threshold:.1%}, {rare_merged_count} categories)",
                "Frequency":         rare_merged_n,
                "Percentage":        round(rp, 2),
                "CumulativePercent": round(min(running, 100.0), 2),
            })

        # ── Full table (for download) ─────────────────────────────────────
        full_table = []
        running_f  = 0.0
        for val, cnt in display_counts.items():
            pct       = cnt / total * 100
            running_f += pct
            full_table.append({
                "Value":             str(val),
                "Frequency":         int(cnt),
                "Percentage":        round(float(pct), 2),
                "CumulativePercent": round(float(running_f), 2),
            })
        if rare_merged_count > 0:
            rp = rare_merged_n / total * 100
            full_table.append({
                "Value":             f"Rare (<{rare_threshold:.1%})",
                "Frequency":         rare_merged_n,
                "Percentage":        round(rp, 2),
                "CumulativePercent": round(min(running_f + rp, 100.0), 2),
            })

        # ── Summary ──────────────────────────────────────────────────────
        summary = {
            "total_count":        total,
            "missing_count":      missing_n,
            "missing_rate":       round(missing_n / (total + missing_n) * 100, 2) if (total + missing_n) > 0 else 0,
            "unique_categories":  n_unique_orig,
            "truncated":          has_others,
            "cumulative_basis":   cumulative_basis,
            "rare_merged_count":  rare_merged_count,
            "rare_merged_n":      rare_merged_n,
            "mode":               mode,
            "mode_pct":           mode_pct,   # Top 1 box
            "top2_pct":           top2_pct,   # Top 2 box
            "top3_pct":           top3_pct,   # Top 3 box
            "entropy":            round(entropy, 4),
            "max_entropy":        round(max_entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "freq_ratio":         round(freq_ratio, 4) if freq_ratio is not None else None,
            "var_type":           type_info["type"],
            "numeric_ratio":      round(type_info["numeric_ratio"], 4),
        }

        # ── Insights & Recommendations ────────────────────────────────────
        insights        = []
        recommendations = []

        if type_info["type"] == "continuous_warning":
            insights.append({
                "type": "warning", "title": "Continuous Variable Detected",
                "description": type_info["message"],
            })
            recommendations.append(
                "Use descriptive statistics (mean, median, std) instead of frequency analysis. "
                "A histogram or box plot is more appropriate."
            )

        if n_unique_orig == 1:
            insights.append({
                "type": "warning", "title": "Zero Variance",
                "description": f"Only one unique value '{mode}' exists. This variable carries no information.",
            })
            recommendations.append("Consider removing this variable from the analysis.")

        elif freq_ratio is not None:
            nzv_thr = _nzv_threshold(total)
            if freq_ratio > nzv_thr:
                insights.append({
                    "type": "warning", "title": "Near-Zero Variance",
                    "description": (
                        f"The most frequent category is {freq_ratio:.1f}x more common than the second "
                        f"(threshold for n={total}: {nzv_thr:.0f}x). This may cause issues in modeling."
                    ),
                })
                recommendations.append(
                    "Consider collapsing rare categories or applying imbalanced-data techniques."
                )

        # Top-N box insight
        insights.append({
            "type": "info", "title": "Top-N Box",
            "description": (
                f"Most frequent category '{mode}': {mode_pct:.1f}% (Top 1 box). "
                f"Top 2 categories combined: {top2_pct:.1f}% (Top 2 box). "
                f"Top 3 categories combined: {top3_pct:.1f}% (Top 3 box)."
            ),
        })

        # Distribution balance (4 levels)
        if n_unique_orig > 1:
            if normalized_entropy >= 0.9:
                insights.append({
                    "type": "info", "title": "Balanced Distribution ✓",
                    "description": (
                        f"Categories are evenly distributed (normalized entropy: {normalized_entropy:.3f}). "
                        "Suitable for standard categorical analyses such as chi-square tests."
                    ),
                })
            elif normalized_entropy >= 0.7:
                insights.append({
                    "type": "info", "title": "Moderately Balanced Distribution",
                    "description": (
                        f"Distribution is reasonably balanced (normalized entropy: {normalized_entropy:.3f}). "
                        "Some categories dominate slightly but this should not affect most analyses."
                    ),
                })
            elif normalized_entropy >= 0.5:
                insights.append({
                    "type": "warning", "title": "Moderately Skewed Distribution",
                    "description": (
                        f"Distribution is moderately skewed (normalized entropy: {normalized_entropy:.3f}). "
                        f"'{mode}' is dominant, though other categories still hold meaningful shares."
                    ),
                })
                recommendations.append("Check whether the imbalance may affect model performance.")
            else:
                insights.append({
                    "type": "warning", "title": "Heavily Skewed Distribution",
                    "description": (
                        f"Distribution is heavily skewed (normalized entropy: {normalized_entropy:.3f}). "
                        f"'{mode}' accounts for {mode_pct:.1f}% of all responses."
                    ),
                })
                recommendations.append("Verify whether the imbalance reflects the population or sampling bias.")

        if rare_merged_count > 0:
            insights.append({
                "type": "info", "title": f"Rare Categories Merged ({rare_merged_count})",
                "description": (
                    f"{rare_merged_count} categories below the {rare_threshold:.1%} threshold "
                    f"({rare_merged_n} observations) were merged into 'Rare'. "
                    "They appear separately at the bottom of the table, distinct from 'Others'."
                ),
            })

        if n_unique_orig > _TOP_N:
            insights.append({
                "type": "warning", "title": "High Cardinality",
                "description": (
                    f"This variable has {n_unique_orig} unique categories. "
                    f"The table shows the top {_TOP_N} + Others; see full_table for all categories."
                ),
            })
            recommendations.append(
                "Use rareThreshold to merge infrequent categories first, which improves readability."
            )

        if small_cells > 0 and n_unique_orig > 1:
            chi_feasible = (small_cells / n_unique_orig) <= 0.20
            test_advice  = (
                f"Small cells represent {small_cells/n_unique_orig:.0%} of categories (≤20%), "
                "so chi-square is still feasible but interpret results with caution."
            ) if chi_feasible else (
                f"Small cells represent {small_cells/n_unique_orig:.0%} of categories, "
                "violating chi-square assumptions. Use Fisher's exact test or collapse categories."
            )
            insights.append({
                "type": "warning", "title": "Small Cell Counts",
                "description": f"{small_cells} categories have fewer than 5 observations. {test_advice}",
            })
            recommendations.append(
                "Merge small-count categories with adjacent ones, or use Fisher's exact test / Monte Carlo simulation."
            )

        if not insights:
            insights.append({
                "type": "info", "title": "No Issues Detected ✓",
                "description": "No distributional anomalies detected. Suitable for standard categorical analyses.",
            })
        if not recommendations:
            recommendations.append("Distribution is suitable for standard categorical analyses.")

        plot_base64 = _create_plot(variable, display_counts, total, n_unique_orig, type_info["type"])

        return {
            "table":           table,
            "full_table":      full_table,
            "summary":         summary,
            "insights":        insights,
            "recommendations": recommendations,
            "plot":            plot_base64,
        }

    except Exception as e:
        logger.exception("analyze_single_variable failed for '%s'", variable)
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def _create_plot(
    variable: str,
    display_counts: pd.Series,
    total: int,
    unique_count: int,
    var_type: str = "categorical",
) -> str:
    if var_type == "binary":
        return _create_binary_plot(variable, display_counts, total)
    return _create_bar_plot(variable, display_counts, total, unique_count)


def _create_binary_plot(variable: str, display_counts: pd.Series, total: int) -> str:
    """Binary variable — vertical bar chart with percentage labels."""
    labels = [str(v) for v in display_counts.index[:2]]
    values = display_counts.values[:2]
    pcts   = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = ["#5B9BD5", "#C44E52"]
    bars    = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, val, pct in zip(bars, values, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Frequency")
    ax.set_title(f"Binary Distribution: {variable}", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, alpha=0.3, axis="y")
    sns.despine()
    plt.tight_layout()
    return _fig_to_b64(fig)


def _create_bar_plot(
    variable: str,
    display_counts: pd.Series,
    total: int,
    unique_count: int,
) -> str:
    """Categorical — horizontal bar chart with dynamic size/font scaling."""
    plot_data  = display_counts.head(_TOP_N)
    has_others = unique_count > _TOP_N
    n_bars     = len(plot_data)

    # Dynamic scaling by category count
    if n_bars <= 5:
        row_h, label_fs, fig_w = 0.70, 10, 10
    elif n_bars <= 10:
        row_h, label_fs, fig_w = 0.55,  9, 10
    elif n_bars <= 20:
        row_h, label_fs, fig_w = 0.45,  8, 11
    else:
        row_h, label_fs, fig_w = 0.38,  7, 12

    fig_h = max(4, n_bars * row_h + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = sns.color_palette("Blues_d", n_bars)
    bars   = ax.barh(range(n_bars), plot_data.values, color=colors)

    max_label_len = 25 if n_bars > 15 else 30
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels([str(v)[:max_label_len] for v in plot_data.index], fontsize=label_fs)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    title_suffix = f" (Top {_TOP_N} of {unique_count})" if has_others else ""
    ax.set_title(f"Frequency Distribution: {variable}{title_suffix}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    max_val = max(plot_data.values) if n_bars > 0 else 1
    for i, (bar, val) in enumerate(zip(bars, plot_data.values)):
        pct = val / total * 100
        ax.text(val + max_val * 0.01, i, f"{val} ({pct:.1f}%)", va="center", fontsize=label_fs - 1)

    if has_others:
        others_cnt = total - int(plot_data.sum())
        ax.text(
            0.99, -0.04,
            f"+ {unique_count - _TOP_N} more categories ({others_cnt} obs, {others_cnt/total*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="gray", style="italic",
        )

    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# Request model + Endpoint
# ─────────────────────────────────────────────
class FrequencyRequest(BaseModel):
    data:          List[Dict[str, Any]]
    variables:     List[str]
    valueOrder:    Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Explicit category order for ordinal variables. e.g. {'grade': ['low','medium','high']}",
    )
    rareThreshold: float = Field(
        default=0.0, ge=0.0, lt=1.0,
        description=(
            "Auto-merge threshold for rare categories (0–1). "
            "e.g. 0.01 merges categories with < 1% share into 'Rare'. 0 = disabled (default)."
        ),
    )


@router.post("/frequency")
async def frequency_analysis(request: FrequencyRequest):
    try:
        df              = pd.DataFrame(request.data)
        variables       = request.variables
        value_order_map = request.valueOrder or {}
        rare_threshold  = request.rareThreshold

        if not variables:
            raise HTTPException(status_code=400, detail="No variables provided")

        missing_vars = [v for v in variables if v not in df.columns]
        if missing_vars:
            raise HTTPException(status_code=400, detail=f"Variables not found: {missing_vars}")

        results = {}
        for var in variables:
            results[var] = analyze_single_variable(
                df, var,
                value_order=value_order_map.get(var),
                rare_threshold=rare_threshold,
            )

        return _to_native({"results": results})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("frequency_analysis endpoint error")
        raise HTTPException(status_code=500, detail=str(e))
