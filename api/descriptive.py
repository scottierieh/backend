from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()


# ─────────────────────────────────────────────
# Missing value handling
# ─────────────────────────────────────────────
_MISSING_PATTERNS = {
    "", " ", "  ", "na", "n/a", "na/", "nan", "null",
    "none", "nil", ".", "-", "--", "?", "unknown", "missing",
}

def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return str(val).strip().lower() in _MISSING_PATTERNS

def _clean_series(series: pd.Series) -> pd.Series:
    """Replace all real-world missing patterns with NaN (vectorized, handles 100k+ rows)."""
    s = series.copy()
    # Step 1: None / float NaN → NaN
    s = s.where(s.notna(), other=np.nan)
    # Step 2: string normalization + pattern match via isin (faster than map)
    str_mask = s.astype(str).str.strip().str.lower().isin(_MISSING_PATTERNS)
    s[str_mask] = np.nan
    return s

def _missing_count(series: pd.Series) -> int:
    """Vectorized missing count covering NaN and string patterns."""
    null_mask = series.isna()
    str_mask  = series.astype(str).str.strip().str.lower().isin(_MISSING_PATTERNS)
    return int((null_mask | str_mask).sum())


# ─────────────────────────────────────────────
# Variable type inference
# ─────────────────────────────────────────────
_MAX_NUMERIC_UNIQUE_RATIO = 0.05
_MAX_LIKERT_UNIQUE        = 10

_BOOL_TRUTHY = {"true",  "1", "yes", "y", "on",  "t"}
_BOOL_FALSY  = {"false", "0", "no",  "n", "off", "f"}
_BOOL_VALUES = _BOOL_TRUTHY | _BOOL_FALSY

# Prevent year-only columns (2020, 2021, ...) from being classified as datetime
_YEAR_PATTERN = r"^\d{4}$"

def _infer_type(series: pd.Series) -> str:
    """
    Returns 'numeric' | 'categorical' | 'datetime' | 'binary'.
    Priority: datetime → binary → numeric → categorical.
    """
    cleaned = _clean_series(series).dropna()
    if len(cleaned) == 0:
        return "categorical"

    str_vals = cleaned.astype(str).str.strip()

    # ── datetime detection (year-only false-positive guard) ──────────────
    # Skip datetime classification if ≥90% of values are 4-digit year strings
    year_only_ratio = str_vals.str.match(_YEAR_PATTERN).sum() / len(str_vals)
    if year_only_ratio < 0.9:
        dt_converted = pd.to_datetime(str_vals, errors="coerce", infer_datetime_format=True)
        if dt_converted.notna().sum() / len(cleaned) >= 0.5:
            return "datetime"

    # ── binary detection ─────────────────────────────────────────────────
    unique_lower = set(str_vals.str.lower().unique())
    if len(unique_lower) == 2 and unique_lower.issubset(_BOOL_VALUES):
        return "binary"
    numeric_check = pd.to_numeric(str_vals, errors="coerce")
    if numeric_check.notna().all() and set(numeric_check.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
        return "binary"

    # ── numeric vs categorical ────────────────────────────────────────────
    numeric_ratio = numeric_check.notna().sum() / len(cleaned)
    if numeric_ratio < 0.5:
        return "categorical"

    num_vals     = numeric_check.dropna()
    n_unique     = num_vals.nunique()
    unique_ratio = n_unique / len(num_vals)
    all_integer  = (num_vals == num_vals.round()).all()

    if all_integer and n_unique <= _MAX_LIKERT_UNIQUE:
        return "categorical"
    if all_integer and unique_ratio <= _MAX_NUMERIC_UNIQUE_RATIO and n_unique <= 20:
        return "categorical"

    return "numeric"


# ─────────────────────────────────────────────
# Serialization helper
# ─────────────────────────────────────────────
def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
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


# ─────────────────────────────────────────────
# Numeric statistics
# ─────────────────────────────────────────────
def compute_numeric_stats(series: pd.Series) -> Optional[dict]:
    cleaned   = _clean_series(series)
    num       = pd.to_numeric(cleaned, errors="coerce").dropna()
    if len(num) == 0:
        return None

    n         = int(len(num))
    missing_n = _missing_count(series)
    mean_val  = float(num.mean())
    std_val   = float(num.std())

    q1  = float(num.quantile(0.25))
    q3  = float(num.quantile(0.75))
    iqr = q3 - q1

    # Extended quantiles
    p5  = float(num.quantile(0.05))
    p10 = float(num.quantile(0.10))
    p90 = float(num.quantile(0.90))
    p95 = float(num.quantile(0.95))

    trimmed_mean = float(stats.trim_mean(num, 0.1))
    se           = std_val / np.sqrt(n) if n > 0 else None
    cv           = (std_val / mean_val * 100) if mean_val != 0 else None

    # MAD — robust dispersion measure
    mad = float(stats.median_abs_deviation(num))

    # Outliers via IQR method
    lower_fence   = q1 - 1.5 * iqr
    upper_fence   = q3 + 1.5 * iqr
    outlier_mask  = (num < lower_fence) | (num > upper_fence)
    outlier_count = int(outlier_mask.sum())
    outlier_rate  = round(outlier_count / n * 100, 2) if n > 0 else 0

    return {
        # Counts
        "count":        n,
        "missing":      missing_n,
        "missingRate":  round(missing_n / (n + missing_n) * 100, 2) if (n + missing_n) > 0 else 0,
        # Central tendency
        "mean":         mean_val,
        "trimmedMean":  trimmed_mean,
        "median":       float(num.median()),
        # Dispersion
        "stdDev":       std_val,
        "variance":     float(num.var()),
        "se":           float(se) if se is not None else None,
        "cv":           float(cv) if cv is not None else None,
        "mad":          mad,
        "range":        float(num.max() - num.min()),
        "iqr":          iqr,
        # Quantiles
        "min":          float(num.min()),
        "p5":           p5,
        "p10":          p10,
        "q1":           q1,
        "q3":           q3,
        "p90":          p90,
        "p95":          p95,
        "max":          float(num.max()),
        # Shape
        "skewness":     float(stats.skew(num))     if n > 2 else 0,
        "kurtosis":     float(stats.kurtosis(num)) if n > 3 else 0,
        # Outliers
        "outlierCount": outlier_count,
        "outlierRate":  outlier_rate,
        "lowerFence":   round(lower_fence, 4),
        "upperFence":   round(upper_fence, 4),
    }


# ─────────────────────────────────────────────
# Datetime statistics
# ─────────────────────────────────────────────
def compute_datetime_stats(series: pd.Series) -> Optional[dict]:
    cleaned   = _clean_series(series)
    dt        = pd.to_datetime(cleaned, errors="coerce", infer_datetime_format=True).dropna()
    if len(dt) == 0:
        return None

    n         = int(len(dt))
    missing_n = _missing_count(series)
    dt_sorted = dt.sort_values().reset_index(drop=True)
    span      = dt_sorted.iloc[-1] - dt_sorted.iloc[0]

    # Frequency detection via median inter-observation gap
    detected_freq = None
    freq_label    = None
    if n >= 3:
        diffs_sec  = dt_sorted.diff().dropna().dt.total_seconds()
        median_sec = float(diffs_sec.median())
        if   median_sec <= 90:           detected_freq = "secondly";  freq_label = "per second"
        elif median_sec <= 90 * 60:      detected_freq = "minutely";  freq_label = "per minute"
        elif median_sec <= 26 * 3600:    detected_freq = "hourly";    freq_label = "per hour"
        elif median_sec <= 2   * 86400:  detected_freq = "daily";     freq_label = "daily"
        elif median_sec <= 10  * 86400:  detected_freq = "weekly";    freq_label = "weekly"
        elif median_sec <= 45  * 86400:  detected_freq = "monthly";   freq_label = "monthly"
        elif median_sec <= 100 * 86400:  detected_freq = "quarterly"; freq_label = "quarterly"
        else:                            detected_freq = "yearly";    freq_label = "yearly"

    return {
        "count":        n,
        "missing":      missing_n,
        "missingRate":  round(missing_n / (n + missing_n) * 100, 2) if (n + missing_n) > 0 else 0,
        "min":          str(dt_sorted.iloc[0]),
        "max":          str(dt_sorted.iloc[-1]),
        "median":       str(dt_sorted.iloc[n // 2]),
        "spanDays":     span.days,
        "unique":       int(dt.nunique()),
        "detectedFreq": detected_freq,
        "freqLabel":    freq_label,
    }


# ─────────────────────────────────────────────
# Binary statistics
# ─────────────────────────────────────────────
def compute_binary_stats(series: pd.Series):
    cleaned   = _clean_series(series).dropna()
    if len(cleaned) == 0:
        return None, None

    str_vals  = cleaned.astype(str).str.strip()
    normed    = str_vals.str.lower().map(lambda v: 1 if v in _BOOL_TRUTHY else 0)

    n         = int(len(normed))
    missing_n = _missing_count(series)
    true_n    = int(normed.sum())
    false_n   = n - true_n

    # Fix True/False label order — do not rely on unique() ordering
    true_label  = str_vals[normed == 1].iloc[0] if true_n  > 0 else "True"
    false_label = str_vals[normed == 0].iloc[0] if false_n > 0 else "False"

    true_pct  = round(true_n  / n * 100, 2) if n > 0 else 0
    false_pct = round(false_n / n * 100, 2) if n > 0 else 0

    summary = {
        "count":       n,
        "missing":     missing_n,
        "missingRate": round(missing_n / (n + missing_n) * 100, 2) if (n + missing_n) > 0 else 0,
        "trueLabel":   true_label,
        "falseLabel":  false_label,
        "trueCount":   true_n,
        "falseCount":  false_n,
        "truePct":     true_pct,
        "falsePct":    false_pct,
    }
    # Table always in [True row, False row] order
    table = [
        {"Value": true_label,  "Frequency": true_n,  "Percentage": true_pct},
        {"Value": false_label, "Frequency": false_n, "Percentage": false_pct},
    ]
    return table, summary


# ─────────────────────────────────────────────
# Categorical statistics
# ─────────────────────────────────────────────
def compute_categorical_stats(series: pd.Series, value_order: Optional[List[str]] = None):
    """
    value_order: explicit category order for ordinal variables.
                 Cumulative percentages follow this order when provided.
                 Defaults to frequency-descending order.
    """
    cleaned  = _clean_series(series).dropna()
    if len(cleaned) == 0:
        return None, None

    freq     = cleaned.value_counts()          # frequency-descending
    total    = int(len(cleaned))
    n_unique = int(cleaned.nunique())

    # Apply value_order for ordinal cumulative calculation
    if value_order:
        ordered      = [v for v in value_order if v in freq.index] + \
                       [v for v in freq.index if v not in value_order]
        display_freq     = freq.reindex(ordered).dropna()
        cumulative_note  = "value_order"
    else:
        display_freq    = freq
        cumulative_note = "frequency_desc"

    # High-cardinality: show top 20 + Others
    _TOP_N = 20
    if n_unique > _TOP_N:
        table_freq = display_freq.iloc[:_TOP_N]
        others_cnt = int(freq.iloc[_TOP_N:].sum())
        has_others = True
    else:
        table_freq = display_freq
        others_cnt = 0
        has_others = False

    # Cumulative follows display order
    running = 0
    table   = []
    for val, cnt in table_freq.items():
        running += int(cnt)
        table.append({
            "Value":      str(val),
            "Frequency":  int(cnt),
            "Percentage": round(float(cnt / total * 100), 2),
            "Cumulative": round(running / total * 100, 2),
        })
    if has_others:
        table.append({
            "Value":      f"Others ({n_unique - _TOP_N} categories)",
            "Frequency":  others_cnt,
            "Percentage": round(others_cnt / total * 100, 2),
            "Cumulative": 100.0,
        })

    top3_pct = round(float(freq.iloc[:3].sum() / total * 100), 2) if n_unique >= 3 else None
    top1_pct = round(float(freq.iloc[0]       / total * 100), 2) if n_unique >  0 else None
    probs    = freq / total
    entropy  = float(-np.sum(probs * np.log2(probs + 1e-12)))

    def _cr(k: int) -> Optional[float]:
        if n_unique < k:
            return None
        return round(float(freq.iloc[:k].sum() / total * 100), 2)

    summary = {
        "count":           total,
        "missing":         _missing_count(series),
        "missingRate":     round(_missing_count(series) / (total + _missing_count(series)) * 100, 2),
        "unique":          n_unique,
        "truncated":       has_others,
        "cumulativeOrder": cumulative_note,
        "mode":            str(freq.index[0]) if n_unique > 0 else None,
        "modeFreq":        int(freq.iloc[0])  if n_unique > 0 else None,
        "modePct":         top1_pct,
        "top3Pct":         top3_pct,
        "entropy":         round(entropy, 4),
        "cr2":             _cr(2),
        "cr3":             _cr(3),
        "cr4":             _cr(4),
    }

    return table, summary


# ─────────────────────────────────────────────
# Plot generation
# ─────────────────────────────────────────────
def create_plot(series: pd.Series, var_name: str, var_type: str) -> str:
    """Always returns 'data:image/png;base64,<data>'."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if var_type == "numeric":
        num = pd.to_numeric(_clean_series(series), errors="coerce").dropna()
        if len(num) > 0:
            use_kde = len(num) >= 50   # suppress KDE for small samples to avoid distortion
            sns.histplot(num, kde=use_kde, ax=ax, color="#5B9BD5")
            ax.axvline(float(num.mean()),   color="#C44E52", linestyle="--",
                       label=f"Mean: {num.mean():.2f}")
            ax.axvline(float(num.median()), color="#55A868", linestyle=":",
                       label=f"Median: {num.median():.2f}")
            ax.legend()

    elif var_type == "datetime":
        dt = pd.to_datetime(_clean_series(series), errors="coerce",
                            infer_datetime_format=True).dropna()
        if len(dt) > 0:
            dt.hist(ax=ax, bins=20, color="#5B9BD5", edgecolor="white")
            ax.set_xlabel("Date")

    elif var_type == "binary":
        cat = _clean_series(series).dropna().astype(str).str.strip()
        if len(cat) > 0:
            freq   = cat.value_counts()
            colors = ["#5B9BD5", "#C44E52"]
            ax.bar(freq.index.astype(str), freq.values,
                   color=colors[:len(freq)], edgecolor="white")
            ax.set_ylabel("Count")
            for i, v in enumerate(freq.values):
                ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")

    else:  # categorical
        cat = _clean_series(series).dropna()
        if len(cat) > 0:
            freq = cat.value_counts().head(20)
            sns.barplot(x=freq.values, y=freq.index.astype(str), ax=ax, palette="crest")
            ax.set_xlabel("Count")

    ax.set_title(f"Distribution of {var_name}", fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ─────────────────────────────────────────────
# Group comparison
# ─────────────────────────────────────────────
def compute_group_comparison(df: pd.DataFrame, var: str, group_by: str, var_type: str) -> dict:
    """Per-group statistics with deviation from overall mean for numeric variables."""
    overall_stats = compute_numeric_stats(df[var]) if var_type == "numeric" else None
    grouped_stats = {}

    for group_name, gdf in df.groupby(group_by):
        key = str(group_name)
        if var_type == "numeric":
            gs = compute_numeric_stats(gdf[var])
            if gs and overall_stats:
                gs["diffFromMean"] = round(gs["mean"] - overall_stats["mean"], 4)
            if gs:
                gs["groupCount"] = int(len(gdf))
            grouped_stats[key] = gs
        elif var_type == "binary":
            g_table, g_summary = compute_binary_stats(gdf[var])
            if g_summary:
                g_summary["groupCount"] = int(len(gdf))
            grouped_stats[key] = {"table": g_table, "summary": g_summary}
        elif var_type == "datetime":
            gs = compute_datetime_stats(gdf[var])
            if gs:
                gs["groupCount"] = int(len(gdf))
            grouped_stats[key] = gs
        else:
            g_table, g_summary = compute_categorical_stats(gdf[var])
            if g_summary:
                g_summary["groupCount"] = int(len(gdf))
            grouped_stats[key] = {"table": g_table, "summary": g_summary}

    comparison_meta = {}
    if var_type == "numeric":
        means = {k: v["mean"] for k, v in grouped_stats.items() if v and "mean" in v}
        if means:
            comparison_meta["highestGroup"]   = max(means, key=means.get)
            comparison_meta["lowestGroup"]    = min(means, key=means.get)
            comparison_meta["groupMeanRange"] = round(
                max(means.values()) - min(means.values()), 4
            )

    return {"groups": grouped_stats, "comparisonMeta": comparison_meta}


# ─────────────────────────────────────────────
# Correlation matrix
# ─────────────────────────────────────────────
def _corr_label(r: float) -> str:
    a = abs(r)
    if a >= 0.9: return "very strong"
    if a >= 0.7: return "strong"
    if a >= 0.5: return "moderate"
    if a >= 0.3: return "weak"
    return "negligible"

_MAX_CORR_VARIABLES = 30   # cap to avoid O(n²) slowdown

def compute_correlation_matrix(df: pd.DataFrame, variables: List[str]) -> Optional[dict]:
    """
    Pearson + Spearman correlation matrices with p-values for all numeric variables.
    Capped at 30 variables; upper triangle masked in heatmap; insight threshold |r| >= 0.5.
    """
    num_cols = []
    for var in variables:
        if var in df.columns:
            converted = pd.to_numeric(_clean_series(df[var]), errors="coerce")
            if converted.notna().sum() >= 2:
                num_cols.append(var)

    if len(num_cols) < 2:
        return None

    truncated_corr = False
    if len(num_cols) > _MAX_CORR_VARIABLES:
        num_cols       = num_cols[:_MAX_CORR_VARIABLES]
        truncated_corr = True

    num_df = pd.DataFrame(
        {col: pd.to_numeric(_clean_series(df[col]), errors="coerce") for col in num_cols}
    )

    def _build_corr_with_pval(method: str):
        n_cols = len(num_cols)
        r_mat  = np.ones((n_cols, n_cols))
        p_mat  = np.full((n_cols, n_cols), np.nan)
        fn     = stats.pearsonr if method == "pearson" else stats.spearmanr

        for i in range(n_cols):
            for j in range(i + 1, n_cols):   # upper triangle only, then mirror
                xi     = num_df[num_cols[i]].dropna()
                xj     = num_df[num_cols[j]].dropna()
                common = xi.index.intersection(xj.index)
                if len(common) >= 3:
                    r_v, p_v     = fn(xi.loc[common], xj.loc[common])
                    r_mat[i, j]  = r_mat[j, i] = float(r_v)
                    p_mat[i, j]  = p_mat[j, i] = float(p_v)
                else:
                    r_mat[i, j]  = r_mat[j, i] = np.nan
                    p_mat[i, j]  = p_mat[j, i] = np.nan

        flat = []
        for i in range(n_cols):
            for j in range(n_cols):
                r_v, p_v = r_mat[i, j], p_mat[i, j]
                flat.append({
                    "var1": num_cols[i],
                    "var2": num_cols[j],
                    "r":    None if np.isnan(r_v) else round(float(r_v), 4),
                    "p":    None if np.isnan(p_v) else round(float(p_v), 4),
                    "sig":  (not np.isnan(p_v)) and (p_v < 0.05),
                })

        matrix_fmt = {
            "columns": num_cols,
            "r": [[None if np.isnan(r_mat[i, j]) else round(float(r_mat[i, j]), 4)
                   for j in range(n_cols)] for i in range(n_cols)],
            "p": [[None if np.isnan(p_mat[i, j]) else round(float(p_mat[i, j]), 4)
                   for j in range(n_cols)] for i in range(n_cols)],
        }
        r_df = pd.DataFrame(r_mat, index=num_cols, columns=num_cols)
        return r_df, p_mat, flat, matrix_fmt

    pearson_r_df,  pearson_p,  pearson_flat,  pearson_matrix  = _build_corr_with_pval("pearson")
    spearman_r_df, spearman_p, spearman_flat, spearman_matrix = _build_corr_with_pval("spearman")

    def _corr_heatmap(r_df: pd.DataFrame, p_mat: np.ndarray, method: str) -> str:
        n    = len(r_df)
        mask = np.triu(np.ones((n, n), dtype=bool))   # mask upper triangle + diagonal
        annot = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    annot[i, j] = ""
                else:
                    r_v, p_v = r_df.iloc[i, j], p_mat[i, j]
                    star = ""
                    if not np.isnan(p_v):
                        if p_v < 0.001:  star = "***"
                        elif p_v < 0.01: star = "**"
                        elif p_v < 0.05: star = "*"
                    annot[i, j] = f"{r_v:.2f}{star}"

        fig_size = max(5, n)
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size - 1))
        sns.heatmap(
            r_df, mask=mask, annot=annot, fmt="", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={"size": 9}, square=True,
        )
        ax.set_title(
            f"{method.capitalize()} Correlation Matrix\n(* p<.05  ** p<.01  *** p<.001)",
            fontweight="bold", fontsize=10,
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    _INSIGHT_THRESHOLD = 0.5
    strong_pairs  = []
    corr_insights = []
    n_cols = len(num_cols)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            r_v = pearson_r_df.iloc[i, j]
            p_v = pearson_p[i, j]
            if np.isnan(r_v):
                continue
            sig  = (not np.isnan(p_v)) and (p_v < 0.05)
            pair = {
                "var1":      num_cols[i],
                "var2":      num_cols[j],
                "pearsonR":  round(float(r_v), 4),
                "p":         round(float(p_v), 4) if not np.isnan(p_v) else None,
                "sig":       sig,
                "direction": "positive" if r_v > 0 else "negative",
            }
            if abs(r_v) >= _INSIGHT_THRESHOLD:
                strong_pairs.append(pair)
                label   = _corr_label(r_v)
                direct  = "positive" if r_v > 0 else "negative"
                sig_str = f"p={p_v:.3f}, {'statistically significant' if sig else 'not significant'}"
                corr_insights.append(
                    f"'{num_cols[i]}' and '{num_cols[j]}' show a {label} {direct} correlation "
                    f"(r={r_v:.2f}, {sig_str})."
                )

    return {
        "variables":   num_cols,
        "truncated":   truncated_corr,
        "pearson":     {"flat": pearson_flat,  "matrix": pearson_matrix,  "plot": _corr_heatmap(pearson_r_df,  pearson_p,  "pearson")},
        "spearman":    {"flat": spearman_flat, "matrix": spearman_matrix, "plot": _corr_heatmap(spearman_r_df, spearman_p, "spearman")},
        "strongPairs": strong_pairs,
        "insights":    corr_insights,
    }


# ─────────────────────────────────────────────
# Insight generation
# ─────────────────────────────────────────────
def generate_insights(series: pd.Series, var_name: str, var_type: str,
                      group_comparison: Optional[dict] = None) -> List[str]:
    insights = []

    if var_type == "datetime":
        dt_stat = compute_datetime_stats(series)
        if dt_stat:
            insights.append(
                f"Date range: {dt_stat['min']} to {dt_stat['max']} ({dt_stat['spanDays']} days)."
            )
            if dt_stat.get("freqLabel"):
                insights.append(
                    f"Observation interval detected as {dt_stat['freqLabel']} "
                    f"(detectedFreq: {dt_stat['detectedFreq']})."
                )
            if dt_stat["unique"] == dt_stat["count"]:
                insights.append("All dates are unique — no duplicate timestamps in the series.")
            elif dt_stat["unique"] < dt_stat["count"]:
                dup = dt_stat["count"] - dt_stat["unique"]
                insights.append(f"{dup} duplicate date(s) detected.")
        return insights

    if var_type == "binary":
        _, summary = compute_binary_stats(series)
        if summary:
            tp, fp = summary["truePct"], summary["falsePct"]
            ratio  = max(tp, fp) / (min(tp, fp) + 1e-9)
            insights.append(f"True: {tp:.1f}% / False: {fp:.1f}%.")
            if ratio > 4:
                insights.append(
                    f"Severe class imbalance detected (ratio ~{ratio:.1f}:1). "
                    "Consider oversampling or class weighting during modeling."
                )
        return insights

    if var_type == "numeric":
        num = pd.to_numeric(_clean_series(series), errors="coerce").dropna()
        if len(num) == 0:
            return insights

        mean_val   = float(num.mean())
        median_val = float(num.median())
        skew_val   = float(stats.skew(num)) if len(num) > 2 else 0
        q1, q3     = float(num.quantile(0.25)), float(num.quantile(0.75))
        iqr        = q3 - q1
        std_val    = float(num.std())

        # Skewness
        if skew_val > 1:
            insights.append(
                f"{var_name} is right-skewed (skewness: {skew_val:.2f}). "
                "The mean exceeds the median; extreme high values may be present."
            )
        elif skew_val < -1:
            insights.append(
                f"{var_name} is left-skewed (skewness: {skew_val:.2f}). "
                "The mean is below the median; extreme low values may be present."
            )
        else:
            insights.append(f"{var_name} is approximately symmetric (skewness: {skew_val:.2f}).")

        # Mean–median divergence
        diff_pct = abs(mean_val - median_val) / (abs(median_val) + 1e-9) * 100
        if diff_pct > 10:
            insights.append(
                f"Mean ({mean_val:.2f}) and median ({median_val:.2f}) diverge by {diff_pct:.1f}%, "
                "suggesting the influence of outliers or an asymmetric distribution."
            )

        # Outliers
        outliers = int(((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum())
        if outliers > 0:
            insights.append(
                f"{outliers} outlier(s) detected via IQR method ({outliers/len(num)*100:.1f}% of observations)."
            )

        # Coefficient of variation
        cv = std_val / mean_val * 100 if mean_val != 0 else 0
        if cv > 100:
            insights.append(f"CV is {cv:.1f}% — very high variability relative to the mean.")
        elif cv < 10:
            insights.append(
                f"CV is {cv:.1f}% — values are tightly clustered around the mean (possible low variance)."
            )

        # Ceiling / floor effects
        if (num == num.max()).mean() * 100 > 5:
            insights.append(f"Responses concentrate at the maximum ({num.max():.2f}) — possible ceiling effect.")
        if (num == num.min()).mean() * 100 > 5:
            insights.append(f"Responses concentrate at the minimum ({num.min():.2f}) — possible floor effect.")

        # Group comparison
        if group_comparison:
            meta = group_comparison.get("comparisonMeta", {})
            if meta.get("groupMeanRange") is not None:
                insights.append(
                    f"Group mean range: {meta['groupMeanRange']:.2f}. "
                    f"Highest group: '{meta['highestGroup']}', lowest group: '{meta['lowestGroup']}'."
                )

    else:  # categorical
        cat = _clean_series(series).dropna()
        if len(cat) == 0:
            return insights

        freq     = cat.value_counts()
        total    = len(cat)
        top_pct  = float(freq.iloc[0] / total * 100) if len(freq) > 0 else 0
        n_unique = int(cat.nunique())

        if top_pct > 50:
            insights.append(
                f"'{freq.index[0]}' accounts for {top_pct:.1f}% of responses — dominant category."
            )
        if n_unique > 20:
            insights.append(
                f"High cardinality: {n_unique} unique values. Consider grouping or encoding."
            )
        if len(freq) >= 2:
            imbalance = float(freq.iloc[0] / freq.iloc[-1])
            if imbalance > 10:
                insights.append(
                    f"Frequency ratio between the most and least common categories is {imbalance:.1f}:1 — severe imbalance."
                )
        probs   = freq / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        max_ent = np.log2(n_unique) if n_unique > 1 else 1
        if entropy / max_ent < 0.5:
            insights.append(
                f"Low entropy ({entropy:.2f}) — distribution is heavily concentrated in a few categories."
            )

    return insights


# ─────────────────────────────────────────────
# Request model + Endpoint
# ─────────────────────────────────────────────
class DescriptiveRequest(BaseModel):
    data:       list[dict[str, Any]]           = Field(...)
    variables:  List[str]                      = Field(...)
    groupBy:    Optional[str]                  = Field(default=None)
    valueOrder: Optional[dict[str, List[str]]] = Field(
        default=None,
        description="Explicit category order for ordinal variables. e.g. {'grade': ['low','medium','high']}",
    )


@router.post("/descriptive")
def descriptive_analysis(req: DescriptiveRequest):
    try:
        df              = pd.DataFrame(req.data)
        variables       = req.variables
        group_by        = req.groupBy
        value_order_map = req.valueOrder or {}
        results         = {}

        for var in variables:
            if var not in df.columns:
                results[var] = {"error": f"Variable '{var}' not found"}
                continue

            series   = df[var]
            var_type = _infer_type(series)

            group_comparison = None
            if group_by and group_by in df.columns and group_by != var:
                group_comparison = compute_group_comparison(df, var, group_by, var_type)

            if var_type == "numeric":
                stat = compute_numeric_stats(series)
                if stat is None:
                    results[var] = {"error": "No valid numeric data"}
                    continue
                results[var] = {
                    "type":            "numeric",
                    "stats":           stat,
                    "plot":            create_plot(series, var, "numeric"),
                    "insights":        generate_insights(series, var, "numeric", group_comparison),
                    "groupComparison": group_comparison,
                }

            elif var_type == "datetime":
                stat = compute_datetime_stats(series)
                if stat is None:
                    results[var] = {"error": "No valid datetime data"}
                    continue
                results[var] = {
                    "type":            "datetime",
                    "stats":           stat,
                    "plot":            create_plot(series, var, "datetime"),
                    "insights":        generate_insights(series, var, "datetime"),
                    "groupComparison": group_comparison,
                }

            elif var_type == "binary":
                table, summary = compute_binary_stats(series)
                if table is None:
                    results[var] = {"error": "No valid binary data"}
                    continue
                results[var] = {
                    "type":            "binary",
                    "table":           table,
                    "summary":         summary,
                    "plot":            create_plot(series, var, "binary"),
                    "insights":        generate_insights(series, var, "binary"),
                    "groupComparison": group_comparison,
                }

            else:  # categorical
                v_order = value_order_map.get(var)
                table, summary = compute_categorical_stats(series, value_order=v_order)
                if table is None:
                    results[var] = {"error": "No valid categorical data"}
                    continue
                results[var] = {
                    "type":            "categorical",
                    "table":           table,
                    "summary":         summary,
                    "plot":            create_plot(series, var, "categorical"),
                    "insights":        generate_insights(series, var, "categorical", group_comparison),
                    "groupComparison": group_comparison,
                }

        return _to_native({
            "results":           results,
            "correlationMatrix": compute_correlation_matrix(df, variables),
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
