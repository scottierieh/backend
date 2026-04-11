from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
import io, base64, logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Request model ────────────────────────────────────────────────────────────

SUPPORTED_METRICS = {
    'euclidean', 'manhattan', 'cosine', 'chebyshev',
    'canberra', 'braycurtis', 'minkowski',
}

class HDBSCANRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(...)
    items: List[str] = Field(...)
    min_cluster_size: int = Field(default=5, ge=2)
    min_samples: Optional[int] = Field(default=None, ge=1)
    scalerType: str = Field(default='standard', pattern='^(standard|robust|minmax)$')
    distanceMetric: str = Field(default='euclidean')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj

def _safe_float(val, default=0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError) as exc:
        logger.debug("_safe_float failed for %r: %s", val, exc)
        return default


# ─── Validation ───────────────────────────────────────────────────────────────

def _validate_input(
    df: pd.DataFrame,
    items: List[str],
    min_cluster_size: int,
    min_samples: Optional[int],
    metric: str = 'euclidean',
) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if min_cluster_size < 2:
        errors.append(f"min_cluster_size must be ≥ 2, got {min_cluster_size}.")
    if min_samples is not None and min_samples < 1:
        errors.append(f"min_samples must be ≥ 1, got {min_samples}.")
    if metric not in SUPPORTED_METRICS:
        errors.append(
            f"Unsupported metric '{metric}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}."
        )

    missing = [c for c in items if c not in df.columns]
    if missing:
        errors.append(f"Column(s) not found: {', '.join(missing)}.")
    if errors:
        return {'errors': errors, 'warnings': warnings}

    # Numeric check
    non_numeric = []
    for col in items:
        converted = pd.to_numeric(df[col], errors='coerce')
        n_invalid = converted.isna().sum() - df[col].isna().sum()
        if n_invalid > 0:
            non_numeric.append(f"'{col}' has {n_invalid} non-numeric value(s)")
    if non_numeric:
        errors.append("Non-numeric data: " + "; ".join(non_numeric))
    if errors:
        return {'errors': errors, 'warnings': warnings}

    num_df = df[items].apply(pd.to_numeric, errors='coerce')
    clean_df = num_df.dropna()

    if len(clean_df) == 0:
        errors.append("No valid rows after removing missing values.")
        return {'errors': errors, 'warnings': warnings}

    # Categorical disguised as numeric
    for col in items:
        raw_col = df[col]
        n_unique = raw_col.nunique(dropna=True)
        n_rows = len(raw_col.dropna())
        is_low_card = n_unique <= 10 and n_rows >= 20 and (n_unique / n_rows) <= 0.05
        is_str = pd.api.types.is_object_dtype(raw_col) or pd.api.types.is_string_dtype(raw_col)
        if is_low_card or is_str:
            warnings.append(
                f"'{col}' ({n_unique} unique values{', string origin' if is_str else ''}) "
                "may be categorical. HDBSCAN assumes continuous numeric features."
            )

    n = len(clean_df)
    if n < 10:
        warnings.append(f"Very small sample (N={n}). HDBSCAN results may be unreliable.")
    elif n < 30:
        warnings.append(f"Small sample (N={n}). Consider whether HDBSCAN is appropriate.")

    if min_cluster_size > n // 2:
        warnings.append(
            f"min_cluster_size={min_cluster_size} is large relative to N={n}. "
            "Most points may be classified as noise."
        )

    # Constant / near-zero variance
    for col in items:
        s = clean_df[col].std()
        if s == 0.0:
            errors.append(f"Column '{col}' is constant (zero variance).")
        elif s < 1e-6:
            warnings.append(f"Column '{col}' has near-zero variance (std={s:.2e}).")

    # Duplicate rows
    n_dupes = clean_df.duplicated().sum()
    if n_dupes / max(len(clean_df), 1) > 0.3:
        warnings.append(
            f"{n_dupes} duplicate rows ({n_dupes/len(clean_df)*100:.1f}%). "
            "High duplication can distort density estimates."
        )

    # Extreme outliers
    z = np.abs((clean_df - clean_df.mean()) / clean_df.std(ddof=0).replace(0, 1))
    n_extreme = int((z > 5).any(axis=1).sum())
    if n_extreme > 0:
        warnings.append(
            f"{n_extreme} row(s) with |z| > 5. "
            "HDBSCAN may assign these as noise — this can be intentional."
        )

    # Metric-specific advice
    if metric == 'cosine':
        warnings.append(
            "Cosine metric measures angle, not magnitude. "
            "Two points with identical direction but different scales are treated as identical. "
            "Ensure this is intentional (e.g. text/high-dim frequency data)."
        )
    elif metric == 'manhattan':
        warnings.append(
            "Manhattan (L1) distance is less sensitive to outliers than Euclidean "
            "but may produce different cluster shapes. Good for high-dimensional data."
        )
    elif metric in ('chebyshev', 'canberra', 'braycurtis'):
        warnings.append(
            f"'{metric}' is an advanced distance metric. "
            "Ensure features are on appropriate scales."
        )

    return {'errors': errors, 'warnings': warnings}


# ─── Parameter advisor ────────────────────────────────────────────────────────

def _parameter_advice(
    n_samples: int,
    n_features: int,
    min_cluster_size: int,
    min_samples_used: int,
) -> Dict[str, Any]:
    notes: List[str] = []
    suggested_mcs = max(2, int(np.sqrt(n_samples)))
    suggested_ms = max(1, n_features)

    if min_cluster_size < max(2, n_samples // 50):
        notes.append(
            f"min_cluster_size={min_cluster_size} may be too small, "
            f"risking many micro-clusters. Consider ≥ {max(2, n_samples // 50)}."
        )
    if min_cluster_size > n_samples // 5:
        notes.append(
            f"min_cluster_size={min_cluster_size} is large relative to N={n_samples}. "
            "Many points may be classified as noise."
        )
    if min_samples_used > min_cluster_size:
        notes.append(
            f"min_samples={min_samples_used} > min_cluster_size={min_cluster_size}. "
            "This is valid but produces very conservative (noise-heavy) results."
        )

    return {
        'suggested_min_cluster_size': suggested_mcs,
        'suggested_min_samples': suggested_ms,
        'notes': notes,
        'summary': (
            f"Used min_cluster_size={min_cluster_size}, min_samples={min_samples_used}. "
            f"Rule-of-thumb suggestions: min_cluster_size≈{suggested_mcs} (√N), "
            f"min_samples≈{suggested_ms} (= n_features)."
        ),
    }


# ─── Probability / stability summary ─────────────────────────────────────────

def _summarise_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Summarise HDBSCAN membership probabilities and outlier scores per cluster.
    Probability = how strongly a point belongs to its cluster (0–1).
    Outlier score = GLOSH score; higher = more outlier-like.
    """
    summary: Dict[str, Any] = {}
    unique = np.unique(labels[labels != -1])

    for lbl in unique:
        mask = labels == lbl
        probs = probabilities[mask]
        summary[f'Cluster {lbl + 1}'] = {
            'mean_probability': round(float(np.mean(probs)), 4),
            'min_probability': round(float(np.min(probs)), 4),
            'soft_members': int((probs < 0.5).sum()),  # weakly assigned points
        }

    noise_mask = labels == -1
    return {
        'per_cluster': summary,
        'overall_mean_probability': round(
            float(np.mean(probabilities[labels != -1])) if (labels != -1).any() else 0.0, 4
        ),
        'high_outlier_count': int((outlier_scores > 0.7).sum()),
        'note': (
            "Membership probability (0–1): how strongly a point belongs to its cluster. "
            "Points with probability < 0.5 are weakly assigned. "
            "Outlier score > 0.7 indicates a strong outlier (GLOSH score)."
        ),
    }


def _cluster_persistence(clusterer) -> Dict[str, Any]:
    """
    Extract cluster stability/persistence from HDBSCAN.
    Stability measures how long a cluster persists across the hierarchy.
    Higher = more robust cluster.
    """
    try:
        labels = clusterer.labels_
        unique = np.unique(labels[labels != -1])
        if not hasattr(clusterer, 'cluster_persistence_') or len(clusterer.cluster_persistence_) == 0:
            return {'clusters': {}, 'note': 'Persistence data unavailable.'}

        persistence = clusterer.cluster_persistence_
        result: Dict[str, float] = {}
        for i, lbl in enumerate(unique):
            if i < len(persistence):
                result[f'Cluster {lbl + 1}'] = round(float(persistence[i]), 4)

        sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return {
            'clusters': dict(sorted_items),
            'most_stable': sorted_items[0][0] if sorted_items else None,
            'note': (
                "Cluster persistence (stability) measures how long a cluster survives "
                "as the density threshold is varied. Higher = more robust and reliable cluster."
            ),
        }
    except Exception as exc:
        logger.debug("Persistence extraction failed: %s", exc)
        return {'clusters': {}, 'note': 'Persistence extraction failed.'}


# ─── Quality warnings ─────────────────────────────────────────────────────────

def _quality_warnings(
    n_clusters: int,
    n_noise: int,
    n_samples: int,
    silhouette: float,
    mean_probability: float,
    high_outlier_count: int,
) -> List[str]:
    warns: List[str] = []
    noise_ratio = n_noise / max(n_samples, 1)

    if n_clusters == 0:
        warns.append(
            "No clusters found — all points are noise. "
            "Try decreasing min_cluster_size or min_samples."
        )
    elif n_clusters == 1:
        warns.append(
            "Only 1 cluster found. Consider decreasing min_cluster_size for finer granularity."
        )
    elif n_clusters > 20:
        warns.append(
            f"{n_clusters} clusters found — unusually high. "
            "Consider increasing min_cluster_size."
        )

    if noise_ratio > 0.5:
        warns.append(
            f"{n_noise} points ({noise_ratio*100:.1f}%) are noise. "
            "Over half the data is unassigned — try decreasing min_cluster_size."
        )
    elif noise_ratio > 0.3:
        warns.append(
            f"{n_noise} points ({noise_ratio*100:.1f}%) are noise. "
            "High noise ratio — consider parameter tuning."
        )

    if n_clusters > 1 and silhouette < 0.25:
        warns.append(
            f"Silhouette score is low ({silhouette:.3f}). "
            "Clusters may overlap or lack clear separation."
        )

    if mean_probability < 0.6 and n_clusters > 0:
        warns.append(
            f"Mean membership probability is low ({mean_probability:.3f}). "
            "Many points are weakly assigned to their clusters."
        )

    if high_outlier_count > 0:
        warns.append(
            f"{high_outlier_count} point(s) have outlier score > 0.7 (GLOSH). "
            "These are strong outliers worth investigating."
        )

    return warns


# ─── Feature drivers (ANOVA) ──────────────────────────────────────────────────

def _compute_feature_drivers(
    cluster_data: pd.DataFrame,
    labels: np.ndarray,
    items: List[str],
) -> Dict[str, Any]:
    try:
        non_noise_mask = labels != -1
        if non_noise_mask.sum() == 0:
            return {'features': [], 'note': 'No non-noise points for ANOVA.'}

        clean_labels = labels[non_noise_mask]
        clean_data = cluster_data.iloc[non_noise_mask]
        unique_labels = np.unique(clean_labels)

        if len(unique_labels) < 2:
            return {'features': [], 'note': 'ANOVA requires ≥ 2 clusters (excluding noise).'}

        drivers: List[Dict[str, Any]] = []
        for col in items:
            groups = [clean_data.loc[clean_labels == lbl, col].dropna().values for lbl in unique_labels]
            if any(len(g) < 2 for g in groups):
                continue
            try:
                f_stat, p_val = f_oneway(*groups)
                if not (np.isfinite(f_stat) and np.isfinite(p_val)):
                    continue
                all_vals = clean_data[col].dropna().values
                grand_mean = all_vals.mean()
                ss_total = float(np.sum((all_vals - grand_mean) ** 2))
                ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups))
                eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0
                drivers.append({
                    'feature': col,
                    'f_stat': float(f_stat),
                    'p_value': float(p_val),
                    'eta_squared': round(eta_sq, 4),
                    'effect_size': 'large' if eta_sq >= 0.14 else 'medium' if eta_sq >= 0.06 else 'small',
                    'is_significant': bool(p_val < 0.05),
                })
            except Exception as exc:
                logger.debug("ANOVA failed for '%s': %s", col, exc)

        drivers.sort(key=lambda x: x['f_stat'], reverse=True)
        for rank, d in enumerate(drivers, start=1):
            d['rank'] = rank

        return {
            'features': drivers,
            'top_driver': drivers[0]['feature'] if drivers else None,
            'note': (
                "One-way ANOVA across non-noise clusters. "
                "η² ≥ 0.14 large, ≥ 0.06 medium, < 0.06 small. "
                "Treat p-values as exploratory (no Bonferroni correction)."
            ),
        }
    except Exception as exc:
        logger.warning("Feature driver analysis failed: %s", exc)
        return {'features': [], 'note': f'Feature driver analysis failed: {exc}'}


# ─── Interpretations ──────────────────────────────────────────────────────────

def _build_interpretations(
    profiles: Dict,
    n_clusters: int,
    n_noise: int,
    n_samples: int,
    silhouette: float,
    davies_bouldin: float,
    calinski: float,
    cluster_data: pd.DataFrame,
    items: List[str],
    mean_probability: float,
) -> Dict[str, Any]:
    noise_ratio = n_noise / max(n_samples, 1)

    if n_clusters == 0:
        quality_desc = "no clusters detected — all points are noise."
    elif silhouette >= 0.7:
        quality_desc = "strong and well-defined."
    elif silhouette >= 0.5:
        quality_desc = "reasonable and distinct."
    elif silhouette >= 0.25:
        quality_desc = "weak with some overlap."
    else:
        quality_desc = "not well-defined; interpret with caution."

    overall_quality = (
        f"HDBSCAN identified <strong>{n_clusters} cluster(s)</strong> and "
        f"<strong>{n_noise} noise point(s)</strong> ({noise_ratio*100:.1f}% of data). "
        f"Mean membership probability: <strong>{mean_probability:.3f}</strong>. "
    )
    if n_clusters > 1:
        overall_quality += (
            f"Silhouette Score: <strong>{silhouette:.3f}</strong> — clustering is {quality_desc} "
            f"Calinski-Harabasz: <strong>{calinski:.2f}</strong>. "
            f"Davies-Bouldin: <strong>{davies_bouldin:.3f}</strong>."
        )
    else:
        overall_quality += "Quality metrics require ≥ 2 clusters to compute."

    overall_means = cluster_data.mean()
    overall_std = cluster_data.std().replace(0, 1)
    cluster_profiles_text: List[str] = []

    for name, profile in profiles.items():
        if profile.get('is_noise'):
            cluster_profiles_text.append(
                f"<strong>Noise ({profile['percentage']:.1f}% of data):</strong> "
                "Points that did not meet the density threshold for any cluster. "
                "May represent outliers or low-density transition regions."
            )
            continue
        centroid = pd.Series(profile['centroid'])
        deviations = (centroid - overall_means) / overall_std
        top_features = deviations.nlargest(2).index.tolist()
        bottom_features = deviations.nsmallest(2).index.tolist()
        prob_info = profile.get('mean_probability', None)
        prob_str = f" (mean membership probability: {prob_info:.3f})" if prob_info else ""
        cluster_profiles_text.append(
            f"<strong>{name} ({profile['percentage']:.1f}% of data){prob_str}:</strong> "
            f"High in <strong>{', '.join(top_features)}</strong>; "
            f"low in <strong>{', '.join(bottom_features)}</strong>."
        )

    non_noise = {k: v for k, v in profiles.items() if not v.get('is_noise')}
    if len(non_noise) > 1:
        sizes = [p['size'] for p in non_noise.values()]
        ratio = max(sizes) / max(min(sizes), 1)
        dist_desc = (
            "Non-noise clusters are relatively balanced in size."
            if ratio < 3 else
            f"Non-noise cluster sizes are imbalanced (ratio {ratio:.1f}×)."
        )
    elif len(non_noise) == 1:
        dist_desc = "A single dense cluster was found alongside noise points."
    else:
        dist_desc = "No clusters found — all points classified as noise."

    return {
        'overall_quality': overall_quality,
        'cluster_profiles': cluster_profiles_text,
        'cluster_distribution': dist_desc,
    }


# ─── Main endpoint ────────────────────────────────────────────────────────────

@router.post("/hdbscan")
def hdbscan_clustering(req: HDBSCANRequest):
    if not HDBSCAN_AVAILABLE:
        raise HTTPException(status_code=500, detail="hdbscan library not installed. Run: pip install hdbscan")

    try:
        df = pd.DataFrame(req.data)
        items = req.items
        min_cluster_size = req.min_cluster_size
        min_samples = req.min_samples
        metric = req.distanceMetric.lower()

        # Validation
        validation = _validate_input(df, items, min_cluster_size, min_samples, metric)
        if validation['errors']:
            raise HTTPException(
                status_code=400,
                detail="Validation failed: " + " | ".join(validation['errors']),
            )
        input_warnings: List[str] = validation['warnings']

        # Prepare data
        num_df = df[items].apply(pd.to_numeric, errors='coerce')
        cluster_data = num_df.dropna().copy()
        n_samples_total, n_features = cluster_data.shape

        # Large-data strategy: subsample for fitting if N > 5000
        LARGE_N_THRESHOLD = 5000
        is_large = n_samples_total > LARGE_N_THRESHOLD
        if is_large:
            subsample_size = min(5000, n_samples_total)
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(n_samples_total, size=subsample_size, replace=False)
            input_warnings.append(
                f"Large dataset (N={n_samples_total}). "
                f"HDBSCAN fitted on a random subsample of {subsample_size} points. "
                "Labels for all points are approximated via nearest-neighbour assignment."
            )

        # Scaler
        _scaler_map = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
        }
        scaler_obj = _scaler_map.get(req.scalerType, StandardScaler())
        X_scaled = scaler_obj.fit_transform(cluster_data)

        if not np.all(np.isfinite(X_scaled)):
            raise HTTPException(
                status_code=400,
                detail="Scaled data contains NaN/inf. Check for constant columns.",
            )

        # min_samples default = min_cluster_size if not provided
        min_samples_used = min_samples if min_samples is not None else min_cluster_size

        # Parameter advice
        param_advice = _parameter_advice(n_samples_total, n_features, min_cluster_size, min_samples_used)

        # Run HDBSCAN — cosine requires algorithm='generic'
        hdbscan_kwargs: Dict[str, Any] = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples_used,
            'metric': metric,
            'gen_min_span_tree': True,
            'prediction_data': True,
        }
        if metric == 'cosine':
            hdbscan_kwargs['algorithm'] = 'generic'

        clusterer = hdbscan_lib.HDBSCAN(**hdbscan_kwargs)

        if is_large:
            clusterer.fit(X_scaled[sub_idx])
            # Approximate labels for full dataset via HDBSCAN approximate_predict
            try:
                labels_full, strengths_full = hdbscan_lib.approximate_predict(clusterer, X_scaled)
                labels = labels_full
                probabilities = strengths_full
            except Exception:
                # Fallback: assign noise to out-of-sample points
                labels_sub = clusterer.labels_
                probs_sub = clusterer.probabilities_
                labels = np.full(n_samples_total, -1, dtype=int)
                probabilities = np.zeros(n_samples_total)
                labels[sub_idx] = labels_sub
                probabilities[sub_idx] = probs_sub
        else:
            labels = clusterer.fit_predict(X_scaled)
            probabilities = clusterer.probabilities_

        outlier_scores = getattr(clusterer, 'outlier_scores_', np.zeros(len(sub_idx if is_large else X_scaled)))

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        unique_labels = np.unique(labels)

        # Probability / stability summaries
        prob_summary = _summarise_probabilities(labels, probabilities, outlier_scores)
        persistence_info = _cluster_persistence(clusterer)

        # Metrics (noise-excluded)
        non_noise_mask = labels != -1
        if n_clusters > 1 and non_noise_mask.sum() > 1:
            try:
                silhouette = _safe_float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
                davies_bouldin = _safe_float(davies_bouldin_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
                calinski = _safe_float(calinski_harabasz_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
            except Exception as exc:
                logger.debug("Metrics failed: %s", exc)
                silhouette = davies_bouldin = calinski = 0.0
        else:
            silhouette = davies_bouldin = calinski = 0.0

        # Profiles
        profiles: Dict[str, Any] = {}
        for label in unique_labels:
            mask = labels == label
            subset = cluster_data.iloc[mask]
            name = 'Noise' if label == -1 else f'Cluster {label + 1}'
            prob_info = prob_summary['per_cluster'].get(name, {})
            profiles[name] = {
                'size': int(mask.sum()),
                'percentage': _safe_float(mask.sum() / n_samples_total * 100),
                'centroid': {col: _safe_float(subset[col].mean()) for col in items},
                'is_noise': bool(label == -1),
                'mean_probability': prob_info.get('mean_probability'),
                'min_probability': prob_info.get('min_probability'),
                'soft_members': prob_info.get('soft_members'),
                'persistence': persistence_info['clusters'].get(name),
            }

        # Quality warnings
        quality_warnings = _quality_warnings(
            n_clusters, n_noise, n_samples_total,
            silhouette,
            prob_summary['overall_mean_probability'],
            prob_summary['high_outlier_count'],
        )

        # Feature drivers
        feature_drivers = _compute_feature_drivers(cluster_data, labels, items)

        # Interpretations
        interpretations = _build_interpretations(
            profiles, n_clusters, n_noise, n_samples_total,
            silhouette, davies_bouldin, calinski,
            cluster_data, items,
            prob_summary['overall_mean_probability'],
        )

        pca_note = None

        # ── Plots: 2×3 grid ───────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        palette = sns.color_palette('husl', n_colors=max(n_clusters, 1))

        # 1. PCA scatter — point size ∝ membership probability (top-left)
        ax1 = axes[0, 0]
        ax1.set_facecolor('#f0f0f0')
        if n_features >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(X_scaled)
            var_explained = sum(pca.explained_variance_ratio_)
            pca_note = (
                f"PCA projection explains {var_explained:.1%} of variance. "
                f"Point size ∝ membership probability. "
                f"Boundaries shown in 2-D PCA space, not original {n_features}-D feature space."
            )

            # Noise
            noise_mask = labels == -1
            if noise_mask.any():
                ax1.scatter(pca_data[noise_mask, 0], pca_data[noise_mask, 1],
                            color='#cccccc', marker='x', s=35, alpha=0.6,
                            linewidth=1.2, label='Noise', zorder=1)

            # Clusters — size encodes probability
            color_idx = 0
            for label in sorted(np.unique(labels[labels != -1])):
                mask = labels == label
                probs = probabilities[mask]
                sizes = 30 + probs * 160  # 30–190 range
                ax1.scatter(pca_data[mask, 0], pca_data[mask, 1],
                            c=[palette[color_idx % len(palette)]] * mask.sum(),
                            s=sizes, alpha=0.75,
                            edgecolors='black', linewidth=0.6,
                            label=f'Cluster {label + 1}', zorder=2)
                color_idx += 1

            ax1.set_title(
                f'PCA Projection ({var_explained:.1%} var) — size ∝ probability',
                fontsize=10, fontweight='bold'
            )
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            ax1.annotate("⚠ PCA 2-D projection only — not original feature space.",
                         xy=(0.5, -0.13), xycoords='axes fraction',
                         ha='center', fontsize=7, color='#666666', style='italic')
            ax1.legend(loc='best', frameon=True, facecolor='white', edgecolor='gray', fontsize=8)
            ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. Membership probability distribution (top-middle)
        ax2 = axes[0, 1]
        ax2.set_facecolor('#f5f5f5')
        if (labels != -1).any():
            color_idx = 0
            for label in sorted(np.unique(labels[labels != -1])):
                mask = labels == label
                probs = probabilities[mask]
                ax2.hist(probs, bins=20, alpha=0.65,
                         color=palette[color_idx % len(palette)],
                         label=f'Cluster {label + 1}', edgecolor='white', linewidth=0.8)
                color_idx += 1
            ax2.axvline(0.5, color='#E8505B', linestyle='--', linewidth=1.5,
                        label='p=0.5 threshold')
            ax2.set_title('Membership Probability Distribution', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Membership Probability', fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.4, axis='y', linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'No cluster members to display',
                     ha='center', va='center', fontsize=11, transform=ax2.transAxes, color='#888')
            ax2.set_title('Membership Probability Distribution', fontsize=11, fontweight='bold')

        # 3. Cluster size bar (top-right)
        ax3 = axes[0, 2]
        ax3.set_facecolor('#f5f5f5')
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        bar_names = ['Noise' if i == -1 else f'Cluster {i+1}' for i in cluster_sizes.index]
        bar_colors = ['#cccccc' if i == -1 else palette[min(j, len(palette)-1)]
                      for j, i in enumerate(cluster_sizes.index)]
        bars = ax3.bar(bar_names, cluster_sizes.values, color=bar_colors,
                       edgecolor='white', linewidth=1.5, alpha=0.9)
        for bar, val in zip(bars, cluster_sizes.values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax3.set_title('Cluster Size Distribution', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Cluster', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.tick_params(axis='x', rotation=30)
        ax3.grid(True, alpha=0.4, axis='y', linestyle='--')

        # 4. Snake plot (bottom-left)
        ax4 = axes[1, 0]
        ax4.set_facecolor('#f5f5f5')
        centroids_data = {k: v['centroid'] for k, v in profiles.items() if not v['is_noise']}
        if centroids_data and len(items) >= 2:
            centroids_df = pd.DataFrame(centroids_data).T
            overall_m = cluster_data.mean()
            overall_s = cluster_data.std().replace(0, 1)
            scaled = (centroids_df - overall_m) / overall_s
            for i, (name, row) in enumerate(scaled.iterrows()):
                ax4.plot(range(len(items)), row.values, 'o-',
                         label=name, color=palette[i % len(palette)], linewidth=2, markersize=7)
            ax4.set_xticks(range(len(items)))
            ax4.set_xticklabels(items, rotation=40, ha='right', fontsize=9)
            ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_title('Snake Plot (Cluster Centroids)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Standardized Value (Z-score)', fontsize=10)
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.4, linestyle='--')
        else:
            ax4.text(0.5, 0.5, 'No clusters to display', ha='center', va='center',
                     fontsize=12, transform=ax4.transAxes, color='#888')
            ax4.set_title('Snake Plot (Cluster Centroids)', fontsize=11, fontweight='bold')

        # 5. Condensed tree (bottom-middle)
        ax5 = axes[1, 1]
        ax5.set_facecolor('#f5f5f5')
        condensed_tree_plotted = False
        try:
            if not is_large and hasattr(clusterer, 'condensed_tree_'):
                ct = clusterer.condensed_tree_
                ct_df = ct.to_pandas()
                # Select cluster nodes only (child_size > 1 = branch nodes)
                branch_df = ct_df[ct_df['child_size'] > 1].copy()
                leaf_df = ct_df[ct_df['child_size'] == 1].copy()

                if len(branch_df) > 0 or len(leaf_df) > 0:
                    # Plot lambda (1/distance) bars representing cluster lifespan
                    all_df = ct_df.copy()
                    parents = all_df['parent'].unique()
                    max_lambda = all_df['lambda_val'].replace([np.inf], np.nan).dropna().max()
                    if max_lambda and np.isfinite(max_lambda):
                        # Draw each cluster span as a horizontal bar
                        cluster_ids = sorted(all_df['parent'].unique())[:30]  # cap at 30 for clarity
                        y_pos = 0
                        for cid in cluster_ids:
                            rows = all_df[all_df['parent'] == cid]
                            lmin = rows['lambda_val'].min()
                            lmax = rows['lambda_val'].replace([np.inf], np.nan).dropna().max()
                            if pd.isna(lmax):
                                lmax = max_lambda
                            size = int(rows['child_size'].sum())
                            color_idx_c = int(cid) % len(palette)
                            ax5.barh(y_pos, lmax - lmin, left=lmin,
                                     height=max(0.3, np.log1p(size) * 0.15),
                                     color=palette[color_idx_c], alpha=0.75, edgecolor='white')
                            y_pos += 0.5
                        ax5.set_xlabel('λ (1 / distance scale)', fontsize=9)
                        ax5.set_ylabel('Cluster nodes', fontsize=9)
                        ax5.set_title('Condensed Tree (Cluster Lifespans)', fontsize=11, fontweight='bold')
                        ax5.grid(True, alpha=0.3, axis='x', linestyle='--')
                        ax5.annotate(
                            "Wider bars = longer-lived (more stable) clusters.",
                            xy=(0.5, -0.12), xycoords='axes fraction',
                            ha='center', fontsize=7, color='#555', style='italic'
                        )
                        condensed_tree_plotted = True
        except Exception as tree_exc:
            logger.debug("Condensed tree plot failed: %s", tree_exc)

        if not condensed_tree_plotted:
            ax5.text(0.5, 0.5,
                     'Condensed tree\nnot available\n(large-data mode or\nno hierarchy)',
                     ha='center', va='center', fontsize=10,
                     transform=ax5.transAxes, color='#888')
            ax5.set_title('Condensed Tree', fontsize=11, fontweight='bold')

        # 6. Outlier (GLOSH) score distribution (bottom-right)
        ax6 = axes[1, 2]
        ax6.set_facecolor('#f5f5f5')
        try:
            scores = clusterer.outlier_scores_
            if scores is not None and len(scores) > 0 and np.any(np.isfinite(scores)):
                finite_scores = scores[np.isfinite(scores)]
                ax6.hist(finite_scores, bins=30, color='#5B7FDB', alpha=0.8,
                         edgecolor='white', linewidth=0.8)
                ax6.axvline(0.7, color='#E8505B', linestyle='--', linewidth=1.5,
                            label='Outlier threshold (0.7)')
                n_outliers = int((finite_scores > 0.7).sum())
                ax6.set_title(
                    f'GLOSH Outlier Score Distribution ({n_outliers} outliers > 0.7)',
                    fontsize=10, fontweight='bold'
                )
                ax6.set_xlabel('Outlier Score', fontsize=10)
                ax6.set_ylabel('Count', fontsize=10)
                ax6.legend(fontsize=8)
                ax6.grid(True, alpha=0.4, axis='y', linestyle='--')
            else:
                raise ValueError("No finite outlier scores")
        except Exception:
            ax6.text(0.5, 0.5, 'Outlier scores\nnot available',
                     ha='center', va='center', fontsize=11,
                     transform=ax6.transAxes, color='#888')
            ax6.set_title('GLOSH Outlier Score Distribution', fontsize=11, fontweight='bold')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

        return _to_native({
            'results': {
                'clustering_summary': {
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'n_samples': n_samples_total,
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples_used,
                    'scaler': req.scalerType,
                    'metric': metric,
                    'is_large_data': is_large,
                    'labels': labels.tolist(),
                },
                'param_advice': param_advice,
                'probability_summary': prob_summary,
                'cluster_persistence': persistence_info,
                'profiles': profiles,
                'feature_drivers': feature_drivers,
                'final_metrics': {
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin,
                    'calinski_harabasz': calinski,
                    'note': 'Metrics computed on non-noise points only. Require ≥ 2 clusters.',
                },
                'interpretations': interpretations,
                'warnings': {
                    'input': input_warnings,
                    'quality': quality_warnings,
                    'pca_note': pca_note,
                },
            },
            'plot': plot,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in hdbscan_clustering")
        raise HTTPException(status_code=500, detail=f"HDBSCAN analysis failed: {exc}")
