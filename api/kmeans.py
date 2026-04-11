from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import io, base64, logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")

router = APIRouter()


# ─── Request model ────────────────────────────────────────────────────────────

class KMeansRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(...)
    items: List[str] = Field(...)
    nClusters: int = Field(default=3, ge=2, le=50)
    boundaryType: Optional[str] = Field(default='hull', pattern='^(hull|ellipse)$')
    useMiniBatch: bool = Field(default=False)
    scalerType: str = Field(default='standard', pattern='^(standard|robust|minmax)$')
    # standard : StandardScaler — zero mean, unit variance. Default, works well for
    #            most distributions.
    # robust   : RobustScaler  — uses median/IQR. Resistant to outliers.
    # minmax   : MinMaxScaler  — scales to [0, 1]. Useful when all features have
    #            natural bounds and outliers are already removed.


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def _safe_float(val, default=0.0) -> float:
    """Convert to float, returning default only for None/NaN/inf — logs other failures."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError) as exc:
        logger.debug("_safe_float conversion failed for %r: %s", val, exc)
        return default


# ─── Fix 1: Comprehensive validation ─────────────────────────────────────────

def _validate_input(
    df: pd.DataFrame,
    items: List[str],
    n_clusters: int,
) -> Dict[str, List[str]]:
    """
    Run all pre-fit checks.  Returns {'errors': [...], 'warnings': [...]}.
    Errors are fatal (raise HTTPException in caller).
    Warnings are surfaced to the user but do not block execution.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ── nClusters basic bound (pydantic covers ge=2, but guard defensively) ──
    if n_clusters < 2:
        errors.append(f"nClusters must be ≥ 2, got {n_clusters}.")

    # ── Column existence ──────────────────────────────────────────────────────
    missing = [c for c in items if c not in df.columns]
    if missing:
        errors.append(f"Column(s) not found in data: {', '.join(missing)}.")

    # Stop here if columns are missing — further checks would fail
    if errors:
        return {'errors': errors, 'warnings': warnings}

    # ── Numeric check ─────────────────────────────────────────────────────────
    non_numeric = []
    for col in items:
        converted = pd.to_numeric(df[col], errors='coerce')
        n_invalid = converted.isna().sum() - df[col].isna().sum()
        if n_invalid > 0:
            non_numeric.append(f"'{col}' has {n_invalid} non-numeric value(s)")
    if non_numeric:
        errors.append("Non-numeric data detected: " + "; ".join(non_numeric) + ". Only numeric columns are supported.")

    if errors:
        return {'errors': errors, 'warnings': warnings}

    # Work on numeric subset from here
    num_df = df[items].apply(pd.to_numeric, errors='coerce')
    clean_df = num_df.dropna()

    # ── Fix 2: Categorical-disguised-as-numeric detection ─────────────────────
    # Heuristics for columns that look numeric but behave like categories:
    #   (a) integer-valued column with very few unique values (≤ 10 and ≤ 5% of N)
    #   (b) string dtype that survived pd.to_numeric (e.g. "1", "2", "3" labels)
    # These won't cause a crash but will produce meaningless cluster geometry.
    categorical_suspects: List[str] = []
    for col in items:
        raw_col = df[col]
        n_unique = raw_col.nunique(dropna=True)
        n_rows = len(raw_col.dropna())

        # Heuristic (a): very low cardinality relative to sample size
        is_low_cardinality = (
            n_unique <= 10 and n_rows >= 20 and (n_unique / n_rows) <= 0.05
        )
        # Heuristic (b): originally string/object dtype (ID-like codes encoded as numbers)
        is_string_origin = pd.api.types.is_object_dtype(raw_col) or pd.api.types.is_string_dtype(raw_col)

        if is_low_cardinality or is_string_origin:
            categorical_suspects.append(
                f"'{col}' ({n_unique} unique values"
                + (", string origin" if is_string_origin else "")
                + ")"
            )

    if categorical_suspects:
        warnings.append(
            "Possible categorical-disguised-as-numeric column(s): "
            + ", ".join(categorical_suspects)
            + ". K-means assumes continuous numeric features. "
            "If these are category codes or IDs, remove them from the feature list."
        )

    # ── Sample size ───────────────────────────────────────────────────────────
    n = len(clean_df)
    if n < n_clusters * 2:
        errors.append(
            f"Need at least {n_clusters * 2} complete observations for {n_clusters} clusters, "
            f"but only {n} found after removing rows with missing values."
        )
    elif n < 30:
        warnings.append(
            f"Small sample (N={n}). K-means results may be unstable. "
            "Interpret cluster assignments with caution."
        )

    if errors:
        return {'errors': errors, 'warnings': warnings}

    # ── Constant / near-zero variance ─────────────────────────────────────────
    for col in items:
        s = clean_df[col].std()
        if s == 0.0:
            errors.append(f"Column '{col}' is constant (zero variance). K-means cannot use it.")
        elif s < 1e-6:
            warnings.append(
                f"Column '{col}' has near-zero variance (std={s:.2e}). "
                "It will contribute almost nothing after standardization."
            )

    # ── Duplicate rows ────────────────────────────────────────────────────────
    n_dupes = clean_df.duplicated().sum()
    if n_dupes / max(len(clean_df), 1) > 0.3:
        warnings.append(
            f"{n_dupes} duplicate rows detected ({n_dupes / len(clean_df) * 100:.1f}%). "
            "High duplication can distort cluster centroids."
        )

    # ── Extreme outliers (z-score > 5 on any column) ─────────────────────────
    z = np.abs((clean_df - clean_df.mean()) / clean_df.std(ddof=0).replace(0, 1))
    n_extreme = int((z > 5).any(axis=1).sum())
    if n_extreme > 0:
        warnings.append(
            f"{n_extreme} row(s) have at least one feature with |z| > 5. "
            "Extreme outliers may pull centroids and distort clustering."
        )

    return {'errors': errors, 'warnings': warnings}


# ─── Fix 2: Consensus optimal-k selection ────────────────────────────────────

def _recommend_k(
    k_range: List[int],
    inertias: List[float],
    silhouette_scores: List[float],
    ch_scores: List[float],
) -> Dict[str, Any]:
    """
    Combine three criteria to recommend optimal k:
      1. Silhouette: highest score
      2. Calinski-Harabasz: highest score
      3. Elbow (inertia): maximum second-derivative (knee)

    Returns the k with the most votes, with a breakdown dict for transparency.
    Minimum-cluster-size constraint is applied upstream.
    """
    n = len(k_range)

    # Silhouette vote
    sil_k = k_range[int(np.argmax(silhouette_scores))] if any(s > -1 for s in silhouette_scores) else k_range[0]

    # CH vote
    ch_k = k_range[int(np.argmax(ch_scores))] if ch_scores else k_range[0]

    # Elbow vote: second derivative of inertia (largest curvature = knee)
    elbow_k = k_range[0]
    if len(inertias) >= 3:
        arr = np.array(inertias)
        d2 = np.diff(arr, 2)           # second derivative
        elbow_k = k_range[int(np.argmax(d2)) + 1]   # +1 offset from double diff

    # Vote tally
    votes: Dict[int, int] = {}
    for k in [sil_k, ch_k, elbow_k]:
        votes[k] = votes.get(k, 0) + 1

    # Break ties: prefer middle k (less extremes)
    winner = max(votes, key=lambda k: (votes[k], -abs(k - np.median(k_range))))

    return {
        'recommended_k': winner,
        'votes': {
            'silhouette_k': sil_k,
            'calinski_harabasz_k': ch_k,
            'elbow_k': elbow_k,
            'tally': votes,
        },
        'note': (
            f"Recommended k={winner} based on consensus of silhouette (k={sil_k}), "
            f"Calinski-Harabasz (k={ch_k}), and elbow method (k={elbow_k})."
        )
    }


# ─── Fix 3: PCA visualisation disclaimer helper ───────────────────────────────

_PCA_DISCLAIMER = (
    "Note: cluster boundaries are projected onto 2-D PCA space for visualisation only. "
    "They do NOT represent the true cluster boundaries in the original {n_features}-dimensional feature space."
)


# ─── Fix 4: Narrow exception handlers ─────────────────────────────────────────

def _draw_convex_hull(ax, points: np.ndarray, color, alpha: float = 0.15) -> None:
    """Draw convex hull; logs and skips on failure instead of silently passing."""
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
        hull_pts = np.append(hull.vertices, hull.vertices[0])
        ax.fill(points[hull_pts, 0], points[hull_pts, 1],
                color=color, alpha=alpha, edgecolor=color, linewidth=2, linestyle='-')
    except Exception as exc:
        logger.debug("ConvexHull failed (likely collinear points): %s", exc)


def _draw_confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, color, n_std: float = 2.0, alpha: float = 0.2) -> None:
    """Draw confidence ellipse; logs and skips on failure."""
    if len(x) < 3:
        return
    try:
        cov = np.cov(x, y)
        if not np.all(np.isfinite(cov)):
            logger.debug("Ellipse skipped: covariance matrix contains non-finite values")
            return
        denom = cov[0, 0] * cov[1, 1]
        pearson = cov[0, 1] / np.sqrt(denom) if denom > 0 else 0.0
        ell_radius_x = np.sqrt(max(1 + pearson, 0))
        ell_radius_y = np.sqrt(max(1 - pearson, 0))
        ellipse = Ellipse(
            (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
            facecolor=color, alpha=alpha, edgecolor=color, linewidth=2, linestyle='--',
        )
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        transf = (
            matplotlib.transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(float(np.mean(x)), float(np.mean(y)))
        )
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
    except Exception as exc:
        logger.debug("Ellipse drawing failed: %s", exc)


# ─── Fix 6: Quality warning generator ────────────────────────────────────────

def _quality_warnings(
    silhouette: float,
    profiles: Dict,
    n_clusters: int,
    recommended_k: int,
    n_samples: int,
) -> List[str]:
    """Emit explicit operational warnings about clustering quality."""
    warns: List[str] = []

    if silhouette < 0.25:
        warns.append(
            f"Silhouette score is very low ({silhouette:.3f} < 0.25). "
            "Clusters are not well-separated. Consider a different k or feature set."
        )
    elif silhouette < 0.50:
        warns.append(
            f"Silhouette score is weak ({silhouette:.3f}). "
            "Some cluster overlap is likely."
        )

    sizes = [p['size'] for p in profiles.values()]
    min_size = min(sizes)
    min_fraction = min_size / n_samples
    if min_fraction < 0.05:
        warns.append(
            f"Smallest cluster has only {min_size} member(s) ({min_fraction*100:.1f}% of data). "
            "Very small clusters may represent outliers rather than meaningful groups."
        )

    if abs(n_clusters - recommended_k) >= 2:
        warns.append(
            f"Selected k={n_clusters} differs substantially from recommended k={recommended_k}. "
            "Consider re-running with the recommended value."
        )

    return warns


# ─── Fix 3: ANOVA-based cluster driver analysis ──────────────────────────────

def _compute_feature_drivers(
    cluster_data: pd.DataFrame,
    labels: np.ndarray,
    items: List[str],
) -> Dict[str, Any]:
    """
    For each feature, run a one-way ANOVA across cluster groups to quantify
    how much that feature discriminates between clusters.

    Returns a list of features sorted by F-statistic (descending), each with:
      - f_stat   : ANOVA F-statistic  (higher = stronger driver)
      - p_value  : significance of between-cluster variance
      - eta_sq   : η² effect size = SS_between / SS_total  (0–1; ≥ 0.14 = large)
      - rank     : 1-based importance rank
      - is_significant : p < 0.05

    Uses scipy.stats.f_oneway; returns empty list with an error note if it fails.
    """
    drivers: List[Dict[str, Any]] = []

    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {'features': [], 'note': 'ANOVA requires ≥ 2 clusters.'}

        for col in items:
            groups = [cluster_data.loc[labels == lbl, col].dropna().values
                      for lbl in unique_labels]
            # Skip degenerate groups
            if any(len(g) < 2 for g in groups):
                continue
            try:
                f_stat, p_val = f_oneway(*groups)
                if not (np.isfinite(f_stat) and np.isfinite(p_val)):
                    continue

                # η² (eta-squared) = SS_between / SS_total
                all_vals = cluster_data[col].dropna().values
                grand_mean = all_vals.mean()
                ss_total = float(np.sum((all_vals - grand_mean) ** 2))
                ss_between = float(sum(
                    len(g) * (g.mean() - grand_mean) ** 2 for g in groups
                ))
                eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0

                drivers.append({
                    'feature': col,
                    'f_stat': float(f_stat),
                    'p_value': float(p_val),
                    'eta_squared': round(eta_sq, 4),
                    'effect_size': (
                        'large'  if eta_sq >= 0.14 else
                        'medium' if eta_sq >= 0.06 else
                        'small'
                    ),
                    'is_significant': bool(p_val < 0.05),
                })
            except Exception as exc:
                logger.debug("ANOVA failed for feature '%s': %s", col, exc)

        # Sort by F-statistic descending and assign rank
        drivers.sort(key=lambda x: x['f_stat'], reverse=True)
        for rank, d in enumerate(drivers, start=1):
            d['rank'] = rank

        return {
            'features': drivers,
            'top_driver': drivers[0]['feature'] if drivers else None,
            'note': (
                "F-statistic from one-way ANOVA across clusters. "
                "η² (eta-squared) is the effect size: ≥0.14 large, ≥0.06 medium, <0.06 small. "
                "Bonferroni correction is NOT applied; treat p-values as exploratory."
            ),
        }

    except Exception as exc:
        logger.warning("Feature driver analysis failed: %s", exc)
        return {'features': [], 'note': f'Feature driver analysis failed: {exc}'}


# ─── Main endpoint ────────────────────────────────────────────────────────────

@router.post("/kmeans")
def kmeans_clustering(req: KMeansRequest):
    try:
        df = pd.DataFrame(req.data)
        items = req.items
        n_clusters = req.nClusters
        boundary_type = req.boundaryType or 'hull'
        use_minibatch = req.useMiniBatch

        # ── Fix 1: Comprehensive validation ──────────────────────────────────
        validation = _validate_input(df, items, n_clusters)
        if validation['errors']:
            raise HTTPException(
                status_code=400,
                detail="Validation failed: " + " | ".join(validation['errors']),
            )
        input_warnings: List[str] = validation['warnings']

        # Prepare numeric data
        num_df = df[items].apply(pd.to_numeric, errors='coerce')
        cluster_data = num_df.dropna().copy()
        n_samples, n_features = cluster_data.shape

        # ── Fix 1: Configurable scaler ────────────────────────────────────────
        _scaler_map = {
            'standard': StandardScaler(),
            'robust':   RobustScaler(),
            'minmax':   MinMaxScaler(),
        }
        scaler = _scaler_map.get(req.scalerType, StandardScaler())
        cluster_data_scaled = scaler.fit_transform(cluster_data)

        # ── Fix 1: post-scaling NaN/inf check ────────────────────────────────
        if not np.all(np.isfinite(cluster_data_scaled)):
            raise HTTPException(
                status_code=400,
                detail="Standardized data contains NaN or inf values. "
                       "Check for constant or near-constant columns.",
            )

        # ── Fix 2: Optimal k search with consensus criterion ─────────────────
        # Adaptive upper bound:
        #   small data  (N ≤ 200) : cap at 10 — elbow/silhouette plots stay readable
        #   large data  (N > 200) : use sqrt(N)/2, capped at 20 — avoids over-searching
        #                           while reflecting that larger datasets can support more clusters
        sqrt_bound = max(2, int(np.sqrt(n_samples) / 2))
        max_k = min(sqrt_bound if n_samples > 200 else 10, n_samples - 1, 20)
        k_range = list(range(2, max_k + 1))
        inertias: List[float] = []
        silhouette_scores: List[float] = []
        ch_scores: List[float] = []

        for k in k_range:
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            km.fit(cluster_data_scaled)
            inertias.append(_safe_float(km.inertia_))
            lbl = km.labels_
            if len(np.unique(lbl)) > 1:
                silhouette_scores.append(_safe_float(silhouette_score(cluster_data_scaled, lbl)))
                ch_scores.append(_safe_float(calinski_harabasz_score(cluster_data_scaled, lbl)))
            else:
                silhouette_scores.append(-1.0)
                ch_scores.append(0.0)

        optimal_k_info = _recommend_k(k_range, inertias, silhouette_scores, ch_scores)
        recommended_k: int = optimal_k_info['recommended_k']

        # ── Fix 5: MiniBatch option for large data ────────────────────────────
        # Heuristic: auto-switch to MiniBatchKMeans when N > 10 000 unless
        # user explicitly set useMiniBatch=False.
        _use_mb = use_minibatch or n_samples > 10_000
        KMeansClass = MiniBatchKMeans if _use_mb else KMeans
        fit_kwargs = dict(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        if _use_mb:
            fit_kwargs['batch_size'] = min(1024, n_samples)
            if n_samples > 10_000 and not use_minibatch:
                input_warnings.append(
                    f"Large dataset (N={n_samples}): automatically using MiniBatchKMeans for performance. "
                    "Results may differ slightly from standard KMeans."
                )

        kmeans = KMeansClass(**fit_kwargs)
        labels = kmeans.fit_predict(cluster_data_scaled)

        # ── Metrics ───────────────────────────────────────────────────────────
        unique_labels, counts = np.unique(labels, return_counts=True)

        if len(unique_labels) > 1:
            silhouette = _safe_float(silhouette_score(cluster_data_scaled, labels))
            davies_bouldin = _safe_float(davies_bouldin_score(cluster_data_scaled, labels))
            calinski = _safe_float(calinski_harabasz_score(cluster_data_scaled, labels))
        else:
            silhouette = davies_bouldin = calinski = 0.0

        # ── Profiles ──────────────────────────────────────────────────────────
        profiles: Dict[str, Any] = {}
        for i, label in enumerate(unique_labels):
            mask = labels == label
            subset = cluster_data.iloc[mask]
            profiles[f'Cluster {label + 1}'] = {
                'size': int(counts[i]),
                'percentage': _safe_float(counts[i] / n_samples * 100),
                'centroid': {col: _safe_float(subset[col].mean()) for col in items},
            }

        # ── Fix 6: Quality warnings ───────────────────────────────────────────
        quality_warnings = _quality_warnings(silhouette, profiles, n_clusters, recommended_k, n_samples)

        # ── Interpretations ───────────────────────────────────────────────────
        overall_means = cluster_data.mean()
        overall_std = cluster_data.std().replace(0, 1)

        cluster_profiles_text: List[str] = []
        for name, profile in profiles.items():
            centroid = pd.Series(profile['centroid'])
            deviations = (centroid - overall_means) / overall_std
            top_features = deviations.nlargest(2).index.tolist()
            bottom_features = deviations.nsmallest(2).index.tolist()
            cluster_profiles_text.append(
                f"<strong>{name} ({profile['percentage']:.1f}% of data):</strong> "
                f"High in <strong>{', '.join(top_features)}</strong>; "
                f"low in <strong>{', '.join(bottom_features)}</strong>."
            )

        if silhouette >= 0.7:
            quality_desc = "strong and well-defined."
        elif silhouette >= 0.5:
            quality_desc = "reasonable and distinct."
        elif silhouette >= 0.25:
            quality_desc = "weak and may have some overlap."
        else:
            quality_desc = "not well-defined; results should be interpreted with caution."

        overall_quality = (
            f"The <strong>Silhouette Score of {silhouette:.3f}</strong> indicates the clustering structure is {quality_desc} "
            f"<strong>Calinski-Harabasz Score: {calinski:.2f}</strong> (higher is better). "
            f"<strong>Davies-Bouldin Score: {davies_bouldin:.3f}</strong> (lower is better). "
            f"<strong>Inertia (WCSS): {_safe_float(kmeans.inertia_):.2f}</strong>."
        )

        percentages = [p['percentage'] for p in profiles.values()]
        if len(percentages) > 1 and min(percentages) > 0:
            ratio = max(percentages) / min(percentages)
            dist_desc = (
                "The cluster sizes are imbalanced (largest/smallest ratio > 3)."
                if ratio > 3
                else "The clusters are relatively balanced in size."
            )
        else:
            dist_desc = "Single or degenerate cluster detected."

        # ── Plots ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        colors = sns.color_palette('husl', n_colors=n_clusters)

        # 1. Elbow
        ax1 = axes[0, 0]
        ax1.set_facecolor('#f5f5f5')
        ax1.plot(k_range, inertias, 'o-', color='#2E86AB', linewidth=2.5, markersize=8,
                 markerfacecolor='white', markeredgewidth=2)
        if optimal_k_info['votes']['elbow_k'] in k_range:
            ek = optimal_k_info['votes']['elbow_k']
            ax1.axvline(ek, color='#E8505B', linestyle='--', alpha=0.7, label=f'Elbow k={ek}')
            ax1.legend(fontsize=9)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Inertia (WCSS)', fontsize=11, fontweight='bold')
        ax1.set_title('Elbow Method for Optimal k', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.4, linestyle='--')

        # 2. Silhouette
        ax2 = axes[0, 1]
        ax2.set_facecolor('#f5f5f5')
        bar_colors = ['#E8505B' if s == max(silhouette_scores) else '#5B9BD5' for s in silhouette_scores]
        ax2.bar(range(len(k_range)), silhouette_scores, color=bar_colors, alpha=0.85,
                edgecolor='white', linewidth=1.5)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        ax2.set_title('Silhouette Scores by k', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(k_range)))
        ax2.set_xticklabels(k_range)
        ax2.grid(True, alpha=0.4, axis='y', linestyle='--')

        # 3. Scatter (PCA)
        ax3 = axes[1, 0]
        ax3.set_facecolor('#f0f0f0')

        pca_disclaimer = ""
        if n_features >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(cluster_data_scaled)
            var_explained = sum(pca.explained_variance_ratio_)

            for i, label in enumerate(unique_labels):
                mask = labels == label
                pts = pca_data[mask]
                if boundary_type == 'hull':
                    _draw_convex_hull(ax3, pts, colors[i], alpha=0.2)
                else:
                    _draw_confidence_ellipse(ax3, pts[:, 0], pts[:, 1], colors[i])

            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax3.scatter(pca_data[mask, 0], pca_data[mask, 1],
                            c=[colors[i]], label=f'Cluster {label + 1}',
                            alpha=0.7, s=60, edgecolors='white', linewidth=0.8)

            centroids_pca = pca.transform(kmeans.cluster_centers_)
            ax3.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                        s=120, c='white', marker='o', edgecolors='black', linewidth=1.5, zorder=10)
            ax3.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                        s=40, c='black', marker='x', linewidth=1, zorder=11)

            # Fix 3: PCA disclaimer as subtitle
            pca_disclaimer = _PCA_DISCLAIMER.format(n_features=n_features)
            ax3.set_title(
                f'Clusters (PCA projection — {var_explained:.1%} variance explained)',
                fontsize=11, fontweight='bold',
            )
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)

            # Fix 3: add disclaimer as figure text below the axis
            ax3.annotate(
                "⚠ Boundaries shown in PCA 2-D space, not original feature space.",
                xy=(0.5, -0.13), xycoords='axes fraction',
                ha='center', fontsize=7, color='#666666', style='italic',
            )

            handles, _ = ax3.get_legend_handles_labels()
            centroid_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                                     markeredgecolor='black', markersize=8, label='Centroid', linewidth=0)
            handles.append(centroid_marker)
            ax3.legend(handles=handles, loc='best', frameon=True, facecolor='white', edgecolor='gray')
            ax3.grid(True, alpha=0.3, linestyle='--')

        # 4. Donut
        ax4 = axes[1, 1]
        ax4.set_facecolor('white')
        sizes_pie = [p['size'] for p in profiles.values()]
        labels_pie = list(profiles.keys())
        wedges, texts, autotexts = ax4.pie(
            sizes_pie, labels=labels_pie, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75, explode=[0.02] * len(sizes_pie),
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
        ax4.text(0, 0, f'n={sum(sizes_pie)}', ha='center', va='center', fontsize=14, fontweight='bold')
        ax4.set_title('Cluster Distribution', fontsize=13, fontweight='bold')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

        # ── Fix 3: Feature driver analysis ───────────────────────────────────
        feature_drivers = _compute_feature_drivers(cluster_data, labels, items)

        return _to_native({
            'results': {
                'optimal_k': {
                    'k_range': k_range,
                    'inertias': inertias,
                    'silhouette_scores': silhouette_scores,
                    'ch_scores': ch_scores,
                    'recommended_k': recommended_k,
                    'recommendation_detail': optimal_k_info,
                },
                'clustering_summary': {
                    'n_clusters': n_clusters,
                    'inertia': _safe_float(kmeans.inertia_),
                    'centroids': kmeans.cluster_centers_.tolist(),
                    'labels': labels.tolist(),
                    'algorithm': 'MiniBatchKMeans' if _use_mb else 'KMeans',
                    'scaler': req.scalerType,
                },
                'profiles': profiles,
                'feature_drivers': feature_drivers,         # Fix 3
                'final_metrics': {
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin,
                    'calinski_harabasz': calinski,
                },
                'interpretations': {
                    'overall_quality': overall_quality,
                    'cluster_profiles': cluster_profiles_text,
                    'cluster_distribution': dist_desc,
                },
                'warnings': {
                    'input': input_warnings,
                    'quality': quality_warnings,
                    'pca_note': pca_disclaimer if n_features >= 2 else None,
                },
            },
            'plot': plot,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in kmeans_clustering")
        raise HTTPException(status_code=500, detail=f"K-means analysis failed: {exc}")
