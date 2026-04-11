from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import f_oneway
import io, base64, logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Request model ────────────────────────────────────────────────────────────

class DBSCANRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(...)
    items: List[str] = Field(...)
    eps: Optional[float] = Field(default=0.5, gt=0)
    min_samples: Optional[int] = Field(default=5, ge=1)
    scalerType: str = Field(default='standard', pattern='^(standard|robust|minmax)$')
    distanceMetric: str = Field(default='euclidean', pattern='^(euclidean|manhattan|cosine)$')
    # euclidean : standard L2. Default for most continuous data.
    # manhattan : L1 / city-block. Better for high-dim or ordinal-heavy data.
    # cosine    : angle-based. Useful when direction matters more than magnitude.


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

def _validate_input(df: pd.DataFrame, items: List[str], eps: float, min_samples: int) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if eps <= 0:
        errors.append(f"eps must be > 0, got {eps}.")
    if min_samples < 1:
        errors.append(f"min_samples must be ≥ 1, got {min_samples}.")

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
                "may be categorical. DBSCAN assumes continuous numeric features."
            )

    n = len(clean_df)
    if n < 10:
        warnings.append(f"Very small sample (N={n}). DBSCAN results may be unreliable.")
    elif n < 30:
        warnings.append(f"Small sample (N={n}). Consider whether DBSCAN is appropriate.")

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
            f"{n_dupes} duplicate rows detected ({n_dupes/len(clean_df)*100:.1f}%). "
            "Duplicates may form artificial dense clusters."
        )

    # Outliers
    z = np.abs((clean_df - clean_df.mean()) / clean_df.std(ddof=0).replace(0, 1))
    n_extreme = int((z > 5).any(axis=1).sum())
    if n_extreme > 0:
        warnings.append(
            f"{n_extreme} row(s) with |z| > 5 detected. "
            "DBSCAN may classify these as noise — this can be intentional."
        )

    return {'errors': errors, 'warnings': warnings}


# ─── eps advisor: k-distance elbow heuristic ─────────────────────────────────

def _suggest_eps(X_scaled: np.ndarray, min_samples: int, metric: str = 'euclidean') -> Dict[str, Any]:
    """
    Compute k-distance (k = min_samples) sorted ascending.
    The 'elbow' of this curve is a principled eps suggestion.
    Returns the sorted distances and suggested eps.
    Note: for cosine metric, distances are in [0, 2] (not [0, ∞]),
    so the elbow may appear at a very different scale than euclidean/manhattan.
    """
    try:
        k = min(min_samples, len(X_scaled) - 1)
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])

        # Elbow: point of maximum curvature via second derivative
        if len(k_distances) >= 5:
            d2 = np.diff(k_distances, 2)
            elbow_idx = int(np.argmax(d2)) + 1
            suggested_eps = float(k_distances[elbow_idx])
        else:
            suggested_eps = float(np.percentile(k_distances, 90))

        metric_note = ""
        if metric == 'cosine':
            metric_note = " Note: cosine distances are in [0, 2] — typical elbow values are much smaller than euclidean."
        elif metric == 'manhattan':
            metric_note = " Note: manhattan distances scale with feature dimensionality — elbow values will be larger than euclidean."

        return {
            'k_distances': k_distances.tolist(),
            'suggested_eps': round(suggested_eps, 4),
            'note': (
                f"k-distance plot uses k={k} (= min_samples), metric={metric}. "
                f"Elbow heuristic suggests eps ≈ {suggested_eps:.4f}.{metric_note} "
                "If too many noise points, increase eps toward this value."
            )
        }
    except Exception as exc:
        logger.debug("eps suggestion failed: %s", exc)
        return {'k_distances': [], 'suggested_eps': None, 'note': 'eps suggestion unavailable.'}


# ─── Quality warnings ─────────────────────────────────────────────────────────

def _quality_warnings(
    n_clusters: int,
    n_noise: int,
    n_samples: int,
    silhouette: float,
    eps: float,
    suggested_eps: Optional[float],
) -> List[str]:
    warns: List[str] = []

    noise_ratio = n_noise / max(n_samples, 1)

    if n_clusters == 0:
        warns.append(
            "No clusters found — all points classified as noise. "
            "Try increasing eps or decreasing min_samples."
        )
    elif n_clusters == 1:
        warns.append(
            "Only 1 cluster found. This may indicate eps is too large (merging everything) "
            "or the data has no meaningful density structure."
        )
    elif n_clusters > 15:
        warns.append(
            f"{n_clusters} clusters found. This is unusually high — "
            "consider increasing eps or min_samples."
        )

    if noise_ratio > 0.5:
        warns.append(
            f"{n_noise} points ({noise_ratio*100:.1f}%) classified as noise. "
            "More than half the data is noise. Consider increasing eps."
        )
    elif noise_ratio > 0.3:
        warns.append(
            f"{n_noise} points ({noise_ratio*100:.1f}%) classified as noise. "
            "High noise ratio — eps may be too small."
        )

    if n_clusters > 1 and silhouette < 0.25:
        warns.append(
            f"Silhouette score is low ({silhouette:.3f}). "
            "Clusters may overlap or be poorly separated."
        )

    if suggested_eps is not None and n_clusters > 0:
        ratio = eps / suggested_eps if suggested_eps > 0 else None
        if ratio is not None and (ratio < 0.3 or ratio > 3.0):
            warns.append(
                f"Current eps={eps:.4f} differs substantially from suggested eps≈{suggested_eps:.4f}. "
                "Consider trying the suggested value for potentially better results."
            )

    return warns


# ─── Feature drivers (ANOVA) ──────────────────────────────────────────────────

def _compute_feature_drivers(
    cluster_data: pd.DataFrame,
    labels: np.ndarray,
    items: List[str],
) -> Dict[str, Any]:
    """ANOVA across non-noise clusters only."""
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
    inertia_proxy: float,
    cluster_data: pd.DataFrame,
    items: List[str],
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
        f"DBSCAN identified <strong>{n_clusters} cluster(s)</strong> and "
        f"<strong>{n_noise} noise point(s)</strong> ({noise_ratio*100:.1f}% of data). "
    )
    if n_clusters > 1:
        overall_quality += (
            f"The <strong>Silhouette Score of {silhouette:.3f}</strong> indicates clustering is {quality_desc} "
            f"<strong>Calinski-Harabasz: {calinski:.2f}</strong> (higher better). "
            f"<strong>Davies-Bouldin: {davies_bouldin:.3f}</strong> (lower better)."
        )
    else:
        overall_quality += "Quality metrics require ≥ 2 clusters to compute."

    # Per-cluster profile text
    overall_means = cluster_data.mean()
    overall_std = cluster_data.std().replace(0, 1)
    cluster_profiles_text: List[str] = []

    for name, profile in profiles.items():
        if name == 'Noise':
            cluster_profiles_text.append(
                f"<strong>Noise ({profile['percentage']:.1f}% of data):</strong> "
                "Points that did not meet the density threshold. "
                "They may be outliers or low-density transition points."
            )
            continue
        centroid = pd.Series(profile['centroid'])
        deviations = (centroid - overall_means) / overall_std
        top_features = deviations.nlargest(2).index.tolist()
        bottom_features = deviations.nsmallest(2).index.tolist()
        cluster_profiles_text.append(
            f"<strong>{name} ({profile['percentage']:.1f}% of data):</strong> "
            f"High in <strong>{', '.join(top_features)}</strong>; "
            f"low in <strong>{', '.join(bottom_features)}</strong>."
        )

    # Distribution
    non_noise = {k: v for k, v in profiles.items() if k != 'Noise'}
    if len(non_noise) > 1:
        sizes = [p['size'] for p in non_noise.values()]
        ratio = max(sizes) / max(min(sizes), 1)
        dist_desc = (
            "Non-noise clusters are relatively balanced in size."
            if ratio < 3
            else f"Non-noise cluster sizes are imbalanced (ratio {ratio:.1f}×)."
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

@router.post("/dbscan")
def dbscan_clustering(req: DBSCANRequest):
    try:
        df = pd.DataFrame(req.data)
        items = req.items
        eps = req.eps or 0.5
        min_samples = req.min_samples or 5
        metric = req.distanceMetric

        # Validation
        validation = _validate_input(df, items, eps, min_samples)
        if validation['errors']:
            raise HTTPException(
                status_code=400,
                detail="Validation failed: " + " | ".join(validation['errors']),
            )
        input_warnings: List[str] = validation['warnings']

        # Prepare data
        num_df = df[items].apply(pd.to_numeric, errors='coerce')
        cluster_data = num_df.dropna().copy()
        n_samples, n_features = cluster_data.shape

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

        # eps advisor
        eps_advice = _suggest_eps(X_scaled, min_samples, metric)

        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        unique_labels = np.unique(labels)

        # Core / border / noise counts
        core_mask = np.zeros(n_samples, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        n_core = int(core_mask.sum())
        n_border = int(((labels != -1) & ~core_mask).sum())

        # Metrics (noise-excluded, requires ≥ 2 clusters)
        non_noise_mask = labels != -1
        if n_clusters > 1 and non_noise_mask.sum() > 1:
            try:
                silhouette = _safe_float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask], metric=metric))
                davies_bouldin = _safe_float(davies_bouldin_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
                calinski = _safe_float(calinski_harabasz_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
            except Exception as exc:
                logger.debug("Metrics failed: %s", exc)
                silhouette = davies_bouldin = calinski = 0.0
        else:
            silhouette = davies_bouldin = calinski = 0.0

        # Inertia proxy (WCSS of non-noise)
        inertia_proxy = 0.0
        if n_clusters > 0:
            centroids = np.array([
                X_scaled[labels == lbl].mean(axis=0)
                for lbl in np.unique(labels[labels != -1])
            ])
            for lbl, cent in zip(np.unique(labels[labels != -1]), centroids):
                pts = X_scaled[labels == lbl]
                inertia_proxy += float(np.sum((pts - cent) ** 2))

        # Profiles
        profiles: Dict[str, Any] = {}
        for label in unique_labels:
            mask = labels == label
            subset = cluster_data.iloc[mask]
            name = 'Noise' if label == -1 else f'Cluster {label + 1}'
            profiles[name] = {
                'size': int(mask.sum()),
                'percentage': _safe_float(mask.sum() / n_samples * 100),
                'centroid': {col: _safe_float(subset[col].mean()) for col in items},
                'is_noise': bool(label == -1),
            }

        # Quality warnings
        quality_warnings = _quality_warnings(
            n_clusters, n_noise, n_samples, silhouette,
            eps, eps_advice.get('suggested_eps')
        )

        # Feature drivers
        feature_drivers = _compute_feature_drivers(cluster_data, labels, items)

        # Interpretations
        interpretations = _build_interpretations(
            profiles, n_clusters, n_noise, n_samples,
            silhouette, davies_bouldin, calinski, inertia_proxy,
            cluster_data, items,
        )

        # PCA note
        pca_note = None

        # ── Plots: 2×2 grid ───────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        palette = sns.color_palette('husl', n_colors=max(n_clusters, 1))

        # 1. PCA scatter — core / border / noise 구분 (top-left)
        ax1 = axes[0, 0]
        ax1.set_facecolor('#f0f0f0')
        if n_features >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(X_scaled)
            var_explained = sum(pca.explained_variance_ratio_)
            pca_note = (
                f"PCA projection explains {var_explained:.1%} of variance. "
                f"Boundaries shown in 2-D PCA space, not original {n_features}-D feature space."
            )

            # Point type arrays
            is_noise = labels == -1
            is_core = core_mask
            is_border = (~is_noise) & (~is_core)

            color_map = {}
            color_idx = 0
            for label in sorted(np.unique(labels)):
                if label != -1:
                    color_map[label] = palette[color_idx % len(palette)]
                    color_idx += 1

            # Layer 1: noise (bottom)
            if is_noise.any():
                ax1.scatter(pca_data[is_noise, 0], pca_data[is_noise, 1],
                            color='#bbbbbb', marker='x', s=35, alpha=0.6,
                            linewidth=1.2, label='Noise', zorder=1)

            # Layer 2: border points (medium)
            for label in sorted(np.unique(labels[labels != -1])):
                mask = (labels == label) & is_border
                if mask.any():
                    ax1.scatter(pca_data[mask, 0], pca_data[mask, 1],
                                color=color_map[label], marker='o', s=55,
                                alpha=0.55, edgecolors='white', linewidth=0.6,
                                zorder=2)

            # Layer 3: core points (top, larger + bold edge)
            for label in sorted(np.unique(labels[labels != -1])):
                mask = (labels == label) & is_core
                if mask.any():
                    ax1.scatter(pca_data[mask, 0], pca_data[mask, 1],
                                color=color_map[label], marker='o', s=90,
                                alpha=0.9, edgecolors='black', linewidth=0.9,
                                label=f'Cluster {label + 1}', zorder=3)

            # Legend patch for border
            from matplotlib.lines import Line2D
            legend_handles = ax1.get_legend_handles_labels()[0]
            legend_labels  = ax1.get_legend_handles_labels()[1]
            if is_border.any():
                border_patch = Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='gray', markeredgecolor='white',
                                      markersize=7, alpha=0.55, label='Border')
                legend_handles.append(border_patch)
                legend_labels.append('Border')

            ax1.legend(legend_handles, legend_labels, loc='best',
                       frameon=True, facecolor='white', edgecolor='gray', fontsize=8)
            ax1.set_title(
                f'PCA Projection ({var_explained:.1%} var) — Core · Border · Noise',
                fontsize=10, fontweight='bold'
            )
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            ax1.annotate("⚠ PCA 2-D projection only — not original feature space.",
                         xy=(0.5, -0.13), xycoords='axes fraction',
                         ha='center', fontsize=7, color='#666666', style='italic')
            ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. k-distance plot (top-right) — eps advisor
        ax2 = axes[0, 1]
        ax2.set_facecolor('#f5f5f5')
        k_distances = eps_advice.get('k_distances', [])
        if k_distances:
            ax2.plot(range(len(k_distances)), k_distances, color='#2E86AB', linewidth=2)
            ax2.axhline(eps, color='#E8505B', linestyle='--', linewidth=1.8, label=f'Current eps={eps:.3f}')
            suggested = eps_advice.get('suggested_eps')
            if suggested:
                ax2.axhline(suggested, color='#27AE60', linestyle=':', linewidth=1.8,
                            label=f'Suggested eps≈{suggested:.3f}')
            ax2.set_title(f'k-Distance Plot (k={min(min_samples, n_samples-1)}, {metric})', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Points (sorted)', fontsize=10)
            ax2.set_ylabel(f'{min(min_samples, n_samples-1)}-NN Distance', fontsize=10)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.4, linestyle='--')

        # 3. Cluster size bar (bottom-left)
        ax3 = axes[1, 0]
        ax3.set_facecolor('#f5f5f5')
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        bar_names = ['Noise' if i == -1 else f'Cluster {i+1}' for i in cluster_sizes.index]
        bar_colors = ['#aaaaaa' if i == -1 else palette[min(j, len(palette)-1)]
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

        # 4. Snake plot (bottom-right)
        ax4 = axes[1, 1]
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
                     fontsize=12, transform=ax4.transAxes, color='#888888')
            ax4.set_title('Snake Plot (Cluster Centroids)', fontsize=11, fontweight='bold')

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
                    'n_samples': n_samples,
                    'n_core': n_core,
                    'n_border': n_border,
                    'eps': eps,
                    'min_samples': min_samples,
                    'scaler': req.scalerType,
                    'metric': metric,
                    'labels': labels.tolist(),
                    'inertia_proxy': _safe_float(inertia_proxy),
                },
                'eps_advice': eps_advice,
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
        logger.exception("Unexpected error in dbscan_clustering")
        raise HTTPException(status_code=500, detail=f"DBSCAN analysis failed: {exc}")
