from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.decomposition import PCA
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

class KMedoidsRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(...)
    items: List[str] = Field(...)
    nClusters: int = Field(default=3, ge=2, le=50)
    maxIter: Optional[int] = 300
    scalerType: str = Field(default='standard', pattern='^(standard|robust|minmax)$')


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
        logger.debug("_safe_float conversion failed for %r: %s", val, exc)
        return default


# ─── Validation ───────────────────────────────────────────────────────────────

def _validate_input(df: pd.DataFrame, items: List[str], n_clusters: int) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if n_clusters < 2:
        errors.append(f"nClusters must be ≥ 2, got {n_clusters}.")

    missing = [c for c in items if c not in df.columns]
    if missing:
        errors.append(f"Column(s) not found in data: {', '.join(missing)}.")
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
        errors.append("Non-numeric data detected: " + "; ".join(non_numeric) + ". Only numeric columns are supported.")
    if errors:
        return {'errors': errors, 'warnings': warnings}

    num_df = df[items].apply(pd.to_numeric, errors='coerce')
    clean_df = num_df.dropna()

    # Categorical-disguised-as-numeric
    categorical_suspects: List[str] = []
    for col in items:
        raw_col = df[col]
        n_unique = raw_col.nunique(dropna=True)
        n_rows = len(raw_col.dropna())
        is_low_cardinality = n_unique <= 10 and n_rows >= 20 and (n_unique / n_rows) <= 0.05
        is_string_origin = pd.api.types.is_object_dtype(raw_col) or pd.api.types.is_string_dtype(raw_col)
        if is_low_cardinality or is_string_origin:
            categorical_suspects.append(
                f"'{col}' ({n_unique} unique values" + (", string origin" if is_string_origin else "") + ")"
            )
    if categorical_suspects:
        warnings.append(
            "Possible categorical-disguised-as-numeric column(s): "
            + ", ".join(categorical_suspects)
            + ". K-Medoids assumes continuous numeric features."
        )

    # Sample size
    n = len(clean_df)
    if n < n_clusters * 2:
        errors.append(
            f"Need at least {n_clusters * 2} complete observations for {n_clusters} clusters, "
            f"but only {n} found after removing rows with missing values."
        )
    elif n < 30:
        warnings.append(f"Small sample (N={n}). K-Medoids results may be unstable.")
    if errors:
        return {'errors': errors, 'warnings': warnings}

    # Constant / near-zero variance
    for col in items:
        s = clean_df[col].std()
        if s == 0.0:
            errors.append(f"Column '{col}' is constant (zero variance). K-Medoids cannot use it.")
        elif s < 1e-6:
            warnings.append(f"Column '{col}' has near-zero variance (std={s:.2e}).")

    # Duplicate rows
    n_dupes = clean_df.duplicated().sum()
    if n_dupes / max(len(clean_df), 1) > 0.3:
        warnings.append(
            f"{n_dupes} duplicate rows detected ({n_dupes / len(clean_df) * 100:.1f}%). "
            "High duplication can distort medoid selection."
        )

    # Extreme outliers
    z = np.abs((clean_df - clean_df.mean()) / clean_df.std(ddof=0).replace(0, 1))
    n_extreme = int((z > 5).any(axis=1).sum())
    if n_extreme > 0:
        warnings.append(
            f"{n_extreme} row(s) have at least one feature with |z| > 5. "
            "Note: K-Medoids is more robust to outliers than K-Means."
        )

    return {'errors': errors, 'warnings': warnings}


# ─── Optimal k ────────────────────────────────────────────────────────────────

def _recommend_k(
    k_range: List[int],
    inertias: List[float],
    silhouette_scores: List[float],
    ch_scores: List[float],
) -> Dict[str, Any]:
    sil_k = k_range[int(np.argmax(silhouette_scores))] if any(s > -1 for s in silhouette_scores) else k_range[0]
    ch_k = k_range[int(np.argmax(ch_scores))] if ch_scores else k_range[0]

    elbow_k = k_range[0]
    if len(inertias) >= 3:
        arr = np.array(inertias)
        d2 = np.diff(arr, 2)
        elbow_k = k_range[int(np.argmax(d2)) + 1]

    votes: Dict[int, int] = {}
    for k in [sil_k, ch_k, elbow_k]:
        votes[k] = votes.get(k, 0) + 1

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


# ─── Quality warnings ─────────────────────────────────────────────────────────

def _quality_warnings(
    silhouette: float,
    profiles: Dict,
    n_clusters: int,
    recommended_k: int,
    n_samples: int,
) -> List[str]:
    warns: List[str] = []

    if silhouette < 0.25:
        warns.append(
            f"Silhouette score is very low ({silhouette:.3f} < 0.25). "
            "Clusters are not well-separated. Consider a different k or feature set."
        )
    elif silhouette < 0.50:
        warns.append(f"Silhouette score is weak ({silhouette:.3f}). Some cluster overlap is likely.")

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


# ─── Feature drivers (ANOVA) ──────────────────────────────────────────────────

def _compute_feature_drivers(
    cluster_data: pd.DataFrame,
    labels: np.ndarray,
    items: List[str],
) -> Dict[str, Any]:
    drivers: List[Dict[str, Any]] = []
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {'features': [], 'note': 'ANOVA requires ≥ 2 clusters.'}

        for col in items:
            groups = [cluster_data.loc[labels == lbl, col].dropna().values for lbl in unique_labels]
            if any(len(g) < 2 for g in groups):
                continue
            try:
                f_stat, p_val = f_oneway(*groups)
                if not (np.isfinite(f_stat) and np.isfinite(p_val)):
                    continue
                all_vals = cluster_data[col].dropna().values
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
                logger.debug("ANOVA failed for feature '%s': %s", col, exc)

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


# ─── PAM algorithm ────────────────────────────────────────────────────────────

def kmedoids_pam(X, n_clusters, max_iter=300, random_state=42):
    """K-Medoids (PAM) implementation using NumPy."""
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    medoid_indices = rng.choice(n_samples, n_clusters, replace=False)
    dist_matrix = pairwise_distances(X, metric='euclidean')
    current_cost = float('inf')

    for _ in range(max_iter):
        labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
        new_cost = np.sum([dist_matrix[j, medoid_indices[labels[j]]] for j in range(n_samples)])

        if new_cost >= current_cost:
            break
        current_cost = new_cost

        new_medoid_indices = medoid_indices.copy()
        for cluster_idx in range(n_clusters):
            cluster_points_indices = np.where(labels == cluster_idx)[0]
            if len(cluster_points_indices) == 0:
                continue
            cluster_dist_matrix = dist_matrix[np.ix_(cluster_points_indices, cluster_points_indices)]
            costs = np.sum(cluster_dist_matrix, axis=1)
            best_point_in_cluster_idx = np.argmin(costs)
            new_medoid_indices[cluster_idx] = cluster_points_indices[best_point_in_cluster_idx]

        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        medoid_indices = new_medoid_indices

    final_labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
    inertia = np.sum([dist_matrix[j, medoid_indices[final_labels[j]]] ** 2 for j in range(n_samples)])

    return final_labels, medoid_indices, inertia


# ─── Main endpoint ────────────────────────────────────────────────────────────

@router.post("/kmedoids")
def kmedoids_clustering(req: KMedoidsRequest):
    try:
        df = pd.DataFrame(req.data)
        items = req.items
        n_clusters = req.nClusters
        max_iter = req.maxIter or 300

        # Validation
        validation = _validate_input(df, items, n_clusters)
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
        scaler = _scaler_map.get(req.scalerType, StandardScaler())
        cluster_data_scaled = scaler.fit_transform(cluster_data)

        if not np.all(np.isfinite(cluster_data_scaled)):
            raise HTTPException(
                status_code=400,
                detail="Standardized data contains NaN or inf values. Check for constant or near-constant columns.",
            )

        # Optimal k search
        sqrt_bound = max(2, int(np.sqrt(n_samples) / 2))
        max_k = min(sqrt_bound if n_samples > 200 else 10, n_samples - 1, 20)
        k_range = list(range(2, max_k + 1))
        inertias: List[float] = []
        silhouette_scores: List[float] = []
        ch_scores: List[float] = []

        for k in k_range:
            try:
                lbl, _, ine = kmedoids_pam(cluster_data_scaled, n_clusters=k, max_iter=100, random_state=42)
                inertias.append(_safe_float(ine))
                if len(np.unique(lbl)) > 1:
                    silhouette_scores.append(_safe_float(silhouette_score(cluster_data_scaled, lbl)))
                    ch_scores.append(_safe_float(calinski_harabasz_score(cluster_data_scaled, lbl)))
                else:
                    silhouette_scores.append(-1.0)
                    ch_scores.append(0.0)
            except Exception as exc:
                logger.debug("Optimal k search failed for k=%d: %s", k, exc)
                inertias.append(0.0)
                silhouette_scores.append(-1.0)
                ch_scores.append(0.0)

        optimal_k_info = _recommend_k(k_range, inertias, silhouette_scores, ch_scores)
        recommended_k: int = optimal_k_info['recommended_k']

        # Main clustering
        labels, medoid_indices, inertia = kmedoids_pam(
            cluster_data_scaled, n_clusters=n_clusters, max_iter=max_iter, random_state=42
        )

        medoids = cluster_data.iloc[medoid_indices].to_dict('records')
        unique_labels, counts = np.unique(labels, return_counts=True)

        if len(unique_labels) > 1:
            silhouette = _safe_float(silhouette_score(cluster_data_scaled, labels))
            davies_bouldin = _safe_float(davies_bouldin_score(cluster_data_scaled, labels))
            calinski = _safe_float(calinski_harabasz_score(cluster_data_scaled, labels))
        else:
            silhouette = davies_bouldin = calinski = 0.0

        # Profiles
        profiles: Dict[str, Any] = {}
        for i, label in enumerate(unique_labels):
            mask = labels == label
            subset = cluster_data.iloc[mask]
            profiles[f'Cluster {label + 1}'] = {
                'size': int(counts[i]),
                'percentage': _safe_float(counts[i] / n_samples * 100),
                'centroid': {col: _safe_float(subset[col].mean()) for col in items},
            }

        # Quality warnings
        quality_warnings = _quality_warnings(silhouette, profiles, n_clusters, recommended_k, n_samples)

        # Interpretations
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
            f"<strong>Inertia (WCSS): {_safe_float(inertia):.2f}</strong>."
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

        # Feature drivers
        feature_drivers = _compute_feature_drivers(cluster_data, labels, items)

        # PCA disclaimer
        pca_disclaimer = ""

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        colors = sns.color_palette('husl', n_colors=n_clusters)

        # 1. Scatter (PCA)
        if n_features >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(cluster_data_scaled)
            var_explained = sum(pca.explained_variance_ratio_)

            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0].scatter(
                    pca_data[mask, 0], pca_data[mask, 1],
                    c=[colors[i]], label=f'Cluster {label + 1}',
                    alpha=0.7, s=60, edgecolors='white', linewidth=0.8
                )

            medoids_pca = pca.transform(cluster_data_scaled[medoid_indices])
            axes[0].scatter(
                medoids_pca[:, 0], medoids_pca[:, 1],
                s=250, c='white', marker='o', edgecolors='black', linewidth=2, zorder=10
            )
            axes[0].scatter(
                medoids_pca[:, 0], medoids_pca[:, 1],
                s=80, c='black', marker='X', linewidth=1, zorder=11, label='Medoids'
            )

            pca_disclaimer = (
                f"Note: cluster boundaries are projected onto 2-D PCA space for visualisation only. "
                f"They do NOT represent the true cluster boundaries in the original {n_features}-dimensional feature space."
            )
            axes[0].set_title(
                f'Clusters (PCA projection — {var_explained:.1%} variance explained)',
                fontsize=11, fontweight='bold'
            )
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            axes[0].annotate(
                "⚠ Boundaries shown in PCA 2-D space, not original feature space.",
                xy=(0.5, -0.13), xycoords='axes fraction',
                ha='center', fontsize=7, color='#666666', style='italic',
            )
            axes[0].legend(loc='best', frameon=True, facecolor='white', edgecolor='gray')
            axes[0].grid(True, alpha=0.3, linestyle='--')

        # 2. Radar chart of medoids
        medoids_df = pd.DataFrame(medoids)
        if not medoids_df.empty and len(items) >= 3:
            min_vals = cluster_data[items].min()
            max_vals = cluster_data[items].max()
            range_vals = (max_vals - min_vals).replace(0, 1)
            medoids_norm = (medoids_df[items] - min_vals) / range_vals

            angles = np.linspace(0, 2 * np.pi, len(items), endpoint=False).tolist()
            angles += angles[:1]

            ax_radar = fig.add_subplot(122, polar=True)
            for i, (_, row) in enumerate(medoids_norm.iterrows()):
                values = row.tolist() + row.tolist()[:1]
                ax_radar.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i + 1}', color=colors[i])
                ax_radar.fill(angles, values, alpha=0.25, color=colors[i])

            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(items, fontsize=9)
            ax_radar.set_title('Cluster Medoids Profile (Normalized)', fontsize=12, fontweight='bold', pad=20)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax_radar.grid(True, alpha=0.3)
            axes[1].set_visible(False)
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

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
                    'inertia': _safe_float(inertia),
                    'medoids': medoids,
                    'medoid_indices': medoid_indices.tolist(),
                    'labels': labels.tolist(),
                    'scaler': req.scalerType,
                },
                'profiles': profiles,
                'feature_drivers': feature_drivers,
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
        logger.exception("Unexpected error in kmedoids_clustering")
        raise HTTPException(status_code=500, detail=f"K-Medoids analysis failed: {exc}")
