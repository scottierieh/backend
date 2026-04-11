from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.feature_selection import f_classif
from scipy import stats
import io
import base64
import logging
import matplotlib.pyplot as plt
import seaborn as sns

router = APIRouter()
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)

# ── Compatibility tables ────────────────────────────────────────────────────
SUPPORTED_LINKAGES = {'ward', 'complete', 'average', 'single', 'weighted', 'centroid', 'median'}
SUPPORTED_METRICS  = {'euclidean', 'manhattan', 'cosine', 'correlation', 'chebyshev', 'canberra'}

# ward, centroid, median require Euclidean
EUCLIDEAN_ONLY_LINKAGES = {'ward', 'centroid', 'median'}

LARGE_DATA_THRESHOLD = 5_000   # subsample above this
LARGE_DATA_SAMPLE    = 5_000


class HcaRequest(BaseModel):
    data: List[Dict[str, Any]]
    items: List[str]
    linkageMethod: str = 'ward'
    distanceMetric: str = 'euclidean'
    nClusters: Optional[int] = None
    scalerType: Optional[str] = 'standard'


def _to_native(obj):
    if isinstance(obj, np.integer):   return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):   return bool(obj)
    elif isinstance(obj, dict):       return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):       return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ── 1. Comprehensive input validation ──────────────────────────────────────
def _validate_input(
    df: pd.DataFrame,
    items: List[str],
    linkage_method: str,
    distance_metric: str,
    n_clusters_req: Optional[int],
) -> tuple[list, pd.DataFrame]:
    input_warnings = []

    # Column existence
    missing = [c for c in items if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {', '.join(missing)}")

    if len(items) < 2:
        raise ValueError("At least 2 variables are required.")

    # Linkage / metric support
    if linkage_method not in SUPPORTED_LINKAGES:
        raise ValueError(
            f"Unsupported linkage '{linkage_method}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_LINKAGES))}"
        )
    if distance_metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{distance_metric}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
        )

    # Linkage-metric compatibility
    if linkage_method in EUCLIDEAN_ONLY_LINKAGES and distance_metric != 'euclidean':
        raise ValueError(
            f"Linkage '{linkage_method}' requires Euclidean distance. "
            f"Either switch to 'euclidean' metric or use a different linkage method."
        )

    # Numeric check
    non_numeric = [
        c for c in items
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(
            f"Non-numeric column(s): {', '.join(non_numeric)}. "
            "HCA requires numeric variables only."
        )

    # Categorical disguised as numeric
    for col in items:
        n_unique = df[col].nunique(dropna=True)
        n_total  = df[col].count()
        if n_total > 20 and n_unique <= 5:
            input_warnings.append(
                f"'{col}' has only {n_unique} unique values — may be categorical. "
                "Consider excluding it from clustering."
            )

    # Constant / near-zero variance
    for col in items:
        col_std  = df[col].std(skipna=True)
        col_mean = abs(df[col].mean(skipna=True))
        if col_std == 0:
            raise ValueError(
                f"'{col}' is constant (zero variance). Remove it before clustering."
            )
        if col_mean > 1e-8 and (col_std / col_mean) < 0.01:
            input_warnings.append(
                f"'{col}' has near-zero variance (CV < 1%). "
                "It may not contribute meaningfully to clustering."
            )

    cluster_data = df[items].apply(pd.to_numeric, errors='coerce').dropna()
    n_total = len(cluster_data)

    # Duplicate rows
    n_dupes = n_total - cluster_data.drop_duplicates().shape[0]
    if n_dupes > 0:
        dupe_pct = n_dupes / n_total * 100
        if dupe_pct > 10:
            input_warnings.append(
                f"{n_dupes} duplicate rows ({dupe_pct:.1f}%) detected. "
                "HCA distances may be distorted — consider de-duplication."
            )

    # Extreme outliers (|z| > 5)
    for col in items:
        z = np.abs(stats.zscore(cluster_data[col].dropna()))
        extreme = int((z > 5).sum())
        if extreme > 0:
            input_warnings.append(
                f"'{col}' has {extreme} extreme outlier(s) (|z| > 5). "
                "Consider Robust scaler or outlier removal."
            )

    # Minimum sample size
    if n_total < 3:
        raise ValueError("Need at least 3 valid observations.")
    if n_total < 20:
        input_warnings.append(
            f"Small sample size (n={n_total}). "
            "HCA results may not generalise well."
        )

    # n_clusters range check
    if n_clusters_req is not None:
        if n_clusters_req < 2 or n_clusters_req >= n_total:
            raise ValueError(
                f"n_clusters must be between 2 and {n_total - 1} (got {n_clusters_req})."
            )

    return input_warnings, cluster_data


# ── 2. Large-data strategy ──────────────────────────────────────────────────
def _maybe_subsample(cluster_data: pd.DataFrame, seed: int = 42):
    """Return (data_for_linkage, is_large_data, full_data)."""
    n = len(cluster_data)
    if n > LARGE_DATA_THRESHOLD:
        sample = cluster_data.sample(n=LARGE_DATA_SAMPLE, random_state=seed)
        return sample, True
    return cluster_data, False


# ── 3. Optimal k search ─────────────────────────────────────────────────────
def find_optimal_k(Z, X_scaled, max_k=10):
    max_k = min(max_k, len(X_scaled) - 1)
    recommendations = {}

    silhouette_scores, ch_scores = [], []
    for k in range(2, max_k + 1):
        lbl = fcluster(Z, k, criterion='maxclust')
        if len(np.unique(lbl)) > 1:
            silhouette_scores.append((k, safe_float(silhouette_score(X_scaled, lbl))))
            ch_scores.append((k, safe_float(calinski_harabasz_score(X_scaled, lbl))))

    if silhouette_scores:
        recommendations['silhouette'] = max(silhouette_scores, key=lambda x: x[1])[0]
    if ch_scores:
        recommendations['calinski_harabasz'] = max(ch_scores, key=lambda x: x[1])[0]

    # Elbow (second derivative of inertia)
    inertias = []
    for k in range(1, max_k + 1):
        lbl = fcluster(Z, k, criterion='maxclust')
        inertia = sum(
            np.sum((X_scaled[lbl == cid] - X_scaled[lbl == cid].mean(axis=0)) ** 2)
            for cid in np.unique(lbl)
        )
        inertias.append(inertia)

    if len(inertias) >= 3:
        diffs2 = np.diff(np.diff(inertias))
        recommendations['elbow'] = int(np.argmax(diffs2) + 2)

    return recommendations


# ── 4. Cophenetic correlation + cluster stability ───────────────────────────
def _cluster_stability(Z, X_scaled, n_clusters: int, metric: str):
    """
    Returns:
      cophenetic_r  – cophenetic correlation (goodness of dendrogram fit)
      stability     – simple bootstrap-like consistency measure
    """
    # Cophenetic correlation
    try:
        dist_condensed = pdist(X_scaled, metric=metric if metric != 'manhattan' else 'cityblock')
        c, _ = cophenet(Z, dist_condensed)
        cophenetic_r = safe_float(c)
    except Exception as e:
        logger.warning(f"Cophenetic failed: {e}")
        cophenetic_r = None

    # Bootstrap stability: resample 80 % × 10 times, measure label agreement
    n = len(X_scaled)
    agreements = []
    rng = np.random.default_rng(42)
    for _ in range(10):
        idx = rng.choice(n, size=int(n * 0.8), replace=False)
        X_boot = X_scaled[idx]
        try:
            Z_boot = linkage(X_boot, method='ward', metric='euclidean')
            lbl_boot = fcluster(Z_boot, n_clusters, criterion='maxclust')
            lbl_full = fcluster(Z, n_clusters, criterion='maxclust')[idx]
            # Map labels by majority vote
            from scipy.stats import mode as scipy_mode
            mapping = {}
            for orig in np.unique(lbl_full):
                candidates = lbl_boot[lbl_full == orig]
                if len(candidates):
                    m = scipy_mode(candidates, keepdims=True)
                    mapping[orig] = int(m.mode[0])
            matched = np.array([mapping.get(l, -1) for l in lbl_full])
            agreements.append(float((matched == lbl_boot).mean()))
        except Exception:
            pass

    stability_score = safe_float(np.mean(agreements)) if agreements else None

    return cophenetic_r, stability_score


# ── 5. Feature drivers (ANOVA) ──────────────────────────────────────────────
def _feature_drivers(cluster_data: pd.DataFrame, labels: np.ndarray, items: List[str]):
    if len(np.unique(labels)) < 2:
        return None
    try:
        X = cluster_data[items].values
        f_stats, p_values = f_classif(X, labels)

        features = []
        for i, col in enumerate(items):
            f  = safe_float(f_stats[i])
            p  = safe_float(p_values[i], default=1.0)
            groups     = [X[labels == k, i] for k in np.unique(labels)]
            grand_mean = X[:, i].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
            ss_total   = ((X[:, i] - grand_mean) ** 2).sum()
            eta_sq     = safe_float(ss_between / ss_total) if ss_total > 0 else 0.0
            effect     = 'large' if eta_sq >= 0.14 else 'medium' if eta_sq >= 0.06 else 'small'
            features.append({
                'feature': col, 'f_stat': f, 'p_value': p,
                'eta_squared': eta_sq, 'effect_size': effect,
                'is_significant': bool(p < 0.05), 'rank': 0,
            })

        features.sort(key=lambda x: x['eta_squared'], reverse=True)
        for rank, feat in enumerate(features, 1):
            feat['rank'] = rank

        return {
            'features': features,
            'top_driver': features[0]['feature'] if features else None,
            'note': 'ANOVA F-test: measures how well each variable separates HCA clusters.',
        }
    except Exception as e:
        logger.warning(f"Feature drivers failed: {e}")
        return None


# ── 6. Quality warnings ──────────────────────────────────────────────────────
def _quality_warnings(
    profiles: dict,
    silhouette: float,
    cophenetic_r: Optional[float],
    stability: Optional[float],
    n_clusters: int,
    optimal_k_rec: dict,
    linkage_method: str,
    n_samples: int,
    is_large_data: bool,
):
    warnings_list = []

    if silhouette < 0.25 and silhouette > 0:
        warnings_list.append(
            f"Silhouette score ({silhouette:.3f}) is low. "
            "Clusters may heavily overlap — consider adjusting k."
        )

    if cophenetic_r is not None and cophenetic_r < 0.75:
        warnings_list.append(
            f"Cophenetic correlation ({cophenetic_r:.3f}) is below 0.75. "
            "The dendrogram may not faithfully represent the original distances."
        )

    if stability is not None and stability < 0.7:
        warnings_list.append(
            f"Cluster stability score ({stability:.2f}) is low. "
            "Results may change with different subsamples — treat with caution."
        )

    for name, p in profiles.items():
        if p['percentage'] < 5.0:
            warnings_list.append(
                f"{name} is very small ({p['percentage']:.1f}%, n={p['size']}). "
                "May be unstable or represent outliers."
            )

    # k mismatch
    sil_rec = optimal_k_rec.get('silhouette')
    if sil_rec and abs(sil_rec - n_clusters) >= 2:
        warnings_list.append(
            f"Silhouette recommends k={sil_rec} but you chose k={n_clusters}. "
            f"Consider re-running with k={sil_rec}."
        )

    if is_large_data:
        warnings_list.append(
            f"Large dataset: dendrogram and linkage built on a {LARGE_DATA_SAMPLE}-row subsample. "
            "Final labels assigned to all rows via nearest-centroid."
        )

    return warnings_list


# ── Main endpoint ────────────────────────────────────────────────────────────
@router.post("/hca")
async def hierarchical_cluster_analysis(request: HcaRequest):
    try:
        df              = pd.DataFrame(request.data)
        items           = request.items
        linkage_method  = request.linkageMethod
        distance_metric = request.distanceMetric
        n_clusters_req  = request.nClusters
        scaler_type     = request.scalerType or 'standard'

        # ── Validation ──────────────────────────────────────────────────────
        input_warnings, cluster_data = _validate_input(
            df, items, linkage_method, distance_metric, n_clusters_req
        )

        # ── Scaling ─────────────────────────────────────────────────────────
        if scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_scaled_full = scaler.fit_transform(cluster_data)

        # ── Large-data strategy ─────────────────────────────────────────────
        sample_data, is_large_data = _maybe_subsample(cluster_data)
        if is_large_data:
            X_scaled_sample = scaler.transform(sample_data)
        else:
            X_scaled_sample = X_scaled_full

        n_samples = len(cluster_data)

        # ward requires euclidean — already validated above, but enforce
        metric_for_linkage = distance_metric
        if linkage_method in EUCLIDEAN_ONLY_LINKAGES:
            metric_for_linkage = 'euclidean'
        # scipy uses 'cityblock' for manhattan
        scipy_metric = 'cityblock' if metric_for_linkage == 'manhattan' else metric_for_linkage

        # ── Linkage ─────────────────────────────────────────────────────────
        Z = linkage(X_scaled_sample, method=linkage_method, metric=scipy_metric)

        # ── Optimal k ───────────────────────────────────────────────────────
        max_k = min(10, len(X_scaled_sample) // 2)
        optimal_k_rec = find_optimal_k(Z, X_scaled_sample, max_k=max_k)

        if n_clusters_req is None:
            n_clusters = optimal_k_rec.get('silhouette', 3)
            n_clusters = max(2, min(int(n_clusters), len(X_scaled_sample) - 1))
        else:
            n_clusters = n_clusters_req

        # ── Labels ──────────────────────────────────────────────────────────
        if is_large_data:
            # Assign full dataset via nearest centroid from sample clustering
            sample_labels = fcluster(Z, n_clusters, criterion='maxclust')
            centroids = np.array([
                X_scaled_sample[sample_labels == k].mean(axis=0)
                for k in np.unique(sample_labels)
            ])
            from sklearn.metrics import pairwise_distances_argmin
            full_labels = pairwise_distances_argmin(X_scaled_full, centroids) + 1
            labels = full_labels
        else:
            labels = fcluster(Z, n_clusters, criterion='maxclust')

        # ── Quality metrics ──────────────────────────────────────────────────
        if len(np.unique(labels)) > 1:
            silhouette = safe_float(silhouette_score(X_scaled_full, labels))
            calinski   = safe_float(calinski_harabasz_score(X_scaled_full, labels))
            davies     = safe_float(davies_bouldin_score(X_scaled_full, labels))
        else:
            silhouette, calinski, davies = 0.0, 0.0, 0.0

        # ── Cophenetic correlation + stability ───────────────────────────────
        cophenetic_r, stability_score = _cluster_stability(
            Z, X_scaled_sample, n_clusters, scipy_metric
        )

        # ── Profiles ─────────────────────────────────────────────────────────
        profiles = {}
        for cluster_id in sorted(np.unique(labels)):
            mask = labels == cluster_id
            subset = cluster_data.iloc[mask] if not is_large_data else cluster_data[mask]
            profiles[f"Cluster {cluster_id}"] = {
                'size':       int(mask.sum()),
                'percentage': safe_float(mask.sum() / n_samples * 100),
                'centroid':   {col: safe_float(subset[col].mean()) for col in items},
                'std':        {col: safe_float(subset[col].std())  for col in items},
            }

        # ── Feature drivers ───────────────────────────────────────────────────
        feature_drivers = _feature_drivers(
            cluster_data if not is_large_data else cluster_data,
            labels, items
        )

        # ── Quality warnings ──────────────────────────────────────────────────
        quality_warnings = _quality_warnings(
            profiles, silhouette, cophenetic_r, stability_score,
            n_clusters, optimal_k_rec, linkage_method, n_samples, is_large_data
        )

        # ── Interpretations ───────────────────────────────────────────────────
        final_metrics = {
            'silhouette':        silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin':    davies,
            'note': 'Silhouette: higher better (-1..1). Davies-Bouldin: lower better. Calinski-Harabasz: higher better.',
        }
        interpretations = _generate_interpretations(profiles, final_metrics, items, n_clusters)

        # ── Visualization ─────────────────────────────────────────────────────
        plot_base64 = _build_plot(
            Z, labels, profiles, items, cluster_data if not is_large_data else sample_data,
            X_scaled_sample, X_scaled_full,
            silhouette, calinski, davies,
            cophenetic_r, stability_score,
            n_clusters, linkage_method, distance_metric, n_samples, is_large_data
        )

        return _to_native({
            'results': {
                'n_clusters':               int(n_clusters),
                'is_large_data':            is_large_data,
                'clustering_summary': {
                    'n_samples':    n_samples,
                    'scaler':       scaler_type,
                    'linkage':      linkage_method,
                    'metric':       distance_metric,
                    'n_clusters':   int(n_clusters),
                },
                'profiles':                 profiles,
                'stability': {
                    'cophenetic_r':   cophenetic_r,
                    'stability_score': stability_score,
                    'note': (
                        'Cophenetic r: how faithfully the dendrogram preserves pairwise distances (>0.75 good). '
                        'Stability: bootstrap label consistency (>0.70 good).'
                    ),
                },
                'feature_drivers':          feature_drivers,
                'final_metrics':            final_metrics,
                'optimal_k_recommendation': optimal_k_rec,
                'interpretations':          interpretations,
                'warnings': {
                    'input':   input_warnings,
                    'quality': quality_warnings,
                },
            },
            'plot': plot_base64,
        })

    except ValueError as e:
        logger.error(f"HCA validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"HCA unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ── Plot builder ─────────────────────────────────────────────────────────────
def _build_plot(
    Z, labels, profiles, items, cluster_data_raw,
    X_scaled_sample, X_scaled_full,
    silhouette, calinski, davies,
    cophenetic_r, stability_score,
    n_clusters, linkage_method, distance_metric, n_samples, is_large_data
):
    fig = plt.figure(figsize=(16, 20))
    colors = plt.cm.Set2(np.linspace(0, 1, len(profiles)))
    cluster_names = list(profiles.keys())

    # 1. Dendrogram
    ax1 = fig.add_subplot(3, 2, 1)
    dendrogram(Z, ax=ax1, truncate_mode='lastp', p=30,
               leaf_rotation=90, leaf_font_size=8)
    ax1.set_title('Dendrogram', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Sample Index or Cluster Size')
    ax1.set_ylabel('Distance')
    if n_clusters >= 2 and len(Z) >= n_clusters - 1:
        cut_height = Z[-(n_clusters - 1), 2]
        ax1.axhline(y=cut_height, color='r', linestyle='--', alpha=0.7,
                    label=f'Cut for {n_clusters} clusters')
    if cophenetic_r is not None:
        ax1.set_title(f'Dendrogram (cophenetic r={cophenetic_r:.3f})',
                      fontweight='bold', fontsize=11)
    ax1.legend(fontsize=8)

    # 2. Cluster sizes
    ax2 = fig.add_subplot(3, 2, 2)
    sizes = [profiles[c]['size'] for c in cluster_names]
    bars  = ax2.bar(cluster_names, sizes, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title('Cluster Sizes', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Observations')
    for bar, size, pct in zip(bars, sizes, [profiles[c]['percentage'] for c in cluster_names]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{size}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)

    # 3. Cluster profile heatmap
    ax3 = fig.add_subplot(3, 2, 3)
    centroid_df = pd.DataFrame({name: p['centroid'] for name, p in profiles.items()})
    row_mean = centroid_df.mean(axis=1).values.reshape(-1, 1)
    row_std  = (centroid_df.std(axis=1).values.reshape(-1, 1) + 1e-10)
    centroid_scaled = (centroid_df - row_mean) / row_std
    sns.heatmap(centroid_scaled, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax3, cbar_kws={'label': 'Z-score'})
    ax3.set_title('Cluster Profiles (Standardized)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Variable')

    # 4. Radar chart
    ax4 = fig.add_subplot(3, 2, 4, projection='polar')
    plot_vars     = items[:min(6, len(items))]
    plot_clusters = cluster_names[:min(3, len(cluster_names))]
    angles = np.linspace(0, 2 * np.pi, len(plot_vars), endpoint=False).tolist()
    angles += angles[:1]

    for i, cluster in enumerate(plot_clusters):
        values = [profiles[cluster]['centroid'].get(v, 0) for v in plot_vars]
        min_v, max_v = min(values), max(values)
        values = [(v - min_v) / (max_v - min_v) for v in values] if max_v > min_v else [0.5] * len(values)
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=cluster, color=colors[i])
        ax4.fill(angles, values, alpha=0.25, color=colors[i])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(plot_vars, size=8)
    ax4.set_title('Cluster Comparison (Top Variables)', fontweight='bold', fontsize=12, pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 5. Silhouette by cluster
    ax5 = fig.add_subplot(3, 2, 5)
    try:
        sil_samples = silhouette_samples(X_scaled_full, labels)
        cluster_silhouettes = [
            safe_float(sil_samples[labels == cid].mean())
            for cid in sorted(np.unique(labels))
        ]
    except Exception:
        cluster_silhouettes = [0.0] * len(cluster_names)

    ax5.barh(cluster_names, cluster_silhouettes, color=colors, edgecolor='black', alpha=0.8)
    ax5.axvline(x=silhouette, color='red', linestyle='--', label=f'Overall: {silhouette:.3f}')
    ax5.set_title('Silhouette Score by Cluster', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Silhouette Score')
    ax5.legend()
    ax5.set_xlim(-0.2, 1)

    # 6. Metrics summary
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    coph_str = f"{cophenetic_r:.4f}" if cophenetic_r is not None else "N/A"
    stab_str = f"{stability_score:.2f}" if stability_score is not None else "N/A"
    large_note = f"\n  ⚠ Large-data mode (subsample={LARGE_DATA_SAMPLE})" if is_large_data else ""
    metrics_text = (
        f"  Clustering Quality Metrics\n"
        f"  {'═' * 34}\n\n"
        f"  Silhouette Score:       {silhouette:.4f}\n"
        f"  {'✓' if silhouette >= 0.5 else '△' if silhouette >= 0.25 else '✗'}"
        f"  {'Good' if silhouette >= 0.5 else 'Moderate' if silhouette >= 0.25 else 'Weak'} separation\n\n"
        f"  Calinski-Harabasz:      {calinski:.2f}\n"
        f"  Davies-Bouldin:         {davies:.4f}\n"
        f"  {'✓' if davies <= 1 else '△' if davies <= 2 else '✗'}"
        f"  {'Good' if davies <= 1 else 'Moderate' if davies <= 2 else 'Consider different k'}\n\n"
        f"  Cophenetic r:           {coph_str}\n"
        f"  Bootstrap Stability:    {stab_str}\n\n"
        f"  {'═' * 34}\n"
        f"  Linkage: {linkage_method}  |  Metric: {distance_metric}\n"
        f"  k={n_clusters}  |  n={n_samples}  |  vars={len(items)}{large_note}"
    )
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')


# ── Interpretations ───────────────────────────────────────────────────────────
def _generate_interpretations(profiles, metrics, items, n_clusters):
    silhouette = metrics['silhouette']

    if silhouette >= 0.7:
        quality = (f"<strong>Excellent clustering quality</strong> (Silhouette = {silhouette:.3f}). "
                   f"The {n_clusters} clusters are well-separated and internally cohesive.")
    elif silhouette >= 0.5:
        quality = (f"<strong>Good clustering quality</strong> (Silhouette = {silhouette:.3f}). "
                   "The clusters show reasonable separation with some overlap.")
    elif silhouette >= 0.25:
        quality = (f"<strong>Fair clustering quality</strong> (Silhouette = {silhouette:.3f}). "
                   "There is noticeable overlap. Consider trying different parameters.")
    else:
        quality = (f"<strong>Weak clustering structure</strong> (Silhouette = {silhouette:.3f}). "
                   "The data may not have clear cluster patterns, or k may need adjustment.")

    cluster_profiles = []
    all_centroids = {k: v['centroid'] for k, v in profiles.items()}
    for name, profile in profiles.items():
        centroid = profile['centroid']
        distinguishing = []
        for var in items:
            vals = [c[var] for c in all_centroids.values()]
            mu, sigma = np.mean(vals), (np.std(vals) if np.std(vals) > 0 else 1)
            z = (centroid[var] - mu) / sigma
            if abs(z) > 0.5:
                distinguishing.append(f"{'high' if z > 0 else 'low'} {var}")
        desc = (
            f"<strong>{name}</strong> ({profile['size']} obs, {profile['percentage']:.1f}%): "
            + (f"Characterized by {', '.join(distinguishing[:3])}" if distinguishing
               else "Average profile across all variables")
        )
        cluster_profiles.append(desc)

    sizes = [p['size'] for p in profiles.values()]
    ratio = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
    if ratio < 2:
        distribution = f"Clusters are <strong>well-balanced</strong> (ratio {ratio:.1f}:1)."
    elif ratio < 4:
        distribution = f"Clusters have <strong>moderate imbalance</strong> (ratio {ratio:.1f}:1)."
    else:
        distribution = (f"Clusters are <strong>highly imbalanced</strong> (ratio {ratio:.1f}:1). "
                        "Consider if this reflects true structure.")

    return {
        'overall_quality':    quality,
        'cluster_profiles':   cluster_profiles,
        'cluster_distribution': distribution,
    }
