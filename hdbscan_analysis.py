import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_selection import f_classif
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import io
import base64
import warnings

warnings.filterwarnings('ignore')

# Try to import hdbscan, fall back to DBSCAN if not available
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Warning: hdbscan not installed, using DBSCAN approximation", file=sys.stderr)

# Datasets larger than this are considered "large data" — mirrors the threshold
# convention used elsewhere in this codebase for large-N handling (e.g.
# som_analysis.py / umap_analysis.py switch behavior around a few thousand rows).
LARGE_DATA_THRESHOLD = 5000


def _fig_to_data_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def safe_float(v, default=None):
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _mad_scale(df):
    """Robust scaling via median / (MAD * 1.4826) — matches R's mad() default
    (Hampel 1974), which differs from sklearn's default IQR-based RobustScaler."""
    median = df.median()
    mad = (df - median).abs().median()
    mad_scaled = mad * 1.4826
    mad_scaled = mad_scaled.replace(0, 1.0)
    return ((df - median) / mad_scaled).values


def _apply_scaler(df, scaler_type):
    """Returns (X_scaled, scaler_name) honoring the scalerType selected in the UI."""
    if scaler_type == 'robust':
        return _mad_scale(df), 'robust'
    elif scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit_transform(df), 'minmax'
    else:
        return StandardScaler().fit_transform(df), 'standard'


def _resolve_metric_for_hdbscan(X_scaled, distance_metric):
    """The `hdbscan` package's tree-based fit only accepts metrics that are
    valid for a KD/Ball tree (euclidean, manhattan, chebyshev, canberra,
    braycurtis, ... all work natively) but rejects 'cosine' outright. R's
    dbscan::hdbscan is Euclidean-only, so for cosine we L2-normalize the
    inputs and cluster with Euclidean distance — mathematically exact and
    rank-preserving (Euclidean distance between unit vectors is a monotonic
    function of cosine distance).

    Returns (X_for_fit, actual_metric, match_r_transform, note).
    """
    if distance_metric == 'cosine':
        norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_scaled / norms
        note = (
            "Requested 'cosine' distance; the hdbscan package's tree-based fit doesn't "
            "accept cosine directly, so inputs were L2-normalized and clustered with "
            "Euclidean distance (mathematically exact, rank-preserving)."
        )
        return X_norm, 'euclidean', True, note
    return X_scaled, distance_metric, False, (
        f"'{distance_metric}' is supported natively by the hdbscan package; no transformation was needed."
    )


def estimate_eps_for_dbscan(X, min_cluster_size):
    """Estimate a good eps value for DBSCAN based on k-distance graph"""
    k = min_cluster_size - 1
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)

    # Find the "elbow" point
    # Use the 90th percentile as a reasonable eps value
    eps = np.percentile(distances, 90)
    return eps


def dbscan_with_probabilities(X, min_cluster_size, min_samples=None):
    """Use DBSCAN as a fallback with pseudo-probabilities, pseudo-persistence and
    a pseudo-outlier-score, for environments where the `hdbscan` package isn't
    installed."""
    if min_samples is None:
        min_samples = min_cluster_size

    # Estimate eps
    eps = estimate_eps_for_dbscan(X, min_cluster_size)

    # Run DBSCAN
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(X)

    # Calculate pseudo-probabilities based on distance to core points
    # This is a simple approximation
    probabilities = np.ones(len(labels))

    # For noise points, set probability to 0
    probabilities[labels == -1] = 0

    # For clustered points, calculate based on density
    for label in set(labels):
        if label != -1:
            mask = labels == label
            cluster_points = X[mask]

            # Calculate distances to cluster center
            center = cluster_points.mean(axis=0)
            distances = np.sqrt(np.sum((cluster_points - center)**2, axis=1))

            # Convert to probabilities (closer = higher probability)
            max_dist = distances.max() if distances.max() > 0 else 1
            cluster_probs = 1 - (distances / max_dist) * 0.5  # Scale to 0.5-1.0
            probabilities[mask] = cluster_probs

    # Pseudo outlier score: 1 - probability (noise gets the max score of 1.0)
    outlier_scores = 1.0 - probabilities

    # Pseudo persistence: fraction of a cluster's points with above-median
    # probability, as a crude stand-in for real GLOSH-based stability.
    persistence = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        persistence[label] = float(np.mean(probabilities[mask]))

    return labels, probabilities, outlier_scores, persistence


def _feature_drivers(raw_df, labels, items):
    """ANOVA F-test / eta-squared per feature across non-noise cluster groups."""
    mask = labels != -1
    if mask.sum() == 0:
        return None
    lc = labels[mask]
    unique = np.unique(lc)
    if len(unique) < 2:
        return None

    try:
        X = raw_df.values[mask]
        f_stats, p_values = f_classif(X, lc)

        features = []
        for i, col in enumerate(items):
            f = safe_float(f_stats[i])
            p = safe_float(p_values[i], default=1.0)

            groups = [X[lc == k, i] for k in unique]
            grand_mean = X[:, i].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
            ss_total = ((X[:, i] - grand_mean) ** 2).sum()
            eta_sq = safe_float(ss_between / ss_total) if ss_total > 0 else 0.0

            effect = 'large' if eta_sq >= 0.14 else 'medium' if eta_sq >= 0.06 else 'small'
            features.append({
                'feature': col,
                'f_stat': f,
                'p_value': p,
                'eta_squared': eta_sq,
                'effect_size': effect,
                'is_significant': bool(p is not None and p < 0.05),
                'rank': 0,
            })

        features.sort(key=lambda x: (x['eta_squared'] if x['eta_squared'] is not None else -1), reverse=True)
        for rank, feat in enumerate(features, 1):
            feat['rank'] = rank

        top_driver = features[0]['feature'] if features else None

        return {
            'features': features,
            'top_driver': top_driver,
            'note': 'ANOVA F-test (noise excluded): measures how well each variable separates the discovered clusters.',
        }
    except Exception:
        return None


def _final_metrics(X_scaled, labels, metric='euclidean'):
    mask = labels != -1
    lc = labels[mask]
    unique = np.unique(lc)
    n_clusters = len(unique)

    if n_clusters < 2 or mask.sum() < 3:
        return {
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None,
            'note': (
                'Quality metrics require at least 2 non-noise clusters (found '
                f'{n_clusters}); metrics were not computed.'
            ),
        }

    try:
        Xc = X_scaled[mask]
        return {
            'silhouette': safe_float(silhouette_score(Xc, lc, metric=metric)),
            'davies_bouldin': safe_float(davies_bouldin_score(Xc, lc)),
            'calinski_harabasz': safe_float(calinski_harabasz_score(Xc, lc)),
            'note': 'Computed on non-noise points only (in the space actually used for fitting). Davies-Bouldin/Calinski-Harabasz are always Euclidean-based (sklearn limitation).',
        }
    except Exception:
        return {
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None,
            'note': 'Quality metrics could not be computed for this configuration.',
        }


def _probability_summary(labels, probabilities, outlier_scores):
    per_cluster = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        probs = probabilities[mask]
        per_cluster[f'Cluster {label}'] = {
            'mean_probability': safe_float(probs.mean(), 0.0),
            'min_probability': safe_float(probs.min(), 0.0),
            'soft_members': int((probs < 0.5).sum()),
        }

    non_noise_mask = labels != -1
    overall_mean = safe_float(probabilities[non_noise_mask].mean(), 0.0) if non_noise_mask.any() else 0.0
    high_outlier_count = int((outlier_scores > 0.7).sum())

    return {
        'per_cluster': per_cluster,
        'overall_mean_probability': overall_mean if overall_mean is not None else 0.0,
        'high_outlier_count': high_outlier_count,
        'note': (
            f'Soft-clustering membership probabilities (mean {overall_mean:.1%} across non-noise points) '
            f'and GLOSH outlier scores; {high_outlier_count} point(s) score above the 0.7 anomaly threshold.'
        ) if overall_mean is not None else 'Membership probability summary.',
    }


def _cluster_persistence_summary(persistence_dict):
    clusters = {f'Cluster {k}': safe_float(v, 0.0) for k, v in persistence_dict.items()}
    most_stable = None
    if clusters:
        most_stable = max(clusters, key=lambda k: clusters[k] if clusters[k] is not None else -1)
    return {
        'clusters': clusters,
        'most_stable': most_stable,
        'note': (
            'Cluster persistence (excess-of-mass stability): how long each cluster survives '
            'as the density threshold rises across the condensed hierarchy. Higher = more stable.'
        ),
    }


def _param_advice(n_samples, min_cluster_size, min_samples_used, n_clusters, n_noise,
                   high_outlier_count, persistence_clusters):
    noise_pct = (n_noise / n_samples * 100) if n_samples > 0 else 0.0
    notes = []

    # ~1% of N, floor 5 — matches the heuristic documented in public/submission-code/hdbscan.py
    suggested_min_cluster_size = max(5, int(round(n_samples * 0.01)))
    suggested_min_samples = min_samples_used if min_samples_used else min_cluster_size

    if n_clusters == 0:
        notes.append('No clusters were found — try decreasing min_cluster_size.')
        suggested_min_cluster_size = max(3, int(round(min_cluster_size * 0.6)))
    elif noise_pct > 30:
        notes.append(f'High noise level ({noise_pct:.1f}%) — consider decreasing min_samples or min_cluster_size.')

    if n_clusters == 1:
        notes.append('Only one cluster was detected — consider decreasing min_cluster_size for finer granularity.')

    if high_outlier_count > 0:
        notes.append(f'{high_outlier_count} point(s) have a GLOSH outlier score above 0.7 — inspect these as candidate anomalies.')

    low_persistence = [name for name, score in persistence_clusters.items() if score is not None and score < 0.1]
    if low_persistence:
        notes.append(f'{len(low_persistence)} cluster(s) have low persistence (<0.10) and may be unstable: {", ".join(low_persistence)}.')

    summary = (
        f'Current: min_cluster_size={min_cluster_size}, min_samples={min_samples_used}, '
        f'n={n_samples}, noise={noise_pct:.1f}%. Suggested starting point: '
        f'min_cluster_size≈{suggested_min_cluster_size}, min_samples≈{suggested_min_samples}.'
    )

    return {
        'suggested_min_cluster_size': int(suggested_min_cluster_size),
        'suggested_min_samples': int(suggested_min_samples),
        'notes': notes,
        'summary': summary,
    }


def _generate_interpretations(profiles, final_metrics, summary_core, raw_df):
    interpretations = {'overall_quality': '', 'cluster_profiles': [], 'cluster_distribution': ''}

    n_clusters = summary_core['n_clusters']
    n_noise = summary_core['n_noise']
    n_samples = summary_core['n_samples']
    noise_pct = (n_noise / n_samples * 100) if n_samples > 0 else 0.0

    if n_clusters == 0:
        quality_line = "No clusters were found — every point was labeled noise. Consider decreasing min_cluster_size."
    elif n_clusters == 1:
        quality_line = "A single cluster was detected. Consider decreasing min_cluster_size for finer granularity."
    else:
        quality_line = f"{n_clusters} distinct clusters were identified. "
        quality_line += (
            "Very low noise indicates well-defined, dense clusters." if noise_pct < 5 else
            "Moderate noise with reasonably clear cluster boundaries." if noise_pct < 20 else
            "Significant noise — consider tuning min_cluster_size/min_samples."
        )

    sil = final_metrics.get('silhouette') if final_metrics else None
    if sil is not None:
        if sil >= 0.7: quality_desc = "strong and well-defined."
        elif sil >= 0.5: quality_desc = "reasonable and distinct."
        elif sil >= 0.25: quality_desc = "weak and could have some overlap."
        else: quality_desc = "not well-defined; results should be interpreted with caution."
        quality_line += f" The <strong>Silhouette Score of {sil:.3f}</strong> (noise-excluded) indicates the clustering structure is {quality_desc}"

    interpretations['overall_quality'] = quality_line

    overall_means = raw_df.mean()
    overall_std = raw_df.std().replace(0, 1)
    for name, profile in profiles.items():
        if profile.get('is_noise'):
            continue
        centroid = pd.Series(profile['centroid'])
        deviations = (centroid - overall_means) / overall_std
        top_features = deviations.nlargest(2).index.tolist()
        bottom_features = deviations.nsmallest(2).index.tolist()
        prob_txt = ''
        if profile.get('mean_probability') is not None:
            prob_txt = f" Average membership probability: {profile['mean_probability']:.1%}."
        profile_desc = (
            f"<strong>{name} ({profile['percentage']:.1f}% of data):</strong> "
            f"Characterized by high values in <strong>{', '.join(top_features)}</strong> "
            f"and low values in <strong>{', '.join(bottom_features)}</strong>.{prob_txt}"
        )
        interpretations['cluster_profiles'].append(profile_desc)

    sizes = [p['size'] for name, p in profiles.items() if not p.get('is_noise')]
    dist_desc = ''
    if len(sizes) > 1:
        ratio = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
        dist_desc = (
            "Highly imbalanced cluster sizes." if ratio > 5 else
            "Moderately imbalanced cluster sizes." if ratio > 2 else
            "Balanced cluster sizes."
        )
    if n_noise > 0:
        dist_desc += f" {n_noise} noise point(s) ({noise_pct:.1f}%)."
    interpretations['cluster_distribution'] = dist_desc.strip()

    return interpretations


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        min_cluster_size = int(payload.get('min_cluster_size', 5))
        min_samples = payload.get('min_samples')
        scaler_type = payload.get('scalerType') or 'standard'
        distance_metric = payload.get('distanceMetric') or 'euclidean'

        if not data or not items:
            raise ValueError("Missing 'data' or 'items'")

        df_raw_full = pd.DataFrame(data)[items]
        df = df_raw_full.dropna()
        n_dropped = len(df_raw_full) - len(df)

        if df.shape[0] == 0:
            raise ValueError("No valid data points for analysis.")

        input_warnings = []
        if n_dropped > 0:
            input_warnings.append(f"{n_dropped} row(s) with missing values were excluded from the analysis.")

        n_samples_ = df.shape[0]
        is_large_data = bool(n_samples_ > LARGE_DATA_THRESHOLD)

        # Standardize data (honors the UI's scalerType selection)
        X_scaled, scaler_name = _apply_scaler(df, scaler_type)

        # Resolve the requested distance metric to one the hdbscan package's
        # tree-based fit actually accepts (cosine needs an L2-normalize + euclidean
        # workaround; everything else is native).
        X_for_fit, actual_metric, match_r_transform, metric_note = _resolve_metric_for_hdbscan(
            X_scaled, distance_metric
        )

        min_samples_used = int(min_samples) if min_samples else min_cluster_size

        # Run clustering
        if HAS_HDBSCAN:
            # Use actual HDBSCAN if available
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples if min_samples else None,
                metric=actual_metric,
                gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(X_for_fit)
            probabilities = clusterer.probabilities_
            outlier_scores = clusterer.outlier_scores_
            # cluster_persistence_ is indexed 0..k-1 in the same order as the
            # real (non-noise) cluster labels HDBSCAN assigns.
            persistence = {int(i): float(v) for i, v in enumerate(clusterer.cluster_persistence_)}
            outlier_score_method = 'GLOSH (Campello et al. 2015)'
        else:
            # Fall back to DBSCAN with pseudo-probabilities
            labels, probabilities, outlier_scores, persistence = dbscan_with_probabilities(
                X_for_fit, min_cluster_size, min_samples
            )
            outlier_score_method = 'Pseudo-GLOSH approximation (hdbscan package unavailable — DBSCAN fallback)'

        # np.nan can appear in outlier_scores_ for degenerate fits; treat as 0
        outlier_scores = np.nan_to_num(outlier_scores, nan=0.0)

        # Analysis Summary
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = int(list(labels).count(-1))

        probability_summary = _probability_summary(labels, probabilities, outlier_scores)
        cluster_persistence = _cluster_persistence_summary(persistence)
        feature_drivers = _feature_drivers(df, labels, items)
        final_metrics = _final_metrics(X_for_fit, labels, metric=actual_metric)
        param_advice = _param_advice(
            n_samples_, min_cluster_size, min_samples_used, n_clusters_, n_noise_,
            probability_summary['high_outlier_count'], cluster_persistence['clusters'],
        )

        # Calculate cluster profiles
        profiles = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = (labels == label)
            cluster_data = df[mask]
            is_noise = bool(label == -1)
            cluster_name = 'Noise' if is_noise else f'Cluster {label}'

            prof_prob = None if is_noise else probability_summary['per_cluster'].get(cluster_name, {})
            profiles[cluster_name] = {
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / len(df) * 100),
                'centroid': cluster_data.mean().to_dict() if not cluster_data.empty else {},
                'is_noise': is_noise,
                'mean_probability': None if is_noise else prof_prob.get('mean_probability'),
                'min_probability': None if is_noise else prof_prob.get('min_probability'),
                'soft_members': None if is_noise else prof_prob.get('soft_members'),
                'persistence': None if is_noise else cluster_persistence['clusters'].get(cluster_name),
            }

        noise_pct = (n_noise_ / n_samples_ * 100) if n_samples_ > 0 else 0.0
        quality_warnings = []
        if n_clusters_ == 0:
            quality_warnings.append('No clusters were found — every point was classified as noise.')
        elif noise_pct > 30:
            quality_warnings.append(f'High noise level ({noise_pct:.1f}%) — consider decreasing min_samples or min_cluster_size.')
        if n_clusters_ == 1:
            quality_warnings.append('Only one cluster was detected — consider decreasing min_cluster_size for finer granularity.')

        pca_note = None
        if df.shape[1] > 2:
            pca_note = f'Data has {df.shape[1]} features; scatter plots are PCA-projected to 2D/3D for visualization.'

        # NOTE ON SHAPE: unlike dbscan-page.tsx, hdbscan-page.tsx has no flat-response
        # fallback — it reads `result.results.clustering_summary.labels` directly.
        # So clustering_summary MUST be an explicit nested key here.
        clustering_summary = {
            'n_clusters': n_clusters_,
            'n_noise': n_noise_,
            'n_samples': n_samples_,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples_used,
            'scaler': scaler_name,
            'metric': distance_metric,
            'is_large_data': is_large_data,
            'labels': labels.tolist(),
            'algorithm': 'HDBSCAN (Campello et al. 2013, deterministic)' if HAS_HDBSCAN else 'DBSCAN approximation (hdbscan package unavailable)',
            'scaler_note': (
                "Robust scaling uses median / (MAD × 1.4826) — matches R mad() default (Hampel 1974)."
                if scaler_name == 'robust' else
                "R scale() ddof=1 equivalent (StandardScaler uses population variance, but result is proportional so cluster membership is unaffected)."
                if scaler_name == 'standard' else
                "Min-max scaling to [0, 1] per feature."
            ),
            'actual_metric': actual_metric,
            'match_r_transform': match_r_transform,
            'metric_note': metric_note,
            'outlier_score_method': outlier_score_method,
        }

        interpretations = _generate_interpretations(profiles, final_metrics, clustering_summary, df)

        summary = {
            'clustering_summary': clustering_summary,
            'probabilities': probabilities.tolist(),
            'outlier_scores': outlier_scores.tolist(),
            'profiles': profiles,
            'param_advice': param_advice,
            'probability_summary': probability_summary,
            'cluster_persistence': cluster_persistence,
            'feature_drivers': feature_drivers,
            'final_metrics': final_metrics,
            'interpretations': interpretations,
            'warnings': {
                'input': input_warnings,
                'quality': quality_warnings,
                'pca_note': pca_note,
            },
        }

        # --- Plotting ---
        plots = []
        if df.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            plot_df['cluster'] = labels
            plot_df['probability'] = probabilities

            fig, ax = plt.subplots(figsize=(10, 8))

            # Use a categorical palette, handle noise points separately
            unique_labels = sorted(list(set(labels)))
            if -1 in unique_labels:
                unique_labels.remove(-1)

            if len(unique_labels) > 0:
                palette = sns.color_palette("viridis", n_colors=len(unique_labels))
                color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

                # Plot non-noise points with size based on probability
                clustered_points = plot_df[plot_df['cluster'] != -1]
                if not clustered_points.empty:
                    # Draw a translucent convex hull behind each real cluster's points
                    for label in unique_labels:
                        cluster_data = clustered_points[clustered_points['cluster'] == label]
                        if len(cluster_data) >= 3:
                            try:
                                hull_points = cluster_data[['PC1', 'PC2']].values
                                hull = ConvexHull(hull_points)
                                polygon = hull_points[hull.vertices]
                                ax.fill(
                                    polygon[:, 0], polygon[:, 1],
                                    color=color_map[label], alpha=0.15, zorder=1
                                )
                            except Exception:
                                # Collinear/degenerate point sets can't form a hull; skip.
                                pass

                    # Create sizes based on probability
                    sizes = clustered_points['probability'] * 150 + 20

                    for i, label in enumerate(unique_labels):
                        cluster_data = clustered_points[clustered_points['cluster'] == label]
                        if not cluster_data.empty:
                            ax.scatter(
                                cluster_data['PC1'],
                                cluster_data['PC2'],
                                s=sizes[clustered_points['cluster'] == label],
                                c=[color_map[label]] * len(cluster_data),
                                label=f'Cluster {label}',
                                alpha=0.7,
                                zorder=2
                            )

            # Plot noise points
            noise_points = plot_df[plot_df['cluster'] == -1]
            if not noise_points.empty:
                ax.scatter(
                    noise_points['PC1'],
                    noise_points['PC2'],
                    color='gray',
                    marker='x',
                    s=50,
                    label='Noise',
                    alpha=0.5
                )

            title = 'HDBSCAN Clustering (PCA Projection)' if HAS_HDBSCAN else 'Hierarchical Clustering Approximation (PCA Projection)'
            ax.set_title(title)
            ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()

            plots.append({'label': 'PCA Cluster Projection', 'image': _fig_to_data_url(fig)})

        # --- 3D PCA scatter (only when there are enough feature columns) ---
        if df.shape[1] >= 3:
            pca3 = PCA(n_components=3)
            X_pca3 = pca3.fit_transform(X_scaled)

            plot_df3 = pd.DataFrame(X_pca3, columns=['PC1', 'PC2', 'PC3'])
            plot_df3['cluster'] = labels
            plot_df3['probability'] = probabilities

            fig3d = plt.figure(figsize=(7, 5.5))
            ax3d = fig3d.add_subplot(111, projection='3d')

            unique_labels_3d = sorted(list(set(labels)))
            if -1 in unique_labels_3d:
                unique_labels_3d.remove(-1)

            if len(unique_labels_3d) > 0:
                palette3d = sns.color_palette("viridis", n_colors=len(unique_labels_3d))
                color_map_3d = {label: palette3d[i] for i, label in enumerate(unique_labels_3d)}

                clustered_points_3d = plot_df3[plot_df3['cluster'] != -1]
                if not clustered_points_3d.empty:
                    # Carry probability over into point size, same idea as the 2D panel
                    sizes_3d = clustered_points_3d['probability'] * 100 + 15

                    for label in unique_labels_3d:
                        cluster_data = clustered_points_3d[clustered_points_3d['cluster'] == label]
                        if not cluster_data.empty:
                            ax3d.scatter(
                                cluster_data['PC1'],
                                cluster_data['PC2'],
                                cluster_data['PC3'],
                                s=sizes_3d[clustered_points_3d['cluster'] == label],
                                c=[color_map_3d[label]] * len(cluster_data),
                                label=f'Cluster {label}',
                                alpha=0.7
                            )

            noise_points_3d = plot_df3[plot_df3['cluster'] == -1]
            if not noise_points_3d.empty:
                ax3d.scatter(
                    noise_points_3d['PC1'],
                    noise_points_3d['PC2'],
                    noise_points_3d['PC3'],
                    color='gray',
                    marker='x',
                    s=25,
                    label='Noise',
                    alpha=0.5
                )

            title_3d = 'HDBSCAN Clustering (3D PCA Projection)' if HAS_HDBSCAN else 'Hierarchical Clustering Approximation (3D PCA Projection)'
            ax3d.set_title(title_3d)
            ax3d.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%})')
            ax3d.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]:.1%})')
            ax3d.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%})')
            ax3d.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            fig3d.tight_layout()

            plots.append({'label': 'Clusters (3D PCA)', 'image': _fig_to_data_url(fig3d)})

        response = {
            'results': summary,
            'plots': plots
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
