
import sys
import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_selection import f_classif
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_theme(style="darkgrid")
import io
import base64
import warnings

warnings.filterwarnings('ignore')

def _fig_to_data_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
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


def _eps_advice(X_scaled, min_samples, metric='euclidean'):
    """K-distance plot advice for choosing eps.

    Uses n_neighbors=min_samples+1 (column 0 is the point itself at distance 0,
    so column `min_samples` is the min_samples-th nearest neighbor excluding
    self) — this matches R's dbscan::kNNdist(k=min_samples) exactly.

    The suggested eps is the "knee" of the sorted k-distance curve, found via
    the classic max-distance-to-chord heuristic (the point on the curve
    furthest from the straight line connecting its first and last points).
    """
    n_samples = X_scaled.shape[0]
    k = int(min_samples)
    try:
        if k + 1 > n_samples:
            return {'k_distances': [], 'suggested_eps': None,
                    'note': 'Not enough points to compute a k-distance plot for the current min_samples.'}

        nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_dist = np.sort(distances[:, k])  # ascending: the classic k-distance-plot shape

        x_idx = np.arange(len(k_dist))
        x0, y0, x1, y1 = x_idx[0], k_dist[0], x_idx[-1], k_dist[-1]
        denom = np.hypot(y1 - y0, x1 - x0)
        if denom > 0:
            dist_to_chord = np.abs((y1 - y0) * x_idx - (x1 - x0) * k_dist + x1 * y0 - y1 * x0) / denom
            knee_idx = int(np.argmax(dist_to_chord))
        else:
            knee_idx = len(k_dist) // 2

        suggested_eps = safe_float(k_dist[knee_idx])

        note = (
            f"Suggested eps ≈ {suggested_eps:.4f} (elbow of the {k}-distance curve, "
            f"n_neighbors={k + 1} excludes self, matches R dbscan::kNNdist(k={k}))."
            if suggested_eps is not None else
            "Could not determine a knee point for the k-distance curve."
        )

        return {
            'k_distances': [safe_float(v, 0.0) for v in k_dist.tolist()],
            'suggested_eps': suggested_eps,
            'note': note,
        }
    except Exception:
        return {'k_distances': [], 'suggested_eps': None, 'note': 'K-distance computation failed.'}


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
        silhouette = safe_float(silhouette_score(Xc, lc, metric=metric))
        davies_bouldin = safe_float(davies_bouldin_score(Xc, lc))
        calinski = safe_float(calinski_harabasz_score(Xc, lc))
        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'note': 'Computed on non-noise points only (scaled feature space). Davies-Bouldin/Calinski-Harabasz are always Euclidean-based (sklearn limitation).',
        }
    except Exception:
        return {
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None,
            'note': 'Quality metrics could not be computed for this configuration.',
        }


def _generate_interpretations(profiles, final_metrics, clustering_summary, raw_df):
    interpretations = {'overall_quality': '', 'cluster_profiles': [], 'cluster_distribution': ''}

    n_clusters = clustering_summary['n_clusters']
    n_noise = clustering_summary['n_noise']
    n_samples = clustering_summary['n_samples']
    noise_pct = (n_noise / n_samples * 100) if n_samples > 0 else 0.0

    if n_clusters == 0:
        quality_line = "No clusters were found — every point was labeled noise. Consider increasing eps or decreasing min_samples."
    elif n_clusters == 1:
        quality_line = "A single cluster was detected. Consider decreasing eps for finer granularity."
    else:
        quality_line = f"{n_clusters} distinct clusters were identified. "
        quality_line += (
            "Very low noise indicates well-defined, dense clusters." if noise_pct < 5 else
            "Moderate noise with reasonably clear cluster boundaries." if noise_pct < 20 else
            "Significant noise — consider tuning eps/min_samples."
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
        profile_desc = (
            f"<strong>{name} ({profile['percentage']:.1f}% of data):</strong> "
            f"Characterized by high values in <strong>{', '.join(top_features)}</strong> "
            f"and low values in <strong>{', '.join(bottom_features)}</strong>."
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
        eps = float(payload.get('eps', 0.5))
        min_samples = int(payload.get('min_samples', 5))
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

        # Standardize data (honors the UI's scalerType selection)
        X_scaled, scaler_name = _apply_scaler(df, scaler_type)

        # Run DBSCAN (honors the UI's distanceMetric selection — sklearn's DBSCAN
        # supports euclidean/manhattan/cosine/chebyshev/etc. natively, no
        # transformation required)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric)
        clusters = dbscan.fit_predict(X_scaled)

        # Analysis Summary
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = int(list(labels).count(-1))
        n_samples_ = len(df)

        # Core / border point counts
        core_mask = np.zeros(n_samples_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        n_core = int(core_mask.sum())
        n_border = int(((labels != -1) & ~core_mask).sum())

        # Inertia proxy: within-cluster sum of squared distances to centroid (scaled space)
        inertia_proxy = 0.0
        for lab in set(labels):
            if lab == -1:
                continue
            pts = X_scaled[labels == lab]
            inertia_proxy += float(((pts - pts.mean(axis=0)) ** 2).sum())

        # Calculate cluster profiles
        profiles = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = (labels == label)
            cluster_data = df[mask]
            is_noise = bool(label == -1)
            cluster_name = 'Noise' if is_noise else f'Cluster {label}'

            profiles[cluster_name] = {
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / len(df) * 100),
                'centroid': cluster_data.mean().to_dict(),
                'is_noise': is_noise,
            }

        eps_advice = _eps_advice(X_scaled, min_samples, metric=distance_metric)
        feature_drivers = _feature_drivers(df, labels, items)
        final_metrics = _final_metrics(X_scaled, labels, metric=distance_metric)

        noise_pct = (n_noise_ / n_samples_ * 100) if n_samples_ > 0 else 0.0
        quality_warnings = []
        if n_clusters_ == 0:
            quality_warnings.append('No clusters were found — every point was classified as noise.')
        elif noise_pct > 30:
            quality_warnings.append(f'High noise level ({noise_pct:.1f}%) — consider increasing eps or decreasing min_samples.')
        if n_clusters_ == 1:
            quality_warnings.append('Only one cluster was detected — consider decreasing eps for finer granularity.')

        pca_note = None
        if df.shape[1] > 2:
            pca_note = f'Data has {df.shape[1]} features; scatter plots are PCA-projected to 2D/3D for visualization.'

        # NOTE ON SHAPE: the frontend normalizes whatever this script returns via
        # `rr.clustering_summary ?? (Array.isArray(rr.labels) ? rr : undefined)`
        # — i.e. if there's no explicit `clustering_summary` key, it treats the
        # entire flat `results` object AS the clustering_summary. We deliberately
        # keep everything flat (no artificial nesting) to match that contract,
        # exactly like the original implementation did.
        summary_core = {
            'n_clusters': n_clusters_,
            'n_noise': n_noise_,
            'n_samples': n_samples_,
            'n_core': n_core,
            'n_border': n_border,
            'eps': eps,
            'min_samples': min_samples,
            'scaler': scaler_name,
            'metric': distance_metric,
            'labels': labels.tolist(),
            'inertia_proxy': safe_float(inertia_proxy, 0.0),
            'algorithm': 'DBSCAN (Ester et al. 1996, deterministic)',
            'scaler_note': (
                "Robust scaling uses median / (MAD × 1.4826) — matches R mad() default (Hampel 1974)."
                if scaler_name == 'robust' else
                "R scale() ddof=1 equivalent (StandardScaler uses population variance, but result is proportional so cluster membership is unaffected)."
                if scaler_name == 'standard' else
                "Min-max scaling to [0, 1] per feature."
            ),
            'knn_distance_note': 'k-distance uses n_neighbors=min_samples+1 (excludes self) — matches R dbscan::kNNdist(k=min_samples) exactly.',
        }

        interpretations = _generate_interpretations(
            profiles, final_metrics, summary_core, df
        )

        summary = {
            **summary_core,
            'profiles': profiles,
            'eps_advice': eps_advice,
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

            fig, ax = plt.subplots(figsize=(7, 5.5))

            # Use a categorical palette, handle noise points separately
            unique_labels = sorted(list(set(labels)))
            palette = sns.color_palette("viridis", n_colors=len(unique_labels) - (1 if -1 in unique_labels else 0))

            label_colors = {}
            for i, label in enumerate(unique_labels):
                if label == -1:
                    # Noise points
                    sns.scatterplot(
                        x=plot_df[plot_df['cluster'] == label]['PC1'],
                        y=plot_df[plot_df['cluster'] == label]['PC2'],
                        color='gray',
                        marker='x',
                        s=50,
                        label='Noise',
                        ax=ax
                    )
                else:
                    color = palette[i - (1 if -1 in unique_labels else 0)]
                    label_colors[label] = color
                    sns.scatterplot(
                        x=plot_df[plot_df['cluster'] == label]['PC1'],
                        y=plot_df[plot_df['cluster'] == label]['PC2'],
                        color=color,
                        label=f'Cluster {label}',
                        s=80,
                        alpha=0.7,
                        ax=ax
                    )

            # Convex hulls around each real cluster (skip noise; need >= 3 points to form a hull)
            for label, color in label_colors.items():
                cluster_points = plot_df.loc[plot_df['cluster'] == label, ['PC1', 'PC2']].values
                if cluster_points.shape[0] < 3:
                    continue
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.15)
                except Exception:
                    # Collinear or degenerate point sets can't form a hull; skip silently
                    pass

            ax.set_title('DBSCAN Clustering (PCA Projection)')
            ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.legend(title='Cluster')
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()

            plots.append({'label': 'Cluster Scatter (PCA Projection)', 'image': _fig_to_data_url(fig)})

        if df.shape[1] >= 3:
            pca3 = PCA(n_components=3)
            X_pca3 = pca3.fit_transform(X_scaled)

            fig3d = plt.figure(figsize=(7, 5.5))
            ax3d = fig3d.add_subplot(111, projection='3d')

            unique_labels_3d = sorted(list(set(labels)))
            n_real_clusters = len(unique_labels_3d) - (1 if -1 in unique_labels_3d else 0)
            palette3d = sns.color_palette("viridis", n_colors=max(n_real_clusters, 1))

            real_idx = 0
            for label in unique_labels_3d:
                mask = (labels == label)
                if label == -1:
                    ax3d.scatter(
                        X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2],
                        c='gray', marker='x', s=30, label='Noise', alpha=0.6
                    )
                else:
                    ax3d.scatter(
                        X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2],
                        color=palette3d[real_idx], marker='o', s=50, label=f'Cluster {label}', alpha=0.7
                    )
                    real_idx += 1

            ax3d.set_title('DBSCAN Clustering (3D PCA Projection)')
            ax3d.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%})')
            ax3d.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]:.1%})')
            ax3d.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%})')
            ax3d.legend(title='Cluster', loc='best', fontsize='small')
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
