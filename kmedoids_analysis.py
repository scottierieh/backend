
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import kmeans_plusplus
from sklearn_extra.cluster import KMedoids
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
import warnings
import io
import base64

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

_SCALER_CLASSES = {
    'standard': ('StandardScaler', StandardScaler),
    'robust': ('RobustScaler', RobustScaler),
    'minmax': ('MinMaxScaler', MinMaxScaler),
}


class KMedoidsAnalysis:
    def __init__(self, data, feature_cols, standardize=True, scaler_type='standard', distance_metric='euclidean'):
        self.data = pd.DataFrame(data)
        self.feature_cols = feature_cols
        self.cluster_data = self.data[self.feature_cols].copy().dropna()
        self.standardize = standardize
        # Actually honor the frontend's scaler/metric selectors (scalerType, distanceMetric)
        # instead of silently hardcoding StandardScaler + euclidean regardless of user choice.
        self.scaler_name, scaler_cls = _SCALER_CLASSES.get(scaler_type, _SCALER_CLASSES['standard'])
        self.distance_metric = distance_metric or 'euclidean'

        if standardize:
            scaler = scaler_cls()
            self.cluster_data_scaled = pd.DataFrame(scaler.fit_transform(self.cluster_data), columns=self.feature_cols, index=self.cluster_data.index)
        else:
            self.cluster_data_scaled = self.cluster_data.copy()

        self.n_samples, self.n_features = self.cluster_data_scaled.shape
        self.results = {}

    def find_optimal_k(self, max_k=10):
        """Elbow (PAM cost) + Silhouette + Calinski-Harabasz curves across candidate k,
        mirroring KMeansAnalysis.find_optimal_k exactly (same shape, same 3-way vote for
        recommended_k) so the app's claim that K-Medoids offers the same k-selection tools
        as K-Means is actually true."""
        k_range = list(range(2, min(max_k + 1, self.n_samples)))
        inertias = []
        silhouette_scores = []
        ch_scores = []

        for k in k_range:
            model = KMedoids(n_clusters=k, method='pam', init='k-medoids++', max_iter=300, random_state=42, metric=self.distance_metric)
            model.fit(self.cluster_data_scaled)
            inertias.append(model.inertia_)
            if len(np.unique(model.labels_)) > 1:
                silhouette_scores.append(silhouette_score(self.cluster_data_scaled, model.labels_))
                ch_scores.append(calinski_harabasz_score(self.cluster_data_scaled, model.labels_))
            else:
                silhouette_scores.append(-1)
                ch_scores.append(-1)

        self.results['optimal_k'] = {
            'k_range': k_range,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'ch_scores': ch_scores,
        }

        if k_range:
            silhouette_k = k_range[int(np.argmax(silhouette_scores))]
            ch_k = k_range[int(np.argmax(ch_scores))]
            x = np.array(k_range, dtype=float)
            y = np.array(inertias, dtype=float)
            x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
            num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
            den = np.hypot(y1 - y0, x1 - x0)
            elbow_k = int(x[int(np.argmax(num / den))]) if den > 0 else k_range[0]
            votes = [silhouette_k, ch_k, elbow_k]
            recommended_k = max(set(votes), key=votes.count)
            self.results['optimal_k']['recommended_k'] = recommended_k
            self.results['optimal_k']['recommendation_detail'] = {
                'recommended_k': recommended_k,
                'votes': {
                    'silhouette_k': silhouette_k,
                    'calinski_harabasz_k': ch_k,
                    'elbow_k': elbow_k,
                },
                'note': (
                    f"k={recommended_k} chosen by majority vote across three criteria: "
                    f"silhouette maximum (k={silhouette_k}), Calinski-Harabasz maximum (k={ch_k}), "
                    f"and the elbow point of the PAM cost curve (k={elbow_k})."
                ),
            }
        else:
            self.results['optimal_k']['recommended_k'] = 3

        return self.results['optimal_k']

    def perform_clustering(self, n_clusters, init_method='k-medoids++', max_iter=300):
        self.n_clusters = n_clusters

        kmedoids = KMedoids(n_clusters=n_clusters, method='pam', init=init_method, max_iter=max_iter, random_state=42, metric=self.distance_metric)
        self.cluster_labels = kmedoids.fit_predict(self.cluster_data_scaled)
        self.medoid_indices_ = kmedoids.medoid_indices_

        self.results['clustering_summary'] = {
            'n_clusters': n_clusters,
            'inertia': kmedoids.inertia_,
            'medoids': self.cluster_data.iloc[kmedoids.medoid_indices_].to_dict('records'),
            'medoid_indices': kmedoids.medoid_indices_.tolist(),
            'labels': self.cluster_labels.tolist(),
            'algorithm': 'PAM',
            'scaler': self.scaler_name if self.standardize else 'None',
            'metric': kmedoids.metric,
        }

        self.analyze_clusters()
        return self.results

    def analyze_clusters(self):
        profiles = {}
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)

        for i, label in enumerate(unique_labels):
            mask = (self.cluster_labels == label)
            cluster_data = self.cluster_data[mask]
            medoid_idx = (
                int(self.medoid_indices_[label])
                if hasattr(self, 'medoid_indices_') and label < len(self.medoid_indices_)
                else None
            )
            profiles[f'Cluster {label + 1}'] = {
                'size': int(counts[i]),
                'percentage': float(counts[i] / self.n_samples * 100),
                'centroid': cluster_data.mean().to_dict(), # Using mean as centroid for profile summary
                'medoid_scaled': self.cluster_data_scaled.iloc[medoid_idx].to_dict() if medoid_idx is not None else None,
                'medoid_index': medoid_idx,
            }
        self.results['profiles'] = profiles

        if len(unique_labels) > 1:
            self.results['final_metrics'] = {
                'silhouette': silhouette_score(self.cluster_data_scaled, self.cluster_labels),
                'davies_bouldin': davies_bouldin_score(self.cluster_data_scaled, self.cluster_labels),
                'calinski_harabasz': calinski_harabasz_score(self.cluster_data_scaled, self.cluster_labels),
            }

            sil_samples = silhouette_samples(self.cluster_data_scaled, self.cluster_labels)
            per_cluster_sil = []
            for i, label in enumerate(unique_labels):
                mask = (self.cluster_labels == label)
                s = sil_samples[mask]
                per_cluster_sil.append({
                    'cluster': f'Cluster {label + 1}',
                    'n': int(counts[i]),
                    'avg_sil': float(s.mean()),
                    'min_sil': float(s.min()),
                    'max_sil': float(s.max()),
                    'negative_sil': int((s < 0).sum()),
                })
            self.results['per_cluster_silhouette'] = per_cluster_sil

        self.results['feature_drivers'] = self._compute_feature_drivers()
        self.results['warnings'] = self._generate_warnings()
        self.results['interpretations'] = self.generate_interpretations()

    def _compute_feature_drivers(self):
        """One-way ANOVA F-test per feature across the fitted clusters — identical
        methodology to KMeansAnalysis._compute_feature_drivers and
        public/submission-code/kmedoids.py's ANOVA block."""
        if len(np.unique(self.cluster_labels)) < 2:
            return None
        try:
            X = self.cluster_data[self.feature_cols].to_numpy()
            f_stats, p_values = f_classif(X, self.cluster_labels)

            features = []
            grand_mean = X.mean(axis=0)
            for i, col in enumerate(self.feature_cols):
                f = float(f_stats[i]) if np.isfinite(f_stats[i]) else 0.0
                p = float(p_values[i]) if np.isfinite(p_values[i]) else 1.0

                groups = [X[self.cluster_labels == k, i] for k in np.unique(self.cluster_labels)]
                ss_between = sum(len(g) * (g.mean() - grand_mean[i]) ** 2 for g in groups if len(g) > 0)
                ss_total = ((X[:, i] - grand_mean[i]) ** 2).sum()
                eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

                effect = 'large' if eta_sq >= 0.14 else 'medium' if eta_sq >= 0.06 else 'small'
                features.append({
                    'feature': col,
                    'f_stat': f,
                    'p_value': p,
                    'eta_squared': eta_sq,
                    'effect_size': effect,
                    'is_significant': bool(p < 0.05),
                    'rank': 0,
                })

            features.sort(key=lambda x: x['eta_squared'], reverse=True)
            for rank, feat in enumerate(features, 1):
                feat['rank'] = rank

            top_driver = features[0]['feature'] if features else None
            return {
                'features': features,
                'top_driver': top_driver,
                'note': 'ANOVA F-test: measures how well each variable separates the k clusters. Higher F / eta-squared = stronger driver.',
            }
        except Exception:
            return None

    def _generate_warnings(self):
        input_warnings = []
        quality_warnings = []

        n = self.n_samples
        if n < 30:
            input_warnings.append(
                f"Small sample size (n={n}). Medoid selection may be unstable and not generalize well."
            )

        n_dupes = n - self.cluster_data.drop_duplicates().shape[0]
        if n > 0 and n_dupes / n > 0.10:
            input_warnings.append(
                f"{n_dupes} duplicate rows ({n_dupes / n * 100:.1f}%) detected. "
                "Duplicates can bias which point ends up selected as the medoid."
            )

        for col in self.feature_cols:
            col_vals = self.cluster_data[col]
            sd = col_vals.std(ddof=1)
            if sd and sd > 0:
                z = np.abs((col_vals - col_vals.mean()) / sd)
                extreme = int((z > 5).sum())
                if extreme > 0:
                    input_warnings.append(
                        f"'{col}' has {extreme} extreme outlier(s) (|z| > 5). K-Medoids is more "
                        "robust than K-Means here, but extreme values can still become medoids."
                    )

        if n < self.n_features * 10:
            quality_warnings.append(
                f"{self.n_features} feature(s) relative to n={n} observations is high-dimensional "
                "for clustering — distances become less meaningful (curse of dimensionality). "
                "Consider dimensionality reduction (e.g. PCA) first."
            )

        if 'profiles' in self.results:
            for name, p in self.results['profiles'].items():
                if p['percentage'] < 5.0:
                    quality_warnings.append(
                        f"{name} is very small ({p['percentage']:.1f}% of data, n={p['size']}). "
                        "May be unstable or represent an outlier acting as its own medoid."
                    )
            percentages = [p['percentage'] for p in self.results['profiles'].values()]
            if len(percentages) > 1 and min(percentages) > 0 and max(percentages) / min(percentages) > 3:
                quality_warnings.append(
                    "Cluster sizes are highly imbalanced (largest is >3x the smallest). "
                    "Consider whether k is appropriate or whether outliers are forming tiny clusters."
                )

        if 'final_metrics' in self.results and 0 < self.results['final_metrics']['silhouette'] < 0.25:
            quality_warnings.append(
                f"Silhouette score ({self.results['final_metrics']['silhouette']:.3f}) is low. "
                "Clusters may overlap substantially — treat cluster assignments with caution."
            )

        pca_note = None
        if self.n_features > 2:
            try:
                pca = PCA(n_components=2)
                pca.fit(self.cluster_data_scaled)
                var_explained = float(pca.explained_variance_ratio_.sum())
                pca_note = (
                    f"Clustering was performed in the full {self.n_features}-dimensional feature "
                    f"space. The 2D scatter plot is a PCA projection explaining {var_explained:.1%} "
                    "of total variance — visual overlap there does not necessarily mean overlap in "
                    "the original feature space."
                )
            except Exception:
                pca_note = None

        return {
            'input': input_warnings,
            'quality': quality_warnings,
            'pca_note': pca_note,
        }

    def generate_interpretations(self):
        if 'profiles' not in self.results or 'final_metrics' not in self.results:
            return {}

        interpretations = {
            'overall_quality': '',
            'cluster_profiles': [],
            'cluster_distribution': ''
        }

        # 1. Overall Quality Interpretation
        metrics = self.results['final_metrics']
        silhouette = metrics['silhouette']
        calinski = metrics['calinski_harabasz']
        davies = metrics['davies_bouldin']
        inertia = self.results['clustering_summary']['inertia']

        if silhouette >= 0.7: quality_desc = "strong and well-defined."
        elif silhouette >= 0.5: quality_desc = "reasonable and distinct."
        elif silhouette >= 0.25: quality_desc = "weak and could have some overlap."
        else: quality_desc = "not well-defined; results should be interpreted with caution."
        
        interpretations['overall_quality'] = (
            f"The **Silhouette Score of {silhouette:.3f}** indicates the clustering structure is {quality_desc}\n"
            f"The **Calinski-Harabasz Score** ({calinski:.2f}, higher is better) measures the ratio of between-cluster to within-cluster variance.\n"
            f"The **Davies-Bouldin Score** ({davies:.3f}, closer to 0 is better) measures the average similarity between clusters.\n"
            f"The **Inertia of {inertia:.2f}** represents the sum of squared distances of samples to their closest cluster medoid; lower is better."
        )

        # 2. Cluster Profile Interpretation
        overall_means = self.cluster_data.mean()
        
        for name, profile in self.results['profiles'].items():
            centroid = pd.Series(profile['centroid'])
            deviations = (centroid - overall_means) / overall_means.std()
            
            top_features = deviations.nlargest(2).index.tolist()
            bottom_features = deviations.nsmallest(2).index.tolist()
            
            profile_desc = f"<strong>{name} ({profile['percentage']:.1f}% of data):</strong> This cluster is characterized by high values in <strong>{', '.join(top_features)}</strong> and low values in <strong>{', '.join(bottom_features)}</strong>."
            interpretations['cluster_profiles'].append(profile_desc)

        # 3. Cluster Distribution Interpretation
        percentages = [p['percentage'] for p in self.results['profiles'].values()]
        if len(percentages) > 1:
            max_p = max(percentages)
            min_p = min(percentages)
            if max_p / min_p > 3: dist_desc = "The cluster sizes are imbalanced, with some clusters being significantly larger than others."
            else: dist_desc = "The clusters are relatively balanced in size."
            interpretations['cluster_distribution'] = dist_desc

        return interpretations

    def _fig_to_data_url(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    def _simulate_clustering_process(self, pca_2d_data):
        """
        Approximate, self-contained PAM-style iteration used ONLY to render the
        'Clustering Process' visualization. This does NOT feed back into
        self.results / self.cluster_labels / the medoid indices used elsewhere
        in the file - it is a lightweight re-derivation purely for the plot.

        Approximation notes:
          - Initial medoids are seeded via k-means++ (on the actual scaled data
            points) and each seed is snapped to the nearest real data point.
          - Each iteration reassigns points to their nearest current medoid,
            then for every cluster picks the *actual data point closest to the
            cluster mean* as the new medoid. This is a fast stand-in for the
            real PAM swap step (which would test every candidate point for the
            total-cost-minimizing swap) - cheap enough to run per-iteration
            purely for illustrating convergence, not for the reported result.
        """
        X = self.cluster_data_scaled.values
        n_samples = X.shape[0]
        k = self.n_clusters

        seeds, _ = kmeans_plusplus(X, n_clusters=k, random_state=42)
        medoid_idx = []
        for seed in seeds:
            dists = np.linalg.norm(X - seed, axis=1)
            for cand in np.argsort(dists):
                if cand not in medoid_idx:
                    medoid_idx.append(int(cand))
                    break
        medoid_idx = np.array(medoid_idx)

        history = []
        max_iter = 8
        for iteration in range(1, max_iter + 1):
            medoid_points = X[medoid_idx]
            dists_to_medoids = np.linalg.norm(X[:, None, :] - medoid_points[None, :, :], axis=2)
            labels = np.argmin(dists_to_medoids, axis=1)
            history.append({'iter': iteration, 'medoid_idx': medoid_idx.copy(), 'labels': labels.copy()})

            new_medoid_idx = medoid_idx.copy()
            for ci in range(k):
                members = np.where(labels == ci)[0]
                if len(members) == 0:
                    continue
                cluster_mean = X[members].mean(axis=0)
                local_dists = np.linalg.norm(X[members] - cluster_mean, axis=1)
                new_medoid_idx[ci] = members[np.argmin(local_dists)]

            if np.array_equal(new_medoid_idx, medoid_idx):
                break  # converged - medoids stopped changing
            medoid_idx = new_medoid_idx

        n_hist = len(history)
        raw_targets = [1, 3, 6, n_hist]
        picked = []
        for t in raw_targets:
            i = min(t, n_hist) - 1
            if i not in picked:
                picked.append(i)
        while len(picked) < 4:
            picked.append(n_hist - 1)
        picked = picked[:4]

        snapshots = [history[i] for i in picked]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_flat = axes.flatten()
        for pos, (ax_p, snap) in enumerate(zip(axes_flat, snapshots)):
            ax_p.scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], c=snap['labels'],
                        cmap='viridis', s=30, alpha=0.6)
            medoid_pts = pca_2d_data[snap['medoid_idx']]
            ax_p.scatter(medoid_pts[:, 0], medoid_pts[:, 1], s=200, c='red', marker='X',
                        edgecolors='black', linewidths=1.5, zorder=5)

            title = 'Converged!' if pos == len(snapshots) - 1 else f"Iteration {snap['iter']}"
            ax_p.set_title(title, fontsize=12, fontweight='bold')
            ax_p.set_xlabel('PC1')
            ax_p.set_ylabel('PC2')
            ax_p.grid(True, alpha=0.3)

        fig.suptitle('K-Medoids (PAM) Clustering Process', fontsize=14)
        plt.tight_layout()
        return self._fig_to_data_url(fig)

    def plot_results(self):
        plots = []

        # 0a. Elbow Plot (PAM cost vs k)
        if 'optimal_k' in self.results:
            opt_k_res = self.results['optimal_k']
            fig, ax = plt.subplots(figsize=(7, 5.5))
            ax.plot(opt_k_res['k_range'], opt_k_res['inertias'], 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Total Cost (Inertia)')
            ax.set_title('Elbow Method for Optimal k')
            ax.grid(True, alpha=0.3)
            plots.append({'label': 'Elbow Method', 'image': self._fig_to_data_url(fig)})

        # 0b. Silhouette Plot
        if 'optimal_k' in self.results and self.results['optimal_k']['silhouette_scores']:
            opt_k_res = self.results['optimal_k']
            fig, ax = plt.subplots(figsize=(7, 5.5))
            sns.barplot(x=opt_k_res['k_range'], y=opt_k_res['silhouette_scores'], ax=ax, color='skyblue')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Average Silhouette Score')
            ax.set_title('Silhouette Scores for Optimal k')
            ax.grid(True, alpha=0.3)
            plots.append({'label': 'Silhouette Scores', 'image': self._fig_to_data_url(fig)})

        pca_2d = None
        pca_2d_data = None

        # 1. Cluster Scatter Plot (PCA)
        if self.n_features >= 2:
            fig, ax = plt.subplots(figsize=(7, 5.5))

            pca_2d = PCA(n_components=2)
            pca_2d_data = pca_2d.fit_transform(self.cluster_data_scaled)

            sns.scatterplot(x=pca_2d_data[:, 0], y=pca_2d_data[:, 1], hue=self.cluster_labels,
                            palette='viridis', ax=ax, legend='full', s=50, alpha=0.7)

            # Convex hulls per cluster, colored to match the scatter's viridis palette
            unique_cluster_labels = sorted(np.unique(self.cluster_labels).tolist())
            hull_palette = sns.color_palette('viridis', n_colors=len(unique_cluster_labels))
            hull_color_map = dict(zip(unique_cluster_labels, hull_palette))
            for label in unique_cluster_labels:
                mask = self.cluster_labels == label
                cluster_pts = pca_2d_data[mask]
                if cluster_pts.shape[0] < 3:
                    continue
                try:
                    hull = ConvexHull(cluster_pts)
                    hull_pts = cluster_pts[hull.vertices]
                    ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.15, color=hull_color_map[label])
                except Exception:
                    pass  # collinear or degenerate points - skip hull for this cluster

            medoids_pca = pca_2d.transform(self.cluster_data_scaled.iloc[self.results['clustering_summary']['medoid_indices']])

            ax.scatter(medoids_pca[:, 0], medoids_pca[:, 1], s=250, c='red', marker='X', label='Medoids', edgecolors='black')

            ax.set_title('Clusters in 2D PCA Space')
            ax.set_xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plots.append({'label': 'Clusters (PCA)', 'image': self._fig_to_data_url(fig)})

        # 1b. Cluster Scatter Plot (3D PCA) - only when there are enough feature columns
        if self.n_features >= 3:
            fig = plt.figure(figsize=(7, 5.5))
            ax3d = fig.add_subplot(111, projection='3d')

            pca_3d = PCA(n_components=3)
            pca_3d_data = pca_3d.fit_transform(self.cluster_data_scaled)

            ax3d.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2],
                        c=self.cluster_labels, cmap='viridis', s=40, alpha=0.7)

            medoids_pca_3d = pca_3d.transform(self.cluster_data_scaled.iloc[self.results['clustering_summary']['medoid_indices']])
            ax3d.scatter(medoids_pca_3d[:, 0], medoids_pca_3d[:, 1], medoids_pca_3d[:, 2],
                        s=250, c='red', marker='X', label='Medoids', edgecolors='black', linewidths=1.2)

            ax3d.set_title('Clusters in 3D PCA Space')
            ax3d.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
            ax3d.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
            ax3d.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
            ax3d.legend()

            plt.tight_layout()
            plots.append({'label': 'Clusters (3D PCA)', 'image': self._fig_to_data_url(fig)})

        # 1c. Clustering Process (K-Medoids/PAM convergence) - combined 2x2 snapshot grid.
        # Purely illustrative: derived from a separate, approximate simulation and
        # does not alter self.results, self.cluster_labels, or the reported medoids.
        if pca_2d_data is not None:
            try:
                process_image = self._simulate_clustering_process(pca_2d_data)
                plots.append({'label': 'Clustering Process', 'image': process_image})
            except Exception:
                pass  # visualization-only helper - never let it break the main response

        # 2. Radar Chart of Medoids
        if 'profiles' in self.results:
            medoids_df = pd.DataFrame(self.results['clustering_summary']['medoids'])
            if not medoids_df.empty:
                min_vals = self.cluster_data[self.feature_cols].min()
                max_vals = self.cluster_data[self.feature_cols].max()
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1 # Avoid division by zero

                medoids_norm = (medoids_df[self.feature_cols] - min_vals) / range_vals

                angles = np.linspace(0, 2 * np.pi, len(self.feature_cols), endpoint=False).tolist()
                angles += angles[:1]

                fig = plt.figure(figsize=(7, 5.5))
                ax_radar = fig.add_subplot(111, polar=True)
                for i, (idx, row) in enumerate(medoids_norm.iterrows()):
                    values = row.tolist()
                    values += values[:1]
                    ax_radar.plot(angles, values, label=f'Cluster {i+1}')
                    ax_radar.fill(angles, values, alpha=0.25)

                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(self.feature_cols)
                ax_radar.set_title('Cluster Medoids (Normalized)', size=12)
                ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

                plt.tight_layout()
                plots.append({'label': 'Cluster Medoids (Radar Chart)', 'image': self._fig_to_data_url(fig)})

        return plots

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        n_clusters = payload.get('nClusters')
        scaler_type = payload.get('scalerType') or 'standard'
        distance_metric = payload.get('distanceMetric') or 'euclidean'

        if not data or not items or n_clusters is None:
            raise ValueError("Missing 'data', 'items', or 'nClusters'")

        kma = KMedoidsAnalysis(data=data, feature_cols=items, scaler_type=scaler_type, distance_metric=distance_metric)
        kma.find_optimal_k()  # Always run this to provide elbow/silhouette/CH suggestions, mirroring K-Means
        kma.perform_clustering(n_clusters=n_clusters)
        
        plots = kma.plot_results()

        response = {
            'results': kma.results,
            'plots': plots
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
