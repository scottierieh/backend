

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
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

class KMeansAnalysis:
    def __init__(self, data, feature_cols, standardize=True):
        self.data = pd.DataFrame(data)
        self.feature_cols = feature_cols
        self.cluster_data_raw = self.data[self.feature_cols].copy().dropna()
        self.standardize = standardize

        if standardize:
            # Standardize using sample SD (ddof=1) to match R's scale() exactly, rather than
            # sklearn's StandardScaler default of population SD (ddof=0). This rescales every
            # feature by the same constant factor sqrt(n/(n-1)), so it does NOT change cluster
            # assignments or the Silhouette / Davies-Bouldin / Calinski-Harabasz scores (all
            # scale-invariant under a uniform rescale) — only the Inertia (WCSS) magnitude shifts.
            means = self.cluster_data_raw.mean()
            stds = self.cluster_data_raw.std(ddof=1).replace(0, 1)
            self.cluster_data_scaled = (self.cluster_data_raw - means) / stds
        else:
            self.cluster_data_scaled = self.cluster_data_raw.copy()

        self.n_samples, self.n_features = self.cluster_data_scaled.shape
        self.results = {}

    def find_optimal_k(self, max_k=10):
        k_range = list(range(2, min(max_k + 1, self.n_samples)))
        inertias = []
        silhouette_scores = []
        ch_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            kmeans.fit(self.cluster_data_scaled)
            inertias.append(kmeans.inertia_)
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette_scores.append(silhouette_score(self.cluster_data_scaled, kmeans.labels_))
                ch_scores.append(calinski_harabasz_score(self.cluster_data_scaled, kmeans.labels_))
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
            # Recommended k = majority vote across three independent criteria: silhouette
            # maximum, Calinski-Harabasz maximum, and the elbow point (max perpendicular
            # distance of the inertia curve from the line joining its first/last points).
            # Mirrors public/submission-code/kmeans.py exactly so the reference script and
            # this handler always agree on how k is recommended.
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
        else:
            self.results['optimal_k']['recommended_k'] = 3

        return self.results['optimal_k']

    def perform_clustering(self, n_clusters, init_method='k-means++', n_init=10, algorithm='lloyd'):
        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, n_init=n_init, random_state=42, algorithm=algorithm)
        self.cluster_labels = kmeans.fit_predict(self.cluster_data_scaled)

        self.results['clustering_summary'] = {
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'centroids': kmeans.cluster_centers_.tolist(),
            'labels': self.cluster_labels.tolist(),
            'algorithm': kmeans.algorithm,
            'init': init_method,
            'scaler': 'StandardScaler' if self.standardize else 'None',
            'scaler_note': (
                "Standardized using sample SD (ddof=1), matching R's scale() exactly — "
                "sklearn's StandardScaler defaults to population SD (ddof=0) instead. This is "
                "a uniform rescale of every feature, so it does not change cluster assignments "
                "or the Silhouette / Davies-Bouldin / Calinski-Harabasz scores — only the "
                "Inertia (WCSS) magnitude."
            ) if self.standardize else None,
            'algorithm_note': (
                "This backend uses Lloyd's algorithm (sklearn KMeans default) — a batch, "
                "coordinate-descent-style update. R's kmeans() defaults to Hartigan-Wong, a "
                "different local-search heuristic that moves individual points between clusters "
                "when it reduces WCSS. The two families can converge to different local optima, "
                "especially near ambiguous cluster boundaries."
            ),
        }

        self.analyze_clusters()
        return self.results

    def analyze_clusters(self):
        profiles = {}
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)

        for i, label in enumerate(unique_labels):
            mask = (self.cluster_labels == label)
            cluster_data = self.cluster_data_raw[mask]
            profiles[f'Cluster {label + 1}'] = {
                'size': int(counts[i]),
                'percentage': float(counts[i] / self.n_samples * 100),
                'centroid': cluster_data.mean().to_dict(),
                'centroid_scaled': self.cluster_data_scaled[mask].mean().to_dict(),
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
        """One-way ANOVA F-test per feature across the fitted clusters: how strongly does
        each variable separate the clusters? Mirrors gmm_analysis.py's _feature_drivers and
        public/submission-code/kmeans.py's ANOVA block so the three agree on methodology."""
        if len(np.unique(self.cluster_labels)) < 2:
            return None
        try:
            X = self.cluster_data_raw[self.feature_cols].to_numpy()
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
                f"Small sample size (n={n}). Cluster centroids may be unstable and not generalize well."
            )

        n_dupes = n - self.cluster_data_raw.drop_duplicates().shape[0]
        if n > 0 and n_dupes / n > 0.10:
            input_warnings.append(
                f"{n_dupes} duplicate rows ({n_dupes / n * 100:.1f}%) detected. "
                "Duplicates can distort centroids and inflate apparent cluster density."
            )

        for col in self.feature_cols:
            col_vals = self.cluster_data_raw[col]
            sd = col_vals.std(ddof=1)
            if sd and sd > 0:
                z = np.abs((col_vals - col_vals.mean()) / sd)
                extreme = int((z > 5).sum())
                if extreme > 0:
                    input_warnings.append(
                        f"'{col}' has {extreme} extreme outlier(s) (|z| > 5). "
                        "K-Means is distance-based and sensitive to outliers pulling centroids."
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
                        "May be unstable or driven by outliers."
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

        if silhouette >= 0.7:
            quality_desc = "strong and well-defined."
        elif silhouette >= 0.5:
            quality_desc = "reasonable and distinct."
        elif silhouette >= 0.25:
            quality_desc = "weak and could have some overlap."
        else:
            quality_desc = "not well-defined; results should be interpreted with caution."
        
        interpretations['overall_quality'] = (
            f"The <strong>Silhouette Score of {silhouette:.3f}</strong> indicates the clustering structure is {quality_desc}\n"
            f"Higher is better for the <strong>Calinski-Harabasz Score ({calinski:.2f})</strong>, which measures the ratio of between-cluster to within-cluster variance.\n"
            f"Lower is better for the <strong>Davies-Bouldin Score ({davies:.3f})</strong>, which measures the average similarity between clusters.\n"
            f"The <strong>Inertia (WCSS) of {inertia:.2f}</strong> represents the sum of squared distances of samples to their closest cluster center; lower is generally better."
        )

        # 2. Cluster Profile Interpretation
        overall_means = self.cluster_data_raw.mean()
        
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
            if max_p / min_p > 3:
                dist_desc = "The cluster sizes are imbalanced, with some clusters being significantly larger than others."
            else:
                dist_desc = "The clusters are relatively balanced in size."
            interpretations['cluster_distribution'] = dist_desc

        return interpretations
        
    def _fig_to_data_url(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    def _plot_clustering_process(self):
        """Manually run Lloyd's algorithm step-by-step so we can snapshot how the
        centroids/assignments evolve, then render 4 sampled iterations as one
        combined 2x2 figure (iteration 1, 3, 6, and the final converged state)."""
        X = self.cluster_data_scaled.values
        k = self.n_clusters
        seed = 42  # same random_state used elsewhere in this file for reproducibility

        try:
            init_centroids, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
        except Exception:
            rng = np.random.RandomState(seed)
            init_idx = rng.choice(X.shape[0], size=k, replace=False)
            init_centroids = X[init_idx].copy()

        # Project once upfront so points don't jump around between panels.
        pca_2d = PCA(n_components=2)
        pca_data_2d = pca_2d.fit_transform(X)

        history = []  # list of (labels, centroids) captured after each iteration's update
        centroids = init_centroids.astype(float).copy()
        max_iters = 15
        tol = 1e-6
        converged = False
        for _ in range(max_iters):
            dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(k)
            ])
            history.append((labels.copy(), new_centroids.copy()))
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < tol:
                converged = True
                break

        final_idx = len(history) - 1

        # Sample iterations 1, 3, 6 (0-indexed: 0, 2, 5), falling back to earlier
        # iterations if convergence happened sooner than that.
        candidate_iters = [1, 3, 6]
        sample_idxs = [c - 1 for c in candidate_iters if c - 1 < final_idx]
        fill = 0
        while len(sample_idxs) < 3 and fill < final_idx:
            if fill not in sample_idxs:
                sample_idxs.append(fill)
            fill += 1
        sample_idxs = sorted(set(sample_idxs))[:3]
        panel_idxs = sample_idxs + [final_idx]
        seen = set()
        panel_idxs = [i for i in panel_idxs if not (i in seen or seen.add(i))]

        final_title = 'Converged!' if converged else f'Iteration {final_idx + 1} (Max Reached)'
        titles = [
            final_title if idx == final_idx else f'Iteration {idx + 1}'
            for idx in panel_idxs
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_flat = axes.flatten()

        for ax_i, idx in enumerate(panel_idxs):
            ax = axes_flat[ax_i]
            labels_i, centroids_i = history[idx]
            centroids_proj = pca_2d.transform(centroids_i)
            ax.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], c=labels_i, cmap='viridis', s=40, alpha=0.8)
            ax.scatter(centroids_proj[:, 0], centroids_proj[:, 1], s=200, c='red', marker='X', edgecolor='black')
            ax.set_title(titles[ax_i])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)

        for ax_j in range(len(panel_idxs), 4):
            axes_flat[ax_j].axis('off')

        fig.suptitle('K-Means Clustering Process', fontsize=14)
        return self._fig_to_data_url(fig)

    def plot_results(self):
        plots = []

        # 1. Elbow Plot
        if 'optimal_k' in self.results:
            opt_k_res = self.results['optimal_k']
            fig, ax = plt.subplots(figsize=(7, 5.5))
            ax.plot(opt_k_res['k_range'], opt_k_res['inertias'], 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia (WCSS)')
            ax.set_title('Elbow Method for Optimal k')
            ax.grid(True, alpha=0.3)
            plots.append({'label': 'Elbow Method', 'image': self._fig_to_data_url(fig)})

        # 2. Silhouette Plot
        if 'optimal_k' in self.results and self.results['optimal_k']['silhouette_scores']:
            opt_k_res = self.results['optimal_k']
            fig, ax = plt.subplots(figsize=(7, 5.5))
            sns.barplot(x=opt_k_res['k_range'], y=opt_k_res['silhouette_scores'], ax=ax, color='skyblue')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Average Silhouette Score')
            ax.set_title('Silhouette Scores for Optimal k')
            ax.grid(True, alpha=0.3)
            plots.append({'label': 'Silhouette Scores', 'image': self._fig_to_data_url(fig)})

        # 3. Cluster Scatter Plot (PCA)
        if self.n_features >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(self.cluster_data_scaled)

            fig, ax = plt.subplots(figsize=(7, 5.5))

            # Draw a translucent convex hull behind each cluster's points, matching
            # the color that sns.scatterplot's viridis hue assignment will use.
            unique_labels_sorted = np.unique(self.cluster_labels)
            hull_palette = sns.color_palette('viridis', n_colors=len(unique_labels_sorted))
            label_color_map = dict(zip(unique_labels_sorted, hull_palette))
            for lbl in unique_labels_sorted:
                cluster_points = pca_data[self.cluster_labels == lbl]
                if cluster_points.shape[0] >= 3:
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_pts = cluster_points[hull.vertices]
                        ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=label_color_map[lbl], alpha=0.15)
                    except Exception:
                        pass

            sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=self.cluster_labels,
                            palette='viridis', ax=ax, legend='full')

            centroids_pca = pca.transform(self.results['clustering_summary']['centroids'])
            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='red', marker='X', label='Centroids')

            ax.set_title('Clusters in 2D PCA Space')
            ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plots.append({'label': 'Clusters (PCA)', 'image': self._fig_to_data_url(fig)})

        # 3b. Cluster Scatter Plot (3D PCA) - only when at least 3 clustering variables were used
        if len(self.feature_cols) >= 3:
            pca3 = PCA(n_components=3)
            pca3_data = pca3.fit_transform(self.cluster_data_scaled)
            centroids_pca3 = pca3.transform(self.results['clustering_summary']['centroids'])

            fig = plt.figure(figsize=(7, 5.5))
            ax3 = fig.add_subplot(111, projection='3d')
            ax3.scatter(pca3_data[:, 0], pca3_data[:, 1], pca3_data[:, 2],
                        c=self.cluster_labels, cmap='viridis', s=40, alpha=0.8)
            ax3.scatter(centroids_pca3[:, 0], centroids_pca3[:, 1], centroids_pca3[:, 2],
                        s=200, c='red', marker='X', label='Centroids')

            ax3.set_title('Clusters in 3D PCA Space')
            ax3.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%})')
            ax3.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]:.1%})')
            ax3.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%})')
            ax3.legend()
            plots.append({'label': 'Clusters (3D PCA)', 'image': self._fig_to_data_url(fig)})

        # 3c. Clustering Process - manually step through Lloyd's algorithm to snapshot convergence
        if self.n_features >= 2 and hasattr(self, 'n_clusters'):
            plots.append({'label': 'Clustering Process', 'image': self._plot_clustering_process()})

        # 4. Radar Chart of Centroids
        if 'profiles' in self.results:
            centroids = pd.DataFrame({name: profile['centroid'] for name, profile in self.results['profiles'].items()}).T
            # Normalize for radar plot
            centroids_norm = (centroids - centroids.min()) / (centroids.max() - centroids.min())

            angles = np.linspace(0, 2 * np.pi, len(self.feature_cols), endpoint=False).tolist()
            angles += angles[:1]

            fig = plt.figure(figsize=(7, 6))
            ax_radar = fig.add_subplot(111, polar=True)
            for i, (name, row) in enumerate(centroids_norm.iterrows()):
                values = row.tolist()
                values += values[:1]
                ax_radar.plot(angles, values, label=name)
                ax_radar.fill(angles, values, alpha=0.25)

            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(self.feature_cols)
            ax_radar.set_title('Cluster Profiles (Normalized)', size=12)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plots.append({'label': 'Cluster Profiles', 'image': self._fig_to_data_url(fig)})

        return plots

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        n_clusters = payload.get('nClusters')

        if not data or not items or n_clusters is None:
            raise ValueError("Missing 'data', 'items', or 'nClusters'")

        kma = KMeansAnalysis(data=data, feature_cols=items)
        kma.find_optimal_k() # Always run this to provide suggestions
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

