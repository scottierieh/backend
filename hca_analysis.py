import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
import warnings
import io
import base64
import math

warnings.filterwarnings('ignore')

# Set seaborn style globally
sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

# Above this many observations, computing the full O(n^2) pairwise-distance matrix
# for linkage becomes prohibitively slow/memory-heavy. Instead we build the
# dendrogram from a random subsample and assign the remaining points to the
# nearest subsample-cluster centroid (see `is_large_data` in clustering_summary).
LARGE_DATA_THRESHOLD = 5000
SUBSAMPLE_SIZE = 2000


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _safe_float(val, default=0.0):
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _feature_drivers(cluster_data_raw, labels, feature_cols):
    """One-way ANOVA (F-test) of each raw feature across the final cluster
    labels, with eta-squared effect size. Mirrors gmm_analysis.py's
    _feature_drivers so the methodology (and Cohen thresholds) is identical
    across the clustering scripts in this codebase."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None

    try:
        X = cluster_data_raw[feature_cols].values
        f_stats, p_values = f_classif(X, labels)

        features = []
        for i, col in enumerate(feature_cols):
            f = _safe_float(f_stats[i])
            p = _safe_float(p_values[i], default=1.0)

            groups = [X[labels == k, i] for k in np.unique(labels)]
            grand_mean = X[:, i].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
            ss_total = ((X[:, i] - grand_mean) ** 2).sum()
            eta_sq = _safe_float(ss_between / ss_total) if ss_total > 0 else 0.0

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
            'note': 'ANOVA F-test: measures how well each variable separates the hierarchical clusters.',
        }
    except Exception:
        return None

class HierarchicalClusterAnalysis:
    def __init__(self, data, feature_cols=None, scaler_type='standard'):
        self.data = pd.DataFrame(data)
        self.scaler_type = (scaler_type or 'standard').lower()
        if self.scaler_type not in ('standard', 'robust', 'minmax'):
            self.scaler_type = 'standard'
        self.feature_cols = feature_cols if feature_cols is not None else self.data.select_dtypes(include=np.number).columns.tolist()

        self.cluster_data = self.data[self.feature_cols].copy().dropna()

        if self.scaler_type == 'robust':
            scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        self.cluster_data_scaled = pd.DataFrame(scaler.fit_transform(self.cluster_data), columns=self.feature_cols, index=self.cluster_data.index)

        self.n_samples, self.n_features = self.cluster_data_scaled.shape
        self.is_large_data = self.n_samples > LARGE_DATA_THRESHOLD
        self.subsample_index = None
        self._basis_data = None
        self.results = {}

    def perform_clustering(self, linkage_method='ward', distance_metric='euclidean', n_clusters=None):
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric

        # ── Large-data guard: build the linkage/dendrogram from a bounded random
        # subsample instead of the full O(n^2) pairwise-distance matrix. ────────
        if self.is_large_data:
            rng = np.random.RandomState(42)
            subsample_n = min(SUBSAMPLE_SIZE, self.n_samples)
            sub_positions = rng.choice(self.n_samples, size=subsample_n, replace=False)
            self.subsample_index = self.cluster_data_scaled.index[sub_positions]
            basis = self.cluster_data_scaled.loc[self.subsample_index]
        else:
            basis = self.cluster_data_scaled
        self._basis_data = basis

        if linkage_method == 'ward' and distance_metric != 'euclidean':
            warnings.warn("Ward linkage requires euclidean distance. Overriding distance_metric to 'euclidean'.")
            self.distance_metric = 'euclidean'
            distances = pdist(basis, metric='euclidean')
            self.linkage_matrix = linkage(distances, method='ward')
        elif linkage_method == 'ward':
            distances = pdist(basis, metric='euclidean')
            self.linkage_matrix = linkage(distances, method='ward')
        else:
            distances = pdist(basis, metric=self.distance_metric)
            self.linkage_matrix = linkage(distances, method=linkage_method)

        if n_clusters is None:
            recommendations = self._find_optimal_clusters(basis)
            n_clusters = recommendations.get('silhouette', 3) # Default to 3 if silhouette fails

        self.n_clusters = n_clusters
        basis_labels = fcluster(self.linkage_matrix, t=n_clusters, criterion='maxclust')

        if self.is_large_data:
            # Assign every observation (including the basis points themselves, for
            # consistency) to the nearest basis-cluster centroid in scaled space.
            unique_basis_labels = sorted(np.unique(basis_labels).tolist())
            centroid_matrix = np.array([
                basis.values[basis_labels == lbl].mean(axis=0) for lbl in unique_basis_labels
            ])
            all_points = self.cluster_data_scaled.values
            dists = np.linalg.norm(all_points[:, np.newaxis, :] - centroid_matrix[np.newaxis, :, :], axis=2)
            nearest = np.argmin(dists, axis=1)
            self.cluster_labels = np.array([unique_basis_labels[i] for i in nearest])
        else:
            self.cluster_labels = basis_labels

        self.results['linkage_method'] = linkage_method
        self.results['distance_metric'] = self.distance_metric
        self.results['n_clusters'] = n_clusters
        self.results['cluster_labels'] = self.cluster_labels.tolist()
        self.results['clustering_summary'] = {
            'n_clusters': n_clusters,
            'linkage': linkage_method,
            'metric': self.distance_metric,
            'scaler': self.scaler_type,
            'n_samples': self.n_samples,
            'is_large_data': self.is_large_data,
            'subsample_size': int(len(basis)) if self.is_large_data else None,
        }

    def _find_optimal_clusters(self, basis=None, max_k=10):
        if basis is None:
            basis = self._basis_data if self._basis_data is not None else self.cluster_data_scaled
        n_basis = len(basis)
        k_range = range(2, min(max_k + 1, n_basis - 1))
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in k_range:
            labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(basis, labels))
                calinski_scores.append(calinski_harabasz_score(basis, labels))
                davies_bouldin_scores.append(davies_bouldin_score(basis, labels))
            else:
                silhouette_scores.append(-1)
                calinski_scores.append(0)
                davies_bouldin_scores.append(np.inf)

        recommendations = {}
        if silhouette_scores:
            recommendations['silhouette'] = k_range[np.argmax(silhouette_scores)]
        if calinski_scores:
            recommendations['calinski_harabasz'] = k_range[np.argmax(calinski_scores)]
        if davies_bouldin_scores:
            recommendations['davies_bouldin'] = k_range[np.argmin(davies_bouldin_scores)]

        self.results['optimal_k_recommendation'] = recommendations
        self.results['validation_scores'] = {'k_range': list(k_range), 'silhouette': silhouette_scores}
        return recommendations

    def analyze_clusters(self):
        if self.cluster_labels is None: return

        profiles = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            mask = (self.cluster_labels == label)
            cluster_data = self.cluster_data[mask]
            profiles[f'Cluster {label}'] = {
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / self.n_samples * 100),
                'centroid': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict(),
                'min': cluster_data.min().to_dict(),
                'max': cluster_data.max().to_dict(),
            }
        self.results['profiles'] = profiles

        if len(unique_labels) > 1:
            self.results['final_metrics'] = {
                'silhouette': silhouette_score(self.cluster_data_scaled, self.cluster_labels),
                'calinski_harabasz': calinski_harabasz_score(self.cluster_data_scaled, self.cluster_labels),
                'davies_bouldin': davies_bouldin_score(self.cluster_data_scaled, self.cluster_labels),
                'note': 'Silhouette: higher better. Calinski-Harabasz: higher better. Davies-Bouldin: lower better.',
            }

        self.results['feature_drivers'] = _feature_drivers(self.cluster_data, self.cluster_labels, self.feature_cols)

        self.results['interpretations'] = self.generate_interpretations()

    def stability_analysis(self, n_bootstrap=None, sample_ratio=0.8):
        # Fewer bootstrap resamples when already operating on the (up to 2000-row)
        # large-data subsample, since each resample re-runs an O(n^2) linkage —
        # keeps runtime bounded without materially changing the stability estimate.
        if n_bootstrap is None:
            n_bootstrap = 12 if self.is_large_data else 30

        # Basis used to build self.linkage_matrix (full data, or the random
        # subsample when is_large_data). Cophenetic correlation and the
        # bootstrap-ARI stability score are both computed against this basis,
        # since it's what the dendrogram actually reflects.
        basis = self._basis_data if self._basis_data is not None else self.cluster_data_scaled
        basis_values = basis.values
        n_basis = len(basis_values)

        # ── Cophenetic correlation: how faithfully the dendrogram preserves the
        # basis's original pairwise distances (1.0 = perfect fidelity). ────────
        cophenetic_r = None
        try:
            basis_distances = pdist(basis_values, metric='euclidean' if self.linkage_method == 'ward' else self.distance_metric)
            coph_corr, _ = cophenet(self.linkage_matrix, basis_distances)
            cophenetic_r = _safe_float(coph_corr, default=None)
        except Exception:
            pass

        # ── Bootstrap stability: mean Adjusted Rand Index between the basis
        # clustering and re-clustered bootstrap resamples of the basis. ───────
        basis_labels = fcluster(self.linkage_matrix, t=self.n_clusters, criterion='maxclust')
        rng = np.random.RandomState(123)
        aris = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_basis, n_basis, replace=True)
            try:
                if self.linkage_method == 'ward':
                    bootstrap_linkage = linkage(basis_values[idx], method='ward')
                else:
                    bootstrap_distances = pdist(basis_values[idx], metric=self.distance_metric)
                    bootstrap_linkage = linkage(bootstrap_distances, method=self.linkage_method)
                bootstrap_labels = fcluster(bootstrap_linkage, self.n_clusters, criterion='maxclust')
                aris.append(adjusted_rand_score(basis_labels[idx], bootstrap_labels))
            except Exception:
                continue

        stability_score = float(np.mean(aris)) if aris else None
        self.results['stability'] = {
            'cophenetic_r': cophenetic_r,
            'stability_score': stability_score,
            'note': ('Cophenetic correlation measures how faithfully the dendrogram represents the original '
                     'pairwise distances (closer to 1 is better). Stability score is the mean Adjusted Rand '
                     'Index between the clustering and bootstrap resamples (0-1, higher = more reproducible).'),
        }

    def generate_interpretations(self):
        if 'profiles' not in self.results:
            return {}

        interpretations = {
            'overall_quality': '',
            'cluster_profiles': [],
            'cluster_distribution': ''
        }

        # 1. Overall Quality Interpretation
        if 'final_metrics' in self.results:
            metrics = self.results['final_metrics']
            silhouette = metrics['silhouette']
            
            if silhouette >= 0.7: quality_desc = "strong and well-defined."
            elif silhouette >= 0.5: quality_desc = "reasonable and distinct."
            elif silhouette >= 0.25: quality_desc = "weak and could have some overlap."
            else: quality_desc = "not well-defined; results should be interpreted with caution."
            
            interpretations['overall_quality'] = (
                f"The <strong>Silhouette Score of {silhouette:.3f}</strong> indicates the clustering structure is {quality_desc} "
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
            if max_p / min_p > 3:
                dist_desc = "The cluster sizes are imbalanced, with some clusters being significantly larger than others."
            else:
                dist_desc = "The clusters are relatively balanced in size."
            interpretations['cluster_distribution'] = dist_desc

        return interpretations

    def _fig_to_data_url(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    def plot_results(self):
        # Ensure seaborn style is applied
        sns.set_theme(style="darkgrid")
        sns.set_context("notebook", font_scale=1.1)

        plots = []

        # 1. Dendrogram
        fig1, ax1 = plt.subplots(figsize=(10, 5.5))
        cut_height = 0
        if self.n_clusters > 1 and len(self.linkage_matrix) >= self.n_clusters - 1:
            cut_height = self.linkage_matrix[-(self.n_clusters - 1), 2]
        # Per-leaf text tick labels are both unreadable and extremely slow to
        # lay out once the linkage has more than a couple hundred leaves
        # (matplotlib text-layout cost dominates runtime) — drop them past that.
        n_leaves = len(self.linkage_matrix) + 1
        dendrogram(self.linkage_matrix, ax=ax1, color_threshold=cut_height, above_threshold_color='gray',
                   no_labels=n_leaves > 200)
        ax1.axhline(y=cut_height, c='red', linestyle='--', linewidth=2, label=f'Cut for {self.n_clusters} clusters')
        dendrogram_title = 'Hierarchical Clustering Dendrogram'
        if self.is_large_data:
            dendrogram_title += f' (subsample n={len(self._basis_data)})'
        ax1.set_title(dendrogram_title, fontsize=12, fontweight='bold')
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Distance', fontsize=12)
        ax1.legend()
        ax1.grid(False)
        plt.tight_layout()
        plots.append({'label': 'Dendrogram', 'image': self._fig_to_data_url(fig1)})

        # 2. PCA Plot
        if self.n_features > 1:
            fig2, ax2 = plt.subplots(figsize=(7, 5.5))
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(self.cluster_data_scaled)

            # Use crest palette
            n_clusters = len(np.unique(self.cluster_labels))
            palette = sns.color_palette('crest', n_colors=n_clusters)

            label_colors = {}
            for i, label in enumerate(np.unique(self.cluster_labels)):
                mask = self.cluster_labels == label
                label_colors[label] = palette[i]
                ax2.scatter(pca_data[mask, 0], pca_data[mask, 1],
                          c=[palette[i]], label=f'Cluster {label}',
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            # Convex hull overlay per cluster
            for label in np.unique(self.cluster_labels):
                mask = self.cluster_labels == label
                cluster_points = pca_data[mask]
                if cluster_points.shape[0] < 3:
                    continue
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    ax2.fill(hull_points[:, 0], hull_points[:, 1],
                             color=label_colors[label], alpha=0.15, zorder=0)
                except Exception:
                    pass

            ax2.set_title('Clusters in PCA Space', fontsize=12, fontweight='bold')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
            ax2.legend(title='Cluster')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', which='major', width=1.5)
            plt.tight_layout()
            plots.append({'label': 'Clusters in PCA Space', 'image': self._fig_to_data_url(fig2)})

        # 3. Silhouette Score Plot
        if 'validation_scores' in self.results:
            fig3, ax3 = plt.subplots(figsize=(7, 5.5))
            opt_k_res = self.results['validation_scores']
            ax3.plot(opt_k_res['k_range'], opt_k_res['silhouette'], 'o-',
                    color='#5B9BD5', linewidth=2, markersize=8)
            ax3.set_title('Silhouette Scores by Number of Clusters', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax3.set_ylabel('Average Silhouette Score', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', which='major', width=1.5)
            if 'optimal_k_recommendation' in self.results and 'silhouette' in self.results['optimal_k_recommendation']:
                rec_k = self.results['optimal_k_recommendation']['silhouette']
                ax3.axvline(x=rec_k, color='red', linestyle='--', linewidth=2, label=f'Recommended k = {rec_k}')
                ax3.legend()
            plt.tight_layout()
            plots.append({'label': 'Silhouette Scores by Number of Clusters', 'image': self._fig_to_data_url(fig3)})

        # 4. Cluster Size Distribution
        fig4, ax4 = plt.subplots(figsize=(7, 5.5))
        cluster_sizes = pd.Series(self.cluster_labels).value_counts().sort_index()

        # Use crest palette for bars
        colors = sns.color_palette('crest', n_colors=len(cluster_sizes))

        ax4.bar(range(len(cluster_sizes)), cluster_sizes.values,
               color=colors, alpha=0.7, edgecolor='black')
        ax4.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Cluster', fontsize=12)
        ax4.set_ylabel('Number of Samples', fontsize=12)
        ax4.set_xticks(range(len(cluster_sizes)))
        ax4.set_xticklabels(cluster_sizes.index)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', width=1.5)
        plt.tight_layout()
        plots.append({'label': 'Cluster Size Distribution', 'image': self._fig_to_data_url(fig4)})

        # 5. Centroid Heatmap
        if 'profiles' in self.results:
            centroids_scaled = []
            cluster_names = []
            for name, profile in sorted(self.results['profiles'].items()):
                scaled_center = self.cluster_data_scaled[self.cluster_labels == int(name.split(' ')[1])].mean().values
                centroids_scaled.append(scaled_center)
                cluster_names.append(name)

            if centroids_scaled:
                fig5, ax5 = plt.subplots(figsize=(7, 5.5))
                centroid_df = pd.DataFrame(centroids_scaled, columns=self.feature_cols, index=cluster_names)
                sns.heatmap(centroid_df, annot=True, cmap='crest', ax=ax5, fmt='.2f')
                ax5.set_title('Scaled Centroid Heatmap', fontsize=12, fontweight='bold')
                ax5.tick_params(axis='x', rotation=45)
                ax5.tick_params(axis='both', which='major', width=1.5)
                plt.tight_layout()
                plots.append({'label': 'Scaled Centroid Heatmap', 'image': self._fig_to_data_url(fig5)})

        # 6. 3D PCA Scatter (only when there are enough features to justify a 3rd dimension)
        if self.n_features >= 3:
            fig6 = plt.figure(figsize=(7, 5.5))
            ax6 = fig6.add_subplot(111, projection='3d')
            pca3 = PCA(n_components=3)
            pca3_data = pca3.fit_transform(self.cluster_data_scaled)

            n_clusters_3d = len(np.unique(self.cluster_labels))
            palette3 = sns.color_palette('viridis', n_colors=n_clusters_3d)

            for i, label in enumerate(np.unique(self.cluster_labels)):
                mask = self.cluster_labels == label
                ax6.scatter(pca3_data[mask, 0], pca3_data[mask, 1], pca3_data[mask, 2],
                           c=[palette3[i]], label=f'Cluster {label}',
                           alpha=0.7, s=40, edgecolors='black', linewidth=0.4)

            ax6.set_title('Clusters (3D PCA)', fontsize=12, fontweight='bold')
            ax6.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%})', fontsize=10)
            ax6.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]:.1%})', fontsize=10)
            ax6.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%})', fontsize=10)
            ax6.legend(title='Cluster')
            plt.tight_layout()
            plots.append({'label': 'Clusters (3D PCA)', 'image': self._fig_to_data_url(fig6)})

        return plots

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        linkage_method = payload.get('linkageMethod', 'ward')
        distance_metric = payload.get('distanceMetric', 'euclidean')
        n_clusters = payload.get('nClusters') # Can be None
        scaler_type = payload.get('scalerType') or 'standard'

        if not data or not items:
            raise ValueError("Missing 'data' or 'items'")

        hca = HierarchicalClusterAnalysis(data=data, feature_cols=items, scaler_type=scaler_type)
        hca.perform_clustering(linkage_method, distance_metric, n_clusters)
        hca.analyze_clusters()
        hca.stability_analysis()
        
        plots = hca.plot_results()

        response = {
            'results': hca.results,
            'plots': plots
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()