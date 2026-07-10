import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
import io
import base64
import math

warnings.filterwarnings('ignore')

# Set seaborn style globally
sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

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

class HierarchicalClusterAnalysis:
    def __init__(self, data, feature_cols=None, standardize=True):
        self.data = pd.DataFrame(data)
        self.standardize = standardize
        self.feature_cols = feature_cols if feature_cols is not None else self.data.select_dtypes(include=np.number).columns.tolist()
        
        self.cluster_data = self.data[self.feature_cols].copy().dropna()
        
        if self.standardize:
            scaler = StandardScaler()
            self.cluster_data_scaled = pd.DataFrame(scaler.fit_transform(self.cluster_data), columns=self.feature_cols, index=self.cluster_data.index)
        else:
            self.cluster_data_scaled = self.cluster_data.copy()
            
        self.n_samples, self.n_features = self.cluster_data_scaled.shape
        self.results = {}

    def perform_clustering(self, linkage_method='ward', distance_metric='euclidean', n_clusters=None):
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        
        if linkage_method == 'ward' and distance_metric != 'euclidean':
            warnings.warn("Ward linkage requires euclidean distance. Overriding distance_metric to 'euclidean'.")
            self.distance_metric = 'euclidean'
            distances = pdist(self.cluster_data_scaled, metric='euclidean')
            self.linkage_matrix = linkage(distances, method='ward')
        elif linkage_method == 'ward':
            distances = pdist(self.cluster_data_scaled, metric='euclidean')
            self.linkage_matrix = linkage(distances, method='ward')
        else:
            distances = pdist(self.cluster_data_scaled, metric=self.distance_metric)
            self.linkage_matrix = linkage(distances, method=linkage_method)
        
        if n_clusters is None:
            recommendations = self._find_optimal_clusters()
            n_clusters = recommendations.get('silhouette', 3) # Default to 3 if silhouette fails
        
        self.n_clusters = n_clusters
        self.cluster_labels = fcluster(self.linkage_matrix, t=n_clusters, criterion='maxclust')
        
        self.results['linkage_method'] = linkage_method
        self.results['distance_metric'] = self.distance_metric
        self.results['n_clusters'] = n_clusters
        self.results['cluster_labels'] = self.cluster_labels.tolist()
        
    def _find_optimal_clusters(self, max_k=10):
        k_range = range(2, min(max_k + 1, self.n_samples - 1))
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in k_range:
            labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(self.cluster_data_scaled, labels))
                calinski_scores.append(calinski_harabasz_score(self.cluster_data_scaled, labels))
                davies_bouldin_scores.append(davies_bouldin_score(self.cluster_data_scaled, labels))
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
            }
        
        self.results['interpretations'] = self.generate_interpretations()

    def stability_analysis(self, n_bootstrap=50, sample_ratio=0.8):
        stability_scores = []
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(self.n_samples, int(self.n_samples * sample_ratio), replace=True)
            bootstrap_data = self.cluster_data_scaled.iloc[sample_indices]
            
            if self.linkage_method == 'ward':
                bootstrap_linkage = linkage(bootstrap_data.values, method='ward')
            else:
                bootstrap_distances = pdist(bootstrap_data.values, metric=self.distance_metric)
                bootstrap_linkage = linkage(bootstrap_distances, method=self.linkage_method)

            bootstrap_labels = fcluster(bootstrap_linkage, self.n_clusters, criterion='maxclust')
            
            if len(np.unique(bootstrap_labels)) > 1:
                stability_scores.append(silhouette_score(bootstrap_data, bootstrap_labels))
        
        if stability_scores:
            mean_stability = np.mean(stability_scores)
            std_stability = np.std(stability_scores)
            self.results['stability'] = {
                'mean': mean_stability,
                'std': std_stability
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

    def plot_results(self):
        # Ensure seaborn style is applied
        sns.set_theme(style="darkgrid")
        sns.set_context("notebook", font_scale=1.1)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))

        # 1. Dendrogram
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, fig=fig)
        cut_height = 0
        if self.n_clusters > 1 and len(self.linkage_matrix) >= self.n_clusters - 1:
            cut_height = self.linkage_matrix[-(self.n_clusters - 1), 2]
        dendrogram(self.linkage_matrix, ax=ax1, color_threshold=cut_height, above_threshold_color='gray')
        ax1.axhline(y=cut_height, c='red', linestyle='--', linewidth=2, label=f'Cut for {self.n_clusters} clusters')
        ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Distance', fontsize=12)
        ax1.legend()
        ax1.grid(False)
        
        # 2. PCA Plot
        ax2 = axes[1, 0]
        if self.n_features > 1:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(self.cluster_data_scaled)
            
            # Use crest palette
            n_clusters = len(np.unique(self.cluster_labels))
            palette = sns.color_palette('crest', n_colors=n_clusters)
            
            for i, label in enumerate(np.unique(self.cluster_labels)):
                mask = self.cluster_labels == label
                ax2.scatter(pca_data[mask, 0], pca_data[mask, 1], 
                          c=[palette[i]], label=f'Cluster {label}', 
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            ax2.set_title('Clusters in PCA Space', fontsize=12, fontweight='bold')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
            ax2.legend(title='Cluster')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', which='major', width=1.5)

        # 3. Silhouette Score Plot
        ax3 = axes[1, 1]
        if 'validation_scores' in self.results:
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

        # 4. Cluster Size Distribution
        ax4 = axes[2, 0]
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

        # 5. Centroid Heatmap
        ax5 = axes[2, 1]
        if 'profiles' in self.results:
            centroids_scaled = []
            cluster_names = []
            for name, profile in sorted(self.results['profiles'].items()):
                scaled_center = self.cluster_data_scaled[self.cluster_labels == int(name.split(' ')[1])].mean().values
                centroids_scaled.append(scaled_center)
                cluster_names.append(name)

            if centroids_scaled:
                centroid_df = pd.DataFrame(centroids_scaled, columns=self.feature_cols, index=cluster_names)
                sns.heatmap(centroid_df, annot=True, cmap='crest', ax=ax5, fmt='.2f')
                ax5.set_title('Scaled Centroid Heatmap', fontsize=12, fontweight='bold')
                ax5.tick_params(axis='x', rotation=45)
                ax5.tick_params(axis='both', which='major', width=1.5)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        linkage_method = payload.get('linkageMethod', 'ward')
        distance_metric = payload.get('distanceMetric', 'euclidean')
        n_clusters = payload.get('nClusters') # Can be None

        if not data or not items:
            raise ValueError("Missing 'data' or 'items'")

        hca = HierarchicalClusterAnalysis(data=data, feature_cols=items, standardize=True)
        hca.perform_clustering(linkage_method, distance_metric, n_clusters)
        hca.analyze_clusters()
        hca.stability_analysis()
        
        plot_image = hca.plot_results()

        response = {
            'results': hca.results,
            'plot': plot_image
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()