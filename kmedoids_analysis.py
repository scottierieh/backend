
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
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

class KMedoidsAnalysis:
    def __init__(self, data, feature_cols, standardize=True):
        self.data = pd.DataFrame(data)
        self.feature_cols = feature_cols
        self.cluster_data = self.data[self.feature_cols].copy().dropna()
        
        if standardize:
            scaler = StandardScaler()
            self.cluster_data_scaled = pd.DataFrame(scaler.fit_transform(self.cluster_data), columns=self.feature_cols, index=self.cluster_data.index)
        else:
            self.cluster_data_scaled = self.cluster_data.copy()
            
        self.n_samples, self.n_features = self.cluster_data_scaled.shape
        self.results = {}

    def perform_clustering(self, n_clusters, init_method='k-medoids++', max_iter=300):
        self.n_clusters = n_clusters
        
        kmedoids = KMedoids(n_clusters=n_clusters, method='pam', init=init_method, max_iter=max_iter, random_state=42)
        self.cluster_labels = kmedoids.fit_predict(self.cluster_data_scaled)
        
        self.results['clustering_summary'] = {
            'n_clusters': n_clusters,
            'inertia': kmedoids.inertia_,
            'medoids': self.cluster_data.iloc[kmedoids.medoid_indices_].to_dict('records'),
            'medoid_indices': kmedoids.medoid_indices_.tolist(),
            'labels': self.cluster_labels.tolist()
        }
        
        self.analyze_clusters()
        return self.results

    def analyze_clusters(self):
        profiles = {}
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        for i, label in enumerate(unique_labels):
            mask = (self.cluster_labels == label)
            cluster_data = self.cluster_data[mask]
            profiles[f'Cluster {label + 1}'] = {
                'size': int(counts[i]),
                'percentage': float(counts[i] / self.n_samples * 100),
                'centroid': cluster_data.mean().to_dict(), # Using mean as centroid for profile summary
            }
        self.results['profiles'] = profiles

        if len(unique_labels) > 1:
            self.results['final_metrics'] = {
                'silhouette': silhouette_score(self.cluster_data_scaled, self.cluster_labels),
                'davies_bouldin': davies_bouldin_score(self.cluster_data_scaled, self.cluster_labels),
                'calinski_harabasz': calinski_harabasz_score(self.cluster_data_scaled, self.cluster_labels),
            }
            
        self.results['interpretations'] = self.generate_interpretations()

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

    def plot_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('K-Medoids Clustering Results', fontsize=16, fontweight='bold')

        # 1. Cluster Scatter Plot (PCA)
        if self.n_features >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(self.cluster_data_scaled)
            
            sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=self.cluster_labels, 
                            palette='viridis', ax=axes[0], legend='full', s=50, alpha=0.7)
            
            medoids_pca = pca.transform(self.cluster_data_scaled.iloc[self.results['clustering_summary']['medoid_indices']])
            
            axes[0].scatter(medoids_pca[:, 0], medoids_pca[:, 1], s=250, c='red', marker='X', label='Medoids', edgecolors='black')
            
            axes[0].set_title('Clusters in 2D PCA Space')
            axes[0].set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
            axes[0].set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

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
                
                ax_radar = fig.add_subplot(122, polar=True)
                for i, (idx, row) in enumerate(medoids_norm.iterrows()):
                    values = row.tolist()
                    values += values[:1]
                    ax_radar.plot(angles, values, label=f'Cluster {i+1}')
                    ax_radar.fill(angles, values, alpha=0.25)
                
                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(self.feature_cols)
                ax_radar.set_title('Cluster Medoids (Normalized)', size=12)
                ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        n_clusters = payload.get('nClusters')

        if not data or not items or n_clusters is None:
            raise ValueError("Missing 'data', 'items', or 'nClusters'")

        kma = KMedoidsAnalysis(data=data, feature_cols=items)
        kma.perform_clustering(n_clusters=n_clusters)
        
        plot_image = kma.plot_results()

        response = {
            'results': kma.results,
            'plot': plot_image
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
