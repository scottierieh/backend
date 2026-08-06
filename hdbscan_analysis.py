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
    return obj

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
    """Use DBSCAN as a fallback with pseudo-probabilities"""
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
    
    return labels, probabilities

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        min_cluster_size = int(payload.get('min_cluster_size', 5))
        min_samples = payload.get('min_samples')

        if not data or not items:
            raise ValueError("Missing 'data' or 'items'")

        df = pd.DataFrame(data)[items].dropna()
        
        if df.shape[0] == 0:
            raise ValueError("No valid data points for analysis.")

        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Run clustering
        if HAS_HDBSCAN:
            # Use actual HDBSCAN if available
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size, 
                min_samples=min_samples if min_samples else None,
                gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(X_scaled)
            probabilities = clusterer.probabilities_
        else:
            # Fall back to DBSCAN with pseudo-probabilities
            labels, probabilities = dbscan_with_probabilities(
                X_scaled, min_cluster_size, min_samples
            )

        # Analysis Summary
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # Calculate cluster profiles
        profiles = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = (labels == label)
            cluster_data = df[mask]
            
            cluster_name = f'Cluster {label}' if label != -1 else 'Noise'
            
            profiles[cluster_name] = {
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / len(df) * 100),
                'centroid': cluster_data.mean().to_dict() if not cluster_data.empty else {},
            }

        summary = {
            'n_clusters': n_clusters_,
            'n_noise': n_noise_,
            'n_samples': len(df),
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'labels': labels.tolist(),
            'probabilities': probabilities.tolist(),
            'profiles': profiles,
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

                # Plot non-noise points with size based on probability
                clustered_points = plot_df[plot_df['cluster'] != -1]
                if not clustered_points.empty:
                    # Create sizes based on probability
                    sizes = clustered_points['probability'] * 150 + 20

                    for i, label in enumerate(unique_labels):
                        cluster_data = clustered_points[clustered_points['cluster'] == label]
                        if not cluster_data.empty:
                            ax.scatter(
                                cluster_data['PC1'],
                                cluster_data['PC2'],
                                s=sizes[clustered_points['cluster'] == label],
                                c=[palette[i]] * len(cluster_data),
                                label=f'Cluster {label}',
                                alpha=0.7
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

    