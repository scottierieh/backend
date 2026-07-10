
import sys
import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

warnings.filterwarnings('ignore')

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

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        items = payload.get('items')
        eps = float(payload.get('eps', 0.5))
        min_samples = int(payload.get('min_samples', 5))

        if not data or not items:
            raise ValueError("Missing 'data' or 'items'")

        df = pd.DataFrame(data)[items].dropna()
        
        if df.shape[0] == 0:
            raise ValueError("No valid data points for analysis.")

        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)

        # Analysis Summary
        labels = dbscan.labels_
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
                'centroid': cluster_data.mean().to_dict(),
            }

        summary = {
            'n_clusters': n_clusters_,
            'n_noise': n_noise_,
            'n_samples': len(df),
            'eps': eps,
            'min_samples': min_samples,
            'labels': labels.tolist(),
            'profiles': profiles,
        }

        # --- Plotting ---
        plot_image = None
        if df.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            plot_df['cluster'] = labels

            plt.figure(figsize=(8, 6))
            
            # Use a categorical palette, handle noise points separately
            unique_labels = sorted(list(set(labels)))
            palette = sns.color_palette("viridis", n_colors=len(unique_labels) - (1 if -1 in unique_labels else 0))
            
            for i, label in enumerate(unique_labels):
                if label == -1:
                    # Noise points
                    sns.scatterplot(
                        x=plot_df[plot_df['cluster'] == label]['PC1'],
                        y=plot_df[plot_df['cluster'] == label]['PC2'],
                        color='gray',
                        marker='x',
                        s=50,
                        label='Noise'
                    )
                else:
                    sns.scatterplot(
                        x=plot_df[plot_df['cluster'] == label]['PC1'],
                        y=plot_df[plot_df['cluster'] == label]['PC2'],
                        color=palette[i - (1 if -1 in unique_labels else 0)],
                        label=f'Cluster {label}',
                        s=80,
                        alpha=0.7
                    )

            plt.title('DBSCAN Clustering (PCA Projection)')
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
            plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
            plt.legend(title='Cluster')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            plot_image = base64.b64encode(buf.read()).decode('utf-8')

        response = {
            'results': summary,
            'plot': f"data:image/png;base64,{plot_image}" if plot_image else None
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
