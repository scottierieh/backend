
import sys
import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
