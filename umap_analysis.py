
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
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
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

class UmapAnalysis:
    def __init__(self, data, variables, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', label_col=None):
        self.data = pd.DataFrame(data).copy()
        self.variables = variables
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.label_col = label_col
        self.results = {}
        self._prepare_data()

    def _prepare_data(self):
        cols = self.variables + ([self.label_col] if self.label_col and self.label_col in self.data.columns else [])
        self.clean_data = self.data[cols].dropna() if cols else self.data[self.variables].dropna()
        if self.clean_data.empty:
            raise ValueError("No valid data for selected variables.")

        self.labels = self.clean_data[self.label_col] if self.label_col and self.label_col in self.clean_data.columns else None

        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.clean_data[self.variables])

    def run_analysis(self):
        n_samples = self.scaled_data.shape[0]
        effective_neighbors = min(self.n_neighbors, max(2, n_samples - 1))

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=effective_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=42
        )
        self.embedding = reducer.fit_transform(self.scaled_data)

        self.results['embedding'] = self.embedding
        self.results['n_neighbors_used'] = effective_neighbors
        self.results['min_dist'] = self.min_dist
        self.results['metric'] = self.metric
        self.results['n_components'] = self.n_components
        self.results['variables'] = self.variables
        self.results['n_samples'] = n_samples

        self.results['interpretation'] = self._generate_interpretation(n_samples)

    def _generate_interpretation(self, n_samples):
        parts = []
        parts.append("**Overall Assessment**")
        parts.append(
            f"→ UMAP embedded {n_samples} observations across {len(self.variables)} input variables "
            f"into {self.n_components} dimensions using n_neighbors = {self.results['n_neighbors_used']}, "
            f"min_dist = {self.min_dist}, metric = '{self.metric}'."
        )
        if self.n_neighbors != self.results['n_neighbors_used']:
            parts.append(
                f"→ Requested n_neighbors ({self.n_neighbors}) was reduced to "
                f"{self.results['n_neighbors_used']} to stay below the sample size."
            )

        parts.append("")
        parts.append("**Statistical Insights**")
        if self.labels is not None:
            try:
                sil = silhouette_score(self.embedding, self.labels)
                sil_desc = "well separated" if sil >= 0.5 else "moderately separated" if sil >= 0.25 else "overlapping"
                parts.append(
                    f"→ Silhouette score of the embedding w.r.t. '{self.label_col}' = {sil:.3f} — groups "
                    f"appear {sil_desc} in the projection."
                )
            except Exception:
                pass
        parts.append(
            f"→ n_neighbors controls the local/global tradeoff: smaller values ({self.results['n_neighbors_used']} "
            "used here) emphasize fine local structure, larger values emphasize broader global structure."
        )
        parts.append(
            f"→ min_dist = {self.min_dist} controls how tightly points are packed; lower values produce "
            "tighter, more separated clusters, higher values preserve a more even spread."
        )

        parts.append("")
        parts.append("**Recommendations**")
        parts.append(
            "→ Unlike t-SNE, UMAP better preserves some global structure, but inter-cluster distances "
            "should still be interpreted cautiously rather than taken as precise dissimilarities."
        )
        parts.append("→ Try a range of n_neighbors (5-50) and min_dist (0.0-0.5) to check how stable the cluster structure is.")
        parts.append("→ Because UMAP is stochastic, re-run with a few random seeds if the embedding will inform downstream decisions.")

        return "\n".join(parts)

    def plot_results(self):
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.suptitle('UMAP Embedding', fontsize=16)

        if self.labels is not None:
            unique_labels = self.labels.unique()
            palette = sns.color_palette('viridis', n_colors=len(unique_labels))
            sns.scatterplot(x=self.embedding[:, 0], y=self.embedding[:, 1], hue=self.labels.astype(str),
                             palette=palette, ax=ax, s=50, alpha=0.8, legend='full')
            ax.legend(title=self.label_col, bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.scatter(self.embedding[:, 0], self.embedding[:, 1], alpha=0.7, s=50, c='steelblue')

        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_title(f"n_neighbors={self.results['n_neighbors_used']}, min_dist={self.min_dist}")
        ax.grid(True, alpha=0.3)

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
        variables = payload.get('variables')
        n_components = int(payload.get('nComponents', 2))
        n_neighbors = int(payload.get('nNeighbors', 15))
        min_dist = float(payload.get('minDist', 0.1))
        metric = payload.get('metric', 'euclidean')
        label_col = payload.get('labelCol')

        if not data or not variables:
            raise ValueError("Missing 'data' or 'variables'")

        analysis = UmapAnalysis(
            data, variables,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            label_col=label_col
        )
        analysis.run_analysis()
        plot_image = analysis.plot_results()

        response = {
            'results': analysis.results,
            'plot': plot_image
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
