
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
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

class TsneAnalysis:
    def __init__(self, data, variables, n_components=2, perplexity=30.0, learning_rate='auto', n_iter=1000, label_col=None):
        self.data = pd.DataFrame(data).copy()
        self.variables = variables
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
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
        effective_perplexity = min(self.perplexity, max(5.0, (n_samples - 1) / 3))

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=effective_perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.n_iter,
            random_state=42,
            init='pca'
        )
        self.embedding = tsne.fit_transform(self.scaled_data)

        self.results['embedding'] = self.embedding
        self.results['kl_divergence'] = tsne.kl_divergence_
        self.results['n_iterations_run'] = tsne.n_iter_
        self.results['perplexity_used'] = effective_perplexity
        self.results['n_components'] = self.n_components
        self.results['variables'] = self.variables
        self.results['n_samples'] = n_samples

        self.results['interpretation'] = self._generate_interpretation(n_samples)

    def _generate_interpretation(self, n_samples):
        parts = []
        parts.append("**Overall Assessment**")
        parts.append(
            f"→ t-SNE embedded {n_samples} observations across {len(self.variables)} input variables "
            f"into {self.n_components} dimensions using perplexity = {self.results['perplexity_used']:.1f} "
            f"(ran {self.results['n_iterations_run']} of {self.n_iter} max iterations)."
        )
        if self.perplexity != self.results['perplexity_used']:
            parts.append(
                f"→ Requested perplexity ({self.perplexity:.1f}) was reduced to "
                f"{self.results['perplexity_used']:.1f} because it must stay well below the sample size."
            )

        parts.append("")
        parts.append("**Statistical Insights**")
        kl = self.results['kl_divergence']
        parts.append(
            f"→ Final KL divergence = {kl:.3f} (lower indicates the low-dimensional embedding better "
            "preserves local neighborhood structure from the original space; not comparable across datasets)."
        )
        if self.labels is not None:
            try:
                sil = silhouette_score(self.embedding, self.labels)
                sil_desc = "well separated" if sil >= 0.5 else "moderately separated" if sil >= 0.25 else "overlapping"
                parts.append(
                    f"→ Silhouette score of the embedding w.r.t. '{self.label_col}' = {sil:.3f} — groups "
                    f"appear {sil_desc} in the 2D projection."
                )
            except Exception:
                pass

        parts.append("")
        parts.append("**Recommendations**")
        parts.append(
            "→ Caveat: in t-SNE plots, cluster sizes, inter-cluster distances, and densities are not "
            "directly meaningful — only relative neighborhood grouping should be interpreted."
        )
        parts.append(
            "→ Try a few different perplexity values (typically 5-50) — results can vary noticeably, "
            "especially on smaller datasets."
        )
        parts.append("→ For a globally faithful (and reproducible) alternative to compare against, consider UMAP or PCA.")

        return "\n".join(parts)

    def plot_results(self):
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.suptitle('t-SNE Embedding', fontsize=16)

        if self.labels is not None:
            unique_labels = self.labels.unique()
            palette = sns.color_palette('viridis', n_colors=len(unique_labels))
            sns.scatterplot(x=self.embedding[:, 0], y=self.embedding[:, 1], hue=self.labels.astype(str),
                             palette=palette, ax=ax, s=50, alpha=0.8, legend='full')
            ax.legend(title=self.label_col, bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.scatter(self.embedding[:, 0], self.embedding[:, 1], alpha=0.7, s=50, c='steelblue')

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(f"Perplexity={self.results['perplexity_used']:.1f}, KL Divergence={self.results['kl_divergence']:.3f}")
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
        perplexity = float(payload.get('perplexity', 30.0))
        n_iter = int(payload.get('nIter', 1000))
        label_col = payload.get('labelCol')

        if not data or not variables:
            raise ValueError("Missing 'data' or 'variables'")

        analysis = TsneAnalysis(
            data, variables,
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
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
