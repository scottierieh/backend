
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.cluster import HDBSCAN
import umap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
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
        random_state = 42

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=effective_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=random_state
        )
        self.embedding = reducer.fit_transform(self.scaled_data)

        self.results['embedding'] = self.embedding
        self.results['n_neighbors_requested'] = self.n_neighbors
        self.results['n_neighbors_used'] = effective_neighbors
        self.results['random_state'] = random_state
        self.results['min_dist'] = self.min_dist
        self.results['metric'] = self.metric
        self.results['n_components'] = self.n_components
        self.results['variables'] = self.variables
        self.results['n_features'] = len(self.variables)
        self.results['n_samples'] = n_samples

        # Silhouette score w.r.t. the optional label column — only meaningful (and only
        # computed) when there are at least 2 distinct non-null classes among the rows
        # that actually made it into the embedding.
        self.results['silhouette_score'] = None
        if self.labels is not None:
            n_classes = self.labels.dropna().nunique()
            if n_classes >= 2:
                try:
                    self.results['silhouette_score'] = float(silhouette_score(self.embedding, self.labels))
                except Exception:
                    self.results['silhouette_score'] = None

        # Trustworthiness / continuity — label-free structure-preservation diagnostics.
        # trustworthiness(X, embedding) penalizes embedding-space neighbors that were NOT
        # true neighbors in the original space (intrusions). continuity is the Venna &
        # Kaski (2001) counterpart: swapping the two arguments penalizes original-space
        # neighbors that are NOT preserved as neighbors in the embedding (extrusions),
        # weighted by their rank in the embedded space instead of the original space.
        # sklearn requires n_neighbors < n_samples / 2, so cap well below that.
        k_tc = max(1, min(effective_neighbors, (n_samples - 1) // 2 - 1)) if n_samples >= 4 else None
        if k_tc and k_tc >= 1:
            try:
                self.results['trustworthiness'] = float(
                    trustworthiness(self.scaled_data, self.embedding, n_neighbors=k_tc)
                )
                self.results['continuity'] = float(
                    trustworthiness(self.embedding, self.scaled_data, n_neighbors=k_tc)
                )
            except Exception:
                self.results['trustworthiness'] = None
                self.results['continuity'] = None
        else:
            self.results['trustworthiness'] = None
            self.results['continuity'] = None

        # Per-dimension coordinate summary statistics.
        coord_stats = []
        for i in range(self.n_components):
            col = self.embedding[:, i]
            coord_stats.append({
                'component': f'UMAP-{i + 1}',
                'min': float(np.min(col)),
                'max': float(np.max(col)),
                'mean': float(np.mean(col)),
                'sd': float(np.std(col, ddof=1)) if len(col) > 1 else None,
            })
        self.results['coordinate_stats'] = coord_stats

        # Automatic cluster detection on the embedding via HDBSCAN (native in
        # scikit-learn >= 1.3, no extra dependency needed).
        cluster_labels, cluster_sil, min_cluster_size = self._detect_clusters(self.embedding, n_samples)
        if cluster_labels is not None:
            unique, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
            n_noise = int(np.sum(cluster_labels == -1))
            self.results['cluster_detection'] = {
                'algorithm': 'HDBSCAN',
                'min_cluster_size': min_cluster_size,
                'n_clusters': int(len(unique)),
                'n_noise': n_noise,
                'noise_pct': float(n_noise / n_samples * 100) if n_samples else None,
                'clusters': [
                    {'cluster': int(u), 'size': int(c), 'pct': float(c / n_samples * 100)}
                    for u, c in zip(unique, counts)
                ],
                'silhouette_score': cluster_sil,
                'labels': cluster_labels.tolist(),
            }
        else:
            self.results['cluster_detection'] = None

        # Feature correlations: how each standardized input feature correlates with
        # each embedding axis, to help interpret what the axes roughly capture.
        feature_corrs = []
        for i, var in enumerate(self.variables):
            row = {'feature': var}
            for j in range(self.n_components):
                with np.errstate(invalid='ignore'):
                    corr = np.corrcoef(self.scaled_data[:, i], self.embedding[:, j])[0, 1]
                row[f'umap_{j + 1}_correlation'] = float(corr) if np.isfinite(corr) else None
            feature_corrs.append(row)
        self.results['feature_correlations'] = feature_corrs

        # Parameter stability: re-run UMAP at a small, bounded set of alternative
        # n_neighbors values and compare the cluster silhouette each produces. A
        # setting whose silhouette holds up across the sweep indicates the detected
        # structure isn't an artifact of one particular n_neighbors choice.
        self.results['parameter_stability'] = self._parameter_stability(n_samples, cluster_sil)

        self.results['interpretation'] = self._generate_interpretation(n_samples)

    def _detect_clusters(self, embedding, n_samples):
        """Run HDBSCAN on an embedding and, if it finds >= 2 clusters, a silhouette
        score over the non-noise points. Returns (labels, silhouette, min_cluster_size)
        — labels/silhouette are None if clustering isn't meaningful for this data."""
        if n_samples < 5:
            return None, None, None
        min_cluster_size = max(3, min(int(n_samples * 0.05), 25))
        try:
            labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(embedding)
        except Exception:
            return None, None, min_cluster_size
        mask = labels != -1
        n_clusters = len(set(labels[mask].tolist()))
        sil = None
        if n_clusters >= 2:
            try:
                sil = float(silhouette_score(embedding[mask], labels[mask]))
            except Exception:
                sil = None
        return labels, sil, min_cluster_size

    def _parameter_stability(self, n_samples, base_silhouette):
        # Bounded, fixed candidate set so re-running UMAP a handful of times never
        # blows up runtime, even for larger datasets.
        candidates = sorted({c for c in (5, 15, 30, 50) if 2 <= c < n_samples})[:4]
        if not candidates or n_samples > 2000:
            return {'sweep': [], 'best_n_neighbors': None}

        sweep = []
        best_n, best_sil = None, None
        for c in candidates:
            if c == self.results['n_neighbors_used']:
                sil = base_silhouette
            else:
                try:
                    emb_c = umap.UMAP(
                        n_components=self.n_components, n_neighbors=c,
                        min_dist=self.min_dist, metric=self.metric, random_state=42
                    ).fit_transform(self.scaled_data)
                    _, sil, _ = self._detect_clusters(emb_c, n_samples)
                except Exception:
                    sil = None
            sweep.append({'n_neighbors': c, 'silhouette': sil})
            if sil is not None and (best_sil is None or sil > best_sil):
                best_sil, best_n = sil, c

        return {'sweep': sweep, 'best_n_neighbors': best_n}

    def _generate_interpretation(self, n_samples):
        sil = self.results.get('silhouette_score')
        clamped = self.n_neighbors != self.results['n_neighbors_used']

        key_insights = []
        key_insights.append({
            'title': 'Embedding overview',
            'description': (
                f"UMAP embedded {n_samples} observations across {len(self.variables)} input variables "
                f"into {self.n_components} dimensions using n_neighbors = {self.results['n_neighbors_used']}, "
                f"min_dist = {self.min_dist}, metric = '{self.metric}'."
            ),
            'status': 'neutral',
        })
        if clamped:
            key_insights.append({
                'title': 'n_neighbors was clamped',
                'description': (
                    f"Requested n_neighbors ({self.n_neighbors}) was reduced to "
                    f"{self.results['n_neighbors_used']} to stay below the sample size."
                ),
                'status': 'warning',
            })
        if sil is not None:
            sil_desc = "well separated" if sil >= 0.5 else "moderately separated" if sil >= 0.25 else "overlapping"
            key_insights.append({
                'title': 'Group separation',
                'description': (
                    f"Silhouette score of the embedding w.r.t. '{self.label_col}' = {sil:.3f} — groups "
                    f"appear {sil_desc} in the projection."
                ),
                'status': 'positive' if sil >= 0.5 else 'neutral' if sil >= 0.25 else 'warning',
            })
        tw = self.results.get('trustworthiness')
        if tw is not None:
            key_insights.append({
                'title': 'Neighborhood preservation',
                'description': f"Trustworthiness = {tw:.3f}, continuity = {self.results.get('continuity'):.3f} "
                                "(both range 0-1; higher means the embedding's local neighborhoods faithfully "
                                "reflect the original feature space).",
                'status': 'positive' if tw >= 0.9 else 'neutral' if tw >= 0.75 else 'warning',
            })
        cd = self.results.get('cluster_detection')
        if cd:
            key_insights.append({
                'title': 'Automatic cluster detection',
                'description': f"HDBSCAN found {cd['n_clusters']} cluster(s) and {cd['n_noise']} noise point(s) "
                                f"({cd['noise_pct']:.1f}% of observations) directly on the embedding.",
                'status': 'neutral',
            })

        parts = []
        parts.append("**Overall Assessment**")
        parts.append(
            f"→ UMAP embedded {n_samples} observations across {len(self.variables)} input variables "
            f"into {self.n_components} dimensions using n_neighbors = {self.results['n_neighbors_used']}, "
            f"min_dist = {self.min_dist}, metric = '{self.metric}'."
        )
        if clamped:
            parts.append(
                f"→ Requested n_neighbors ({self.n_neighbors}) was reduced to "
                f"{self.results['n_neighbors_used']} to stay below the sample size."
            )

        parts.append("")
        parts.append("**Statistical Insights**")
        if sil is not None:
            sil_desc = "well separated" if sil >= 0.5 else "moderately separated" if sil >= 0.25 else "overlapping"
            parts.append(
                f"→ Silhouette score of the embedding w.r.t. '{self.label_col}' = {sil:.3f} — groups "
                f"appear {sil_desc} in the projection."
            )
        if tw is not None:
            parts.append(
                f"→ Trustworthiness = {tw:.3f}, continuity = {self.results.get('continuity'):.3f} — both range "
                "0-1 and measure, label-free, how faithfully the embedding's local neighborhoods reflect the "
                "original feature space (trustworthiness penalizes false neighbors introduced by the embedding, "
                "continuity penalizes true neighbors lost by it)."
            )
        if cd:
            parts.append(
                f"→ HDBSCAN detected {cd['n_clusters']} cluster(s) directly on the embedding, with "
                f"{cd['n_noise']} noise point(s) ({cd['noise_pct']:.1f}%)."
            )
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

        return {
            'key_insights': key_insights,
            'recommendation': "\n".join(parts),
        }

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

        # Flattened response: the frontend's UMAPResponse type expects every field
        # (n_samples, embedding, trustworthiness, ...) at the top level, not nested
        # under "results". `plot` is kept for backward compatibility; `embedding_plot`
        # is the field name the frontend actually reads.
        response = {
            **analysis.results,
            'plot': plot_image,
            'embedding_plot': plot_image,
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
