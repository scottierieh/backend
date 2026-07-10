
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import f_oneway
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

warnings.filterwarnings('ignore')

try:
    from minisom import MiniSom
    HAS_MINISOM = True
except ImportError:
    HAS_MINISOM = False


class _NumpySom:
    """Minimal self-organizing map fallback used only if minisom is unavailable."""

    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, random_seed=42):
        self.x, self.y, self.input_len = x, y, input_len
        self.sigma, self.learning_rate = sigma, learning_rate
        rng = np.random.RandomState(random_seed)
        self.weights = rng.normal(size=(x, y, input_len)) * 0.1
        grid_x, grid_y = np.meshgrid(np.arange(x), np.arange(y), indexing='ij')
        self._grid = np.stack([grid_x, grid_y], axis=-1).astype(float)

    def _winner(self, sample):
        dists = np.linalg.norm(self.weights - sample, axis=-1)
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        return idx

    def winner(self, sample):
        return self._winner(sample)

    def train_random(self, data, num_iteration):
        rng = np.random.RandomState(42)
        n = len(data)
        for t in range(num_iteration):
            sample = data[rng.randint(0, n)]
            bmu = self._winner(sample)
            frac = t / max(num_iteration, 1)
            sigma_t = self.sigma * (1 - frac) + 0.3
            lr_t = self.learning_rate * (1 - frac) + 0.01
            bmu_pos = np.array(bmu, dtype=float)
            dist_sq = np.sum((self._grid - bmu_pos) ** 2, axis=-1)
            influence = np.exp(-dist_sq / (2 * sigma_t ** 2))
            self.weights += lr_t * influence[..., None] * (sample - self.weights)

    def quantization_error(self, data):
        errors = [np.linalg.norm(self.weights[self._winner(s)] - s) for s in data]
        return float(np.mean(errors))

    def get_weights(self):
        return self.weights


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

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def _humanize_feature(name):
    return str(name).replace('_', ' ').replace('-', ' ').strip().title()


def _get_bmu_matrix(som, grid_x, grid_y):
    """Return codebook weights as a (grid_x, grid_y, n_features) array regardless of backend."""
    return som.weights if hasattr(som, 'weights') else som.get_weights()


def _grid_dist(a, b):
    """Chebyshev distance between two (i, j) grid coordinates."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        features = payload.get('items') or payload.get('features')

        grid_x = int(payload.get('gridX', 6))
        grid_y = int(payload.get('gridY', 6))
        sigma = float(payload.get('sigma', 1.0))
        learning_rate = float(payload.get('learningRate', 0.5))
        num_iteration = int(payload.get('numIteration', 2000))

        if not data or not features:
            raise ValueError("Missing 'data' or 'features'")

        df = pd.DataFrame(data)
        clean_df = df[features].dropna()
        if clean_df.empty:
            raise ValueError("No valid data after removing missing values.")

        # Try to find an obvious id-like column for reporting representative samples.
        id_col = None
        for cand in ['id', 'ID', 'Id', 'customer_id', 'CustomerID', 'customer id']:
            if cand in df.columns:
                id_col = cand
                break

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df)
        n_samples, n_features = scaled_data.shape

        if HAS_MINISOM:
            som = MiniSom(grid_x, grid_y, n_features, sigma=sigma, learning_rate=learning_rate, random_seed=42)
            som.random_weights_init(scaled_data)
            som.train_random(scaled_data, num_iteration)
            q_error = float(som.quantization_error(scaled_data))
        else:
            som = _NumpySom(grid_x, grid_y, n_features, sigma=sigma, learning_rate=learning_rate)
            som.train_random(scaled_data, num_iteration)
            q_error = som.quantization_error(scaled_data)

        bmu_indices = np.array([som.winner(x) for x in scaled_data])
        cluster_ids = bmu_indices[:, 0] * grid_y + bmu_indices[:, 1]

        unique_clusters = np.unique(cluster_ids)
        n_effective_clusters = len(unique_clusters)

        weights = _get_bmu_matrix(som, grid_x, grid_y)
        codebook_flat = weights.reshape(-1, n_features)  # (grid_x*grid_y, n_features)

        # ---- Existing: per-cluster raw centroid profiles ----
        profiles = {}
        cluster_scaled_centroid = {}
        for cid in unique_clusters:
            mask = cluster_ids == cid
            cluster_data = clean_df[mask]
            cluster_scaled = scaled_data[mask]
            profiles[f'Node {int(cid)}'] = {
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / n_samples * 100),
                'centroid': cluster_data.mean().to_dict()
            }
            cluster_scaled_centroid[cid] = cluster_scaled.mean(axis=0)

        silhouette = None
        if n_effective_clusters > 1:
            silhouette = float(silhouette_score(scaled_data, cluster_ids))

        u_matrix = np.zeros((grid_x, grid_y))
        for i in range(grid_x):
            for j in range(grid_y):
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_x and 0 <= nj < grid_y:
                        neighbors.append(np.linalg.norm(weights[i, j] - weights[ni, nj]))
                u_matrix[i, j] = np.mean(neighbors) if neighbors else 0.0

        hit_map = np.zeros((grid_x, grid_y))
        for (bi, bj) in bmu_indices:
            hit_map[bi, bj] += 1

        results = {
            'grid_size': [grid_x, grid_y],
            'quantization_error': q_error,
            'n_nodes_activated': n_effective_clusters,
            'silhouette_score': silhouette,
            'node_assignments': cluster_ids.tolist(),
            'bmu_coordinates': bmu_indices.tolist(),
            'profiles': profiles,
            'features': features
        }
        results['interpretation'] = _generate_interpretation(
            grid_x, grid_y, n_effective_clusters, q_error, silhouette, profiles, n_samples
        )

        # =========================================================================
        # 1. Rule-based automatic cluster labeling
        # =========================================================================
        grand_mean = scaled_data.mean(axis=0)  # ~0 by construction of StandardScaler, kept explicit for clarity
        auto_labels = {}
        for cid in unique_clusters:
            z = cluster_scaled_centroid[cid] - grand_mean
            order = np.argsort(-np.abs(z))
            top_idx = order[:2]
            parts = []
            for idx in top_idx:
                direction = 'High' if z[idx] > 0 else 'Low'
                parts.append(f"{direction} {_humanize_feature(features[idx])}")
            label = ", ".join(parts) if parts else "Undifferentiated"
            auto_labels[f'Node {int(cid)}'] = {
                'auto_label': label,
                'label_basis': 'rule-based (top |z-score| deviations from grand mean)',
                'top_features': [
                    {'feature': features[idx], 'z_score': float(z[idx])} for idx in top_idx
                ]
            }
            profiles[f'Node {int(cid)}']['auto_label'] = label

        # =========================================================================
        # 11. centroid in z-score units
        # =========================================================================
        for cid in unique_clusters:
            z = cluster_scaled_centroid[cid]
            profiles[f'Node {int(cid)}']['centroid_zscore'] = {
                feat: float(z[i]) for i, feat in enumerate(features)
            }

        # =========================================================================
        # 2. Per-variable contribution to map separation (ANOVA F-stat across clusters)
        # =========================================================================
        feature_contribution = []
        for i, feat in enumerate(features):
            groups = [scaled_data[cluster_ids == cid, i] for cid in unique_clusters if (cluster_ids == cid).sum() > 0]
            groups = [g for g in groups if len(g) > 0]
            try:
                if len(groups) > 1 and any(len(g) > 1 for g in groups):
                    f_stat, p_val = f_oneway(*groups)
                    f_stat = float(f_stat) if np.isfinite(f_stat) else 0.0
                    p_val = float(p_val) if np.isfinite(p_val) else 1.0
                else:
                    f_stat, p_val = 0.0, 1.0
            except Exception:
                f_stat, p_val = 0.0, 1.0
            codebook_variance = float(np.var(codebook_flat[:, i]))
            feature_contribution.append({
                'feature': feat,
                'f_stat': f_stat,
                'p_value': p_val,
                'codebook_variance': codebook_variance
            })

        # Rank and bucket by quartiles of f_stat
        f_stats_sorted = sorted([f['f_stat'] for f in feature_contribution])
        ranks = np.argsort([-f['f_stat'] for f in feature_contribution])  # descending order indices
        rank_of = {idx: r for r, idx in enumerate(ranks)}
        n_feat = len(feature_contribution)
        for i, f in enumerate(feature_contribution):
            pct_rank = 1.0 - (rank_of[i] / max(n_feat - 1, 1))  # 1.0 = highest
            if pct_rank >= 0.75:
                bucket = 'Very High'
            elif pct_rank >= 0.5:
                bucket = 'High'
            elif pct_rank >= 0.25:
                bucket = 'Moderate'
            else:
                bucket = 'Low'
            f['contribution_level'] = bucket
            f['rank'] = int(rank_of[i]) + 1
        feature_contribution.sort(key=lambda x: x['rank'])

        # =========================================================================
        # 3. Component planes (top 8 features by contribution if > 8 total)
        # =========================================================================
        if n_feat <= 8:
            plane_features_idx = list(range(n_feat))
        else:
            ranked_feature_names = [f['feature'] for f in feature_contribution][:8]
            plane_features_idx = [features.index(f) for f in ranked_feature_names]

        n_planes = len(plane_features_idx)
        ncols = min(4, n_planes) if n_planes > 0 else 1
        nrows = int(np.ceil(n_planes / ncols)) if n_planes > 0 else 1
        component_planes_plot = None
        if n_planes > 0:
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
            axes_flat = np.array(axes).reshape(-1) if n_planes > 1 else [axes]
            for k, fidx in enumerate(plane_features_idx):
                plane = weights[:, :, fidx]  # z-scored codebook weights (documented: z-score units, not inverse-transformed)
                sns.heatmap(plane, cmap='coolwarm', center=0, ax=axes_flat[k],
                            cbar_kws={'label': 'z-score weight'})
                axes_flat[k].set_title(_humanize_feature(features[fidx]), fontsize=10)
                axes_flat[k].set_xlabel('Grid Y')
                axes_flat[k].set_ylabel('Grid X')
            for k in range(n_planes, len(axes_flat)):
                axes_flat[k].axis('off')
            fig.suptitle('Component Planes (standardized codebook weights per feature)', fontsize=14)
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            component_planes_plot = fig_to_base64(fig)

        # =========================================================================
        # 4. Cluster-to-cluster distance matrix (standardized centroid space)
        # =========================================================================
        cluster_names = [f'Node {int(cid)}' for cid in unique_clusters]
        centroid_matrix = np.array([cluster_scaled_centroid[cid] for cid in unique_clusters])
        dist_matrix = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
        cluster_distance_matrix = {
            'clusters': cluster_names,
            'matrix': dist_matrix.tolist()
        }
        closest_pair = None
        farthest_pair = None
        if len(cluster_names) > 1:
            iu = np.triu_indices(len(cluster_names), k=1)
            pair_dists = dist_matrix[iu]
            min_i = np.argmin(pair_dists)
            max_i = np.argmax(pair_dists)
            closest_pair = {
                'clusters': [cluster_names[iu[0][min_i]], cluster_names[iu[1][min_i]]],
                'distance': float(pair_dists[min_i])
            }
            farthest_pair = {
                'clusters': [cluster_names[iu[0][max_i]], cluster_names[iu[1][max_i]]],
                'distance': float(pair_dists[max_i])
            }
        cluster_distance_matrix['note'] = (
            f"Closest clusters: {' & '.join(closest_pair['clusters'])} (d={closest_pair['distance']:.2f}). "
            f"Farthest clusters: {' & '.join(farthest_pair['clusters'])} (d={farthest_pair['distance']:.2f})."
            if closest_pair and farthest_pair else "Only one active cluster; no pairwise distances to report."
        )

        # =========================================================================
        # 5. Representative sample per cluster (nearest to centroid in standardized space)
        # =========================================================================
        representative_samples = {}
        clean_index_list = clean_df.index.tolist()
        for cid in unique_clusters:
            mask = cluster_ids == cid
            idxs_in_cluster = np.where(mask)[0]
            centroid = cluster_scaled_centroid[cid]
            sub = scaled_data[idxs_in_cluster]
            dists = np.linalg.norm(sub - centroid, axis=1)
            best_local = idxs_in_cluster[np.argmin(dists)]
            orig_row_index = clean_index_list[best_local]
            row_values = clean_df.loc[orig_row_index].to_dict()
            rep = {
                'row_index': int(orig_row_index),
                'distance_to_centroid': float(dists.min()),
                'values': {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                           for k, v in row_values.items()}
            }
            if id_col is not None and id_col in df.columns:
                rep['id'] = _to_native_type(df.loc[orig_row_index, id_col])
            representative_samples[f'Node {int(cid)}'] = rep

        # =========================================================================
        # 6. Node density interpretation
        # =========================================================================
        hit_flat = hit_map.flatten()
        nonzero_hits = hit_flat[hit_flat > 0]
        if len(nonzero_hits) > 0:
            q25 = np.percentile(nonzero_hits, 25)
            q75 = np.percentile(nonzero_hits, 75)
        else:
            q25 = q75 = 0.0

        node_density = {'empty': 0, 'sparse': 0, 'typical': 0, 'dense': 0}
        total_nodes = grid_x * grid_y
        for h in hit_flat:
            if h == 0:
                node_density['empty'] += 1
            elif h < q25:
                node_density['sparse'] += 1
            elif h > q75:
                node_density['dense'] += 1
            else:
                node_density['typical'] += 1
        node_density_summary = {
            'counts': node_density,
            'percentages': {k: float(v / total_nodes * 100) for k, v in node_density.items()},
            'total_nodes': int(total_nodes)
        }

        # =========================================================================
        # 7. Outlier detection (distance to own BMU codebook vector)
        # =========================================================================
        bmu_vectors = np.array([weights[bi, bj] for (bi, bj) in bmu_indices])
        sample_bmu_dist = np.linalg.norm(scaled_data - bmu_vectors, axis=1)
        dist_mean, dist_std = sample_bmu_dist.mean(), sample_bmu_dist.std()
        outlier_threshold = dist_mean + 2 * dist_std
        outlier_mask = sample_bmu_dist > outlier_threshold
        outlier_local_idx = np.where(outlier_mask)[0]
        order_by_dist = outlier_local_idx[np.argsort(-sample_bmu_dist[outlier_local_idx])]
        capped = order_by_dist[:20]
        outliers = {
            'count': int(outlier_mask.sum()),
            'threshold': float(outlier_threshold),
            'rows': [
                {'row_index': int(clean_index_list[i]), 'distance': float(sample_bmu_dist[i])}
                for i in capped
            ],
            'note': (f"Showing top {len(capped)} of {int(outlier_mask.sum())} flagged outliers."
                     if outlier_mask.sum() > len(capped) else None)
        }

        # =========================================================================
        # 8. Cluster stability via bootstrap (ARI)
        # =========================================================================
        n_boot = 40
        rng = np.random.RandomState(123)
        ari_scores = []
        for b in range(n_boot):
            sample_idx = rng.randint(0, n_samples, size=n_samples)
            boot_data = scaled_data[sample_idx]
            boot_bmu = np.array([som.winner(x) for x in boot_data])
            boot_cluster_ids = boot_bmu[:, 0] * grid_y + boot_bmu[:, 1]
            original_labels_for_sample = cluster_ids[sample_idx]
            try:
                ari = adjusted_rand_score(original_labels_for_sample, boot_cluster_ids)
                ari_scores.append(ari)
            except Exception:
                pass
        cluster_stability = {
            'mean_ari': float(np.mean(ari_scores)) if ari_scores else None,
            'std_ari': float(np.std(ari_scores)) if ari_scores else None,
            'n_bootstrap': len(ari_scores),
            'note': 'Higher mean ARI (closer to 1.0) indicates more stable cluster assignments under resampling.'
        }

        # =========================================================================
        # 9. Variable correlation structure across component planes
        # =========================================================================
        codebook_df = pd.DataFrame(codebook_flat, columns=features)
        with np.errstate(invalid='ignore'):
            plane_corr = codebook_df.corr().fillna(0.0)
        pairs = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                r = float(plane_corr.iloc[i, j])
                pairs.append((features[i], features[j], r))
        pairs_sorted_pos = sorted(pairs, key=lambda x: -x[2])[:5]
        pairs_sorted_neg = sorted(pairs, key=lambda x: x[2])[:5]
        variable_correlation_highlights = {
            'strongest_positive': [
                {'feature_a': a, 'feature_b': b, 'r': r,
                 'note': f"{_humanize_feature(a)} and {_humanize_feature(b)} move together across the map (r={r:.2f})."}
                for a, b, r in pairs_sorted_pos if r > 0
            ],
            'strongest_negative': [
                {'feature_a': a, 'feature_b': b, 'r': r,
                 'note': f"{_humanize_feature(a)} and {_humanize_feature(b)} move oppositely across the map (r={r:.2f})."}
                for a, b, r in pairs_sorted_neg if r < 0
            ]
        }

        # =========================================================================
        # 10. Expanded quality metrics
        # =========================================================================
        # topographic_error
        topo_errors = 0
        for x in scaled_data:
            dists_to_codebook = np.linalg.norm(codebook_flat - x, axis=1)
            order2 = np.argsort(dists_to_codebook)
            best_idx, second_idx = order2[0], order2[1]
            best_pos = np.unravel_index(best_idx, (grid_x, grid_y))
            second_pos = np.unravel_index(second_idx, (grid_x, grid_y))
            manhattan = abs(best_pos[0] - second_pos[0]) + abs(best_pos[1] - second_pos[1])
            if manhattan != 1:
                topo_errors += 1
        topographic_error = float(topo_errors / n_samples)

        total_sq_qe = float(np.sum(sample_bmu_dist ** 2))
        total_variance = float(np.sum((scaled_data - scaled_data.mean(axis=0)) ** 2))
        explained_variance = float(1 - total_sq_qe / total_variance) if total_variance > 0 else None

        empty_node_rate = float(node_density['empty'] / total_nodes)
        average_hit_count = float(nonzero_hits.mean()) if len(nonzero_hits) > 0 else 0.0
        dead_unit_rate = empty_node_rate  # synonymous with empty_node_rate in this implementation

        # neighbor_preservation: for each sample, fraction of its k=5 NN (original space)
        # whose BMU grid position is within Chebyshev distance <= 1 of this sample's own BMU.
        k_nn = min(5, n_samples - 1)
        neighbor_preservation = None
        if k_nn > 0:
            pairwise = cdist(scaled_data, scaled_data, metric='euclidean')
            np.fill_diagonal(pairwise, np.inf)
            nn_idx = np.argsort(pairwise, axis=1)[:, :k_nn]
            preserved_fracs = []
            for i in range(n_samples):
                own_bmu = tuple(bmu_indices[i])
                neighbor_bmus = [tuple(bmu_indices[j]) for j in nn_idx[i]]
                preserved = sum(1 for nb in neighbor_bmus if _grid_dist(own_bmu, nb) <= 1)
                preserved_fracs.append(preserved / k_nn)
            neighbor_preservation = float(np.mean(preserved_fracs))

        # trustworthiness & continuity (Venna & Kaski) — skip gracefully for large n (O(n^2))
        trustworthiness = None
        continuity = None
        trust_cont_note = None
        if n_samples > 3000:
            trust_cont_note = "Skipped (trustworthiness/continuity) — sample size exceeds 3000, computation is O(n^2)."
        else:
            k_tc = min(5, n_samples - 1)
            if k_tc > 0:
                orig_dist = cdist(scaled_data, scaled_data, metric='euclidean')
                map_coords = bmu_indices.astype(float)
                map_dist = cdist(map_coords, map_coords, metric='euclidean')

                np.fill_diagonal(orig_dist, -1)
                orig_rank = np.argsort(np.argsort(orig_dist, axis=1), axis=1)
                np.fill_diagonal(orig_dist, np.inf)

                np.fill_diagonal(map_dist, -1)
                map_rank = np.argsort(np.argsort(map_dist, axis=1), axis=1)
                np.fill_diagonal(map_dist, np.inf)

                orig_nn = np.argsort(orig_dist, axis=1)[:, :k_tc]
                map_nn = np.argsort(map_dist, axis=1)[:, :k_tc]

                n = n_samples
                trust_penalty = 0.0
                cont_penalty = 0.0
                for i in range(n):
                    map_neighbors = set(map_nn[i].tolist())
                    orig_neighbors = set(orig_nn[i].tolist())
                    # trustworthiness: points that are map-neighbors but not orig-neighbors
                    for j in map_neighbors - orig_neighbors:
                        rank_in_orig = orig_rank[i, j]
                        trust_penalty += (rank_in_orig - k_tc)
                    # continuity: points that are orig-neighbors but not map-neighbors
                    for j in orig_neighbors - map_neighbors:
                        rank_in_map = map_rank[i, j]
                        cont_penalty += (rank_in_map - k_tc)

                norm = (2.0 / (n * k_tc * (2 * n - 3 * k_tc - 1))) if (2 * n - 3 * k_tc - 1) > 0 else 0.0
                trustworthiness = float(1 - norm * trust_penalty) if norm > 0 else None
                continuity = float(1 - norm * cont_penalty) if norm > 0 else None

        quality_metrics = {
            'topographic_error': topographic_error,
            'explained_variance': explained_variance,
            'empty_node_rate': empty_node_rate,
            'average_hit_count': average_hit_count,
            'dead_unit_rate': dead_unit_rate,
            'neighbor_preservation': neighbor_preservation,
            'trustworthiness': trustworthiness,
            'continuity': continuity,
            'trust_continuity_note': trust_cont_note
        }

        # ---- assemble additional results ----
        results['auto_labels'] = auto_labels
        results['feature_contribution'] = feature_contribution
        results['cluster_distance_matrix'] = cluster_distance_matrix
        results['representative_samples'] = representative_samples
        results['node_density'] = node_density_summary
        results['outliers'] = outliers
        results['cluster_stability'] = cluster_stability
        results['variable_correlation_highlights'] = variable_correlation_highlights
        results['quality_metrics'] = quality_metrics

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Self-Organizing Map (SOM) Analysis', fontsize=16)

        sns.heatmap(u_matrix, cmap='bone_r', ax=axes[0], cbar_kws={'label': 'Avg. Distance to Neighbors'})
        axes[0].set_title('U-Matrix (Node Distance Map)')
        axes[0].set_xlabel('Grid Y')
        axes[0].set_ylabel('Grid X')

        sns.heatmap(hit_map, annot=True, fmt='.0f', cmap='viridis', ax=axes[1], cbar_kws={'label': 'Hit Count'})
        axes[1].set_title('Hit Map (Samples per Node)')
        axes[1].set_xlabel('Grid Y')
        axes[1].set_ylabel('Grid X')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_image = fig_to_base64(fig)

        response = {
            'results': results,
            'plot': plot_image
        }
        if component_planes_plot is not None:
            response['component_planes_plot'] = component_planes_plot

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
