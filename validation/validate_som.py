# Validates som_analysis.py (Self-Organizing Map). SOM training is stochastic
# (minisom seed, or the deterministic numpy fallback when minisom is absent), so
# the codebook weights are not independently reproducible here. Instead every
# returned quantity that is a DERIVED function of the handler's own node
# assignments / BMU coordinates + the (deterministic) StandardScaler'd data is
# recomputed independently and checked value-by-value: node-assignment
# consistency, silhouette, per-node profiles (size, percentage, raw centroid,
# z-score centroid), auto-label z-scores, per-feature ANOVA F/p contribution,
# the cluster-distance matrix, representative samples, node-density buckets,
# neighbor preservation, and the finite/range sanity of the weight-dependent
# metrics (quantization error, topographic error, explained variance,
# trustworthiness, continuity). Where minisom is installed the quantization
# error is additionally reproduced exactly from a direct MiniSom fit.
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
from scipy.spatial.distance import cdist
from _pyharness import run_script, chk, report

iris = load_iris(as_frame=True)
df = iris.frame.copy()
feat = list(iris.feature_names)
gx, gy, sigma, lr = 4, 4, 1.0, 0.5
payload = {'data': df.to_dict('records'), 'features': feat,
           'gridX': gx, 'gridY': gy, 'sigma': sigma, 'learning_rate': lr}
res = run_script('som_analysis.py', payload)
r = res['results'] if 'results' in res else res

Xs = StandardScaler().fit_transform(df[feat].values)
n_samples, n_feat = Xs.shape
assign = np.asarray(r['node_assignments'])
bmu = np.asarray(r['bmu_coordinates'])
unique = np.unique(assign)

# ---- echoed params / structural ----
chk("som.grid_size[0]", r['grid_size'][0], gx)
chk("som.grid_size[1]", r['grid_size'][1], gy)
chk("som.features.len", len(r['features']), n_feat)
for i, f in enumerate(feat):
    chk(f"som.features[{i}]", r['features'][i], f)
chk("som.assignments_count", len(assign), n_samples)
chk("som.bmu_count", len(bmu), n_samples)
chk("som.n_nodes_activated", r['n_nodes_activated'], len(unique))
chk("som.nodes_within_grid", 1.0 if r['n_nodes_activated'] <= gx * gy else 0.0, 1.0)

# node_assignments must equal bmu_x*gy + bmu_y (sample first 20, logged cap)
CAP = 20
print(f"LOG | checking node_assignment==bmu consistency for first {CAP} of {n_samples} points")
for i in range(min(CAP, n_samples)):
    chk(f"som.assign_from_bmu[{i}]", int(assign[i]), int(bmu[i, 0] * gy + bmu[i, 1]))

# ---- silhouette (recomputed independently on handler inputs) ----
chk("som.silhouette", r['silhouette_score'], float(silhouette_score(Xs, assign)), tol=1e-9)

# ---- weight-dependent metrics: finite / in-range sanity (weights not reproducible here) ----
qe = r['quantization_error']
chk("som.qerror_finite_nonneg", 1.0 if np.isfinite(qe) and qe >= 0 else 0.0, 1.0)
qm = r['quality_metrics']
te = qm['topographic_error']
chk("som.topographic_error_in[0,1]", 1.0 if 0.0 <= te <= 1.0 else 0.0, 1.0)
ev = qm['explained_variance']
chk("som.explained_variance<=1", 1.0 if ev is None or ev <= 1.0 + 1e-9 else 0.0, 1.0)
for key in ('trustworthiness', 'continuity', 'neighbor_preservation'):
    v = qm[key]
    chk(f"som.{key}_in[0,1]", 1.0 if v is None or (0.0 <= v <= 1.0 + 1e-9) else 0.0, 1.0)

# ---- per-node profiles: size, percentage, raw centroid, z-score centroid ----
grand_mean = Xs.mean(axis=0)
scaled_centroid = {}
print(f"LOG | checking per-node profiles for all {len(unique)} activated nodes")
for cid in unique:
    mask = assign == cid
    key = f'Node {int(cid)}'
    prof = r['profiles'][key]
    chk(f"som.{key}.size", prof['size'], int(mask.sum()))
    chk(f"som.{key}.percentage", prof['percentage'], float(mask.sum() / n_samples * 100), tol=1e-9)
    raw_centroid = df[feat][mask].mean()
    for f in feat:
        chk(f"som.{key}.centroid[{f}]", prof['centroid'][f], float(raw_centroid[f]), tol=1e-9)
    zc = Xs[mask].mean(axis=0)
    scaled_centroid[cid] = zc
    for i, f in enumerate(feat):
        chk(f"som.{key}.centroid_zscore[{f}]", prof['centroid_zscore'][f], float(zc[i]), tol=1e-9)

# ---- auto_labels: top-2 |z| deviation z-scores ----
print("LOG | checking auto_label top-feature z-scores for all activated nodes")
for cid in unique:
    key = f'Node {int(cid)}'
    z = scaled_centroid[cid] - grand_mean
    top_idx = np.argsort(-np.abs(z))[:2]
    got = r['auto_labels'][key]['top_features']
    for k, idx in enumerate(top_idx):
        chk(f"som.{key}.auto.top[{k}].feature", got[k]['feature'], feat[idx])
        chk(f"som.{key}.auto.top[{k}].z_score", got[k]['z_score'], float(z[idx]), tol=1e-9)

# ---- feature_contribution: ANOVA F-stat and p-value per feature ----
fc = {f['feature']: f for f in r['feature_contribution']}
print(f"LOG | checking feature_contribution F/p for all {n_feat} features")
for i, f in enumerate(feat):
    groups = [Xs[assign == cid, i] for cid in unique if (assign == cid).sum() > 0]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) > 1 and any(len(g) > 1 for g in groups):
        fs, pv = f_oneway(*groups)
        fs = float(fs) if np.isfinite(fs) else 0.0
        pv = float(pv) if np.isfinite(pv) else 1.0
    else:
        fs, pv = 0.0, 1.0
    chk(f"som.fc[{f}].f_stat", fc[f]['f_stat'], fs, tol=1e-6)
    chk(f"som.fc[{f}].p_value", fc[f]['p_value'], pv, tol=1e-9)

# ---- cluster_distance_matrix (standardized centroid space) ----
names = [f'Node {int(cid)}' for cid in unique]
cmat = np.array([scaled_centroid[cid] for cid in unique])
dmat = cdist(cmat, cmat, metric='euclidean')
got_mat = np.array(r['cluster_distance_matrix']['matrix'])
chk("som.cdm.clusters", r['cluster_distance_matrix']['clusters'] == names, True)
RCAP = min(8, len(names))
print(f"LOG | checking cluster_distance_matrix first {RCAP}x{len(names)} cells")
for i in range(RCAP):
    for j in range(len(names)):
        chk(f"som.cdm[{i},{j}]", float(got_mat[i, j]), float(dmat[i, j]), tol=1e-9)

# ---- representative samples: row index + distance to standardized centroid ----
print("LOG | checking representative_samples for all activated nodes")
for cid in unique:
    key = f'Node {int(cid)}'
    idxs = np.where(assign == cid)[0]
    d = np.linalg.norm(Xs[idxs] - scaled_centroid[cid], axis=1)
    best = idxs[np.argmin(d)]
    rep = r['representative_samples'][key]
    chk(f"som.{key}.rep.row_index", rep['row_index'], int(best))
    chk(f"som.{key}.rep.dist", rep['distance_to_centroid'], float(d.min()), tol=1e-9)

# ---- node density buckets (from BMU hit map over full grid) ----
hit = np.zeros((gx, gy))
for (bi, bj) in bmu:
    hit[bi, bj] += 1
hit_flat = hit.flatten()
nz = hit_flat[hit_flat > 0]
q25, q75 = (np.percentile(nz, 25), np.percentile(nz, 75)) if len(nz) else (0.0, 0.0)
dens = {'empty': 0, 'sparse': 0, 'typical': 0, 'dense': 0}
for h in hit_flat:
    if h == 0: dens['empty'] += 1
    elif h < q25: dens['sparse'] += 1
    elif h > q75: dens['dense'] += 1
    else: dens['typical'] += 1
total = gx * gy
nd = r['node_density']
chk("som.nd.total_nodes", nd['total_nodes'], total)
for k in ('empty', 'sparse', 'typical', 'dense'):
    chk(f"som.nd.counts[{k}]", nd['counts'][k], dens[k])
    chk(f"som.nd.pct[{k}]", nd['percentages'][k], float(dens[k] / total * 100), tol=1e-9)
chk("som.qm.empty_node_rate", qm['empty_node_rate'], float(dens['empty'] / total), tol=1e-9)
chk("som.qm.dead_unit_rate", qm['dead_unit_rate'], float(dens['empty'] / total), tol=1e-9)
chk("som.qm.average_hit_count", qm['average_hit_count'],
    float(nz.mean()) if len(nz) else 0.0, tol=1e-9)

# ---- neighbor_preservation recomputed independently (k=5 NN, Chebyshev<=1 on BMU grid) ----
k_nn = min(5, n_samples - 1)
pw = cdist(Xs, Xs, metric='euclidean')
np.fill_diagonal(pw, np.inf)
nn_idx = np.argsort(pw, axis=1)[:, :k_nn]
fracs = []
for i in range(n_samples):
    own = bmu[i]
    fracs.append(sum(1 for j in nn_idx[i]
                     if max(abs(own[0] - bmu[j][0]), abs(own[1] - bmu[j][1])) <= 1) / k_nn)
chk("som.qm.neighbor_preservation", qm['neighbor_preservation'], float(np.mean(fracs)), tol=1e-9)

# ---- exact quantization-error reproduction when minisom is available (deploy env) ----
try:
    from minisom import MiniSom
    som = MiniSom(gx, gy, n_feat, sigma=sigma, learning_rate=lr, random_seed=42)
    som.random_weights_init(Xs)
    som.train_random(Xs, 2000)  # handler default numIteration
    chk("som.quantization_error(minisom)", qe, float(som.quantization_error(Xs)), tol=1e-6)
except ImportError:
    print("SKIP | minisom not installed — exact quantization-error reproduction skipped (deploy env has it)")

report("SOM (Python)")
