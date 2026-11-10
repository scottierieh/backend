# Validates kmedoids_analysis.py (PAM via scikit-learn-extra KMedoids).
# Run it in the pinned deploy environment (requirements.txt: numpy==1.26.4,
# scikit-learn==1.5.2, scikit-learn-extra==0.3.0 — scikit-learn-extra's compiled
# extensions target the numpy 1.x ABI).
#
# Two independent lines of reference:
#  1. Library reproduction: sklearn_extra KMedoids(method='pam') refit with the
#     handler's exact settings (StandardScaler, euclidean, k-medoids++ start,
#     random_state=42, max_iter=300) -> labels, medoid indices, cost, and the
#     whole k=2..10 search.
#  2. From first principles with numpy/scipy on an independently standardized
#     matrix z = (x - mean) / sd (population SD, as StandardScaler):
#       - medoids are actual data rows (raw and scaled values, value-by-value);
#       - every point is assigned to its nearest returned medoid (labels);
#       - the reported cost equals sum_i min_m ||z_i - z_m|| (KMedoids' inertia_
#         is the sum of *unsquared* distances, the PAM objective);
#       - PAM's defining property: no single medoid <-> non-medoid swap lowers
#         that cost (the result is swap-optimal);
#       - exhaustive search over all C(150,3) medoid triples gives the global
#         minimum of the k=3 objective; the reported cost must not be below it.
#         PAM (BUILD/k-medoids++ start + SWAP) only guarantees swap-optimality,
#         not the global optimum: on iris (k=3) every KMedoids init and R's
#         cluster::pam (BUILD+SWAP) stop at medoids rows {7,55,112}, cost
#         131.7958, while the exhaustive optimum is {7,94,147}, cost 130.7333
#         (it needs two simultaneous swaps). So equality is not the PAM spec;
#         the gap is printed for the record;
#       - silhouette (overall and per cluster), Davies-Bouldin and
#         Calinski-Harabasz from their textbook formulas;
#       - cluster sizes / percentages / raw-space means; per-feature one-way
#         ANOVA F, p (scipy.stats.f_oneway) and eta^2.
from itertools import combinations
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn_extra.cluster import KMedoids
from _pyharness import run_script, chk, report

iris = load_iris(as_frame=True); df = iris.frame.copy(); feat = list(iris.feature_names)
# space-free feature keys for check names ('sepal length (cm)' -> 'sepal_length')
SHORT = {c: c.split(' (')[0].replace(' ', '_') for c in feat}
P = dict(data=df.to_dict('records'), items=feat, nClusters=3)
res = run_script('kmedoids_analysis.py', P)['results']
cs = res['clustering_summary']
K = 3
Xraw = df[feat].to_numpy(dtype=float)
n, p = Xraw.shape

labels = np.asarray(cs['labels'])
med = np.asarray(cs['medoid_indices'])

# ---- independent standardization (population SD, i.e. StandardScaler) ----
Z = (Xraw - Xraw.mean(axis=0)) / Xraw.std(axis=0, ddof=0)
D = cdist(Z, Z)                       # full euclidean distance matrix


def silhouette_def(lab):
    """s_i = (b_i - a_i) / max(a_i, b_i); 0 for singletons."""
    uq = np.unique(lab)
    size = np.array([(lab == lab[i]).sum() for i in range(n)])
    a = np.array([D[i, lab == lab[i]].sum() / max(size[i] - 1, 1) for i in range(n)])
    b = np.array([min(D[i, lab == c].mean() for c in uq if c != lab[i]) for i in range(n)])
    return np.where(size > 1, (b - a) / np.maximum(a, b), 0.0)


def calinski_harabasz_def(lab):
    """[B/(k-1)] / [W/(n-k)] with cluster means in scaled space."""
    uq = np.unique(lab); k = len(uq); gm = Z.mean(axis=0)
    B = sum((lab == c).sum() * np.sum((Z[lab == c].mean(axis=0) - gm) ** 2) for c in uq)
    W = sum(np.sum((Z[lab == c] - Z[lab == c].mean(axis=0)) ** 2) for c in uq)
    return float((B / (k - 1)) / (W / (n - k)))


# ---- 1. library reproduction ----
km = KMedoids(n_clusters=K, method='pam', init='k-medoids++', max_iter=300,
              random_state=42, metric='euclidean').fit(Z)
chk("kmedoids.labels(ARI=1)", adjusted_rand_score(labels, km.labels_), 1.0, tol=1e-12)
chk("kmedoids.labels_exact.vs_KMedoids", int((labels == km.labels_).all()), 1)
for i in range(K):
    chk(f"kmedoids.medoid_index[{i}].vs_KMedoids", int(med[i]), int(km.medoid_indices_[i]))
chk("kmedoids.cost.vs_KMedoids_inertia", cs['inertia'], float(km.inertia_), tol=1e-9)
chk("kmedoids.n_iter.vs_KMedoids", cs['n_iter'], int(km.n_iter_))
chk("kmedoids.converged(n_iter<max_iter)", int(cs['n_iter'] < cs['max_iter']), 1)
chk("kmedoids.n_clusters", cs['n_clusters'], K)

# ---- 2. first principles ----
# medoids are data rows: raw values (clustering_summary.medoids)
for i in range(K):
    for j, col in enumerate(feat):
        chk(f"kmedoids.medoid[{i}][{SHORT[col]}]", cs['medoids'][i][col], Xraw[med[i], j], tol=1e-12)
# nearest-medoid assignment
# Tolerance for costs recomputed here with exact distances (scipy cdist): the
# handler's cost comes from scikit-learn's euclidean_distances, which uses
# ||x||^2 + ||m||^2 - 2 x.m, so a medoid's distance to itself is sqrt(rounding
# residual) ~ 1e-8 .. 6e-8 instead of 0 (checked: 1.05e-8 at k=6, 5.96e-8 at
# k=8). With up to k such terms the summed cost differs by O(k * 1e-8).
COST_TOL = 1e-6
Dm = D[:, med]                        # n x K distances to the returned medoids
chk("kmedoids.labels_exact.nearest_medoid", int((labels == Dm.argmin(axis=1)).all()), 1)
# PAM objective: sum of (unsquared) distances to the nearest medoid
cost = float(Dm.min(axis=1).sum())
chk("kmedoids.cost.sum_of_distances", cs['inertia'], cost, tol=COST_TOL)

# swap optimality: the best single medoid <-> non-medoid swap cannot beat the result
best_swap = np.inf
non_med = np.setdiff1d(np.arange(n), med)
for a in range(K):
    base = D[:, np.delete(med, a)].min(axis=1)
    best_swap = min(best_swap, float(np.minimum(base[:, None], D[:, non_med]).sum(axis=0).min()))
chk("kmedoids.swap_optimal(best_swap>=cost)", 1.0 if best_swap >= cost - 1e-9 else 0.0, 1.0)

# exhaustive global minimum of the k=3 objective over all C(n,3) medoid triples
best = np.inf
trip = np.array(list(combinations(range(n), K)), dtype=np.int32)
for s in range(0, len(trip), 20000):
    t = trip[s:s + 20000]
    c = np.minimum(np.minimum(D[:, t[:, 0]], D[:, t[:, 1]]), D[:, t[:, 2]]).sum(axis=0)
    best = min(best, float(c.min()))
print(f"LOG | k=3 PAM cost {cs['inertia']:.10g} vs exhaustive global minimum {best:.10g} (gap {cs['inertia'] - best:.6g})")
chk("kmedoids.cost>=exhaustive_global_min", 1.0 if cs['inertia'] >= best - COST_TOL else 0.0, 1.0)

# internal validity indices from their definitions
uniq = np.unique(labels)
sil = silhouette_def(labels)
fm = res['final_metrics']
chk("kmedoids.silhouette", fm['silhouette'], float(sil.mean()), tol=1e-9)
chk("kmedoids.calinski_harabasz", fm['calinski_harabasz'], calinski_harabasz_def(labels), tol=1e-6)
# Davies-Bouldin = mean_i max_{j!=i} (S_i + S_j) / ||mu_i - mu_j||, S = mean distance to cluster mean
mu = np.array([Z[labels == c].mean(axis=0) for c in uniq])
S = np.array([np.linalg.norm(Z[labels == c] - mu[k], axis=1).mean() for k, c in enumerate(uniq)])
R = [max((S[i] + S[j]) / np.linalg.norm(mu[i] - mu[j]) for j in range(len(uniq)) if j != i)
     for i in range(len(uniq))]
chk("kmedoids.davies_bouldin", fm['davies_bouldin'], float(np.mean(R)), tol=1e-9)

# per-cluster silhouette summary
pcs = {d['cluster']: d for d in res['per_cluster_silhouette']}
for c in uniq:
    d = pcs[f'Cluster {c + 1}']; s = sil[labels == c]
    chk(f"kmedoids.sil[{c + 1}].n", d['n'], int(s.size))
    chk(f"kmedoids.sil[{c + 1}].avg", d['avg_sil'], float(s.mean()), tol=1e-9)
    chk(f"kmedoids.sil[{c + 1}].min", d['min_sil'], float(s.min()), tol=1e-9)
    chk(f"kmedoids.sil[{c + 1}].max", d['max_sil'], float(s.max()), tol=1e-9)
    chk(f"kmedoids.sil[{c + 1}].negative", d['negative_sil'], int((s < 0).sum()))

# profiles: size, %, medoid index, scaled medoid row, raw-space cluster means
prof = res['profiles']
for c in uniq:
    pr = prof[f'Cluster {c + 1}']; m = labels == c
    chk(f"kmedoids.size[{c + 1}]", pr['size'], int(m.sum()))
    chk(f"kmedoids.pct[{c + 1}]", pr['percentage'], float(m.sum() / n * 100), tol=1e-9)
    chk(f"kmedoids.profile_medoid_index[{c + 1}]", pr['medoid_index'], int(med[c]))
    for j, col in enumerate(feat):
        chk(f"kmedoids.medoid_scaled[{c + 1}][{SHORT[col]}]", pr['medoid_scaled'][col], float(Z[med[c], j]), tol=1e-9)
        chk(f"kmedoids.rawmean[{c + 1}][{SHORT[col]}]", pr['centroid'][col], float(Xraw[m, j].mean()), tol=1e-9)

# feature drivers: one-way ANOVA across clusters on raw values, eta^2 = SSB/SST
fd = {f['feature']: f for f in res['feature_drivers']['features']}
for j, col in enumerate(feat):
    groups = [Xraw[labels == c, j] for c in uniq]
    F, pv = stats.f_oneway(*groups)
    ssb = sum(len(g) * (g.mean() - Xraw[:, j].mean()) ** 2 for g in groups)
    sst = ((Xraw[:, j] - Xraw[:, j].mean()) ** 2).sum()
    chk(f"kmedoids.anova[{SHORT[col]}].F", fd[col]['f_stat'], float(F), tol=1e-6)
    chk(f"kmedoids.anova[{SHORT[col]}].p", fd[col]['p_value'], float(pv), tol=1e-12)
    chk(f"kmedoids.anova[{SHORT[col]}].eta2", fd[col]['eta_squared'], float(ssb / sst), tol=1e-9)

# k search (k = 2..10): PAM cost by library refit and from the refit's medoids;
# silhouette and CH from their definitions on the refit's labels
ok = res['optimal_k']
k_range = list(range(2, 11))
chk("kmedoids.optk.k_range_len", len(ok['k_range']), len(k_range))
ref_sil = []; ref_ch = []
for ik, k in enumerate(k_range):
    mk = KMedoids(n_clusters=k, method='pam', init='k-medoids++', max_iter=300,
                  random_state=42, metric='euclidean').fit(Z)
    chk(f"kmedoids.optk.k[{ik}]", ok['k_range'][ik], k)
    chk(f"kmedoids.optk.cost[{ik}].vs_KMedoids", ok['inertias'][ik], float(mk.inertia_), tol=1e-9)
    chk(f"kmedoids.optk.cost[{ik}].sum_of_distances", ok['inertias'][ik],
        float(D[:, mk.medoid_indices_].min(axis=1).sum()), tol=COST_TOL)
    ref_sil.append(float(silhouette_def(mk.labels_).mean()))
    ref_ch.append(calinski_harabasz_def(mk.labels_))
    chk(f"kmedoids.optk.sil[{ik}]", ok['silhouette_scores'][ik], ref_sil[-1], tol=1e-9)
    chk(f"kmedoids.optk.ch[{ik}]", ok['ch_scores'][ik], ref_ch[-1], tol=1e-6)
# recommended k: majority of silhouette max, CH max and the elbow (max distance
# from the chord joining the first and last points of the cost curve)
sil_k = k_range[int(np.argmax(ref_sil))]; ch_k = k_range[int(np.argmax(ref_ch))]
x = np.array(k_range, float); y = np.array(ok['inertias'], float)
dist = (np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
        / np.hypot(y[-1] - y[0], x[-1] - x[0]))
elbow_k = int(x[int(np.argmax(dist))])
votes = [sil_k, ch_k, elbow_k]
rd = ok['recommendation_detail']['votes']
chk("kmedoids.optk.silhouette_k", rd['silhouette_k'], sil_k)
chk("kmedoids.optk.ch_k", rd['calinski_harabasz_k'], ch_k)
chk("kmedoids.optk.elbow_k", rd['elbow_k'], elbow_k)
chk("kmedoids.optk.recommended_k", ok['recommended_k'], max(set(votes), key=votes.count))

report("KMEDOIDS (Python, numpy<2)")
