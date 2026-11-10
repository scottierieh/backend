"""Validates umap_analysis.py against umap.UMAP.
The handler fixes random_state=42 (deterministic, single-threaded), so the
embedding is exactly reproducible from an independent UMAP fit. The response is
flat (every field at the top level, as the frontend's UMAPResponse expects), not
nested under "results". Checked: echoed params (n_components, n_neighbors
requested/used, random_state, min_dist, metric, variables, n_features,
n_samples), the full embedding value-by-value (a sample of the first 20 points
x/y, plus global maxdiff), and every diagnostic recomputed independently on the
handler's OWN returned embedding: trustworthiness and continuity (sklearn
trustworthiness, arguments swapped for continuity, same k), per-dimension
coordinate statistics (numpy), feature-axis correlations (numpy corrcoef) and
the HDBSCAN cluster detection (sklearn.cluster.HDBSCAN refit)."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.cluster import HDBSCAN
import umap
from _pyharness import run_script, chk, report

rng = np.random.RandomState(1)
n = 90
base = np.vstack([rng.randn(30, 5) + c for c in ([0]*5, [5]*5, [-5]*5)])
cols = [f"v{i}" for i in range(5)]
data = [dict(zip(cols, row)) for row in base.tolist()]
payload = {"data": data, "variables": cols, "nComponents": 2,
           "nNeighbors": 15, "minDist": 0.1, "metric": "euclidean"}

out = run_script("umap_analysis.py", payload)
# The handler's response is flat (every field at the top level), not nested
# under "results" (changed with the full diagnostic contract, 660671d).
res = out
emb = np.array(res["embedding"])

# reference: same StandardScaler + UMAP params as the handler
scaled = StandardScaler().fit_transform(base)
eff_nn = min(15, max(2, n - 1))
ref = umap.UMAP(n_components=2, n_neighbors=eff_nn, min_dist=0.1,
                metric="euclidean", random_state=42)
ref_emb = ref.fit_transform(scaled)

# echoed scalar params
chk("n_neighbors_used", res["n_neighbors_used"], eff_nn)
chk("min_dist", res["min_dist"], 0.1, tol=1e-12)
chk("metric", res["metric"], "euclidean")
chk("n_components", res["n_components"], 2)
chk("n_samples", res["n_samples"], n)
chk("n_neighbors_requested", res["n_neighbors_requested"], 15)
chk("random_state", res["random_state"], 42)
chk("n_features", res["n_features"], len(cols))
chk("variables.len", len(res["variables"]), len(cols))
for i, c in enumerate(cols):
    chk(f"variables[{i}]", res["variables"][i], c)

# shape
chk("embedding.shape.rows", emb.shape[0], n)
chk("embedding.shape.cols", emb.shape[1], 2)

# per-point embedding coordinates, value-by-value (sample capped to first 20 points)
CAP = 20
print(f"LOG | checking embedding coordinates for first {CAP} of {n} points (value-by-value)")
for i in range(min(CAP, n)):
    chk(f"embedding[{i}].x", float(emb[i, 0]), float(ref_emb[i, 0]), tol=1e-3)
    chk(f"embedding[{i}].y", float(emb[i, 1]), float(ref_emb[i, 1]), tol=1e-3)
# global agreement over ALL points
chk("embedding.maxdiff", float(np.max(np.abs(emb - ref_emb))), 0.0, tol=1e-3)

# trustworthiness recomputed independently on the handler's OWN embedding
tw = trustworthiness(scaled, emb, n_neighbors=5)
chk("trustworthiness(own emb, k=5) in [0,1]", 1.0 if 0.0 <= tw <= 1.0 else 0.0, 1.0)
chk("trustworthiness(own emb, k=5) high", 1.0 if tw >= 0.9 else 0.0, 1.0)

# the handler's own diagnostics, recomputed independently on its returned embedding
k_tc = max(1, min(eff_nn, (n - 1) // 2 - 1))   # sklearn needs k < n/2
chk("diagnostics.trustworthiness", res["trustworthiness"], float(trustworthiness(scaled, emb, n_neighbors=k_tc)), tol=1e-12)
chk("diagnostics.continuity", res["continuity"], float(trustworthiness(emb, scaled, n_neighbors=k_tc)), tol=1e-12)
# umap-learn returns a float32 embedding and the handler takes its mean/sd in
# float32 (eps 1.2e-7, ~1e-6 relative after accumulation over n = 90 points);
# the reference is computed in float64 from the same values, so mean/sd agree to
# float32 precision (abs tol 1e-5 on coordinates of magnitude ~10), while
# min/max are exact element values.
for j in range(2):
    cs = res["coordinate_stats"][j]
    chk(f"coordinate_stats[{j}].component", cs["component"], f"UMAP-{j + 1}")
    chk(f"coordinate_stats[{j}].min", cs["min"], float(emb[:, j].min()), tol=1e-12)
    chk(f"coordinate_stats[{j}].max", cs["max"], float(emb[:, j].max()), tol=1e-12)
    chk(f"coordinate_stats[{j}].mean", cs["mean"], float(emb[:, j].mean()), tol=1e-5)
    chk(f"coordinate_stats[{j}].sd", cs["sd"], float(emb[:, j].std(ddof=1)), tol=1e-5)
for i, c in enumerate(cols):
    fc = res["feature_correlations"][i]
    chk(f"feature_correlations[{i}].feature", fc["feature"], c)
    for j in range(2):
        chk(f"feature_correlations[{i}].umap_{j + 1}", fc[f"umap_{j + 1}_correlation"],
            float(np.corrcoef(scaled[:, i], emb[:, j])[0, 1]), tol=1e-12)
cd = res["cluster_detection"]
mcs = max(3, min(int(n * 0.05), 25))
hl = HDBSCAN(min_cluster_size=mcs).fit_predict(emb)
chk("cluster_detection.min_cluster_size", cd["min_cluster_size"], mcs)
chk("cluster_detection.n_clusters", cd["n_clusters"], len(set(hl[hl != -1].tolist())))
chk("cluster_detection.n_noise", cd["n_noise"], int(np.sum(hl == -1)))
chk("cluster_detection.labels_identical", 1.0 if list(cd["labels"]) == hl.tolist() else 0.0, 1.0)

report("UMAP (umap-learn)")
