"""Validates umap_analysis.py against umap.UMAP.
The handler fixes random_state=42 (deterministic, single-threaded), so the
embedding is exactly reproducible from an independent UMAP fit. Output embedding
lives at results.embedding. Every returned leaf value is checked: echoed params
(n_components, n_neighbors_used, min_dist, metric, variables, n_samples), the
full embedding value-by-value (a sample of the first 20 points x/y, plus global
maxdiff), and trustworthiness recomputed independently with sklearn on the
handler's OWN returned embedding."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
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
res = out["results"]
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

report("UMAP (umap-learn)")
