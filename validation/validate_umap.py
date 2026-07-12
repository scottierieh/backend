"""Validates umap_analysis.py against umap.UMAP.
The script fixes random_state=42 (deterministic, single-threaded), so the
embedding is exactly reproducible. Output embedding at results.embedding."""
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# reference: same StandardScaler + UMAP params
scaled = StandardScaler().fit_transform(base)
eff_nn = min(15, max(2, n - 1))
ref = umap.UMAP(n_components=2, n_neighbors=eff_nn, min_dist=0.1,
                metric="euclidean", random_state=42)
ref_emb = ref.fit_transform(scaled)

chk("n_neighbors_used", res["n_neighbors_used"], eff_nn)
chk("n_samples", res["n_samples"], n)
chk("embedding.shape.rows", emb.shape[0], n)
chk("embedding.shape.cols", emb.shape[1], 2)
chk("embedding.maxdiff", float(np.max(np.abs(emb - ref_emb))), 0.0, tol=1e-3)

report("UMAP (umap-learn)")
