"""Validates tsne_analysis.py against sklearn.manifold.TSNE.
The handler fixes random_state=42, init='pca', learning_rate='auto', so the
embedding + KL divergence are exactly reproducible from an independent TSNE fit.
Output embedding lives at results.embedding. Every returned leaf value is
checked: echoed params (n_components, perplexity_used, variables, n_samples),
n_iterations_run, kl_divergence, the full embedding value-by-value (a sample of
the first 20 points x/y, plus global maxdiff), and trustworthiness recomputed
independently with sklearn on the handler's OWN returned embedding."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness
from _pyharness import run_script, chk, report

rng = np.random.RandomState(0)
n = 90
# three separated blobs in 5D
base = np.vstack([rng.randn(30, 5) + c for c in ([0]*5, [4]*5, [-4]*5)])
cols = [f"v{i}" for i in range(5)]
data = [dict(zip(cols, row)) for row in base.tolist()]
payload = {"data": data, "variables": cols, "nComponents": 2,
           "perplexity": 30.0, "nIter": 1000}

out = run_script("tsne_analysis.py", payload)
res = out["results"]
emb = np.array(res["embedding"])

# reference: same StandardScaler + TSNE params as the handler
scaled = StandardScaler().fit_transform(base)
eff_perp = min(30.0, max(5.0, (n - 1) / 3))
ref = TSNE(n_components=2, perplexity=eff_perp, learning_rate='auto',
           max_iter=1000, random_state=42, init='pca')
ref_emb = ref.fit_transform(scaled)

# echoed scalar params
chk("perplexity_used", res["perplexity_used"], eff_perp, tol=1e-9)
chk("n_components", res["n_components"], 2)
chk("n_samples", res["n_samples"], n)
chk("n_iterations_run", res["n_iterations_run"], int(ref.n_iter_))
chk("variables.len", len(res["variables"]), len(cols))
for i, c in enumerate(cols):
    chk(f"variables[{i}]", res["variables"][i], c)

# shape
chk("embedding.shape.rows", emb.shape[0], n)
chk("embedding.shape.cols", emb.shape[1], 2)

# KL divergence (exactly reproducible)
chk("kl_divergence", res["kl_divergence"], float(ref.kl_divergence_), tol=1e-4)

# per-point embedding coordinates, value-by-value (sample capped to first 20 points)
CAP = 20
print(f"LOG | checking embedding coordinates for first {CAP} of {n} points (value-by-value)")
for i in range(min(CAP, n)):
    chk(f"embedding[{i}].x", float(emb[i, 0]), float(ref_emb[i, 0]), tol=1e-3)
    chk(f"embedding[{i}].y", float(emb[i, 1]), float(ref_emb[i, 1]), tol=1e-3)
# global agreement over ALL points
chk("embedding.maxdiff", float(np.max(np.abs(emb - ref_emb))), 0.0, tol=1e-3)

# trustworthiness recomputed independently on the handler's OWN embedding
# (reproducible derived quantity; the handler does not report it but it must be
# well-defined and high for a faithful embedding of separated blobs).
tw = trustworthiness(scaled, emb, n_neighbors=5)
chk("trustworthiness(own emb, k=5) in [0,1]", 1.0 if 0.0 <= tw <= 1.0 else 0.0, 1.0)
chk("trustworthiness(own emb, k=5) high", 1.0 if tw >= 0.9 else 0.0, 1.0)

report("t-SNE (sklearn TSNE)")
