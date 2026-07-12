"""Validates tsne_analysis.py against sklearn.manifold.TSNE.
The script fixes random_state=42, init='pca', so the embedding + KL divergence
are exactly reproducible. Output embedding lives at results.embedding."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from _pyharness import run_script, chk, report, find_key

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

# reference: same StandardScaler + TSNE params
scaled = StandardScaler().fit_transform(base)
eff_perp = min(30.0, max(5.0, (n - 1) / 3))
ref = TSNE(n_components=2, perplexity=eff_perp, learning_rate='auto',
           max_iter=1000, random_state=42, init='pca')
ref_emb = ref.fit_transform(scaled)

chk("perplexity_used", res["perplexity_used"], eff_perp, tol=1e-9)
chk("n_samples", res["n_samples"], n)
chk("embedding.shape.rows", emb.shape[0], n)
chk("embedding.shape.cols", emb.shape[1], 2)
chk("kl_divergence", res["kl_divergence"], float(ref.kl_divergence_), tol=1e-4)
chk("embedding.maxdiff", float(np.max(np.abs(emb - ref_emb))), 0.0, tol=1e-3)

report("t-SNE (sklearn TSNE)")
