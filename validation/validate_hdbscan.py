import numpy as np, pandas as pd, hdbscan
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,min_cluster_size=10)
r=run_script('hdbscan_analysis.py',P)
res=r['results']

# ---- independent reference (HDBSCAN is deterministic) ----
Xraw=df[feat]
Xs=StandardScaler().fit_transform(Xraw)
clu=hdbscan.HDBSCAN(min_cluster_size=10,min_samples=None,gen_min_span_tree=True).fit(Xs)
ref=clu.labels_
refprob=clu.probabilities_

labels=np.array(find_key(res,'labels'))
chk("hdbscan.labels(ARI=1)", adjusted_rand_score(labels,ref), 1.0, tol=1e-12)
chk("hdbscan.labels_exact", int((labels==ref).all()), 1)

n_clusters=len(set(ref))-(1 if -1 in ref else 0)
n_noise=int((ref==-1).sum())
chk("hdbscan.n_clusters", res['n_clusters'], n_clusters)
chk("hdbscan.n_noise", res['n_noise'], n_noise)
chk("hdbscan.n_samples", res['n_samples'], 150)
chk("hdbscan.min_cluster_size", res['min_cluster_size'], 10)

# probabilities value-by-value
prob=np.array(res['probabilities'])
chk("hdbscan.prob_len", len(prob), 150)
chk("hdbscan.prob_maxdiff", float(np.abs(prob-refprob).max()), 0.0, tol=1e-9)

# profiles per unique label (incl Noise), value-by-value
prof=res['profiles']
Rdf=Xraw.reset_index(drop=True)
for label in sorted(set(ref)):
    name='Noise' if label==-1 else f'Cluster {label}'
    mask=ref==label
    p=prof[name]
    chk(f"hdbscan.size[{name}]", p['size'], int(mask.sum()))
    chk(f"hdbscan.pct[{name}]", p['percentage'], float(mask.sum()/150*100), tol=1e-9)
    rm=Rdf[mask].mean()
    for col in feat:
        chk(f"hdbscan.centroid[{name}][{col}]", p['centroid'][col], rm[col], tol=1e-9)

report("HDBSCAN (Python)")
