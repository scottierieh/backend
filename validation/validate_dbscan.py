import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,eps=0.8,min_samples=5)
r=run_script('dbscan_analysis.py',P)
res=r['results']

# ---- independent reference (DBSCAN is deterministic) ----
Xraw=df[feat]
Xs=StandardScaler().fit_transform(Xraw)
ref=DBSCAN(eps=0.8,min_samples=5).fit_predict(Xs)

labels=np.array(find_key(res,'labels'))
chk("dbscan.labels(ARI=1)", adjusted_rand_score(labels,ref), 1.0, tol=1e-12)
chk("dbscan.labels_exact", int((labels==ref).all()), 1)

n_clusters=len(set(ref))-(1 if -1 in ref else 0)
n_noise=int((ref==-1).sum())
chk("dbscan.n_clusters", res['n_clusters'], n_clusters)
chk("dbscan.n_noise", res['n_noise'], n_noise)
chk("dbscan.n_samples", res['n_samples'], 150)
chk("dbscan.eps", res['eps'], 0.8, tol=1e-12)
chk("dbscan.min_samples", res['min_samples'], 5)

# profiles per unique label (incl Noise), value-by-value
prof=res['profiles']
for label in sorted(set(ref)):
    name='Noise' if label==-1 else f'Cluster {label}'
    mask=ref==label
    p=prof[name]
    chk(f"dbscan.size[{name}]", p['size'], int(mask.sum()))
    chk(f"dbscan.pct[{name}]", p['percentage'], float(mask.sum()/150*100), tol=1e-9)
    ref_centroid=Xraw[mask].mean()
    for col in feat:
        chk(f"dbscan.centroid[{name}][{col}]", p['centroid'][col], ref_centroid[col], tol=1e-9)

report("DBSCAN (Python)")
