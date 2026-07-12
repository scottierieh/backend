import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,nClusters=3)
r=run_script('kmeans_analysis.py',P)
res=r['results']

# ---- independent reference (KMeans is deterministic under fixed seed) ----
Xraw=df[feat]
Xs=StandardScaler().fit_transform(Xraw)
km=KMeans(n_clusters=3,init='k-means++',n_init=10,random_state=42).fit(Xs)
lab=km.labels_

# labels reproduce exactly (same seed/params) -> ARI==1
labels=np.array(find_key(res,'labels'))
chk("km.labels(ARI=1)", adjusted_rand_score(labels, lab), 1.0, tol=1e-12)
chk("km.labels_exact", int((labels==lab).all()), 1)

# clustering_summary
cs=res['clustering_summary']
chk("km.n_clusters", cs['n_clusters'], 3)
chk("km.inertia", cs['inertia'], km.inertia_, tol=1e-6)
# centroids (scaled cluster centers), value-by-value
cen=np.array(cs['centroids'])
for i in range(3):
    for j in range(len(feat)):
        chk(f"km.centroid[{i}][{j}]", cen[i][j], km.cluster_centers_[i][j], tol=1e-6)

# final_metrics (recomputed independently from reproduced labels)
fm=res['final_metrics']
chk("km.silhouette", fm['silhouette'], silhouette_score(Xs, lab), tol=1e-9)
chk("km.davies_bouldin", fm['davies_bouldin'], davies_bouldin_score(Xs, lab), tol=1e-9)
chk("km.calinski_harabasz", fm['calinski_harabasz'], calinski_harabasz_score(Xs, lab), tol=1e-6)

# profiles: sizes, percentages, raw-space centroids
sizes=np.bincount(lab, minlength=3)
prof=res['profiles']
for label in range(3):
    p=prof[f'Cluster {label+1}']
    chk(f"km.size[{label}]", p['size'], int(sizes[label]))
    chk(f"km.pct[{label}]", p['percentage'], float(sizes[label]/150*100), tol=1e-9)
    ref_centroid=Xraw[lab==label].mean()
    for col in feat:
        chk(f"km.rawcentroid[{label}][{col}]", p['centroid'][col], ref_centroid[col], tol=1e-9)

# optimal_k search: k_range, inertias, silhouette_scores, recommended_k
ok=res['optimal_k']
k_range=list(range(2, min(11, 150)))
chk("km.optk.k_range_len", len(ok['k_range']), len(k_range))
ref_inertias=[]; ref_sils=[]
for k in k_range:
    m=KMeans(n_clusters=k,init='k-means++',n_init=10,random_state=42).fit(Xs)
    ref_inertias.append(m.inertia_)
    ref_sils.append(silhouette_score(Xs, m.labels_) if len(np.unique(m.labels_))>1 else -1)
for idx,k in enumerate(k_range):
    chk(f"km.optk.k[{idx}]", ok['k_range'][idx], k)
    chk(f"km.optk.inertia[{idx}]", ok['inertias'][idx], ref_inertias[idx], tol=1e-6)
    chk(f"km.optk.sil[{idx}]", ok['silhouette_scores'][idx], ref_sils[idx], tol=1e-9)
chk("km.optk.recommended_k", ok['recommended_k'], k_range[int(np.argmax(ref_sils))])

report("KMEANS (Python)")
