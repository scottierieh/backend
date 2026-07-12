import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,linkageMethod='ward',distanceMetric='euclidean',nClusters=3)
r=run_script('hca_analysis.py',P)
res=r['results']

# ---- independent reference (linkage + fcluster are deterministic) ----
Xraw=df[feat]
Xs=StandardScaler().fit_transform(Xraw)
Z=linkage(Xs,method='ward'); ref=fcluster(Z,t=3,criterion='maxclust')

labels=np.array(find_key(res,'cluster_labels'))
chk("hca.labels(ARI=1)", adjusted_rand_score(labels,ref), 1.0, tol=1e-12)
chk("hca.labels_exact", int((labels==ref).all()), 1)
chk("hca.n_clusters", res['n_clusters'], 3)
chk("hca.linkage_method", res['linkage_method'], 'ward')
chk("hca.distance_metric", res['distance_metric'], 'euclidean')

# final_metrics recomputed independently from reproduced labels
fm=res['final_metrics']
chk("hca.silhouette", fm['silhouette'], silhouette_score(Xs, ref), tol=1e-9)
chk("hca.calinski_harabasz", fm['calinski_harabasz'], calinski_harabasz_score(Xs, ref), tol=1e-6)
chk("hca.davies_bouldin", fm['davies_bouldin'], davies_bouldin_score(Xs, ref), tol=1e-9)

# profiles: size, percentage, and per-feature centroid/std/min/max on raw data
prof=res['profiles']
Rdf=Xraw.reset_index(drop=True)
for label in sorted(set(ref)):
    mask=ref==label
    p=prof[f'Cluster {label}']
    sub=Rdf[mask]
    chk(f"hca.size[{label}]", p['size'], int(mask.sum()))
    chk(f"hca.pct[{label}]", p['percentage'], float(mask.sum()/150*100), tol=1e-9)
    m=sub.mean(); s=sub.std(); mn=sub.min(); mx=sub.max()
    for col in feat:
        chk(f"hca.centroid[{label}][{col}]", p['centroid'][col], m[col], tol=1e-9)
        chk(f"hca.std[{label}][{col}]", p['std'][col], s[col], tol=1e-9)
        chk(f"hca.min[{label}][{col}]", p['min'][col], mn[col], tol=1e-9)
        chk(f"hca.max[{label}][{col}]", p['max'][col], mx[col], tol=1e-9)

# NOTE: results['stability'] (mean/std) uses np.random bootstrap resampling with
# no fixed seed in the handler, so it is not reproducible value-by-value and is
# intentionally not chk()'d here.

report("HCA (Python)")
