import numpy as np, pandas as pd, hdbscan
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,min_cluster_size=10)
r=run_script('hdbscan_analysis.py',P); labels=find_key(r,'labels'); nc=find_key(r,'n_clusters')
Xs=StandardScaler().fit_transform(df[feat])
ref=hdbscan.HDBSCAN(min_cluster_size=10,min_samples=None).fit_predict(Xs)
chk("hdbscan.labels(ARI=1)", adjusted_rand_score(labels,ref), 1.0, tol=1e-12)
chk("hdbscan.n_clusters", nc, len(set(ref))-(1 if -1 in ref else 0))
report("HDBSCAN (Python)")
