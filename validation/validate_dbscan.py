import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,eps=0.8,min_samples=5)
r=run_script('dbscan_analysis.py',P); labels=find_key(r,'labels'); nc=find_key(r,'n_clusters')
Xs=StandardScaler().fit_transform(df[feat]); ref=DBSCAN(eps=0.8,min_samples=5).fit_predict(Xs)
chk("dbscan.labels(ARI=1)", adjusted_rand_score(labels,ref), 1.0, tol=1e-12)
chk("dbscan.n_clusters", nc, len(set(ref))-(1 if -1 in ref else 0))
report("DBSCAN (Python)")
