# Validates kmedoids_analysis.py vs sklearn_extra KMedoids(method='pam').
# Requires numpy<2 (scikit-learn-extra is not importable under numpy 2.x).
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,nClusters=3)
r=run_script('kmedoids_analysis.py',P); labels=find_key(r,'labels')
Xs=StandardScaler().fit_transform(df[feat])
km=KMedoids(n_clusters=3,method='pam',init='k-medoids++',max_iter=300,random_state=42).fit(Xs)
chk("kmedoids.labels(ARI=1)", adjusted_rand_score(labels,km.labels_), 1.0, tol=1e-12)
report("KMEDOIDS (Python, numpy<2)")
