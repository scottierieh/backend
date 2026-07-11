import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,nClusters=3)
r=run_script('kmeans_analysis.py',P); inertia=find_key(r,'inertia'); labels=find_key(r,'labels')
Xs=StandardScaler().fit_transform(df[feat])
km=KMeans(n_clusters=3,init='k-means++',n_init=10,random_state=42).fit(Xs)
chk("km.inertia", inertia, km.inertia_, tol=1e-6)
chk("km.labels(ARI=1)", adjusted_rand_score(labels, km.labels_), 1.0, tol=1e-12)
report("KMEANS (Python)")
