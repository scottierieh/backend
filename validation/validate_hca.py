import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,linkageMethod='ward',distanceMetric='euclidean',nClusters=3)
r=run_script('hca_analysis.py',P); labels=find_key(r,'cluster_labels')
Xs=StandardScaler().fit_transform(df[feat])
Z=linkage(Xs,method='ward'); ref=fcluster(Z,t=3,criterion='maxclust')
chk("hca.labels(ARI=1)", adjusted_rand_score(labels,ref), 1.0, tol=1e-12)
report("HCA (Python)")
