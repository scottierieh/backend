import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,nComponents=3,covarianceType='full',scalerType='standard')
r=run_script('gmm_analysis.py',P); labels=find_key(r,'labels')
Xs=StandardScaler().fit_transform(df[feat])
g=GaussianMixture(n_components=3,covariance_type='full',n_init=10,random_state=42,max_iter=300).fit(Xs)
chk("gmm.labels(ARI=1)", adjusted_rand_score(labels,g.predict(Xs)), 1.0, tol=1e-12)
chk("gmm.n_components", find_key(r,'n_components'), 3)
chk("gmm.bic", find_key(r,'bic'), float(g.bic(Xs)), tol=1e-3)
chk("gmm.aic", find_key(r,'aic'), float(g.aic(Xs)), tol=1e-3)
report("GMM (Python)")
