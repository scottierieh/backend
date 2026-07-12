import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.feature_selection import f_classif
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
P=dict(data=df.to_dict('records'),items=feat,nComponents=3,covarianceType='full',scalerType='standard')
r=run_script('gmm_analysis.py',P)
res=r['results']

# ---- independent reference (fixed seed/n_init -> reproducible) ----
Xraw=df[feat]
Xs=StandardScaler().fit_transform(Xraw)
g=GaussianMixture(n_components=3,covariance_type='full',n_init=10,random_state=42,max_iter=300).fit(Xs)
lab=g.predict(Xs)
proba=g.predict_proba(Xs)

labels=np.array(find_key(res,'labels'))
chk("gmm.labels(ARI=1)", adjusted_rand_score(labels,lab), 1.0, tol=1e-12)
chk("gmm.labels_exact", int((labels==lab).all()), 1)

# clustering_summary
cs=res['clustering_summary']
chk("gmm.n_components", cs['n_components'], 3)
chk("gmm.converged", cs['converged'], bool(g.converged_))
chk("gmm.n_iter", cs['n_iter'], int(g.n_iter_))
chk("gmm.n_samples", cs['n_samples'], 150)
for i in range(3):
    chk(f"gmm.weight[{i}]", cs['weights'][i], g.weights_[i], tol=1e-9)
    for j in range(len(feat)):
        chk(f"gmm.mean[{i}][{j}]", cs['means'][i][j], g.means_[i][j], tol=1e-6)

# final_metrics
fm=res['final_metrics']
chk("gmm.silhouette", fm['silhouette'], silhouette_score(Xs, lab), tol=1e-9)
chk("gmm.davies_bouldin", fm['davies_bouldin'], davies_bouldin_score(Xs, lab), tol=1e-9)
chk("gmm.calinski_harabasz", fm['calinski_harabasz'], calinski_harabasz_score(Xs, lab), tol=1e-6)
chk("gmm.bic", fm['bic'], float(g.bic(Xs)), tol=1e-3)
chk("gmm.aic", fm['aic'], float(g.aic(Xs)), tol=1e-3)
chk("gmm.log_likelihood", fm['log_likelihood'], float(g.score(Xs)*150), tol=1e-3)

# profiles: size, percentage, raw mean, weight, avg_probability
sizes=np.bincount(lab, minlength=3)
prof=res['profiles']
Rdf=Xraw.reset_index(drop=True)
for label in range(3):
    p=prof[f'Component {label+1}']
    mask=lab==label
    chk(f"gmm.size[{label}]", p['size'], int(sizes[label]))
    chk(f"gmm.pct[{label}]", p['percentage'], float(sizes[label]/150*100), tol=1e-9)
    chk(f"gmm.prof.weight[{label}]", p['weight'], g.weights_[label], tol=1e-9)
    chk(f"gmm.avg_prob[{label}]", p['avg_probability'], float(proba[mask,label].mean()), tol=1e-9)
    rm=Rdf[mask].mean()
    for col in feat:
        chk(f"gmm.prof.mean[{label}][{col}]", p['mean'][col], rm[col], tol=1e-9)

# soft diagnostics
sd=res['soft_diagnostics']
max_probs=proba.max(axis=1)
chk("gmm.sd.avg_probability", sd['avg_probability'], float(max_probs.mean()), tol=1e-9)
chk("gmm.sd.low_confidence_count", sd['low_confidence_count'], int((max_probs<0.6).sum()))
sp=np.sort(proba,axis=1)[:,::-1]
chk("gmm.sd.ambiguous_count", sd['ambiguous_count'], int(((sp[:,0]-sp[:,1])<0.1).sum()))

# feature drivers (ANOVA f_classif recomputed independently)
fstat,pval=f_classif(Rdf.values, lab)
fd={f['feature']:f for f in res['feature_drivers']['features']}
for i,col in enumerate(feat):
    chk(f"gmm.fdriver.f_stat[{col}]", fd[col]['f_stat'], float(fstat[i]), tol=1e-4)
    chk(f"gmm.fdriver.p_value[{col}]", fd[col]['p_value'], float(pval[i]), tol=1e-9)

# optimal_k search: bic/aic/silhouette per k, recommended_k
ok=res['optimal_k']
k_range=list(range(2, min(10,149)+1))
ref_bic=[]
for idx,k in enumerate(k_range):
    gk=GaussianMixture(n_components=k,covariance_type='full',n_init=10,random_state=42,max_iter=300).fit(Xs)
    ref_bic.append(float(gk.bic(Xs)))
    chk(f"gmm.optk.bic[{k}]", ok['bic_scores'][idx], float(gk.bic(Xs)), tol=1e-3)
    chk(f"gmm.optk.aic[{k}]", ok['aic_scores'][idx], float(gk.aic(Xs)), tol=1e-3)
    lk=gk.predict(Xs)
    ref_sil=silhouette_score(Xs,lk) if len(np.unique(lk))>1 else -1.0
    chk(f"gmm.optk.sil[{k}]", ok['silhouette_scores'][idx], ref_sil, tol=1e-9)
chk("gmm.optk.recommended_k", ok['recommended_k'], k_range[int(np.argmin(ref_bic))])

report("GMM (Python)")
