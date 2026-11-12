import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),features=feat,target='species',model='random_forest',search_method='grid',cv_folds=5,scoring='accuracy')
r=run_script('hyperparameter_tuning_analysis.py',P)
Xs=StandardScaler().fit_transform(df[feat].values); y=df['species']
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.3,random_state=42,stratify=y)
grid={'n_estimators':[100,200,300],'max_depth':[5,10,None],'min_samples_split':[2,5,10]}
gs=GridSearchCV(RandomForestClassifier(random_state=42),grid,cv=5,scoring='accuracy',n_jobs=-1,refit=True).fit(Xtr,ytr)
chk("hp.best_cv_score", find_key(r,'best_cv_score'), gs.best_score_, tol=1e-9)
bp=find_key(r,'best_params') or {}
chk("hp.best_n_estimators", bp.get('n_estimators'), gs.best_params_['n_estimators'])
chk("hp.best_max_depth", str(bp.get('max_depth')), str(gs.best_params_['max_depth']))
chk("hp.best_min_samples_split", bp.get('min_samples_split'), gs.best_params_['min_samples_split'])
# --- candidates (exploration tab): every cv_results_ row ---
cand=r.get('candidates') or []
cvres=gs.cv_results_
chk("hp.n_candidates", len(cand), len(cvres['params']))
chk("hp.scoring", r.get('scoring'), 'accuracy')
chk("hp.greater_is_better", r.get('greater_is_better'), True)
chk("hp.param_names", str(sorted(r.get("param_names") or [])), str(sorted(grid.keys())))
def _norm(p): return {k:(None if v is None else v) for k,v in p.items()}
best=[c for c in cand if c['rank']==1]
chk("hp.best_rank_params==best_params", any(_norm(c['params'])==_norm(bp) for c in best), True)
chk("hp.best_rank_params==reference_best", any(_norm(c['params'])==_norm(gs.best_params_) for c in best), True)
mx=0.0; mxf=0.0; mxs=0.0; rank_bad=0
for i,c in enumerate(cand):
    mx=max(mx,abs(c['mean_score']-cvres['mean_test_score'][i]))
    mxs=max(mxs,abs(c['std_score']-cvres['std_test_score'][i]))
    mxf=max(mxf,max(abs(c['fold_scores'][k]-cvres[f'split{k}_test_score'][i]) for k in range(5)))
    rank_bad+=int(c['rank']!=int(cvres['rank_test_score'][i]))
chk("hp.candidates.max|mean_score-ref|", mx, 0.0, tol=1e-9)
chk("hp.candidates.max|std_score-ref|", mxs, 0.0, tol=1e-9)
chk("hp.candidates.max|fold_score-ref|", mxf, 0.0, tol=1e-9)
chk("hp.candidates.rank_mismatches", rank_bad, 0)
chk("hp.candidates.fold_mean==mean_score", max(abs(np.mean(c['fold_scores'])-c['mean_score']) for c in cand), 0.0, tol=1e-12)
chk("hp.best_cv_score==max mean_score", max(c['mean_score'] for c in cand), find_key(r,'best_cv_score'), tol=1e-12)
report("HYPERPARAMETER TUNING (Python)")
