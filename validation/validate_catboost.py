import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['y']=(iris.target==0).astype(int).map({1:'setosa',0:'other'})
P=dict(data=df.to_dict('records'),target_col='y',feature_cols=feat,task_type='classification',test_size=0.2,
       iterations=100,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_state=42)
r=run_script('catboost_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['y']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=CatBoostClassifier(iterations=100,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_seed=42,verbose=False,allow_writing_files=False).fit(Xtr,le.fit_transform(ytr))
classifier_checks("cat", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# ---- METHOD-SPECIFIC boosting blocks (independently recomputed) ----
# Faithfully reproduce the backend's classifier training: Pool + eval_set +
# use_best_model with the default early_stopping_rounds=20 (payload omits it, so
# the handler applies its default of 20). This drives best_iteration, native
# feature_importance (PredictionValuesChange), per-class metrics, and SHAP.
Xdf=df[feat]; ys=df['y']
Xtr2,Xte2,ytr2,yte2=train_test_split(Xdf,ys,test_size=0.2,random_state=42,stratify=ys)
Xtr2=Xtr2.reset_index(drop=True); Xte2=Xte2.reset_index(drop=True)
ytr2=ytr2.reset_index(drop=True); yte2=yte2.reset_index(drop=True)
le2=LabelEncoder(); ytr2e=le2.fit_transform(ytr2); yte2e=le2.transform(yte2)
_common=dict(iterations=100,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_seed=42,verbose=False,allow_writing_files=False)
mb=CatBoostClassifier(**_common)
trp=Pool(Xtr2,ytr2e,cat_features=[]); tep=Pool(Xte2,yte2e,cat_features=[])
mb.fit(trp,eval_set=tep,use_best_model=True,early_stopping_rounds=20)

# best_iteration
chk("cat.best_iteration", find_key(r,'best_iteration'),
    int(mb.get_best_iteration() or 100))

# feature_importance: value, importance_pct, normalized_importance and rank order
fi=np.asarray(mb.get_feature_importance(trp))
fi_map=dict(zip(feat,fi))
tot=fi.sum() if fi.sum()>0 else 1.0; mx=fi.max() if fi.max()>0 else 1.0
fi_list=find_key(r,'feature_importance')
emit={d['feature']:d for d in fi_list}
for fname,val in fi_map.items():
    chk(f"cat.fi[{fname}].importance", emit[fname]['importance'], float(val), tol=1e-6)
    chk(f"cat.fi[{fname}].importance_pct", emit[fname]['importance_pct'], float(val/tot*100), tol=1e-6)
    chk(f"cat.fi[{fname}].normalized_importance", emit[fname]['normalized_importance'], float(val/mx), tol=1e-6)
# ranking: sorted descending by importance
exp_rank={f:i+1 for i,f in enumerate(sorted(fi_map,key=lambda k:fi_map[k],reverse=True))}
for fname in feat:
    chk(f"cat.fi[{fname}].rank", emit[fname]['rank'], exp_rank[fname])

# per_class_metrics (precision/recall/f1/support) aligned to LabelEncoder class order
predb=mb.predict(tep).ravel().astype(int)
pr,rc,f1,sp=precision_recall_fscore_support(yte2e,predb,labels=range(len(le2.classes_)),zero_division=0)
pcm=find_key(r,'per_class_metrics'); pcm_map={d['class']:d for d in pcm}
for i,cls in enumerate(le2.classes_):
    d=pcm_map[str(cls)]
    chk(f"cat.per_class[{cls}].precision", d['precision'], float(pr[i]), tol=1e-9)
    chk(f"cat.per_class[{cls}].recall", d['recall'], float(rc[i]), tol=1e-9)
    chk(f"cat.per_class[{cls}].f1", d['f1_score'], float(f1[i]), tol=1e-9)
    chk(f"cat.per_class[{cls}].support", d['support'], int(sp[i]))

# SHAP importance (native CatBoost exact SHAP, bias column dropped, mean|.|)
raw=np.array(mb.get_feature_importance(tep,type='ShapValues'))
ms=np.abs(raw[:,:,:-1]).mean(axis=(0,1)) if raw.ndim==3 else np.abs(raw[:,:-1]).mean(axis=0)
shap_map=dict(zip(feat,ms))
shap_emit={d['feature']:d for d in find_key(r,'shap_importance')}
for fname,val in shap_map.items():
    chk(f"cat.shap[{fname}].mean_abs_shap", shap_emit[fname]['mean_abs_shap'], float(val), tol=1e-6)

# cross-validation (StratifiedKFold, shuffle, seed=42) accuracy mean/std
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
ye=LabelEncoder().fit_transform(ys)
cvs=[]
for tri,tei in cv.split(Xdf,ye):
    mm=CatBoostClassifier(**_common)
    mm.fit(Pool(Xdf.iloc[tri],ye[tri],cat_features=[]),verbose=False)
    cvs.append(accuracy_score(ye[tei],mm.predict(Pool(Xdf.iloc[tei],cat_features=[])).ravel().astype(int)))
cvs=np.array(cvs)
cvr=find_key(r,'cv_results')
chk("cat.cv_mean", cvr['cv_mean'], float(np.mean(cvs)), tol=1e-9)
chk("cat.cv_std", cvr['cv_std'], float(np.std(cvs)), tol=1e-9)

report("CATBOOST (Python)")
