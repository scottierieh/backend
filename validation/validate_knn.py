import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from _pyharness import run_script, chk, report, classifier_checks, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,'task_type':'classification',
         'test_size':0.2,'n_neighbors':5,'weights':'uniform','metric':'minkowski','p':2,'find_optimal_k':False,'random_state':42,'cv_folds':5}
r=run_script('knn_analysis.py',payload); r=r.get('results',r); m=r['metrics']
Xs=StandardScaler().fit_transform(df[feat].values); y=df['species']
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski',p=2).fit(Xtr,ytr_e)
classifier_checks("knn", r, mdl, Xtr, ytr_e, Xte, yte_e)

# ── Method-specific block 1: cross_validation (StratifiedKFold on full scaled X) ──
# backend perform_cross_validation: StratifiedKFold(n_splits=cv_folds, shuffle, rs=42)
yfull=LabelEncoder().fit_transform(y)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_scores=cross_val_score(KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski',p=2),
                          Xs,yfull,cv=cv,scoring='accuracy')
cvr=find_key(r,'cv_results')
chk("knn.cv_mean", cvr['cv_mean'], float(np.mean(cv_scores)))
chk("knn.cv_std",  cvr['cv_std'],  float(np.std(cv_scores)))
for i,s in enumerate(cv_scores):
    chk(f"knn.cv_scores[{i}]", cvr['cv_scores'][i], float(s))

# ── Method-specific block 2: per-class precision/recall/f1/support ──
pred=mdl.predict(Xte)
cr=classification_report(yte_e,pred,target_names=[str(c) for c in le.classes_],
                         output_dict=True,zero_division=0)
pcm={d['class']:d for d in find_key(r,'per_class_metrics')}
for cls in le.classes_:
    ref=cr[str(cls)]; got=pcm[str(cls)]
    chk(f"knn.per_class[{cls}].precision", got['precision'], ref['precision'])
    chk(f"knn.per_class[{cls}].recall",    got['recall'],    ref['recall'])
    chk(f"knn.per_class[{cls}].f1_score",  got['f1_score'],  ref['f1-score'])
    chk(f"knn.per_class[{cls}].support",   got['support'],   int(ref['support']))

# ── Method-specific block 3: permutation feature_importance ──
perm=permutation_importance(mdl,Xte,yte_e,n_repeats=10,random_state=42,n_jobs=1)
ref_imp=dict(zip(feat,perm.importances_mean)); ref_std=dict(zip(feat,perm.importances_std))
fi={d['feature']:d for d in find_key(r,'feature_importance')}
for f in feat:
    chk(f"knn.feature_importance[{f}].importance", fi[f]['importance'], float(ref_imp[f]), tol=1e-6)
    chk(f"knn.feature_importance[{f}].std",        fi[f]['std'],        float(ref_std[f]), tol=1e-6)

# ── Method-specific block 4: optimal-K search (find_optimal_k) ──
# separate payload enabling the CV k-search the STEP6 page plots
krange=[1,3,5,7,9,11,13,15]
kp=dict(payload); kp['find_optimal_k']=True; kp['k_range']=krange
rk=run_script('knn_analysis.py',kp); rk=rk.get('results',rk)
ksr=find_key(rk,'k_search_result')
kres=[]
for k in krange:
    if k>len(Xtr)-1: continue
    kcv=StratifiedKFold(n_splits=min(5,len(Xtr)//k+1),shuffle=True,random_state=42)
    s=cross_val_score(KNeighborsClassifier(n_neighbors=k,weights='uniform',metric='minkowski',p=2),
                      Xtr,ytr_e,cv=kcv,scoring='accuracy')
    kres.append({'k':k,'mean':float(np.mean(s)),'std':float(np.std(s))})
best=max(range(len(kres)),key=lambda i:kres[i]['mean'])
chk("knn.optimal_k",     ksr['optimal_k'],     kres[best]['k'])
chk("knn.optimal_score", ksr['optimal_score'], kres[best]['mean'])
for i,kr in enumerate(kres):
    chk(f"knn.k_scores[{kr['k']}].mean", ksr['k_scores'][i]['mean_score'], kr['mean'])
    chk(f"knn.k_scores[{kr['k']}].std",  ksr['k_scores'][i]['std_score'],  kr['std'])

report("KNN (Python)")
