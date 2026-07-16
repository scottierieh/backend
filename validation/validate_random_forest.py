import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=True,random_state=42,cv_folds=5)
r=run_script('random_forest_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
m=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=True,random_state=42,n_jobs=-1).fit(Xtr,ytr_e)
classifier_checks("rf", r, m, Xtr, ytr_e, Xte, yte_e)

# ---- method-specific blocks (mirror what STEP6 page shows) ----

# oob_score: sklearn's out-of-bag generalization estimate
chk("rf.oob_score", find_key(r,'oob_score'), float(m.oob_score_), tol=1e-9)

# feature_importance: Gini importance, importance_pct (imp/sum*100), normalized (imp/max), rank
imp=m.feature_importances_
total=imp.sum(); mx=imp.max()
fi=find_key(r,'feature_importance')
fi_by={d['feature']:d for d in fi}
for name,val in zip(feat,imp):
    d=fi_by[name]
    chk(f"rf.fi[{name}].importance", d['importance'], float(val), tol=1e-9)
    chk(f"rf.fi[{name}].importance_pct", d['importance_pct'], float(val/total*100), tol=1e-9)
    chk(f"rf.fi[{name}].normalized_importance", d['normalized_importance'], float(val/mx), tol=1e-9)
# ranking: descending by importance, rank 1 = most important feature
exp_order=[feat[i] for i in np.argsort(-imp, kind='stable')]
got_order=[d['feature'] for d in sorted(fi,key=lambda x:x['rank'])]
chk("rf.fi.rank_order", ",".join(got_order), ",".join(exp_order))

# cross-validation: StratifiedKFold(shuffle,rs=42) accuracy scores
y_all_e=LabelEncoder().fit_transform(y)
mcv=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,random_state=42,n_jobs=-1)
cv_split=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_scores=cross_val_score(mcv,X,y_all_e,cv=cv_split,scoring='accuracy')
chk("rf.cv_mean", find_key(r,'cv_mean'), float(np.mean(cv_scores)), tol=1e-9)
chk("rf.cv_std", find_key(r,'cv_std'), float(np.std(cv_scores)), tol=1e-9)
cv_got=find_key(r,'cv_scores')
for i,s in enumerate(cv_scores):
    chk(f"rf.cv_scores[{i}]", cv_got[i], float(s), tol=1e-9)

# permutation_importance: n_repeats=10, random_state=42 on the test set
perm=permutation_importance(m, Xte, yte_e, n_repeats=10, random_state=42, n_jobs=-1)
pm=find_key(r,'perm_importance')
pm_by={d['feature']:d for d in pm}
for name,mean,std in zip(feat, perm.importances_mean, perm.importances_std):
    chk(f"rf.perm[{name}].mean", pm_by[name]['importance_mean'], float(mean), tol=1e-9)
    chk(f"rf.perm[{name}].std", pm_by[name]['importance_std'], float(std), tol=1e-9)

report("RANDOM FOREST (Python)")
