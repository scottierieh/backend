import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['y']=(iris.target==0).astype(int).map({1:'setosa',0:'other'})
P=dict(data=df.to_dict('records'),target_col='y',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=50,max_depth=-1,num_leaves=31,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,
       min_child_samples=20,reg_alpha=0.0,reg_lambda=0.0,random_state=42)
r=run_script('lightgbm_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['y']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=lgb.LGBMClassifier(n_estimators=50,max_depth=-1,num_leaves=31,learning_rate=0.1,subsample=1.0,
   colsample_bytree=1.0,min_child_samples=20,reg_alpha=0.0,reg_lambda=0.0,random_state=42,n_jobs=-1,verbosity=-1).fit(Xtr,le.fit_transform(ytr))
classifier_checks("lgb", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# ---- Method-specific block checks (STEP6 boosting page) ----
# Reproduce the backend's exact classifier pipeline: DataFrame X (feature
# names), eval_set + record_evaluation + early_stopping(20), binary_logloss.
cp=dict(n_estimators=50,max_depth=-1,num_leaves=31,learning_rate=0.1,subsample=1.0,
        colsample_bytree=1.0,min_child_samples=20,reg_alpha=0.0,reg_lambda=0.0,
        random_state=42,n_jobs=-1,verbosity=-1)
Xdf=df[feat].copy(); ydf=df['y'].copy()
Xtr2,Xte2,ytr2,yte2=train_test_split(Xdf,ydf,test_size=0.2,random_state=42,stratify=ydf)
le2=LabelEncoder(); ytr2e=le2.fit_transform(ytr2); yte2e=le2.transform(yte2)
mb=lgb.LGBMClassifier(**cp)
mb.fit(Xtr2,ytr2e,eval_set=[(Xtr2,ytr2e),(Xte2,yte2e)],eval_names=['train','test'],
       eval_metric='binary_logloss',callbacks=[lgb.early_stopping(20,verbose=False)])

# best_iteration
chk("lgb.best_iteration", find_key(r,'best_iteration'), int(mb.best_iteration_ or 50))

# feature_importance: split-count importances, pct, normalized, ranking order
imp=mb.feature_importances_.astype(float)
tot=imp.sum() if imp.sum()>0 else 1.0; mx=imp.max() if imp.max()>0 else 1.0
exp_fi={f:v for f,v in zip(feat,imp)}
fi=find_key(r,'feature_importance')
for d in fi:
    chk(f"lgb.fi[{d['feature']}].importance", d['importance'], exp_fi[d['feature']])
    chk(f"lgb.fi[{d['feature']}].importance_pct", d['importance_pct'], exp_fi[d['feature']]/tot*100)
    chk(f"lgb.fi[{d['feature']}].normalized_importance", d['normalized_importance'], exp_fi[d['feature']]/mx)
# ranking: sorted by importance descending, rank 1..n
exp_order=[f for f,_ in sorted(exp_fi.items(),key=lambda kv:kv[1],reverse=True)]
for i,d in enumerate(fi):
    chk(f"lgb.fi.rank[{i}]", d['rank'], i+1)
    chk(f"lgb.fi.order[{i}]", d['feature'], exp_order[i])

# per_class_metrics: precision/recall/f1/support vs sklearn on encoded labels
predb=mb.predict(Xte2)
pcm={d['class']:d for d in find_key(r,'per_class_metrics')}
for i,cls in enumerate(le2.classes_):
    d=pcm[str(cls)]
    chk(f"lgb.per_class[{cls}].precision", d['precision'],
        precision_score(yte2e,predb,labels=[i],average='macro',zero_division=0))
    chk(f"lgb.per_class[{cls}].recall", d['recall'],
        recall_score(yte2e,predb,labels=[i],average='macro',zero_division=0))
    chk(f"lgb.per_class[{cls}].f1_score", d['f1_score'],
        f1_score(yte2e,predb,labels=[i],average='macro',zero_division=0))
    chk(f"lgb.per_class[{cls}].support", d['support'], int((yte2e==i).sum()))

# metrics.auc: binary ROC-AUC on positive-class probability
proba=mb.predict_proba(Xte2)[:,1]
chk("lgb.metrics.auc", find_key(r,'metrics').get('auc'), float(roc_auc_score(yte2e,proba)), tol=1e-6)

# cv_results: StratifiedKFold(5, shuffle, rs=42) accuracy mean/std over full X
cvle=LabelEncoder(); ye=cvle.fit_transform(ydf)
sp=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
sc=cross_val_score(lgb.LGBMClassifier(**cp),Xdf,ye,cv=sp,scoring='accuracy')
cvr=find_key(r,'cv_results')
chk("lgb.cv_mean", cvr['cv_mean'], float(np.mean(sc)))
chk("lgb.cv_std", cvr['cv_std'], float(np.std(sc)))
chk("lgb.cv_folds", cvr['cv_folds'], 5)

report("LIGHTGBM (Python)")
