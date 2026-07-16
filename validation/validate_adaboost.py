import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=50,learning_rate=1.0,base_max_depth=1,random_state=42)
r=run_script('adaboost_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); base=DecisionTreeClassifier(max_depth=1,random_state=42)
ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
m=AdaBoostClassifier(estimator=base,n_estimators=50,learning_rate=1.0,random_state=42).fit(Xtr,ytr_e)
classifier_checks("ada", r, m, Xtr, ytr_e, Xte, yte_e)

# ---- METHOD-SPECIFIC BLOCKS (boosting) ----------------------------------
# estimator_weights_ : AdaBoost alpha per weak learner (SAMME)
ew_got=r.get('estimator_weights'); ew_exp=list(m.estimator_weights_)
chk("ada.n_estimator_weights", len(ew_got), len(ew_exp))
for i,(g,e) in enumerate(zip(ew_got,ew_exp)):
    chk(f"ada.estimator_weight[{i}]", g, e, tol=1e-9)

# estimator_errors_ : weighted error of each weak learner
ee_got=r.get('estimator_errors'); ee_exp=list(m.estimator_errors_)
chk("ada.n_estimator_errors", len(ee_got), len(ee_exp))
for i,(g,e) in enumerate(zip(ee_got,ee_exp)):
    chk(f"ada.estimator_error[{i}]", g, e, tol=1e-9)

# staged scores : learning curve (accuracy per boosting round)
st_tr_got=r.get('staged_train_scores'); st_tr_exp=list(m.staged_score(Xtr,ytr_e))
st_te_got=r.get('staged_test_scores'); st_te_exp=list(m.staged_score(Xte,yte_e))
chk("ada.n_staged_train", len(st_tr_got), len(st_tr_exp))
chk("ada.n_staged_test", len(st_te_got), len(st_te_exp))
for i,(g,e) in enumerate(zip(st_tr_got,st_tr_exp)):
    chk(f"ada.staged_train[{i}]", g, e, tol=1e-9)
for i,(g,e) in enumerate(zip(st_te_got,st_te_exp)):
    chk(f"ada.staged_test[{i}]", g, e, tol=1e-9)

# feature_importance block : impurity importances, pct/normalized, sorted rank
fi_exp=m.feature_importances_
tot=fi_exp.sum() if fi_exp.sum()>0 else 1.0
mx=fi_exp.max() if fi_exp.max()>0 else 1.0
exp_by_name={f:fi_exp[i] for i,f in enumerate(feat)}
fi_got=r.get('feature_importance')
prev=None
for item in fi_got:
    name=item['feature']; e=exp_by_name[name]
    chk(f"ada.importance[{name}]", item['importance'], e, tol=1e-9)
    chk(f"ada.importance_pct[{name}]", item['importance_pct'], e/tot*100, tol=1e-9)
    chk(f"ada.norm_importance[{name}]", item['normalized_importance'], e/mx, tol=1e-9)
    if prev is not None: chk(f"ada.importance_sorted_desc[{name}]", item['importance']<=prev+1e-12, True)
    prev=item['importance']
chk("ada.importance_rank1", fi_got[0]['rank'], 1)
chk("ada.importance_rankN", fi_got[-1]['rank'], len(feat))

# per_class_metrics : precision/recall/f1/support per class
pred=m.predict(Xte)
pcm=r.get('per_class_metrics')
for row in pcm:
    idx=int(le.transform([row['class']])[0])
    yb=(yte_e==idx).astype(int); pb=(pred==idx).astype(int)
    chk(f"ada.pc_precision[{row['class']}]", row['precision'], precision_score(yb,pb,zero_division=0), tol=1e-9)
    chk(f"ada.pc_recall[{row['class']}]", row['recall'], recall_score(yb,pb,zero_division=0), tol=1e-9)
    chk(f"ada.pc_f1[{row['class']}]", row['f1_score'], f1_score(yb,pb,zero_division=0), tol=1e-9)
    chk(f"ada.pc_support[{row['class']}]", row['support'], int((yte_e==idx).sum()))

# auc : multiclass macro OvR
proba=m.predict_proba(Xte)
chk("ada.auc", find_key(r,'auc'), roc_auc_score(yte_e,proba,multi_class='ovr',average='macro'), tol=1e-9)

# cv_results : StratifiedKFold(shuffle,rs=42) accuracy across full data
le2=LabelEncoder(); y_enc=le2.fit_transform(y)
cvm=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,random_state=42),
                       n_estimators=50,learning_rate=1.0,random_state=42)
splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cvs=cross_val_score(cvm,X,y_enc,cv=splitter,scoring='accuracy')
cv=r.get('cv_results')
chk("ada.cv_mean", cv['cv_mean'], float(np.mean(cvs)), tol=1e-9)
chk("ada.cv_std", cv['cv_std'], float(np.std(cvs)), tol=1e-9)
for i,(g,e) in enumerate(zip(cv['cv_scores'],cvs)):
    chk(f"ada.cv_score[{i}]", g, e, tol=1e-9)

report("ADABOOST (Python)")
