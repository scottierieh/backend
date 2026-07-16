import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       hidden_layer_sizes=[50],activation='relu',solver='adam',alpha=0.0001,learning_rate_init=0.001,max_iter=300,early_stopping=False,random_state=42)
r=run_script('mlp_analysis.py',P); acc=find_key(r,'accuracy')
Xs=StandardScaler().fit_transform(df[feat].values); y=df['species']
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=MLPClassifier(hidden_layer_sizes=(50,),activation='relu',solver='adam',alpha=0.0001,learning_rate_init=0.001,max_iter=300,early_stopping=False,random_state=42).fit(Xtr,le.fit_transform(ytr))
classifier_checks("mlp", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# ---- MLP-specific blocks (mirror what the STEP6 page shows) --------------
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

# 1) Training/convergence block: n_iterations (model.n_iter_), final_loss (model.loss_)
chk("mlp.n_iterations", find_key(r,'n_iterations'), int(m.n_iter_))
chk("mlp.final_loss", find_key(r,'final_loss'), float(m.loss_), tol=1e-9)

# 2) Per-class metrics block (precision/recall/f1/support in le.classes_ order)
ytr_enc=le.transform(ytr); yte_enc=le.transform(yte)
pred=m.predict(Xte)
prec=precision_score(yte_enc,pred,average=None,zero_division=0)
rec =recall_score(yte_enc,pred,average=None,zero_division=0)
f1  =f1_score(yte_enc,pred,average=None,zero_division=0)
pcm=find_key(r,'per_class_metrics') or []
by_class={d['class']:d for d in pcm}
for i,cls in enumerate(le.classes_):
    d=by_class.get(str(cls))
    if d is not None:
        chk(f"mlp.per_class[{cls}].precision", d['precision'], prec[i], tol=1e-9)
        chk(f"mlp.per_class[{cls}].recall",    d['recall'],    rec[i],  tol=1e-9)
        chk(f"mlp.per_class[{cls}].f1_score",  d['f1_score'],  f1[i],   tol=1e-9)
        chk(f"mlp.per_class[{cls}].support",   d['support'], int((yte_enc==i).sum()))

# 3) Macro ROC-AUC (one-vs-rest macro average over predict_proba)
auc_exp=float(roc_auc_score(yte_enc, m.predict_proba(Xte), multi_class='ovr', average='macro'))
if find_key(r,'auc') is not None:
    chk("mlp.auc", find_key(r,'auc'), auc_exp, tol=1e-9)

# 4) Cross-validation block: StratifiedKFold(shuffle, rs=42) accuracy on full scaled X
y_full_enc=LabelEncoder().fit_transform(y)
cv_model=MLPClassifier(hidden_layer_sizes=(50,),activation='relu',solver='adam',alpha=0.0001,
                       learning_rate_init=0.001,max_iter=300,early_stopping=False,random_state=42)
cv_split=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_scores=cross_val_score(cv_model, Xs, y_full_enc, cv=cv_split, scoring='accuracy')
cvr=find_key(r,'cv_results') or {}
chk("mlp.cv_mean", cvr.get('cv_mean'), float(np.mean(cv_scores)), tol=1e-9)
chk("mlp.cv_std",  cvr.get('cv_std'),  float(np.std(cv_scores)),  tol=1e-9)
chk("mlp.cv_folds",cvr.get('cv_folds'),5)
for i,s in enumerate(cv_scores):
    chk(f"mlp.cv_scores[{i}]", cvr['cv_scores'][i], float(s), tol=1e-9)

# 5) Permutation feature importance block (mean importance per feature, rank ordering)
perm=permutation_importance(m, Xte, yte_enc, n_repeats=10, random_state=42, n_jobs=-1)
exp_imp=dict(zip(feat, perm.importances_mean))
pi=find_key(r,'perm_importance') or []
for d in pi:
    if d['feature'] in exp_imp:
        chk(f"mlp.perm_importance[{d['feature']}].mean", d['importance_mean'], float(exp_imp[d['feature']]), tol=1e-9)
# rank ordering: emitted list must be sorted descending by importance_mean
ranked=sorted(feat, key=lambda fn: exp_imp[fn], reverse=True)
chk("mlp.perm_importance.rank_order", " > ".join(d['feature'] for d in pi), " > ".join(ranked))

report("MLP (Python)")
