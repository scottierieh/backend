import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from _pyharness import run_script, chk, report, classifier_checks, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,'test_size':0.2,
         'nb_type':'gaussian','var_smoothing':1e-9,'random_state':42,'cv_folds':5}
r=run_script('naive_bayes_analysis.py',payload); r=r.get('results',r); m=r['metrics']
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=GaussianNB(var_smoothing=1e-9).fit(Xtr,ytr_e)
classifier_checks("nb", r, mdl, Xtr, ytr_e, Xte, yte_e)

# ── METHOD-SPECIFIC BLOCKS (STEP6 page) ──────────────────────────────
classes=[str(c) for c in le.classes_]

# 1. Class priors — GaussianNB.class_prior_ (training class frequencies)
cp=find_key(r,'class_priors')
for i,c in enumerate(classes):
    chk(f"nb.class_prior[{c}]", cp.get(c), mdl.class_prior_[i], tol=1e-9)

# 2. Feature importance — gaussian: 1/(mean_var+1e-10) normalized, sorted desc
avg_var=np.mean(mdl.var_,axis=0)
imp=1.0/(avg_var+1e-10); imp=imp/imp.sum()
exp_imp={n:v for n,v in zip(feat,imp)}
fi=find_key(r,'feature_importance')
got_imp={d['feature']:d['importance'] for d in fi}
for n in feat:
    chk(f"nb.importance[{n}]", got_imp.get(n), exp_imp[n], tol=1e-9)
    chk(f"nb.importance_pct[{n}]", next(d['importance_pct'] for d in fi if d['feature']==n), exp_imp[n]*100, tol=1e-6)
# descending-sort ordering
got_order=[d['feature'] for d in fi]
exp_order=[n for n,_ in sorted(exp_imp.items(),key=lambda kv:kv[1],reverse=True)]
chk("nb.importance_sorted_desc", got_order==exp_order, True)

# 3. Cross-validation — StratifiedKFold(5, shuffle, random_state=42) on full data
y_all_e=LabelEncoder().fit_transform(y)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(GaussianNB(var_smoothing=1e-9),X,y_all_e,cv=cv,scoring='accuracy')
chk("nb.cv_mean", find_key(r,'cv_mean'), float(np.mean(scores)), tol=1e-9)
chk("nb.cv_std",  find_key(r,'cv_std'),  float(np.std(scores)),  tol=1e-9)
cvs=find_key(r,'cv_scores')
for i,s in enumerate(scores):
    chk(f"nb.cv_score[{i}]", cvs[i], float(s), tol=1e-9)

# 4. Per-class precision/recall/f1/support
pred=mdl.predict(Xte)
pcm=find_key(r,'per_class_metrics')
got_pc={d['class']:d for d in pcm}
for i,c in enumerate(classes):
    yb_true=(yte_e==i); # per-class via sklearn on encoded labels
    chk(f"nb.per_class.precision[{c}]", got_pc[c]['precision'],
        precision_score(yte_e,pred,labels=[i],average='macro',zero_division=0), tol=1e-9)
    chk(f"nb.per_class.recall[{c}]", got_pc[c]['recall'],
        recall_score(yte_e,pred,labels=[i],average='macro',zero_division=0), tol=1e-9)
    chk(f"nb.per_class.f1[{c}]", got_pc[c]['f1_score'],
        f1_score(yte_e,pred,labels=[i],average='macro',zero_division=0), tol=1e-9)
    chk(f"nb.per_class.support[{c}]", got_pc[c]['support'], int((yte_e==i).sum()))

report("NAIVE BAYES (Python)")
