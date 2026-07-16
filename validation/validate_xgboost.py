import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['y']=(iris.target==0).astype(int).map({1:'setosa',0:'other'})  # binary
P=dict(data=df.to_dict('records'),target_col='y',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=50,max_depth=3,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,min_child_weight=1,
       gamma=0.0,reg_alpha=0.0,reg_lambda=1.0,random_state=42)
r=run_script('xgboost_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['y']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=xgb.XGBClassifier(n_estimators=50,max_depth=3,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,
   min_child_weight=1,gamma=0.0,reg_alpha=0.0,reg_lambda=1.0,random_state=42,n_jobs=-1).fit(Xtr,le.fit_transform(ytr))
classifier_checks("xgb", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# ---- Method-specific (boosting) block checks -----------------------------
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

ytr_e = le.transform(ytr); yte_e = le.transform(yte)

# 1) feature_importance ranking block: recompute from model.feature_importances_
fi = find_key(r, 'feature_importance')
imp = np.asarray(m.feature_importances_, dtype=float)
total = imp.sum() if imp.sum() > 0 else 1.0
maxi = imp.max() if imp.max() > 0 else 1.0
exp_fi = [{'feature': f, 'importance': float(v), 'importance_pct': float(v/total*100),
           'normalized_importance': float(v/maxi)} for f, v in zip(feat, imp)]
exp_fi.sort(key=lambda d: d['importance'], reverse=True)
for i, e in enumerate(exp_fi):
    e['rank'] = i + 1
chk("xgb.feature_importance.len", len(fi), len(exp_fi))
by_feat = {d['feature']: d for d in fi}
for e in exp_fi:
    g = by_feat[e['feature']]
    chk(f"xgb.fi[{e['feature']}].importance", g['importance'], e['importance'], tol=1e-6)
    chk(f"xgb.fi[{e['feature']}].importance_pct", g['importance_pct'], e['importance_pct'], tol=1e-4)
    chk(f"xgb.fi[{e['feature']}].normalized_importance", g['normalized_importance'], e['normalized_importance'], tol=1e-6)
    chk(f"xgb.fi[{e['feature']}].rank", g['rank'], e['rank'])

# 2) AUC (binary) — positive-class probability
auc_got = find_key(r, 'auc')
proba = m.predict_proba(Xte)[:, 1]
chk("xgb.metrics.auc", auc_got, float(roc_auc_score(yte_e, proba)), tol=1e-6)

# 3) per_class_metrics — precision/recall/f1/support per class label
pcm = find_key(r, 'per_class_metrics')
pred_te = m.predict(Xte)
by_cls = {d['class']: d for d in pcm}
for idx, cname in enumerate(le.classes_):
    d = by_cls[str(cname)]
    yb = (yte_e == idx).astype(int); pb = (pred_te == idx).astype(int)
    chk(f"xgb.pcm[{cname}].precision", d['precision'], float(precision_score(yb, pb, zero_division=0)), tol=1e-9)
    chk(f"xgb.pcm[{cname}].recall", d['recall'], float(recall_score(yb, pb, zero_division=0)), tol=1e-9)
    chk(f"xgb.pcm[{cname}].f1_score", d['f1_score'], float(f1_score(yb, pb, zero_division=0)), tol=1e-9)
    chk(f"xgb.pcm[{cname}].support", d['support'], int((yte_e == idx).sum()))

# 4) cv_results — StratifiedKFold accuracy over the FULL dataset (backend contract)
cvr = find_key(r, 'cv_results')
y_all = le.fit_transform(df['y'])
cv_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=1.0,
    colsample_bytree=1.0, objective='binary:logistic', random_state=42, n_jobs=-1)
cvfolds = max(2, min(5, int(np.min(np.bincount(y_all)))))
splitter = StratifiedKFold(n_splits=cvfolds, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_model, X, y_all, cv=splitter, scoring='accuracy')
chk("xgb.cv_results.cv_folds", cvr['cv_folds'], cvfolds)
chk("xgb.cv_results.cv_mean", cvr['cv_mean'], float(np.mean(cv_scores)), tol=1e-6)
chk("xgb.cv_results.cv_std", cvr['cv_std'], float(np.std(cv_scores)), tol=1e-6)
for i, s in enumerate(cv_scores):
    chk(f"xgb.cv_results.cv_scores[{i}]", cvr['cv_scores'][i], float(s), tol=1e-6)

# 5) tree_rules — n_leaves counted from booster tree-0 dump
tr = find_key(r, 'tree_rules')
dump0 = m.get_booster().get_dump(with_stats=False)[0]
exp_leaves = sum(1 for ln in dump0.strip().split('\n') if 'leaf' in ln)
chk("xgb.tree_rules.n_leaves", tr['n_leaves'], exp_leaves)

report("XGBOOST (Python)")
