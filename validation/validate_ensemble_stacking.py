import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       ensemble_method='voting',voting_type='soft',
       base_estimators=['logistic_regression','decision_tree','random_forest'],
       scale_features=True,random_state=42)
r=run_script('ensemble_stacking_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xs=StandardScaler().fit_transform(X); Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
ests=[('logistic_regression',LogisticRegression(max_iter=1000,random_state=42)),
      ('decision_tree',DecisionTreeClassifier(max_depth=5,random_state=42)),
      ('random_forest',RandomForestClassifier(n_estimators=100,random_state=42))]
m=VotingClassifier(estimators=ests,voting='soft').fit(Xtr,ytr)
classifier_checks("stack.voting", r, m, Xtr, ytr, Xte, yte)

# ----- METHOD-SPECIFIC BLOCKS (what the STEP6 ensemble page shows) -----
# The backend fits each base learner standalone on the training split and reports
# its test-set score, plus the ensemble's own score, in individual_scores.
ind = find_key(r, 'individual_scores')
for name, est in ests:
    est.fit(Xtr, ytr)
    chk(f"stack.individual_scores.{name}", ind[name], accuracy_score(yte, est.predict(Xte)))
chk("stack.individual_scores.ensemble", ind['Voting Ensemble (soft)'], accuracy_score(yte, m.predict(Xte)))
# base_estimators list (names, in order)
chk("stack.base_estimators", ",".join(find_key(r, 'base_estimators')),
    ",".join(n for n, _ in ests))

# cross_validation block: StratifiedKFold(shuffle, rs=42) accuracy on the FULL data,
# with labels LabelEncoded exactly as the backend does it.
le = LabelEncoder(); y_enc = le.fit_transform(y)
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mv = VotingClassifier(estimators=[('logistic_regression', LogisticRegression(max_iter=1000, random_state=42)),
                                  ('decision_tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
                                  ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42))],
                      voting='soft')
cv_scores_exp = cross_val_score(mv, Xs, y_enc, cv=cv_splitter, scoring='accuracy')
cvr = find_key(r, 'cv_results')
chk("stack.cv_folds", cvr['cv_folds'], 5)
for i, s in enumerate(cv_scores_exp):
    chk(f"stack.cv_scores[{i}]", cvr['cv_scores'][i], float(s))
chk("stack.cv_mean", cvr['cv_mean'], float(np.mean(cv_scores_exp)))
chk("stack.cv_std", cvr['cv_std'], float(np.std(cv_scores_exp)))

# per_class_metrics: precision/recall/f1/support recomputed per class (label order = sorted classes = le.classes_)
pred = m.predict(Xte)
classes = list(le.classes_)
prec = precision_score(yte, pred, average=None, labels=classes, zero_division=0)
rec = recall_score(yte, pred, average=None, labels=classes, zero_division=0)
f1 = f1_score(yte, pred, average=None, labels=classes, zero_division=0)
pcm = {d['class']: d for d in find_key(r, 'per_class_metrics')}
yte_arr = np.asarray(yte)
for i, cls in enumerate(classes):
    d = pcm[str(cls)]
    chk(f"stack.per_class[{cls}].precision", d['precision'], float(prec[i]))
    chk(f"stack.per_class[{cls}].recall", d['recall'], float(rec[i]))
    chk(f"stack.per_class[{cls}].f1_score", d['f1_score'], float(f1[i]))
    chk(f"stack.per_class[{cls}].support", d['support'], int((yte_arr == cls).sum()))

# permutation feature importance ranking (n_repeats=10, rs=42) — the backend reports
# both perm_importance (mean/std) and a normalized feature_importance (positive share -> pct).
perm = permutation_importance(m, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)
perm_exp = dict(zip(feat, perm.importances_mean))
std_exp = dict(zip(feat, perm.importances_std))
pi = {d['feature']: d for d in find_key(r, 'perm_importance')}
for f in feat:
    chk(f"stack.perm_importance[{f}].mean", pi[f]['importance_mean'], float(perm_exp[f]))
    chk(f"stack.perm_importance[{f}].std", pi[f]['importance_std'], float(std_exp[f]))
# feature_importance pct = clip(mean,0) / sum(clip(mean,0)) * 100
pos = {f: max(perm_exp[f], 0.0) for f in feat}
tot = sum(pos.values()) or 1.0
fi = {d['feature']: d for d in find_key(r, 'feature_importance')}
for f in feat:
    chk(f"stack.feature_importance[{f}].pct", fi[f]['importance_pct'], pos[f] / tot * 100)

report("ENSEMBLE (VOTING) (Python)")
