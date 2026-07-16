# Validates svm_analysis.py (classification accuracy) against a direct sklearn SVC.
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from _pyharness import run_script, chk, report, classifier_checks, find_key

iris=load_iris(as_frame=True); df=iris.frame.copy()
feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,
         'task_type':'classification','test_size':0.2,'kernel':'rbf','C':1.0,'gamma':'scale','random_state':42}
res=run_script('svm_analysis.py', payload)
r=res['results'] if 'results' in res else res
m=r['metrics'] if 'metrics' in r else r

# reproduce: scale full X -> stratified split -> label encode -> SVC
X=df[feat].values; y=df['species']
Xs=StandardScaler().fit_transform(X)
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=SVC(kernel='rbf',C=1.0,gamma='scale',degree=3,coef0=0.0,random_state=42,probability=True).fit(Xtr,ytr_e)
classifier_checks("svm", r, mdl, Xtr, ytr_e, Xte, yte_e)

# ── METHOD-SPECIFIC BLOCKS (SVM STEP6) ──────────────────────────────────
# 1. Support vectors — total (metrics.n_support_vectors = sum(model.n_support_))
chk("svm.n_support_vectors", find_key(r, 'n_support_vectors'), int(sum(mdl.n_support_)))

# 2. Support vectors per class — model.n_support_ ordered by sorted class label
spc = find_key(r, 'support_per_class')
if spc is not None:
    # backend emits [{'class': str(cls), 'n_support_vectors': int}] in le.classes_ order
    for i, (cls, n_sv) in enumerate(zip(le.classes_, mdl.n_support_)):
        chk(f"svm.support_per_class[{i}].class", spc[i]['class'], str(cls))
        chk(f"svm.support_per_class[{i}].n_sv", spc[i]['n_support_vectors'], int(n_sv))

# 3. Cross-validation — StratifiedKFold(5, shuffle, rs=42), accuracy on full scaled X
cv = find_key(r, 'cv_results')
if cv is not None:
    y_full_e = LabelEncoder().fit_transform(y)
    cv_mdl = SVC(kernel='rbf', C=1.0, gamma='scale', degree=3, random_state=42)
    cvk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_mdl, Xs, y_full_e, cv=cvk, scoring='accuracy')
    chk("svm.cv_mean", cv['cv_mean'], float(np.mean(cv_scores)))
    chk("svm.cv_std", cv['cv_std'], float(np.std(cv_scores)))
    for i, s in enumerate(cv_scores):
        chk(f"svm.cv_scores[{i}]", cv['cv_scores'][i], float(s))

# 4. Permutation feature importance — n_repeats=10, rs=42 (deterministic)
fi = find_key(r, 'feature_importance')
if fi is not None:
    perm = permutation_importance(mdl, Xte, yte_e, n_repeats=10, random_state=42, n_jobs=-1)
    exp_imp = dict(zip(feat, perm.importances_mean))
    exp_std = dict(zip(feat, perm.importances_std))
    for item in fi:
        name = item['feature']
        chk(f"svm.fi[{name}].importance", item['importance'], float(exp_imp[name]))
        chk(f"svm.fi[{name}].std", item['std'], float(exp_std[name]))
    # importance ranking is sorted descending
    imps = [item['importance'] for item in fi]
    chk("svm.fi_sorted_desc", imps == sorted(imps, reverse=True), True)

report("SVM (Python)")
