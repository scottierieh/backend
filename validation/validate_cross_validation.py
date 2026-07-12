# Validates cross_validation_analysis.py against an independent sklearn
# reproduction (KFold + cross_val_score/cross_val_predict) of the handler's
# exact preprocessing / splitter / model / seed. Every returned value is
# checked: per-fold scores + sizes, mean/std/min/max/median, additional
# cross-validated metrics, and the sample/feature counts.
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from _pyharness import run_script, chk, report

iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',
       cv_method='kfold',n_folds=5,shuffle=True,model_type='random_forest',scoring='accuracy',random_state=42)
r=run_script('cross_validation_analysis.py',P); cvr=r["cv_results"]

# --- independent reproduction ---
Xs=StandardScaler().fit_transform(df[feat].values); y=LabelEncoder().fit_transform(df['species'])
cv=KFold(n_splits=5,shuffle=True,random_state=42)
mdl=RandomForestClassifier(n_estimators=100,random_state=42)
sc=cross_val_score(mdl,Xs,y,cv=cv,scoring='accuracy')
y_pred=cross_val_predict(mdl,Xs,y,cv=cv)

# --- top-level counts / task ---
chk("cv.task_type", r["task_type"], "classification")
chk("cv.n_samples", r["n_samples"], len(y))
chk("cv.n_features", r["n_features"], len(feat))

# --- aggregate score statistics ---
chk("cv.mean_score", cvr["mean"], sc.mean(), tol=1e-9)
chk("cv.std_score",  cvr["std"],  sc.std(),  tol=1e-9)
chk("cv.min_score",  cvr["min"],  sc.min(),  tol=1e-9)
chk("cv.max_score",  cvr["max"],  sc.max(),  tol=1e-9)
chk("cv.median_score", cvr["median"], float(np.median(sc)), tol=1e-9)
chk("cv.n_folds",    cvr["n_folds"], 5)

# --- every fold: score + train/test sizes ---
splits=list(cv.split(Xs,y))
for i,(got,exp) in enumerate(zip(cvr["scores"], sc)):
    chk(f"cv.fold{i}_score", got, float(exp), tol=1e-9)
for i,(tr_idx,te_idx) in enumerate(splits):
    fd=cvr["fold_details"][i]
    chk(f"cv.fold{i}.fold", fd["fold"], i+1)
    chk(f"cv.fold{i}.train_size", fd["train_size"], len(tr_idx))
    chk(f"cv.fold{i}.test_size", fd["test_size"], len(te_idx))

# --- additional cross-validated metrics ---
am=r["additional_metrics"]
chk("cv.add.accuracy",  am["accuracy"],  accuracy_score(y,y_pred), tol=1e-9)
chk("cv.add.precision", am["precision_macro"], precision_score(y,y_pred,average='macro',zero_division=0), tol=1e-9)
chk("cv.add.recall",    am["recall_macro"], recall_score(y,y_pred,average='macro',zero_division=0), tol=1e-9)
chk("cv.add.f1",        am["f1_macro"], f1_score(y,y_pred,average='macro',zero_division=0), tol=1e-9)

# --- plots present ---
for k in ['scores_plot','stability_plot']:
    chk(f"has.{k}", isinstance(r.get(k),str) and len(r[k])>0, True)
report("CROSS VALIDATION (Python)")
