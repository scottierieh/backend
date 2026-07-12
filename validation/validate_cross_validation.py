import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',
       cv_method='kfold',n_folds=5,shuffle=True,model_type='random_forest',scoring='accuracy',random_state=42)
r=run_script('cross_validation_analysis.py',P); cvr=r["cv_results"]
Xs=StandardScaler().fit_transform(df[feat].values); y=LabelEncoder().fit_transform(df['species'])
cv=KFold(n_splits=5,shuffle=True,random_state=42)
sc=cross_val_score(RandomForestClassifier(n_estimators=100,random_state=42),Xs,y,cv=cv,scoring='accuracy')
chk("cv.mean_score", cvr["mean"], sc.mean(), tol=1e-9)
chk("cv.std_score",  cvr["std"],  sc.std(),  tol=1e-9)
chk("cv.min_score",  cvr["min"],  sc.min(),  tol=1e-9)
chk("cv.max_score",  cvr["max"],  sc.max(),  tol=1e-9)
chk("cv.median_score", cvr["median"], float(np.median(sc)), tol=1e-9)
chk("cv.n_folds",    cvr["n_folds"], 5)
for i,(got,exp) in enumerate(zip(cvr["scores"], sc)):
    chk(f"cv.fold{i}_score", got, float(exp), tol=1e-9)
report("CROSS VALIDATION (Python)")
