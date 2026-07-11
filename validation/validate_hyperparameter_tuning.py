import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),features=feat,target='species',model='random_forest',search_method='grid',cv_folds=5,scoring='accuracy')
r=run_script('hyperparameter_tuning_analysis.py',P); best=find_key(r,'best_cv_score')
Xs=StandardScaler().fit_transform(df[feat].values); y=df['species']
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.3,random_state=42,stratify=y)
grid={'n_estimators':[100,200,300],'max_depth':[5,10,None],'min_samples_split':[2,5,10]}
gs=GridSearchCV(RandomForestClassifier(random_state=42),grid,cv=5,scoring='accuracy',n_jobs=-1,refit=True).fit(Xtr,ytr)
chk("hp.best_cv_score", best, gs.best_score_, tol=1e-9)
report("HYPERPARAMETER TUNING (Python)")
