import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=False,random_state=42)
r=run_script('random_forest_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=False,random_state=42,n_jobs=-1).fit(Xtr,le.fit_transform(ytr))
classifier_checks("rf", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))
report("RANDOM FOREST (Python)")
