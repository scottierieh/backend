import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target='species',features=feat,problemType='classification',
       nEstimators=100,learningRate=0.1,maxDepth=3)
r=run_script('gbm_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
m=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42).fit(Xtr,ytr)
classifier_checks("gbm", r, m, Xtr, ytr, Xte, yte)
report("GBM (Python)")
