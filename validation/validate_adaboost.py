import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=50,learning_rate=1.0,base_max_depth=1,random_state=42)
r=run_script('adaboost_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); base=DecisionTreeClassifier(max_depth=1,random_state=42)
m=AdaBoostClassifier(estimator=base,n_estimators=50,learning_rate=1.0,random_state=42).fit(Xtr,le.fit_transform(ytr))
classifier_checks("ada", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))
report("ADABOOST (Python)")
