import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
report("ENSEMBLE (VOTING) (Python)")
