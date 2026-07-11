import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       max_depth=5,min_samples_split=2,min_samples_leaf=1,max_features=None,criterion='gini',splitter='best',max_leaf_nodes=None,random_state=42)
r=run_script('decision_tree_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=DecisionTreeClassifier(max_depth=5,min_samples_split=2,min_samples_leaf=1,max_features=None,criterion='gini',splitter='best',max_leaf_nodes=None,random_state=42).fit(Xtr,le.fit_transform(ytr))
chk("dt.accuracy", acc, accuracy_score(le.transform(yte),m.predict(Xte)), tol=1e-9)
report("DECISION TREE (Python)")
