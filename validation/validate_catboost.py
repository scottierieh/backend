import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['y']=(iris.target==0).astype(int).map({1:'setosa',0:'other'})
P=dict(data=df.to_dict('records'),target_col='y',feature_cols=feat,task_type='classification',test_size=0.2,
       iterations=100,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_state=42)
r=run_script('catboost_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['y']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=CatBoostClassifier(iterations=100,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_seed=42,verbose=False,allow_writing_files=False).fit(Xtr,le.fit_transform(ytr))
chk("cat.accuracy", acc, accuracy_score(le.transform(yte),m.predict(Xte).ravel()), tol=1e-9)
report("CATBOOST (Python)")
