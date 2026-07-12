import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       hidden_layer_sizes=[50],activation='relu',solver='adam',alpha=0.0001,learning_rate_init=0.001,max_iter=300,early_stopping=False,random_state=42)
r=run_script('mlp_analysis.py',P); acc=find_key(r,'accuracy')
Xs=StandardScaler().fit_transform(df[feat].values); y=df['species']
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=MLPClassifier(hidden_layer_sizes=(50,),activation='relu',solver='adam',alpha=0.0001,learning_rate_init=0.001,max_iter=300,early_stopping=False,random_state=42).fit(Xtr,le.fit_transform(ytr))
classifier_checks("mlp", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))
report("MLP (Python)")
