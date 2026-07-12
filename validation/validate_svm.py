# Validates svm_analysis.py (classification accuracy) against a direct sklearn SVC.
import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, classifier_checks

iris=load_iris(as_frame=True); df=iris.frame.copy()
feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,
         'task_type':'classification','test_size':0.2,'kernel':'rbf','C':1.0,'gamma':'scale','random_state':42}
res=run_script('svm_analysis.py', payload)
r=res['results'] if 'results' in res else res
m=r['metrics'] if 'metrics' in r else r

# reproduce: scale full X -> stratified split -> label encode -> SVC
X=df[feat].values; y=df['species']
Xs=StandardScaler().fit_transform(X)
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=SVC(kernel='rbf',C=1.0,gamma='scale',degree=3,coef0=0.0,random_state=42,probability=True).fit(Xtr,ytr_e)
classifier_checks("svm", r, mdl, Xtr, ytr_e, Xte, yte_e)
report("SVM (Python)")
