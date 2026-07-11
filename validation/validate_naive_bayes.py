import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,'test_size':0.2,
         'nb_type':'gaussian','var_smoothing':1e-9,'random_state':42}
r=run_script('naive_bayes_analysis.py',payload); r=r.get('results',r); m=r['metrics']
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=GaussianNB(var_smoothing=1e-9).fit(Xtr,ytr_e)
chk("nb.accuracy", m['accuracy'], accuracy_score(yte_e,mdl.predict(Xte)), tol=1e-9)
report("NAIVE BAYES (Python)")
