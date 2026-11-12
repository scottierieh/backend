import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,'task_type':'classification',
         'test_size':0.2,'n_neighbors':5,'weights':'uniform','metric':'minkowski','p':2,'find_optimal_k':False,'random_state':42}
r=run_script('knn_analysis.py',payload); r=r.get('results',r); m=r['metrics']
Xs=StandardScaler().fit_transform(df[feat].values); y=df['species']
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski',p=2).fit(Xtr,ytr_e)
classifier_checks("knn", r, mdl, Xtr, ytr_e, Xte, yte_e)

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
PB={'data':bdf.to_dict('records'),'target_col':'diagnosis','feature_cols':bfeat,'task_type':'classification',
    'test_size':0.2,'n_neighbors':7,'weights':'distance','metric':'minkowski','p':2,'find_optimal_k':False,'random_state':42}
rb=run_script('knn_analysis.py',PB); rb=rb.get('results',rb)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis',scale=True)
mb=KNeighborsClassifier(n_neighbors=7,weights='distance',metric='minkowski',p=2).fit(Xtr_b,ytr_b)
threshold_curve_checks("knn.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
report("KNN (Python)")
