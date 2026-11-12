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

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
PB=dict(data=bdf.to_dict('records'),target_col='diagnosis',feature_cols=bfeat,task_type='classification',test_size=0.2,
        hidden_layer_sizes=[50],activation='relu',solver='adam',alpha=0.0001,learning_rate_init=0.001,max_iter=300,early_stopping=False,random_state=42)
rb=run_script('mlp_analysis.py',PB)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis',scale=True)
mb=MLPClassifier(hidden_layer_sizes=(50,),activation='relu',solver='adam',alpha=0.0001,learning_rate_init=0.001,max_iter=300,early_stopping=False,random_state=42).fit(Xtr_b,ytr_b)
threshold_curve_checks("mlp.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
report("MLP (Python)")
