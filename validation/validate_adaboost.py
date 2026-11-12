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

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
from sklearn.utils.class_weight import compute_sample_weight
PB=dict(data=bdf.to_dict('records'),target_col='diagnosis',feature_cols=bfeat,task_type='classification',test_size=0.2,
        n_estimators=50,learning_rate=1.0,base_max_depth=1,random_state=42)
rb=run_script('adaboost_analysis.py',PB)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis')
mb=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,random_state=42),n_estimators=50,learning_rate=1.0,random_state=42)
mb.fit(Xtr_b,ytr_b,sample_weight=compute_sample_weight('balanced',ytr_b))
threshold_curve_checks("ada.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
learning_curve_checks("ada.bin", rb.get('learning_curve_data'), list(mb.staged_score(Xtr_b,ytr_b)), list(mb.staged_score(Xte_b,yte_b)),
                      None, 'accuracy', False)
report("ADABOOST (Python)")
