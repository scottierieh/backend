import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,'test_size':0.2,
         'nb_type':'gaussian','var_smoothing':1e-9,'random_state':42}
r=run_script('naive_bayes_analysis.py',payload); r=r.get('results',r); m=r['metrics']
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=GaussianNB(var_smoothing=1e-9).fit(Xtr,ytr_e)
classifier_checks("nb", r, mdl, Xtr, ytr_e, Xte, yte_e)

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
from sklearn.utils.class_weight import compute_sample_weight
PB={'data':bdf.to_dict('records'),'target_col':'diagnosis','feature_cols':bfeat,'test_size':0.2,
    'nb_type':'gaussian','var_smoothing':1e-9,'random_state':42}
rb=run_script('naive_bayes_analysis.py',PB); rb=rb.get('results',rb)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis')
mb=GaussianNB(var_smoothing=1e-9).fit(Xtr_b,ytr_b,sample_weight=compute_sample_weight('balanced',ytr_b))
threshold_curve_checks("nb.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
report("NAIVE BAYES (Python)")
