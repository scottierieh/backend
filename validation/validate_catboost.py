import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['y']=(iris.target==0).astype(int).map({1:'setosa',0:'other'})
P=dict(data=df.to_dict('records'),target_col='y',feature_cols=feat,task_type='classification',test_size=0.2,
       iterations=100,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_state=42)
r=run_script('catboost_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['y']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=CatBoostClassifier(iterations=100,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_seed=42,verbose=False,allow_writing_files=False).fit(Xtr,le.fit_transform(ytr))
classifier_checks("cat", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
from catboost import Pool
PB=dict(data=bdf.to_dict('records'),target_col='diagnosis',feature_cols=bfeat,task_type='classification',test_size=0.2,
        iterations=200,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_state=42,early_stopping_rounds=20)
rb=run_script('catboost_analysis.py',PB)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis')
mb=CatBoostClassifier(iterations=200,depth=6,learning_rate=0.1,l2_leaf_reg=3.0,random_seed=42,verbose=False,
   allow_writing_files=False,auto_class_weights='Balanced')
mb.fit(Pool(Xtr_b,ytr_b),eval_set=Pool(Xte_b,yte_b),use_best_model=True,early_stopping_rounds=20)
ev=mb.get_evals_result(); _mk=list(ev['learn'].keys())[0]; _vk=list(ev['validation'].keys())[0]
threshold_curve_checks("cat.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Pool(Xte_b))[:,1], lab_b, rb['metrics'].get('auc'))
learning_curve_checks("cat.bin", rb.get('learning_curve_data'), ev['learn'][_mk], ev['validation'][_vk],
                      mb.get_best_iteration()+1, _mk, True)
report("CATBOOST (Python)")
