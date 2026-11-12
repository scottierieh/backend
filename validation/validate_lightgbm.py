import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['y']=(iris.target==0).astype(int).map({1:'setosa',0:'other'})
P=dict(data=df.to_dict('records'),target_col='y',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=50,max_depth=-1,num_leaves=31,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,
       min_child_samples=20,reg_alpha=0.0,reg_lambda=0.0,random_state=42)
r=run_script('lightgbm_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['y']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=lgb.LGBMClassifier(n_estimators=50,max_depth=-1,num_leaves=31,learning_rate=0.1,subsample=1.0,
   colsample_bytree=1.0,min_child_samples=20,reg_alpha=0.0,reg_lambda=0.0,random_state=42,n_jobs=-1,verbosity=-1).fit(Xtr,le.fit_transform(ytr))
classifier_checks("lgb", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
PB=dict(data=bdf.to_dict('records'),target_col='diagnosis',feature_cols=bfeat,task_type='classification',test_size=0.2,
        n_estimators=100,max_depth=-1,num_leaves=31,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,
        min_child_samples=20,reg_alpha=0.0,reg_lambda=0.0,random_state=42,early_stopping_rounds=20)
rb=run_script('lightgbm_analysis.py',PB)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis')
ev={}
mb=lgb.LGBMClassifier(n_estimators=100,max_depth=-1,num_leaves=31,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,
   min_child_samples=20,reg_alpha=0.0,reg_lambda=0.0,random_state=42,n_jobs=-1,verbosity=-1,class_weight='balanced')
mb.fit(Xtr_b,ytr_b,eval_set=[(Xtr_b,ytr_b),(Xte_b,yte_b)],eval_names=['train','test'],eval_metric='binary_logloss',
       callbacks=[lgb.record_evaluation(ev),lgb.early_stopping(20,verbose=False)])
threshold_curve_checks("lgb.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
learning_curve_checks("lgb.bin", rb.get('learning_curve_data'), ev['train']['binary_logloss'], ev['test']['binary_logloss'],
                      (mb.best_iteration_ or None), 'binary_logloss', True)
report("LIGHTGBM (Python)")
