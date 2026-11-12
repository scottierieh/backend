import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names)
df['y']=(iris.target==0).astype(int).map({1:'setosa',0:'other'})  # binary
P=dict(data=df.to_dict('records'),target_col='y',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=50,max_depth=3,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,min_child_weight=1,
       gamma=0.0,reg_alpha=0.0,reg_lambda=1.0,random_state=42)
r=run_script('xgboost_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['y']; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=xgb.XGBClassifier(n_estimators=50,max_depth=3,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,
   min_child_weight=1,gamma=0.0,reg_alpha=0.0,reg_lambda=1.0,random_state=42,n_jobs=-1).fit(Xtr,le.fit_transform(ytr))
classifier_checks("xgb", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
PB=dict(data=bdf.to_dict('records'),target_col='diagnosis',feature_cols=bfeat,task_type='classification',test_size=0.2,
        n_estimators=100,max_depth=3,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,min_child_weight=1,
        gamma=0.0,reg_alpha=0.0,reg_lambda=1.0,random_state=42)
rb=run_script('xgboost_analysis.py',PB)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis')
_c=np.bincount(ytr_b)
mb=xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=1.0,colsample_bytree=1.0,min_child_weight=1,
   gamma=0.0,reg_alpha=0.0,reg_lambda=1.0,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=10,
   random_state=42,n_jobs=-1,scale_pos_weight=float(_c.max())/float(_c.min()))
mb.fit(Xtr_b,ytr_b,eval_set=[(Xtr_b,ytr_b),(Xte_b,yte_b)],verbose=False)
ev=mb.evals_result()
threshold_curve_checks("xgb.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
learning_curve_checks("xgb.bin", rb.get('learning_curve_data'), ev['validation_0']['logloss'], ev['validation_1']['logloss'],
                      mb.best_iteration+1, 'logloss', True)
chk("xgb.bin.learning_curve_data.early_stopping_rounds", rb['learning_curve_data'].get('early_stopping_rounds'), 10)
report("XGBOOST (Python)")
