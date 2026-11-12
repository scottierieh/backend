import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target='species',features=feat,problemType='classification',
       nEstimators=100,learningRate=0.1,maxDepth=3)
r=run_script('gbm_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
m=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42).fit(Xtr,ytr)
classifier_checks("gbm", r, m, Xtr, ytr, Xte, yte)

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.datasets import load_diabetes
PB=dict(data=bdf.to_dict('records'),target='diagnosis',features=bfeat,problemType='classification',nEstimators=100,learningRate=0.1,maxDepth=3)
rb=run_script('gbm_analysis.py',PB); rbr=rb['results']
X=bdf[bfeat].values; yb=bdf['diagnosis']
Xtr_b,Xte_b,ytr_b,yte_b=train_test_split(X,yb,test_size=0.2,random_state=42,stratify=yb)
mb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42).fit(Xtr_b,ytr_b,sample_weight=compute_sample_weight('balanced',ytr_b))
threshold_curve_checks("gbm.bin", rbr.get('threshold_curve'), (np.asarray(yte_b)==mb.classes_[1]).astype(int), mb.predict_proba(Xte_b)[:,1],
                       [str(c) for c in mb.classes_], rbr['metrics'].get('auc'))
learning_curve_checks("gbm.bin", rbr.get('learning_curve_data'),
                      [log_loss(ytr_b,p,labels=mb.classes_) for p in mb.staged_predict_proba(Xtr_b)],
                      [log_loss(yte_b,p,labels=mb.classes_) for p in mb.staged_predict_proba(Xte_b)], None, 'log_loss', True)
chk("gbm.bin.learning_curve_data.last_valid==log_loss(final model)", rbr['learning_curve_data']['valid'][-1], log_loss(yte_b,mb.predict_proba(Xte_b)), tol=1e-9)
chk("gbm.iris.threshold_curve_absent_multiclass", 'threshold_curve' in r['results'], False)
_db=load_diabetes(as_frame=True); ddf=_db.frame.copy(); dfeat=list(_db.feature_names)
PR=dict(data=ddf.to_dict('records'),target='target',features=dfeat,problemType='regression',nEstimators=100,learningRate=0.1,maxDepth=3)
rr=run_script('gbm_analysis.py',PR)['results']
Xtr_r,Xte_r,ytr_r,yte_r=train_test_split(ddf[dfeat].values,ddf['target'].values,test_size=0.2,random_state=42)
mr=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42,validation_fraction=0.1,n_iter_no_change=5,tol=0.01).fit(Xtr_r,ytr_r)
learning_curve_checks("gbm.reg", rr.get('learning_curve_data'),
                      [mean_squared_error(ytr_r,p) for p in mr.staged_predict(Xtr_r)],
                      [mean_squared_error(yte_r,p) for p in mr.staged_predict(Xte_r)], None, 'mse', True)
chk("gbm.reg.learning_curve_data.n_iterations==n_estimators_", rr['learning_curve_data']['n_iterations'], int(mr.n_estimators_))
chk("gbm.reg.learning_curve_data.last_valid==mse(final model)", rr['learning_curve_data']['valid'][-1], mean_squared_error(yte_r,mr.predict(Xte_r)), tol=1e-9)
report("GBM (Python)")
