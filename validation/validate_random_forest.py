import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=False,random_state=42)
r=run_script('random_forest_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# The handler applies balanced class weighting by default (balanced_weighting ->
# class_weight='balanced', since 3cc890f), so the reference forest does too. On
# iris' stratified split the training classes are 40/40/40 and the balanced
# weights are all 1, but that is not a no-op everywhere: scikit-learn >= 1.8
# draws each tree's bootstrap sample from the sample weights, so an unweighted
# reference grows a different forest there (accuracy 0.9 vs the handler's 0.967
# under scikit-learn 1.9). Mirroring the handler's setting keeps the comparison
# like-for-like on every scikit-learn version.
le=LabelEncoder(); m=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=False,random_state=42,n_jobs=-1,class_weight='balanced').fit(Xtr,le.fit_transform(ytr))
cw=find_key(r,'class_weighting') or {}
chk("rf.class_weighting.applied", cw.get('applied'), True)
chk("rf.class_weighting.method", cw.get('method'), "class_weight='balanced'")
classifier_checks("rf", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
PB=dict(data=bdf.to_dict('records'),target_col='diagnosis',feature_cols=bfeat,task_type='classification',test_size=0.2,
        n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=False,random_state=42)
rb=run_script('random_forest_analysis.py',PB)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis')
mb=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',bootstrap=True,random_state=42,n_jobs=-1,class_weight='balanced').fit(Xtr_b,ytr_b)
threshold_curve_checks("rf.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
chk("rf.iris.threshold_curve_absent_multiclass", 'threshold_curve' in r, False)
report("RANDOM FOREST (Python)")
