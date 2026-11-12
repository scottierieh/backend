import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from _pyharness import run_script, chk, report, find_key, classifier_checks
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
P=dict(data=df.to_dict('records'),target_col='species',feature_cols=feat,task_type='classification',test_size=0.2,
       max_depth=5,min_samples_split=2,min_samples_leaf=1,max_features=None,criterion='gini',splitter='best',max_leaf_nodes=None,random_state=42)
r=run_script('decision_tree_analysis.py',P); acc=find_key(r,'accuracy')
X=df[feat].values; y=df['species']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
le=LabelEncoder(); m=DecisionTreeClassifier(max_depth=5,min_samples_split=2,min_samples_leaf=1,max_features=None,criterion='gini',splitter='best',max_leaf_nodes=None,random_state=42).fit(Xtr,le.fit_transform(ytr))
classifier_checks("dt", r, m, Xtr, le.transform(ytr), Xte, le.transform(yte))

# --- Exploration-tab payloads on a binary target (breast cancer, 10 features) ---
from sklearn.datasets import load_breast_cancer
from _pyharness import binary_split, threshold_curve_checks, learning_curve_checks, tree_structure_checks
_bc=load_breast_cancer(as_frame=True); bdf=_bc.frame.copy(); bfeat=list(_bc.feature_names)[:10]
bdf['diagnosis']=np.where(_bc.target==1,'benign','malignant')
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
# iris (multiclass) run above: the tree shown is the tree that predicts
tree_structure_checks("dt.iris", r.get('tree_structure'), m, Xte, feat, 'classification', le.classes_)
chk("dt.iris.threshold_curve_absent_multiclass", 'threshold_curve' in r, False)
# binary: threshold curve + tree
PB=dict(data=bdf.to_dict('records'),target_col='diagnosis',feature_cols=bfeat,task_type='classification',test_size=0.2,
        max_depth=5,min_samples_split=2,min_samples_leaf=1,max_features=None,criterion='gini',splitter='best',max_leaf_nodes=None,random_state=42)
rb=run_script('decision_tree_analysis.py',PB)
Xtr_b,Xte_b,ytr_b,yte_b,lab_b=binary_split(bdf,bfeat,'diagnosis')
mb=DecisionTreeClassifier(max_depth=5,min_samples_split=2,min_samples_leaf=1,criterion='gini',splitter='best',random_state=42,class_weight='balanced').fit(Xtr_b,ytr_b)
threshold_curve_checks("dt.bin", rb.get('threshold_curve'), yte_b, mb.predict_proba(Xte_b)[:,1], lab_b, rb['metrics'].get('auc'))
tree_structure_checks("dt.bin", rb.get('tree_structure'), mb, np.vstack([Xtr_b,Xte_b]), bfeat, 'classification', lab_b)
# regression (diabetes): leaf means in target units; depth 6 fits under 255 nodes
_db=load_diabetes(as_frame=True); ddf=_db.frame.copy(); dfeat=list(_db.feature_names)
PR=dict(data=ddf.to_dict('records'),target_col='target',feature_cols=dfeat,task_type='regression',test_size=0.2,
        max_depth=6,min_samples_split=2,min_samples_leaf=1,max_features=None,criterion='squared_error',splitter='best',max_leaf_nodes=None,random_state=42)
rr=run_script('decision_tree_analysis.py',PR)
Xtr_r,Xte_r,ytr_r,yte_r=train_test_split(ddf[dfeat].values,ddf['target'].values,test_size=0.2,random_state=42)
mr=DecisionTreeRegressor(max_depth=6,random_state=42).fit(Xtr_r,ytr_r)
tree_structure_checks("dt.reg", rr.get('tree_structure'), mr, Xte_r, dfeat, 'regression')
chk("dt.reg.threshold_curve_absent", 'threshold_curve' in rr, False)
# unlimited depth: 600+ nodes -> truncated to <= 255 shown nodes
PR2=dict(PR); PR2['max_depth']=None
rr2=run_script('decision_tree_analysis.py',PR2); ts2=rr2.get('tree_structure') or {}
mr2=DecisionTreeRegressor(random_state=42).fit(Xtr_r,ytr_r)
chk("dt.reg_full.truncated", ts2.get('truncated'), mr2.tree_.node_count>255)
chk("dt.reg_full.n_nodes_total", ts2.get('n_nodes_total'), int(mr2.tree_.node_count))
chk("dt.reg_full.shown_nodes<=255", len(ts2.get('left',[]))<=255, True)
_ids=ts2.get('node_id',[])
chk("dt.reg_full.shown_values==tree_.value", float(max(abs(ts2['value'][k][0]-mr2.tree_.value[i][0][0]) for k,i in enumerate(_ids))), 0.0, tol=1e-9)
report("DECISION TREE (Python)")
