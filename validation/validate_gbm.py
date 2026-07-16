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

# --- Method-specific block: feature_importance (boosting) ---
# Backend emits results.feature_importance as an array of
# {feature, importance, importance_pct}, sorted descending by importance,
# where importance = model.feature_importances_ and
# importance_pct = importance / importance.sum() * 100.
fi = find_key(r, 'feature_importance')
if fi is not None:
    imp = np.asarray(m.feature_importances_, dtype=float)
    total = imp.sum() or 1.0
    ref = {name: (imp[i], imp[i] / total * 100) for i, name in enumerate(feat)}
    # ordering: emitted list must be sorted descending by importance
    got_vals = [row['importance'] for row in fi]
    chk("gbm.feature_importance.sorted_desc",
        all(got_vals[i] >= got_vals[i+1] for i in range(len(got_vals)-1)), True)
    chk("gbm.feature_importance.n", len(fi), len(feat))
    chk("gbm.feature_importance.sum", float(sum(got_vals)), float(imp.sum()), tol=1e-9)
    chk("gbm.feature_importance.pct_sum", float(sum(row['importance_pct'] for row in fi)), 100.0, tol=1e-6)
    for row in fi:
        exp_imp, exp_pct = ref[row['feature']]
        chk(f"gbm.feature_importance[{row['feature']}].importance", row['importance'], exp_imp, tol=1e-9)
        chk(f"gbm.feature_importance[{row['feature']}].importance_pct", row['importance_pct'], exp_pct, tol=1e-6)

report("GBM (Python)")
