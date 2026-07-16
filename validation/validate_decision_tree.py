import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
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

# ── METHOD-SPECIFIC block checks (independent recomputation) ──

# tree_info: node count / actual depth / leaf count from the reproduced tree
ti = find_key(r, 'tree_info')
chk("dt.tree_info.n_nodes",          ti['n_nodes'],          int(m.tree_.node_count))
chk("dt.tree_info.max_depth_actual", ti['max_depth_actual'], int(m.get_depth()))
chk("dt.tree_info.n_leaves",         ti['n_leaves'],         int(m.get_n_leaves()))

# feature_importance: raw Gini importance + normalized + pct + descending rank
fi = find_key(r, 'feature_importance')
raw = m.feature_importances_
raw_by_name = {n: float(v) for n, v in zip(feat, raw)}
mx = raw.max() if raw.max() > 0 else 1.0
exp_sorted = sorted(feat, key=lambda n: raw_by_name[n], reverse=True)
for rank, row in enumerate(fi, 1):
    nm = row['feature']
    chk(f"dt.feature_importance[{nm}].importance",            row['importance'],            raw_by_name[nm])
    chk(f"dt.feature_importance[{nm}].normalized_importance", row['normalized_importance'], raw_by_name[nm] / mx)
    chk(f"dt.feature_importance[{nm}].importance_pct",        row['importance_pct'],        raw_by_name[nm] * 100.0)
    chk(f"dt.feature_importance[rank{rank}].feature",         row['feature'],               exp_sorted[rank - 1])
    chk(f"dt.feature_importance[{nm}].rank",                  row['rank'],                  rank)

# cv_results: 5-fold accuracy cross_val_score on the full dataset (backend uses X_arr, y_enc)
X_full = df[feat].values.astype(float)
y_full = LabelEncoder().fit_transform(df['species'])
m_cv = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1,
                              max_features=None, criterion='gini', splitter='best',
                              max_leaf_nodes=None, random_state=42)
cv_exp = cross_val_score(m_cv, X_full, y_full, cv=5, scoring='accuracy')
cvr = find_key(r, 'cv_results')
chk("dt.cv_results.cv_folds", cvr['cv_folds'], 5)
chk("dt.cv_results.cv_mean",  cvr['cv_mean'],  float(np.mean(cv_exp)))
chk("dt.cv_results.cv_std",   cvr['cv_std'],   float(np.std(cv_exp)))
for i, s in enumerate(cvr['cv_scores']):
    chk(f"dt.cv_results.cv_scores[{i}]", s, float(cv_exp[i]))

# tree_rules: leaf count reported by the rule extractor must match the tree
tr = find_key(r, 'tree_rules')
chk("dt.tree_rules.n_leaves", tr['n_leaves'], int(m.get_n_leaves()))

report("DECISION TREE (Python)")
