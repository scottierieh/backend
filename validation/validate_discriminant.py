import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from scipy import stats
from _pyharness import run_script, chk, report, classifier_checks, find_key
iris=load_iris(as_frame=True); df=iris.frame.copy(); feat=list(iris.feature_names); df['species']=iris.target_names[iris.target]
payload={'data':df.to_dict('records'),'target_col':'species','feature_cols':feat,'method':'lda',
         'test_size':0.2,'solver':'svd','random_state':42}
r=run_script('discriminant_analysis.py',payload); r=r.get('results',r); m=r['metrics']
X=df[feat].values; y=df['species']
Xtr_raw,Xte_raw,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr_raw); Xte=sc.transform(Xte_raw)
le=LabelEncoder(); ytr_e=le.fit_transform(ytr); yte_e=le.transform(yte)
mdl=LinearDiscriminantAnalysis(solver='svd').fit(Xtr,ytr_e)
classifier_checks("lda", r, mdl, Xtr, ytr_e, Xte, yte_e)

# ── Method-specific SPSS-level LDA blocks (STEP6 page) ─────────────────────
# Recompute each block independently, mirroring discriminant_analysis.py's
# documented formulas, and cross-check against the backend's emitted values.
n_samples, n_features = Xtr.shape
n_classes = len(np.unique(ytr_e))

# lda_info: priors, explained_variance_ratio, class_means (from the reproduced model)
li = find_key(r, 'lda_info') or {}
for i, pr in enumerate(mdl.priors_):
    chk(f"lda_info.priors[{i}]", li.get('priors', [None]*n_classes)[i], float(pr))
for i, ev in enumerate(mdl.explained_variance_ratio_):
    chk(f"lda_info.explained_variance_ratio[{i}]", li.get('explained_variance_ratio', [None]*len(mdl.explained_variance_ratio_))[i], float(ev))
cm_means = np.asarray(li.get('class_means'))
for i in range(n_classes):
    for j in range(n_features):
        chk(f"lda_info.class_means[{i},{j}]", cm_means[i, j], float(mdl.means_[i, j]))

# lda_statistics: eigenvalues / canonical correlations / Wilks' Lambda
# S_B (between-class) and S_W (within-class) scatter on the standardized train set.
overall = Xtr.mean(axis=0)
S_B = np.zeros((n_features, n_features)); S_W = np.zeros((n_features, n_features))
for i in range(n_classes):
    mask = ytr_e == i; n_k = mask.sum(); mean_k = Xtr[mask].mean(axis=0)
    d = (mean_k - overall).reshape(-1, 1); S_B += n_k * (d @ d.T)
    c = Xtr[mask] - mean_k; S_W += c.T @ c
ev_raw = np.real(np.linalg.eig(np.linalg.pinv(S_W) @ S_B)[0])
eig = ev_raw[np.argsort(ev_raw)[::-1]]; eig = eig[eig > 1e-10][:n_classes - 1]
canon = np.sqrt(eig / (1 + eig))
wilks = float(np.prod([1 / (1 + e) for e in eig]))
p, g = n_features, n_classes
chi2_exp = -(n_samples - 1 - (p + g) / 2) * np.log(wilks + 1e-10)
df_exp = p * (g - 1)

ls = find_key(r, 'lda_statistics') or {}
emit_eig = ls.get('eigenvalues', [])
for i in range(len(eig)):
    chk(f"lda_statistics.eigenvalue[LD{i+1}]", emit_eig[i]['eigenvalue'], float(eig[i]), tol=1e-6)
    chk(f"lda_statistics.canonical_correlation[LD{i+1}]", emit_eig[i]['canonical_correlation'], float(canon[i]), tol=1e-6)
tot_eig = eig.sum()
for i in range(len(eig)):
    chk(f"lda_statistics.variance_explained_pct[LD{i+1}]", emit_eig[i]['variance_explained_pct'], float(eig[i] / tot_eig * 100), tol=1e-6)
wl = ls.get('wilks_lambda') or {}
chk("lda_statistics.wilks_lambda.lambda", wl.get('lambda'), wilks, tol=1e-6)
chk("lda_statistics.wilks_lambda.chi2", wl.get('chi2'), float(chi2_exp), tol=1e-4)
chk("lda_statistics.wilks_lambda.df", wl.get('df'), int(df_exp))

# structure_matrix: correlation of each feature with each LD score (X_train @ scalings)
sm = {row['feature']: row for row in ls.get('structure_matrix', [])}
for fi, fn in enumerate(feat):
    for comp in range(mdl.scalings_.shape[1]):
        scores = Xtr @ mdl.scalings_[:, comp]
        corr = np.corrcoef(Xtr[:, fi], scores)[0, 1]
        chk(f"lda_statistics.structure_matrix[{fn}].LD{comp+1}", sm[fn][f'LD{comp+1}'], float(corr), tol=1e-6)

# group_centroids: mean LD score per class (X_train[class] @ scalings)
gc = {row['class']: row for row in ls.get('group_centroids', [])}
for i, lab in enumerate(le.classes_):
    mask = ytr_e == i; class_scores = Xtr[mask] @ mdl.scalings_
    for comp in range(class_scores.shape[1]):
        chk(f"lda_statistics.group_centroids[{lab}].LD{comp+1}", gc[str(lab)][f'LD{comp+1}'], float(class_scores[:, comp].mean()), tol=1e-6)

# anova_f_statistics: one-way ANOVA F per feature across classes
af = {row['feature']: row for row in ls.get('anova_f_statistics', [])}
for fi, fn in enumerate(feat):
    groups = [Xtr[ytr_e == i, fi] for i in range(n_classes)]
    f_stat, p_val = stats.f_oneway(*groups)
    chk(f"lda_statistics.anova_f[{fn}].f_statistic", af[fn]['f_statistic'], float(f_stat), tol=1e-6)
    chk(f"lda_statistics.anova_f[{fn}].p_value", af[fn]['p_value'], float(p_val), tol=1e-6)

report("DISCRIMINANT ANALYSIS (Python)")
