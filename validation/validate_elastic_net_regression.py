# Validates elastic_net_regression_analysis.py against an independent sklearn
# ElasticNet fit reproducing the handler's split / standardization / seed.
# Every returned leaf value is checked: all coefficients, intercept, alpha,
# l1_ratio, n_nonzero_coefficients, and train/test R2/RMSE/MAE.
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from _pyharness import run_script, chk, report

rng=np.random.RandomState(2); n=140
X=pd.DataFrame({'x1':rng.normal(size=n),'x2':rng.normal(size=n),'x3':rng.normal(size=n)})
y=2*X['x1']-1.5*X['x2']+0.0*X['x3']+rng.normal(scale=0.5,size=n)
df=pd.concat([X,y.rename('y')],axis=1)
feats=['x1','x2','x3']
ALPHA, L1 = 0.1, 0.5
payload={'data':df.to_dict('records'),'target':'y','features':feats,
         'alpha':ALPHA,'l1_ratio':L1,'test_size':0.2,'use_cv':False,'cv_folds':5}
res=run_script('elastic_net_regression_analysis.py', payload)
r=res['results']

# --- independent reproduction (same split/scaler/alpha/l1_ratio, random_state=42) ---
Xa=df[feats]; ya=pd.to_numeric(df['y'])
Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=0.2,random_state=42)
sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
m=ElasticNet(alpha=ALPHA,l1_ratio=L1,random_state=42,max_iter=10000).fit(Xtr_s,ytr)
pred_tr=m.predict(Xtr_s); pred_te=m.predict(Xte_s)

# --- scalar params ---
chk("enet.intercept", r['intercept'], m.intercept_, tol=1e-8)
chk("enet.alpha", r['alpha'], ALPHA, tol=1e-12)
chk("enet.l1_ratio", r['l1_ratio'], L1, tol=1e-12)
chk("enet.n_nonzero_coefficients", r['n_nonzero_coefficients'], int(np.sum(m.coef_!=0)))

# --- every coefficient value-by-value ---
for i,f in enumerate(feats):
    chk(f"enet.coef.{f}", r['coefficients'][f], m.coef_[i], tol=1e-8)

# --- every metric, train & test ---
chk("enet.test.r2",   r['metrics']['test']['r2_score'], r2_score(yte,pred_te), tol=1e-8)
chk("enet.test.rmse", r['metrics']['test']['rmse'], np.sqrt(mean_squared_error(yte,pred_te)), tol=1e-8)
chk("enet.test.mae",  r['metrics']['test']['mae'], mean_absolute_error(yte,pred_te), tol=1e-8)
chk("enet.train.r2",  r['metrics']['train']['r2_score'], r2_score(ytr,pred_tr), tol=1e-8)
chk("enet.train.rmse",r['metrics']['train']['rmse'], np.sqrt(mean_squared_error(ytr,pred_tr)), tol=1e-8)
chk("enet.train.mae", r['metrics']['train']['mae'], mean_absolute_error(ytr,pred_tr), tol=1e-8)

# --- plot images present ---
for k in ['plot','path_plot']:
    chk(f"has.{k}", isinstance(res.get(k),str) and res[k].startswith('data:image/png'), True)

report("ELASTIC NET REGRESSION (Python)")
