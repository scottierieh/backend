# Validates lasso_regression_analysis.py against an independent sklearn Lasso
# fit reproducing the handler's split / standardization / seed. Every returned
# leaf value is checked: all coefficients, intercept, alpha, and train/test
# R2/RMSE/MAE.
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from _pyharness import run_script, chk, report

rng=np.random.RandomState(1)
n=120
X=pd.DataFrame({'x1':rng.normal(size=n),'x2':rng.normal(size=n),'x3':rng.normal(size=n)})
y=2*X['x1']-1.5*X['x2']+0.5*X['x3']+rng.normal(scale=0.5,size=n)
df=pd.concat([X,y.rename('y')],axis=1)
feats=['x1','x2','x3']; ALPHA=0.1
payload={'data':df.to_dict('records'),'target':'y','features':feats,'alpha':ALPHA,'test_size':0.2}
res=run_script('lasso_regression_analysis.py', payload)
r=res['results']

Xa=df[feats]; ya=pd.to_numeric(df['y'])
Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=0.2,random_state=42)
sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
m=Lasso(alpha=ALPHA,random_state=42).fit(Xtr_s,ytr)
pred_tr=m.predict(Xtr_s); pred_te=m.predict(Xte_s)

chk("lasso.intercept", r['intercept'], m.intercept_, tol=1e-8)
chk("lasso.alpha", r['alpha'], ALPHA, tol=1e-12)
for i,f in enumerate(feats):
    chk(f"lasso.coef.{f}", r['coefficients'][f], m.coef_[i], tol=1e-8)
chk("lasso.test.r2",   r['metrics']['test']['r2_score'], r2_score(yte,pred_te), tol=1e-8)
chk("lasso.test.rmse", r['metrics']['test']['rmse'], np.sqrt(mean_squared_error(yte,pred_te)), tol=1e-8)
chk("lasso.test.mae",  r['metrics']['test']['mae'], mean_absolute_error(yte,pred_te), tol=1e-8)
chk("lasso.train.r2",  r['metrics']['train']['r2_score'], r2_score(ytr,pred_tr), tol=1e-8)
chk("lasso.train.rmse",r['metrics']['train']['rmse'], np.sqrt(mean_squared_error(ytr,pred_tr)), tol=1e-8)
chk("lasso.train.mae", r['metrics']['train']['mae'], mean_absolute_error(ytr,pred_tr), tol=1e-8)
for k in ['plot','path_plot']:
    chk(f"has.{k}", isinstance(res.get(k),str) and res[k].startswith('data:image/png'), True)
report("LASSO REGRESSION (Python)")
