# Validates elastic_net_regression_analysis.py against a direct sklearn ElasticNet fit.
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from _pyharness import run_script, chk, report

rng=np.random.RandomState(2)
n=140
X=pd.DataFrame({'x1':rng.normal(size=n),'x2':rng.normal(size=n),'x3':rng.normal(size=n)})
y=2*X['x1']-1.5*X['x2']+0.0*X['x3']+rng.normal(scale=0.5,size=n)
df=pd.concat([X,y.rename('y')],axis=1)
payload={'data':df.to_dict('records'),'target':'y','features':['x1','x2','x3'],
         'alpha':0.1,'l1_ratio':0.5,'test_size':0.2,'use_cv':False}
res=run_script('elastic_net_regression_analysis.py', payload)
r=res['results'] if 'results' in res else res

Xa=df[['x1','x2','x3']]; ya=pd.to_numeric(df['y'])
Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=0.2,random_state=42)
sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
m=ElasticNet(alpha=0.1,l1_ratio=0.5,random_state=42,max_iter=10000).fit(Xtr_s,ytr)
chk("enet.intercept", r['intercept'], m.intercept_, tol=1e-8)
for i,f in enumerate(['x1','x2','x3']):
    chk(f"enet.coef.{f}", r['coefficients'][f], m.coef_[i], tol=1e-8)
chk("enet.test.r2", r['metrics']['test']['r2_score'], r2_score(yte,m.predict(Xte_s)), tol=1e-8)
chk("enet.test.rmse", r['metrics']['test']['rmse'], np.sqrt(mean_squared_error(yte,m.predict(Xte_s))), tol=1e-8)
chk("enet.train.r2", r['metrics']['train']['r2_score'], r2_score(ytr,m.predict(Xtr_s)), tol=1e-8)
report("ELASTIC NET REGRESSION (Python)")
