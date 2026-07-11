# Validates relative_importance_analysis.py (semi-partial R² relative weights) vs statsmodels.
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from _pyharness import run_script, chk, report
rng=np.random.RandomState(0); n=200
x1=rng.normal(size=n); x2=rng.normal(size=n); x3=rng.normal(size=n)
y=2*x1+1*x2+0.2*x3+rng.normal(size=n)
df=pd.DataFrame({'y':y,'x1':x1,'x2':x2,'x3':x3})
r=run_script('relative_importance_analysis.py', {'data':df.to_dict('records'),'dependent_var':'y','independent_vars':['x1','x2','x3']})
rows=r['results'] if isinstance(r,dict) and 'results' in r else r
recs={d['predictor']:d for d in (rows if isinstance(rows,list) else rows.get('data',rows))}
# reproduce with statsmodels
xs=['x1','x2','x3']; sc=StandardScaler(); ds=pd.DataFrame(sc.fit_transform(df[['y']+xs]),columns=['y']+xs)
full=sm.OLS(ds['y'],sm.add_constant(ds[xs])).fit(); fr=full.rsquared
sp={p: fr - sm.OLS(ds['y'],sm.add_constant(ds[[c for c in xs if c!=p]])).fit().rsquared for p in xs}
tot=sum(sp.values())
for p in xs:
    chk(f"ri.{p}.semi_partial_r2", recs[p]['semi_partial_r2'], sp[p], tol=1e-6)
    chk(f"ri.{p}.relative_weight", recs[p]['relative_weight_pct'], sp[p]/tot*100, tol=1e-4)
report("RELATIVE IMPORTANCE (Python)")
