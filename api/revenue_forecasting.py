"""
Revenue Forecasting Router for FastAPI
Multi-stream revenue forecasting with target tracking
Reuses the same forecasting engine as demand_forecasting
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()


class RevenueForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    value_col: str
    stream_col: Optional[str] = None
    forecast_periods: int = 12
    frequency: Literal["D", "W", "M", "Q", "Y"] = "M"
    confidence_level: float = 0.95
    revenue_target: Optional[float] = None


def _safe(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)): return obj.isoformat()
    if isinstance(obj, np.ndarray): return [_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict): return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_safe(x) for x in obj]
    return obj


# ── Forecasting Methods (same as demand_forecasting) ──

def moving_average_forecast(series, periods, window=3):
    forecast, history = [], list(series)
    for _ in range(periods): avg=np.mean(history[-window:]); forecast.append(avg); history.append(avg)
    fitted=[np.mean(series[:i+1]) if i<window else np.mean(series[i-window:i]) for i in range(len(series))]
    return {'forecast':np.array(forecast),'fitted':np.array(fitted),'method':f'Moving Average'}

def exponential_smoothing_forecast(series, periods, alpha=None):
    if alpha is None:
        def sse(a): a=float(a[0]); f=[float(series[0])]; [f.append(a*series[i-1]+(1-a)*f[-1]) for i in range(1,len(series))]; return np.sum((series-np.array(f))**2)
        alpha=float(minimize(sse,x0=[0.5],bounds=[(0.01,0.99)]).x[0])
    fitted=[float(series[0])]
    for i in range(1,len(series)): fitted.append(alpha*series[i-1]+(1-alpha)*fitted[-1])
    last=alpha*series[-1]+(1-alpha)*fitted[-1]
    return {'forecast':np.array([float(last)]*periods),'fitted':np.array(fitted),'method':'Exponential Smoothing'}

def holt_forecast(series, periods, alpha=None, beta=None):
    n=len(series)
    if alpha is None or beta is None:
        def sse(p):
            a,b=float(p[0]),float(p[1]);lv=float(series[0]);tr=float(series[1]-series[0]) if n>1 else 0.;f=[]
            for i in range(n):
                f.append(lv+tr)
                if i<n-1:nl=a*series[i]+(1-a)*(lv+tr);nt=b*(nl-lv)+(1-b)*tr;lv,tr=float(nl),float(nt)
            return np.sum((series-np.array(f))**2)
        r=minimize(sse,x0=[0.5,0.5],bounds=[(0.01,0.99)]*2);alpha,beta=float(r.x[0]),float(r.x[1])
    lv=float(series[0]);tr=float(series[1]-series[0]) if n>1 else 0.;fitted=[]
    for i in range(n):
        fitted.append(lv+tr)
        if i<n-1:nl=alpha*series[i]+(1-alpha)*(lv+tr);nt=beta*(nl-lv)+(1-beta)*tr;lv,tr=float(nl),float(nt)
    lv=alpha*series[-1]+(1-alpha)*(lv+tr);tr=beta*(lv-(fitted[-1]-tr))+(1-beta)*tr
    return {'forecast':np.array([lv+tr*(i+1) for i in range(periods)]),'fitted':np.array(fitted),'method':'Holt Linear'}

def holt_winters_forecast(series, periods, sp=12):
    n=len(series)
    if n<2*sp: return holt_forecast(series,periods)
    seas=np.zeros(sp)
    for i in range(sp): seas[i]=np.mean([series[j] for j in range(i,min(n,sp*2),sp)])-np.mean(series[:sp*2])
    def sse(p):
        a,b,g=float(p[0]),float(p[1]),float(p[2]);lv=float(np.mean(series[:sp]));tr=float((np.mean(series[sp:2*sp])-np.mean(series[:sp]))/sp);s=seas.copy();f=[]
        for i in range(n):
            si=i%sp;f.append(lv+tr+s[si])
            if i<n-1:nl=a*(series[i]-s[si])+(1-a)*(lv+tr);nt=b*(nl-lv)+(1-b)*tr;s[si]=g*(series[i]-nl)+(1-g)*s[si];lv,tr=float(nl),float(nt)
        return np.sum((series-np.array(f))**2)
    r=minimize(sse,x0=[0.5,0.1,0.5],bounds=[(0.01,0.99)]*3);a,b,g=float(r.x[0]),float(r.x[1]),float(r.x[2])
    lv=float(np.mean(series[:sp]));tr=float((np.mean(series[sp:2*sp])-np.mean(series[:sp]))/sp);s=seas.copy();fitted=[]
    for i in range(n):
        si=i%sp;fitted.append(lv+tr+s[si])
        if i<n-1:nl=a*(series[i]-s[si])+(1-a)*(lv+tr);nt=b*(nl-lv)+(1-b)*tr;s[si]=g*(series[i]-nl)+(1-g)*s[si];lv,tr=float(nl),float(nt)
    si=(n-1)%sp;lv=a*(series[-1]-s[si])+(1-a)*(lv+tr);tr=b*(lv-(fitted[-1]-tr-s[si]))+(1-b)*tr;s[si]=g*(series[-1]-lv)+(1-g)*s[si]
    fc=[lv+tr*(i+1)+s[(n+i)%sp] for i in range(periods)]
    return {'forecast':np.array(fc),'fitted':np.array(fitted),'method':'Holt-Winters'}

def linear_trend_forecast(series, periods):
    n=len(series);x=np.arange(n);slope,intercept,r_value,_,_=stats.linregress(x,series)
    return {'forecast':intercept+slope*np.arange(n,n+periods),'fitted':intercept+slope*x,'method':'Linear Trend'}

def ensemble_forecast(series, periods, sp=12):
    methods=[moving_average_forecast(series,periods),exponential_smoothing_forecast(series,periods),holt_forecast(series,periods),linear_trend_forecast(series,periods)]
    if len(series)>=2*sp: methods.append(holt_winters_forecast(series,periods,sp))
    return {'forecast':np.mean(np.vstack([m['forecast'] for m in methods]),axis=0),'fitted':np.mean(np.vstack([m['fitted'] for m in methods]),axis=0),'method':f'Ensemble ({len(methods)} models)'}


def auto_select(series, sp):
    n=len(series);_,_,r,_,_=stats.linregress(np.arange(n),series)
    if sp and n>=2*sp and abs(r)>0.5: return 'holt_winters'
    elif abs(r)>0.5: return 'holt'
    elif n<10: return 'moving_average'
    return 'exponential'


def detect_seasonality(series, max_lag=24):
    n=len(series);max_lag=min(max_lag,n//2);mean=np.mean(series);var=np.var(series)
    if var==0: return {'has_seasonality':False,'seasonal_period':None}
    acf=[1.0]+[np.mean((series[:n-lag]-mean)*(series[lag:]-mean))/var for lag in range(1,max_lag+1)]
    peaks=sorted([(i,acf[i]) for i in range(2,len(acf)-1) if acf[i]>acf[i-1] and acf[i]>acf[i+1] and acf[i]>0.2],key=lambda x:x[1],reverse=True)
    sp=peaks[0][0] if peaks else None
    return {'has_seasonality':sp is not None,'seasonal_period':sp}


def forecast_single_stream(series, periods, freq_str, confidence):
    """Forecast a single time series and return results"""
    seas = detect_seasonality(series)
    sp = seas.get('seasonal_period')
    method_name = auto_select(series, sp)

    methods = {'moving_average': moving_average_forecast, 'exponential': exponential_smoothing_forecast,
               'holt': holt_forecast, 'linear': linear_trend_forecast}
    if method_name == 'holt_winters': result = holt_winters_forecast(series, periods, sp or 12)
    elif method_name == 'ensemble': result = ensemble_forecast(series, periods, sp or 12)
    else: result = methods.get(method_name, exponential_smoothing_forecast)(series, periods)

    forecast = result['forecast']; fitted = result['fitted']
    std_err = np.std(series - fitted); z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * std_err * np.sqrt(np.arange(1, len(forecast) + 1))

    # Accuracy
    err = series - fitted; ae = np.abs(err); pe = np.abs(err / series) * 100; pe = pe[~np.isinf(pe)]
    mape = float(np.mean(pe)) if len(pe) > 0 else None

    return {
        'method': result['method'],
        'mape': mape,
        'forecast': forecast,
        'lower': forecast - margin,
        'upper': forecast + margin,
        'fitted': fitted,
        'seasonality': seas,
    }


@router.post("/revenue-forecasting")
async def revenue_forecasting(request: RevenueForecastRequest):
    try:
        df = pd.DataFrame(request.data)
        if request.date_col not in df.columns: raise HTTPException(400, f"Column '{request.date_col}' not found")
        if request.value_col not in df.columns: raise HTTPException(400, f"Column '{request.value_col}' not found")

        df[request.date_col] = pd.to_datetime(df[request.date_col], errors='coerce')
        df = df.dropna(subset=[request.date_col]).sort_values(request.date_col)
        df[request.value_col] = pd.to_numeric(df[request.value_col], errors='coerce').fillna(0)

        freq_map = {'D':'D','W':'W','M':'MS','Q':'QS','Y':'YS'}
        freq = freq_map.get(request.frequency, 'MS')

        # Group by stream
        if request.stream_col and request.stream_col in df.columns:
            groups = {name: grp for name, grp in df.groupby(request.stream_col)}
        else:
            groups = {'Total': df}

        stream_results = []
        all_forecast_dates = None

        for stream_name, grp in groups.items():
            agg = grp.groupby(request.date_col)[request.value_col].sum().reset_index().sort_values(request.date_col)
            dates = pd.DatetimeIndex(agg[request.date_col])
            series = agg[request.value_col].values.astype(float)

            if len(series) < 5:
                stream_results.append({
                    'stream': str(stream_name), 'method': 'Insufficient Data', 'mape': None,
                    'original_data': [{'date': d.strftime('%Y-%m-%d'), 'value': _safe(v)} for d, v in zip(dates, series)],
                    'forecast': []
                })
                continue

            res = forecast_single_stream(series, request.forecast_periods, request.frequency, request.confidence_level)

            last_date = dates[-1]
            fc_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1) if request.frequency == 'M' else last_date + timedelta(days=1),
                periods=request.forecast_periods, freq=freq
            )
            if all_forecast_dates is None: all_forecast_dates = fc_dates

            stream_results.append({
                'stream': str(stream_name),
                'method': res['method'],
                'mape': _safe(res['mape']),
                'seasonality_detected': res['seasonality'].get('has_seasonality', False),
                'seasonal_period': res['seasonality'].get('seasonal_period'),
                'original_data': [{'date': d.strftime('%Y-%m-%d'), 'value': _safe(v)} for d, v in zip(dates, series)],
                'forecast': [{'date': d.strftime('%Y-%m-%d'), 'forecast': _safe(float(res['forecast'][i])), 'lower': _safe(float(res['lower'][i])), 'upper': _safe(float(res['upper'][i]))} for i, d in enumerate(fc_dates)]
            })

        # Build totals
        date_totals = {}
        for sr in stream_results:
            for d in sr['original_data']:
                date_totals.setdefault(d['date'], 0)
                date_totals[d['date']] += (d['value'] or 0)
        total_original = [{'date': k, 'value': v} for k, v in sorted(date_totals.items())]

        fc_totals = {}
        for sr in stream_results:
            for f in sr['forecast']:
                fc_totals.setdefault(f['date'], {'mean': 0, 'lo': 0, 'hi': 0})
                fc_totals[f['date']]['mean'] += (f['forecast'] or 0)
                fc_totals[f['date']]['lo'] += (f['lower'] or 0)
                fc_totals[f['date']]['hi'] += (f['upper'] or 0)
        total_forecast = [{'date': k, 'forecast': v['mean'], 'lower': v['lo'], 'upper': v['hi']} for k, v in sorted(fc_totals.items())]

        # Metrics
        mapes = [s['mape'] for s in stream_results if s['mape'] is not None]
        avg_mape = sum(mapes) / len(mapes) if mapes else None

        # YTD, run rate, YoY
        now = datetime.now()
        cy = now.year
        ytd = sum(d['value'] for d in total_original if d['date'].startswith(str(cy)))
        last3 = total_original[-3:] if len(total_original) >= 3 else total_original
        run_rate = (sum(d['value'] for d in last3) / len(last3)) * 12 if last3 else 0

        fc_total_sum = sum(f['forecast'] for f in total_forecast)
        forecast_total = ytd + fc_total_sum

        py = cy - 1
        py_total = sum(d['value'] for d in total_original if d['date'].startswith(str(py)))
        yoy_growth = ((run_rate - py_total) / py_total * 100) if py_total > 0 else None

        target = request.revenue_target or 0
        target_gap = forecast_total - target if target > 0 else 0
        target_pct = (forecast_total / target * 100) if target > 0 else 0

        return _safe({
            'success': True,
            'streams': stream_results,
            'total_original': total_original,
            'total_forecast': total_forecast,
            'metrics': {
                'avg_mape': avg_mape,
                'ytd_revenue': ytd,
                'run_rate': run_rate,
                'forecast_total': forecast_total,
                'yoy_growth': yoy_growth,
                'target_gap': target_gap,
                'target_pct': target_pct,
                'stream_count': len(stream_results),
            }
        })

    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Revenue forecasting failed: {str(e)}")
