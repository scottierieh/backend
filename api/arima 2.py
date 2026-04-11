from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import io
import base64
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class ArimaRequest(BaseModel):
    data: List[Dict[str, Any]]
    timeCol: str
    valueCol: str
    order: Optional[List[int]] = None   # [p, d, q] — None triggers auto_order
    seasonalOrder: Optional[List[int]] = None  # [P, D, Q, s]
    exogCols: Optional[List[str]] = None
    forecastPeriods: int = 12
    auto_order: bool = True             # AIC grid search when order is None
    max_p: int = 3                      # grid search bound for p
    max_q: int = 3                      # grid search bound for q
    max_d: int = 2                      # grid search bound for d


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except:
        return default



# ══════════════════════════════════════════════════════════════════
# ① Auto ARIMA order selection via AIC grid search
# ══════════════════════════════════════════════════════════════════

def _auto_order(series: pd.Series,
                max_p: int = 3, max_d: int = 2, max_q: int = 3,
                seasonal_order: tuple = (0, 0, 0, 0),
                exog=None) -> dict:
    """
    Exhaustive AIC-minimising grid search over ARIMA(p,d,q).

    Search space: p ∈ [0..max_p], d ∈ [0..max_d], q ∈ [0..max_q].
    Total candidates = (max_p+1) × (max_d+1) × (max_q+1) — bounded by
    defaults 4×3×4 = 48 fits, typically < 5 s on n ≤ 500.

    Differencing bound d: we additionally constrain d by running ADF;
    if the series is already stationary at d=0 we cap the search at d≤1
    to avoid over-differencing.

    Returns:
      best_order (p,d,q), best_aic, comparison_table (sorted by AIC),
      adf_suggested_d.
    """
    # ADF-guided d upper bound
    adf_d = 0
    try:
        p_val = adfuller(series.dropna(), autolag='AIC')[1]
        if p_val > 0.05:
            adf_d = 1
            # check if d=1 is still non-stationary
            s1 = series.diff().dropna()
            if len(s1) > 4 and adfuller(s1, autolag='AIC')[1] > 0.05:
                adf_d = 2
    except Exception:
        adf_d = max_d

    d_cap = min(max_d, max(adf_d, 1))   # always allow at least d=0,1

    best_aic   = np.inf
    best_order = (1, 1, 1)
    table      = []

    for d in range(0, d_cap + 1):
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                if p == 0 and q == 0:
                    continue   # pure random walk — skip
                try:
                    m = SARIMAX(series, exog=exog,
                                order=(p, d, q),
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                    f = m.fit(disp=False)
                    aic = float(f.aic)
                    if np.isfinite(aic):
                        table.append({'order': [p, d, q],
                                      'aic': round(aic, 3),
                                      'bic': round(float(f.bic), 3)})
                        if aic < best_aic:
                            best_aic   = aic
                            best_order = (p, d, q)
                except Exception:
                    pass

    table.sort(key=lambda x: x['aic'])
    return {
        'best_order':      list(best_order),
        'best_aic':        round(best_aic, 3) if np.isfinite(best_aic) else None,
        'adf_suggested_d': adf_d,
        'n_candidates':    len(table),
        'top_candidates':  table[:5],
    }


# ══════════════════════════════════════════════════════════════════
# ② In-sample residual diagnostics
# ══════════════════════════════════════════════════════════════════

def _residual_diagnostics(model_fit, series: pd.Series) -> dict:
    """
    In-sample error metrics + Ljung-Box white-noise test on residuals.

    Metrics
    -------
    RMSE  : Root Mean Squared Error
    MAE   : Mean Absolute Error
    MAPE  : Mean Absolute Percentage Error (skipped when any actual = 0)
    SMAPE : Symmetric MAPE
    ME    : Mean Error (signed bias)

    Ljung-Box
    ---------
    lags = min(20, n//4).  Significant (p < 0.05) → residuals are not
    white noise → model leaves structure unexplained.
    """
    resid = model_fit.resid.dropna()
    fv    = model_fit.fittedvalues.reindex(resid.index)
    n     = len(resid)

    actual = series.reindex(resid.index).values
    rv     = resid.values

    rmse  = float(np.sqrt(np.mean(rv ** 2)))
    mae   = float(np.mean(np.abs(rv)))
    me    = float(np.mean(rv))

    if (np.abs(actual) > 1e-8).all():
        mape = float(np.mean(np.abs(rv / actual)) * 100)
    else:
        mape = None

    denom  = (np.abs(actual) + np.abs(fv.values) + 1e-8)
    smape  = float(np.mean(2 * np.abs(rv) / denom) * 100)

    # Ljung-Box on residuals (model_df = p+q to correct for estimated params)
    lb_lags = max(1, min(20, n // 4))
    order   = model_fit.model.order
    mdf     = order[0] + order[2]   # p + q
    try:
        lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True,
                            boxpierce=False, model_df=mdf)
        lb_stat   = float(lb['lb_stat'].iloc[0])
        lb_pvalue = float(lb['lb_pvalue'].iloc[0])
        lb_sig    = lb_pvalue < 0.05
    except Exception:
        lb_stat = lb_pvalue = None; lb_sig = None

    resid_desc = {
        'mean':     round(float(resid.mean()), 6),
        'std':      round(float(resid.std()),  6),
        'min':      round(float(resid.min()),  6),
        'max':      round(float(resid.max()),  6),
        'skew':     round(float(resid.skew()), 4),
        'kurtosis': round(float(resid.kurtosis()), 4),
    }

    lb_interp = None
    if lb_sig is not None:
        lb_interp = ('Residuals show significant autocorrelation — '
                     'model may be under-specified (try higher p or q).'
                     if lb_sig else
                     'No significant autocorrelation in residuals — '
                     'model appears well-specified.')

    return {
        'rmse':  round(rmse, 4),
        'mae':   round(mae,  4),
        'mape':  round(mape, 4) if mape is not None else None,
        'smape': round(smape, 4),
        'me':    round(me,   6),
        'n_residuals': n,
        'residual_summary': resid_desc,
        'ljung_box': {
            'statistic':      lb_stat,
            'p_value':        lb_pvalue,
            'lags':           lb_lags,
            'model_df':       mdf,
            'is_significant': lb_sig,
            'interpretation': lb_interp,
        },
    }


# ══════════════════════════════════════════════════════════════════
# ③ Robust forecast index generation
# ══════════════════════════════════════════════════════════════════

def _forecast_index(series: pd.Series, h: int) -> pd.DatetimeIndex:
    """
    Generate a future DatetimeIndex of length h starting after the
    last observation, using one of three strategies in priority order:

    1. pd.infer_freq — standard path for regular series
    2. Median timedelta — robust to 1-2 irregular gaps (e.g. missing months)
    3. Integer fallback — if the index is not datetime at all

    Using the median rather than mean avoids outlier gaps (e.g. a leap-year
    or a data collection pause) distorting the spacing.
    """
    last = series.index[-1]
    try:
        freq = pd.infer_freq(series.index)
        if freq:
            return pd.date_range(start=last, periods=h + 1, freq=freq)[1:]
    except Exception:
        pass

    # Median timedelta approach
    try:
        deltas = pd.Series(series.index).diff().dropna()
        median_delta = deltas.median()
        if pd.isnull(median_delta) or median_delta.total_seconds() <= 0:
            raise ValueError
        return pd.DatetimeIndex([last + median_delta * (i + 1) for i in range(h)])
    except Exception:
        pass

    # Last resort: integer steps
    return pd.RangeIndex(start=int(last) + 1, stop=int(last) + h + 1)                if not isinstance(last, pd.Timestamp)                else pd.date_range(start=last, periods=h + 1, freq='ME')[1:]


@router.post("/arima")
async def arima_analysis(request: ArimaRequest):
    try:
        df               = pd.DataFrame(request.data)
        time_col         = request.timeCol
        value_col        = request.valueCol
        seasonal_order   = tuple(request.seasonalOrder) if request.seasonalOrder else (0, 0, 0, 0)
        exog_cols        = request.exogCols
        forecast_periods = request.forecastPeriods

        if time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{time_col}' not found")
        if value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{value_col}' not found")

        # ── Data preparation ──────────────────────────────────────────────────
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[time_col, value_col]).set_index(time_col).sort_index()
        series = df[value_col]

        # ── Exogenous variables ───────────────────────────────────────────────
        exog_data = None
        if exog_cols:
            valid_exog = [c for c in exog_cols if c in df.columns]
            if valid_exog:
                exog_data = df[valid_exog].apply(pd.to_numeric, errors='coerce').dropna()
                series, exog_data = series.align(exog_data, join='inner')

        # ── ① Order: auto or manual ───────────────────────────────────────────
        auto_info = None
        if request.order is None or request.auto_order:
            auto_info = _auto_order(
                series,
                max_p=request.max_p, max_d=request.max_d, max_q=request.max_q,
                seasonal_order=seasonal_order, exog=exog_data
            )
            order = tuple(auto_info['best_order'])
        else:
            order = tuple(request.order)

        if len(series) < sum(order[:3]) + 4:
            raise HTTPException(status_code=400,
                detail="Not enough data to fit the ARIMA model with the chosen order")

        # ── Fit SARIMAX ───────────────────────────────────────────────────────
        model = SARIMAX(
            series, exog=exog_data,
            order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)

        # ── ② Residual diagnostics ────────────────────────────────────────────
        diag = _residual_diagnostics(model_fit, series)

        # ── ③ Forecast with robust index ─────────────────────────────────────
        exog_forecast = None
        if exog_data is not None and not exog_data.empty:
            fc_idx = _forecast_index(series, forecast_periods)
            exog_forecast = pd.DataFrame(
                [exog_data.iloc[-1].values] * forecast_periods,
                index=fc_idx, columns=list(exog_data.columns)
            )

        forecast     = model_fit.get_forecast(steps=forecast_periods, exog=exog_forecast)
        forecast_df  = forecast.summary_frame(alpha=0.05)
        forecast_df.index.name = 'forecast_date'

        # Use robust index for consistent date labelling
        fc_index = _forecast_index(series, forecast_periods)
        if len(fc_index) == len(forecast_df):
            forecast_df.index = fc_index
        forecast_df.index.name = 'forecast_date'

        # ── Plots ─────────────────────────────────────────────────────────────
        # Plot 1: Forecast
        fig, ax = plt.subplots(figsize=(14, 6))
        title_order = f'SARIMA{order}×{seasonal_order}' if any(seasonal_order[:3]) else f'ARIMA{order}'
        title_auto  = '  (auto-selected)' if auto_info else ''
        fig.suptitle(f'{title_order} Forecast{title_auto}', fontsize=14, fontweight='bold')

        ax.plot(series.index, series.values,
                label='Observed', color='#1f77b4', lw=1.5, alpha=0.8)
        ax.plot(model_fit.fittedvalues.index, model_fit.fittedvalues.values,
                label='In-sample fit', color='#2ca02c', lw=1, alpha=0.6, linestyle='--')
        ax.plot(forecast_df.index, forecast_df['mean'].values,
                label='Forecast', color='#ff7f0e', lw=2)
        ax.fill_between(forecast_df.index,
                        forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'],
                        color='#ff7f0e', alpha=0.20, label='95% CI')

        # Info box: order + diagnostics
        info = (f'Order: {order}\n'
                f'AIC={model_fit.aic:.1f}\n'
                f'RMSE={diag["rmse"]}  MAE={diag["mae"]}\n'
                f'MAPE={diag["mape"]}%' if diag["mape"] else
                f'Order: {order}\nAIC={model_fit.aic:.1f}\nRMSE={diag["rmse"]}')
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.65))
        ax.set_xlabel(time_col, fontsize=11)
        ax.set_ylabel(value_col, fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.4)
        plt.tight_layout()

        buf1 = io.BytesIO()
        fig.savefig(buf1, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf1.seek(0)
        forecast_plot_b64 = base64.b64encode(buf1.read()).decode('utf-8')

        # Plot 2: Statsmodels diagnostics (unchanged)
        diag_fig = model_fit.plot_diagnostics(figsize=(14, 10))
        buf2 = io.BytesIO()
        diag_fig.savefig(buf2, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(diag_fig)
        buf2.seek(0)
        diag_plot_b64 = base64.b64encode(buf2.read()).decode('utf-8')

        # ── Auto-order comparison plot ─────────────────────────────────────────
        auto_plot_b64 = None
        if auto_info and auto_info['top_candidates']:
            cands = auto_info['top_candidates'][:8]
            labels = [str(tuple(c['order'])) for c in cands]
            aics   = [c['aic'] for c in cands]
            colors = ['#2ecc71' if i == 0 else '#4C72B0' for i in range(len(cands))]

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            bars = ax3.bar(labels, aics, color=colors, alpha=0.85, edgecolor='white')
            ax3.set_title('Auto ARIMA — AIC by Order (green = selected)', fontweight='bold')
            ax3.set_xlabel('ARIMA(p,d,q)')
            ax3.set_ylabel('AIC')
            ax3.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, aics):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            from matplotlib.patches import Patch
            ax3.legend(handles=[Patch(color='#2ecc71', label=f'Selected {tuple(auto_info["best_order"])}'),
                                 Patch(color='#4C72B0', label='Other candidates')], fontsize=8)
            plt.tight_layout()
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format='png', dpi=110, bbox_inches='tight', facecolor='white')
            plt.close(fig3)
            buf3.seek(0)
            auto_plot_b64 = base64.b64encode(buf3.read()).decode('utf-8')

        # ── Summary tables ────────────────────────────────────────────────────
        summary_obj = model_fit.summary()
        summary_data = [{'caption': getattr(t, 'title', None),
                         'data': [list(row) for row in t.data]}
                        for t in summary_obj.tables]

        # ── Forecast records ──────────────────────────────────────────────────
        forecast_records = []
        for _, row in forecast_df.reset_index().iterrows():
            dt = row['forecast_date']
            forecast_records.append({
                'forecast_date':   dt.isoformat() if hasattr(dt, 'isoformat') else str(dt),
                'mean':            safe_float(row['mean']),
                'mean_se':         safe_float(row.get('mean_se', 0)),
                'mean_ci_lower':   safe_float(row['mean_ci_lower']),
                'mean_ci_upper':   safe_float(row['mean_ci_upper']),
            })

        # ── Response ─────────────────────────────────────────────────────────
        response = {
            'results': {
                'summary_data':   summary_data,
                'aic':            safe_float(model_fit.aic),
                'bic':            safe_float(model_fit.bic),
                'hqic':           safe_float(model_fit.hqic),
                'forecast':       forecast_records,
                'n_observations': len(series),
                # ① Auto order
                'order_used':     list(order),
                'seasonal_order': list(seasonal_order),
                'auto_selected':  auto_info is not None,
                'auto_order_info': auto_info,
                # ② Diagnostics
                'diagnostics':    diag,
            },
            'plot':             f'data:image/png;base64,{forecast_plot_b64}',
            'diagnostics_plot': f'data:image/png;base64,{diag_plot_b64}',
            'auto_order_plot':  f'data:image/png;base64,{auto_plot_b64}' if auto_plot_b64 else None,
        }

        return _to_native(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
