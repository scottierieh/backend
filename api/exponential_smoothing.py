from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class ExponentialSmoothingRequest(BaseModel):
    data: List[Dict[str, Any]]
    timeCol: str
    valueCol: str
    smoothingType: str = 'auto'         # 'auto' | 'simple' | 'holt' | 'holt-winters'
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    trendType: Optional[str] = None     # 'add' | 'mul'
    seasonalType: Optional[str] = None  # 'add' | 'mul'
    seasonalPeriods: Optional[int] = None
    forecastPeriods: int = 12           # how many steps ahead to forecast
    confidenceLevel: float = 0.95       # PI coverage: 0.80 | 0.90 | 0.95
    nSimulations: int = 1000            # Monte-Carlo draws for prediction intervals


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=None):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except:
        return default



# ══════════════════════════════════════════════════════════════════
# ① Auto model selection
# ══════════════════════════════════════════════════════════════════

def _auto_select_model(series: pd.Series, seasonal_periods: int) -> dict:
    """
    Select simple / Holt / Holt-Winters by comparing AICc of candidate fits.

    Rules (if explicit candidates fail, fall back gracefully):
      1. Simple ES           — no trend, no seasonality
      2. Holt additive       — additive trend, no seasonality
      3. Holt multiplicative — multiplicative trend (positive series only)
      4. HW add+add          — additive trend + additive seasonal
      5. HW add+mul          — additive trend + multiplicative seasonal (positive only)

    Returns the name and fitted result of the winner, plus a comparison table.
    """
    n = len(series)
    positive = bool((series > 0).all())

    candidates = [
        ('simple',      dict(trend=None,  seasonal=None)),
        ('holt_add',    dict(trend='add', seasonal=None)),
    ]
    if positive:
        candidates.append(('holt_mul', dict(trend='mul', seasonal=None)))
    if seasonal_periods and n >= seasonal_periods * 2:
        candidates.append(('hw_add_add', dict(trend='add', seasonal='add',
                                              seasonal_periods=seasonal_periods)))
        if positive:
            candidates.append(('hw_add_mul', dict(trend='add', seasonal='mul',
                                                  seasonal_periods=seasonal_periods)))

    best_name, best_fit, best_aicc = None, None, np.inf
    comparison = []
    for name, kwargs in candidates:
        try:
            m = ExponentialSmoothing(series, initialization_method='estimated', **kwargs)
            f = m.fit(optimized=True)
            aicc = float(f.aicc) if np.isfinite(f.aicc) else np.inf
            comparison.append({'model': name, 'aicc': round(aicc, 2),
                                'aic': round(float(f.aic), 2),
                                'bic': round(float(f.bic), 2)})
            if aicc < best_aicc:
                best_aicc, best_name, best_fit = aicc, name, f
        except Exception:
            comparison.append({'model': name, 'aicc': None, 'aic': None, 'bic': None})

    # Map internal name → user-facing type label
    label_map = {
        'simple':     'simple',
        'holt_add':   'holt',
        'holt_mul':   'holt',
        'hw_add_add': 'holt-winters',
        'hw_add_mul': 'holt-winters',
    }
    return {
        'selected_model':   best_name,
        'selected_label':   label_map.get(best_name, best_name),
        'selected_aicc':    round(best_aicc, 2),
        'comparison_table': comparison,
        'fitted':           best_fit,
    }


# ══════════════════════════════════════════════════════════════════
# ② Prediction intervals via Monte-Carlo simulation
# ══════════════════════════════════════════════════════════════════

def _prediction_intervals(fitted, h: int, level: float = 0.95,
                           n_sim: int = 1000) -> dict:
    """
    Bootstrap prediction intervals using statsmodels simulate().

    fitted.simulate(h, repetitions=n_sim, error='add') draws n_sim
    future sample paths of length h.  We take empirical quantiles at
    (1−level)/2 and 1−(1−level)/2.

    Returns dict with 'lower', 'upper', 'level', 'point_forecast'.
    """
    alpha_half = (1 - level) / 2
    try:
        sim = fitted.simulate(h, repetitions=n_sim, error='add')  # (h, n_sim)
        lo  = np.percentile(sim, alpha_half * 100,      axis=1)
        hi  = np.percentile(sim, (1 - alpha_half) * 100, axis=1)
    except Exception:
        # Fallback: symmetric normal interval from residual std
        sigma = float(fitted.resid.std())
        z     = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(level, 1.960)
        fc    = fitted.forecast(h).values
        lo    = fc - z * sigma * np.sqrt(np.arange(1, h + 1))
        hi    = fc + z * sigma * np.sqrt(np.arange(1, h + 1))

    fc = fitted.forecast(h).values
    return {
        'lower':         lo.tolist(),
        'upper':         hi.tolist(),
        'point_forecast': fc.tolist(),
        'level':         level,
    }


# ══════════════════════════════════════════════════════════════════
# ③ Diagnostics
# ══════════════════════════════════════════════════════════════════

def _diagnostics(fitted, series: pd.Series) -> dict:
    """
    In-sample error metrics + Ljung-Box residual test.

    Metrics
    -------
    MAE   : Mean Absolute Error
    RMSE  : Root Mean Squared Error
    MAPE  : Mean Absolute Percentage Error  (undefined if any y=0, skipped)
    SMAPE : Symmetric MAPE  (avoids y=0 issue)
    ME    : Mean Error (bias)

    Ljung-Box
    ---------
    Tests residuals for remaining autocorrelation.
    lags = min(20, n//4).  Significant → model under-fits structure.
    """
    resid = fitted.resid.dropna()
    fv    = fitted.fittedvalues.dropna()
    n     = len(resid)

    mae  = float(np.abs(resid).mean())
    rmse = float(np.sqrt((resid ** 2).mean()))
    me   = float(resid.mean())

    # MAPE — skip if any actual = 0
    actual = series.iloc[-n:].values
    if (np.abs(actual) > 1e-8).all():
        mape  = float(np.abs(resid.values / actual).mean() * 100)
    else:
        mape  = None

    # SMAPE
    smape = float(
        (2 * np.abs(resid.values) / (np.abs(actual) + np.abs(fv.values) + 1e-8)).mean() * 100
    )

    # Ljung-Box
    lb_lags = max(1, min(20, n // 4))
    try:
        lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True, boxpierce=False)
        lb_stat   = float(lb['lb_stat'].iloc[0])
        lb_pvalue = float(lb['lb_pvalue'].iloc[0])
        lb_sig    = lb_pvalue < 0.05
    except Exception:
        lb_stat = lb_pvalue = None; lb_sig = None

    resid_desc = {
        'mean':  round(float(resid.mean()), 6),
        'std':   round(float(resid.std()),  6),
        'min':   round(float(resid.min()),  6),
        'max':   round(float(resid.max()),  6),
        'skew':  round(float(resid.skew()), 4),
        'kurtosis': round(float(resid.kurtosis()), 4),
    }

    # Interpretation
    lb_interp = None
    if lb_sig is not None:
        lb_interp = ('Residuals show significant autocorrelation — model may under-fit.'
                     if lb_sig else
                     'No significant autocorrelation in residuals — model well-specified.')

    return {
        'mae':   round(mae,  4),
        'rmse':  round(rmse, 4),
        'mape':  round(mape, 4) if mape is not None else None,
        'smape': round(smape, 4),
        'me':    round(me,   6),
        'n_residuals': n,
        'residual_summary': resid_desc,
        'ljung_box': {
            'statistic': lb_stat,
            'p_value':   lb_pvalue,
            'lags':      lb_lags,
            'is_significant': lb_sig,
            'interpretation': lb_interp,
        },
    }


# ══════════════════════════════════════════════════════════════════
# ④ Date-aware forecast index
# ══════════════════════════════════════════════════════════════════

def _forecast_dates(last_date: pd.Timestamp, h: int,
                    freq_hint: str = None) -> list:
    """
    Infer frequency from last_date + freq_hint and generate h future dates.
    Falls back to integer indices if inference fails.
    """
    try:
        freq_map = {
            'M': 'MS', 'Q': 'QS', 'A': 'AS', 'Y': 'AS',
            'D': 'D',  'W': 'W',  'H': 'h',
        }
        freq = None
        if freq_hint:
            freq = freq_map.get(freq_hint.upper()[:1])
        if freq is None:
            freq = 'MS'   # default: monthly
        future = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]
        return [str(d)[:10] for d in future]
    except Exception:
        return [f't+{i+1}' for i in range(h)]


@router.post("/exponential-smoothing")
async def exponential_smoothing(request: ExponentialSmoothingRequest):
    try:
        df            = pd.DataFrame(request.data)
        time_col      = request.timeCol
        value_col     = request.valueCol
        smoothing_type = request.smoothingType.lower().strip()
        alpha         = request.alpha
        beta          = request.beta
        gamma         = request.gamma
        trend_type    = request.trendType
        seasonal_type = request.seasonalType
        seasonal_periods = request.seasonalPeriods
        h             = max(1, int(request.forecastPeriods))
        ci_level      = min(0.999, max(0.5, float(request.confidenceLevel)))
        n_sim         = max(200, int(request.nSimulations))

        if time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{time_col}' not found")
        if value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{value_col}' not found")

        # ── Parse & sort ─────────────────────────────────────────────────────
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col, value_col]).sort_values(time_col).reset_index(drop=True)
        series   = pd.to_numeric(df[value_col], errors='coerce').dropna().reset_index(drop=True)
        dates    = df[time_col].reset_index(drop=True)
        n        = len(series)
        last_date = dates.iloc[-1]

        if n < 10:
            raise HTTPException(status_code=400, detail=f"Need at least 10 observations, got {n}")

        # ── ① Auto model selection ────────────────────────────────────────────
        if smoothing_type == 'auto':
            auto = _auto_select_model(series, seasonal_periods or 12)
            fitted       = auto['fitted']
            smoothing_type = auto['selected_label']
            auto_info    = {k: v for k, v in auto.items() if k != 'fitted'}
            auto_selected = True
        else:
            auto_info    = None
            auto_selected = False

            # Manual model build (original logic, preserved)
            if smoothing_type == 'simple':
                model = ExponentialSmoothing(series, trend=None, seasonal=None,
                                             initialization_method='estimated')
            elif smoothing_type == 'holt':
                trend = 'mul' if trend_type == 'mul' else 'add'
                model = ExponentialSmoothing(series, trend=trend, seasonal=None,
                                             initialization_method='estimated')
            elif smoothing_type == 'holt-winters':
                if not seasonal_periods or seasonal_periods < 2:
                    raise HTTPException(status_code=400,
                        detail="seasonal_periods must be >= 2 for Holt-Winters")
                if n < seasonal_periods * 2:
                    raise HTTPException(status_code=400,
                        detail=f"Need at least {seasonal_periods*2} obs for period {seasonal_periods}")
                trend    = 'mul' if trend_type    == 'mul' else 'add'
                seasonal = 'mul' if seasonal_type == 'mul' else 'add'
                model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal,
                                             seasonal_periods=seasonal_periods,
                                             initialization_method='estimated')
            else:
                raise HTTPException(status_code=400, detail=f"Unknown smoothing type: {smoothing_type}")

            fit_params = {}
            if alpha is not None: fit_params['smoothing_level']    = alpha
            if beta  is not None and smoothing_type != 'simple':
                                   fit_params['smoothing_trend']   = beta
            if gamma is not None and smoothing_type == 'holt-winters':
                                   fit_params['smoothing_seasonal'] = gamma

            fitted = model.fit(optimized=not bool(fit_params), **fit_params)

        # ── ② Forecast + prediction intervals ────────────────────────────────
        fc_point = fitted.forecast(h).values
        pi       = _prediction_intervals(fitted, h, level=ci_level, n_sim=n_sim)
        fc_dates = _forecast_dates(last_date, h)

        # ── ③ Diagnostics ─────────────────────────────────────────────────────
        diag = _diagnostics(fitted, series)

        # ── Model parameters ──────────────────────────────────────────────────
        model_params = {}
        for attr in ('smoothing_level', 'smoothing_trend', 'smoothing_seasonal'):
            v = getattr(fitted, attr, None)
            if v is not None:
                model_params[attr] = round(float(v), 6)

        aic  = safe_float(fitted.aic, 0)
        bic  = safe_float(fitted.bic, 0)
        aicc = safe_float(fitted.aicc, 0)

        # ── ④ Date-enriched result rows ────────────────────────────────────────
        fitted_values = fitted.fittedvalues
        result_data = []
        for i in range(n):
            result_data.append({
                'index':    i,
                'date':     str(dates.iloc[i])[:10],
                'original': safe_float(series.iloc[i]),
                'fitted':   safe_float(fitted_values.iloc[i]),
                'residual': safe_float(fitted.resid.iloc[i]) if i < len(fitted.resid) else None,
            })

        forecast_data = []
        for i in range(h):
            forecast_data.append({
                'step':         i + 1,
                'date':         fc_dates[i],
                'forecast':     safe_float(fc_point[i]),
                'lower':        safe_float(pi['lower'][i]),
                'upper':        safe_float(pi['upper'][i]),
                'ci_level':     ci_level,
            })

        # ── Plots ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            f'Exponential Smoothing — {smoothing_type.replace("-"," ").title()}'
            + (' (auto-selected)' if auto_selected else ''),
            fontsize=14, fontweight='bold'
        )

        date_labels = [str(d)[:10] for d in dates]
        x_in = np.arange(n)

        # Panel 1: Fitted + forecast
        ax1 = axes[0, 0]
        ax1.plot(x_in, series.values, 'b-', lw=1.5, alpha=0.7, label='Actual')
        ax1.plot(x_in, fitted_values.values, 'r-', lw=2, label='Fitted')
        x_fc = np.arange(n, n + h)
        ax1.plot(x_fc, fc_point, 'g--', lw=2, label=f'Forecast ({h} steps)')
        ax1.fill_between(x_fc, pi['lower'], pi['upper'],
                         alpha=0.20, color='green',
                         label=f'{int(ci_level*100)}% PI')
        # Date x-ticks (max 8)
        tick_step = max(1, n // 8)
        ax1.set_xticks(x_in[::tick_step])
        ax1.set_xticklabels(date_labels[::tick_step], rotation=30, ha='right', fontsize=7)
        ax1.set_title('Fitted Values & Forecast', fontweight='bold')
        ax1.set_ylabel(value_col)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        # Info box
        info = (f'α={model_params.get("smoothing_level","—")}\n'
                f'AICc={aicc:.1f}\n'
                f'RMSE={diag["rmse"]}\n'
                f'MAPE={diag["mape"]}%' if diag["mape"] else f'AICc={aicc:.1f}\nRMSE={diag["rmse"]}')
        ax1.text(0.02, 0.98, info, transform=ax1.transAxes, fontsize=8,
                 va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        # Panel 2: Residuals over time
        ax2 = axes[0, 1]
        resid_vals = fitted.resid.values
        ax2.plot(x_in, resid_vals, color='purple', lw=1, alpha=0.8)
        ax2.axhline(0, color='gray', lw=1, linestyle='--')
        ax2.fill_between(x_in, resid_vals, 0,
                         where=resid_vals > 0, alpha=0.2, color='green')
        ax2.fill_between(x_in, resid_vals, 0,
                         where=resid_vals < 0, alpha=0.2, color='red')
        ax2.set_xticks(x_in[::tick_step])
        ax2.set_xticklabels(date_labels[::tick_step], rotation=30, ha='right', fontsize=7)
        ax2.set_title(f'Residuals  (RMSE={diag["rmse"]}, ME={diag["me"]})',
                      fontweight='bold')
        ax2.set_ylabel('Residual')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Residual histogram + KDE
        ax3 = axes[1, 0]
        ax3.hist(resid_vals, bins=20, color='steelblue', alpha=0.7,
                 density=True, edgecolor='white', label='Residuals')
        from scipy.stats import norm as sp_norm
        mu, s = resid_vals.mean(), resid_vals.std()
        xs = np.linspace(resid_vals.min(), resid_vals.max(), 200)
        ax3.plot(xs, sp_norm.pdf(xs, mu, s), 'r-', lw=2, label='Normal fit')
        ax3.set_title('Residual Distribution', fontweight='bold')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Density')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Forecast fan chart (close-up of forecast window)
        ax4 = axes[1, 1]
        # Show last 30% of history + full forecast
        tail = max(10, n // 3)
        x_tail = x_in[-tail:]
        ax4.plot(x_tail, series.values[-tail:], 'b-', lw=1.5, alpha=0.8, label='Actual')
        ax4.plot(np.append(x_tail[-1], x_fc),
                 np.append(series.values[-1], fc_point),
                 'g--', lw=2, label='Forecast')
        ax4.fill_between(x_fc, pi['lower'], pi['upper'],
                         alpha=0.25, color='green', label=f'{int(ci_level*100)}% PI')
        ax4.set_xticks(list(x_tail[::max(1,tail//4)]) + list(x_fc[::max(1,h//4)]))
        tail_labels  = date_labels[-tail:][::max(1,tail//4)]
        fc_tl_labels = fc_dates[::max(1,h//4)]
        ax4.set_xticklabels(tail_labels + fc_tl_labels,
                             rotation=30, ha='right', fontsize=7)
        ax4.axvline(x=n - 1, color='gray', lw=1, linestyle=':', alpha=0.7)
        ax4.set_title(f'Forecast Fan Chart  ({h} steps ahead)', fontweight='bold')
        ax4.set_ylabel(value_col)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # ── Response ─────────────────────────────────────────────────────────
        return _to_native({
            'results': {
                'data':          result_data,
                'forecast':      forecast_data,
                'model_params':  model_params,
                'aic':           aic,
                'bic':           bic,
                'aicc':          aicc,
                'n_observations': n,
                'smoothing_type': smoothing_type,
                'auto_selected': auto_selected,
                'auto_selection': auto_info,
                'diagnostics':   diag,
            },
            'plot': f'data:image/png;base64,{plot_b64}'
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
