from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import io
import base64
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class StationarityRequest(BaseModel):
    data: List[Dict[str, Any]]
    timeCol: str
    valueCol: str
    period: int = 12


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
# ① ADF + KPSS with explicit stationarity verdict
# ══════════════════════════════════════════════════════════════════

def _run_adf(series):
    """
    Augmented Dickey-Fuller test.
    H0: unit root (non-stationary). p < 0.05 → reject H0 → stationary.
    Ref: Dickey & Fuller (1979), JASA 74(366):427-431.
    """
    s = series.dropna()
    try:
        res = adfuller(s, autolag='AIC')
        return {
            'statistic':     safe_float(res[0]),
            'p_value':       safe_float(res[1]),
            'n_lags':        int(res[2]),
            'n_obs':         int(res[3]),
            'critical_1pct': safe_float(res[4]['1%']),
            'critical_5pct': safe_float(res[4]['5%']),
            'critical_10pct':safe_float(res[4]['10%']),
            'stationary':    bool(res[1] < 0.05),
        }
    except Exception as e:
        return {'error': str(e), 'stationary': False}


def _run_kpss(series):
    """
    KPSS test.
    H0: stationary. p < 0.05 → reject H0 → non-stationary.
    Ref: Kwiatkowski et al. (1992), J. Econometrics 54(1-3):159-178.
    """
    s = series.dropna()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = kpss(s, regression='c', nlags='auto')
        return {
            'statistic':     safe_float(res[0]),
            'p_value':       safe_float(res[1]),
            'n_lags':        int(res[2]),
            'critical_1pct': safe_float(res[3]['1%']),
            'critical_5pct': safe_float(res[3]['5%']),
            'critical_10pct':safe_float(res[3]['10%']),
            'stationary':    bool(res[1] >= 0.05),
        }
    except Exception as e:
        return {'error': str(e), 'stationary': True}


def _decide_stationarity(adf: dict, kpss_r: dict) -> dict:
    """
    Combine ADF and KPSS to produce a definitive verdict.

    Four cases (Hobijn et al. 2004 / Hyndman & Athanasopoulos 2021):
    ┌────────────────┬──────────────────┬──────────────────────────────────────┐
    │   ADF          │   KPSS           │   Verdict                            │
    ├────────────────┼──────────────────┼──────────────────────────────────────┤
    │ stationary     │ stationary       │ Stationary (both agree)              │
    │ non-stationary │ non-stationary   │ Non-stationary (both agree) → d++    │
    │ stationary     │ non-stationary   │ Trend-stationary → detrend or d++    │
    │ non-stationary │ stationary       │ Difference-stationary → d=0 likely   │
    └────────────────┴──────────────────┴──────────────────────────────────────┘
    """
    adf_stat  = adf.get('stationary',   False)
    kpss_stat = kpss_r.get('stationary', True)
    adf_p     = adf.get('p_value',  1.0)
    kpss_p    = kpss_r.get('p_value', 1.0)

    if adf_stat and kpss_stat:
        verdict = 'Stationary'
        needs_diff = False
        explanation = (f'Both tests agree: stationary. '
                       f'ADF p={adf_p:.4f} (reject unit root), '
                       f'KPSS p={kpss_p:.4f} (fail to reject stationarity).')
    elif not adf_stat and not kpss_stat:
        verdict = 'Non-stationary'
        needs_diff = True
        explanation = (f'Both tests agree: non-stationary. '
                       f'ADF p={adf_p:.4f} (fail to reject unit root), '
                       f'KPSS p={kpss_p:.4f} (reject stationarity). '
                       f'Differencing recommended.')
    elif adf_stat and not kpss_stat:
        verdict = 'Trend-stationary'
        needs_diff = True
        explanation = (f'ADF p={adf_p:.4f} passes (no unit root) but '
                       f'KPSS p={kpss_p:.4f} fails (deterministic trend present). '
                       f'Detrend or apply d=1.')
    else:  # not adf_stat and kpss_stat
        verdict = 'Difference-stationary'
        needs_diff = False
        explanation = (f'KPSS p={kpss_p:.4f} passes but ADF p={adf_p:.4f} fails. '
                       f'Series may be borderline; inspect ACF and visual plot.')

    return {
        'verdict':     verdict,
        'is_stationary': verdict == 'Stationary',
        'needs_differencing': needs_diff,
        'explanation': explanation,
    }


def run_stationarity_tests(series) -> dict:
    """ADF + KPSS + combined verdict."""
    adf    = _run_adf(series)
    kpss_r = _run_kpss(series)
    decision = _decide_stationarity(adf, kpss_r)
    return {
        'adf':      adf,
        'kpss':     kpss_r,
        'decision': decision,
        # legacy flat fields (backward compat)
        'adf_statistic': adf.get('statistic', 0.0),
        'adf_p_value':   adf.get('p_value',   1.0),
        'kpss_statistic':kpss_r.get('statistic', 0.0),
        'kpss_p_value':  kpss_r.get('p_value',   1.0),
    }


# ══════════════════════════════════════════════════════════════════
# ② Auto differencing recommendation
# ══════════════════════════════════════════════════════════════════

def _auto_diff_recommendation(series: pd.Series, period: int) -> dict:
    """
    Sequentially test d=0,1,2 and D=0,1 to find the minimum differencing
    needed to achieve stationarity.

    Strategy:
      1. Test original (d=0).  If stationary → done, d=0.
      2. Test d=1 (first difference).  If stationary → d=1.
      3. Test d=2 (second difference). If stationary → d=2.
         (d>2 is unusual; warn if still non-stationary.)
      4. Independently test D=1 seasonal difference.

    Verdict uses the joint ADF+KPSS decision from _decide_stationarity().
    Ref: Box & Jenkins (1970); Hyndman & Athanasopoulos (2021) §9.1.
    """
    results = {}

    def _test(s, label):
        if len(s.dropna()) < 4:
            return None
        adf = _run_adf(s); kpss_r = _run_kpss(s)
        dec = _decide_stationarity(adf, kpss_r)
        return {'label': label, 'n': int(len(s.dropna())),
                'is_stationary': dec['is_stationary'],
                'verdict': dec['verdict'],
                'adf_p': safe_float(adf.get('p_value', 1.0)),
                'kpss_p': safe_float(kpss_r.get('p_value', 0.0))}

    d0 = _test(series,                   'd=0 (original)')
    d1 = _test(series.diff().dropna(),   'd=1 (first diff)')
    d2 = _test(series.diff().diff().dropna(), 'd=2 (second diff)')

    # Seasonal
    D0_series = series.diff(period).dropna() if len(series) > period + 4 else None
    D1 = _test(D0_series, f'D=1 (seasonal diff, s={period})') if D0_series is not None else None

    # Determine recommended d
    recommended_d = 0
    if d0 and d0['is_stationary']:
        recommended_d = 0
    elif d1 and d1['is_stationary']:
        recommended_d = 1
    elif d2 and d2['is_stationary']:
        recommended_d = 2
    else:
        recommended_d = 1   # default to 1 if unclear

    # Determine recommended D
    recommended_D = 1 if (D1 and D1['is_stationary'] and
                          d0 and not d0['is_stationary']) else 0

    # Recommendation string
    parts = [f'd={recommended_d}']
    if recommended_D > 0:
        parts.append(f'D={recommended_D} (s={period})')
    recommendation = ', '.join(parts)

    return {
        'recommended_d':   recommended_d,
        'recommended_D':   recommended_D,
        'recommended_s':   period if recommended_D > 0 else None,
        'recommendation':  recommendation,
        'test_sequence':   [r for r in [d0, d1, d2, D1] if r is not None],
        'arima_hint':      (f'ARIMA({recommended_d},d,q) with d={recommended_d}'
                            + (f', SARIMA seasonal D={recommended_D}' if recommended_D else '')),
    }


# ══════════════════════════════════════════════════════════════════
# ③ Zivot-Andrews structural break test
# ══════════════════════════════════════════════════════════════════

def _run_zivot_andrews(series: pd.Series, dates=None) -> dict:
    """
    Zivot-Andrews test for unit root with a single unknown structural break.

    Extends ADF to allow one break in level or trend at an unknown date.
    H0: unit root with structural break.
    Reject H0 (p<0.05) → stationary despite apparent break.

    Three regression modes tested:
      'c'  — break in intercept only
      't'  — break in trend only
      'ct' — break in both (most common for economic data)

    The break date is the point minimising the ADF t-statistic
    (most evidence for stationarity conditional on that break).

    Ref: Zivot & Andrews (1992), JBES 10(3):251-270.
    """
    s = series.dropna()
    n = len(s)
    if n < 20:
        return {'error': 'Need ≥20 observations for Zivot-Andrews test'}

    out = {}
    for reg in ('c', 't', 'ct'):
        try:
            # statsmodels ZA returns: (stat, p_value, critvals_dict, n_lags, break_index)
            za_res    = zivot_andrews(s.values, trim=0.15, maxlag=None,
                                      regression=reg, autolag='AIC')
            za_stat   = float(za_res[0])
            za_p      = float(za_res[1])
            za_cv     = za_res[2]          # dict {'1%', '5%', '10%'}
            bp_idx    = int(za_res[4])     # break index
            bp_label  = (dates.iloc[bp_idx].isoformat()
                         if dates is not None and bp_idx < len(dates)
                         else str(bp_idx))
            out[reg] = {
                'statistic':         safe_float(za_stat),
                'p_value':           safe_float(za_p),
                'break_index':       bp_idx,
                'break_date':        bp_label,
                'critical_1pct':     safe_float(za_cv.get('1%', None)),
                'critical_5pct':     safe_float(za_cv.get('5%', None)),
                'critical_10pct':    safe_float(za_cv.get('10%', None)),
                'rejects_unit_root': bool(za_p < 0.05),
                'conclusion': (
                    f'Reject unit root at break (p={za_p:.4f}) — stationary with structural break at {bp_label}'
                    if za_p < 0.05
                    else f'Fail to reject unit root (p={za_p:.4f}) — non-stationary even with break at {bp_label}'
                ),
            }
        except Exception as e:
            out[reg] = {'error': str(e)}

    # Overall verdict: use 'ct' if available, else 'c'
    primary = out.get('ct') or out.get('c') or {}
    if not primary.get('error'):
        overall_rejects = primary.get('rejects_unit_root', False)
        break_date      = primary.get('break_date', 'unknown')
        verdict = (
            f'Structural break detected at {break_date} — '
            + ('series is stationary conditional on break' if overall_rejects
               else 'unit root persists even with break — differencing needed')
        )
    else:
        verdict = 'Zivot-Andrews test could not be completed'

    out['summary'] = {
        'primary_regression': 'ct',
        'break_date':   primary.get('break_date'),
        'rejects_unit_root': primary.get('rejects_unit_root', False),
        'verdict': verdict,
        'interpretation': (
            'If ADF says non-stationary but Zivot-Andrews rejects unit root, '
            'the series may be stationary with a structural break — '
            'detrend at the break point rather than differencing.'
        ),
    }
    return out


def create_plot(series, title, color='#1f77b4'):
    """Creates a plot for a given series with consistent styling."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(series.index, series.values, color=color, linewidth=1.5, alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(True, alpha=0.6)

    # Add mean line
    mean_val = series.mean()
    ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


# ══════════════════════════════════════════════════════════════════
# ARCH-LM test — variance stationarity
# ══════════════════════════════════════════════════════════════════

def _run_arch_lm(series: pd.Series, max_lag: int = None) -> dict:
    """
    Engle's ARCH-LM test for conditional heteroscedasticity (variance clustering).

    Tests whether the squared residuals are autocorrelated — i.e. whether
    variance changes systematically over time (ARCH / GARCH effects).

    H0: No ARCH effects (variance is constant).
    Reject H0 (p < 0.05) → variance is non-stationary → GARCH modeling needed.

    Procedure:
      1. Demean the series (remove mean non-stationarity).
      2. Run OLS of squared residuals on lagged squared residuals (lags=1..nlags).
      3. LM statistic = n * R² ~ χ²(nlags) under H0.

    nlags auto-selected as min(10, n//5) — Engle (1982) recommends 4-8 lags
    for monthly data; we scale with series length.

    Multi-lag profile: test each lag 1..max_lag individually to show
    at which lag the ARCH effect first appears.

    Reference: Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity
               with Estimates of the Variance of United Kingdom Inflation."
               Econometrica 50(4): 987-1007.
    """
    s = series.dropna()
    n = len(s)
    if n < 10:
        return {'error': 'Need ≥10 observations for ARCH-LM test'}

    if max_lag is None:
        max_lag = min(10, n // 5)
    max_lag = max(1, max_lag)

    # ── Primary test at max_lag ───────────────────────────────────────────────
    try:
        lm_stat, lm_p, f_stat, f_p = het_arch(s.values, nlags=max_lag)
        has_arch = bool(lm_p < 0.05)
        verdict  = ('ARCH effects present' if has_arch
                    else 'No ARCH effects — variance is stationary')
        conclusion = (
            f'Reject H0 (p={lm_p:.4f}): variance is non-stationary. '
            f'Conditional heteroscedasticity detected — consider GARCH/EGARCH modeling.'
            if has_arch else
            f'Fail to reject H0 (p={lm_p:.4f}): variance appears stationary. '
            f'No ARCH effects at lag {max_lag}.'
        )
    except Exception as e:
        return {'error': str(e)}

    # ── Lag profile: test each lag individually ───────────────────────────────
    lag_profile = []
    for lag in range(1, max_lag + 1):
        try:
            lm_l, p_l, _, _ = het_arch(s.values, nlags=lag)
            lag_profile.append({
                'lag':       lag,
                'lm_stat':   safe_float(lm_l),
                'p_value':   safe_float(p_l),
                'significant': bool(p_l < 0.05),
            })
        except Exception:
            break

    # First lag where ARCH appears
    first_arch_lag = next((r['lag'] for r in lag_profile if r['significant']), None)

    # Squared-residual autocorrelation (visual evidence)
    sq_resid = (s - s.mean()) ** 2
    sq_acf_vals = None
    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        sq_acf_vals = [safe_float(v) for v in sm_acf(sq_resid, nlags=max_lag, fft=True)]
    except Exception:
        pass

    return {
        'lm_statistic':    safe_float(lm_stat),
        'lm_p_value':      safe_float(lm_p),
        'f_statistic':     safe_float(f_stat),
        'f_p_value':       safe_float(f_p),
        'n_lags':          max_lag,
        'has_arch_effects':has_arch,
        'verdict':         verdict,
        'conclusion':      conclusion,
        'first_arch_lag':  first_arch_lag,
        'lag_profile':     lag_profile,
        'sq_resid_acf':    sq_acf_vals,
        'recommendation':  (
            'Fit GARCH(1,1) or EGARCH to model volatility clustering.'
            if has_arch else
            'OLS / ARIMA residuals are homoscedastic — no variance model needed.'
        ),
    }


def _create_arch_plot(series: pd.Series, arch_result: dict) -> str:
    """
    4-panel ARCH-LM diagnostic plot:
      Top-left  : squared residuals over time (variance clustering visible)
      Top-right : ACF of squared residuals (ARCH signature = significant lags)
      Bottom-left: ARCH-LM p-value by lag (lag profile)
      Bottom-right: rolling variance (window=10% n) to show heteroscedasticity
    """
    s       = series.dropna()
    n       = len(s)
    sq_res  = (s - s.mean()) ** 2
    ci      = 1.96 / np.sqrt(n)
    max_lag = arch_result.get('n_lags', 10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('ARCH-LM Variance Stationarity Diagnostics', fontsize=13, fontweight='bold')

    # ── Panel 1: squared residuals ────────────────────────────────────────────
    axes[0, 0].plot(np.arange(n), sq_res.values, '#C44E52', linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Squared Residuals (ε²)')
    axes[0, 0].set_xlabel('Index'); axes[0, 0].grid(True, alpha=0.3)
    # Annotate verdict
    color_v = '#e74c3c' if arch_result.get('has_arch_effects') else '#2ecc71'
    axes[0, 0].text(0.97, 0.95, arch_result.get('verdict', ''),
                    transform=axes[0, 0].transAxes, ha='right', va='top',
                    fontsize=9, color=color_v,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── Panel 2: ACF of squared residuals ────────────────────────────────────
    acf_v = arch_result.get('sq_resid_acf', [])
    if acf_v:
        lags_x = np.arange(len(acf_v))
        bar_colors = ['#e74c3c' if abs(v) > ci and i > 0 else '#4C72B0'
                      for i, v in enumerate(acf_v)]
        axes[0, 1].bar(lags_x, acf_v, color=bar_colors, alpha=0.8)
        axes[0, 1].axhline( ci, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95% CI')
        axes[0, 1].axhline(-ci, color='red', linestyle='--', linewidth=1, alpha=0.7)
        axes[0, 1].set_title('ACF of Squared Residuals\n(red bars = significant ARCH)')
        axes[0, 1].set_xlabel('Lag'); axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

    # ── Panel 3: ARCH-LM p-value by lag ──────────────────────────────────────
    lag_prof = arch_result.get('lag_profile', [])
    if lag_prof:
        lags_p  = [r['lag']     for r in lag_prof]
        pvals   = [r['p_value'] for r in lag_prof]
        bar_c   = ['#e74c3c' if p < 0.05 else '#2ecc71' for p in pvals]
        axes[1, 0].bar(lags_p, pvals, color=bar_c, alpha=0.8, edgecolor='white')
        axes[1, 0].axhline(0.05, color='navy', linestyle='--', linewidth=1.5, label='α=0.05')
        for lag, p in zip(lags_p, pvals):
            axes[1, 0].text(lag, min(p + 0.02, 0.96), f'{p:.3f}',
                            ha='center', va='bottom', fontsize=8)
        axes[1, 0].set_title('ARCH-LM p-value by Lag\n(red = ARCH detected)')
        axes[1, 0].set_xlabel('Lag'); axes[1, 0].set_ylabel('p-value')
        axes[1, 0].set_ylim(0, 1.05); axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        from matplotlib.patches import Patch
        axes[1, 0].legend(handles=[
            Patch(color='#e74c3c', label='ARCH detected'),
            Patch(color='#2ecc71', label='No ARCH'),
            plt.Line2D([0], [0], color='navy', linestyle='--', label='α=0.05'),
        ], fontsize=8)

    # ── Panel 4: rolling variance ─────────────────────────────────────────────
    w = max(4, n // 10)
    roll_var = sq_res.rolling(w, center=True, min_periods=w // 2).mean()
    axes[1, 1].fill_between(np.arange(n), 0, roll_var.values,
                             color='#DD8452', alpha=0.5, label=f'Rolling Var (w={w})')
    axes[1, 1].plot(np.arange(n), roll_var.values, '#DD8452', linewidth=1.5)
    axes[1, 1].axhline(float(sq_res.mean()), color='gray', linestyle='--',
                        linewidth=1, label='Mean variance')
    axes[1, 1].set_title('Rolling Variance\n(spikes = volatility clustering)')
    axes[1, 1].set_xlabel('Index'); axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ══════════════════════════════════════════════════════════════════
# Ljung-Box + Box-Pierce serial correlation test
# ══════════════════════════════════════════════════════════════════

def _run_ljung_box(series: pd.Series, model_df: int = 0) -> dict:
    """
    Ljung-Box Q test (and Box-Pierce Q') for serial autocorrelation.

    Complements ADF/KPSS (unit root) and ARCH-LM (variance clustering)
    by testing whether the series has any autocorrelation structure:
      • Is the series white noise?
      • Does it have AR/MA structure worth modelling?

    Lag auto-selection: min(20, n//4)  — standard recommendation
    (Box & Jenkins 1970; Ljung & Box 1978).

    Returns per-lag LB and BP p-values, first significant lag, and a
    plain-language interpretation.

    Refs:
      Ljung, G.M. & Box, G.E.P. (1978). Biometrika 65(2): 297-303.
      Box, G.E.P. & Pierce, D.A. (1970). JASA 65: 1509-1526.
    """
    s = series.dropna()
    n = len(s)
    if n < 8:
        return {'error': f'Need at least 8 observations for Ljung-Box, got {n}'}

    auto_lags = max(1, min(20, n // 4))

    try:
        full = acorr_ljungbox(
            s, lags=list(range(1, auto_lags + 1)),
            return_df=True, boxpierce=True, model_df=model_df
        )
    except Exception as e:
        return {'error': str(e)}

    lb_stat   = safe_float(full['lb_stat'].iloc[-1])
    lb_pvalue = safe_float(full['lb_pvalue'].iloc[-1])
    bp_stat   = safe_float(full['bp_stat'].iloc[-1])
    bp_pvalue = safe_float(full['bp_pvalue'].iloc[-1])

    lb_sig = lb_pvalue < 0.05
    bp_sig = bp_pvalue < 0.05

    lb_pvals = [safe_float(p) for p in full['lb_pvalue'].tolist()]
    bp_pvals = [safe_float(p) for p in full['bp_pvalue'].tolist()]

    first_sig = next(
        (int(full.index[i]) for i, p in enumerate(lb_pvals) if p < 0.05), None
    )

    if lb_sig:
        verdict = (f'Significant autocorrelation at lag {first_sig} — '
                   f'series has AR/MA structure; not white noise.')
    else:
        verdict = (f'No significant autocorrelation up to lag {auto_lags} — '
                   f'series is consistent with white noise.')

    agreement = ('LB and BP agree.' if lb_sig == bp_sig else
                 'LB and BP disagree — prefer Ljung-Box (better χ² approx for small n).')

    return {
        'ljung_box': {
            'statistic':             lb_stat,
            'p_value':               lb_pvalue,
            'is_significant':        lb_sig,
            'first_significant_lag': first_sig,
            'p_values_by_lag':       lb_pvals,
            'interpretation':        f'Ljung-Box Q({auto_lags}) = {lb_stat:.4f}, p = {lb_pvalue:.4f}. {verdict}',
        },
        'box_pierce': {
            'statistic':       bp_stat,
            'p_value':         bp_pvalue,
            'is_significant':  bp_sig,
            'p_values_by_lag': bp_pvals,
            'interpretation':  f"Box-Pierce Q'({auto_lags}) = {bp_stat:.4f}, p = {bp_pvalue:.4f}.",
        },
        'agreement':          agreement,
        'lags':               auto_lags,
        'n_observations':     n,
        'model_df':           model_df,
        'has_autocorrelation': lb_sig,
        'verdict':            verdict,
    }


def _create_lb_plot(series: pd.Series, lb_result: dict) -> str:
    """
    2-panel Ljung-Box / Box-Pierce plot:
      Left : LB p-values by lag  (red bars = significant)
      Right: BP p-values by lag  (orange bars = significant)
    Dashed reference line at α = 0.05.
    """
    if lb_result.get('error'):
        return None

    lags      = lb_result['lags']
    lag_range = list(range(1, lags + 1))
    lb_pvals  = lb_result['ljung_box']['p_values_by_lag']
    bp_pvals  = lb_result['box_pierce']['p_values_by_lag']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Serial Autocorrelation Tests  (lags 1–{lags},  n={lb_result["n_observations"]})',
        fontsize=12, fontweight='bold'
    )

    def _bar_panel(ax, pvals, title, sig_col, ok_col):
        colors = [sig_col if p < 0.05 else ok_col for p in pvals]
        ax.bar(lag_range, pvals, color=colors, alpha=0.82, edgecolor='white')
        ax.axhline(0.05, color='#d62728', linestyle='--', linewidth=1.8, label='α = 0.05')
        ax.set_xlabel('Lag', fontsize=10)
        ax.set_ylabel('p-value', fontsize=10)
        ax.set_ylim(0, min(1.08, max(pvals) * 1.2 + 0.05) if pvals else 1.08)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35)
        n_sig = sum(1 for p in pvals if p < 0.05)
        face  = '#ffdddd' if n_sig > 0 else '#ddffdd'
        ax.annotate(f'{n_sig}/{len(pvals)} lags significant',
                    xy=(0.97, 0.97), xycoords='axes fraction',
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=face, alpha=0.85))

    lb_s = lb_result['ljung_box']
    bp_s = lb_result['box_pierce']
    _bar_panel(ax1, lb_pvals,
               f"Ljung-Box  Q({lags}) = {lb_s['statistic']:.2f},  p = {lb_s['p_value']:.4f}",
               '#d62728', '#1f77b4')
    _bar_panel(ax2, bp_pvals,
               f"Box-Pierce  Q'({lags}) = {bp_s['statistic']:.2f},  p = {bp_s['p_value']:.4f}",
               '#e07b39', '#2ca02c')

    fig.text(0.5, 0.01, lb_result['agreement'],
             ha='center', va='bottom', fontsize=8, color='#555', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def _create_za_plot(series: pd.Series, za_result: dict, dates=None) -> str:
    """
    2-panel Zivot-Andrews plot:
      Left : time series with vertical line at detected break date
      Right: rolling mean ± std to visualise structural shift
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x = np.arange(len(series))

    # Panel 1: series + break line
    axes[0].plot(x, series.values, '#1f77b4', linewidth=1.5, alpha=0.85)
    ct = za_result.get('ct') or za_result.get('c') or {}
    bp = ct.get('break_index')
    if bp is not None and 0 < bp < len(series):
        axes[0].axvline(bp, color='red', linestyle='--', linewidth=2,
                        label=f"Break: {ct.get('break_date', bp)}")
        axes[0].legend(fontsize=9)
    axes[0].set_title('Zivot-Andrews Structural Break', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Index'); axes[0].grid(True, alpha=0.3)

    # Panel 2: rolling statistics (window = 10% of n)
    w = max(4, len(series) // 10)
    roll_mean = series.rolling(w, center=True).mean()
    roll_std  = series.rolling(w, center=True).std()
    axes[1].plot(x, series.values,    '#1f77b4', linewidth=1,   alpha=0.5, label='Original')
    axes[1].plot(x, roll_mean.values, '#ff7f0e', linewidth=2,   label=f'Rolling Mean (w={w})')
    axes[1].fill_between(x,
        (roll_mean - roll_std).values,
        (roll_mean + roll_std).values,
        alpha=0.2, color='orange', label='±1 std')
    if bp is not None and 0 < bp < len(series):
        axes[1].axvline(bp, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Rolling Mean ± Std', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Index'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def _create_summary_plot(seq_results: list) -> str:
    """
    Bar chart of ADF p-values across d=0,1,2 and seasonal diff
    with 5% significance line — visually shows which order achieves stationarity.
    """
    if not seq_results:
        return None
    labels = [r['label'] for r in seq_results]
    adf_ps = [r['adf_p']  for r in seq_results]
    colors = ['#2ecc71' if r['is_stationary'] else '#e74c3c' for r in seq_results]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, adf_ps, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(0.05, color='navy', linestyle='--', linewidth=1.5, label='α=0.05 threshold')
    for bar, p in zip(bars, adf_ps):
        ax.text(bar.get_x() + bar.get_width()/2, min(p + 0.01, 0.97),
                f'{p:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('ADF p-value'); ax.set_ylim(0, 1.05)
    ax.set_title('ADF p-value by Differencing Order (green = stationary)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#2ecc71', label='Stationary'),
        Patch(color='#e74c3c', label='Non-stationary'),
        plt.Line2D([0],[0], color='navy', linestyle='--', label='α=0.05'),
    ], fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


@router.post("/stationarity")
async def stationarity_analysis(request: StationarityRequest):
    try:
        df = pd.DataFrame(request.data)
        time_col  = request.timeCol
        value_col = request.valueCol
        period    = request.period

        if time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{time_col}' not found")
        if value_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{value_col}' not found")

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[time_col, value_col]).set_index(time_col).sort_index()

        series = df[value_col]
        dates  = series.index.to_series()

        if len(series) < 4:
            raise HTTPException(status_code=400, detail="Series must have at least 4 observations")

        # ── ① ADF + KPSS + verdict on original, d=1, seasonal diff ──────────
        series_diff1         = series.diff().dropna()
        series_seasonal_diff = series.diff(periods=period).dropna()

        original_results      = run_stationarity_tests(series)
        diff1_results         = run_stationarity_tests(series_diff1) if len(series_diff1) > 3 else None
        seasonal_diff_results = run_stationarity_tests(series_seasonal_diff) if len(series_seasonal_diff) > 3 else None

        # ── ② Auto differencing recommendation ──────────────────────────────
        auto_diff = _auto_diff_recommendation(series, period)

        # ── ③ Zivot-Andrews structural break test ────────────────────────────
        za_result = _run_zivot_andrews(series, dates=dates)

        # ── ④ ARCH-LM variance stationarity test ─────────────────────────────
        arch_result = _run_arch_lm(series)

        # ── ⑤ Ljung-Box serial autocorrelation test ───────────────────────────
        lb_result = _run_ljung_box(series)

        # ── Plots ─────────────────────────────────────────────────────────────
        original_plot      = create_plot(series,        "Original Time Series",                  color='#1f77b4')
        diff1_plot         = create_plot(series_diff1,  "First-Differenced Series (d=1)",         color='#ff7f0e') if diff1_results else None
        seasonal_diff_plot = create_plot(series_seasonal_diff,
                                         f"Seasonally-Differenced (D=1, s={period})",             color='#2ca02c') if seasonal_diff_results else None
        za_plot            = _create_za_plot(series, za_result, dates=dates) if not za_result.get('error') else None
        arch_plot          = _create_arch_plot(series, arch_result) if not arch_result.get('error') else None
        lb_plot            = _create_lb_plot(series, lb_result) if not lb_result.get('error') else None
        summary_plot       = _create_summary_plot(auto_diff.get('test_sequence', []))

        # ── Overall recommendation summary ───────────────────────────────────
        orig_verdict  = original_results['decision']['verdict']
        is_stationary = original_results['decision']['is_stationary']
        za_breaks_ur  = za_result.get('summary', {}).get('rejects_unit_root', False)

        if is_stationary:
            overall = 'Mean: stationary. No differencing needed.'
        elif za_breaks_ur:
            break_date = za_result.get('summary', {}).get('break_date', 'unknown')
            overall = (f'Mean: structural break at {break_date}. '
                       f'Consider detrending at the break rather than differencing. '
                       f'Recommended: {auto_diff["recommendation"]}.')
        else:
            overall = (f'Mean: non-stationary ({orig_verdict}). '
                       f'Recommended differencing: {auto_diff["recommendation"]}.')

        # Append variance verdict
        if not arch_result.get('error'):
            var_verdict = ('Variance: non-stationary (ARCH effects detected — GARCH modeling recommended).'
                           if arch_result.get('has_arch_effects')
                           else 'Variance: stationary (homoscedastic).')
            overall += f' | {var_verdict}'

        # Append serial correlation verdict
        if not lb_result.get('error'):
            lb_verdict = ('Serial correlation: present (AR/MA structure detected — consider ARIMA).'
                          if lb_result.get('has_autocorrelation')
                          else 'Serial correlation: none (consistent with white noise).')
            overall += f' | {lb_verdict}'

        response = {
            # ① Per-series test results with verdicts
            'original': {
                'test_results': original_results,
                'plot':         original_plot,
            },
            'first_difference': {
                'test_results': diff1_results,
                'plot':         diff1_plot,
            } if diff1_results else None,
            'seasonal_difference': {
                'test_results': seasonal_diff_results,
                'plot':         seasonal_diff_plot,
            } if seasonal_diff_results else None,

            # ② Auto differencing recommendation
            'differencing_recommendation': auto_diff,

            # ③ Zivot-Andrews
            'zivot_andrews': za_result,
            'zivot_andrews_plot': za_plot,

            # ④ ARCH-LM variance stationarity
            'arch_lm': arch_result,
            'arch_lm_plot': arch_plot,

            # ⑤ Ljung-Box + Box-Pierce serial autocorrelation
            'ljung_box': lb_result,
            'ljung_box_plot': lb_plot,

            # Summary plot (ADF p by diff order)
            'summary_plot': summary_plot,

            # Top-level verdict
            'overall_recommendation': overall,
            'is_stationary': is_stationary,
            'recommended_d': auto_diff['recommended_d'],
            'recommended_D': auto_diff['recommended_D'],
            'variance_stationary': not arch_result.get('has_arch_effects', False),
            'has_autocorrelation': lb_result.get('has_autocorrelation', None),
        }

        return _to_native(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
