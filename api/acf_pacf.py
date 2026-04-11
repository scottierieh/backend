from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import io
import base64
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()


class AcfPacfRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]]
    valueCol: Optional[str] = None
    lags: int = 40


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


def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return float(val)
    except:
        return default


def _detect_acf_pattern(acf_vals, conf_int):
    """Classify ACF decay pattern."""
    sig = [i for i, v in enumerate(acf_vals) if i > 0 and abs(v) > conf_int]
    if not sig:
        return "no_sig", "No significant lags — white noise"
    # Check for cutoff: significant lags stop after lag q, then nothing
    if len(sig) <= 3:
        max_sig = max(sig)
        tail = [i for i in range(max_sig + 1, min(max_sig + 5, len(acf_vals))) if abs(acf_vals[i]) > conf_int]
        if not tail:
            return "cutoff", f"ACF cuts off after lag {max_sig} — MA({max_sig}) component"
    # Check sinusoidal / exponential decay
    vals = [acf_vals[i] for i in range(1, min(6, len(acf_vals)))]
    if all(abs(vals[i]) <= abs(vals[i-1]) for i in range(1, len(vals))):
        return "decay", "ACF tails off gradually — AR component present"
    return "mixed", "ACF shows mixed pattern — ARMA process likely"


def _detect_pacf_pattern(pacf_vals, conf_int):
    """Classify PACF decay pattern."""
    sig = [i for i, v in enumerate(pacf_vals) if i > 0 and abs(v) > conf_int]
    if not sig:
        return "no_sig", "No significant PACF lags — white noise"
    if len(sig) <= 3:
        max_sig = max(sig)
        tail = [i for i in range(max_sig + 1, min(max_sig + 5, len(pacf_vals))) if abs(pacf_vals[i]) > conf_int]
        if not tail:
            return "cutoff", f"PACF cuts off after lag {max_sig} — AR({max_sig}) component"
    vals = [pacf_vals[i] for i in range(1, min(6, len(pacf_vals)))]
    if all(abs(vals[i]) <= abs(vals[i-1]) for i in range(1, len(vals))):
        return "decay", "PACF tails off gradually — MA component present"
    return "mixed", "PACF shows mixed pattern — ARMA process likely"


def _detect_seasonality(acf_vals, conf_int, n):
    """Detect seasonal candidates from ACF spikes at regular intervals."""
    sig = [i for i, v in enumerate(acf_vals) if i > 0 and abs(v) > conf_int]
    candidates = []
    for period in [4, 6, 7, 12, 24, 52]:
        lags_at_period = [lag for lag in sig if lag % period == 0]
        if len(lags_at_period) >= 2:
            strength = np.mean([abs(acf_vals[lag]) for lag in lags_at_period])
            label_map = {4: 'Quarterly', 6: 'Bi-monthly', 7: 'Weekly',
                         12: 'Monthly (yearly)', 24: 'Monthly (2-yr)', 52: 'Weekly (yearly)'}
            candidates.append({
                'period': period,
                'label': label_map.get(period, f'Period {period}'),
                'significant_at': lags_at_period,
                'acf_strength': float(strength)
            })
    has_seasonality = len(candidates) > 0
    notable = sig[:8] if sig else []
    if has_seasonality:
        best = candidates[0]
        note = f"Seasonal pattern detected at period {best['period']} ({best['label']}). Consider SARIMA with s={best['period']}."
    else:
        note = "No clear seasonal pattern detected. Standard ARIMA should be sufficient."
    return {
        'has_seasonality': has_seasonality,
        'candidates': candidates,
        'notable_lags': notable,
        'note': note
    }


def _build_interpretation(ar_order, ma_order, acf_pattern, pacf_pattern,
                           acf_pattern_label, pacf_pattern_label, model_rec, interp_extra):
    """Build the full interpretation block."""
    if ar_order == 0 and ma_order == 0:
        process = "white_noise"
        process_label = "White Noise — no ARIMA modeling needed. Series appears random."
    elif ar_order > 0 and ma_order == 0:
        process = "ar"
        process_label = f"Pure AR({ar_order}) process — past values predict current values."
    elif ar_order == 0 and ma_order > 0:
        process = "ma"
        process_label = f"Pure MA({ma_order}) process — past forecast errors predict current values."
    else:
        process = "arma"
        process_label = f"Mixed ARMA({ar_order},{ma_order}) process — both AR and MA components needed."

    next_steps = []
    if ar_order == 0 and ma_order == 0:
        next_steps = [
            "Verify stationarity with ADF/KPSS test.",
            "If series is non-stationary, apply differencing and re-run ACF/PACF.",
            "No ARIMA model needed if series is truly white noise."
        ]
    else:
        next_steps = [
            f"Run stationarity test (ADF/KPSS) to determine d.",
            f"Fit ARIMA({ar_order}, d, {ma_order}) as a starting point.",
            f"Try ARIMA({max(0,ar_order-1)}, d, {ma_order}) and ARIMA({ar_order+1}, d, {ma_order}) and compare AIC/BIC.",
            "Check residual ACF/PACF — should show no significant lags after fitting."
        ]
        if interp_extra.get('has_seasonality'):
            next_steps.append("Seasonal pattern detected — also consider SARIMA.")

    return {
        'acf_pattern': acf_pattern,
        'acf_pattern_label': acf_pattern_label,
        'pacf_pattern': pacf_pattern,
        'pacf_pattern_label': pacf_pattern_label,
        'suggested_process': process,
        'process_label': process_label,
        'seasonality_hint': interp_extra.get('seasonality_note', ''),
        'white_noise_hint': 'Series appears to be white noise — no modeling needed.' if process == 'white_noise' else '',
        'next_steps': next_steps
    }


@router.post("/acf-pacf")
async def acf_pacf_analysis(request: AcfPacfRequest):
    try:
        data = request.data
        value_col = request.valueCol
        lags = request.lags

        # Handle both formats: array of numbers OR array of objects
        if len(data) > 0 and isinstance(data[0], (int, float)):
            series = pd.Series(data).dropna()
        else:
            df = pd.DataFrame(data)
            if value_col and value_col in df.columns:
                series = pd.to_numeric(df[value_col], errors='coerce').dropna()
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise HTTPException(status_code=400, detail="No numeric columns found")
                series = df[numeric_cols[0]].dropna()

        if len(series) < lags * 2:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data. Need at least {lags * 2} points, have {len(series)}."
            )

        # Limit lags to valid range
        max_lags = min(lags, len(series) // 2 - 1)
        if max_lags < 1:
            max_lags = 1

        # ── ACF / PACF ──────────────────────────────────────────────
        acf_values  = sm.tsa.acf(series, nlags=max_lags, fft=True)
        pacf_values = sm.tsa.pacf(series, nlags=max_lags, method='ywm')

        conf_int = 1.96 / np.sqrt(len(series))

        # Significant lags (new structure)
        sig_acf  = [i for i, v in enumerate(acf_values)  if i > 0 and abs(v) > conf_int]
        sig_pacf = [i for i, v in enumerate(pacf_values) if i > 0 and abs(v) > conf_int]

        # ── AR / MA order ────────────────────────────────────────────
        ar_order = 0
        for i in range(1, len(pacf_values)):
            if abs(pacf_values[i]) > conf_int:
                ar_order = i
            else:
                break

        ma_order = 0
        for i in range(1, len(acf_values)):
            if abs(acf_values[i]) > conf_int:
                ma_order = i
            else:
                break

        # Model recommendation
        if ar_order == 0 and ma_order == 0:
            model_rec = "White noise — no ARIMA modeling needed"
        elif ar_order > 0 and ma_order == 0:
            model_rec = f"Pure AR({ar_order}) process suggested"
        elif ar_order == 0 and ma_order > 0:
            model_rec = f"Pure MA({ma_order}) process suggested"
        else:
            model_rec = f"ARIMA({ar_order},d,{ma_order}) suggested — determine d via stationarity tests"

        # ── Pattern detection ────────────────────────────────────────
        acf_pattern,  acf_pattern_label  = _detect_acf_pattern(acf_values,  conf_int)
        pacf_pattern, pacf_pattern_label = _detect_pacf_pattern(pacf_values, conf_int)

        # ── Seasonality ──────────────────────────────────────────────
        seasonality = _detect_seasonality(acf_values, conf_int, len(series))

        # ── Ljung-Box ────────────────────────────────────────────────
        lb_lags = min(max_lags, 20)
        try:
            lb_result = acorr_ljungbox(series, lags=list(range(1, lb_lags + 1)), return_df=True)
            lb_rows = []
            for lag_val, row in lb_result.iterrows():
                lb_rows.append({
                    'lag': int(lag_val),
                    'statistic': safe_float(row['lb_stat']),
                    'p_value': safe_float(row['lb_pvalue']),
                    'is_white_noise': bool(row['lb_pvalue'] > 0.05)
                })
            overall_wn = all(r['is_white_noise'] for r in lb_rows) if lb_rows else None
            lb_note = (
                "Ljung-Box test: no significant autocorrelation detected — residuals behave as white noise."
                if overall_wn
                else f"Ljung-Box test: significant autocorrelation detected at one or more lags (p < .05)."
            )
        except Exception:
            lb_rows = []
            overall_wn = None
            lb_note = "Ljung-Box test could not be computed."

        diagnostics = {
            'ljung_box': {
                'results': lb_rows,
                'overall_white_noise': overall_wn,
                'note': lb_note
            },
            'decay': {
                'acf_pattern_label': acf_pattern_label,
                'pacf_pattern_label': pacf_pattern_label,
                'process_label': model_rec
            }
        }

        # ── Interpretation ───────────────────────────────────────────
        interpretation = _build_interpretation(
            ar_order, ma_order,
            acf_pattern, pacf_pattern,
            acf_pattern_label, pacf_pattern_label,
            model_rec,
            {'has_seasonality': seasonality['has_seasonality'],
             'seasonality_note': seasonality['note']}
        )

        # ── Recommended lags hint ────────────────────────────────────
        recommended_lags = min(40, max(10, int(np.floor(np.log(len(series)) * 10)),
                                       int(len(series) // 4)))

        # ── Plot ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('ACF & PACF Analysis', fontsize=14, fontweight='bold')

        sm.graphics.tsa.plot_acf(series, lags=max_lags, ax=axes[0], fft=True)
        axes[0].set_title(f'Autocorrelation Function (ACF) — MA order suggestion: q={ma_order}', fontsize=11)
        axes[0].axhline(y= conf_int, color='red', linestyle='--', alpha=0.5, label='95% CI')
        axes[0].axhline(y=-conf_int, color='red', linestyle='--', alpha=0.5)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        sm.graphics.tsa.plot_pacf(series, lags=max_lags, ax=axes[1], method='ywm')
        axes[1].set_title(f'Partial Autocorrelation Function (PACF) — AR order suggestion: p={ar_order}', fontsize=11)
        axes[1].axhline(y= conf_int, color='red', linestyle='--', alpha=0.5, label='95% CI')
        axes[1].axhline(y=-conf_int, color='red', linestyle='--', alpha=0.5)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # ── Final results ────────────────────────────────────────────
        results = {
            # Core arrays
            'acf':  [safe_float(v) for v in acf_values.tolist()],
            'pacf': [safe_float(v) for v in pacf_values.tolist()],
            'lags': max_lags,
            'recommended_lags': recommended_lags,
            'confidence_interval': safe_float(conf_int),
            'n_observations': len(series),

            # New structure (frontend primary)
            'significant_lags': {
                'acf':  sig_acf,
                'pacf': sig_pacf
            },
            'model_suggestion': {
                'ar_order': ar_order,
                'ma_order': ma_order,
                'model_recommendation': model_rec
            },
            'interpretation': interpretation,
            'seasonality': seasonality,
            'diagnostics': diagnostics,

            # Backward-compat flat fields
            'significant_acf_lags':  sig_acf,
            'significant_pacf_lags': sig_pacf,
            'ar_order_suggestion':   ar_order,
            'ma_order_suggestion':   ma_order,
            'model_recommendation':  model_rec,
        }

        return _to_native({
            'results': results,
            'plot': f"data:image/png;base64,{plot_b64}"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
