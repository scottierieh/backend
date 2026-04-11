from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
import io
import base64
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class LjungBoxRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]]
    valueCol: Optional[str] = None
    lags: Optional[int] = None          # None → auto: min(20, n//4)
    include_box_pierce: bool = True     # also compute Box-Pierce Q′ statistic
    input_type: str = "series"          # "series" | "residual"
    model_df: int = 0                   # degrees of freedom used by the fitted model
                                        # (p+q for ARIMA). Adjusts the χ² df when
                                        # input_type="residual". Ignored for raw series.


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


@router.post("/ljung-box")
async def ljung_box_test(request: LjungBoxRequest):
    try:
        data       = request.data
        value_col  = request.valueCol
        include_bp = request.include_box_pierce
        input_type = request.input_type.lower().strip()
        if input_type not in ("series", "residual"):
            raise HTTPException(status_code=400,
                detail="input_type must be 'series' or 'residual'")
        is_residual = input_type == "residual"
        model_df    = max(0, int(request.model_df)) if is_residual else 0

        # ── Parse series ──────────────────────────────────────────────────────
        if len(data) > 0 and isinstance(data[0], (int, float)):
            series = pd.Series(data).dropna().reset_index(drop=True)
        else:
            df = pd.DataFrame(data)
            if value_col and value_col in df.columns:
                series = pd.to_numeric(df[value_col], errors='coerce').dropna().reset_index(drop=True)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise HTTPException(status_code=400, detail="No numeric columns found")
                series = df[numeric_cols[0]].dropna().reset_index(drop=True)

        n = len(series)

        # ① Auto lag: min(20, n//4)  — Box & Jenkins (1970) recommend h << n;
        #   common practical rule: min(20, n//4) avoids overfitting for large n
        auto_lags = max(1, min(20, n // 4))
        lags      = request.lags if request.lags is not None else auto_lags
        lags_auto = request.lags is None

        if n <= lags:
            raise HTTPException(
                status_code=400,
                detail=f"Need more than {lags} observations. Have {n}."
            )

        max_plot_lags = min(lags, n - 1)
        lags_range    = list(range(1, max_plot_lags + 1))

        # ── Single call: Ljung-Box + Box-Pierce simultaneously ────────────────
        # boxpierce=True adds bp_stat / bp_pvalue columns at no extra cost.
        # Ref: Ljung & Box (1978) Biometrika 65(2):297-303;
        #      Box & Pierce (1970) JASA 65(332):1509-1526.
        # model_df shifts the effective χ² df: df_eff = lag - model_df
        # This is the correction for ARIMA residuals (Box & Jenkins 1970, §8.2).
        # statsmodels acorr_ljungbox accepts model_df directly.
        full_result = acorr_ljungbox(
            series, lags=lags_range, return_df=True,
            boxpierce=True, model_df=model_df
        )

        # ── Results at the chosen max lag ─────────────────────────────────────
        lb_stat   = safe_float(full_result['lb_stat'].iloc[-1])
        lb_pvalue = safe_float(full_result['lb_pvalue'].iloc[-1])
        bp_stat   = safe_float(full_result['bp_stat'].iloc[-1])
        bp_pvalue = safe_float(full_result['bp_pvalue'].iloc[-1])

        lb_significant = lb_pvalue < 0.05
        bp_significant = bp_pvalue < 0.05

        # Per-lag p-value series
        lb_pvals = [safe_float(p) for p in full_result['lb_pvalue'].tolist()]
        bp_pvals = [safe_float(p) for p in full_result['bp_pvalue'].tolist()]

        # First lag where LB is significant
        first_sig_lag = next(
            (int(full_result.index[i]) for i, p in enumerate(lb_pvals) if p < 0.05),
            None
        )

        # ── Interpretations ───────────────────────────────────────────────────
        def _interp(test_name, stat, pval, sig, h, residual_mode, mdf):
            df_note = f"  (χ² df = {h - mdf}, adjusted for {mdf} model parameter{'s' if mdf != 1 else ''})"                       if residual_mode and mdf > 0 else ""
            if residual_mode:
                verdict = ("Significant remaining autocorrelation — "
                           "model is under-fitted; consider increasing p or q."
                           if sig else
                           "No significant autocorrelation in residuals — "
                           "model appears well-specified (residuals ≈ white noise).")
            else:
                verdict = ("Significant autocorrelation present — "
                           "series is not white noise."
                           if sig else
                           "No significant autocorrelation — series is consistent with white noise.")
            return f"{test_name} Q({h}) = {stat:.4f},  p = {pval:.4f}.{df_note}  {verdict}"

        lb_interpretation = _interp("Ljung-Box",  lb_stat, lb_pvalue, lb_significant, lags, is_residual, model_df)
        bp_interpretation = _interp("Box-Pierce", bp_stat, bp_pvalue, bp_significant, lags, is_residual, model_df)

        # Agreement note
        if lb_significant == bp_significant:
            agreement = "Ljung-Box and Box-Pierce agree."
        else:
            agreement = (
                "Tests disagree — this is uncommon. "
                "Prefer Ljung-Box: it has better finite-sample size properties "
                "(Ljung & Box 1978 showed LB has a closer χ² approximation than BP for small n)."
            )

        # ── Plot ──────────────────────────────────────────────────────────────
        # 2-panel when include_bp=True, 1-panel otherwise
        nrows = 2 if include_bp else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(13, 5 * nrows), squeeze=False)
        lag_label   = f'lags 1–{lags}' + ('  (auto-selected)' if lags_auto else '')
        input_label = 'Residuals' if is_residual else 'Raw Series'
        df_label    = f'  |  model df={model_df}' if is_residual and model_df > 0 else ''
        fig.suptitle(
            f'Ljung-Box{"  +  Box-Pierce" if include_bp else ""} Test\n'
            f'({input_label}{df_label}  |  {lag_label})',
            fontsize=13, fontweight='bold'
        )

        def _panel(ax, lag_idx, pvals, title, sig_color, ok_color):
            bar_cols = [sig_color if p < 0.05 else ok_color for p in pvals]
            ax.bar(lag_idx, pvals, color=bar_cols, alpha=0.80, edgecolor='white')
            ax.axhline(y=0.05, color='#d62728', linestyle='--',
                       linewidth=1.8, label='α = 0.05')
            ax.set_xlabel('Lag', fontsize=11)
            ax.set_ylabel('p-value', fontsize=11)
            ymax = min(1.05, max(pvals) * 1.18 + 0.05) if pvals else 1.05
            ax.set_ylim(0, ymax)
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)
            n_sig = sum(1 for p in pvals if p < 0.05)
            face  = '#ffdddd' if n_sig > 0 else '#ddffdd'
            ax.annotate(
                f'{n_sig}/{len(pvals)} lags significant',
                xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=face, alpha=0.85)
            )

        _panel(
            axes[0, 0], lags_range, lb_pvals,
            f'Ljung-Box  |  Q({lags}) = {lb_stat:.3f},  p = {lb_pvalue:.4f}',
            '#d62728', '#1f77b4'
        )

        if include_bp:
            _panel(
                axes[1, 0], lags_range, bp_pvals,
                f"Box-Pierce  |  Q'({lags}) = {bp_stat:.3f},  p = {bp_pvalue:.4f}",
                '#e07b39', '#2ca02c'
            )
            fig.text(0.5, 0.005, agreement,
                     ha='center', va='bottom', fontsize=8.5, color='#444', style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # ── Response ──────────────────────────────────────────────────────────
        response = {
            'results': {
                'ljung_box': {
                    'statistic':           lb_stat,
                    'p_value':             lb_pvalue,
                    'is_significant':      lb_significant,
                    'interpretation':      lb_interpretation,
                    'p_values_by_lag':     lb_pvals,
                    'first_significant_lag': first_sig_lag,
                },
                'box_pierce': {
                    'statistic':       bp_stat,
                    'p_value':         bp_pvalue,
                    'is_significant':  bp_significant,
                    'interpretation':  bp_interpretation,
                    'p_values_by_lag': bp_pvals,
                } if include_bp else None,
                'agreement':          agreement if include_bp else None,
                # Meta
                'lags':               lags,
                'lags_auto_selected': lags_auto,
                'recommended_lags':   auto_lags,
                'n_observations':     n,
                'input_type':         input_type,
                'model_df':           model_df,
                # Legacy flat keys (backward compat)
                'lb_statistic':       lb_stat,
                'p_value':            lb_pvalue,
                'is_significant':     lb_significant,
                'interpretation':     lb_interpretation,
                'p_values_by_lag':    lb_pvals,
            },
            'plot': f"data:image/png;base64,{plot_b64}"
        }

        return _to_native(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
