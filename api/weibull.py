from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P
from reliability.Probability_plotting import Weibull_probability_plot
from reliability.Other_functions import make_right_censored_data
import io
import base64
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()


# ── Request / Response Models ──

class WeibullRequest(BaseModel):
    data: Union[List[float], List[Dict[str, Any]]]
    valueCol: Optional[str] = None
    censorCol: Optional[str] = None       # 1=failure, 0=censored (default: all failures)
    model: str = "2P"                      # "2P" or "3P"
    confidenceLevel: float = 0.95
    targetTimes: Optional[List[float]] = None  # R(t) at these times


# ── Helpers ──

def _to_native(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def failure_mode_label(beta: float) -> str:
    if beta < 0.95:
        return "Infant mortality (decreasing failure rate)"
    elif beta <= 1.05:
        return "Random / exponential (constant failure rate)"
    elif beta <= 2.5:
        return "Early wear-out (increasing failure rate)"
    else:
        return "Rapid wear-out (strongly increasing failure rate)"


# ── Main Endpoint ──

@router.post("/weibull")
async def weibull_analysis(request: WeibullRequest):
    try:
        data = request.data
        value_col = request.valueCol
        censor_col = request.censorCol
        model_type = request.model.upper()
        conf = request.confidenceLevel
        target_times = request.targetTimes or []

        # ── 1. Parse Input ──
        if len(data) > 0 and isinstance(data[0], (int, float)):
            times = np.array([float(x) for x in data if x is not None and float(x) > 0])
            censored_flag = np.ones(len(times))
        else:
            df = pd.DataFrame(data)
            if value_col and value_col in df.columns:
                ts = pd.to_numeric(df[value_col], errors='coerce')
            else:
                ncols = df.select_dtypes(include=[np.number]).columns
                if len(ncols) == 0:
                    raise HTTPException(status_code=400, detail="No numeric columns found")
                ts = df[ncols[0]]

            if censor_col and censor_col in df.columns:
                cf = pd.to_numeric(df[censor_col], errors='coerce').fillna(1)
            else:
                cf = pd.Series(np.ones(len(df)))

            mask = ts.notna() & (ts > 0)
            times = ts[mask].values.astype(float)
            censored_flag = cf[mask].values.astype(float)

        if len(times) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 data points")

        failures = times[censored_flag == 1].tolist()
        right_censored = times[censored_flag == 0].tolist()

        n_total = len(times)
        n_fail = len(failures)
        n_cens = len(right_censored)

        if n_fail < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 failure observations")

        # ── 2. Fit Weibull ──
        if model_type == "3P" and n_fail >= 5:
            fit = Fit_Weibull_3P(
                failures=failures,
                right_censored=right_censored if right_censored else None,
                CI=conf,
                show_probability_plot=False,
                print_results=False,
            )
            beta = fit.beta
            eta = fit.alpha  # reliability pkg uses alpha for scale
            gamma = fit.gamma
            beta_ci = [safe_float(fit.beta_lower), safe_float(fit.beta_upper)]
            eta_ci = [safe_float(fit.alpha_lower), safe_float(fit.alpha_upper)]
            gamma_ci = [safe_float(fit.gamma_lower), safe_float(fit.gamma_upper)]
            log_lik = safe_float(fit.loglik)
            aic = safe_float(fit.AICc)
            bic = safe_float(fit.BIC)
            ad = safe_float(fit.AD)
            gof_str = fit.goodness_of_fit.to_dict() if hasattr(fit, 'goodness_of_fit') else {}
        else:
            model_type = "2P"
            fit = Fit_Weibull_2P(
                failures=failures,
                right_censored=right_censored if right_censored else None,
                CI=conf,
                show_probability_plot=False,
                print_results=False,
            )
            beta = fit.beta
            eta = fit.alpha
            gamma = 0.0
            beta_ci = [safe_float(fit.beta_lower), safe_float(fit.beta_upper)]
            eta_ci = [safe_float(fit.alpha_lower), safe_float(fit.alpha_upper)]
            gamma_ci = [0, 0]
            log_lik = safe_float(fit.loglik)
            aic = safe_float(fit.AICc)
            bic = safe_float(fit.BIC)
            ad = safe_float(fit.AD)
            gof_str = {}

        # ── 3. Derived Metrics ──
        from scipy.special import gamma as gamma_func

        mttf = gamma * eta * gamma_func(1 + 1 / beta) if model_type == "3P" else eta * gamma_func(1 + 1 / beta)
        if model_type == "3P":
            mttf = gamma + eta * gamma_func(1 + 1 / beta)

        median_life = gamma + eta * np.log(2) ** (1 / beta)

        # B-lives
        def b_life(reliability_pct):
            p = 1 - reliability_pct / 100
            return gamma + eta * (-np.log(1 - p)) ** (1 / beta)

        b1 = b_life(99)
        b5 = b_life(95)
        b10 = b_life(90)
        b50 = b_life(50)

        # Reliability at target times
        def reliability_at(t):
            if t <= gamma:
                return 100.0
            return np.exp(-((t - gamma) / eta) ** beta) * 100

        target_results = [{'time': t, 'reliability_pct': safe_float(reliability_at(t))} for t in target_times]

        # Failure rate at MTTF
        t_mttf = mttf - gamma if mttf > gamma else 0.001
        fr_mttf = (beta / eta) * (t_mttf / eta) ** (beta - 1) if t_mttf > 0 else 0

        # ── 4. Curve Data for Frontend Charts ──
        t_max = max(failures) * 2 if failures else 100
        t_arr = np.linspace(gamma + 0.001, t_max, 150)

        rel_curve = np.exp(-((t_arr - gamma) / eta) ** beta) * 100
        haz_curve = (beta / eta) * ((t_arr - gamma) / eta) ** (beta - 1)
        pdf_curve = (beta / eta) * ((t_arr - gamma) / eta) ** (beta - 1) * np.exp(-((t_arr - gamma) / eta) ** beta)
        cdf_curve = 1 - np.exp(-((t_arr - gamma) / eta) ** beta)

        # Weibull probability plot data (median rank)
        sorted_f = np.sort(failures)
        nf = len(sorted_f)
        median_ranks = (np.arange(1, nf + 1) - 0.3) / (nf + 0.4)

        # ── 5. Matplotlib Composite Plot ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title_suffix = f'β={beta:.4f}, η={eta:.2f}' + (f', γ={gamma:.2f}' if gamma != 0 else '')
        fig.suptitle(f'Weibull Analysis — {title_suffix}', fontsize=14, fontweight='bold')

        # Weibull Probability Plot
        ax = axes[0, 0]
        wpp_y = np.log(np.log(1 / (1 - median_ranks)))
        wpp_x = np.log(sorted_f - gamma) if gamma > 0 else np.log(sorted_f)
        ax.scatter(wpp_x, wpp_y, color='#1E40AF', s=30, zorder=3, label='Data')
        fit_t = np.linspace(sorted_f[0], sorted_f[-1] * 1.5, 100)
        fit_cdf = 1 - np.exp(-((fit_t - gamma) / eta) ** beta)
        fit_cdf_clip = np.clip(fit_cdf, 1e-15, 1 - 1e-15)
        fit_wpp_y = np.log(np.log(1 / (1 - fit_cdf_clip)))
        fit_wpp_x = np.log(fit_t - gamma) if gamma > 0 else np.log(fit_t)
        valid = np.isfinite(fit_wpp_y) & np.isfinite(fit_wpp_x)
        ax.plot(fit_wpp_x[valid], fit_wpp_y[valid], color='#EF4444', linewidth=2, label='Fitted line')
        ax.set_xlabel('ln(Time' + (' − γ)' if gamma > 0 else ')'))
        ax.set_ylabel('ln(ln(1/(1−F)))')
        ax.set_title('Weibull Probability Plot')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Reliability
        ax = axes[0, 1]
        ax.plot(t_arr, rel_curve, color='#1E40AF', linewidth=2.5)
        ax.axhline(y=90, color='#F59E0B', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=50, color='#EF4444', linestyle='--', alpha=0.5, linewidth=1)
        if b10 > gamma:
            ax.axvline(x=b10, color='#F59E0B', linestyle=':', alpha=0.4)
        if median_life > gamma:
            ax.axvline(x=median_life, color='#EF4444', linestyle=':', alpha=0.4)
        ax.set_xlabel('Time')
        ax.set_ylabel('Reliability (%)')
        ax.set_title('Reliability Function R(t)')
        ax.set_ylim([0, 105])
        ax.grid(True, linestyle='--', alpha=0.5)

        # Hazard
        ax = axes[1, 0]
        ax.plot(t_arr, haz_curve, color='#DC2626', linewidth=2.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('h(t)')
        ax.set_title(f'Hazard Function — {failure_mode_label(beta)}')
        ax.grid(True, linestyle='--', alpha=0.5)

        # PDF
        ax = axes[1, 1]
        ax.plot(t_arr, pdf_curve, color='#059669', linewidth=2.5)
        ax.fill_between(t_arr, pdf_curve, alpha=0.08, color='#059669')
        ax.axvline(x=mttf, color='#1E40AF', linestyle='--', alpha=0.5, label=f'MTTF={mttf:.1f}')
        ax.axvline(x=median_life, color='#7C3AED', linestyle='--', alpha=0.5, label=f'Median={median_life:.1f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('f(t)')
        ax.set_title('Probability Density Function')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # ── 6. Build Response ──
        results = {
            'parameters': {
                'model': model_type,
                'beta': safe_float(beta),
                'eta': safe_float(eta),
                'gamma': safe_float(gamma),
                'beta_ci': [safe_float(beta_ci[0]), safe_float(beta_ci[1])],
                'eta_ci': [safe_float(eta_ci[0]), safe_float(eta_ci[1])],
                'gamma_ci': [safe_float(gamma_ci[0]), safe_float(gamma_ci[1])],
                'log_likelihood': log_lik,
                'AICc': aic,
                'BIC': bic,
            },
            'metrics': {
                'mttf': safe_float(mttf),
                'median_life': safe_float(median_life),
                'b1_life': safe_float(b1),
                'b5_life': safe_float(b5),
                'b10_life': safe_float(b10),
                'b50_life': safe_float(b50),
                'characteristic_life': safe_float(eta),
                'failure_rate_at_mttf': safe_float(fr_mttf),
                'failure_mode': failure_mode_label(beta),
            },
            'data_summary': {
                'n_total': n_total,
                'n_failures': n_fail,
                'n_censored': n_cens,
                'min_time': safe_float(float(np.min(times))),
                'max_time': safe_float(float(np.max(times))),
                'mean_time': safe_float(float(np.mean(times))),
            },
            'goodness_of_fit': {
                'anderson_darling': safe_float(ad),
                'interpretation': 'Good fit' if ad < 0.757 else ('Marginal' if ad < 1.038 else 'Poor fit'),
            },
            'reliability_targets': target_results,
            'curves': {
                'weibull_plot': {
                    'data_x': [safe_float(x) for x in sorted_f.tolist()],
                    'data_y': [safe_float(y) for y in median_ranks.tolist()],
                },
                'reliability': {
                    't': [safe_float(x) for x in t_arr.tolist()],
                    'r': [safe_float(x) for x in rel_curve.tolist()],
                },
                'hazard': {
                    't': [safe_float(x) for x in t_arr.tolist()],
                    'h': [safe_float(x) for x in haz_curve.tolist()],
                },
                'pdf': {
                    't': [safe_float(x) for x in t_arr.tolist()],
                    'f': [safe_float(x) for x in pdf_curve.tolist()],
                },
                'cdf': {
                    't': [safe_float(x) for x in t_arr.tolist()],
                    'F': [safe_float(x) for x in cdf_curve.tolist()],
                },
            },
            'confidence_level': conf,
        }

        return _to_native({
            'results': results,
            'plot': f"data:image/png;base64,{plot_b64}",
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
