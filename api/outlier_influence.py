from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()

class OutlierRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    dependent: str = Field(...)
    independents: List[str] = Field(...)

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except: return default

@router.post("/outlier-influence")
def outlier_influence(req: OutlierRequest):
    try:
        df = pd.DataFrame(req.data)
        dep = req.dependent
        indeps = req.independents
        all_vars = [dep] + indeps
        clean = df[all_vars].dropna()
        if len(clean) < 10:
            raise ValueError("Need at least 10 observations")
        y = clean[dep].astype(float).values
        X = clean[indeps].astype(float).values
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        influence = OLSInfluence(model)
        n = len(y)
        p = len(indeps) + 1
        cooks_d = influence.cooks_distance[0]
        leverage = influence.hat_matrix_diag
        student_resid = influence.resid_studentized_external
        dffits = influence.dffits[0]
        covratio = influence.cov_ratio
        fitted = model.fittedvalues
        residuals = model.resid
        # Thresholds
        cooks_thresh = 4 / n
        leverage_thresh = 2 * p / n
        dffits_thresh = 2 * np.sqrt(p / n)
        student_thresh = 2
        covratio_lower = 1 - 3 * p / n
        covratio_upper = 1 + 3 * p / n
        thresholds = {
            'cooks_d': {'moderate': cooks_thresh, 'high': 1.0, 'rule': '4/n'},
            'leverage': {'moderate': leverage_thresh, 'high': 0.5, 'rule': '2p/n'},
            'dffits': {'moderate': dffits_thresh, 'rule': '2*sqrt(p/n)'},
            'dfbetas': {'moderate': 2/np.sqrt(n), 'rule': '2/sqrt(n)'},
            'studentized_residual': {'moderate': 2, 'high': 3, 'rule': '|t| > 2'},
            'covratio': {'lower': covratio_lower, 'upper': covratio_upper, 'rule': '1 ± 3p/n'}
        }
        # Diagnostic data
        diagnostic_data = []
        for i in range(n):
            influential = (cooks_d[i] > cooks_thresh and abs(student_resid[i]) > 2) or cooks_d[i] > 1
            diagnostic_data.append({
                'index': int(i),
                'fitted': safe_float(fitted[i]),
                'residual': safe_float(residuals[i]),
                'studentized_residual': safe_float(student_resid[i]),
                'leverage': safe_float(leverage[i]),
                'cooks_d': safe_float(cooks_d[i]),
                'dffits': safe_float(dffits[i]),
                'covratio': safe_float(covratio[i]),
                'influential': influential
            })
        # Metrics
        high_cooks = np.sum(cooks_d > cooks_thresh)
        high_leverage = np.sum(leverage > leverage_thresh)
        outliers = np.sum(np.abs(student_resid) > 2)
        highly_influential = [d['index'] for d in diagnostic_data if d['influential']]
        metrics = {
            'n_observations': n,
            'n_predictors': len(indeps),
            'r_squared': safe_float(model.rsquared),
            'max_cooks_d': safe_float(np.max(cooks_d)),
            'max_leverage': safe_float(np.max(leverage)),
            'n_high_cooks': int(high_cooks),
            'n_high_leverage': int(high_leverage),
            'n_outliers': int(outliers),
            'n_highly_influential': len(highly_influential),
            'highly_influential_indices': highly_influential
        }
        # Top influential (sorted by Cook's D)
        sorted_data = sorted(diagnostic_data, key=lambda x: x['cooks_d'], reverse=True)
        top_influential = sorted_data[:10]
        # Insights
        insights = []
        if metrics['n_highly_influential'] > 0:
            insights.append({'type': 'warning', 'title': 'Influential Points Detected', 'description': f"{metrics['n_highly_influential']} observations meet multiple influence criteria"})
        if high_cooks > n * 0.1:
            insights.append({'type': 'warning', 'title': 'Many High Cook\'s D', 'description': f'{high_cooks} observations exceed threshold'})
        if high_leverage > 0:
            insights.append({'type': 'info', 'title': 'High Leverage Points', 'description': f'{high_leverage} observations have unusual X values'})
        if outliers > 0:
            insights.append({'type': 'warning', 'title': 'Outliers Present', 'description': f'{outliers} observations have |studentized residual| > 2'})
        if len(insights) == 0:
            insights.append({'type': 'info', 'title': 'No Major Issues', 'description': 'No observations with extreme influence detected'})
        # Recommendations
        recommendations = []
        if metrics['n_highly_influential'] > 0:
            recommendations.append('Investigate flagged observations for data entry errors')
            recommendations.append('Consider running sensitivity analysis without influential points')
            recommendations.append('Use robust regression methods if influence persists')
        else:
            recommendations.append('Results appear stable across observations')
            recommendations.append('Continue with standard inference')
        # Plots
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors = ['red' if c > cooks_thresh else 'steelblue' for c in cooks_d]
        ax1.bar(range(n), cooks_d, color=colors, alpha=0.7)
        ax1.axhline(cooks_thresh, color='red', linestyle='--', label=f'Threshold (4/n = {cooks_thresh:.4f})')
        ax1.set_xlabel('Observation Index')
        ax1.set_ylabel("Cook's Distance")
        ax1.set_title("Cook's Distance Plot", fontweight='bold')
        ax1.legend()
        plt.tight_layout()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', dpi=100)
        plt.close(fig1)
        buf1.seek(0)
        plot1 = base64.b64encode(buf1.read()).decode()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors2 = ['red' if d['influential'] else 'steelblue' for d in diagnostic_data]
        ax2.scatter(leverage, student_resid, c=colors2, alpha=0.6)
        ax2.axhline(2, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(-2, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(leverage_thresh, color='orange', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Leverage')
        ax2.set_ylabel('Studentized Residual')
        ax2.set_title('Leverage vs Studentized Residual', fontweight='bold')
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100)
        plt.close(fig2)
        buf2.seek(0)
        plot2 = base64.b64encode(buf2.read()).decode()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sizes = (cooks_d / np.max(cooks_d) * 200 + 20) if np.max(cooks_d) > 0 else np.ones(n) * 50
        ax3.scatter(leverage, student_resid, s=sizes, c=cooks_d, cmap='Reds', alpha=0.6)
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax3.axvline(leverage_thresh, color='orange', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Leverage')
        ax3.set_ylabel('Studentized Residual')
        ax3.set_title('Influence Plot (size = Cook\'s D)', fontweight='bold')
        plt.colorbar(ax3.collections[0], ax=ax3, label="Cook's D")
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=100)
        plt.close(fig3)
        buf3.seek(0)
        plot3 = base64.b64encode(buf3.read()).decode()
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        colors4 = ['red' if abs(d) > dffits_thresh else 'steelblue' for d in dffits]
        ax4.bar(range(n), dffits, color=colors4, alpha=0.7)
        ax4.axhline(dffits_thresh, color='red', linestyle='--')
        ax4.axhline(-dffits_thresh, color='red', linestyle='--')
        ax4.set_xlabel('Observation Index')
        ax4.set_ylabel('DFFITS')
        ax4.set_title('DFFITS Plot', fontweight='bold')
        plt.tight_layout()
        buf4 = io.BytesIO()
        plt.savefig(buf4, format='png', dpi=100)
        plt.close(fig4)
        buf4.seek(0)
        plot4 = base64.b64encode(buf4.read()).decode()
        # Coefficients
        coefs = {v: safe_float(model.params[i+1]) for i, v in enumerate(indeps)}
        return _to_native({
            'metrics': metrics,
            'thresholds': thresholds,
            'insights': insights,
            'recommendations': recommendations,
            'plots': {'cooks_distance': plot1, 'leverage_residual': plot2, 'influence': plot3, 'dffits': plot4},
            'diagnostic_data': diagnostic_data,
            'top_influential': top_influential,
            'model_summary': {'dependent': dep, 'independents': indeps, 'coefficients': coefs, 'intercept': safe_float(model.params[0])}
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
