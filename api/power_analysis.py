from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
from scipy import stats
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class PowerAnalysisRequest(BaseModel):
    test_type: str = Field(default="two_sample_t")
    analysis_type: str = Field(default="power")  # 'power' or 'sample_size'
    effect_size: float = Field(default=0.5)
    n: Optional[int] = Field(default=30)
    n2: Optional[int] = Field(default=None)
    alpha: float = Field(default=0.05)
    power_target: float = Field(default=0.8)
    alternative: str = Field(default="two-sided")
    k: int = Field(default=3)
    df: int = Field(default=1)


# Power calculation functions
def power_t_test_one_sample(effect_size, n, alpha=0.05, alternative='two-sided'):
    df = n - 1
    ncp = effect_size * np.sqrt(n)
    if alternative == 'two-sided':
        critical_t = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
    else:
        critical_t = stats.t.ppf(1 - alpha, df)
        power = 1 - stats.nct.cdf(critical_t, df, ncp) if alternative == 'greater' else stats.nct.cdf(-critical_t, df, ncp)
    return float(power)

def power_t_test_two_sample(effect_size, n1, n2=None, alpha=0.05, alternative='two-sided'):
    if n2 is None:
        n2 = n1
    df = n1 + n2 - 2
    pooled_n = np.sqrt(1/n1 + 1/n2)
    ncp = effect_size / pooled_n
    if alternative == 'two-sided':
        critical_t = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
    else:
        critical_t = stats.t.ppf(1 - alpha, df)
        power = 1 - stats.nct.cdf(critical_t, df, ncp) if alternative == 'greater' else stats.nct.cdf(-critical_t, df, ncp)
    return float(power)

def power_anova(effect_size, k, n, alpha=0.05):
    df1 = k - 1
    df2 = k * (n - 1)
    ncp = n * k * effect_size**2
    critical_f = stats.f.ppf(1 - alpha, df1, df2)
    power = 1 - stats.ncf.cdf(critical_f, df1, df2, ncp)
    return float(power)

def power_correlation(effect_size, n, alpha=0.05, alternative='two-sided'):
    z_r = 0.5 * np.log((1 + effect_size) / (1 - effect_size + 1e-10))
    se = 1 / np.sqrt(n - 3) if n > 3 else 1
    if alternative == 'two-sided':
        z_crit = stats.norm.ppf(1 - alpha/2)
        power = 1 - stats.norm.cdf(z_crit - z_r/se) + stats.norm.cdf(-z_crit - z_r/se)
    else:
        z_crit = stats.norm.ppf(1 - alpha)
        power = 1 - stats.norm.cdf(z_crit - z_r/se) if alternative == 'greater' else stats.norm.cdf(-z_crit - z_r/se)
    return float(min(max(power, 0), 1))

def power_chi_square(effect_size, df, n, alpha=0.05):
    ncp = n * effect_size**2
    critical_chi2 = stats.chi2.ppf(1 - alpha, df)
    power = 1 - stats.ncx2.cdf(critical_chi2, df, ncp)
    return float(power)


# Sample size functions
def sample_size_t_one(effect_size, power=0.8, alpha=0.05, alternative='two-sided'):
    for n in range(5, 10000):
        if power_t_test_one_sample(effect_size, n, alpha, alternative) >= power:
            return n
    return None

def sample_size_t_two(effect_size, power=0.8, alpha=0.05, alternative='two-sided'):
    for n in range(5, 10000):
        if power_t_test_two_sample(effect_size, n, n, alpha, alternative) >= power:
            return n, n
    return None, None

def sample_size_anova(effect_size, k, power=0.8, alpha=0.05):
    for n in range(5, 10000):
        if power_anova(effect_size, k, n, alpha) >= power:
            return n
    return None

def sample_size_correlation(effect_size, power=0.8, alpha=0.05, alternative='two-sided'):
    for n in range(10, 10000):
        if power_correlation(effect_size, n, alpha, alternative) >= power:
            return n
    return None


# Effect size interpretation
def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    return "large"

def interpret_f(f):
    f = abs(f)
    if f < 0.1: return "negligible"
    elif f < 0.25: return "small"
    elif f < 0.4: return "medium"
    return "large"


# Plot functions
def create_power_curve(test_type, effect_size, alpha, alternative, k=3, df=1):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if test_type == 'one_sample_t':
        n_range = np.arange(5, 201)
        powers = [power_t_test_one_sample(effect_size, n, alpha, alternative) for n in n_range]
    elif test_type == 'two_sample_t':
        n_range = np.arange(5, 151)
        powers = [power_t_test_two_sample(effect_size, n, n, alpha, alternative) for n in n_range]
    elif test_type == 'paired_t':
        n_range = np.arange(5, 201)
        powers = [power_t_test_one_sample(effect_size, n, alpha, alternative) for n in n_range]
    elif test_type == 'anova':
        n_range = np.arange(5, 101)
        powers = [power_anova(effect_size, k, n, alpha) for n in n_range]
    elif test_type == 'correlation':
        n_range = np.arange(10, 201)
        powers = [power_correlation(effect_size, n, alpha, alternative) for n in n_range]
    elif test_type == 'chi_square':
        n_range = np.arange(20, 501, 5)
        powers = [power_chi_square(effect_size, df, n, alpha) for n in n_range]
    else:
        return None
    
    ax.plot(n_range, powers, 'b-', linewidth=2.5)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Power = 0.80')
    ax.axhline(y=0.9, color='green', linestyle='--', label='Power = 0.90')
    
    for i, p in enumerate(powers):
        if p >= 0.8:
            ax.axvline(x=n_range[i], color='red', linestyle=':', alpha=0.7)
            ax.annotate(f'n={n_range[i]}', xy=(n_range[i], 0.8), xytext=(n_range[i]+5, 0.75))
            break
    
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Power')
    ax.set_title(f'Power Curve (Effect Size = {effect_size}, α = {alpha})')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_effect_size_plot(test_type, n, alpha, alternative, k=3, df=1):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if test_type in ['one_sample_t', 'two_sample_t', 'paired_t']:
        es_range = np.linspace(0.1, 1.5, 50)
        if test_type == 'two_sample_t':
            powers = [power_t_test_two_sample(es, n, n, alpha, alternative) for es in es_range]
        else:
            powers = [power_t_test_one_sample(es, n, alpha, alternative) for es in es_range]
        xlabel = "Cohen's d"
    elif test_type == 'anova':
        es_range = np.linspace(0.05, 0.6, 50)
        powers = [power_anova(es, k, n, alpha) for es in es_range]
        xlabel = "Cohen's f"
    elif test_type == 'correlation':
        es_range = np.linspace(0.05, 0.7, 50)
        powers = [power_correlation(es, n, alpha, alternative) for es in es_range]
        xlabel = "Correlation r"
    elif test_type == 'chi_square':
        es_range = np.linspace(0.05, 0.7, 50)
        powers = [power_chi_square(es, df, n, alpha) for es in es_range]
        xlabel = "Cohen's w"
    else:
        return None
    
    ax.plot(es_range, powers, 'b-', linewidth=2.5)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Power = 0.80')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Power')
    ax.set_title(f'Power by Effect Size (n = {n}, α = {alpha})')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@router.post("/power-analysis")
def power_analysis(req: PowerAnalysisRequest):
    try:
        test_type = req.test_type
        analysis_type = req.analysis_type
        effect_size = req.effect_size
        n = req.n or 30
        n2 = req.n2
        alpha = req.alpha
        power_target = req.power_target
        alternative = req.alternative
        k = req.k
        df = req.df
        
        results = {'test_type': test_type, 'effect_size': effect_size, 'alpha': alpha}
        
        if test_type == 'one_sample_t':
            test_name = "One-Sample t-Test"
            power = power_t_test_one_sample(effect_size, n, alpha, alternative)
            required_n = sample_size_t_one(effect_size, 0.8, alpha, alternative)
            results['n'] = n
            
        elif test_type == 'two_sample_t':
            test_name = "Two-Sample t-Test"
            if n2 is None: n2 = n
            power = power_t_test_two_sample(effect_size, n, n2, alpha, alternative)
            req_n, _ = sample_size_t_two(effect_size, 0.8, alpha, alternative)
            required_n = req_n
            results['n1'] = n
            results['n2'] = n2
            
        elif test_type == 'paired_t':
            test_name = "Paired t-Test"
            power = power_t_test_one_sample(effect_size, n, alpha, alternative)
            required_n = sample_size_t_one(effect_size, 0.8, alpha, alternative)
            results['n_pairs'] = n
            
        elif test_type == 'anova':
            test_name = f"One-Way ANOVA ({k} groups)"
            power = power_anova(effect_size, k, n, alpha)
            required_n = sample_size_anova(effect_size, k, 0.8, alpha)
            results['n_per_group'] = n
            results['k'] = k
            
        elif test_type == 'correlation':
            test_name = "Correlation Test"
            power = power_correlation(effect_size, n, alpha, alternative)
            required_n = sample_size_correlation(effect_size, 0.8, alpha, alternative)
            results['n'] = n
            
        elif test_type == 'chi_square':
            test_name = f"Chi-Square Test (df={df})"
            power = power_chi_square(effect_size, df, n, alpha)
            required_n = None
            for test_n in range(20, 5000):
                if power_chi_square(effect_size, df, test_n, alpha) >= 0.8:
                    required_n = test_n
                    break
            results['n'] = n
            results['df'] = df
        else:
            raise ValueError(f"Unknown test: {test_type}")
        
        results['test_name'] = test_name
        results['power'] = power
        results['required_n_for_80'] = required_n
        
        # Insights
        insights = []
        if power >= 0.8:
            insights.append({'type': 'info', 'title': 'Adequate Power ✓', 'description': f'Power of {power:.1%} meets the 80% threshold.'})
        else:
            insights.append({'type': 'warning', 'title': 'Low Power', 'description': f'Power of {power:.1%} is below 80%. Consider increasing sample size.'})
        
        # Effect size interpretation
        if test_type in ['one_sample_t', 'two_sample_t', 'paired_t']:
            es_interp = interpret_d(effect_size)
        else:
            es_interp = interpret_f(effect_size)
        insights.append({'type': 'info', 'title': f'Effect Size: {es_interp.title()}', 'description': f'The specified effect size is considered {es_interp}.'})
        
        recommendations = []
        if power < 0.8 and required_n:
            recommendations.append(f'Increase sample size to at least {required_n} for 80% power.')
        recommendations.append('Report power analysis in your methods section.')
        
        # Plots
        plots = {
            'power_curve': create_power_curve(test_type, effect_size, alpha, alternative, k, df),
            'power_by_effect_size': create_effect_size_plot(test_type, n, alpha, alternative, k, df)
        }
        
        return {
            'results': results,
            'insights': insights,
            'recommendations': recommendations,
            'plots': plots
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
