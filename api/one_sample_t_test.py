from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np
from scipy import stats
from scipy.stats import t, shapiro
import math
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

router = APIRouter()


def get_effect_size_interpretation(d: float) -> str:
    """
    Cohen's d 효과 크기 해석 (Cohen, 1988 기준).
    
    주의: 이 기준은 관행적인 것이며, 분야와 맥락에 따라 
    "small" 효과도 실질적으로 중요할 수 있습니다.
    """
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "large"
    elif abs_d >= 0.5:
        return "medium"
    elif abs_d >= 0.2:
        return "small"
    else:
        return "negligible"


class OneSampleTTestRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    params: dict = Field(..., description="Test parameters")


# ── Robust SE helper (HC0 / HC1 / HC3 / Newey-West) ──────────────
def compute_robust_se(
    data_values: np.ndarray,
    test_value: float,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    n_boot: int = 2000,
    seed: int = 42
) -> dict:
    """
    One-sample 맥락의 Robust SE.

    OLS에서 H_A: μ ≠ μ₀와 동치인 회귀:
        (Xᵢ - μ₀) ~ 1 + ε  →  intercept = μ - μ₀
    절편의 robust SE가 곧 mean difference의 robust SE.

    HC variants (MacKinnon & White, 1985; Long & Ervin, 2000):
    - HC0: White (1980) — 점근적으로 유효, 소표본 과소추정
    - HC1: HC0 × n/(n-k) — 자유도 보정
    - HC3: 제거-1 재가중 (가장 보수적, 소표본 권장)

    Newey-West: HAC SE — 자기상관 + 이분산 동시 보정
                시계열 데이터에 필요 (Newey & West, 1987)
    """
    import statsmodels.api as sm

    n = len(data_values)
    # 절편만 있는 회귀: y - μ₀ ~ 1
    y_centered = data_values - test_value
    X_ones = sm.add_constant(np.ones(n), has_constant='add')
    # X_ones는 상수열 두 개가 되므로 단순하게 ones 열만 사용
    X_ones = np.ones((n, 1))
    ols = sm.OLS(y_centered, X_ones).fit()

    results = {}
    for cov in ['HC0', 'HC1', 'HC3']:
        try:
            rob = ols.get_robustcov_results(cov_type=cov)
            # bse / tvalues / pvalues 는 numpy array (길이 1 — 절편)
            rob_se    = float(rob.bse[0])
            rob_t     = float(rob.tvalues[0])
            # 단측/양측 p값 직접 계산
            if alternative == 'two-sided':
                rob_p = float(2 * t.sf(abs(rob_t), df=n - 1))
            elif alternative == 'greater':
                rob_p = float(t.sf(rob_t, df=n - 1))
            else:
                rob_p = float(t.cdf(rob_t, df=n - 1))
            # CI
            if alternative == 'two-sided':
                t_crit = t.ppf(1 - alpha / 2, df=n - 1)
                mean_d = float(np.mean(y_centered))
                ci = [float(mean_d - t_crit * rob_se), float(mean_d + t_crit * rob_se)]
            elif alternative == 'greater':
                t_crit = t.ppf(1 - alpha, df=n - 1)
                mean_d = float(np.mean(y_centered))
                ci = [float(mean_d - t_crit * rob_se), None]
            else:
                t_crit = t.ppf(1 - alpha, df=n - 1)
                mean_d = float(np.mean(y_centered))
                ci = [None, float(mean_d + t_crit * rob_se)]

            results[cov] = {
                'se':        rob_se,
                't_stat':    rob_t,
                'p_value':   rob_p,
                'ci':        ci,
                'significant': bool(rob_p < alpha)
            }
        except Exception as e:
            results[cov] = {'error': str(e)}

    # Newey-West (HAC) — 자기상관 + 이분산 동시 보정
    try:
        rob_nw = ols.get_robustcov_results(cov_type='HAC', maxlags=int(np.floor(4*(n/100)**0.25)))
        nw_se  = float(rob_nw.bse[0])
        nw_t   = float(rob_nw.tvalues[0])
        mean_d = float(np.mean(y_centered))
        if alternative == 'two-sided':
            nw_p = float(2 * t.sf(abs(nw_t), df=n - 1))
            t_crit = t.ppf(1 - alpha / 2, df=n - 1)
            nw_ci = [float(mean_d - t_crit * nw_se), float(mean_d + t_crit * nw_se)]
        elif alternative == 'greater':
            nw_p = float(t.sf(nw_t, df=n - 1))
            t_crit = t.ppf(1 - alpha, df=n - 1)
            nw_ci = [float(mean_d - t_crit * nw_se), None]
        else:
            nw_p = float(t.cdf(nw_t, df=n - 1))
            t_crit = t.ppf(1 - alpha, df=n - 1)
            nw_ci = [None, float(mean_d + t_crit * nw_se)]
        results['Newey_West'] = {
            'se': nw_se, 't_stat': nw_t, 'p_value': nw_p,
            'ci': nw_ci, 'significant': bool(nw_p < alpha),
            'maxlags': int(np.floor(4*(n/100)**0.25))
        }
    except Exception as e:
        results['Newey_West'] = {'error': str(e)}

    results['_note'] = (
        'HC3 권장 (Long & Ervin, 2000) — 소표본에서 가장 보수적이고 정확. '
        'HC0는 점근적으로 유효하나 소표본 SE 과소추정. '
        'HC1은 HC0의 자유도 보정 버전. '
        'Newey-West는 시계열 자기상관 + 이분산 동시 보정 (HAC).'
    )
    return results


# ── Bootstrap CI helper ───────────────────────────────────────────
def compute_bootstrap_ci(
    data_values: np.ndarray,
    test_value: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    seed: int = 42
) -> dict:
    """
    Bootstrapped confidence interval for the mean.

    세 가지 방법 동시 제공:
    - Percentile: 부트스트랩 분포 직접 백분위 (Efron, 1979)
    - BCa (Bias-Corrected accelerated): 편향 + 가속 보정, 가장 정확
                                         (Efron & Tibshirani, 1993)
    - Studentized (Bootstrap-t): t 통계량 부트스트랩 → 가장 이론적으로 엄밀
    """
    rng = np.random.default_rng(seed)
    n   = len(data_values)
    obs_mean = float(np.mean(data_values))
    obs_diff = obs_mean - test_value

    # 부트스트랩 표본 평균 분포
    boot_means = np.array([
        np.mean(rng.choice(data_values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    boot_diffs = boot_means - test_value

    # ── Percentile ────────────────────────────────────────────────
    if alternative == 'two-sided':
        pct_lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
        pct_hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
        pct_ci = [pct_lo, pct_hi]
    elif alternative == 'greater':
        pct_lo = float(np.percentile(boot_diffs, 100 * alpha))
        pct_ci = [pct_lo, None]
    else:
        pct_hi = float(np.percentile(boot_diffs, 100 * (1 - alpha)))
        pct_ci = [None, pct_hi]

    # ── BCa (Bias-Corrected accelerated) ─────────────────────────
    try:
        from scipy.special import ndtri, ndtr
        # 편향 보정 z₀
        z0 = float(ndtri(np.mean(boot_diffs < obs_diff) + 1e-12))
        # 가속도 a: jackknife 추정
        jk_means = np.array([(np.sum(data_values) - v) / (n - 1) for v in data_values])
        jk_mean  = np.mean(jk_means)
        num  = np.sum((jk_mean - jk_means) ** 3)
        denom= np.sum((jk_mean - jk_means) ** 2) ** 1.5
        a    = float(num / (6 * denom)) if denom != 0 else 0.0

        def bca_quantile(prob):
            z_p  = ndtri(prob)
            adj  = ndtr(z0 + (z0 + z_p) / (1 - a * (z0 + z_p)))
            return float(np.percentile(boot_diffs, 100 * adj))

        if alternative == 'two-sided':
            bca_ci = [bca_quantile(alpha / 2), bca_quantile(1 - alpha / 2)]
        elif alternative == 'greater':
            bca_ci = [bca_quantile(alpha), None]
        else:
            bca_ci = [None, bca_quantile(1 - alpha)]
    except Exception as e:
        bca_ci = {'error': str(e)}

    # ── Studentized (Bootstrap-t) ─────────────────────────────────
    try:
        obs_se = float(np.std(data_values, ddof=1) / np.sqrt(n))
        boot_t_stats = []
        for _ in range(n_boot):
            bs = rng.choice(data_values, size=n, replace=True)
            bs_mean = float(np.mean(bs))
            bs_se   = float(np.std(bs, ddof=1) / np.sqrt(n))
            if bs_se > 0:
                boot_t_stats.append((bs_mean - obs_mean) / bs_se)
        boot_t_stats = np.array(boot_t_stats)

        if alternative == 'two-sided':
            t_lo = float(np.percentile(boot_t_stats, 100 * alpha / 2))
            t_hi = float(np.percentile(boot_t_stats, 100 * (1 - alpha / 2)))
            stud_ci = [float(obs_diff - t_hi * obs_se), float(obs_diff - t_lo * obs_se)]
        elif alternative == 'greater':
            t_lo = float(np.percentile(boot_t_stats, 100 * alpha))
            stud_ci = [float(obs_diff - t_lo * obs_se), None]
        else:
            t_hi = float(np.percentile(boot_t_stats, 100 * (1 - alpha)))
            stud_ci = [None, float(obs_diff - t_hi * obs_se)]
    except Exception as e:
        stud_ci = {'error': str(e)}

    # ── Bootstrap p-value ─────────────────────────────────────────
    # H₀ 중심화: 귀무가설 하에서 부트스트랩 분포 = mean centered at 0
    null_diffs = boot_diffs - np.mean(boot_diffs)
    if alternative == 'two-sided':
        boot_p = float(np.mean(np.abs(null_diffs) >= abs(obs_diff)))
    elif alternative == 'greater':
        boot_p = float(np.mean(null_diffs >= obs_diff))
    else:
        boot_p = float(np.mean(null_diffs <= obs_diff))

    return {
        'n_bootstrap':  n_boot,
        'obs_mean_diff': float(obs_diff),
        'bootstrap_se': float(np.std(boot_diffs, ddof=1)),
        'bootstrap_p_value': boot_p,
        'ci': {
            'percentile':   pct_ci,
            'bca':          bca_ci,
            'studentized':  stud_ci
        },
        'significant': bool(boot_p < alpha),
        '_note': (
            f'Bootstrap n_resamples={n_boot}, seed={seed}. '
            'BCa 권장 (Efron & Tibshirani, 1993) — 편향·가속 보정으로 가장 정확. '
            'Percentile은 단순하나 작은 표본에서 CI 범위 과소추정 가능. '
            'Studentized는 이론적으로 가장 엄밀하나 계산 비용이 가장 큼.'
        )
    }


def compute_confidence_interval(
    sample_mean: float,
    standard_error: float,
    df: int,
    alpha: float,
    alternative: str
) -> tuple[float | None, float | None]:
    """
    alternative 가설에 맞는 신뢰구간 계산.
    
    - two-sided: 양측 (1-α) CI → [lower, upper]
    - greater: 단측 (1-α) CI → [lower, +∞] (upper=None)
    - less: 단측 (1-α) CI → [-∞, upper] (lower=None)
    """
    if alternative == 'two-sided':
        ci_lower, ci_upper = t.interval(1 - alpha, df, loc=sample_mean, scale=standard_error)
        return float(ci_lower), float(ci_upper)
    elif alternative == 'greater':
        # H1: μ > test_value → 하한만 계산
        t_crit = t.ppf(alpha, df)  # 하위 α 백분위
        ci_lower = sample_mean + t_crit * standard_error
        return float(ci_lower), None
    elif alternative == 'less':
        # H1: μ < test_value → 상한만 계산
        t_crit = t.ppf(1 - alpha, df)  # 상위 α 백분위
        ci_upper = sample_mean + t_crit * standard_error
        return None, float(ci_upper)
    else:
        # 알 수 없는 alternative → 양측으로 fallback
        ci_lower, ci_upper = t.interval(1 - alpha, df, loc=sample_mean, scale=standard_error)
        return float(ci_lower), float(ci_upper)


def generate_plot(
    data_values: np.ndarray,
    sample_mean: float,
    test_value: float,
    t_stat: float,
    df: int,
    variable_name: str,
    alternative: str = 'two-sided',
    alpha: float = 0.05
) -> str:
    """
    alternative에 맞게 t-분포 음영 영역을 시각화.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 데이터 분포 히스토그램
    sns.histplot(data_values, ax=axes[0, 0], color='#5B9BD5', kde=True)
    axes[0, 0].axvline(sample_mean, color='red', linestyle='--', label=f'Sample Mean ({sample_mean:.2f})')
    axes[0, 0].axvline(test_value, color='orange', linestyle='--', label=f'Test Value ({test_value})')
    axes[0, 0].set_title('Data Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel(variable_name, fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].legend()
    
    # 2. Q-Q Plot
    stats.probplot(data_values, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    
    # 3. t-분포와 검정통계량 (alternative에 맞게 음영 처리)
    if df > 0 and np.isfinite(df):
        # [4] x 범위: df 크거나 t_stat 극단값일 때 ±4로 잘리지 않도록 동적 확장
        _x_bound = max(6.0, abs(t_stat) + 2.0)
        x = np.linspace(-_x_bound, _x_bound, 800)
        y = t.pdf(x, df)
        axes[1, 0].plot(x, y, label=f't-distribution (df={df:.0f})', color='#5B9BD5')
        axes[1, 0].axvline(t_stat, color='red', linestyle='--', linewidth=1.8,
                           label=f't-stat = {t_stat:.3f}')

        # [4b] p-value 영역 (관측 t_stat 기준) vs 기각역 (α 기준 t_crit) 분리
        # ─ p-value region: 빨강 (관측값보다 극단적인 영역)
        # ─ rejection region: 주황 점선 (α 기준 t_crit)
        if alternative == 'two-sided':
            _t_crit = t.ppf(1 - alpha / 2, df)
            axes[1, 0].fill_between(
                x, 0, y,
                where=(x >= abs(t_stat)) | (x <= -abs(t_stat)),
                color='red', alpha=0.25, label=f'p-value region (|t| ≥ {abs(t_stat):.2f})'
            )
            axes[1, 0].axvline( _t_crit, color='orange', linestyle=':', linewidth=1.5,
                                label=f'α/2 critical (±{_t_crit:.2f})')
            axes[1, 0].axvline(-_t_crit, color='orange', linestyle=':', linewidth=1.5)
            axes[1, 0].set_title('t-Distribution: p-value & Rejection Region (Two-sided)',
                                 fontsize=11, fontweight='bold')
        elif alternative == 'greater':
            _t_crit = t.ppf(1 - alpha, df)
            axes[1, 0].fill_between(
                x, 0, y,
                where=(x >= t_stat),
                color='red', alpha=0.25, label=f'p-value region (t ≥ {t_stat:.2f})'
            )
            axes[1, 0].axvline(_t_crit, color='orange', linestyle=':', linewidth=1.5,
                               label=f'α critical ({_t_crit:.2f})')
            axes[1, 0].set_title('t-Distribution: p-value & Rejection Region (One-sided: greater)',
                                 fontsize=11, fontweight='bold')
        elif alternative == 'less':
            _t_crit = t.ppf(alpha, df)
            axes[1, 0].fill_between(
                x, 0, y,
                where=(x <= t_stat),
                color='red', alpha=0.25, label=f'p-value region (t ≤ {t_stat:.2f})'
            )
            axes[1, 0].axvline(_t_crit, color='orange', linestyle=':', linewidth=1.5,
                               label=f'α critical ({_t_crit:.2f})')
            axes[1, 0].set_title('t-Distribution: p-value & Rejection Region (One-sided: less)',
                                 fontsize=11, fontweight='bold')

        axes[1, 0].set_xlabel('t', fontsize=11)
        axes[1, 0].legend(fontsize=8)
    
    # 4. Box Plot
    sns.boxplot(x=data_values, ax=axes[1, 1], color='#5B9BD5')
    axes[1, 1].axvline(test_value, color='orange', linestyle='--', label=f'Test Value ({test_value})')
    axes[1, 1].set_title('Box Plot', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def generate_interpretation(
    variable: str,
    test_value: float,
    sample_mean: float,
    sample_std: float,
    t_stat: float,
    df: int,
    p_value: float,
    significant: bool,
    ci: tuple[float | None, float | None],
    cohens_d: float,
    hedges_g: float,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> str:
    """
    alternative에 따라 적절한 해석 문구 생성.
    """
    p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
    sig_text = "statistically significant" if significant else "not statistically significant"
    effect_interp = get_effect_size_interpretation(cohens_d)
    ci_pct = int((1 - alpha) * 100)
    
    # 가설 방향에 맞는 설명
    if alternative == 'two-sided':
        direction_text = "different from"
    elif alternative == 'greater':
        direction_text = "greater than"
    elif alternative == 'less':
        direction_text = "less than"
    else:
        direction_text = "different from"
    
    interpretation = (
        f"A one-sample t-test was conducted to determine whether the mean of '{variable}' "
        f"was {direction_text} the test value of {test_value}.\n\n"
    )
    
    interpretation += (
        f"There was a {sig_text} difference between the sample mean "
        f"(M={sample_mean:.2f}, SD={sample_std:.2f}) and the test value of {test_value}, "
        f"t({df}) = {t_stat:.2f}, {p_text}.\n\n"
    )
    
    # CI 해석 (alternative에 따라 다르게)
    ci_lower, ci_upper = ci
    
    if alternative == 'two-sided':
        # 양측 검정: CI와 test value 포함 여부로 유의성 설명 가능
        interpretation += f"The {ci_pct}% confidence interval for the mean is [{ci_lower:.2f}, {ci_upper:.2f}]. "
        
        if not (ci_lower <= test_value <= ci_upper):
            interpretation += (
                f"Since this interval does not contain the test value of {test_value}, "
                f"the result is significant.\n\n"
            )
        else:
            interpretation += (
                f"Since this interval contains the test value of {test_value}, "
                f"the result is not significant.\n\n"
            )
    elif alternative == 'greater':
        # 단측 (greater): 하한만 있음
        interpretation += (
            f"The {ci_pct}% one-sided confidence interval for the mean is [{ci_lower:.2f}, +∞). "
        )
        if ci_lower > test_value:
            interpretation += (
                f"Since the lower bound ({ci_lower:.2f}) is greater than the test value of {test_value}, "
                f"the result is significant.\n\n"
            )
        else:
            interpretation += (
                f"Since the lower bound ({ci_lower:.2f}) is not greater than the test value of {test_value}, "
                f"the result is not significant.\n\n"
            )
    elif alternative == 'less':
        # 단측 (less): 상한만 있음
        interpretation += (
            f"The {ci_pct}% one-sided confidence interval for the mean is (-∞, {ci_upper:.2f}]. "
        )
        if ci_upper < test_value:
            interpretation += (
                f"Since the upper bound ({ci_upper:.2f}) is less than the test value of {test_value}, "
                f"the result is significant.\n\n"
            )
        else:
            interpretation += (
                f"Since the upper bound ({ci_upper:.2f}) is not less than the test value of {test_value}, "
                f"the result is not significant.\n\n"
            )
    
    # 효과 크기 해석 — Cohen's d + Hedges' g (bias-corrected)
    interpretation += (
        f"Cohen's d = {cohens_d:.3f} indicates a {effect_interp} effect size "
        f"(Cohen, 1988 benchmarks: small=0.2, medium=0.5, large=0.8). "
        f"Hedges' g = {hedges_g:.3f} (bias-corrected for small samples; "
        f"Hedges & Olkin, 1985). "
        f"Note that practical significance may vary depending on the research context."
    )
    
    return interpretation.strip()


@router.post("/one-sample-ttest")
def one_sample_t_test(req: OneSampleTTestRequest):
    try:
        data = req.data
        params = req.params
        
        variable   = params.get('variable')
        test_value = float(params.get('test_value', 0))
        alternative= params.get('alternative', 'two-sided')
        alpha      = float(params.get('alpha', 0.05))
        n_boot     = int(params.get('n_bootstrap', 2000))
        boot_seed  = int(params.get('bootstrap_seed', 42))
        
        # alternative 값 검증
        if alternative not in ('two-sided', 'greater', 'less'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid alternative '{alternative}'. Must be 'two-sided', 'greater', or 'less'."
            )
        
        if not variable:
            raise HTTPException(status_code=400, detail="Variable name is required")
        
        dropped_rows = []
        data_values = []
        
        for idx, row in enumerate(data):
            val = row.get(variable)
            if val is None or val == '' or (isinstance(val, float) and math.isnan(val)):
                dropped_rows.append(idx)
            else:
                try:
                    data_values.append(float(val))
                except (ValueError, TypeError):
                    dropped_rows.append(idx)
        
        data_values = np.array(data_values)
        
        if len(data_values) < 2:
            raise HTTPException(status_code=400, detail="Not enough valid data points (minimum 2 required)")
        
        n = int(len(data_values))
        sample_mean = float(np.mean(data_values))
        sample_std = float(np.std(data_values, ddof=1))
        standard_error = float(sample_std / np.sqrt(n))
        
        # [3] Shapiro-Wilk: 3 ≤ n ≤ 5000 범위에서만 신뢰도 높음
        normality_test = None
        if 3 <= n <= 5000:
            stat, p = shapiro(data_values)
            _sw_note = None
            if n > 50:
                _sw_note = (
                    f"n = {n} > 50: The Central Limit Theorem suggests the sampling "
                    "distribution of the mean is approximately normal regardless of "
                    "the population distribution. Normality assumption may be relaxed."
                )
            normality_test = {
                variable: {
                    'statistic': float(stat),
                    'p_value': float(p),
                    'assumption_met': bool(p > alpha),
                    'note': _sw_note
                }
            }
        elif n > 5000:
            normality_test = {
                variable: {
                    'statistic': None,
                    'p_value': None,
                    'assumption_met': None,
                    'note': (
                        f"n = {n} > 5000: Shapiro-Wilk is unreliable at this sample size. "
                        "Apply CLT — normality assumption is effectively met."
                    )
                }
            }
        
        t_stat, p_value = stats.ttest_1samp(data_values, test_value, alternative=alternative)
        t_stat = float(t_stat)
        p_value = float(p_value)
        df = int(n - 1)
        
        if np.isnan(p_value):
            raise HTTPException(status_code=400, detail="Cannot compute p-value. Data values may be constant.")
        
        # alternative에 맞는 신뢰구간 계산
        ci_lower, ci_upper = compute_confidence_interval(
            sample_mean=sample_mean,
            standard_error=standard_error,
            df=df,
            alpha=alpha,
            alternative=alternative
        )
        
        # [1] constant data 조기 방어: std == 0이면 t-test 자체 불가
        if sample_std == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"All {n} values of '{variable}' are identical ({sample_mean}). "
                    "A t-test requires variation in the data."
                )
            )

        cohens_d = float((sample_mean - test_value) / sample_std)
        # [2] Hedges' g — bias-corrected effect size (SPSS 방식)
        # J(df) correction factor: Hedges & Olkin (1985)
        _j_correction = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
        hedges_g = float(cohens_d * _j_correction)

        significant = bool(p_value < alpha)
        
        # ── Robust Standard Errors (HC0 / HC1 / HC3 / Newey-West) ──
        robust_se_results = compute_robust_se(
            data_values=data_values,
            test_value=test_value,
            alpha=alpha,
            alternative=alternative
        )

        # ── Bootstrapped Confidence Intervals ────────────────────
        bootstrap_results = compute_bootstrap_ci(
            data_values=data_values,
            test_value=test_value,
            n_boot=n_boot,
            alpha=alpha,
            alternative=alternative,
            seed=boot_seed
        )

        interpretation = generate_interpretation(
            variable=variable,
            test_value=test_value,
            sample_mean=sample_mean,
            sample_std=sample_std,
            t_stat=t_stat,
            df=df,
            p_value=p_value,
            significant=significant,
            ci=(ci_lower, ci_upper),
            cohens_d=cohens_d,
            hedges_g=hedges_g,
            alpha=alpha,
            alternative=alternative
        )
        
        plot = generate_plot(
            data_values=data_values,
            sample_mean=sample_mean,
            test_value=test_value,
            t_stat=t_stat,
            df=df,
            variable_name=variable,
            alternative=alternative,
            alpha=alpha
        )
        
        # 응답에서 CI 형식 결정
        if alternative == 'two-sided':
            ci_response = [ci_lower, ci_upper]
        elif alternative == 'greater':
            ci_response = {"lower": ci_lower, "upper": None, "type": "one-sided-greater"}
        elif alternative == 'less':
            ci_response = {"lower": None, "upper": ci_upper, "type": "one-sided-less"}
        else:
            ci_response = [ci_lower, ci_upper]
        
        return {
            "results": {
                "test_type": "one_sample",
                "variable": variable,
                "test_value": float(test_value),
                "alternative": alternative,
                "n": n,
                "sample_mean": sample_mean,
                "se_diff": standard_error,
                "t_statistic": t_stat,
                "degrees_of_freedom": df,
                "p_value": p_value,
                "significant": significant,
                "confidence_interval": ci_response,
                "cohens_d": cohens_d,
                "hedges_g": hedges_g,
                "interpretation": interpretation,
                "dropped_rows": dropped_rows,
                "n_dropped": int(len(dropped_rows)),
                "normality_test": normality_test,
                "robust_se": robust_se_results,
                "bootstrap": bootstrap_results,
                "descriptives": {
                    variable: {
                        "n": n,
                        "mean": sample_mean,
                        "std_dev": sample_std,
                        "se_mean": standard_error
                    }
                }
            },
            "plot": plot
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
