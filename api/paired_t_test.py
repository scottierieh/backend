from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np
from scipy import stats
from scipy.stats import t, shapiro
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

router = APIRouter()


class PairedTTestRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Array of data objects")
    params: dict = Field(..., description="Test parameters")


def get_effect_size_interpretation(d: float) -> str:
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "large"
    elif abs_d >= 0.5:
        return "medium"
    elif abs_d >= 0.2:
        return "small"
    else:
        return "negligible"


def compute_hedges_g(mean_diff: float, std_diff: float, n: int) -> float:
    """Hedges' g correction for small sample bias."""
    if std_diff == 0 or n < 2:
        return 0.0
    d = mean_diff / std_diff
    # Correction factor J
    j = 1.0 - (3.0 / (4.0 * (n - 1) - 1))
    return float(d * j)


def compute_pair_correlation(data1: np.ndarray, data2: np.ndarray) -> dict:
    """Pearson correlation between the two paired variables."""
    r, p = stats.pearsonr(data1, data2)
    abs_r = abs(r)
    if abs_r >= 0.7:
        interp = "strong"
    elif abs_r >= 0.4:
        interp = "moderate"
    elif abs_r >= 0.2:
        interp = "weak"
    else:
        interp = "negligible"
    return {
        "r": float(r),
        "p_value": float(p),
        "interpretation": interp,
        "pairing_effective": bool(r > 0.3),
        "note": "Positive correlation suggests pairing reduces error variance." if r > 0.3 else "Low correlation — pairing may not reduce error variance much."
    }


def compute_bootstrap(differences: np.ndarray, n_bootstrap: int = 5000, alpha: float = 0.05) -> dict:
    """Bootstrap confidence intervals (BCa, percentile, studentized)."""
    np.random.seed(42)
    n = len(differences)
    obs_mean = float(np.mean(differences))

    boot_means = np.array([
        np.mean(np.random.choice(differences, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Percentile CI
    lo_pct = float(np.percentile(boot_means, 100 * alpha / 2))
    hi_pct = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # BCa CI
    z0 = float(stats.norm.ppf(np.mean(boot_means < obs_mean) + 1e-10))
    jk_means = np.array([
        np.mean(np.delete(differences, i)) for i in range(n)
    ])
    jk_bar = np.mean(jk_means)
    num = np.sum((jk_bar - jk_means) ** 3)
    den = 6.0 * (np.sum((jk_bar - jk_means) ** 2) ** 1.5)
    a_hat = float(num / den) if den != 0 else 0.0
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)
    alpha1 = float(stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha))))
    alpha2 = float(stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha))))
    lo_bca = float(np.percentile(boot_means, 100 * alpha1))
    hi_bca = float(np.percentile(boot_means, 100 * alpha2))

    # Studentized (bootstrap-t) CI
    boot_t = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(differences, size=n, replace=True)
        se_b = np.std(sample, ddof=1) / np.sqrt(n)
        if se_b > 0:
            boot_t.append((np.mean(sample) - obs_mean) / se_b)
    boot_t = np.array(boot_t)
    se_orig = np.std(differences, ddof=1) / np.sqrt(n)
    lo_stud = float(obs_mean - np.percentile(boot_t, 100 * (1 - alpha / 2)) * se_orig)
    hi_stud = float(obs_mean - np.percentile(boot_t, 100 * alpha / 2) * se_orig)

    # Bootstrap p-value (two-sided)
    shifted = differences - obs_mean
    null_means = np.array([np.mean(np.random.choice(shifted, size=n, replace=True)) for _ in range(n_bootstrap)])
    p_boot = float(np.mean(np.abs(null_means) >= np.abs(obs_mean)))

    return {
        "p_value": p_boot,
        "n_bootstrap": n_bootstrap,
        "ci": [lo_pct, hi_pct],
        "ci_bca": [lo_bca, hi_bca],
        "ci_percentile": [lo_pct, hi_pct],
        "ci_studentized": [lo_stud, hi_stud],
        "boot_mean": float(np.mean(boot_means)),
        "boot_se": float(np.std(boot_means, ddof=1)),
        "bca_z0": z0,
        "bca_a_hat": a_hat,
        "observed_mean_diff": obs_mean,
    }


def compute_permutation(differences: np.ndarray, n_permutations: int = 10000) -> dict:
    """Sign-permutation test for paired data."""
    np.random.seed(42)
    n = len(differences)
    obs_mean = float(np.mean(differences))
    count = 0
    for _ in range(n_permutations):
        signs = np.random.choice([-1, 1], size=n)
        perm_mean = np.mean(differences * signs)
        if abs(perm_mean) >= abs(obs_mean):
            count += 1
    return {
        "p_value": float(count / n_permutations),
        "n_permutations": n_permutations,
        "observed_mean_diff": obs_mean,
    }


def compute_yuen_trimmed_t(data1: np.ndarray, data2: np.ndarray, trim: float = 0.2) -> dict | None:
    """Yuen's trimmed-mean paired t-test."""
    differences = data1 - data2
    n = len(differences)
    g = int(np.floor(trim * n))
    if n - 2 * g < 2:
        return None
    trimmed_diffs = np.sort(differences)[g: n - g]
    h = n - 2 * g
    tm = float(np.mean(trimmed_diffs))
    # Winsorized variance
    win = np.clip(differences, differences[np.argsort(differences)[g]], differences[np.argsort(differences)[n - g - 1]])
    sw2 = float(np.sum((win - np.mean(win)) ** 2) / (h * (h - 1)))
    se = float(np.sqrt(sw2))
    if se == 0:
        return None
    t_stat = float(tm / se)
    df = float(h - 1)
    p_val = float(2 * t.sf(abs(t_stat), df))
    return {
        "trimmed_mean_diff": tm,
        "t_statistic": t_stat,
        "df": df,
        "p_value": p_val,
        "trim_proportion": trim,
    }


def generate_plot(data1, data2, differences, mean_diff, t_stat, df, var1, var2, alternative='two-sided'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Boxplot
    sns.boxplot(data=[data1, data2], ax=axes[0, 0], palette=['#5B9BD5', '#F4A582'])
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels([var1, var2])
    axes[0, 0].set_title('Paired Variables Distribution', fontsize=12, fontweight='bold')

    # Plot 2: Differences histogram
    sns.histplot(differences, ax=axes[0, 1], color='#5B9BD5', kde=True)
    axes[0, 1].axvline(0, color='black', linestyle='--', label='No difference')
    axes[0, 1].axvline(mean_diff, color='red', linestyle='--', label=f'Mean diff = {mean_diff:.2f}')
    axes[0, 1].set_title('Distribution of Differences', fontsize=12, fontweight='bold')
    axes[0, 1].legend()

    # Plot 3: T-distribution
    if df > 0 and np.isfinite(df):
        x = np.linspace(-4, 4, 500)
        y = t.pdf(x, df)
        axes[1, 0].plot(x, y, label=f't-distribution (df={df})', color='#5B9BD5')
        axes[1, 0].axvline(t_stat, color='red', linestyle='--', label=f't-stat = {t_stat:.2f}')
        if alternative == 'two-sided':
            shade = (x >= abs(t_stat)) | (x <= -abs(t_stat))
            title_suffix = '(two-sided)'
        elif alternative == 'greater':
            shade = x >= t_stat
            title_suffix = '(right-tailed)'
        else:  # less
            shade = x <= t_stat
            title_suffix = '(left-tailed)'

        axes[1, 0].fill_between(x, 0, y, where=shade, color='red', alpha=0.3)
        axes[1, 0].set_title(f'Test Statistic on t-Distribution {title_suffix}', fontsize=12, fontweight='bold')
        axes[1, 0].legend()

    # Plot 4: Q-Q plot
    stats.probplot(differences, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Differences', fontsize=12, fontweight='bold')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


@router.post("/paired-ttest")
def paired_t_test(req: PairedTTestRequest):
    try:
        data = req.data
        params = req.params

        variable1 = params.get('variable1')
        variable2 = params.get('variable2')
        alternative = params.get('alternative', 'two-sided')
        alpha = float(params.get('alpha', 0.05))

        if not variable1 or not variable2:
            raise HTTPException(status_code=400, detail="Both variable1 and variable2 are required")

        # Extract and clean data
        dropped_rows = []
        data1_list = []
        data2_list = []

        for idx, row in enumerate(data):
            val1 = row.get(variable1)
            val2 = row.get(variable2)
            if val1 is None or val1 == '' or val2 is None or val2 == '':
                dropped_rows.append(idx)
            else:
                try:
                    data1_list.append(float(val1))
                    data2_list.append(float(val2))
                except (ValueError, TypeError):
                    dropped_rows.append(idx)

        data1 = np.array(data1_list)
        data2 = np.array(data2_list)

        if len(data1) < 2:
            raise HTTPException(status_code=400, detail="Not enough complete pairs (minimum 2 required)")

        # Calculate differences
        differences = data1 - data2

        # Descriptive statistics
        n = int(len(differences))
        mean_diff = float(np.mean(differences))
        std_diff = float(np.std(differences, ddof=1))
        se_diff = float(std_diff / np.sqrt(n))

        # Normality test on differences
        normality_test = None
        if n >= 3:
            stat, p = shapiro(differences)
            normality_test = {
                'differences': {
                    'statistic': float(stat),
                    'p_value': float(p),
                    'assumption_met': bool(p > alpha)
                }
            }

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)
        t_stat = float(t_stat)
        p_value = float(p_value)
        df = int(n - 1)

        # Confidence interval
        ci_lower, ci_upper = t.interval(1 - alpha, df, loc=mean_diff, scale=se_diff)
        ci_lower = float(ci_lower)
        ci_upper = float(ci_upper)

        # Effect sizes
        cohens_dz = float(mean_diff / std_diff) if std_diff > 0 else 0.0
        hedges_g = compute_hedges_g(mean_diff, std_diff, n)
        effect_interp = get_effect_size_interpretation(cohens_dz)

        significant = bool(p_value < alpha)

        # Descriptives
        mean1, mean2 = float(np.mean(data1)), float(np.mean(data2))
        std1, std2 = float(np.std(data1, ddof=1)), float(np.std(data2, ddof=1))

        # Skewness / kurtosis of differences
        diffs_skewness = float(stats.skew(differences))
        diffs_kurtosis = float(stats.kurtosis(differences))

        descriptives = {
            variable1: {
                "n": n,
                "mean": mean1,
                "std_dev": std1,
                "se_mean": float(std1 / np.sqrt(n)),
                "skewness": float(stats.skew(data1)),
                "kurtosis": float(stats.kurtosis(data1)),
            },
            variable2: {
                "n": n,
                "mean": mean2,
                "std_dev": std2,
                "se_mean": float(std2 / np.sqrt(n)),
                "skewness": float(stats.skew(data2)),
                "kurtosis": float(stats.kurtosis(data2)),
            },
            "differences": {
                "n": n,
                "mean": mean_diff,
                "std_dev": std_diff,
                "se_mean": se_diff,
                "skewness": diffs_skewness,
                "kurtosis": diffs_kurtosis,
            }
        }

        # Pair correlation
        pair_correlation = compute_pair_correlation(data1, data2)

        # Bootstrap
        bootstrap = compute_bootstrap(differences, n_bootstrap=5000, alpha=alpha)

        # Permutation
        permutation = compute_permutation(differences, n_permutations=10000)

        # Yuen trimmed t
        yuen_trimmed_t = compute_yuen_trimmed_t(data1, data2, trim=0.2)

        # Stability warnings
        stability_warnings = []
        if n < 10:
            stability_warnings.append("Small sample size (n < 10) — interpret results with caution.")
        if abs(diffs_skewness) > 2:
            stability_warnings.append("Differences are highly skewed — consider robust methods.")
        if not normality_test or not normality_test.get('differences', {}).get('assumption_met', True):
            stability_warnings.append("Normality assumption violated — bootstrap / Yuen results recommended.")

        # Interpretation
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        sig_text = "statistically significant" if significant else "not statistically significant"

        interpretation = (
            f"A paired-samples t-test was conducted to compare '{variable1}' and '{variable2}'.\n\n"
            f"There was a {sig_text} difference between '{variable1}' (M={mean1:.2f}, SD={std1:.2f}) "
            f"and '{variable2}' (M={mean2:.2f}, SD={std2:.2f}), "
            f"t({df}) = {t_stat:.2f}, {p_text}.\n\n"
            f"The mean difference was {mean_diff:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}].\n\n"
            f"Cohen's dz = {cohens_dz:.3f} ({effect_interp}), Hedges' g = {hedges_g:.3f}."
        )

        # Generate plot
        plot = generate_plot(data1, data2, differences, mean_diff, t_stat, df, variable1, variable2, alternative)

        return {
            "results": {
                "test_type": "paired_samples",
                "variable1": variable1,
                "variable2": variable2,
                "n": n,
                "mean_diff": mean_diff,
                "se_diff": se_diff,
                "t_statistic": t_stat,
                "degrees_of_freedom": df,
                "p_value": p_value,
                "significant": significant,

                # ── Effect sizes ──────────────────────────────────────────
                "cohens_d": cohens_dz,          # legacy alias kept
                "cohens_dz": cohens_dz,         # primary field expected by frontend
                "hedges_g": hedges_g,
                "effect_size_interpretation": effect_interp,

                # ── Intervals ────────────────────────────────────────────
                "confidence_interval": [ci_lower, ci_upper],

                # ── Assumption checks ─────────────────────────────────────
                "normality_test": normality_test,

                # ── Additional diagnostics ────────────────────────────────
                "differences_skewness": diffs_skewness,
                "differences_kurtosis": diffs_kurtosis,
                "stability_warnings": stability_warnings,

                # ── Robust / non-parametric methods ──────────────────────
                "pair_correlation": pair_correlation,
                "bootstrap": bootstrap,
                "permutation": permutation,
                "yuen_trimmed_t": yuen_trimmed_t,

                # ── Descriptives ─────────────────────────────────────────
                "descriptives": descriptives,

                # ── Metadata ─────────────────────────────────────────────
                "interpretation": interpretation,
                "dropped_rows": dropped_rows,
                "n_dropped": int(len(dropped_rows)),
            },
            "plot": plot
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
