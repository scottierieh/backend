"""
Hierarchical Bayes Multinomial Logit (HB-MNL) — shared individual-level
choice-model estimator used by CBC-HB (conjoint_hb.py) and ACBC
(acbc_analysis.py).

Unlike conjoint_analysis.py's aggregate conditional logit (one utility
vector for the whole sample), HB-MNL estimates one utility vector PER
RESPONDENT, shrunk toward a population mean/covariance that is estimated
jointly. That per-respondent heterogeneity is HB's actual value-add for
conjoint work — two respondents who both "prefer brand A on average" can
still have very different price sensitivity, and HB recovers that instead
of averaging it away.

Algorithm — the standard "Sawtooth-style" CBC/HB scheme (Allenby & Rossi,
1998; see Sawtooth Software's CBC/HB Technical Paper for the reference
write-up), a hybrid Gibbs sampler:
  (a) Metropolis-Hastings random-walk draw of each respondent's own
      part-worths beta_r, conditional on the current population mean mu and
      covariance Sigma (their prior) and that respondent's own choice data
      (their likelihood).
  (b) Closed-form Normal-Inverse-Wishart Gibbs draw of (mu, Sigma),
      conditional on all respondents' current beta_r.
Iterate (a)+(b); after a `tune`-iteration burn-in, `draws` further iterations
are kept as the posterior sample.

Deliberately pure numpy/scipy — no PyMC/Stan/JAX. Those need a C compiler or
GPU-oriented backend that isn't guaranteed inside the Cloud Run image this
API already runs in; this repo's requirements.txt is otherwise a plain
numpy/scipy/scikit-learn stack, so this file ships with zero new
dependencies (scipy.stats.invwishart has existed since SciPy 0.17).

Performance note: this is a Python-level double loop (iterations ×
respondents), so wall-clock scales roughly with
R × (tune + draws) × avg_tasks_per_respondent. That's fine for the
draws/tune=1000 defaults CBC-HB ships with with a few hundred respondents
(the current template's typical sample size) — seconds to low tens of
seconds — but this endpoint does real MCMC synchronously in an HTTP
request, so pushing draws/tune much higher on a very large sample will be
slow. No async job queue exists for this API yet; if that becomes a real
usage pattern, that's the next thing to add, not a change to this file.
"""
import numpy as np
import pandas as pd
from scipy.stats import invwishart
from typing import List, Dict, Any

from conjoint_analysis import (
    parse_conjoint_data, compute_partworths_and_importance, safe_float, _to_native,
)


def _resp_loglik(beta_r, X, y, group_row_idx, groups_for_resp):
    """Sum of conditional-logit log-likelihood across one respondent's own choice tasks."""
    ll = 0.0
    v_all = X @ beta_r
    for g in groups_for_resp:
        idx = group_row_idx[g]
        v = v_all[idx]
        y_g = y[idx]
        m = np.max(v)
        ll += np.sum(y_g * (v - (m + np.log(np.sum(np.exp(v - m))))))
    return ll


def _split_rhat(chain_draws: np.ndarray) -> float:
    """Gelman-Rubin R-hat for one scalar parameter, given draws shaped (chains, n_draws).

    Each chain is split in half (giving 2*chains "chains" of n/2 draws) before
    computing the classic between/within-chain variance ratio — the standard
    "split R-hat" refinement, which also catches within-chain non-stationarity
    that plain R-hat (whole chains only) can miss.
    """
    m, n = chain_draws.shape
    half = n // 2
    if half < 2:
        return float('nan')
    split = chain_draws[:, :2 * half].reshape(m * 2, half)
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    W = variances.mean()
    if W <= 0:
        return 1.0
    B = half * means.var(ddof=1)
    var_hat = ((half - 1) / half) * W + B / half
    return float(np.sqrt(var_hat / W))


def run_hb_mnl(
    data: List[Dict[str, Any]],
    attribute_cols: List[str],
    resp_col: str,
    task_col: str,
    choice_col: str,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    none_option: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    df, design_cols, attribute_map, ref_levels, data_summary = parse_conjoint_data(
        raw_data=data,
        response_col=choice_col,
        task_col=task_col,
        resp_col=resp_col,
        alt_col=None,
        attribute_cols=attribute_cols,
        none_option=none_option,
    )

    X_all = df[design_cols].values.astype(np.float64)
    y_all = df[choice_col].values.astype(np.float64)
    K = len(design_cols)

    respondents = df[resp_col].unique().tolist()
    R = len(respondents)
    if R < 5:
        raise ValueError("Hierarchical Bayes needs at least 5 respondents to estimate a population distribution.")

    groups = df['_choice_id'].values
    unique_groups = pd.unique(groups)
    group_row_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    group_resp = {g: df[resp_col].values[group_row_idx[g][0]] for g in unique_groups}
    resp_groups = {r: [g for g in unique_groups if group_resp[g] == r] for r in respondents}

    # ── Weakly-informative priors ──
    # mu ~ Normal(0, 1/k0) per component (k0 small = vague); Sigma ~ Inverse-Wishart(nu0, S0).
    nu0 = K + 2
    S0 = np.eye(K) * 1.0
    k0 = 0.01
    mu0 = np.zeros(K)

    def run_chain(chain_seed: int):
        rng = np.random.default_rng(chain_seed)
        mu = np.zeros(K)
        Sigma = np.eye(K) * 2.0
        beta_r = {r: mu.copy() for r in respondents}
        step = np.full(R, 0.3)
        accept = np.zeros(R)
        n_iter = tune + draws
        mu_draws = np.zeros((draws, K))
        beta_draws = np.zeros((draws, R, K))

        for it in range(n_iter):
            Sigma_inv = np.linalg.inv(Sigma)

            # (a) respondent-level RW-Metropolis
            for i, r in enumerate(respondents):
                cur = beta_r[r]
                prop = cur + rng.normal(0.0, step[i], size=K)
                d_cur, d_prop = cur - mu, prop - mu
                cur_lp = _resp_loglik(cur, X_all, y_all, group_row_idx, resp_groups[r]) - 0.5 * d_cur @ Sigma_inv @ d_cur
                prop_lp = _resp_loglik(prop, X_all, y_all, group_row_idx, resp_groups[r]) - 0.5 * d_prop @ Sigma_inv @ d_prop
                if np.log(rng.random()) < (prop_lp - cur_lp):
                    beta_r[r] = prop
                    accept[i] += 1

            # crude step-size adaptation during burn-in only (targets ~30% acceptance)
            if it < tune and (it + 1) % 50 == 0:
                acc_rate = accept / 50.0
                step *= np.where(acc_rate > 0.35, 1.2, np.where(acc_rate < 0.15, 0.8, 1.0))
                accept[:] = 0.0

            # (b) population (mu, Sigma) | all beta_r — Normal-Inverse-Wishart Gibbs step
            B = np.array([beta_r[r] for r in respondents])
            beta_bar = B.mean(axis=0)
            mu_n = (k0 * mu0 + R * beta_bar) / (k0 + R)
            kn = k0 + R
            centered = B - beta_bar
            Sn = S0 + centered.T @ centered + (k0 * R / (k0 + R)) * np.outer(beta_bar - mu0, beta_bar - mu0)
            Sigma = invwishart.rvs(df=nu0 + R, scale=Sn, random_state=rng)
            mu = rng.multivariate_normal(mu_n, Sigma / kn)

            if it >= tune:
                mu_draws[it - tune] = mu
                beta_draws[it - tune] = B

        return mu_draws, beta_draws

    chain_mu, chain_beta = [], []
    for c in range(max(chains, 1)):
        mu_d, beta_d = run_chain(seed + c * 10_007)
        chain_mu.append(mu_d)
        chain_beta.append(beta_d)

    # ── Convergence: split R-hat per population-mean component, across chains ──
    if len(chain_mu) >= 2:
        stacked = np.stack(chain_mu, axis=0)  # (chains, draws, K)
        max_rhat = max(_split_rhat(stacked[:, :, k]) for k in range(K))
    else:
        max_rhat = float('nan')  # R-hat is undefined with a single chain

    # ── Posterior summaries ──
    mu_all = np.concatenate(chain_mu, axis=0)          # (chains*draws, K)
    mu_hat = mu_all.mean(axis=0)
    mu_se = mu_all.std(axis=0, ddof=1)
    z_values = np.divide(mu_hat, mu_se, out=np.zeros_like(mu_hat), where=mu_se > 1e-12)
    from scipy.stats import norm as _norm
    p_values = 2 * (1 - _norm.cdf(np.abs(z_values)))

    # Per-respondent posterior mean part-worths (the actual HB deliverable).
    beta_all = np.concatenate(chain_beta, axis=0)      # (chains*draws, R, K)
    beta_hat = beta_all.mean(axis=0)                   # (R, K)

    # ── Reuse the aggregate model's part-worth/importance/chart shaping so the
    #    response matches ConjointResults exactly (see conjoint_analysis.py). ──
    partworths, importance = compute_partworths_and_importance(
        beta=mu_hat, se=mu_se, z_values=z_values, p_values=p_values,
        design_cols=design_cols, attribute_map=attribute_map, ref_levels=ref_levels,
    )

    pw_chart_data, coef_table, pw_flat = [], [], []
    for attr, pw_list in partworths.items():
        for pw in pw_list:
            row = {
                'attribute': attr, 'level': pw['level'],
                'coef': safe_float(pw['coef']), 'se': safe_float(pw['se']),
                'ci_lower': safe_float(pw['coef'] - 1.96 * pw['se']),
                'ci_upper': safe_float(pw['coef'] + 1.96 * pw['se']),
                'significant': pw['significant'], 'is_reference': pw['is_reference'],
                'label': f"{attr}: {pw['level']}",
            }
            pw_chart_data.append(row)
            coef_table.append({
                'attribute': attr, 'level': pw['level'],
                'coef': row['coef'], 'se': row['se'], 'z': safe_float(pw['z']), 'p': safe_float(pw['p']),
                'significant': pw['significant'], 'is_reference': pw['is_reference'],
                'ci_lower': row['ci_lower'], 'ci_upper': row['ci_upper'],
            })
            # `value` (not `coef`) — the key conjoint_hb.py/acbc_analysis.py's WTP
            # helpers read off this flat list.
            pw_flat.append({'attribute': attr, 'level': pw['level'], 'value': row['coef']})

    imp_chart_data = [{'attribute': a, 'importance': safe_float(v)} for a, v in sorted(importance.items(), key=lambda x: -x[1])]

    # ── Fit statistics at the posterior-mean population beta (a point-estimate
    #    summary for the same fields the aggregate model reports — not a
    #    Bayesian model-comparison metric, just interface parity). ──
    V_all = X_all @ mu_hat
    ll_model = 0.0
    for g in unique_groups:
        idx = group_row_idx[g]
        v = V_all[idx]
        y_g = y_all[idx]
        m = np.max(v)
        ll_model += np.sum(y_g * (v - (m + np.log(np.sum(np.exp(v - m))))))
    ll_null = 0.0
    for g in unique_groups:
        idx = group_row_idx[g]
        n_alts = len(idx)
        chosen = np.sum(y_all[idx])
        ll_null += chosen * np.log(1.0 / n_alts)
    n_choices = len(unique_groups)
    aic = 2 * K - 2 * ll_model
    bic = K * np.log(n_choices) - 2 * ll_model
    aicc = aic + (2 * K * (K + 1)) / max(n_choices - K - 1, 1)
    mcfadden_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0
    adj_mcfadden_r2 = 1 - ((ll_model - K) / ll_null) if ll_null != 0 else 0

    # Hit rate using each respondent's OWN posterior-mean part-worths — the
    # individual-level prediction HB is meant to improve over the aggregate model.
    hit, total = 0, 0
    for i, r in enumerate(respondents):
        v_r = X_all @ beta_hat[i]
        for g in resp_groups[r]:
            idx = group_row_idx[g]
            predicted = np.argmax(v_r[idx])
            actual = np.argmax(y_all[idx])
            hit += int(predicted == actual)
            total += 1
    hit_rate = hit / total if total > 0 else 0

    # ── Per-respondent part-worths (HB's actual deliverable) in the same
    #    {attribute: {level: coef}} shape as ACA's individualPartWorths, so
    #    downstream consumers (market-share simulators, bootstrap box-plots)
    #    work identically regardless of which estimator produced the data.
    #    Same reference-level-at-0 coding as the aggregate `partworths` above
    #    (not centered) — softmax/logit shares only depend on utility
    #    differences, so the choice of coding origin doesn't affect them.
    col_index = {col: i for i, col in enumerate(design_cols)}
    individual_part_worths = []
    for i in range(R):
        b = beta_hat[i]
        util_map = {}
        for attr, levels in attribute_map.items():
            ref = ref_levels[attr]
            util_map[attr] = {ref: 0.0}
            for lvl in levels:
                if lvl == ref:
                    continue
                col = f"{attr}__{lvl}"
                if col in col_index:
                    util_map[attr][lvl] = float(b[col_index[col]])
        individual_part_worths.append(util_map)

    results = {
        'n': R,
        'model_info': {
            'type': 'Hierarchical Bayes MNL',
            'n_params': K,
            'converged': bool(max_rhat == max_rhat and max_rhat < 1.1),  # NaN-safe (NaN != NaN)
        },
        'fit_statistics': {
            'log_likelihood': safe_float(ll_model),
            'log_likelihood_null': safe_float(ll_null),
            'aic': safe_float(aic),
            'aicc': safe_float(aicc),
            'bic': safe_float(bic),
            'mcfadden_r2': safe_float(mcfadden_r2),
            'adj_mcfadden_r2': safe_float(adj_mcfadden_r2),
            'hit_rate': safe_float(hit_rate),
        },
        'data_summary': data_summary,
        'attribute_map': attribute_map,
        'ref_levels': ref_levels,
        'partworths': {attr: [_to_native(pw) for pw in pw_list] for attr, pw_list in partworths.items()},
        'importance': {attr: safe_float(v) for attr, v in importance.items()},
        'simulation': None,
        'charts': {
            'partworth_data': _to_native(pw_chart_data),
            'importance_data': _to_native(imp_chart_data),
            'coefficient_table': _to_native(coef_table),
        },
        'convergence': {'maxRhat': safe_float(max_rhat) if max_rhat == max_rhat else None, 'chains': chains, 'draws': draws, 'tune': tune},
        # Flat list for the calling routers' WTP helpers (conjoint_hb.py / acbc_analysis.py).
        'partWorths': _to_native(pw_flat),
        'partWorthsFlat': _to_native(pw_flat),
        'individualPartWorths': _to_native(individual_part_worths),
    }
    return results
