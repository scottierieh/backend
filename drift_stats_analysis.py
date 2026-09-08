"""
drift_stats_analysis.py — real PSI / KS-test drift statistics, per feature
(route: /api/analysis/drift-stats).

CLI-script contract (like every *_analysis.py here): read one JSON object
from stdin, print one JSON object to stdout; on error print {"error": ...}
to stderr and exit(1).

THE HONEST LIMIT ON WHAT THIS CAN COMPUTE: a real KS test or PSI needs
actual per-instance live values, not just their mean. This endpoint does
the correct statistics once given those values — it does not, and cannot,
manufacture them. If the only thing being fed in is a running mean per
predict call (which is what Model Lab's PredictionLog.inputSummary stores
today — see predict-dialog.tsx), the result here is still a real
computation over whatever series it was given, but it is only as
meaningful as that series: a KS test over per-call MEANS is a test of
"has the average of each call's rows shifted", not "has the underlying
row-level distribution shifted". Getting the second, sharper question
requires logging individual row values (or at least a running histogram)
per predict call instead of just their mean — a frontend/schema decision
this script doesn't make on its own.

Numeric features: the baseline is summarized as Normal(baseline_mean,
baseline_std) (that's what featureBaseline already stores — see
models_api.py's _compute_feature_baseline). A one-sample KS test compares
the live values against that reference normal directly. PSI bins the same
reference normal into deciles (via its own inverse-CDF) as the "expected"
distribution and compares live values' bin proportions against it.

Categorical features: PSI only (KS doesn't apply to unordered categories) —
observed frequencies in live_values vs baseline_frequencies.

PSI convention used for `drifted` (standard industry thresholds): < 0.1 no
material shift, 0.1-0.2 moderate, > 0.2 significant — flagged at > 0.2.
"""

import sys
import json
import numpy as np
from scipy import stats

from analysis_common import _to_native_type

PSI_DRIFT_THRESHOLD = 0.2
MIN_LIVE_SAMPLES = 5


def _psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    # Standard PSI formula; a bin with 0% on either side is nudged to a
    # small epsilon so a single empty bin doesn't blow up to +/-inf.
    eps = 1e-4
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _numeric_drift(mean: float, std: float, live_values: list) -> dict:
    live = np.asarray([v for v in live_values if v is not None], dtype=float)
    live = live[np.isfinite(live)]
    n = len(live)
    if n < MIN_LIVE_SAMPLES:
        return {'type': 'numeric', 'available': False, 'reason': f'need >= {MIN_LIVE_SAMPLES} live values, got {n}'}
    if std <= 0:
        std = 1e-6  # a constant baseline feature — avoid a divide-by-zero, not a meaningful case anyway

    # A frozen distribution's .cdf, not the 'norm' + args=(mean, std) string
    # form — the latter dispatches internally in a way that breaks on at
    # least one scipy release actually seen in testing; this form is
    # unambiguous regardless of scipy version.
    ks_stat, ks_pvalue = stats.kstest(live, stats.norm(loc=mean, scale=std).cdf)

    # Deciles of the reference Normal(mean, std) as bin edges -> each bin
    # holds exactly 10% of the baseline ("expected") by construction.
    edges = stats.norm.ppf(np.linspace(0, 1, 11), loc=mean, scale=std)
    edges[0], edges[-1] = -np.inf, np.inf
    expected_pct = np.full(10, 0.1)
    actual_counts, _ = np.histogram(live, bins=edges)
    actual_pct = actual_counts / n
    psi = _psi(expected_pct, actual_pct)

    return {
        'type': 'numeric',
        'available': True,
        'n_samples': n,
        'baseline_mean': _to_native_type(mean),
        'baseline_std': _to_native_type(std),
        'live_mean': _to_native_type(float(live.mean())),
        'live_std': _to_native_type(float(live.std())),
        'ks_stat': _to_native_type(float(ks_stat)),
        'ks_pvalue': _to_native_type(float(ks_pvalue)),
        'psi': _to_native_type(psi),
        'drifted': bool(psi > PSI_DRIFT_THRESHOLD),
    }


def _categorical_drift(baseline_frequencies: dict, live_values: list) -> dict:
    live = [v for v in live_values if v is not None]
    n = len(live)
    if n < MIN_LIVE_SAMPLES:
        return {'type': 'categorical', 'available': False, 'reason': f'need >= {MIN_LIVE_SAMPLES} live values, got {n}'}

    categories = sorted(set(baseline_frequencies.keys()) | set(str(v) for v in live))
    expected_pct = np.array([max(baseline_frequencies.get(c, 0.0), 1e-4) for c in categories])
    expected_pct = expected_pct / expected_pct.sum()
    counts = {c: 0 for c in categories}
    for v in live:
        counts[str(v)] = counts.get(str(v), 0) + 1
    actual_pct = np.array([counts[c] / n for c in categories])
    psi = _psi(expected_pct, actual_pct)

    return {
        'type': 'categorical',
        'available': True,
        'n_samples': n,
        'live_frequencies': {c: _to_native_type(float(counts[c] / n)) for c in categories},
        'psi': _to_native_type(psi),
        'drifted': bool(psi > PSI_DRIFT_THRESHOLD),
    }


def main():
    try:
        payload = json.load(sys.stdin)
        features = payload.get('features')
        if not features or not isinstance(features, dict):
            raise ValueError("Missing 'features' — expected {feature_name: {type, baseline_..., live_values}}")

        out = {}
        for name, spec in features.items():
            try:
                ftype = spec.get('type')
                live_values = spec.get('live_values') or []
                if ftype == 'numeric':
                    out[name] = _numeric_drift(
                        float(spec.get('baseline_mean', 0.0)),
                        float(spec.get('baseline_std', 0.0)),
                        live_values,
                    )
                elif ftype == 'categorical':
                    out[name] = _categorical_drift(spec.get('baseline_frequencies') or {}, live_values)
                else:
                    out[name] = {'type': ftype, 'available': False, 'reason': f"unknown type {ftype!r}"}
            except Exception as e:
                out[name] = {'available': False, 'reason': str(e)}

        print(json.dumps(out, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
