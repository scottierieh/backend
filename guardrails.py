"""
Shared model-quality guardrails, computed after training and attached to the
response as a `guardrails` list. The frontend (model-lab-auto-compare.tsx /
src/lib/types/model-result.ts) has read this field and driven leakage-based
exclusion from Model Lab's ranking/comparison table since it was built, but no
backend script ever actually populated it -- every model always reported an
empty list, so the whole safety feature was silently a no-op. This module is
the first real implementation.

Each warning dict has the shape the frontend expects:
    {id, severity: 'critical'|'warning'|'info', title, detail, feature}

Known ids the frontend specifically keys off of (do not rename without also
updating EXCLUDE_IDS / LEAKAGE_EXCLUDE_IDS in model-lab-auto-compare.tsx and
src/lib/types/model-result.ts):
    leakage_suspect   (critical) -- a feature is almost perfectly correlated
                                     with the target
    target_duplicate  (critical) -- a feature is (near-)identical to the target
    perfect_score     (warning)  -- test score is suspiciously perfect
    class_imbalance   (warning)  -- smallest class is very underrepresented
    redundant_pair    (warning)  -- two features carry the same information
"""

import numpy as np
import pandas as pd


# Two features above this absolute correlation are treated as one piece of
# information. Deliberately below the .97 used against the target: that one is
# a leakage alarm and should be rare, this one is a modelling note.
REDUNDANT_PAIR_R = 0.95


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    return pd.Series(pd.factorize(s.astype(str))[0], index=s.index)


def compute_guardrails(X: pd.DataFrame, y: pd.Series, feature_names, task_type: str,
                        metrics: dict) -> list:
    """
    X: the model's feature matrix as actually used for training (pre- or
       post-encoding both work; only relative correlation with y matters).
    y: the raw target column (same length/index as X).
    feature_names: the feature column names to check for leakage.
    task_type: 'classification' or 'regression'.
    metrics: the model's own computed metrics dict (looks for 'accuracy' or 'r2').
    """
    warnings = []

    try:
        y_num = _to_numeric_series(y)
        y_std = float(y_num.std())
        for col in feature_names:
            if col not in X.columns:
                continue
            col_vals = _to_numeric_series(X[col])
            if col_vals.std() == 0 or y_std == 0:
                continue
            # Exact/near-exact duplicate of the target (stronger check than correlation
            # alone -- catches e.g. a re-labeled copy of a binary target).
            if len(col_vals) == len(y_num) and (col_vals.values == y_num.values).mean() > 0.999:
                warnings.append({
                    'id': 'target_duplicate',
                    'severity': 'critical',
                    'title': 'Feature duplicates the target',
                    'detail': f"'{col}' is nearly identical to the target column — almost certainly a copy or direct derivative of it. Remove it before trusting this model.",
                    'feature': col,
                })
                continue
            corr = float(np.corrcoef(col_vals.values, y_num.values)[0, 1])
            if not np.isnan(corr) and abs(corr) > 0.97:
                warnings.append({
                    'id': 'leakage_suspect',
                    'severity': 'critical',
                    'title': 'Possible data leakage',
                    'detail': f"'{col}' correlates {corr:.3f} with the target — check whether it's computed from, or only known after, the target (a common source of unrealistically high scores).",
                    'feature': col,
                })
    except Exception:
        pass

    # Two features that say the same thing. The check above only looks at each
    # feature against the target, which is blind to this: a column and its own
    # restatement (a signup date beside a tenure in months, a price beside a
    # log price) both pass it, and then permutation importance reports both as
    # unimportant because shuffling either one leaves the other to answer. The
    # pair is reported, not removed -- which of the two to keep is the caller's
    # call, and the importance step uses the pairs to measure them together.
    try:
        numeric = {}
        for col in feature_names:
            if col not in X.columns:
                continue
            vals = _to_numeric_series(X[col])
            if vals.std() == 0 or vals.isna().all():
                continue
            numeric[col] = vals
        names = list(numeric)
        seen = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                r = float(np.corrcoef(numeric[a].values, numeric[b].values)[0, 1])
                if np.isnan(r) or abs(r) <= REDUNDANT_PAIR_R:
                    continue
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                warnings.append({
                    'id': 'redundant_pair',
                    'severity': 'warning',
                    'title': 'Two features carry the same information',
                    'detail': (
                        f"'{a}' and '{b}' correlate {r:.3f} with each other. Importance "
                        f"splits between them and both look smaller than they are; "
                        f"consider keeping one."
                    ),
                    'feature': a,
                    'pair': [a, b],
                    'r': round(r, 4),
                })
    except Exception:
        pass

    try:
        score = metrics.get('accuracy') if task_type == 'classification' else metrics.get('r2')
        if score is not None and score >= 0.999:
            warnings.append({
                'id': 'perfect_score',
                'severity': 'warning',
                'title': 'Suspiciously perfect score',
                'detail': f"Test score is {score:.3f} — verify there's no leakage or an unrealistically easy/synthetic dataset before trusting this result.",
                'feature': None,
            })
    except Exception:
        pass

    try:
        if task_type == 'classification':
            counts = y.value_counts(normalize=True)
            min_frac = float(counts.min())
            if min_frac < 0.05:
                warnings.append({
                    'id': 'class_imbalance',
                    'severity': 'warning',
                    'title': 'Class imbalance',
                    'detail': f"The smallest class is only {min_frac:.1%} of the data — accuracy alone can be misleading here; weigh F1/precision-recall more heavily.",
                    'feature': None,
                })
    except Exception:
        pass

    return warnings
