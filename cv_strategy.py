"""
cv_strategy.py — one shared cross-validation module for every Model Lab script.

Before this, each *_analysis.py rolled its own cross_val_score with slightly
different splitters, variable names, and (for regression) shuffle behavior — so
CV was copy-pasted ~13 times and couldn't be improved in one place. This module
centralizes the splitter choice and scoring so:

  1. every model scores CV the same, reproducible way, and
  2. a future time-series / grouped split is added HERE once, not in 13 files.

Scripts keep their own local variable names; they just call run_cv(...) and read
back the same {cv_mean, cv_std, cv_scores, cv_folds} keys they emitted before.

The `time_order` / `groups` arguments are the built-in extension point for the
statistically-honest splits (random KFold is optimistic for time-series and
grouped data). They default off, so today's behavior is unchanged until a caller
opts in.
"""

import numpy as np
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit, GroupKFold,
)


def make_cv_splitter(task_type, cv_folds=5, random_state=42, *, time_order=False, groups=None):
    """Pick the right CV splitter. Random KFold/StratifiedKFold by default; a
    time-ordered or grouped splitter when the caller says the data has that
    structure (so scores aren't optimistic from leaking across time/group)."""
    if time_order:
        return TimeSeriesSplit(n_splits=cv_folds)
    if groups is not None:
        return GroupKFold(n_splits=cv_folds)
    if task_type == 'classification':
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def run_cv(estimator, X, y, task_type, cv_folds=5, random_state=42, scoring=None,
           *, time_order=False, groups=None):
    """Cross-validate `estimator` on (X, y) and return the standard result dict.

    estimator : any fitted-or-unfitted sklearn-compatible model (or Pipeline).
    X, y      : feature matrix and the target already in the form the model expects
                (label-encoded for classification, numeric for regression) — the
                caller passes whatever its local variables are named.
    task_type : 'classification' | 'regression'.
    Returns   : {cv_mean, cv_std, cv_scores, cv_folds, cv_scoring, cv_strategy}.
                Superset of the keys the scripts emitted before, so it's drop-in.
    """
    scoring = scoring or ('accuracy' if task_type == 'classification' else 'r2')
    splitter = make_cv_splitter(task_type, cv_folds, random_state,
                                time_order=time_order, groups=groups)
    kwargs = {'groups': groups} if groups is not None else {}
    scores = np.asarray(
        cross_val_score(estimator, X, y, cv=splitter, scoring=scoring, **kwargs),
        dtype=float,
    )
    return {
        'cv_scores': [float(s) for s in scores],
        'cv_mean': float(np.mean(scores)),
        'cv_std': float(np.std(scores)),
        'cv_folds': int(cv_folds),
        'cv_scoring': scoring,
        'cv_strategy': type(splitter).__name__,
    }
