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
           *, time_order=False, groups=None, sample_weight=None):
    """Cross-validate `estimator` on (X, y) and return the standard result dict.

    estimator : any fitted-or-unfitted sklearn-compatible model (or Pipeline).
    X, y      : feature matrix and the target already in the form the model expects
                (label-encoded for classification, numeric for regression) — the
                caller passes whatever its local variables are named.
    task_type : 'classification' | 'regression'.
    sample_weight : per-row weights for estimators whose only route to class
                balancing is fit(sample_weight=). Indexed per fold by sklearn,
                so each fold weights its own rows. None for the rest, which
                carry the weighting as a parameter instead.
    Returns   : {cv_mean, cv_std, cv_scores, cv_folds, cv_scoring, cv_strategy}.
                Superset of the keys the scripts emitted before, so it's drop-in.
    """
    default_scoring = 'accuracy' if task_type == 'classification' else 'r2'
    requested = scoring or default_scoring
    scoring = requested

    # A binary-only scorer on a target that is not binary does NOT raise once
    # the labels are integers: average_precision takes pos_label=1 and quietly
    # scores class 1 against the rest. On a three-class target that returns
    # about 0.33 and it reaches the board labelled PR-AUC, which is a number
    # about one class presented as a number about the model. Nothing throws,
    # so the fallback below would never fire -- this has to be checked, not
    # caught.
    _binary_only = {'average_precision', 'roc_auc', 'f1', 'precision', 'recall'}
    _pre_reason = None
    if scoring in _binary_only and len(np.unique(np.asarray(y))) != 2:
        _pre_reason = (f'{scoring} is a binary metric and this target has '
                       f'{len(np.unique(np.asarray(y)))} classes')
        scoring = default_scoring

    splitter = make_cv_splitter(task_type, cv_folds, random_state,
                                time_order=time_order, groups=groups)
    kwargs = {'groups': groups} if groups is not None else {}
    # Class weighting reaches most estimators as a parameter they re-derive per
    # fit, which cross-validation then gets for free. Three of them
    # (AdaBoost, GBM, Naive Bayes) have no such parameter and take weights at
    # fit time instead, and without this those models would be weighted in the
    # reported metrics and unweighted in the CV beside them -- two numbers
    # about two different fits, printed as if they described one.
    #
    # `params` rather than the deprecated `fit_params`, and sklearn indexes it
    # per fold, so a fold gets its own rows' weights rather than all of them.
    if sample_weight is not None:
        # A Pipeline does not take fit parameters for itself -- it routes them
        # to a named step, and an unprefixed sample_weight raises rather than
        # being ignored, so every fold fails and the CV comes back as an
        # error. The estimator is the last step by construction here
        # (cv_pipeline puts the preprocessing before it).
        key = 'sample_weight'
        steps = getattr(estimator, 'steps', None)
        if steps:
            key = f'{steps[-1][0]}__sample_weight'
        kwargs['params'] = {
            **kwargs.pop('params', {}),
            key: np.asarray(sample_weight),
        }
    # A caller asks for the metric its leaderboard ranks on, and not every
    # estimator can produce every metric: average_precision needs
    # predict_proba and a binary target, roc_auc the same. Letting that raise
    # would lose the cross-validation entirely -- and CV is what the shortlist
    # is chosen on -- so an unusable scorer falls back to the default and the
    # response says which number it actually is. A model silently scored on a
    # different metric than its neighbours is the failure this whole field
    # exists to prevent.
    fallback_reason = _pre_reason
    try:
        scores = np.asarray(
            cross_val_score(estimator, X, y, cv=splitter, scoring=scoring, **kwargs),
            dtype=float,
        )
        if not np.isfinite(scores).any():
            raise ValueError(f'every fold returned a non-finite {scoring}')
    except Exception as e:
        # Already on the default -- there is nothing left to fall back to, and
        # a silent empty CV would be worse than the error. (Comparing
        # `requested` here instead would retry the default with the default
        # when the pre-check above had already switched to it.)
        if scoring == default_scoring:
            raise
        fallback_reason = f'{type(e).__name__}: {e}'[:200]
        scoring = default_scoring
        scores = np.asarray(
            cross_val_score(estimator, X, y, cv=splitter, scoring=scoring, **kwargs),
            dtype=float,
        )

    out = {
        'cv_scores': [float(s) for s in scores],
        'cv_mean': float(np.mean(scores)),
        'cv_std': float(np.std(scores)),
        'cv_folds': int(cv_folds),
        'cv_scoring': scoring,
        'cv_strategy': type(splitter).__name__,
    }
    if fallback_reason is not None:
        out['cv_scoring_requested'] = requested
        out['cv_scoring_fallback'] = fallback_reason
    return out
