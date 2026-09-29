"""
tune_analysis.py — Model Lab "Tune the winner" backend (route: /api/analysis/tune).

After Auto Compare finds the top model(s), this re-tunes ONE model's
hyperparameters within a preset search budget and re-runs the leakage
guardrails so a higher score can't quietly hide leakage. The frontend panel
that called this (model-lab-tune-panel.tsx) was deleted with the old Model
Lab; this script is being brought back up to the current AutoML pipeline's
contract before a new panel is wired to it — see
docs/automl-hyperparameter-search.md for the request and what's still open.

Distinct from hyperparameter_tuning_analysis.py (the standalone wizard page):
this one speaks the AutoML "tune the winner" contract (model key + preset in,
best_score/improvement/guardrails out) and supports five of Auto Compare's
tunable candidates — XGBoost, Random Forest, GBM, LightGBM, CatBoost.

CLI-script contract (like every *_analysis.py here): read one JSON object from
stdin, print one JSON object to stdout; on error print {"error": ...} to stderr
and exit(1).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import (
    ParameterSampler, cross_val_score, KFold, StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CB = True
except Exception:
    _HAS_CB = False

# Reuse the shared leakage/imbalance/perfect-score guardrails (guardrails.py in
# repo root) — the same module wired into all 14 Auto Compare scripts, so the
# post-tuning re-check is identical to the pre-tuning one.
from guardrails import compute_guardrails

# Same train-only imputation/encoding every Auto Compare script now uses (see
# docs/automl-preprocessing-leakage.md) — this script had the same
# fit-on-everything-then-split leak (get_dummies before train_test_split) plus
# no imputation at all, so a missing input value crashed the whole search.
from analysis_common import balanced_weighting, leak_safe_prepare_onehot


def _to_native(o):
    # Plain Python scalars pass through untouched. They used to fall to the
    # str() below -- best_params came back as {"n_estimators": "300",
    # "max_depth": "None"}, every number a string and a null spelled out as
    # one. Harmless as a json.dumps `default=` (which is only called for what
    # json cannot serialize), but best_params calls this on every value.
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# TunePanel sends the Auto Compare def.route as `model`, mapped there to these keys:
#   'xgboost' -> 'xgboost', 'randomforest' -> 'random-forest', 'gradient-boosting' -> 'gbm'
# Accept those exact strings (plus a few natural aliases). Each entry builds a fresh
# default estimator (the baseline) and a search space for RandomizedSearchCV.
def _model_space(model_key, task_type):
    key = (model_key or '').lower()
    is_clf = task_type == 'classification'

    if key in ('random-forest', 'random_forest', 'randomforest', 'rf'):
        est = RandomForestClassifier(random_state=42) if is_clf else RandomForestRegressor(random_state=42)
        space = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
        }
        return est, space

    if key in ('gbm', 'gradient-boosting', 'gradient_boosting', 'gradientboosting'):
        est = GradientBoostingClassifier(random_state=42) if is_clf else GradientBoostingRegressor(random_state=42)
        space = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.7, 0.85, 1.0],
        }
        return est, space

    if key in ('xgboost', 'xgb'):
        if not _HAS_XGB:
            raise ValueError("XGBoost is not installed on the server")
        # n_jobs=1, not -1: RandomizedSearchCV below already parallelizes
        # across candidates/folds with its own n_jobs=-1. Nesting a second
        # n_jobs=-1 inside each candidate is fork+OpenMP oversubscription at
        # best and a genuine deadlock at worst — confirmed hanging (no error,
        # no output, still running past 300s) against the live Cloud Run
        # container on the very first deploy of this, though it never showed
        # up locally (Windows' spawn-based multiprocessing doesn't hit the
        # same fork/thread-pool interaction Linux does).
        common = dict(random_state=42, n_jobs=1, verbosity=0)
        est = (XGBClassifier(eval_metric='logloss', **common) if is_clf
               else XGBRegressor(**common))
        space = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 8],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'min_child_weight': [1, 3, 5],
        }
        return est, space

    if key in ('lightgbm', 'lgbm', 'lgb'):
        if not _HAS_LGB:
            raise ValueError("LightGBM is not installed on the server")
        # n_jobs=1 — see the XGBoost branch above for why nesting under
        # RandomizedSearchCV's own n_jobs=-1 is refused here.
        common = dict(random_state=42, n_jobs=1, verbose=-1)
        est = (lgb.LGBMClassifier(**common) if is_clf else lgb.LGBMRegressor(**common))
        space = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63, 127],
            'max_depth': [-1, 5, 10, 20],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
        }
        return est, space

    if key in ('catboost', 'cb'):
        if not _HAS_CB:
            raise ValueError("CatBoost is not installed on the server")
        # verbose=False (not just a low value) and allow_writing_files=False:
        # CatBoost's training log goes to stdout by default, which would
        # corrupt this script's one-JSON-line-on-stdout contract, and it
        # writes scratch files to disk by default, which a stateless Cloud
        # Run container has no lasting place for.
        # thread_count=1 for the same reason n_jobs is pinned to 1 above —
        # CatBoost's own default (all cores) would otherwise nest under
        # RandomizedSearchCV's n_jobs=-1 too.
        common = dict(random_state=42, verbose=False, allow_writing_files=False, thread_count=1)
        est = (CatBoostClassifier(**common) if is_clf else CatBoostRegressor(**common))
        space = {
            'iterations': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
        }
        return est, space

    raise ValueError(
        f"Unsupported model '{model_key}' — tunable models are "
        + ', '.join(e['key'] for e in tunable_models() if e['available'])
    )


# Which models this script can tune, and whether the library is actually on
# this server. The screen has to know BEFORE it draws anything: a "Tune"
# button on a row this script would reject with a ValueError is a button that
# only fails. Hard-coding the list in the frontend would drift the moment a
# model is added here, so it is served from the one place that decides it --
# `{"list_only": true}` answers with just this and reads no data.
_CATALOG = [
    ('random-forest', 'Random Forest', True,
     ('random_forest', 'randomforest', 'rf')),
    ('gbm', 'Gradient Boosting', True,
     ('gradient-boosting', 'gradient_boosting', 'gradientboosting')),
    ('xgboost', 'XGBoost', _HAS_XGB, ('xgb',)),
    ('lightgbm', 'LightGBM', _HAS_LGB, ('lgbm', 'lgb')),
    ('catboost', 'CatBoost', _HAS_CB, ('cb',)),
]


def tunable_models():
    return [
        {'key': key, 'label': label, 'available': bool(available), 'aliases': list(aliases)}
        for key, label, available, aliases in _CATALOG
    ]


# Preset -> search effort. RandomizedSearchCV has no native wall-clock budget, so the
# preset maps to (n_iter candidates, cv folds); the "~N min" the UI shows is the rough
# envelope these produce on typical Model Lab datasets. Kept well within the backend's
# 600s Cloud Run request timeout even for 'thorough'.
_PRESETS = {
    'fast':     {'n_iter': 15, 'cv': 3, 'label': 'Fast', 'budget_seconds': 60},
    'balanced': {'n_iter': 40, 'cv': 5, 'label': 'Balanced', 'budget_seconds': 180},
    'thorough': {'n_iter': 80, 'cv': 5, 'label': 'Thorough', 'budget_seconds': 420},
}

# The request's own ceiling, whatever a caller asks for. Cloud Run cuts the
# request at 600s and the reply is then lost along with the whole search, so a
# budget that could reach it is not a budget.
_MAX_BUDGET_SECONDS = 480


def _score_candidate(pipeline, params, X_raw, y, splitter, scoring, fit_params):
    """One candidate's cross-validated score. Runs in a worker process."""
    try:
        pipeline = clone(pipeline).set_params(**params)
        scores = cross_val_score(
            pipeline, X_raw, y, cv=splitter, scoring=scoring, n_jobs=1,
            params=fit_params or None, error_score=np.nan,
        )
        mean = float(np.nanmean(scores))
        return mean if np.isfinite(mean) else None
    except Exception:
        # A candidate that cannot be fitted is a dead point in the space, not
        # a failed search -- a max_features the data is too narrow for, say.
        # It scores nothing and the rest carry on.
        return None


def _budgeted_search(build_pipeline, space, X_raw, y, splitter, scoring,
                     n_iter, budget_seconds, fit_params, seed=42):
    """RandomizedSearchCV's sampling, under a wall-clock budget.

    RandomizedSearchCV has no wall-clock stop: `thorough` is 80 x 5 = 400 fits
    and on a large enough frame that runs past the 600s request timeout, which
    loses the whole search rather than returning the best of what it had. So
    the candidates are drawn the same way and evaluated in parallel chunks,
    with the clock read between chunks -- the answer is the best candidate
    actually scored, and the response says how many that was out of how many
    were planned.

    The stop is predictive: a chunk is started only if the time the last one
    took would still fit. Stopping after overrunning would make the budget a
    suggestion.
    """
    candidates = list(ParameterSampler(space, n_iter=n_iter, random_state=seed))
    workers = max(1, min(len(candidates), os.cpu_count() or 1))

    started = time.time()
    best_score, best_params = None, None
    evaluated, failed = 0, 0
    stopped_early = False
    chunk_seconds = 0.0

    for i in range(0, len(candidates), workers):
        elapsed = time.time() - started
        # After the first chunk we know roughly what one costs.
        if i and elapsed + chunk_seconds > budget_seconds:
            stopped_early = True
            break

        chunk = candidates[i:i + workers]
        chunk_started = time.time()
        scores = Parallel(n_jobs=len(chunk))(
            delayed(_score_candidate)(
                build_pipeline(), params, X_raw, y, splitter, scoring, fit_params)
            for params in chunk
        )
        chunk_seconds = time.time() - chunk_started

        for params, score in zip(chunk, scores):
            evaluated += 1
            if score is None:
                failed += 1
                continue
            if best_score is None or score > best_score:
                best_score, best_params = score, params

    if best_params is None:
        raise ValueError(
            'No candidate could be fitted on this data — the search space and the '
            'data disagree about something (too few rows for the folds, or a '
            'parameter no column supports).'
        )

    return {
        'best_params': best_params,
        'best_cv_score': best_score,
        'n_trials': evaluated,
        'n_candidates': len(candidates),
        'n_failed': failed,
        'stopped_early': stopped_early,
    }


def _detect_task_type(y: pd.Series) -> str:
    vals = y.dropna()
    if vals.empty:
        return 'classification'
    if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 20:
        return 'regression'
    return 'classification'


def main():
    try:
        payload = json.load(sys.stdin)

        # The screen asks this before it has anything to tune, to know which
        # rows may carry a Tune button at all. No data, no model, no search.
        if payload.get('list_only'):
            print(json.dumps({'tunable_models': tunable_models()}))
            return

        data = payload.get('data')
        target = payload.get('target_col') or payload.get('target')
        features = payload.get('feature_cols') or payload.get('features')
        model_key = payload.get('model') or payload.get('model_type')
        preset_key = (payload.get('preset') or 'balanced').lower()
        task_req = payload.get('task_type', 'auto')

        if not all([data, features, target, model_key]):
            raise ValueError("Missing data, features, target, or model")

        preset = _PRESETS.get(preset_key, _PRESETS['balanced'])

        # AutoML sends only split.train here (never the sealed holdout — see
        # docs/automl-hyperparameter-search.md §8); task type is decided on
        # the full target column, before any row is dropped, same as every
        # Auto Compare script.
        df = pd.DataFrame(data)
        task_type = task_req if task_req in ('classification', 'regression') else _detect_task_type(df[target])

        # Split BEFORE any imputation/one-hot statistic is computed, so those
        # values come from THIS split's train rows alone, and impute rather
        # than drop rows with missing features — previously any missing input
        # went straight to the estimator and RandomForest/GBM (unlike XGBoost)
        # raise on NaN, so tuning failed outright on data with any gaps at
        # all. See docs/automl-preprocessing-leakage.md.
        prep = leak_safe_prepare_onehot(df, features, target, task_type, test_size=0.25, random_state=42)
        X_tr_df, X_te_df = prep['X_train'], prep['X_test']
        y_tr_raw, y_te_raw = prep['y_train'], prep['y_test']
        row_counts = prep['row_counts']

        if task_type == 'regression':
            y_tr = y_tr_raw.to_numpy(dtype=float)
            y_te = y_te_raw.to_numpy(dtype=float)
        else:
            # Label-encode so XGBoost (which requires numeric labels) works
            # alongside sklearn. Fit on train+test together — a label
            # vocabulary, not a feature statistic, so this isn't the leak the
            # imputation/encoding above closes; see decision_tree_analysis.py
            # for the same reasoning.
            _le = LabelEncoder()
            _le.fit(pd.concat([y_tr_raw, y_te_raw]).astype(str))
            y_tr = _le.transform(y_tr_raw.astype(str))
            y_te = _le.transform(y_te_raw.astype(str))

        # AutoML sends the same metric its ranking policy actually ranks on
        # (PR-AUC for a rare-minority binary target, Macro F1 for multiclass,
        # RMSE for regression, ...) — optimizing for f1_weighted and then
        # ranking by a different metric was measuring one thing and reporting
        # another. No override -> unchanged default.
        scoring = payload.get('scoring') or ('f1_weighted' if task_type == 'classification' else 'r2')
        scorer = get_scorer(scoring)

        estimator, space = _model_space(model_key, task_type)

        # Baseline = the model's default hyperparameters on the same split, so
        # "before -> after" is an apples-to-apples comparison on identical
        # data — fit directly on the already-imputed/encoded train frame
        # (a single fit, not cross-validated, so there's no fold to leak
        # into).
        baseline_est, _ = _model_space(model_key, task_type)

        # The same balancing Auto Compare now applies, on both sides of the
        # before/after. Tuning the one model on the board that was fitted as
        # if the classes were even would put its gain and that difference in
        # the same number -- see docs/automl-class-imbalance.md.
        if task_type == 'classification':
            requested_weighting = payload.get('class_weight', 'balanced')
            baseline_est, baseline_fit_kwargs, class_weighting = balanced_weighting(
                baseline_est, y_tr, requested_weighting)
            estimator, search_fit_kwargs, _ = balanced_weighting(
                estimator, y_tr, requested_weighting)
        else:
            # Regression has no classes to balance. Reporting it as "turned
            # off" would describe a choice nobody made.
            baseline_fit_kwargs, search_fit_kwargs, class_weighting = {}, {}, None

        baseline_est.fit(X_tr_df, y_tr, **baseline_fit_kwargs)
        baseline_score = float(scorer(baseline_est, X_te_df, y_te))
        # A Pipeline routes fit parameters to a named step; unprefixed, its
        # fit raises rather than ignoring them, and every fold would fail.
        fit_params = ({'model__sample_weight': np.asarray(search_fit_kwargs['sample_weight'])}
                      if search_fit_kwargs.get('sample_weight') is not None else None)

        # The search itself DOES need to be leakage-safe per fold: each of
        # RandomizedSearchCV's CV folds must fit its own imputer/encoder on
        # that fold's train portion, not reuse the outer prep's. cv_pipeline
        # wraps the estimator in a fresh Pipeline(prep, model) each call, so
        # search.fit on the RAW (unimputed) train frame does exactly that —
        # see docs/automl-preprocessing-leakage.md §2.
        space_prefixed = {f'model__{k}': v for k, v in space.items()}

        # One splitter for every candidate, so they are compared on identical
        # folds -- a candidate that merely drew an easier split would
        # otherwise win on that.
        n_splits = preset['cv']
        if task_type == 'classification':
            smallest = int(np.min(np.bincount(y_tr))) if len(y_tr) else 0
            n_splits = max(2, min(n_splits, smallest))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        budget_seconds = payload.get('budget_seconds') or preset['budget_seconds']
        budget_seconds = max(10, min(float(budget_seconds), _MAX_BUDGET_SECONDS))

        started = time.time()
        found = _budgeted_search(
            lambda: prep['cv_pipeline'](clone(estimator)), space_prefixed,
            prep['X_train_raw'], y_tr, splitter, scoring,
            preset['n_iter'], budget_seconds, fit_params,
        )
        seconds_used = round(time.time() - started, 1)

        # Refit the winner on the whole raw train frame -- the search scored
        # it on folds, and the number reported next to the baseline has to
        # come from a model fitted on the same rows the baseline was.
        best_model = prep['cv_pipeline'](clone(estimator)).set_params(**found['best_params'])
        best_model.fit(prep['X_train_raw'], y_tr, **(fit_params or {}))
        tuned_score = float(scorer(best_model, prep['X_test_raw'], y_te))
        best_params = {
            (k[len('model__'):] if k.startswith('model__') else k): _to_native(v)
            for k, v in found['best_params'].items()
        }

        # Post-tuning guardrail re-check — identical logic to Auto Compare's pre-tuning
        # pass, on the raw features/target, so a leakage-inflated gain can't slip through.
        metrics = {'accuracy': tuned_score} if task_type == 'classification' else {'r2': tuned_score}
        guardrails = compute_guardrails(prep['X_train_raw'], y_tr_raw, features, task_type, metrics)

        response = {
            'model': model_key,
            'task_type': task_type,
            'scoring': scoring,
            'preset_label': preset['label'],
            'baseline_score': baseline_score,
            'best_score': tuned_score,
            'improvement': tuned_score - baseline_score,
            'best_params': best_params,
            'best_cv_score': found['best_cv_score'],
            'class_weighting': class_weighting,
            # "80 of 80 ran" and "the budget ran out at 43" are different
            # answers and the screen has to be able to tell them apart.
            'n_trials': found['n_trials'],
            'n_candidates': found['n_candidates'],
            'n_failed': found['n_failed'],
            'stopped_early': found['stopped_early'],
            'budget_seconds': budget_seconds,
            'seconds_used': seconds_used,
            'tunable_models': tunable_models(),
            'model_id': None,  # no model persistence yet — panel treats this as optional
            'guardrails': guardrails,
            'row_counts': row_counts,
            # best_score/baseline_score come from THIS script's own inner
            # split of whatever it was sent (never the sealed holdout) — not
            # the same rows the leaderboard's numbers are scored on. The
            # screen should read this pair as an improvement delta, not swap
            # it in as a final score; see docs/automl-hyperparameter-search.md §8.
            'scored_on': 'inner_split',
        }
        print(json.dumps(response, default=_to_native))

    except Exception as e:  # noqa: BLE001 — CLI contract: any failure -> stderr + exit(1)
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
