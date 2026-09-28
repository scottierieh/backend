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

import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
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
from analysis_common import leak_safe_prepare_onehot


def _to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
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
        f"Unsupported model '{model_key}' — tunable models are xgboost, random-forest, gbm, lightgbm, catboost"
    )


# Preset -> search effort. RandomizedSearchCV has no native wall-clock budget, so the
# preset maps to (n_iter candidates, cv folds); the "~N min" the UI shows is the rough
# envelope these produce on typical Model Lab datasets. Kept well within the backend's
# 600s Cloud Run request timeout even for 'thorough'.
_PRESETS = {
    'fast':     {'n_iter': 15, 'cv': 3, 'label': 'Fast'},
    'balanced': {'n_iter': 40, 'cv': 5, 'label': 'Balanced'},
    'thorough': {'n_iter': 80, 'cv': 5, 'label': 'Thorough'},
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
        baseline_est.fit(X_tr_df, y_tr)
        baseline_score = float(scorer(baseline_est, X_te_df, y_te))

        # The search itself DOES need to be leakage-safe per fold: each of
        # RandomizedSearchCV's CV folds must fit its own imputer/encoder on
        # that fold's train portion, not reuse the outer prep's. cv_pipeline
        # wraps the estimator in a fresh Pipeline(prep, model) each call, so
        # search.fit on the RAW (unimputed) train frame does exactly that —
        # see docs/automl-preprocessing-leakage.md §2.
        pipeline_estimator = prep['cv_pipeline'](estimator)
        space_prefixed = {f'model__{k}': v for k, v in space.items()}

        n_iter = preset['n_iter']
        started = time.time()
        search = RandomizedSearchCV(
            pipeline_estimator, space_prefixed, n_iter=n_iter, cv=preset['cv'],
            scoring=scoring, random_state=42, n_jobs=-1, refit=True,
        )
        search.fit(prep['X_train_raw'], y_tr)
        seconds_used = round(time.time() - started, 1)

        # search.best_estimator_ is the whole fitted Pipeline (refit on the
        # full raw train frame), so it takes the raw test frame too.
        best_model = search.best_estimator_
        tuned_score = float(scorer(best_model, prep['X_test_raw'], y_te))
        best_params = {
            (k[len('model__'):] if k.startswith('model__') else k): _to_native(v)
            for k, v in search.best_params_.items()
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
            'n_trials': int(len(search.cv_results_['params'])),
            'seconds_used': seconds_used,
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
