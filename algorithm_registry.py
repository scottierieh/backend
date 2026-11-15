"""
algorithm_registry.py — maps an Auto Compare algorithm label to a real,
persistable sklearn-compatible estimator.

register-model.ts sends `algorithm` verbatim from the winning run's label —
the exact strings in src/lib/model-lab/predictive-registry.ts's `autoCompare.label`
fields on the frontend (Random Forest, XGBoost, GBM, ...). It sends no
hyperparameters (Auto Compare's preview scripts run with their own defaults
and are never persisted — see models_api.py's docstring), so the estimators
built here use reasonable defaults of their own; they don't need to be
identical to the discarded preview run's, since the persisted model's own
reported metrics (computed in models_api.py after fitting) are what the
registry and the user actually see and judge it by.
"""

import math
from typing import Literal, Optional

Task = Literal['classification', 'regression']


# ---------------------------------------------------------------------------
# Which `params` /train accepts. Until 2026-11 they went straight into
# estimator.set_params(**params), and LightGBM / XGBoost / CatBoost accept far
# more than hyperparameters: LightGBM's distributed-training options
# (`machines`, `local_listen_port`, `num_machines`, `tree_learner`) bound a
# port, connected to a caller-chosen host and aborted the whole server
# process; CatBoost's `train_dir` / `snapshot_file` / `save_snapshot` wrote
# files at any path. So: only the names tune_analysis.py's _model_space
# searches over (the only thing that produces `params` -- its `best_params`),
# each with a type and a range a little wider than the search space. Anything
# else -- thread counts, paths, network, logging, callbacks -- is refused.
#
# Spec: ('int', lo, hi) | ('float', lo, hi) with lo < x <= hi when lo is
# exclusive (`'float_pos'`) | ('choice', {...}). `None` allowed where listed.
# ---------------------------------------------------------------------------
_N_TREES = ('int', 1, 1000)
_RATE = ('float_pos', 0.0, 1.0)
_FRACTION = ('float_pos', 0.0, 1.0)

TUNABLE_PARAMS = {
    'Random Forest': {
        'n_estimators': _N_TREES,
        'max_depth': ('int', 1, 100, 'none'),
        'min_samples_split': ('int', 2, 1000),
        'min_samples_leaf': ('int', 1, 1000),
        'max_features': ('choice', {'sqrt', 'log2', None}),
    },
    'GBM': {
        'n_estimators': _N_TREES,
        'learning_rate': _RATE,
        'max_depth': ('int', 1, 32),
        'subsample': _FRACTION,
    },
    'XGBoost': {
        'n_estimators': _N_TREES,
        'learning_rate': _RATE,
        'max_depth': ('int', 1, 16),
        'subsample': _FRACTION,
        'colsample_bytree': _FRACTION,
        'min_child_weight': ('float', 0.0, 100.0),
    },
    'LightGBM': {
        'n_estimators': _N_TREES,
        'learning_rate': _RATE,
        'num_leaves': ('int', 2, 1024),
        # -1 is LightGBM's "no limit", and in the search space.
        'max_depth': ('int', -1, 64, 'no_zero'),
        'subsample': _FRACTION,
        'colsample_bytree': _FRACTION,
    },
    'CatBoost': {
        'iterations': _N_TREES,
        'learning_rate': _RATE,
        'depth': ('int', 1, 12),
        'l2_leaf_reg': ('float', 0.0, 100.0),
    },
}


def _ok_value(spec, v) -> bool:
    kind = spec[0]
    if kind == 'choice':
        return v in spec[1] if (v is None or isinstance(v, str)) else False
    flags = spec[3:] if len(spec) > 3 else ()
    if v is None:
        return 'none' in flags
    if isinstance(v, bool):
        return False
    if kind == 'int':
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if not isinstance(v, int):
            return False
        if 'no_zero' in flags and v == 0:
            return False
        return spec[1] <= v <= spec[2]
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        return False
    if kind == 'float_pos':
        return spec[1] < v <= spec[2]
    return spec[1] <= v <= spec[2]


def validate_params(algorithm: str, params: Optional[dict]) -> dict:
    """The tuned parameters /train may apply to `algorithm`, or ValueError.

    Returns a clean copy (whole-number floats for int parameters become ints)."""
    if not params:
        return {}
    if not isinstance(params, dict):
        raise ValueError('params must be an object')
    allowed = TUNABLE_PARAMS.get(algorithm)
    if allowed is None:
        raise ValueError(f'{algorithm} takes no tuned parameters here.')
    refused = [k for k in params if k not in allowed]
    if refused:
        raise ValueError(
            f'{algorithm} does not accept '
            + ', '.join(repr(str(k)[:40]) for k in refused[:5])
            + f' here (tunable: {", ".join(sorted(allowed))})')
    bad = [k for k, v in params.items() if not _ok_value(allowed[k], v)]
    if bad:
        raise ValueError(f'{algorithm}: value out of range for ' + ', '.join(repr(k) for k in bad))
    out = {}
    for k, v in params.items():
        if allowed[k][0] == 'int' and isinstance(v, float):
            v = int(v)
        out[k] = v
    return out


def _build_default_estimator(algorithm: str, task: Task):
    """Return a fresh, unfitted estimator for `algorithm` + `task`. Raises
    ValueError for an algorithm/task combination this backend can't serve
    (e.g. Naive Bayes for regression — it's classification-only on the
    frontend's own lineup, so this should never actually be reached, but a
    clear error beats a confusing one if the lineup ever changes)."""
    clf = task == 'classification'

    if algorithm == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return RandomForestClassifier(n_estimators=300, random_state=42) if clf \
            else RandomForestRegressor(n_estimators=300, random_state=42)

    if algorithm == 'XGBoost':
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier(n_estimators=300, eval_metric='logloss', random_state=42) if clf \
            else XGBRegressor(n_estimators=300, random_state=42)

    if algorithm == 'LightGBM':
        from lightgbm import LGBMClassifier, LGBMRegressor
        return LGBMClassifier(n_estimators=300, random_state=42, verbosity=-1) if clf \
            else LGBMRegressor(n_estimators=300, random_state=42, verbosity=-1)

    if algorithm == 'CatBoost':
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostClassifier(iterations=300, random_state=42, verbose=False) if clf \
            else CatBoostRegressor(iterations=300, random_state=42, verbose=False)

    if algorithm == 'GBM':
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        return GradientBoostingClassifier(n_estimators=200, random_state=42) if clf \
            else GradientBoostingRegressor(n_estimators=200, random_state=42)

    if algorithm == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
        return AdaBoostClassifier(n_estimators=200, random_state=42) if clf \
            else AdaBoostRegressor(n_estimators=200, random_state=42)

    if algorithm == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        return DecisionTreeClassifier(random_state=42) if clf else DecisionTreeRegressor(random_state=42)

    if algorithm == 'Support Vector Machine (SVM)':
        from sklearn.svm import SVC, SVR
        return SVC(probability=True, random_state=42) if clf else SVR()

    if algorithm == 'K-Nearest Neighbors (KNN)':
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        return KNeighborsClassifier() if clf else KNeighborsRegressor()

    if algorithm == 'Naive Bayes':
        if not clf:
            raise ValueError('Naive Bayes is classification-only')
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

    if algorithm == 'Discriminant Analysis (LDA)':
        if not clf:
            raise ValueError('LDA is classification-only')
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()

    if algorithm == 'Elastic Net Regression':
        if clf:
            raise ValueError('Elastic Net Regression is regression-only')
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

    if algorithm == 'Artificial Neural Network (MLP)':
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42) if clf \
            else MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    if algorithm == 'Voting / Stacking Ensemble':
        # No base_estimators selection reaches this endpoint (Auto Compare's
        # own ensemble step picks those for the *preview* run only) — build a
        # small, fixed, generally-solid voting ensemble instead.
        from sklearn.ensemble import (
            VotingClassifier, VotingRegressor, RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
        )
        if clf:
            from sklearn.linear_model import LogisticRegression
            return VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000)),
            ], voting='soft')
        from sklearn.linear_model import Ridge
        return VotingRegressor(estimators=[
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=150, random_state=42)),
            ('ridge', Ridge()),
        ])

    raise ValueError(f'Unknown algorithm: {algorithm!r}')


def build_estimator(algorithm: str, task: Task, params: Optional[dict] = None):
    """The estimator this backend serves for `algorithm` + `task`.

    `params` overrides the defaults above -- this is how a TUNED model
    reaches the thing that actually gets served. Without it the search in
    tune_analysis.py ends at a number on a screen: the model deployed
    afterwards is built from those defaults, so the improvement the person
    was shown is not the model they got.
    """
    estimator = _build_default_estimator(algorithm, task)
    params = validate_params(algorithm, params)
    if not params:
        return estimator
    try:
        return estimator.set_params(**params)
    except ValueError as e:
        # A parameter this estimator does not take. Naming it beats
        # sklearn's own message, which lists every parameter it does take
        # and buries the one that was wrong.
        unknown = [k for k in params if k not in estimator.get_params(deep=True)]
        raise ValueError(
            f'{algorithm} does not take '
            + (', '.join(repr(k) for k in unknown) if unknown else 'those parameters')
            + f' ({e})'
        ) from e
