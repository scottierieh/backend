"""blend_registry.py — the members a blend can be built from, in one place.

Auto Compare ranks fourteen models and the reader then picks one row. Blending
the top of that board is the largest single thing the comparison was missing,
and it is asked for by NAMING models -- so there has to be one list of what can
be named, and one function that turns names into a fitted ensemble.

Two callers need exactly that and would otherwise keep their own copies:

  ensemble_stacking_analysis.py   the board row -- fits the blend and scores it
                                  the way every other Auto Compare script does
  models_api.py (/train)          the sealed-row score and the served model

A second copy in the second caller is how a blend comes to mean one set of
members on the leaderboard and a different, fixed set once it is scored for
real or deployed -- the same failure as a tuned model that is served untuned.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from typing import Any, Dict, List, Optional

from analysis_common import balanced_weighting

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CB = True
except Exception:
    _HAS_CB = False


BASE_ESTIMATORS = {
    'classification': {
        'logistic_regression': lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
        'decision_tree': lambda rs: DecisionTreeClassifier(max_depth=5, random_state=rs),
        'random_forest': lambda rs: RandomForestClassifier(n_estimators=100, random_state=rs),
        'gbm': lambda rs: GradientBoostingClassifier(random_state=rs),
        'svm': lambda rs: SVC(probability=True, random_state=rs),
        'knn': lambda rs: KNeighborsClassifier(n_neighbors=5),
        # The rest of Auto Compare's lineup. A blend is asked for by naming
        # the models that won the board, and a winner this registry did not
        # know was silently dropped -- a "top 3 blend" of two.
        'adaboost': lambda rs: AdaBoostClassifier(n_estimators=200, random_state=rs),
        'naive_bayes': lambda rs: GaussianNB(),
        'discriminant': lambda rs: LinearDiscriminantAnalysis(),
        'mlp': lambda rs: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=rs),
    },
    'regression': {
        'ridge': lambda rs: Ridge(random_state=rs),
        'decision_tree': lambda rs: DecisionTreeRegressor(max_depth=5, random_state=rs),
        'random_forest': lambda rs: RandomForestRegressor(n_estimators=100, random_state=rs),
        'gbm': lambda rs: GradientBoostingRegressor(random_state=rs),
        'svm': lambda rs: SVR(),
        'knn': lambda rs: KNeighborsRegressor(n_neighbors=5),
        'adaboost': lambda rs: AdaBoostRegressor(n_estimators=200, random_state=rs),
        'mlp': lambda rs: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=rs),
        'elasticnet': lambda rs: ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=rs, max_iter=10000),
    }
}

# Boosting learners registered only if their lib imported — keeps blends able to
# include the common Auto Compare winners without making them a hard dependency.
if _HAS_XGB:
    BASE_ESTIMATORS['classification']['xgboost'] = lambda rs: XGBClassifier(random_state=rs, n_jobs=-1, verbosity=0, eval_metric='logloss')
    BASE_ESTIMATORS['regression']['xgboost'] = lambda rs: XGBRegressor(random_state=rs, n_jobs=-1, verbosity=0)
if _HAS_LGBM:
    BASE_ESTIMATORS['classification']['lightgbm'] = lambda rs: LGBMClassifier(random_state=rs, n_jobs=-1, verbose=-1)
    BASE_ESTIMATORS['regression']['lightgbm'] = lambda rs: LGBMRegressor(random_state=rs, n_jobs=-1, verbose=-1)
if _HAS_CB:
    # verbose=False and allow_writing_files=False for the same reasons
    # tune_analysis.py pins them: CatBoost logs to stdout, which would corrupt
    # this script's one-JSON-object-on-stdout contract, and it writes scratch
    # files a stateless container has no lasting place for.
    BASE_ESTIMATORS['classification']['catboost'] = lambda rs: CatBoostClassifier(
        iterations=300, random_state=rs, verbose=False, allow_writing_files=False)
    BASE_ESTIMATORS['regression']['catboost'] = lambda rs: CatBoostRegressor(
        iterations=300, random_state=rs, verbose=False, allow_writing_files=False)

DEFAULT_ESTIMATORS = {
    'classification': ['logistic_regression', 'decision_tree', 'random_forest'],
    'regression': ['ridge', 'decision_tree', 'random_forest']
}


def _weight_member(estimator, y_train, requested):
    """Class-weight one ensemble member, parameter routes only.

    An ensemble takes a single sample_weight and hands the same array to every
    member, so a member whose only route is fit(sample_weight=) cannot be
    weighted on its own -- passing it at the ensemble level would also weight
    the members that already carry class_weight='balanced', counting the
    correction twice. Such a member is left unweighted and says so.
    """
    estimator, fit_kwargs, report = balanced_weighting(estimator, y_train, requested)
    if fit_kwargs.get('sample_weight') is not None:
        report = {
            'applied': False, 'method': None,
            'reason': f'{type(estimator).__name__} can only be weighted through '
                      f'fit(sample_weight=), which an ensemble cannot route to one member',
        }
    return estimator, report


def blendable_models(task_type: str = None):
    """Which models can be members of a blend, and whether the library is on
    THIS server.

    Auto Compare asks for a blend by naming the models that won its board, so
    the screen has to know which of its rows can be named before it offers the
    button. Repeating the list on the frontend would drift the first time one
    is added here -- `{"list_only": true}` answers this and reads no data.
    """
    LABELS = {
        'logistic_regression': 'Logistic Regression', 'ridge': 'Ridge',
        'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest',
        'gbm': 'Gradient Boosting', 'svm': 'Support Vector Machine (SVM)',
        'knn': 'K-Nearest Neighbors (KNN)', 'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM', 'catboost': 'CatBoost', 'adaboost': 'AdaBoost',
        'naive_bayes': 'Naive Bayes', 'discriminant': 'Discriminant Analysis (LDA)',
        'mlp': 'Artificial Neural Network (MLP)', 'elasticnet': 'Elastic Net Regression',
    }
    tasks = [task_type] if task_type else ['classification', 'regression']
    return {
        t: [{'key': k, 'label': LABELS.get(k, k)} for k in BASE_ESTIMATORS[t]]
        for t in tasks
    }


class UnknownEstimators(ValueError):
    """An explicit request named models this script cannot build."""


def build_estimators(task_type: str, names: Optional[List[str]], random_state: int,
                     y_train=None, class_weight='balanced'):
    """Returns (estimators, weighting_reports). The reports are empty for
    regression and whenever y_train is not supplied.

    A name this registry does not know used to be dropped without a word, and
    a request where NONE matched fell through to the default three. So "blend
    the top 3" could come back as a blend of two, or as the default blend
    wearing the answer to a question nobody asked. An explicit request is now
    either buildable or an error.
    """
    registry = BASE_ESTIMATORS[task_type]
    if names:
        unknown = [n for n in names if n not in registry]
        chosen = [n for n in names if n in registry]
        if not chosen:
            raise UnknownEstimators(
                f"None of the requested base estimators exist for {task_type}: "
                f"{', '.join(map(repr, names))}. Available: {', '.join(registry)}"
            )
        if unknown:
            raise UnknownEstimators(
                f"No {task_type} base estimator named {', '.join(map(repr, unknown))}. "
                f"Available: {', '.join(registry)}"
            )
    else:
        chosen = DEFAULT_ESTIMATORS[task_type]
    built = [(name, registry[name](random_state)) for name in chosen]
    if task_type != 'classification' or y_train is None:
        return built, {}

    weighted, reports = [], {}
    for name, est in built:
        est, report = _weight_member(est, y_train, class_weight)
        weighted.append((name, est))
        reports[name] = report
    return weighted, reports


def _ensemble_weighting_report(member_reports: Dict[str, Any], final_name=None,
                               final_report=None) -> Dict[str, Any]:
    """Fold the per-member reports into the {applied, method, reason} shape the
    Compare screen reads. `applied` is true only when EVERY member was
    weighted: a blend with one unweighted member is not on the same footing as
    a fully weighted model, and the screen's footnote exists to say so."""
    reports = dict(member_reports)
    if final_report is not None:
        reports[f'{final_name} (meta)'] = final_report
    if not reports:
        return {'applied': False, 'method': None, 'reason': 'not a classification task'}

    missed = [n for n, r in reports.items() if not r.get('applied')]
    done = [n for n, r in reports.items() if r.get('applied')]
    if not missed:
        return {
            'applied': True,
            'method': f'class-weighted every member ({len(done)})',
            'reason': None, 'members': reports,
        }
    return {
        'applied': False,
        'method': (f'class-weighted {len(done)} of {len(reports)} members'
                   if done else None),
        'reason': ('unweighted: ' + ', '.join(missed)),
        'members': reports,
    }



def build_blend(task_type: str, members: Optional[List[str]], method: str = 'voting',
                final_estimator: str = 'logistic_regression', voting_type: str = 'soft',
                random_state: int = 42, y_train=None, class_weight='balanced'):
    """The fitted-shape ensemble for `members`, plus its weighting report.

    Returns (model, chosen_names, weighting_report). `model` is unfitted.

    This is the one place a list of names becomes an ensemble, so the blend on
    the leaderboard, the blend scored on the sealed rows, and the blend that
    gets deployed are the same object built the same way.
    """
    estimators, member_weighting = build_estimators(
        task_type, members, random_state,
        y_train if task_type == 'classification' else None, class_weight)

    final_weighting, final_name = None, None
    if method == 'stacking':
        registry = BASE_ESTIMATORS[task_type]
        final = registry.get(final_estimator,
                             registry[DEFAULT_ESTIMATORS[task_type][0]])(random_state)
        if task_type == 'classification' and y_train is not None:
            # The meta-learner trains on the base models' out-of-fold
            # predictions against the SAME uneven target.
            final, final_weighting = _weight_member(final, y_train, class_weight)
            final_name = final_estimator
        model = (StackingClassifier if task_type == 'classification' else StackingRegressor)(
            estimators=estimators, final_estimator=final, cv=5)
    elif task_type == 'classification':
        model = VotingClassifier(estimators=estimators, voting=voting_type)
    else:
        model = VotingRegressor(estimators=estimators)

    report = _ensemble_weighting_report(member_weighting, final_name, final_weighting)
    return model, [n for n, _ in estimators], report
