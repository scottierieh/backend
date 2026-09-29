"""check_class_weighting.py — the smaller class is weighted, or the response says why not.

    python scripts/check_class_weighting.py          # a few seconds

The frontend has been sending `class_weight: "balanced"` to every
classification route since it was written, and nothing read it. The
leaderboard moves to PR-AUC because accuracy rewards ignoring the smaller
class — and then every model was fitted as if the classes were even.

`class_weight='balanced'` on all of them raises TypeError on about half, so
the difficulty is entirely in the per-estimator route. That is what this
pins down: which route each one takes, and — for the two that have none —
that the response says so rather than going quiet. Ten weighted models and
two unweighted ones on one PR-AUC leaderboard is a comparison under two
conditions, and the screen can only mark it if it is told. (Elastic Net is
regression only, so there is nothing to balance; the ensemble reports member
by member, since a blend can be weighted in part.)

See docs/automl-class-imbalance.md for the table this checks against.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_common import balanced_weighting  # noqa: E402

_ok = 0
_failed = 0


def check(cond, msg, *extra):
    global _ok, _failed
    if cond:
        _ok += 1
        print('ok   ', msg)
    else:
        _failed += 1
        print('FAIL ', msg, *extra)


def estimators():
    from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    # (script, estimator, the route docs/automl-class-imbalance.md specifies)
    return [
        ('random_forest', RandomForestClassifier(), 'class_weight'),
        ('decision_tree', DecisionTreeClassifier(), 'class_weight'),
        ('svm', SVC(), 'class_weight'),
        ('lightgbm', lgb.LGBMClassifier(verbose=-1), 'class_weight'),
        ('elastic_net', LogisticRegression(), 'class_weight'),
        ('xgboost', xgb.XGBClassifier(), 'scale_pos_weight'),
        ('catboost', CatBoostClassifier(verbose=0), 'auto_class_weights'),
        ('adaboost', AdaBoostClassifier(), 'sample_weight'),
        ('gbm', GradientBoostingClassifier(), 'sample_weight'),
        ('naive_bayes', GaussianNB(), 'sample_weight'),
        ('discriminant', LinearDiscriminantAnalysis(), 'priors'),
        ('knn', KNeighborsClassifier(), None),
        ('mlp', MLPClassifier(), None),
    ]


def main() -> int:
    y = np.array([0] * 870 + [1] * 130)

    for label, est, want in estimators():
        _, fit_kwargs, rep = balanced_weighting(est, y)
        if want is None:
            check(rep['applied'] is False and rep['reason'],
                  f'{label}: has no route, and the response says which estimator and why '
                  f'— "{(rep.get("reason") or "")[:52]}"', rep)
            check(rep['method'] is None, f'{label}: and claims no method it did not use')
            continue
        check(rep['applied'] is True and want in (rep['method'] or ''),
              f"{label}: {rep['method']}", rep)
        # sample_weight is the only route that has to travel to .fit(); the
        # others set a parameter so the estimator re-derives per fit, which is
        # what keeps cross-validation honest.
        if want == 'sample_weight':
            check('sample_weight' in fit_kwargs and len(fit_kwargs['sample_weight']) == len(y),
                  f'{label}: with weights for every training row', list(fit_kwargs))
        else:
            check(not fit_kwargs,
                  f'{label}: set on the estimator, so each CV fold derives its own', fit_kwargs)

    # A balanced target must come out as a no-op rather than as a special case:
    # a condition on imbalance would make it impossible to tell afterwards
    # which regime a run was under.
    from sklearn.utils.class_weight import compute_sample_weight
    even = np.array([0, 1] * 200)
    check(np.allclose(compute_sample_weight('balanced', even), 1.0),
          'on a balanced target every weight is 1.0, so this needs no condition')

    # And it can be turned off, for a caller that wants the unweighted fit.
    from sklearn.ensemble import RandomForestClassifier
    est, kw, rep = balanced_weighting(RandomForestClassifier(), y, requested=None)
    check(rep['applied'] is False and not kw and est.get_params()['class_weight'] is None,
          'class_weight: null turns it off and says so', rep)

    # ---- and /train actually uses it ------------------------------------
    # The helper being right is half of it. A helper that nothing calls is
    # what this whole request already was: the frontend has been sending
    # class_weight since it was written and no route read it.
    # The data generator comes from the threshold check, which installs its own
    # model_store stand-in on import. Importing it FIRST and then installing
    # this one means the store models_api writes to is the one read back here
    # — the other way round, the artifact lands in a dict this function does
    # not hold and the lookup fails on a key that looks almost right.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_decision_threshold import make_rows, FEATURES, TARGET

    import types
    _store: dict = {}
    fake = types.ModuleType('model_store')
    fake.save_pipeline = lambda i, a: (_store.__setitem__(f'gs://c/{i}', a) or f'gs://c/{i}')
    fake.load_pipeline = lambda u: _store[u]
    fake.delete_artifact = lambda u: None
    sys.modules['model_store'] = fake
    import models_api
    models_api.model_store = fake

    rows = make_rows(900)
    res = models_api.train_model('m-cw', models_api.TrainRequest(
        data=rows, algorithm='Random Forest', target=TARGET,
        features=FEATURES, task='classification'))
    rep = res.get('classWeighting')
    check(rep and rep.get('applied') is True,
          f"/train weights the model it serves: {(rep or {}).get('method')}", rep)

    art = _store[res['artifactUri']]
    est = art['pipeline'].named_steps['est']
    check(est.get_params().get('class_weight') == 'balanced',
          'and the estimator in the saved artifact carries it, not just the report',
          est.get_params().get('class_weight'))

    off = models_api.train_model('m-cw-off', models_api.TrainRequest(
        data=rows, algorithm='Random Forest', target=TARGET,
        features=FEATURES, task='classification', classWeight=None))
    off_est = _store[off['artifactUri']]['pipeline'].named_steps['est']
    check(off_est.get_params().get('class_weight') is None
          and (off.get('classWeighting') or {}).get('applied') is False,
          'and a caller that turns it off gets the unweighted fit it asked for',
          off.get('classWeighting'))

    reg = models_api.train_model('m-cw-reg', models_api.TrainRequest(
        data=rows, algorithm='Random Forest', target='income',
        features=['age', 'tenure', 'complaints', 'region'], task='regression'))
    check(reg.get('classWeighting') is None,
          'regression reports none, rather than a balance it cannot have')

    # The estimators with no route must reach /train the same way.
    knn = models_api.train_model('m-cw-knn', models_api.TrainRequest(
        data=rows, algorithm='K-Nearest Neighbors (KNN)', target=TARGET,
        features=FEATURES, task='classification'))
    krep = knn.get('classWeighting') or {}
    check(krep.get('applied') is False and 'KNeighbors' in (krep.get('reason') or ''),
          f"and KNN comes back unweighted with its reason: \"{(krep.get('reason') or '')[:46]}\"", krep)

    # Discriminant analysis is the one estimator whose route — the class
    # priors — is also something a caller can set deliberately. Replacing a
    # supplied prior with a uniform one answers a question nobody asked.
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    y_sk = np.array([0] * 870 + [1] * 130)
    _, _, unset = balanced_weighting(LinearDiscriminantAnalysis(), y_sk)
    check(unset.get('applied') is True and 'priors' in (unset.get('method') or ''),
          f"discriminant with priors unset: {unset.get('method')}", unset)

    _, _, given = balanced_weighting(LinearDiscriminantAnalysis(priors=[0.9, 0.1]), y_sk)
    check(given.get('applied') is False and 'caller' in (given.get('reason') or ''),
          'and a caller-supplied prior is left alone, not overwritten', given)

    # An ensemble can be weighted in part: members with a parameter route get
    # it, members whose only route is fit(sample_weight=) cannot, because the
    # weights would reach every member and count the correction twice on the
    # ones already carrying class_weight.
    import ensemble_stacking_analysis as ens
    y_ens = np.array(['no'] * 780 + ['yes'] * 120)

    _, full = ens.build_estimators(
        'classification', ['logistic_regression', 'decision_tree', 'random_forest'], 42, y_ens)
    full_rep = ens._ensemble_weighting_report(full)
    check(full_rep.get('applied') is True and all(r['applied'] for r in full.values()),
          f"ensemble, all members weightable: {full_rep.get('method')}", full_rep)

    _, mixed = ens.build_estimators(
        'classification', ['logistic_regression', 'gbm', 'knn'], 42, y_ens)
    mixed_rep = ens._ensemble_weighting_report(mixed)
    check(mixed_rep.get('applied') is False
          and 'gbm' in (mixed_rep.get('reason') or '')
          and 'knn' in (mixed_rep.get('reason') or ''),
          f"and a partly weighted blend is NOT applied: \"{mixed_rep.get('reason')}\"", mixed_rep)
    check('sample_weight' in (mixed['gbm'].get('reason') or '')
          and 'route' in (mixed['knn'].get('reason') or '')
          or 'neither' in (mixed['knn'].get('reason') or ''),
          'and each member says which of the two reasons it is', mixed['gbm'], mixed['knn'])

    reg_members = ens.build_estimators('regression', ['ridge', 'random_forest'], 42)[1]
    check(reg_members == {},
          'regression members report no weighting at all, rather than a false one')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
