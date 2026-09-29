"""check_blend.py — a blend of the winners is the winners, or it says otherwise.

    python scripts/check_blend.py          # ~2 min

Auto Compare ranks fourteen models and then throws the ranking away: the
reader picks one row. Blending the top of the board is the largest single
thing missing from that (docs/automl-gap-analysis.md, "상위 N 앙상블"), and
`ensemble_stacking_analysis.py` could already do it — it takes
`base_estimators` by name.

Except its registry knew eight of the fourteen, and an unknown name was
dropped without a word. So "blend the top 3" could come back as a blend of
two, and a request where none of the three were known fell through to the
DEFAULT three — a blend of models nobody asked for, reported as the answer.
Both are well-shaped responses. Neither is the thing that was requested.

What this pins down: every model on the board can be named, a name that
cannot be built is an error rather than a silent omission, and the members
that come back are the members that were asked for.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(HERE, 'ensemble_stacking_analysis.py')

sys.path.insert(0, os.path.join(HERE, 'scripts'))
from check_decision_threshold import make_rows, TARGET, FEATURES  # noqa: E402

_ok = 0
_failed = 0


def check(cond, msg, *extra):
    global _ok, _failed
    if cond:
        _ok += 1
        print(f'ok    {msg}')
    else:
        _failed += 1
        print(f'FAIL  {msg}')
        for e in extra:
            print(f'        {e}')


def run(payload, timeout=900):
    p = subprocess.run([sys.executable, SCRIPT], input=json.dumps(payload), text=True,
                       capture_output=True, timeout=timeout, cwd=HERE)
    if p.returncode != 0:
        try:
            return None, json.loads(p.stderr.strip().splitlines()[-1]).get('error', p.stderr)
        except Exception:
            return None, p.stderr.strip()[-300:]
    body = json.loads(p.stdout)
    return body.get('result', body), None


def base(rows, **kw):
    out = {'data': rows, 'target_col': TARGET, 'feature_cols': FEATURES,
           'task_type': 'classification', 'cv_scoring': 'average_precision'}
    out.update(kw)
    return out


def main():
    rows = make_rows(700)

    # ---------------------------------------------------------- the catalog
    cat, err = run({'list_only': True}, timeout=120)
    clf = [m['key'] for m in (cat or {}).get('blendable_models', {}).get('classification', [])]
    reg = [m['key'] for m in (cat or {}).get('blendable_models', {}).get('regression', [])]
    check(err is None and clf and reg, f'list_only answers with no data: {len(clf)} + {len(reg)}', err)

    # Every model Auto Compare can rank must be nameable, or a blend of the
    # top 3 silently becomes a blend of fewer.
    want_clf = {'random_forest', 'xgboost', 'lightgbm', 'catboost', 'gbm', 'decision_tree',
                'svm', 'knn', 'naive_bayes', 'discriminant', 'adaboost', 'mlp'}
    missing = want_clf - set(clf)
    check(not missing, f'every classification model on the board can be a member', missing)
    want_reg = {'random_forest', 'xgboost', 'lightgbm', 'catboost', 'gbm', 'decision_tree',
                'svm', 'knn', 'adaboost', 'mlp', 'elasticnet'}
    check(not (want_reg - set(reg)), 'and every regression one too', want_reg - set(reg))

    # The screen joins its BOARD ROWS to these by route. Matching on the label
    # would put "LightGBM" and "GBM" one substring apart.
    entries = (cat or {}).get('blendable_models', {}).get('classification', [])
    routed = {m['key']: m.get('routes') for m in entries}
    check(routed.get('lightgbm') == ['lightgbm'] and routed.get('gbm') == ['gradient-boosting'],
          'each member names the Auto Compare route it is, so the join is not on a label',
          routed.get('lightgbm'), routed.get('gbm'))
    check(routed.get('logistic_regression') == [],
          'and a member with no board row of its own claims no route')

    # ------------------------------------------------- a named blend is that blend
    asked = ['catboost', 'svm', 'discriminant']
    res, err = run(base(rows, base_estimators=asked, ensemble_method='voting', voting_type='soft'))
    check(err is None and res.get('base_estimators') == asked,
          f"the members that come back are the members asked for: {(res or {}).get('base_estimators')}",
          err)
    check(err is None and all(a in (res.get('individual_scores') or {}) for a in asked),
          'and each member is scored on its own, so the blend can be read against them',
          (res or {}).get('individual_scores'))

    # Three of these were not in the registry before. A blend that quietly
    # became the default three would still return a well-shaped response.
    default_three = {'logistic_regression', 'decision_tree', 'random_forest'}
    check(err is None and not default_three.issubset(set(res.get('base_estimators') or [])),
          'and it is not the default blend wearing the answer to another question')

    cv = (res or {}).get('cv_results') or {}
    check(cv.get('cv_scoring') == 'average_precision',
          f"the blend is cross-validated on the metric the board ranks on: {cv.get('cv_scoring')}",
          cv)
    w = (res or {}).get('class_weighting') or {}
    check(w.get('applied') is True,
          f"and every member carries the class weighting: {w.get('method')}", w)

    # ------------------------------------------------------ stacking, and losing
    st, err = run(base(rows, base_estimators=['random_forest', 'lightgbm', 'xgboost'],
                       ensemble_method='stacking', final_estimator='logistic_regression'))
    check(err is None and 'Stacking' in (st.get('model_label') or ''),
          f"a stacking blend names its meta-learner: {(st or {}).get('model_label')}", err)
    # A blend is not automatically better than its members, and the response
    # has to carry enough for a screen to show that rather than crown it.
    scores = (st or {}).get('individual_scores') or {}
    blend_label = (st or {}).get('model_label')
    members = {k: v for k, v in scores.items() if k != blend_label}
    check(blend_label in scores and len(members) == 3,
          f"and the blend's own score sits beside its members', whichever way it went: "
          f"blend {scores.get(blend_label)} vs {members}")

    # --------------------------------------------- a name that cannot be built
    _, err = run(base(rows, base_estimators=['random_forest', 'autogluon']))
    check(err and 'autogluon' in err and 'random_forest' not in err.split('Available')[0],
          f"one unbuildable name is an error naming it, not a blend of the rest: "
          f"\"{(err or '')[:70]}\"")

    _, err = run(base(rows, base_estimators=['autogluon', 'nope']))
    check(err and 'None of the requested' in err,
          f"and a request where none can be built does NOT fall through to the "
          f"defaults: \"{(err or '')[:60]}\"")

    # No list at all is still the old behaviour: the default three.
    res, err = run(base(rows))
    check(err is None and set(res.get('base_estimators') or []) == default_three,
          'naming nothing still gets the default blend, so an older caller is unaffected',
          (res or {}).get('base_estimators'))

    # ------------------------------ the blend that is served is the blend
    # Stage 2 scores the shortlist on the sealed rows through /train, and the
    # deploy step serves what that fitted. If /train built the fixed three
    # from algorithm_registry, the sealed-row number would describe a
    # different ensemble from the one on the leaderboard, under its name.
    import types
    store: dict = {}
    fake = types.ModuleType('model_store')
    fake.save_pipeline = lambda mid, art: (
        store.__setitem__(f'gs://check/{mid}', art) or f'gs://check/{mid}')
    fake.load_pipeline = lambda uri: store[uri]
    fake.delete_artifact = lambda uri: store.pop(uri, None)
    sys.modules['model_store'] = fake
    import models_api  # noqa: E402
    models_api.model_store = fake

    got = models_api.train_model('m-blend', models_api.TrainRequest(
        data=rows, algorithm='Voting / Stacking Ensemble', target=TARGET,
        features=FEATURES, task='classification',
        ensemble={'members': asked, 'method': 'voting', 'votingType': 'soft'}))
    est = store[got['artifactUri']]['pipeline'].named_steps['est']
    served = [n for n, _ in getattr(est, 'estimators', [])]
    check(served == asked,
          f'/train serves the members that were blended, not a fixed three: {served}', asked)
    check((got.get('classWeighting') or {}).get('applied') is True,
          'and the served blend is class-weighted member by member',
          got.get('classWeighting'))

    fixed = models_api.train_model('m-fixed', models_api.TrainRequest(
        data=rows, algorithm='Voting / Stacking Ensemble', target=TARGET,
        features=FEATURES, task='classification'))
    fixed_est = store[fixed['artifactUri']]['pipeline'].named_steps['est']
    check([n for n, _ in getattr(fixed_est, 'estimators', [])] != served,
          'while no member list still gets the old fixed ensemble, so the two differ',
          [n for n, _ in getattr(fixed_est, 'estimators', [])])

    stacked = models_api.train_model('m-stack', models_api.TrainRequest(
        data=rows, algorithm='Voting / Stacking Ensemble', target=TARGET,
        features=FEATURES, task='classification',
        ensemble={'members': ['random_forest', 'knn'], 'method': 'stacking',
                  'finalEstimator': 'logistic_regression'}))
    st_est = store[stacked['artifactUri']]['pipeline'].named_steps['est']
    check(type(st_est).__name__ == 'StackingClassifier',
          f'a stacking blend is served as one, not voted instead: {type(st_est).__name__}')

    try:
        models_api.train_model('m-bad-blend', models_api.TrainRequest(
            data=rows, algorithm='Voting / Stacking Ensemble', target=TARGET,
            features=FEATURES, task='classification',
            ensemble={'members': ['autogluon']}))
        check(False, 'an unbuildable member list is refused')
    except Exception as e:
        detail = str(getattr(e, 'detail', e))
        check(getattr(e, 'status_code', None) == 400 and 'autogluon' in detail,
              f'an unbuildable member list is a 400 naming it: "{detail[:60]}"', e)

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
