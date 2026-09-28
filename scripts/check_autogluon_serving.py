"""check_autogluon_serving.py — an AutoGluon model can be saved and served.

    python scripts/check_autogluon_serving.py          # ~90s, needs autogluon

A TabularPredictor is a directory and model_store writes a single joblib blob,
so serving one means packing a directory into bytes, carrying it inside the
artifact dict, and getting a working predictor back on the other side. Every
step of that is real here -- the actual request handlers, the actual fit, the
actual pack and unpack -- with model_store replaced by an in-memory stand-in so
the check needs no bucket and no credentials.

What it is guarding against is a round trip that looks fine and is not: a
bundle that unpacks but loads without its models, a probability frame whose
columns moved, a predictor re-untarred on every prediction. None of those
raise; they return plausible numbers.

The data is generated here rather than read from a fixture so the check runs
anywhere, and it is deliberately imbalanced -- roughly one positive in eight --
because that is the shape the app's ranking policy exists for.
"""

import os
import sys
import types
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = 'churn'
FEATURES = ['age', 'income', 'tenure', 'complaints', 'region']
REGIONS = ['서울', '부산', '대구']


def make_rows(n: int, seed: int = 20260928) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        tenure = max(0, round(rnd.gauss(26, 15)))
        complaints = 0 if rnd.random() < 0.72 else math.ceil(rnd.random() * 4)
        income = round(math.exp(rnd.gauss(8.05, 0.42)) * 10) * 10
        logit = -0.9 - tenure * 0.055 + complaints * 0.62 - (0.35 if income > 4000 else 0)
        rows.append({
            'age': min(78, max(19, round(rnd.gauss(41, 12)))),
            'income': income,
            'tenure': tenure,
            'complaints': complaints,
            'region': REGIONS[rnd.randrange(len(REGIONS))],
            TARGET: 1 if rnd.random() < 1 / (1 + math.exp(-logit)) else 0,
        })
    return rows


# model_store, in memory. Everything else is the real module.
_store: dict = {}
_fake = types.ModuleType('model_store')
_fake.save_pipeline = lambda model_id, artifact: (
    _store.__setitem__(f'gs://check/{model_id}', artifact) or f'gs://check/{model_id}')
_fake.load_pipeline = lambda uri: _store[uri]
_fake.delete_artifact = lambda uri: _store.pop(uri, None)
sys.modules['model_store'] = _fake

import models_api  # noqa: E402  — after the stand-in is registered
from fastapi import HTTPException  # noqa: E402

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


def refuses(fn, code, msg):
    try:
        fn()
        check(False, f'{msg} — it was accepted')
    except HTTPException as e:
        check(e.status_code == code, f'{msg} — HTTP {e.status_code}: {str(e.detail)[:64]}', e.status_code)


def main() -> int:
    try:
        import autogluon.tabular  # noqa: F401
    except ImportError:
        print('autogluon.tabular is not installed — nothing to check.')
        return 0

    rows = make_rows(1000)
    train_rows, holdout_rows = rows[:800], rows[800:]
    positives = sum(r[TARGET] for r in rows)
    print(f'{len(rows)} rows, {positives} positive ({positives / len(rows) * 100:.1f}%)\n')

    req = models_api.TrainRequest(
        data=train_rows, algorithm='AutoGluon · WeightedEnsemble_L2', target=TARGET,
        features=FEATURES, task='classification', holdout=holdout_rows,
        engine='autogluon', timeLimit=45, evalMetric='average_precision',
    )
    res = models_api.train_model('m-check', req)
    art = _store[res['artifactUri']]

    check(art.get('kind') == 'autogluon', 'the artifact says which engine it is', art.get('kind'))
    size = len(art.get('bundle') or b'')
    check(size > 0, f'and carries a packed predictor ({size / 1e6:.2f}MB gzipped)')
    # clone_for_deployment is what keeps the training leftovers out. Without it
    # the same predictor packs to several times this.
    check(size < 40_000_000, 'small enough to be a blob rather than a directory')
    check(art.get('class_labels') == [0, 1], 'with the class labels as they were at train time',
          art.get('class_labels'))

    check(res['evaluatedOn'] == 'holdout', "scored on the caller's sealed rows, not an internal split",
          res['evaluatedOn'])
    check((res.get('metrics') or {}).get('n_eval') == float(len(holdout_rows)),
          f'over all {len(holdout_rows)} of them', res.get('metrics'))
    ap = (res.get('metrics') or {}).get('average_precision')
    check(ap is not None, f'with the metric it was told to optimise (PR-AUC {ap:.3f})' if ap else
          'the requested metric is missing from the score')
    # A beeswarm this response did not compute is worse than no beeswarm.
    check(res['shapBeeswarm'] is None, 'and no SHAP beeswarm, rather than one it never computed')
    check(set(res.get('featureBaseline') or {}) == set(FEATURES),
          'the form schema covers every feature the model needs')

    score_rows = [{k: r[k] for k in FEATURES} for r in holdout_rows[:6]]
    uri = res['artifactUri']
    p = models_api.predict_model('m-check', models_api.PredictRequest(artifactUri=uri, rows=score_rows))

    check(len(p['predictions']) == len(score_rows), 'every row comes back with a prediction')
    check(all(v in (0, 1) for v in p['predictions']), 'in the label space it was trained on',
          p['predictions'])
    check(p['probabilities'] is not None and all(len(r) == 2 for r in p['probabilities']),
          'one probability column per class, not a single winning number',
          (p['probabilities'] or [None])[0])
    check(all(abs(sum(r) - 1.0) < 1e-6 for r in p['probabilities']), 'and each row sums to 1')
    check(p['shapContributions'] is None, 'row contributions are declared absent, not faked')

    # Untarring on every call would work and would be slow enough to matter.
    loaded = art.get('_ag_loaded')
    check(loaded is not None, 'the predictor is kept on the artifact after the first call')
    models_api.predict_model('m-check', models_api.PredictRequest(artifactUri=uri, rows=score_rows))
    check(art.get('_ag_loaded') is loaded, 'and a second call reuses it rather than unpacking again')

    refuses(lambda: models_api.predict_model('m-check', models_api.PredictRequest(
        artifactUri=uri, rows=[{k: v for k, v in score_rows[0].items() if k != 'age'}])),
        400, "a row missing a column the model needs")

    refuses(lambda: models_api.train_model('m-bad', models_api.TrainRequest(
        data=train_rows, algorithm='AutoGluon', target=TARGET, features=FEATURES,
        task='classification', engine='autogluon', timeLimit=20, agModel='NoSuchModel')),
        400, 'a model name that is not on the predictor')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
