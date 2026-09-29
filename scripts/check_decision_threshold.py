"""check_decision_threshold.py — a probability is not a decision.

    python scripts/check_decision_threshold.py          # ~20s

Everything this app does about an uneven target stops at the ranking. The
leaderboard moves to PR-AUC because accuracy rewards ignoring the smaller
class; the table marks every accuracy that fails to beat "always answer the
larger class". Then the saved model answers at argmax — 0.5 — and the class
someone built the model to find goes missing.

What this guards is not that a cut exists but that it is honest and that it
is used. Three ways it could be neither:

  - chosen on the rows the model was fitted on, which picks the cut that best
    separates memorised data;
  - stored on the artifact and never applied, so the screen reports one number
    and the service answers by another;
  - applied without being reported, so a row whose top probability is 0.40
    comes back positive and reads as a bug.

None of those raise. All three return a well-shaped response.
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


def make_rows(n: int, seed: int = 20260929) -> list[dict]:
    """An uneven target with real structure — about one positive in eight."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        tenure = max(0, round(rnd.gauss(26, 15)))
        complaints = 0 if rnd.random() < 0.72 else math.ceil(rnd.random() * 4)
        income = round(math.exp(rnd.gauss(8.05, 0.42)) * 10) * 10
        logit = -0.9 - tenure * 0.055 + complaints * 0.62 - (0.35 if income > 4000 else 0)
        out.append({
            'age': min(78, max(19, round(rnd.gauss(41, 12)))),
            'income': income, 'tenure': tenure, 'complaints': complaints,
            'region': REGIONS[rnd.randrange(len(REGIONS))],
            TARGET: 1 if rnd.random() < 1 / (1 + math.exp(-logit)) else 0,
        })
    return out


_store: dict = {}
_fake = types.ModuleType('model_store')
_fake.save_pipeline = lambda model_id, artifact: (
    _store.__setitem__(f'gs://check/{model_id}', artifact) or f'gs://check/{model_id}')
_fake.load_pipeline = lambda uri: _store[uri]
_fake.delete_artifact = lambda uri: _store.pop(uri, None)
sys.modules['model_store'] = _fake

import models_api  # noqa: E402

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


def main() -> int:
    rows = make_rows(1200)
    train, hold = rows[:900], rows[900:]
    positives = sum(r[TARGET] for r in rows)
    print(f'{len(rows)} rows, {positives} positive ({positives / len(rows) * 100:.1f}%)\n')

    res = models_api.train_model('m-th', models_api.TrainRequest(
        data=train, algorithm='Random Forest', target=TARGET, features=FEATURES,
        task='classification', holdout=hold))
    t = res.get('threshold')

    check(t is not None, 'an uneven binary target gets a chosen cut')
    if not t:
        print(f'\n{_ok} ok, {_failed} failure(s)')
        return 1

    check(t['chosen_on'] == 'holdout' and t['n_eval'] == len(hold),
          f"chosen on the {t['n_eval']} held-out rows, not on the ones it was fitted on",
          t.get('chosen_on'), t.get('n_eval'))

    # The smaller class, named. `classes_[1]` is a convention the reader cannot
    # see, and on an uneven target it is the class that goes missing at 0.5.
    check(str(t['positive_class']) == '1', 'and aimed at the smaller class, by name',
          t['positive_class'])

    lo, hi = t['at_default'], t['at_chosen']
    check(abs(lo['threshold'] - 0.5) < 1e-9, 'the default it is compared against is 0.5')
    check(hi['f1'] >= lo['f1'], f"F1 at the chosen cut is no worse: {lo['f1']:.3f} -> {hi['f1']:.3f}")
    check(hi['recall'] > lo['recall'],
          f"and the smaller class is found more often: recall {lo['recall']:.3f} -> {hi['recall']:.3f}")
    # Both points, not just the better one. A screen that shows only the gain
    # is selling a trade as a free lunch — precision usually falls.
    check(all(k in lo and k in hi for k in ('precision', 'recall', 'f1', 'predicted_positive')),
          'both operating points come back, so the cost can be stated too')

    # ---- and it is what the service actually answers by ------------------
    score = [{k: r[k] for k in FEATURES} for r in hold]
    p = models_api.predict_model('m-th', models_api.PredictRequest(
        artifactUri=res['artifactUri'], rows=score))

    idx = t['positive_index']
    cut = t['value']
    served = sum(1 for v in p['predictions'] if str(v) == str(t['positive_class']))
    at_argmax = sum(1 for row in p['probabilities'] if row[idx] >= 0.5)
    by_cut = sum(1 for row in p['probabilities'] if row[idx] >= cut)

    check(served == by_cut, f'the served answers follow the stored cut ({served} positives)',
          served, by_cut)
    check(served != at_argmax,
          f'and differ from argmax, which is the whole point ({at_argmax} at 0.5)')
    check(p.get('threshold') is not None,
          'the response says which cut joined those probabilities to those labels')

    # Every row must obey it, not just the count.
    disagree = [i for i, row in enumerate(p['probabilities'])
                if (str(p['predictions'][i]) == str(t['positive_class'])) != (row[idx] >= cut)]
    check(not disagree, 'every row individually, not just the totals', disagree[:5])

    # ---- when there is no cut to choose ----------------------------------
    # Regression has no classes to cut between. A number here would be one
    # nobody could act on.
    reg = models_api.train_model('m-reg', models_api.TrainRequest(
        data=train, algorithm='Random Forest', target='income',
        features=['age', 'tenure', 'complaints', 'region'], task='regression',
        holdout=hold))
    check(reg.get('threshold') is None, 'regression gets no cut, rather than a meaningless one')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
