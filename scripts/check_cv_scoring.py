"""check_cv_scoring.py — the cross-validation measures what the board ranks on.

    python scripts/check_cv_scoring.py          # ~15s

Auto Compare ranks an uneven binary target on PR-AUC, because accuracy rises
as a model ignores the smaller class. Every script cross-validated on accuracy
regardless. So the column the shortlist is supposed to be chosen on held
exactly the measure the ranking had rejected, and the screen could not use it
to settle an order it was not measured in.

`run_cv` always took a `scoring` argument. Nothing passed one.

Two ways this fails quietly rather than loudly, and both are the point:

  - an estimator that cannot produce the metric would take the whole
    cross-validation down with it, and CV is what the shortlist rests on;
  - a BINARY-only metric on a target that is not binary does not raise at all
    once the labels are integers. average_precision takes pos_label=1 and
    scores class 1 against the rest — about 0.33 on three balanced classes,
    arriving on the board labelled PR-AUC. Nothing throws, so this has to be
    checked rather than caught.

See docs/automl-gap-analysis.md ("CV로 순위").
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from analysis_common import cv_scoring_of, cv_score_value  # noqa: E402
from cv_strategy import run_cv  # noqa: E402

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


def binary(n=400, seed=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = (X[:, 0] + rng.randn(n) * 0.7 > 1.1).astype(int)   # about one in seven
    return X, y


def multi(n=300, seed=5):
    rng = np.random.RandomState(seed)
    return rng.randn(n, 4), rng.choice([0, 1, 2], n)


def main():
    X, y = binary()

    out = run_cv(RandomForestClassifier(n_estimators=40, random_state=0), X, y,
                 'classification', 5, 42, scoring='average_precision')
    check(out['cv_scoring'] == 'average_precision' and 'cv_scoring_fallback' not in out,
          f"the requested metric is the one scored: {out['cv_scoring']} = {out['cv_mean']:.4f}", out)

    default = run_cv(RandomForestClassifier(n_estimators=40, random_state=0), X, y,
                     'classification', 5, 42)
    check(default['cv_scoring'] == 'accuracy',
          'and no request keeps the old default, so an older caller is unaffected',
          default['cv_scoring'])
    check(abs(default['cv_mean'] - out['cv_mean']) > 0.05,
          f"the two are genuinely different numbers — accuracy {default['cv_mean']:.3f} "
          f"vs PR-AUC {out['cv_mean']:.3f}, which is the whole argument")

    # ---- the binary-only metric on a target that is not binary ----------
    Xm, ym = multi()
    mc = run_cv(RandomForestClassifier(n_estimators=40, random_state=0), Xm, ym,
                'classification', 5, 42, scoring='average_precision')
    check(mc['cv_scoring'] == 'accuracy'
          and mc.get('cv_scoring_requested') == 'average_precision'
          and 'binary' in (mc.get('cv_scoring_fallback') or ''),
          f"a binary metric on 3 classes falls back and says so: "
          f"\"{mc.get('cv_scoring_fallback')}\"", mc)
    # The number it would otherwise have reported is class-1-against-the-rest,
    # which lands near 1/3 on three balanced classes and reads as a PR-AUC.
    check(abs(mc['cv_mean'] - 0.33) > 0.1 or mc['cv_scoring'] == 'accuracy',
          'and the number reported is the accuracy, not the one-vs-rest it would have been')

    # ---- an estimator that cannot produce the metric --------------------
    class NoProba(DummyClassifier):
        def __init__(self):
            super().__init__(strategy='most_frequent')
        def predict_proba(self, X):  # noqa: D401
            raise AttributeError('this estimator has no probabilities')

    lost = run_cv(NoProba(), X, y, 'classification', 5, 42, scoring='average_precision')
    check(lost['cv_scoring'] == 'accuracy' and lost.get('cv_scoring_fallback'),
          'an estimator with no probabilities keeps its CV instead of losing it',
          lost.get('cv_scoring_fallback'))

    # ---- regression: the negated scorers ---------------------------------
    Xr = np.random.RandomState(1).randn(300, 3)
    yr = Xr[:, 0] * 3 + np.random.RandomState(2).randn(300) * 0.5
    reg = run_cv(RandomForestRegressor(n_estimators=40, random_state=0), Xr, yr,
                 'regression', 5, 42, scoring='neg_root_mean_squared_error')
    check(reg['cv_scoring'] == 'neg_root_mean_squared_error' and reg['cv_mean'] < 0,
          f"a regression board gets RMSE in its negated form, so higher is still "
          f"better: {reg['cv_mean']:.3f}", reg)

    # ---- the payload reader ----------------------------------------------
    check(cv_scoring_of({'cv_scoring': 'f1_macro'}, 'classification') == 'f1_macro',
          'cv_scoring_of passes the metric through')
    check(cv_scoring_of({}, 'classification') is None,
          'and a caller that predates the key gets no metric, not a guess')
    check(cv_scoring_of({'cv_scoring': 'r2'}, 'classification') is None,
          'a regression scorer asked of a classification run is ignored, not fallen back from')
    check(cv_scoring_of({'cv_scoring': 'accuracy'}, 'regression') is None,
          'and the reverse too')

    # ---- the hand-written fold loop (CatBoost) ---------------------------
    # It fits on a Pool carrying the categorical columns, so an sklearn scorer
    # does not apply to it; it computes the metric from predictions instead,
    # and must agree with what run_cv would have produced.
    yb = np.array([0] * 80 + [1] * 20)
    pred = np.array([0] * 75 + [1] * 5 + [0] * 8 + [1] * 12)
    proba = np.column_stack([1 - (pred * 0.8 + 0.1), pred * 0.8 + 0.1])
    from sklearn.metrics import f1_score, average_precision_score
    check(abs(cv_score_value('f1_macro', yb, pred)
              - f1_score(yb, pred, average='macro')) < 1e-9,
          'cv_score_value agrees with sklearn on f1_macro')
    check(abs(cv_score_value('average_precision', yb, pred, proba)
              - average_precision_score(yb, proba[:, 1])) < 1e-9,
          'and on average_precision from the positive column')
    check(cv_score_value('average_precision', np.array([0, 1, 2, 0, 1, 2]),
                         np.array([0, 1, 2, 0, 1, 2]),
                         np.eye(3)[[0, 1, 2, 0, 1, 2]]) is None,
          'and refuses a binary metric on three classes, exactly as run_cv does')
    check(cv_score_value('average_precision', yb, pred, None) is None,
          'and refuses a probability metric with no probabilities')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
