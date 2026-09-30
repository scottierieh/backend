"""check_baseline.py — the reference line, against the thing it stands in for.

    python scripts/check_baseline.py          # a few seconds

Model Lab's Compare screen draws a baseline row from a CLOSED FORM over the
holdout's class counts (src/lib/model-lab/baseline.ts) rather than by fitting
anything. A constant predictor's metrics are fixed by the class distribution,
so this is the same answer the short way — but "the same answer" is a claim,
and a closed form that is subtly wrong yields a plausible reference line that
every model on the board is then read against.

So the same distributions go through sklearn's DummyClassifier here and the
two are compared. If they ever diverge, the screen is the one that is wrong.

The formulas being checked, for "always answer the largest class":

  accuracy   the largest class's share
  macro F1   that class gets recall 1 and precision = its share; every other
             class is never predicted, so zero_division=0 scores it 0
  ROC-AUC    0.5 — one score for every row ranks nothing
  PR-AUC     the positive class's share
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, f1_score, roc_auc_score,
)

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


def closed_form(counts):
    """What baseline.ts computes, transcribed. Nothing is fitted."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    top = int(np.argmax(counts))
    share = counts[top] / n
    f1_top = (2 * share) / (share + 1)
    out = {
        'accuracy': share,
        'f1_macro': f1_top / len(counts),
        'f1': share * f1_top,
        'auc': 0.5,
    }
    if len(counts) == 2:
        out['pr_auc'] = counts[1] / n
    return out, top


def from_sklearn(counts):
    """The same thing by fitting a DummyClassifier and scoring it."""
    y = np.concatenate([np.full(int(c), i) for i, c in enumerate(counts)])
    X = np.zeros((len(y), 1))
    m = DummyClassifier(strategy='most_frequent').fit(X, y)
    pred = m.predict(X)
    out = {
        'accuracy': accuracy_score(y, pred),
        'f1_macro': f1_score(y, pred, average='macro', zero_division=0),
        'f1': f1_score(y, pred, average='weighted', zero_division=0),
    }
    if len(counts) == 2:
        # `prior` gives a CONSTANT probability — the class priors — which is
        # what a majority-class answer's confidence actually is. The ranking
        # metrics are about that constant, not about the hard label.
        pm = DummyClassifier(strategy='prior').fit(X, y)
        proba = pm.predict_proba(X)[:, 1]
        out['auc'] = roc_auc_score(y, proba)
        out['pr_auc'] = average_precision_score(y, proba)
    return out


def main():
    cases = [
        ('the Compare footnote\'s own shape, 164:23', [164, 23]),
        ('a rarer minority, 380:20', [380, 20]),
        ('balanced, 100:100', [100, 100]),
        ('three classes, 40:20:20', [40, 20, 20]),
        ('five classes, uneven', [50, 25, 12, 8, 5]),
    ]

    for label, counts in cases:
        mine, top = closed_form(counts)
        theirs = from_sklearn(counts)
        wrong = [
            f'{k}: closed form {mine[k]:.10f} vs sklearn {theirs[k]:.10f}'
            for k in theirs if abs(mine[k] - theirs[k]) > 1e-9
        ]
        check(not wrong,
              f'{label}: every figure matches sklearn — '
              + ', '.join(f'{k}={mine[k]:.4f}' for k in sorted(mine)),
              *wrong)

    # The two readings that make the baseline worth drawing at all.
    mine, _ = closed_form([164, 23])
    check(abs(mine['accuracy'] - 0.877) < 1e-3,
          f"87.7% accuracy from answering \"no\" to everything — the number a "
          f"model scoring 87% is actually competing with")
    check(abs(mine['pr_auc'] - 23 / 187) < 1e-9 and mine['auc'] == 0.5,
          'and on the metric that board is RANKED by, the line is the positive '
          f"rate ({23 / 187:.4f}), not 0.5")

    # A DummyClassifier that predicts the majority has no skill, and both
    # rankings say so in their own units: 0.5 for ROC, the prevalence for PR.
    balanced, _ = closed_form([100, 100])
    check(abs(balanced['pr_auc'] - 0.5) < 1e-9,
          'on a balanced target the PR-AUC line sits at 0.5 too, which is why '
          'a PR-AUC of 0.5 means nothing there and everything on an uneven one')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
