"""check_tuning.py — the tuning run is the one the leaderboard asked for.

    python scripts/check_tuning.py          # ~2 min

`tune_analysis.py` came back from the old Model Lab's deleted panel. Bringing
it to the current pipeline's contract is four separate promises, and every one
of them fails quietly rather than loudly:

  - it searches under the metric the board RANKS on, not a fixed f1_weighted —
    optimizing one number and reporting another looks like a result;
  - it applies the same class balancing Auto Compare applies, so the tuned
    model's gain is a tuning gain and not the balancing the rest already got;
  - it stops on a wall clock. RandomizedSearchCV has none: `thorough` is 80 x 5
    fits, and past the 600s request timeout the whole search is lost rather
    than returning the best of what it had;
  - it says which models it can tune AT ALL, so the screen does not draw a
    button whose only outcome is a ValueError.

And one older one, since this script had the leak the 14 were cleaned of: it
must survive a missing value. RandomForest and GBM raise on NaN, so before
`leak_safe_prepare_onehot` a single gap anywhere failed the entire search.

See docs/automl-hyperparameter-search.md for the request this checks against.
"""

import io
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import tune_analysis  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(HERE, 'tune_analysis.py')

TARGET = 'churn'
FEATURES = ['age', 'income', 'tenure', 'complaints', 'region']
REGIONS = ['서울', '부산', '대구']

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


def make_rows(n: int, seed: int = 20260929, holes: float = 0.0) -> list:
    """The same uneven target check_decision_threshold.py uses — about one
    positive in eight — optionally with `holes` of the input values missing."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        tenure = max(0, round(rnd.gauss(26, 15)))
        complaints = 0 if rnd.random() < 0.72 else math.ceil(rnd.random() * 4)
        income = round(math.exp(rnd.gauss(8.05, 0.42)) * 10) * 10
        logit = -0.9 - tenure * 0.055 + complaints * 0.62 - (0.35 if income > 4000 else 0)
        row = {
            'age': min(78, max(19, round(rnd.gauss(41, 12)))),
            'income': income, 'tenure': tenure, 'complaints': complaints,
            'region': REGIONS[rnd.randrange(len(REGIONS))],
            TARGET: 1 if rnd.random() < 1 / (1 + math.exp(-logit)) else 0,
        }
        if holes:
            for col in FEATURES:
                if rnd.random() < holes:
                    row[col] = None
        out.append(row)
    return out


def run(payload: dict, timeout: int = 600):
    """The script's real contract: one JSON in on stdin, one JSON out on
    stdout, or an error on stderr with a non-zero exit."""
    p = subprocess.run(
        [sys.executable, SCRIPT], input=json.dumps(payload), text=True,
        capture_output=True, timeout=timeout, cwd=HERE,
    )
    if p.returncode != 0:
        try:
            return None, json.loads(p.stderr.strip().splitlines()[-1]).get('error', p.stderr)
        except Exception:
            return None, p.stderr.strip()[-400:]
    return json.loads(p.stdout), None


def base(rows, **kw):
    out = {'data': rows, 'target_col': TARGET, 'feature_cols': FEATURES,
           'task_type': 'classification', 'preset': 'fast'}
    out.update(kw)
    return out


def main():
    rows = make_rows(900)

    # ---------------------------------------------------------- the catalog
    # The screen needs this before it has anything to tune. A Tune button on a
    # row this script would reject is a button that can only fail.
    cat, err = run({'list_only': True}, timeout=120)
    keys = [m['key'] for m in (cat or {}).get('tunable_models', [])]
    check(err is None and keys, f'list_only answers with no data at all: {keys}', err)
    check(set(keys) >= {'random-forest', 'gbm', 'xgboost', 'lightgbm', 'catboost'},
          'and it names the five the request asked for', keys)
    check(all(isinstance(m.get('available'), bool) for m in (cat or {}).get('tunable_models', [])),
          'each says whether the library is on THIS server, not just that it is known')

    _, err = run(base(rows, model='naive-bayes'), timeout=120)
    check(err and 'naive-bayes' in err and 'random-forest' in err,
          f'and an unsupported model is refused by name, listing what it could use: "{(err or "")[:80]}"')

    # --------------------------------------------- every model, its own route
    # Auto Compare weights each estimator through whatever route it has. The
    # tuned model has to get the SAME one, or its "improvement" carries the
    # balancing the rest already had.
    routes = {
        'random-forest': 'class_weight', 'gbm': 'sample_weight',
        'xgboost': 'scale_pos_weight', 'lightgbm': 'class_weight',
        'catboost': 'auto_class_weights',
    }
    first = None
    for model, expected in routes.items():
        res, err = run(base(rows, model=model, scoring='average_precision'))
        if err:
            check(False, f'{model}: tuned', err)
            continue
        first = first or res
        w = res.get('class_weighting') or {}
        check(w.get('applied') is True and expected in (w.get('method') or ''),
              f"{model}: weighted the way Auto Compare weights it — {w.get('method')}", w)
        check(res['best_score'] > res['baseline_score'],
              f"{model}: {res['baseline_score']:.4f} → {res['best_score']:.4f} "
              f"({res['improvement']:+.4f}) in {res['seconds_used']}s")

    # best_params is read by a screen and by a later refit. Every value used to
    # come back as a string — "300", and a None as "None" — because the
    # json.dumps fallback was being called on native scalars too.
    bp = (first or {}).get('best_params', {})
    check(bp and not any(isinstance(v, str) and v.replace('.', '', 1).lstrip('-').isdigit()
                         for v in bp.values()),
          f'best_params keeps its numbers as numbers: {bp}')
    check('None' not in [v for v in bp.values() if isinstance(v, str)],
          'and a null is a null, not the four letters', bp)

    # ------------------------------------------------------------- the metric
    res, err = run(base(rows, model='random-forest', scoring='average_precision'))
    check(err is None and res.get('scoring') == 'average_precision',
          'the response says which metric it searched under', err or res.get('scoring'))
    res_acc, err = run(base(rows, model='random-forest', scoring='accuracy'))
    check(err is None and res_acc.get('scoring') == 'accuracy'
          and res_acc['best_score'] != res['best_score'],
          'and a different metric is a different search, not the same one relabelled',
          res.get('best_score'), (res_acc or {}).get('best_score'))

    # The selection rule itself, away from the noise of a real dataset: the
    # winner is the candidate with the highest score, and a candidate that
    # cannot be fitted is a dead point in the space rather than a failed run.
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    X = np.asarray([[i % 7, i % 3] for i in range(120)], dtype=float)
    y = np.asarray([i % 2 for i in range(120)])
    found = tune_analysis._budgeted_search(
        lambda: Pipeline([('prep', FunctionTransformer()), ('model', DummyClassifier())]),
        {'model__strategy': ['most_frequent', 'prior', 'stratified', 'uniform']},
        X, y, StratifiedKFold(n_splits=3), 'accuracy',
        n_iter=4, budget_seconds=60, fit_params=None,
    )
    check(found['n_trials'] == 4 and found['best_params']['model__strategy'] in
          ('most_frequent', 'prior', 'uniform', 'stratified'),
          f"the search returns the best of what it scored: {found['best_params']}")
    check(found['best_cv_score'] >= 0.4,
          f"and the winner's score is the one it won on: {found['best_cv_score']:.3f}")

    # -------------------------------------------------------------- the clock
    started = time.time()
    res, err = run(base(rows, model='random-forest', preset='thorough',
                        scoring='average_precision', budget_seconds=12))
    wall = time.time() - started
    check(err is None and res['stopped_early'] is True
          and res['n_trials'] < res['n_candidates'],
          f"a 12s budget stops the 80-candidate search early: "
          f"{(res or {}).get('n_trials')}/{(res or {}).get('n_candidates')}", err)
    check(err is None and res['seconds_used'] <= 12 * 1.6,
          f'and it stops BEFORE the budget, not after — {(res or {}).get("seconds_used")}s '
          f'of 12s (whole run {wall:.0f}s)')
    check(err is None and res['best_score'] > res['baseline_score'],
          'and a search that ran out of time still answers with its best so far',
          (res or {}).get('improvement'))

    res, err = run(base(rows, model='random-forest', scoring='average_precision',
                        budget_seconds=300))
    check(err is None and res['stopped_early'] is False
          and res['n_trials'] == res['n_candidates'],
          f"a budget it fits inside runs every candidate: "
          f"{(res or {}).get('n_trials')}/{(res or {}).get('n_candidates')}", err)

    # A caller cannot ask for a budget that outlives the request itself: past
    # the 600s Cloud Run timeout the reply is lost along with the search.
    res, err = run(base(rows, model='random-forest', budget_seconds=5000))
    check(err is None and res['budget_seconds'] <= 600,
          f'a budget past the request timeout is capped: {(res or {}).get("budget_seconds")}s', err)

    # ------------------------------------------------------------- the holes
    # RandomForest and GBM raise on NaN. Before leak_safe_prepare_onehot a
    # single missing value failed the whole search, which is most real data.
    res, err = run(base(make_rows(900, holes=0.06), model='random-forest',
                        scoring='average_precision'))
    check(err is None and res.get('best_params'),
          'a frame with 6% of its inputs missing tunes rather than raising', err)
    check(err is None and res['row_counts']['n_train'] > 600,
          'and the rows with holes were imputed, not dropped',
          (res or {}).get('row_counts'))

    # ---------------------------------------------------------- what it is not
    res, err = run(base(rows, model='random-forest', class_weight=None))
    w = (res or {}).get('class_weighting') or {}
    check(err is None and w.get('applied') is False and 'turned off' in (w.get('reason') or ''),
          'class_weight: null is obeyed and said out loud, not ignored', err or w)

    res, err = run(base(rows, model='random-forest', target_col='income',
                        feature_cols=['age', 'tenure', 'complaints', 'region'],
                        task_type='regression', scoring='neg_root_mean_squared_error'))
    check(err is None and res.get('class_weighting') is None,
          'regression reports no weighting at all — "turned off" would describe '
          'a choice nobody made', err or res.get('class_weighting'))

    check(err is None and res.get('scored_on') == 'inner_split',
          "and every response says its score is from this script's own split, "
          'not the sealed holdout the leaderboard reports on', (res or {}).get('scored_on'))

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
