"""check_feature_scoring.py — what a column is worth, and what it already knows.

    python scripts/check_feature_scoring.py          # ~40s

The Features screen lets a person choose columns, and until now it offered no
evidence to choose on. Two things this endpoint has to get right, and both
fail as a well-shaped answer rather than as an error:

  THE LEAK THE LINEAR CHECK CANNOT SEE. guardrails.py finds a leaked column by
  correlating it with the target, which works for a numeric copy and misses a
  categorical one. An account status set when the customer churns correlates
  -0.19 with churn -- nothing -- and alone predicts it perfectly. Measured
  here: compute_guardrails returns NOTHING on the fixture below, and this
  endpoint names the column.

  NOT CRYING WOLF. A continuous measurement has all-distinct values, and so
  does a row id. Flagging every float column as an identifier would make the
  whole panel noise, which is worse than no panel.

The scores are guidance for a person, so the response also has to say which
rows it read: choosing columns on the rows a model is later measured on leaks,
mildly but really.
"""

import json
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from guardrails import compute_guardrails  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(HERE, 'feature_scoring_analysis.py')

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


REGIONS = ['서울', '부산', '대구']


def make_rows(n=900, seed=20260929):
    """An uneven target with real structure, plus four columns planted to be
    found: a leaked category, a constant, a key, and pure noise."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        tenure = max(0, round(rnd.gauss(26, 15)))
        complaints = 0 if rnd.random() < 0.72 else math.ceil(rnd.random() * 4)
        logit = -0.9 - tenure * 0.055 + complaints * 0.62
        churn = 1 if rnd.random() < 1 / (1 + math.exp(-logit)) else 0
        out.append({
            'age': min(78, max(19, round(rnd.gauss(41, 12)))),
            'tenure': tenure,
            'complaints': complaints,
            'region': REGIONS[rnd.randrange(len(REGIONS))],
            # Set when the customer churns. Ordinally coded it correlates with
            # the target barely at all, which is the point.
            'account_status': 'closed' if churn else rnd.choice(['active', 'dormant', 'suspended']),
            'everyone': 'same',                 # constant
            'customer_no': 100000 + i,          # a key: whole numbers, all distinct
            'sensor': rnd.gauss(0, 1),          # a measurement: floats, all distinct
            'churn': churn,
        })
    return out


FEATURES = ['age', 'tenure', 'complaints', 'region', 'account_status',
            'everyone', 'customer_no', 'sensor']


def run(payload, timeout=900):
    p = subprocess.run([sys.executable, SCRIPT], input=json.dumps(payload), text=True,
                       capture_output=True, timeout=timeout, cwd=HERE)
    if p.returncode != 0:
        try:
            return None, json.loads(p.stderr.strip().splitlines()[-1]).get('error', p.stderr)
        except Exception:
            return None, p.stderr.strip()[-300:]
    return json.loads(p.stdout), None


def main():
    rows = make_rows()
    res, err = run({'data': rows, 'target_col': 'churn', 'feature_cols': FEATURES,
                    'task_type': 'classification', 'scored_on': 'train_half'})
    if err:
        check(False, 'the endpoint answers', err)
        print(f'\n{_ok} ok, {_failed} failure(s)')
        return 1

    by_name = {f['feature']: f for f in res['features']}

    # ---- the leak the linear check cannot see -------------------------------
    df = pd.DataFrame(rows)
    flagged = compute_guardrails(df[FEATURES], df['churn'], FEATURES,
                                 'classification', {'accuracy': 0.86})
    check(not flagged,
          'the existing linear guardrail finds nothing on this data — which is '
          'why this endpoint exists', [w['id'] for w in flagged])

    leaked = by_name['account_status']
    check(abs(leaked['corr_with_target'] or 0) < 0.5,
          f"and the leaked column's correlation with the target is only "
          f"{leaked['corr_with_target']:.3f}, so correlation was never going to find it")
    check(leaked['alone_score'] is not None and leaked['alone_score'] > 0.97,
          f"fitted ALONE it reaches {leaked['alone_score']:.3f} — that is the reading "
          f"that finds it")
    check('account_status' in res['leak_suspects'],
          'so it is named as a leak suspect', res['leak_suspects'])
    check(leaked['rank'] == 1,
          'and it ranks first, which is exactly what a column that is the answer does')

    # ---- not crying wolf ----------------------------------------------------
    check('identifier_like' in by_name['customer_no']['flags'],
          'a key of whole numbers, all distinct, is called a key')
    check('identifier_like' not in by_name['sensor']['flags'],
          'a float measurement, also all distinct, is NOT — flagging every '
          'continuous column would make the panel noise', by_name['sensor']['flags'])
    check('constant' in by_name['everyone']['flags'],
          'a column with one value is called constant')
    for real in ('tenure', 'complaints'):
        check(not by_name[real]['flags'],
              f"and a column that genuinely carries signal is left unflagged: {real}")

    # ---- the two lists are different questions ------------------------------
    check(not (set(res['leak_suspects']) & set(res['no_signal'])),
          'no column is both the answer and nothing — the lists are disjoint',
          res['leak_suspects'], res['no_signal'])
    check(set(res['no_signal']) >= {'everyone', 'customer_no'},
          'the no-signal list holds what it should', res['no_signal'])

    # ---- the ranking is comparable across column types ----------------------
    check(res['ranking_metric'] == 'mutual_info',
          'the ranking is on mutual information — the one score a numeric and a '
          'categorical column can share')
    ranks = sorted(f['rank'] for f in res['features'])
    check(ranks == list(range(1, len(FEATURES) + 1)),
          'every column gets exactly one rank', ranks)
    check(by_name['tenure']['rank'] < by_name['sensor']['rank'],
          'and a column with signal outranks pure noise')

    # ---- which rows it read -------------------------------------------------
    check(res.get('scored_on') == 'train_half',
          'the response says which rows the scores came from — choosing columns '
          'on the rows a model is measured on leaks, mildly but really')
    check(res['row_counts']['n_scored'] == len(rows),
          'and how many it scored', res['row_counts'])

    # ---- regression, and the correlation filter -----------------------------
    reg_rows = [dict(r, income=r['tenure'] * 40 + r['age'] * 3 + random.Random(1).random())
                for r in rows]
    for r in reg_rows:
        r['income_copy'] = r['income']
    reg, err = run({'data': reg_rows, 'target_col': 'income',
                    'feature_cols': ['age', 'tenure', 'complaints', 'region',
                                     'income_copy', 'sensor'],
                    'task_type': 'regression', 'scored_on': 'all_rows'})
    if err:
        check(False, 'regression answers', err)
    else:
        rby = {f['feature']: f for f in reg['features']}
        check('target_duplicate' in rby['income_copy']['flags'],
              'an exact copy of a numeric target is called a duplicate, not merely correlated',
              rby['income_copy']['flags'])
        check(reg['task_type'] == 'regression', 'and the task is read as regression')

    # Two columns carrying the same thing: reported, never applied.
    dup_rows = [dict(r, tenure_months=r['tenure'] * 1.0) for r in rows]
    dup, err = run({'data': dup_rows, 'target_col': 'churn',
                    'feature_cols': ['tenure', 'tenure_months', 'age'],
                    'task_type': 'classification', 'scored_on': 'train_half'})
    if err:
        check(False, 'the redundancy pass answers', err)
    else:
        dby = {f['feature']: f for f in dup['features']}
        pair = dby['tenure'].get('redundant_with') or dby['tenure_months'].get('redundant_with')
        check(pair and pair['with'] in ('tenure', 'tenure_months') and abs(pair['corr']) > 0.99,
              f'a duplicate column is reported as redundant with the one that outranks it: {pair}')
        check(not (dby['age'].get('redundant_with')),
              'and an unrelated column is not')

    # ---- refusals ------------------------------------------------------------
    _, err = run({'data': rows, 'target_col': 'churn', 'feature_cols': ['nope']})
    check(err and 'nope' in err, f'a column that is not in the data is refused by name: "{err}"')
    _, err = run({'data': rows[:5], 'target_col': 'churn', 'feature_cols': FEATURES})
    check(err and 'rows' in err.lower(),
          f'and too few rows is refused rather than scored on nothing: "{err}"')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
