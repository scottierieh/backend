"""check_ivd_plan.py — the sample size, and why it is not the first n that works.

    python scripts/check_ivd_plan.py          # a few seconds

/api/analysis/ivd-plan answers the one question step 01 has to answer and
`power-analysis` cannot: not "what power does this test have" but "how many
specimens before the interval is narrow enough for the goal to be reachable".

The number it returns is NOT the first n that clears the goal, and this check
exists mostly to pin that. The observed count is a whole number of specimens,
so the achieved proportion — and the bound with it — wobbles as n grows. The
first n that clears is followed by a run of n that do not, and a plan written
on the first n and executed one specimen later misses.

The floor is checked by brute force here: every n from the returned value to
the search cap must clear, and the n below it must not. That is the property,
stated directly, rather than the formula re-run.
"""

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

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


def run(payload):
    proc = subprocess.run(
        [sys.executable, os.path.join(_ROOT, 'ivd_plan_analysis.py')],
        input=json.dumps(payload), capture_output=True, text=True)
    if proc.returncode != 0:
        try:
            return {'__error': json.loads(proc.stderr)['error']}
        except Exception:
            return {'__error': proc.stderr.strip() or proc.stdout.strip()}
    return json.loads(proc.stdout)


def req(result, mid):
    return next(r for r in result['requirements'] if r['id'] == mid)


# The bound from its DEFINITION — the p at which seeing x or more successes has
# probability alpha — rather than from the Beta quantile the module uses. Same
# reason check_ivd_accuracy.py does it: agreeing with the shortcut is evidence,
# agreeing with itself is not.
def cp_lower(x, n, conf=0.95):
    a = (1 - conf) / 2
    if x <= 0:
        return 0.0
    if x >= n:
        return a ** (1 / n)
    return brentq(lambda p: binom.sf(x - 1, n, p) - a, 1e-12, 1 - 1e-12, xtol=1e-14)


def clears(assume, goal, n, conf=0.95):
    return cp_lower(int(np.rint(assume * n)), n, conf) >= goal


def main():
    # ---- 1. the floor is a floor ----------------------------------------
    cases = [(0.95, 0.90), (0.97, 0.95), (0.99, 0.98), (0.98, 0.95), (0.95, 0.85)]
    for assume, goal in cases:
        res = run({'goals': {'sensitivity': goal}, 'assume': {'sensitivity': assume},
                   'max_n': 4000})
        r = req(res, 'sensitivity')
        need, first, cap = r['n_required'], r['n_first_pass'], r['searched_to']

        holds_above = all(clears(assume, goal, n) for n in range(need, cap + 1))
        below_fails = not clears(assume, goal, need - 1)
        check(holds_above and below_fails,
              f'assume {assume:.0%}, goal {goal:.0%}: {need} is a floor — every n from '
              f'{need} to {cap} clears, and {need - 1} does not',
              f'holds above {holds_above}, {need - 1} fails {below_fails}')

    # ---- 2. and the first n that works is NOT it ------------------------
    res = run({'goals': {'sensitivity': 0.90}, 'assume': {'sensitivity': 0.95}})
    r = req(res, 'sensitivity')
    broken = [n for n in range(r['n_first_pass'], r['n_required'])
              if not clears(0.95, 0.90, n)]
    check(r['n_first_pass'] == 127 and r['n_required'] == 154 and len(broken) > 0,
          f"the first n that clears is {r['n_first_pass']} and {len(broken)} of the n above "
          f"it do NOT ({broken[:6]}…) — a plan written on {r['n_first_pass']} and executed "
          f"with {broken[0]} specimens misses, which is why {r['n_required']} is returned",
          r)
    check(r['n_required'] / r['n_first_pass'] > 1.15,
          f"and the gap is {r['n_required'] / r['n_first_pass']:.2f}x, not a rounding "
          f'detail — it is the difference between a study that reads out and one that '
          f'does not')
    check(any('하한' in (n.get('ko') or '') for n in res['notes']),
          'the response says both numbers and which one to plan on')

    # ---- 3. the bound at the floor is the bound the analysis will give --
    # A plan written against one interval and analysed with another is a plan
    # that proves nothing, so the two have to be the same quantile.
    want = cp_lower(r['x_at_required'], r['n_required'])
    check(abs(r['bound_at_required'] - want) < 1e-9
          and r['bound_at_required'] >= r['goal'],
          f"at {r['n_required']} the bound is {r['bound_at_required']:.4f} — the same "
          f'Clopper-Pearson /api/analysis/ivd-accuracy reports, and it clears '
          f"{r['goal']:.0%}")

    # ---- 4. a goal at or above the assumption is not a sample size ------
    for assume, goal in [(0.95, 0.95), (0.90, 0.95)]:
        res = run({'goals': {'sensitivity': goal}, 'assume': {'sensitivity': assume}})
        r = req(res, 'sensitivity')
        check(r['reachable'] is False and r['n_required'] is None,
              f'assume {assume:.0%} against goal {goal:.0%}: unreachable at any n, because '
              f'the bound converges UP to the assumption and never past it',
              r)
    res = run({'goals': {'sensitivity': 0.95}, 'assume': {'sensitivity': 0.95}})
    note = ' '.join(n.get('en', '') for n in res['notes'])
    check('Change the goal or the assumption, not the sample size' in note,
          'and the advice is to change the plan, not to collect more — "more specimens" '
          'would be wrong advice here')

    # ---- 5. how brittle the answer is to the assumption -----------------
    # The planner's assumption is a guess, and the answer moves enormously with
    # it. Worth knowing that this is real before trusting any single number.
    sizes = {}
    for assume in (0.96, 0.97, 0.98, 0.99):
        res = run({'goals': {'sensitivity': 0.95}, 'assume': {'sensitivity': assume},
                   'max_n': 4000})
        sizes[assume] = req(res, 'sensitivity')['n_required']
    check(sizes[0.96] > 6 * sizes[0.99],
          f'the answer swings {sizes[0.96] / sizes[0.99]:.0f}x across one point of assumed '
          f'performance: {", ".join(f"{a:.0%}→{n}" for a, n in sizes.items())}',
          sizes)
    check(all(sizes[a] > sizes[b] for a, b in zip([0.96, 0.97, 0.98], [0.97, 0.98, 0.99])),
          'and it moves monotonically, so a planner can read the direction off it')

    # ---- 6. both metrics, and which specimens each counts ---------------
    res = run({'goals': {'sensitivity': 0.95, 'specificity': 0.98},
               'assume': {'sensitivity': 0.97, 'specificity': 0.99}})
    sens, spec = req(res, 'sensitivity'), req(res, 'specificity')
    check(sens['n_required'] == 490 and spec['n_required'] == 785,
          f"both goals sized together — {sens['n_required']} positive and "
          f"{spec['n_required']} negative specimens")
    check('양성' in sens['counts']['ko'] and '음성' in spec['counts']['ko'],
          'and each says WHICH specimens its number counts — a study that collected 400 '
          'specimens of which 30 were positive has met neither goal',
          sens['counts'], spec['counts'])

    # ---- 7. the names follow the comparator, as in the analysis ---------
    part = run({'goals': {'sensitivity': 0.95}, 'assume': {'sensitivity': 0.97},
                'reference_complete': False})
    check(req(part, 'sensitivity')['label']['ko'] == '양성 일치율 (PPA)'
          and req(part, 'sensitivity')['n_required'] == 490,
          'an incomplete comparator renames the metric here too, and changes no number — '
          'the same rule the analysis follows')

    # ---- 8. judging the point estimate makes this vacuous ---------------
    res = run({'goals': {'sensitivity': 0.95}, 'assume': {'sensitivity': 0.97},
               'goal_basis': 'point'})
    note = ' '.join(n.get('en', '') for n in res['notes'])
    check('no sample size to compute' in note,
          'on a point-estimate basis the response says there is no sample size to compute — '
          'one specimen above the goal would pass, which is an argument for the bound')

    # ---- 9. refusals ----------------------------------------------------
    check('__error' in run({'goals': {}, 'assume': {}}),
          'no goals is an error rather than an empty answer')
    bad = run({'goals': {'sensitivity': 0.95}, 'assume': {}})
    check('__error' in bad and 'assumed performance' in bad['__error'],
          'and a goal with no assumption is refused, naming why one is needed',
          bad.get('__error', '')[:140])
    bad = run({'goals': {'accuracy': 0.95}, 'assume': {'accuracy': 0.97}})
    check('__error' in bad and 'accuracy' in bad['__error'],
          'a goal on a metric this cannot size names it rather than ignoring it',
          bad.get('__error', '')[:120])

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
