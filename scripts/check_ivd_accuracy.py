"""check_ivd_accuracy.py — the interval, and the three ways to get this wrong.

    python scripts/check_ivd_accuracy.py          # a few seconds

/api/analysis/ivd-accuracy turns one row per specimen into the table a clinical
performance study report opens with. Everything it computes is arithmetic a
spreadsheet could do; what it is FOR is the three decisions around that
arithmetic, each of which fails by returning an ordinary-looking number:

  1. THE INTERVAL. A goal is judged against a confidence bound. The Wald
     approximation breaks where these studies live, and it breaks in the
     direction of handing out a pass.
  2. WHICH WAY ROUND. Sensitivity with the columns swapped is still plausible.
  3. WHAT IT IS CALLED. Percent agreement is not sensitivity, and the
     computation cannot tell them apart.

The intervals here are NOT checked against the module's own function — that
would be an identity. They are checked against the two closed forms anyone can
verify by hand (x = n and x = 0), against scipy's Beta quantile reached by a
different route, and against the textbook relation between a Clopper-Pearson
bound and the binomial tail it is defined by.
"""

import json
import math
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

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
    """Through the real CLI contract — stdin in, stdout out."""
    proc = subprocess.run(
        [sys.executable, os.path.join(_ROOT, 'ivd_accuracy_analysis.py')],
        input=json.dumps(payload), capture_output=True, text=True)
    if proc.returncode != 0:
        try:
            return {'__error': json.loads(proc.stderr)['error']}
        except Exception:
            return {'__error': proc.stderr.strip() or proc.stdout.strip()}
    return json.loads(proc.stdout)


def rows(tp, fp, fn, tn, ind_pos=0, ind_neg=0, kit='kit', comp='ref'):
    out = []
    for _ in range(tp):
        out.append({kit: 'positive', comp: 'positive'})
    for _ in range(fp):
        out.append({kit: 'positive', comp: 'negative'})
    for _ in range(fn):
        out.append({kit: 'negative', comp: 'positive'})
    for _ in range(tn):
        out.append({kit: 'negative', comp: 'negative'})
    for _ in range(ind_pos):
        out.append({kit: 'indeterminate', comp: 'positive'})
    for _ in range(ind_neg):
        out.append({kit: 'indeterminate', comp: 'negative'})
    return out


def by_id(result, mid):
    return next(m for m in result['metrics'] if m['id'] == mid)


# ---------------------------------------------------------------------------
# An independent Clopper-Pearson, from its DEFINITION rather than its formula.
#
# The lower bound is the p at which the chance of seeing x or more successes is
# exactly alpha; the upper, the p at which the chance of x or fewer is alpha.
# Solved numerically here. That is the thing the Beta quantile is a shortcut
# for, so agreeing with it is evidence rather than a tautology.
# ---------------------------------------------------------------------------

def cp_from_tail(x, n, conf=0.95):
    a = (1 - conf) / 2
    lo = 0.0 if x == 0 else brentq(
        lambda p: binom.sf(x - 1, n, p) - a, 1e-12, 1 - 1e-12, xtol=1e-14)
    hi = 1.0 if x == n else brentq(
        lambda p: binom.cdf(x, n, p) - a, 1e-12, 1 - 1e-12, xtol=1e-14)
    return lo, hi


def main():
    # ---- 1. the interval, against three independent references ----------
    # Each fixture carries one true negative as well. Not padding: with 30/30
    # the table would hold the word "positive" and nothing else, and the label
    # inference REFUSES that rather than deciding on its own which single value
    # means positive — which is section 4's point, arrived at from here first.
    # Sensitivity is x/n either way; the extra row only lands in specificity.
    cases = [(142, 147), (326, 332), (99, 100), (57, 60), (30, 30), (0, 20), (1, 50)]
    worst = 0.0
    for x, n in cases:
        got = by_id(run({'data': rows(x, 0, n - x, 1), 'kit_col': 'kit',
                         'comparator_col': 'ref'}), 'sensitivity')
        want = cp_from_tail(x, n)
        worst = max(worst, abs(got['ci'][0] - want[0]), abs(got['ci'][1] - want[1]))
    check(worst < 1e-9,
          f'the interval matches Clopper-Pearson solved from the binomial tail '
          f'on {len(cases)} cases (worst gap {worst:.2e})')

    # The two closed forms. 30/30 is the one to keep in mind: the exact lower
    # bound is 88.4% and Wald says the uncertainty is zero.
    got = by_id(run({'data': rows(30, 0, 0, 1), 'kit_col': 'kit', 'comparator_col': 'ref'}),
                'sensitivity')
    hand = 0.025 ** (1 / 30)
    check(abs(got['ci'][0] - hand) < 1e-12 and got['ci'][1] == 1.0,
          f'with x = n the lower bound is alpha**(1/n) — {hand:.4f}, checkable by hand',
          got['ci'])
    got = by_id(run({'data': rows(0, 0, 20, 1), 'kit_col': 'kit', 'comparator_col': 'ref'}),
                'sensitivity')
    check(got['ci'][0] == 0.0 and abs(got['ci'][1] - (1 - 0.025 ** (1 / 20))) < 1e-12,
          'and with x = 0 the upper bound is 1 - alpha**(1/n), the mirror of it',
          got['ci'])

    # ---- 2. the goal is the BOUND, and the approximation disagrees ------
    res = run({'data': rows(99, 0, 1, 1), 'kit_col': 'kit', 'comparator_col': 'ref',
               'goals': {'sensitivity': 0.95}})
    m = by_id(res, 'sensitivity')
    check(m['point_verdict'] == 'pass' and m['wald_verdict'] == 'pass'
          and m['verdict'] == 'fail',
          f"99/100: the point estimate passes, Wald passes, the exact bound "
          f"({m['ci'][0] * 100:.1f}%) does NOT — this is the case the endpoint exists for",
          m)
    check(any('Wald' in (n.get('en') or '') for n in res['notes']),
          'and the response says so, rather than leaving a reader to notice')

    # The mockup's own numbers, which the screens quote.
    res = run({'data': rows(142, 6, 5, 326, ind_pos=3, ind_neg=4),
               'kit_col': 'kit', 'comparator_col': 'ref',
               'goals': {'sensitivity': 0.95, 'specificity': 0.98},
               'prevalences': [0.02, 0.05, 0.30]})
    sens, spec = by_id(res, 'sensitivity'), by_id(res, 'specificity')
    check(sens['x'] == 142 and sens['n'] == 147
          and spec['x'] == 326 and spec['n'] == 332,
          'indeterminate results come OUT of both denominators — 142/147 and 326/332, '
          'not /150 and /336',
          sens, spec)
    # Compared at the precision the SCREEN renders — one decimal of a percent,
    # which is what a report quotes. An earlier version of this check pinned
    # four decimals taken from a figure that had only ever been printed to one,
    # so two of them were invented and the check failed against correct code.
    def shown(v):
        return round(v * 100, 1)

    check(shown(sens['estimate']) == 96.6 and shown(sens['ci'][0]) == 92.2
          and shown(sens['ci'][1]) == 98.9
          and shown(spec['estimate']) == 98.2 and shown(spec['ci'][0]) == 96.1
          and shown(spec['ci'][1]) == 99.3,
          f"the figures the screen quotes: sensitivity {shown(sens['estimate'])}% "
          f"[{shown(sens['ci'][0])}, {shown(sens['ci'][1])}], specificity "
          f"{shown(spec['estimate'])}% [{shown(spec['ci'][0])}, {shown(spec['ci'][1])}]")
    check(sens['point_verdict'] == 'pass' and sens['verdict'] == 'fail'
          and spec['point_verdict'] == 'pass' and spec['verdict'] == 'fail',
          'both goals: met by the point estimate, missed by the bound')
    check(abs(by_id(res, 'indeterminate_rate')['estimate'] - 7 / 486) < 1e-12,
          f'and the indeterminate rate is reported on its own — 7/486',
          by_id(res, 'indeterminate_rate'))

    # `point` basis is available and says the opposite, so the plan's choice of
    # basis is doing real work rather than decorating the response.
    res2 = run({'data': rows(142, 6, 5, 326, ind_pos=3, ind_neg=4),
                'kit_col': 'kit', 'comparator_col': 'ref', 'goal_basis': 'point',
                'goals': {'sensitivity': 0.95}})
    check(by_id(res2, 'sensitivity')['verdict'] == 'pass',
          "on a 'point' basis the same data passes — which is why the plan has to fix "
          'the basis before anyone sees the number')

    # ---- 3. predictive values move with prevalence ----------------------
    pv = {round(p['prevalence'], 2): p for p in res['predictive']}
    check(abs(pv[0.02]['ppv'] - 0.522) < 2e-3 and abs(pv[0.30]['ppv'] - 0.958) < 2e-3,
          f"PPV at 2% prevalence is {pv[0.02]['ppv'] * 100:.1f}% and at 30% is "
          f"{pv[0.30]['ppv'] * 100:.1f}% — the same kit, which is why a prevalence has "
          f'to be named')
    # From sensitivity and specificity, not from this table's own mix: the
    # specimens were collected, not sampled from a population.
    se, sp, prev = sens['estimate'], spec['estimate'], 0.05
    want = se * prev / (se * prev + (1 - sp) * (1 - prev))
    check(abs(pv[0.05]['ppv'] - want) < 1e-12,
          'and is derived from sensitivity and specificity, not from the study mix')

    # ---- 4. which way round ---------------------------------------------
    straight = run({'data': rows(142, 6, 5, 326), 'kit_col': 'kit', 'comparator_col': 'ref'})
    swapped = run({'data': rows(142, 6, 5, 326), 'kit_col': 'ref', 'comparator_col': 'kit'})
    s1 = by_id(straight, 'sensitivity')['estimate']
    s2 = by_id(swapped, 'sensitivity')['estimate']
    check(abs(s1 - s2) > 0.005,
          f'swapping the two columns gives a DIFFERENT and still plausible sensitivity '
          f'({s1 * 100:.1f}% vs {s2 * 100:.1f}%) — nothing downstream could catch it, '
          f'which is why the labels are never guessed loosely')

    bad = run({'data': [{'kit': 'A', 'ref': 'B'}] * 10, 'kit_col': 'kit', 'comparator_col': 'ref'})
    check('__error' in bad and 'positive_label' in bad['__error'],
          'a column whose values it cannot read is an error naming them, not a guess',
          bad.get('__error', '')[:120])
    told = run({'data': [{'kit': 'A', 'ref': 'B'}, {'kit': 'B', 'ref': 'B'}],
                'kit_col': 'kit', 'comparator_col': 'ref',
                'positive_label': 'A', 'negative_label': 'B'})
    check('__error' not in told and by_id(told, 'sensitivity')['n'] == 0
          and by_id(told, 'specificity')['x'] == 1,
          'and the same columns work once the caller names them')

    unknown = run({'data': rows(5, 0, 0, 5) + [{'kit': 'weak positive', 'ref': 'negative'}],
                   'kit_col': 'kit', 'comparator_col': 'ref'})
    check('__error' in unknown and 'weak positive' in unknown['__error'],
          'a third category is named in the error, never binned as indeterminate — '
          'that would shrink a denominator and flatter the kit',
          unknown.get('__error', '')[:120])

    same = run({'data': rows(5, 0, 0, 5), 'kit_col': 'kit', 'comparator_col': 'kit'})
    check('__error' in same and 'itself' in same['__error'],
          'and a column compared with itself is refused rather than scored at 100%')

    # Korean labels read, and read the right way round.
    ko = run({'data': [{'kit': '양성', 'ref': '양성'}] * 9 + [{'kit': '음성', 'ref': '양성'}]
                      + [{'kit': '음성', 'ref': '음성'}] * 10,
              'kit_col': 'kit', 'comparator_col': 'ref'})
    check(by_id(ko, 'sensitivity')['x'] == 9 and by_id(ko, 'sensitivity')['n'] == 10,
          '양성 / 음성 are read, and 양성 is the positive one')

    # ---- 5. what it is called -------------------------------------------
    full = run({'data': rows(142, 6, 5, 326), 'kit_col': 'kit', 'comparator_col': 'ref',
                'reference_complete': True})
    part = run({'data': rows(142, 6, 5, 326), 'kit_col': 'kit', 'comparator_col': 'ref',
                'reference_complete': False})
    a, b = by_id(full, 'sensitivity'), by_id(part, 'sensitivity')
    check(a['estimate'] == b['estimate'] and a['ci'] == b['ci']
          and a['label']['en'] != b['label']['en'],
          f"an incomplete comparator changes the NAME and not one number: "
          f"\"{a['label']['ko']}\" -> \"{b['label']['ko']}\"")
    check(any('agreement' in (n.get('en') or '') for n in part['notes']),
          'and the response carries the distinction as a note a report can quote')

    # ---- 6. the analysis set ---------------------------------------------
    res = run({'data': rows(10, 0, 0, 10)
                      + [{'kit': 'positive', 'ref': None}] * 3
                      + [{'kit': 'positive', 'ref': 'indeterminate'}] * 2
                      + [{'kit': None, 'ref': 'positive'}] * 4,
               'kit_col': 'kit', 'comparator_col': 'ref'})
    d = res['row_counts']['dropped']
    check(res['row_counts']['n_input'] == 29 and res['row_counts']['n_analysed'] == 20
          and d['comparator_missing'] == 3 and d['comparator_indeterminate'] == 2
          and d['kit_missing'] == 4,
          'every specimen that left the analysis set is counted, by reason — '
          '29 in, 20 analysed, 3 + 2 + 4 out',
          res['row_counts'])

    empty = run({'data': [{'kit': 'positive', 'ref': None}], 'kit_col': 'kit',
                 'comparator_col': 'ref'})
    check('__error' in empty,
          'and a table where nothing is analysable says so rather than returning 0/0')

    # ---- 7. the response names its own method ---------------------------
    check(full['ci_method'] == 'clopper-pearson' and full['conf_level'] == 0.95,
          'the response states which interval it used — a report that quotes a bound '
          'has to be able to say what kind it is')

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
