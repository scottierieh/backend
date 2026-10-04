"""
ivd_plan_analysis.py — how many specimens a performance goal needs.

Route: /api/analysis/ivd-plan. The IVD Lab's step 01, which has no specimens
yet: given a goal and an assumed true performance, how many positive (or
negative) specimens before the goal is reachable at all.

NOT A POWER CALCULATION. The usual sample-size question is "what power does
this test have to detect a difference". This is a different question with a
different answer: a performance study does not test a hypothesis, it reports an
interval, and the goal is judged against that interval's bound. So the question
is HOW WIDE THE INTERVAL WILL BE -- and `power-analysis` cannot answer it,
which is why this exists rather than reusing it.

THE NUMBER IS NOT "THE FIRST n THAT WORKS", AND THAT MATTERS.

The observed count is a whole number of specimens, so the achieved proportion
wobbles above and below the assumed one as n grows, and so does the bound. The
first n that clears a goal is followed by a run of n that do not:

    assume 95%, goal 90%     first clears at 127, then 131-153 do NOT
    assume 99%, goal 98%     first clears at 650, then 651-784 do NOT

A plan written on 127 and executed with 131 specimens misses. So `n_required`
is the smallest n from which the goal holds for EVERY larger n within the
search -- a floor, not a first hit -- and `n_first_pass` is reported beside it
so a planner can see how far apart they are. The gap is 20-30% in ordinary
cases, which is the difference between a study that reads out and one that does
not.

CLI-script contract, like every *_analysis.py here: one JSON object in on
stdin, one JSON object out on stdout; on error print {"error": ...} to stderr
and exit(1).
"""

import json
import sys

import numpy as np
from scipy.stats import beta

from analysis_common import _to_native_type

_DEFAULT_MAX_N = 4000

# The metrics a plan can set a goal on. Deliberately the ids
# ivd_accuracy_analysis.py uses, so a plan's goals can be handed to it
# unchanged -- a second vocabulary between the two would be a mapping table
# nobody maintains.
_METRIC_LABELS = {
    'sensitivity': {
        True: {'en': 'Sensitivity', 'ko': '민감도'},
        False: {'en': 'Positive percent agreement', 'ko': '양성 일치율 (PPA)'},
    },
    'specificity': {
        True: {'en': 'Specificity', 'ko': '특이도'},
        False: {'en': 'Negative percent agreement', 'ko': '음성 일치율 (NPA)'},
    },
}

# Which specimens each metric's n counts. The planner needs this spelled out:
# a sensitivity goal is a constraint on how many POSITIVE specimens have to be
# collected, and a study that collects 400 specimens of which 30 are positive
# has met neither.
_COUNTS = {
    'sensitivity': {'en': 'comparator-positive specimens', 'ko': '기준검사 양성 검체'},
    'specificity': {'en': 'comparator-negative specimens', 'ko': '기준검사 음성 검체'},
}


def _err(msg):
    print(json.dumps({'error': str(msg)}), file=sys.stderr)
    sys.exit(1)


def _bounds(assume: float, conf: float, max_n: int):
    """The Clopper-Pearson lower bound at every n from 2 to max_n.

    Vectorised, and the same quantile ivd_accuracy_analysis.py uses -- the two
    have to agree about the interval or a plan is written against a bound the
    analysis will not reproduce.
    """
    n = np.arange(2, max_n + 1)
    x = np.rint(assume * n).astype(int)
    alpha = (1.0 - conf) / 2.0
    lo = np.where(x <= 0, 0.0, beta.ppf(alpha, np.maximum(x, 1), n - x + 1))
    return n, x, lo


def _requirement(mid, goal, assume, conf, max_n, reference_complete):
    label = (_METRIC_LABELS.get(mid) or {}).get(reference_complete) \
        or {'en': mid, 'ko': mid}
    out = {
        'id': mid,
        'label': label,
        'counts': _COUNTS.get(mid, {'en': 'specimens', 'ko': '검체'}),
        'goal': goal,
        'assume': assume,
        'n_required': None,
        'n_first_pass': None,
        'x_at_required': None,
        'bound_at_required': None,
        'reachable': False,
        'searched_to': max_n,
    }

    # A goal at or above the assumed performance is not a sample-size problem.
    # The bound converges UP to the assumed value, so it never reaches a goal
    # that sits on or above it, at any n. This is a plan to change, not a study
    # to enlarge, and saying "more specimens" here would be wrong advice.
    if assume <= goal:
        return out

    n, x, lo = _bounds(assume, conf, max_n)
    passes = lo >= goal
    if not passes.any():
        return out

    out['reachable'] = True
    out['n_first_pass'] = int(n[np.argmax(passes)])

    fails = np.flatnonzero(~passes)
    # The floor: one past the last n that fails. Exact within the search, and
    # the response says what it searched to rather than implying more.
    need_idx = (fails[-1] + 1) if len(fails) else 0
    if need_idx >= len(n):
        # Still failing at the cap — reachable somewhere, but not provably
        # inside the window, so no floor is reported rather than a guess.
        out['reachable'] = False
        return out

    out['n_required'] = int(n[need_idx])
    out['x_at_required'] = int(x[need_idx])
    out['bound_at_required'] = float(lo[need_idx])
    return out


def main():
    try:
        payload = json.load(sys.stdin)
        goals = payload.get('goals') or {}
        assume = payload.get('assume') or {}
        if not goals:
            _err('No goals to size. Send goals, e.g. {"sensitivity": 0.95}.')

        conf = float(payload.get('conf_level', 0.95))
        if not 0.5 < conf < 1:
            _err(f'conf_level must be between 0.5 and 1, got {conf}.')
        max_n = int(payload.get('max_n', _DEFAULT_MAX_N))
        if not 10 <= max_n <= 200000:
            _err(f'max_n must be between 10 and 200000, got {max_n}.')
        basis = payload.get('goal_basis', 'lower_bound')
        if basis not in ('lower_bound', 'point'):
            _err("goal_basis must be 'lower_bound' or 'point'.")
        reference_complete = bool(payload.get('reference_complete', True))

        unknown = [k for k in goals if k not in _METRIC_LABELS]
        if unknown:
            _err(f'Goals can be set on {sorted(_METRIC_LABELS)}; got {unknown}.')

        requirements = []
        for mid, goal in goals.items():
            goal = float(goal)
            if not 0 < goal < 1:
                _err(f"The goal for '{mid}' must be between 0 and 1, got {goal}.")
            a = assume.get(mid)
            if a is None:
                _err(f"No assumed performance for '{mid}'. A sample size needs one: "
                     f'the answer is "how many before the bound clears the goal", and '
                     f'that depends entirely on how far above the goal the kit really is.')
            a = float(a)
            if not 0 < a <= 1:
                _err(f"The assumed '{mid}' must be between 0 and 1, got {a}.")
            requirements.append(
                _requirement(mid, goal, a, conf, max_n, reference_complete))

        notes = []
        if basis == 'point':
            notes.append({
                'en': 'This plan judges the point estimate, so there is no sample size to '
                      'compute — a single specimen scoring above the goal would pass. The '
                      'figures below are what a bound-based plan would need, and the gap '
                      'between them is what judging the point estimate gives up.',
                'ko': '이 계획은 점추정으로 판정하므로 계산할 표본수가 없습니다 — 검체 한 '
                      '건이 기준을 넘으면 통과입니다. 아래 수치는 경계로 판정하는 계획이 '
                      '필요한 양이고, 그 차이가 점추정 판정이 포기하는 것입니다.',
            })
        for r in requirements:
            if r['assume'] <= r['goal']:
                notes.append({
                    'en': f"The assumed {r['label']['en'].lower()} ({r['assume']:.1%}) is not "
                          f"above its goal ({r['goal']:.1%}), so no number of specimens makes "
                          f"the bound reach it. Change the goal or the assumption, not the "
                          f"sample size.",
                    'ko': f"가정한 {r['label']['ko']}({r['assume']:.1%})가 기준"
                          f"({r['goal']:.1%})보다 높지 않아서, 검체를 얼마나 모아도 경계가 "
                          f"기준에 닿지 않습니다. 표본수가 아니라 기준이나 가정을 바꿔야 "
                          f"합니다.",
                })
            elif r['n_required'] is None:
                notes.append({
                    'en': f"{r['label']['en']} needs more than {max_n} specimens at the "
                          f"assumed {r['assume']:.1%}. Raise max_n to see how many, or widen "
                          f"the gap between the goal and the assumption.",
                    'ko': f"{r['label']['ko']}는 가정 {r['assume']:.1%}에서 {max_n}건보다 "
                          f"많이 필요합니다. max_n을 올려 보거나, 기준과 가정의 간격을 "
                          f"벌려야 합니다.",
                })
            elif r['n_first_pass'] and r['n_required'] > r['n_first_pass']:
                notes.append({
                    'en': f"{r['label']['en']}: the bound first clears the goal at "
                          f"{r['n_first_pass']}, but not at every n above it — the observed "
                          f"count is a whole number of specimens, so the bound wobbles. "
                          f"{r['n_required']} is the floor from which it holds throughout, "
                          f"and is the number to plan on.",
                    'ko': f"{r['label']['ko']}: 경계가 기준을 처음 넘는 것은 "
                          f"{r['n_first_pass']}건이지만 그 위의 모든 n에서 넘는 것은 "
                          f"아닙니다 — 관측 건수가 정수라서 경계가 흔들립니다. 계획에 쓸 "
                          f"수는 줄곧 넘는 하한인 {r['n_required']}건입니다.",
                })

        print(json.dumps({
            'ci_method': 'clopper-pearson',
            'conf_level': conf,
            'goal_basis': basis,
            'reference_complete': reference_complete,
            'searched_to': max_n,
            'requirements': requirements,
            'notes': notes,
        }, default=_to_native_type))

    except Exception as e:  # noqa: BLE001 — CLI contract: any failure -> stderr + exit(1)
        _err(e)


if __name__ == '__main__':
    main()
