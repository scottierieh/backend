"""
ivd_accuracy_analysis.py — a diagnostic kit against its comparator.

Route: /api/analysis/ivd-accuracy. The IVD Lab's section 03: one row per
specimen, the kit's result beside the comparator's, and out of it the table
every clinical performance study report opens with.

WHY THIS IS NOT A 2x2 HELPER. Three decisions in it are easy to get wrong in a
way that produces a perfectly ordinary-looking number:

  1. THE INTERVAL. A performance goal is judged against a confidence bound, not
     a point estimate, and the usual Wald approximation breaks exactly where
     these studies live -- proportions near 1, modest n. On 99/100 it returns
     97.0-101.0%: an upper bound above 100%, and a LOWER bound that clears a
     95% goal the exact interval (94.6%) does not. The approximation hands out
     a pass. So this computes Clopper-Pearson and nothing else, and says so in
     the response.

  2. WHICH WAY ROUND. Sensitivity with the kit and the comparator swapped is
     still a number between 0 and 1, and on a good kit it is a plausible one.
     Nothing downstream can detect it. So the positive and negative labels are
     either given or inferred from a known vocabulary, and a column this cannot
     read is an error naming what it found -- never a guess.

  3. WHAT IT IS CALLED. When the comparator is not a complete reference
     standard -- the common case -- the same arithmetic may not be called
     sensitivity and specificity. It is positive and negative percent
     agreement. Identical computation, different claim, and regulators read the
     difference. `reference_complete` picks the names; it changes no number.

Indeterminate results come out of the denominators and are reported as their
own rate. A specimen the kit could not read is not evidence about the kit's
accuracy, but hiding it inflates both metrics, so it is counted in the open.

INDETERMINATE, INVALID AND RETEST (section 05). Three things that get merged
and should not be: a kit that answered "unclear", a kit that produced no valid
result at all, and a kit whose answer CHANGED when the same specimen was run
again. The first two leave the metric denominators by different routes -- an
indeterminate specimen is in the analysis set and out of the denominators, an
invalid one never entered it -- and the third never shows up in a 2x2 at all
while being the plainest repeatability signal in the data. `retest_col` is the
kit's result on a repeat, and `use_retest` applies the plan's rule for it, with
the first-result reading staying the top level for the same reason the
unresolved one does.

DISCORDANT RESOLUTION (section 06). When the plan says a specimen the two
methods disagree on is re-adjudicated by a third method, that changes the
table. Both tables are returned from ONE request, computed by one function on
one set of rows, so a screen cannot show the resolved figures without the
unresolved ones beside them — and the TOP LEVEL is the unresolved reading,
because a resolution can only move a specimen in the direction that helps the
kit. Only the COMPARATOR is ever re-adjudicated: the kit's result is the thing
under test.

CLI-script contract, like every *_analysis.py here: one JSON object in on
stdin, one JSON object out on stdout; on error print {"error": ...} to stderr
and exit(1).
"""

import json
import sys

import numpy as np
from scipy.stats import beta, norm

from analysis_common import _to_native_type


def _err(msg):
    print(json.dumps({'error': str(msg)}), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# The interval
# ---------------------------------------------------------------------------

def clopper_pearson(x: int, n: int, conf: float = 0.95):
    """The exact binomial interval, as Beta quantiles.

    Exact in the sense that its coverage is never BELOW the nominal level --
    it is conservative, which is the direction a performance claim should err
    in. The two closed forms at the ends are the reason it is trusted here:
    with x = n the lower bound is alpha**(1/n), a number anyone can check by
    hand (30/30 at 95% -> 0.025**(1/30) = 0.8843), where Wald reports that the
    uncertainty is zero.
    """
    if n <= 0:
        return None, None
    alpha = (1.0 - conf) / 2.0
    lo = 0.0 if x <= 0 else float(beta.ppf(alpha, x, n - x + 1))
    hi = 1.0 if x >= n else float(beta.ppf(1.0 - alpha, x + 1, n - x))
    return lo, hi


def _wald(x: int, n: int, conf: float = 0.95):
    """Only ever reported as a FOIL, never as a result.

    The response carries it beside the exact interval when the two disagree
    about a goal, because "the approximation would have passed this" is the one
    thing that makes the choice of interval visible to a reader.
    """
    if n <= 0:
        return None, None
    z = float(norm.ppf(1 - (1 - conf) / 2))
    p = x / n
    se = (p * (1 - p) / n) ** 0.5
    return p - z * se, p + z * se


# ---------------------------------------------------------------------------
# Reading the two columns
# ---------------------------------------------------------------------------

# A result means positive or it means negative, and which one decides the
# direction of every metric below. These are the tokens this will read without
# being told; anything else has to be named by the caller. Deliberately not a
# heuristic -- no "starts with p", no alphabetical order.
_POSITIVE_TOKENS = {
    'positive', 'pos', 'p', '+', '1', 'true', 'yes', 'y', 'detected',
    'reactive', 'present', 'abnormal',
    '양성', '검출', '있음', '유',
}
_NEGATIVE_TOKENS = {
    'negative', 'neg', 'n', '-', '0', 'false', 'no', 'not detected',
    'nondetected', 'non-reactive', 'nonreactive', 'absent', 'normal',
    '음성', '미검출', '없음', '무',
}
# Not an answer. Kept out of the denominators and reported on its own.
_INDETERMINATE_TOKENS = {
    'indeterminate', 'equivocal', 'inconclusive', 'invalid', 'borderline',
    'unclear', 'na', 'n/a', 'error', 'void',
    '판정보류', '보류', '미결정', '무효', '재검',
}

_MISSING = {'', 'none', 'null', 'nan', '-9', '.'}


def _norm(v) -> str:
    if v is None:
        return ''
    if isinstance(v, float) and np.isnan(v):
        return ''
    s = str(v).strip().lower()
    # 1.0 and 1 are the same answer; a float that came out of a spreadsheet
    # should not be a third category.
    if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
        s = s[:-2]
    return s


def _classify(values, positive, negative, indeterminate, column: str):
    """Each value as 'pos' | 'neg' | 'ind' | 'missing'. Unknown values raise.

    An unrecognised value is NOT quietly binned as indeterminate: that would
    turn a typo, or a third category nobody mentioned, into a silently smaller
    denominator and a better-looking kit.
    """
    pos = {_norm(v) for v in positive}
    neg = {_norm(v) for v in negative}
    ind = {_norm(v) for v in indeterminate}
    out = []
    unknown = {}
    for v in values:
        s = _norm(v)
        if s in _MISSING:
            out.append('missing')
        elif s in pos:
            out.append('pos')
        elif s in neg:
            out.append('neg')
        elif s in ind:
            out.append('ind')
        else:
            out.append('unknown')
            unknown[str(v)] = unknown.get(str(v), 0) + 1
    if unknown:
        named = ', '.join(f'{k!r} ({n})' for k, n in sorted(
            unknown.items(), key=lambda kv: -kv[1])[:6])
        raise ValueError(
            f"Column '{column}' holds values this cannot read as a result: {named}. "
            f"Name them with positive_label / negative_label / indeterminate_labels — "
            f"guessing would change which metric is which."
        )
    return out


def _infer_labels(values, column: str):
    """The positive and negative labels, from the vocabulary above.

    Returns (positive, negative, indeterminate) as lists of the values as they
    actually appear, so the response can echo what it read.
    """
    seen = {}
    for v in values:
        s = _norm(v)
        if s in _MISSING:
            continue
        seen.setdefault(s, v)

    positive = [orig for s, orig in seen.items() if s in _POSITIVE_TOKENS]
    negative = [orig for s, orig in seen.items() if s in _NEGATIVE_TOKENS]
    indeterminate = [orig for s, orig in seen.items() if s in _INDETERMINATE_TOKENS]
    leftover = [orig for s, orig in seen.items() if s not in _POSITIVE_TOKENS
                and s not in _NEGATIVE_TOKENS and s not in _INDETERMINATE_TOKENS]

    if leftover or not positive or not negative:
        found = ', '.join(repr(str(v)) for v in sorted(seen.values(), key=str)[:8])
        raise ValueError(
            f"Could not tell which value of '{column}' means positive. Found: {found}. "
            f"Send positive_label and negative_label — reading them the wrong way round "
            f"swaps sensitivity and specificity and still returns a plausible number."
        )
    return positive, negative, indeterminate


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------

_NAMES = {
    True: {
        'sensitivity': {'en': 'Sensitivity', 'ko': '민감도'},
        'specificity': {'en': 'Specificity', 'ko': '특이도'},
    },
    False: {
        'sensitivity': {'en': 'Positive percent agreement', 'ko': '양성 일치율 (PPA)'},
        'specificity': {'en': 'Negative percent agreement', 'ko': '음성 일치율 (NPA)'},
    },
}


def _metric(mid, x, n, conf, goals, basis, reference_complete, extra_label=None):
    est = (x / n) if n else None
    lo, hi = clopper_pearson(x, n, conf)
    name = _NAMES[reference_complete].get(mid) or extra_label or {'en': mid, 'ko': mid}
    out = {
        'id': mid,
        'label': name,
        'x': int(x),
        'n': int(n),
        'estimate': est,
        'ci': [lo, hi],
        'goal': None,
        'basis': basis,
        'verdict': None,
        # What the point estimate alone would have said. The gap between this
        # and `verdict` is the reason the section exists, so it is a field
        # rather than something a screen has to recompute.
        'point_verdict': None,
        'wald_verdict': None,
    }
    goal = (goals or {}).get(mid)
    if goal is None or n == 0:
        return out
    goal = float(goal)
    out['goal'] = goal
    out['point_verdict'] = 'pass' if est >= goal else 'fail'
    judged = lo if basis == 'lower_bound' else est
    out['verdict'] = 'pass' if judged is not None and judged >= goal else 'fail'
    if basis == 'lower_bound':
        wlo, _ = _wald(x, n, conf)
        out['wald_verdict'] = 'pass' if wlo is not None and wlo >= goal else 'fail'
    return out


def _analyse(kit, comp, keep, conf, goals, basis, reference_complete, prevalences):
    """The table, the metrics and the predictive values, from one reading of the
    two columns.

    A function because it is called TWICE when the plan specifies a discordant
    resolution: once on the comparator as observed and once on the comparator
    as re-adjudicated. Two copies of this arithmetic could disagree about the
    before and the after, which is the one comparison section 06 exists to make.
    """
    cell = {(a, b): 0 for a in ('pos', 'neg', 'ind') for b in ('pos', 'neg')}
    for i in keep:
        cell[(kit[i], comp[i])] += 1

    tp = cell[('pos', 'pos')]
    fp = cell[('pos', 'neg')]
    fn = cell[('neg', 'pos')]
    tn = cell[('neg', 'neg')]
    ind_pos = cell[('ind', 'pos')]
    ind_neg = cell[('ind', 'neg')]

    # Indeterminate out of the denominators. Its own rate is reported below;
    # leaving them in as errors would punish the kit for not answering, and
    # dropping them without a figure would flatter it.
    sens = _metric('sensitivity', tp, tp + fn, conf, goals, basis, reference_complete)
    spec = _metric('specificity', tn, tn + fp, conf, goals, basis, reference_complete)
    n_analysed = len(keep)
    ind_rate = _metric(
        'indeterminate_rate', ind_pos + ind_neg, n_analysed, conf, goals, basis,
        reference_complete,
        extra_label={'en': 'Indeterminate rate', 'ko': '판정보류율'})

    # Predictive values at the prevalences the PLAN named. Derived from
    # sensitivity and specificity rather than from this table's own prevalence,
    # which is a property of how the specimens were collected and almost never
    # the prevalence the kit will be used at.
    predictive = []
    if sens['estimate'] is not None and spec['estimate'] is not None:
        se, sp = sens['estimate'], spec['estimate']
        for pv in prevalences:
            pv = float(pv)
            ppv_den = se * pv + (1 - sp) * (1 - pv)
            npv_den = (1 - se) * pv + sp * (1 - pv)
            predictive.append({
                'prevalence': pv,
                'ppv': (se * pv / ppv_den) if ppv_den > 0 else None,
                'npv': (sp * (1 - pv) / npv_den) if npv_den > 0 else None,
            })

    return {
        'n_analysed': n_analysed,
        'table': {
            'kit_positive': {'comparator_positive': tp, 'comparator_negative': fp},
            'kit_negative': {'comparator_positive': fn, 'comparator_negative': tn},
            'kit_indeterminate': {'comparator_positive': ind_pos,
                                  'comparator_negative': ind_neg},
            'comparator_positive_total': tp + fn + ind_pos,
            'comparator_negative_total': fp + tn + ind_neg,
        },
        'metrics': [sens, spec, ind_rate],
        'predictive': predictive,
    }


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        if not isinstance(data, list) or not data:
            _err('No specimen rows provided.')

        # Every column name any row carries. Checking only the FIRST row looks
        # right and is not: a caller that omits a key where the value is empty
        # — which is ordinary for a column only a few specimens have, like a
        # retest result — would be told its column does not exist. The error
        # would be about the first row and read as being about the table.
        present = set()
        for row in data:
            if isinstance(row, dict):
                present.update(row.keys())

        kit_col = payload.get('kit_col')
        comp_col = payload.get('comparator_col')
        if not kit_col or not comp_col:
            _err('kit_col and comparator_col are both required.')
        if kit_col == comp_col:
            _err('kit_col and comparator_col are the same column — that compares '
                 'a result with itself and agrees perfectly.')

        missing_cols = [c for c in (kit_col, comp_col) if c not in present]
        if missing_cols:
            _err(f'Column(s) not found in the specimen table: {missing_cols}')

        conf = float(payload.get('conf_level', 0.95))
        if not 0.5 < conf < 1:
            _err(f'conf_level must be between 0.5 and 1, got {conf}.')
        reference_complete = bool(payload.get('reference_complete', True))
        basis = payload.get('goal_basis', 'lower_bound')
        if basis not in ('lower_bound', 'point'):
            _err("goal_basis must be 'lower_bound' or 'point'.")
        goals = payload.get('goals') or {}

        kit_raw = [r.get(kit_col) for r in data]
        comp_raw = [r.get(comp_col) for r in data]

        # Labels: given, or read from the vocabulary. The kit column decides
        # them, and the comparator is read with the same ones — two columns of
        # the same measurement should not be allowed to use different words for
        # positive without the caller saying so.
        pos_l = payload.get('positive_label')
        neg_l = payload.get('negative_label')
        ind_l = payload.get('indeterminate_labels')
        if pos_l is None or neg_l is None:
            pos_l, neg_l, inferred_ind = _infer_labels(kit_raw + comp_raw, kit_col)
            labels_from = 'inferred'
        else:
            inferred_ind = []
            labels_from = 'given'
        pos_l = pos_l if isinstance(pos_l, list) else [pos_l]
        neg_l = neg_l if isinstance(neg_l, list) else [neg_l]
        if ind_l is None:
            ind_l = inferred_ind
        ind_l = ind_l if isinstance(ind_l, list) else [ind_l]

        kit = _classify(kit_raw, pos_l, neg_l, ind_l, kit_col)
        comp = _classify(comp_raw, pos_l, neg_l, ind_l, comp_col)

        # The analysis set. A specimen with no comparator result cannot be
        # scored against anything, and one whose comparator is indeterminate is
        # the same: there is no answer to be right or wrong about. Both are
        # counted out loud rather than dropped.
        dropped = {'comparator_missing': 0, 'comparator_indeterminate': 0, 'kit_missing': 0}
        keep = []
        for i, (k, c) in enumerate(zip(kit, comp)):
            if c == 'missing':
                dropped['comparator_missing'] += 1
            elif c == 'ind':
                dropped['comparator_indeterminate'] += 1
            elif k == 'missing':
                dropped['kit_missing'] += 1
            else:
                keep.append(i)
        if not keep:
            _err('No specimen has both a kit result and a comparator result.')

        prevalences = payload.get('prevalences') or []
        for pv in prevalences:
            if not 0 < float(pv) < 1:
                _err(f'Each prevalence must be between 0 and 1, got {pv}.')

        observed = _analyse(kit, comp, keep, conf, goals, basis,
                            reference_complete, prevalences)

        # ---- the discordant specimens, and the plan's resolution -------
        #
        # Always listed, resolutions or not: a screen cannot offer to
        # adjudicate specimens it was never told about.
        names = None
        spec_col = payload.get('specimen_col')
        if spec_col:
            if spec_col not in present:
                _err(f"specimen_col '{spec_col}' is not a column of the specimen table.")
            names = [str(r.get(spec_col)) for r in data]

        def name_of(i):
            return names[i] if names else f'row {i + 1}'

        discordant = []
        for i in keep:
            if kit[i] == 'pos' and comp[i] == 'neg':
                kind = 'kit_positive'
            elif kit[i] == 'neg' and comp[i] == 'pos':
                kind = 'kit_negative'
            else:
                continue
            discordant.append({'specimen': name_of(i), 'row': i, 'kind': kind,
                               'resolved_to': None, 'method': None})
        by_name = {d['specimen']: d for d in discordant}

        resolutions = payload.get('resolutions') or []
        resolved_block = None
        if resolutions:
            if not spec_col:
                _err('Resolutions name specimens, so specimen_col is required to apply '
                     'them — matching by row order would silently re-adjudicate the '
                     'wrong specimen if the table were ever re-sorted.')
            comp2 = list(comp)
            seen = set()
            for r in resolutions:
                who = str(r.get('specimen', ''))
                if who in seen:
                    _err(f"Specimen {who!r} is resolved twice; one adjudication each.")
                seen.add(who)
                target = by_name.get(who)
                if target is None:
                    _err(f"Specimen {who!r} is not one of the {len(discordant)} discordant "
                         f'specimens. Only a specimen the two methods disagree on can be '
                         f're-adjudicated — resolving an agreeing one would be changing a '
                         f'result nobody disputed.')
                verdict = _norm(r.get('comparator'))
                if verdict in {_norm(v) for v in pos_l}:
                    new = 'pos'
                elif verdict in {_norm(v) for v in neg_l}:
                    new = 'neg'
                else:
                    _err(f"The resolution for {who!r} must say what the COMPARATOR is now "
                         f'(one of the positive or negative labels), got '
                         f'{r.get("comparator")!r}. The kit\'s own result is the thing '
                         f'under test and is never re-adjudicated.')
                comp2[target['row']] = new
                target['resolved_to'] = new
                target['method'] = r.get('method') or None

            # The analysis set can only grow or stay the same: a resolution
            # turns an indeterminate or missing comparator into an answer, and
            # never the other way.
            keep2 = [i for i, (k, c) in enumerate(zip(kit, comp2))
                     if c in ('pos', 'neg') and k != 'missing']
            resolved_block = _analyse(kit, comp2, keep2, conf, goals, basis,
                                      reference_complete, prevalences)

        # ---- section 05: indeterminate, invalid, and the retest ---------
        retest_col = payload.get('retest_col')
        retest = None
        if retest_col:
            if retest_col not in present:
                _err(f"retest_col '{retest_col}' is not a column of the specimen table.")
            retest = _classify([r.get(retest_col) for r in data],
                               pos_l, neg_l, ind_l, retest_col)

        # An indeterminate specimen is IN the analysis set and out of the metric
        # denominators. An invalid one never entered it. Two different routes
        # out of a 2x2 and two different things to report, so two rates.
        comparator_known = [i for i, c in enumerate(comp) if c in ('pos', 'neg')]
        invalid_n = len(comparator_known)
        invalid_x = sum(1 for i in comparator_known if kit[i] == 'missing')
        repeat = {
            'indeterminate': _metric(
                'indeterminate_rate', observed['metrics'][2]['x'], observed['n_analysed'],
                conf, {}, basis, reference_complete,
                extra_label={'en': 'Indeterminate', 'ko': '판정보류'}),
            'invalid': _metric(
                'invalid_rate', invalid_x, invalid_n, conf, {}, basis, reference_complete,
                extra_label={'en': 'No valid result', 'ko': '무효'}),
            'retested': 0,
            'resolved_by_retest': 0,
            'changed_on_retest': 0,
            'changed': [],
            'use_retest': False,
        }

        retested_block = None
        if retest is not None:
            unclear = ('ind', 'missing')
            for i in comparator_known:
                if retest[i] in unclear:
                    continue
                repeat['retested'] += 1
                if kit[i] in unclear:
                    repeat['resolved_by_retest'] += 1
                elif retest[i] != kit[i]:
                    # The same specimen, the same kit, a different answer. It
                    # never appears in a 2x2 and it is the plainest
                    # repeatability signal the study holds.
                    repeat['changed_on_retest'] += 1
                    repeat['changed'].append({
                        'specimen': name_of(i),
                        'first': kit[i],
                        'retest': retest[i],
                        'comparator': comp[i],
                    })

            if payload.get('use_retest'):
                repeat['use_retest'] = True
                # The plan's rule: a repeat result stands in for a first result
                # that was unclear or invalid. It never overrides a first result
                # the kit gave cleanly — that would be choosing between two
                # answers after seeing both.
                kit2 = [retest[i] if (kit[i] in unclear and retest[i] not in unclear)
                        else kit[i] for i in range(len(kit))]
                keep2 = [i for i, (k, c) in enumerate(zip(kit2, comp))
                         if c in ('pos', 'neg') and k != 'missing']
                retested_block = _analyse(kit2, comp, keep2, conf, goals, basis,
                                          reference_complete, prevalences)

        notes = []
        if not reference_complete:
            notes.append({
                'en': 'The comparator is not a complete reference standard, so these are '
                      'percent agreement, not sensitivity and specificity. The arithmetic '
                      'is the same; the claim is not.',
                'ko': '기준검사가 완전한 표준이 아니므로 이 값은 민감도·특이도가 아니라 '
                      '일치율입니다. 계산은 같고 주장이 다릅니다.',
            })
        _scored = [m for m in observed['metrics'] if m['id'] != 'indeterminate_rate']
        if any(m['goal'] is not None and m['verdict'] != m['point_verdict'] for m in _scored):
            notes.append({
                'en': 'At least one goal is met by the point estimate and missed by the '
                      'confidence bound. The plan judges the bound.',
                'ko': '점추정으로는 넘고 신뢰구간 경계로는 못 넘는 기준이 있습니다. '
                      '계획이 판정하는 것은 경계입니다.',
            })
        if any(m['goal'] is not None and m['wald_verdict'] == 'pass' and m['verdict'] == 'fail'
               for m in _scored):
            notes.append({
                'en': 'A Wald approximation would have passed a goal the exact interval '
                      'does not. This endpoint reports Clopper-Pearson only.',
                'ko': 'Wald 근사로는 통과하는 기준을 정확구간은 통과시키지 않습니다. '
                      '이 분석은 Clopper-Pearson만 보고합니다.',
            })
        if repeat['changed_on_retest'] > 0:
            notes.append({
                'en': f"{repeat['changed_on_retest']} specimen(s) got a DIFFERENT answer "
                      f'when the same specimen was run again. That never appears in the '
                      f'table above — the first result is what is scored — and it is a '
                      f'repeatability finding in its own right.',
                'ko': f"{repeat['changed_on_retest']}건은 같은 검체를 다시 검사했을 때 "
                      f'다른 답이 나왔습니다. 위 표에는 전혀 나타나지 않고(채점되는 것은 '
                      f'첫 결과입니다), 그 자체로 재현성에 대한 결과입니다.',
            })
        if retested_block is not None:
            notes.append({
                'en': "The retest reading replaces a first result that was unclear or "
                      'invalid, never one the kit gave cleanly — choosing between two '
                      'clean answers after seeing both is not a rule, it is a preference. '
                      'The first-result reading stays the top level.',
                'ko': '재검 결과는 애매하거나 무효였던 첫 결과를 대신하고, 키트가 분명하게 '
                      '낸 결과는 절대 덮지 않습니다 — 분명한 두 답을 보고 나서 고르는 것은 '
                      '규칙이 아니라 선호입니다. 첫 결과 기준이 최상위로 남습니다.',
            })
        if resolved_block is not None:
            notes.append({
                'en': 'Both tables are reported. A resolution can only move a specimen in '
                      'the direction that helps the kit — a re-adjudication that confirms '
                      'the comparator changes nothing — so the unresolved figures are the '
                      'ones that rest on no judgement, and are the top-level result here.',
                'ko': '두 표를 모두 보고합니다. 해결은 검체를 키트에 유리한 방향으로만 '
                      '움직일 수 있고(기준검사를 확인하는 재판정은 아무것도 바꾸지 '
                      '않습니다), 그래서 판단이 개입하지 않은 해결 전 수치가 이 응답의 '
                      '최상위 결과입니다.',
            })

        print(json.dumps({
            'ci_method': 'clopper-pearson',
            'conf_level': conf,
            'reference_complete': reference_complete,
            'goal_basis': basis,
            'labels': {
                'from': labels_from,
                'positive': [str(v) for v in pos_l],
                'negative': [str(v) for v in neg_l],
                'indeterminate': [str(v) for v in ind_l],
            },
            'row_counts': {
                'n_input': len(data),
                'n_analysed': observed['n_analysed'],
                'dropped': dropped,
            },
            # The top level is the comparator AS OBSERVED. Deliberately: a
            # caller that ignores `resolved` below gets the figures that rest
            # on no judgement, which is the right way round for a default.
            'table': observed['table'],
            'metrics': observed['metrics'],
            'predictive': observed['predictive'],
            'discordant': discordant,
            'resolved': resolved_block,
            'repeat': repeat,
            'retested': retested_block,
            'notes': notes,
        }, default=_to_native_type))

    except Exception as e:  # noqa: BLE001 — CLI contract: any failure -> stderr + exit(1)
        _err(e)


if __name__ == '__main__':
    main()
