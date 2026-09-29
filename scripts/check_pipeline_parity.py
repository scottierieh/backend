"""check_pipeline_parity.py — two engines, one recipe.

    python scripts/check_pipeline_parity.py          # a few seconds

The Feature Engineering recipe is implemented twice: once in the browser
(src/lib/feature-engineering/transforms.ts) so the screen can show what the
table becomes, and once here (feature_pipeline.py) because /train re-fits it
server-side rather than trusting rows the client already transformed.

That is the right design and it has one failure mode, which is silent. A
column named `signed_up_dow` on one side and `signed_up_weekday` on the other,
a Sunday numbered 0 in the browser and 6 here, a group unseen in training
falling back to the overall mean there and to NaN here -- each one is a model
trained on different features than the screen showed, and none of them raises.
Both sides return a well-shaped table.

So the TS engine writes its answer out
(src/lib/feature-engineering/__tests__/emit-parity-fixture.ts ->
scripts/fixtures/pipeline-parity.json) and this reproduces it. Regenerate the
fixture whenever the recipe changes; a stale fixture fails here rather than
passing quietly, because the steps are read from it too.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from feature_pipeline import FeatureEngineer  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'statistica-frontend', 'scripts', 'fixtures', 'pipeline-parity.json')

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


def same(a, b) -> bool:
    """One cell, compared the way the two languages can agree."""
    a_missing = a is None or (isinstance(a, float) and math.isnan(a))
    b_missing = b is None or (isinstance(b, float) and math.isnan(b))
    if a_missing or b_missing:
        return a_missing and b_missing
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # Group means are sums in a different order; an exact match would be
        # a check on float addition, not on the recipe.
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return str(a) == str(b)


def compare(label, expected_table, got: pd.DataFrame):
    exp_headers = expected_table['headers']
    check(list(got.columns) == exp_headers,
          f'{label}: the same columns, in the same order',
          f'browser: {exp_headers}', f'here:    {list(got.columns)}')
    if list(got.columns) != exp_headers:
        return

    mismatches = []
    for i, row in enumerate(expected_table['rows']):
        for h in exp_headers:
            a = row.get(h)
            b = got.iloc[i][h]
            if not same(a, b):
                mismatches.append(f'row {i} · {h}: browser {a!r} vs here {b!r}')
    check(not mismatches, f'{label}: every cell agrees ({len(expected_table["rows"])} rows '
                          f'x {len(exp_headers)} columns)', *mismatches[:8])


def main():
    if not os.path.exists(FIXTURE):
        print(f'FAIL  the fixture is missing: {FIXTURE}')
        print('      npx tsx src/lib/feature-engineering/__tests__/emit-parity-fixture.ts')
        return 1
    fx = json.load(open(FIXTURE, encoding='utf-8'))

    kinds = {s['kind'] for s in fx['steps']}
    check(kinds >= {'date_parts', 'text_stats', 'group_stats'},
          f'the fixture exercises the new transforms: {sorted(kinds)}')

    train = pd.DataFrame(fx['train']['rows'], columns=fx['train']['headers'])
    holdout = pd.DataFrame(fx['holdout']['rows'], columns=fx['holdout']['headers'])

    fe = FeatureEngineer(fx['steps'])
    fe.fit(train)

    compare('train', fx['train_out'], fe.transform(train))
    # The half the recipe was NOT fitted on. Every fallback lives here: an
    # unseen group, a date that does not parse, a missing value.
    compare('holdout', fx['holdout_out'], fe.transform(holdout))

    # The two engines also have to agree on what was LEARNED, not only on the
    # output: the params travel with the saved model and a disagreement here
    # shows up later, on rows neither side has seen yet.
    ts_params = {s['kind']: p for s, p in zip(fx['steps'], fx['params'])}
    here = {s['kind']: s['params'] for s in fe.fitted_steps_}

    check(here.get('date_parts', {}).get('parts') == ts_params.get('date_parts', {}).get('parts'),
          f"both keep the same date parts: {here.get('date_parts', {}).get('parts')}",
          ts_params.get('date_parts', {}).get('parts'))
    check('year' not in (here.get('date_parts', {}).get('parts') or []),
          'and a part that is constant in training is dropped rather than carried')

    gs_ts = ts_params.get('group_stats', {})
    gs_py = here.get('group_stats', {})
    check(gs_py.get('groupCol') == gs_ts.get('groupCol')
          and gs_py.get('valueCol') == gs_ts.get('valueCol'),
          f"both read the same roles off the column types: group={gs_py.get('groupCol')}, "
          f"value={gs_py.get('valueCol')}", gs_ts)
    check(set((gs_py.get('groupMeans') or {}).keys()) == set((gs_ts.get('groupMeans') or {}).keys())
          and all(same(v, (gs_py.get('groupMeans') or {}).get(k))
                  for k, v in (gs_ts.get('groupMeans') or {}).items()),
          'and the same group means', gs_ts.get('groupMeans'), gs_py.get('groupMeans'))

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
