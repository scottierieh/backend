"""check_conjoint_wtp.py — willingness to pay from the conjoint routers.

    python scripts/check_conjoint_wtp.py          # under a minute, no network

Two things went wrong here, each silently:

  1. CBC with a "none" alternative. compute_wtp_cbc refitted the model with a
     linear price but without the none constant the main model carries, so
     the none row (reference levels, price 0) read as a free product people
     kept refusing. beta_price was pulled toward 0 and WTP came out ~15x too
     big (planted -0.00030 came back as -0.000015 on the data below).
  2. Won prices. RC, ACA, CBC-HB and ACBC each read a price level with
     float() over digits-and-commas, so "12,000원" never parsed and WTP was
     simply missing for every won-priced survey.

So this plants a price coefficient, generates choices from it, and asks the
routers for it back — with and without a none alternative — and reads the
price labels people actually type. Planted values, not the module's own
formula, are the reference: agreeing with itself is not evidence.

The CBC-HB endpoint is called as the coroutine it is, not through
fastapi.testclient (httpx is not in the image; see check_startup.py).
"""

import asyncio
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from price_levels import parse_price_level, price_slope  # noqa: E402
import conjoint_analysis as cbc  # noqa: E402
import conjoint_hb  # noqa: E402
import rc_analysis  # noqa: E402
import acbc_analysis  # noqa: E402

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


# ── planted CBC data (the frontend's row shape) ───────────────────────────
PRICES = [12000, 15000, 18000, 21000]
BETA_PRICE = -0.0003
ASC_NONE = -6.5
UTIL = {
    '브랜드': {'A사': 0.0, 'B사': -0.8, 'C사': -1.1},
    '배송': {'3일 이내': 0.0, '내일 도착': 0.4, '오늘 도착': 0.85},
    '품질 보증': {'1년': 0.0, '2년': 0.25},
}
ATTRS = ['가격'] + list(UTIL)


def cbc_rows(none, n_resp=300, n_tasks=10, n_alts=3, seed=7):
    """{respondent_id, task_id, alternative: 1..n | 'none', choice, <attr>: level};
    none rows carry '' for every attribute, as CBCAnalysisTab.tsx sends them."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_resp):
        for t in range(n_tasks):
            alts = []
            for _ in range(n_alts):
                p = int(rng.choice(PRICES))
                lv = {a: str(rng.choice(list(UTIL[a]))) for a in UTIL}
                alts.append((p, lv, BETA_PRICE * p + sum(UTIL[a][lv[a]] for a in UTIL)))
            v = np.array([u for *_, u in alts] + ([ASC_NONE] if none else []))
            pr = np.exp(v - v.max())
            pick = int(rng.choice(len(v), p=pr / pr.sum()))
            for j, (p, lv, _) in enumerate(alts):
                rows.append({'respondent_id': f'R{r + 1}', 'task_id': f'T{t + 1}', 'alternative': j + 1,
                             'choice': int(pick == j), '가격': f'{p:,}원', **lv})
            if none:
                rows.append({'respondent_id': f'R{r + 1}', 'task_id': f'T{t + 1}', 'alternative': 'none',
                             'choice': int(pick == n_alts), **{a: '' for a in ATTRS}})
    return rows


def cbc_wtp(rows, none):
    df, design_cols, amap, ref, _ = cbc.parse_conjoint_data(
        rows, 'choice', 'task_id', 'respondent_id', 'alternative', ATTRS, none_option=none)
    wtp, beta_price = cbc.compute_wtp_cbc(df, 'choice', '_choice_id', '가격', ATTRS[1:], amap, ref)
    return {(w['attribute'], w['level']): w['wtp'] for w in (wtp or []) if not w['is_reference']}, beta_price, design_cols


def main():
    # --- 1. price labels --------------------------------------------------
    readable = {
        '12,000원': 12000, '₩12,000': 12000, '$1,299.99': 1299.99, '월 9,900원': 9900,
        '9,900원/월': 9900, '1.2만원': 12000, '12000': 12000, '12,000 KRW': 12000,
        '1만 2천원': 12000, '만5천원': 15000, '12만9,900원': 129900, '$1.2k': 1200,
        '１２，０００円': 12000, '무료': 0, 12000: 12000, 9.9: 9.9,
    }
    wrong = {s: parse_price_level(s) for s, v in readable.items() if parse_price_level(s) != v}
    check(not wrong, f'all {len(readable)} price labels read as their amount',
          *[f'{s!r} -> {v!r} (want {readable[s]!r})' for s, v in wrong.items()])
    unreadable = ['', 'none', None, '3개월 29,700원', '1~2만원', '12,00원', '만족', float('nan'), True]
    read = {repr(s): parse_price_level(s) for s in unreadable if parse_price_level(s) is not None}
    check(not read, 'labels that are not exactly one amount stay None — a caller skips them '
          'instead of using a wrong number', *[f'{s} -> {v}' for s, v in read.items()])
    check(abs(price_slope([('12,000원', 1.0), ('15,000원', 0.1), ('18,000원', -0.8)]) - (-0.0003)) < 1e-12,
          'price_slope: utility per won from won-formatted levels')

    # --- 2. CBC WTP, planted beta_price -----------------------------------
    planted = {(a, lv): -u / BETA_PRICE for a, lvs in UTIL.items() for lv, u in lvs.items() if u}
    for none in (True, False):
        label = 'with a none alternative' if none else 'without none'
        wtp, beta_price, design_cols = cbc_wtp(cbc_rows(none), none)
        check(('_ASC_none' in design_cols) == none,
              f'CBC {label}: the main model has a none constant exactly when there are none rows')
        check(beta_price is not None and abs(beta_price / BETA_PRICE - 1) < 0.10,
              f'CBC {label}: beta_price {beta_price:.6f} within 10% of planted {BETA_PRICE}'
              if beta_price is not None else f'CBC {label}: beta_price returned')
        off = {k: (wtp.get(k), v) for k, v in planted.items()
               if k not in wtp or abs(wtp[k] - v) > max(0.35 * abs(v), 400)}
        check(not off, f'CBC {label}: every WTP within 35% (or 400원) of planted',
              *[f'{a} {lv}: got {g}, planted {p:,.0f}' for (a, lv), (g, p) in off.items()])

    # A price level that does not read must not be filled in as free.
    rows = cbc_rows(True, n_resp=40, n_tasks=4)
    for r in rows:
        if r['가격'] == '21,000원':
            r['가격'] = '21,000원 (3개월)'
    wtp, beta_price, _ = cbc_wtp(rows, True)
    check(beta_price is None and not wtp,
          'CBC: a price level that is not one readable amount turns WTP off, not into a free product')

    # --- 3. rating conjoint: won price levels give WTP ---------------------
    rng = np.random.default_rng(3)
    attributes = [
        {'id': 'p', 'name': '가격', 'levels': [{'value': f'{p:,}원'} for p in PRICES]},
        {'id': 'b', 'name': '브랜드', 'levels': [{'value': v} for v in UTIL['브랜드']]},
    ]
    profiles = [{'id': f'x{i}', 'levels': {'p': f'{p:,}원', 'b': b}}
                for i, (p, b) in enumerate((p, b) for p in PRICES for b in UTIL['브랜드'])]
    responses = []
    for r in range(60):
        ratings = {}
        for prof in profiles:
            p = parse_price_level(prof['levels']['p'])
            ratings[prof['id']] = float(4 + BETA_PRICE * (p - 16500) + UTIL['브랜드'][prof['levels']['b']]
                                        + rng.normal(0, 0.5))
        responses.append({'respondentId': f'R{r}', 'ratings': ratings})
    res = rc_analysis.run_rc_analysis(profiles, responses, attributes, price_attribute_id='p')
    pw = {p['attributeId']: {l['value']: l['utility'] for l in p['levels']} for p in res['partWorths']}
    slope = np.polyfit(PRICES, [pw['p'][f'{p:,}원'] for p in PRICES], 1)[0]
    want = {f'브랜드_{lv}': -u / slope for lv, u in pw['b'].items()}
    got = res.get('wtp') or {}
    check(set(got) == set(want) and all(abs(got[k] - want[k]) < 1e-6 for k in want),
          'rating conjoint: "12,000원" price levels give WTP = -utility / price slope',
          f'got {got}', f'want {want}')
    check(abs(slope / BETA_PRICE - 1) < 0.15,
          f'rating conjoint: price slope {slope:.6f} near planted {BETA_PRICE}')

    # --- 4. CBC-HB and ACBC ------------------------------------------------
    flat = [{'attribute': '가격', 'level': f'{p:,}원', 'value': BETA_PRICE * (p - 12000)} for p in PRICES] + \
           [{'attribute': '브랜드', 'level': lv, 'value': u} for lv, u in UTIL['브랜드'].items()]
    got = acbc_analysis._wtp(flat, '가격') or {}
    check(abs(got.get('브랜드_B사', 0) - (-UTIL['브랜드']['B사'] / BETA_PRICE)) < 1e-6,
          'ACBC: "12,000원" price levels give WTP', f'got {got}')

    hb_rows = [r for r in cbc_rows(False, n_resp=30, n_tasks=8, seed=11)]
    payload = conjoint_hb.ConjointHBPayload(data=hb_rows, attributeCols=ATTRS, priceCol='가격',
                                            draws=150, tune=150, chains=1)
    res = asyncio.run(conjoint_hb.analyze_conjoint_hb(payload))['results']
    wtp = res.get('wtp')
    keys = {'attribute', 'level', 'wtp', 'wtp_ci_lower', 'wtp_ci_upper', 'is_reference', 'label'}
    check(isinstance(wtp, list) and wtp and all(keys <= set(w) for w in wtp),
          'CBC-HB: wtp is a list of rows in the aggregate endpoint\'s shape (the CBC screen '
          'walks it as a list)', f'got {type(wtp).__name__}: {str(wtp)[:200]}')
    check(res.get('price_col') == '가격' and (res.get('beta_price') or 0) < 0
          and res.get('charts', {}).get('wtp_data') == wtp,
          'CBC-HB: price_col, a negative beta_price and charts.wtp_data come with it')
    if isinstance(wtp, list):
        bad = [w for w in wtp if not w['is_reference'] and not (w['wtp_ci_lower'] <= w['wtp'] <= w['wtp_ci_upper'])]
        check(not bad, 'CBC-HB: each WTP sits inside its own range', *[str(b) for b in bad])

    print(f'\n{_ok} ok, {_failed} failure(s)')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
