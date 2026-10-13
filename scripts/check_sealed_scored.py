"""The sealed pairs the money screens read.

Every assertion here was run against a deliberately broken version of
`_sealed_scored` first and seen to fail; otherwise it is not a check.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autogluon_analysis as A


def fail(msg):
    print('❌ ' + msg)
    sys.exit(1)


# ── under the cap: every row, nothing marked as thinned ───────────────
score = np.array([0.1, 0.9, 0.4, 0.8])
y = np.array([0, 1, 0, 1])
got = A._sealed_scored(y, score)

if got['n'] != 4 or got['n_total'] != 4:
    fail(f"작은 홀드아웃에서 행 수가 틀립니다: {got['n']}/{got['n_total']}")
if got['sampled']:
    fail('솎아내지 않았는데 sampled가 참입니다')
if len(got['pred']) != len(got['actual']):
    fail('pred와 actual의 길이가 다릅니다')
if [round(v, 6) for v in score] != got['pred']:
    fail(f"점수가 그대로 오지 않았습니다: {got['pred']}")
if list(y) != got['actual']:
    fail(f"실제 결과가 그대로 오지 않았습니다: {got['actual']}")

# ── the pairing, which is the one thing a screen cannot recover ───────
# score encodes its own row index, actual is that index's parity: any
# mis-pairing of the two lists shows up as a broken parity.
n = A.SEALED_SCORED_MAX * 3 + 7
score = np.arange(n, dtype=float) / n
y = (np.arange(n) % 2).astype(int)
got = A._sealed_scored(y, score)

for p, a in zip(got['pred'], got['actual']):
    idx = round(p * n)
    if idx % 2 != a:
        fail(f'솎아낸 뒤 점수와 실제 결과의 짝이 어긋났습니다 (행 {idx}: 실제 {a})')

# ── past the cap: thinned, labelled, and honest about the total ───────
if got['n_total'] != n:
    fail(f"원래 행 수를 {got['n_total']}로 보고했습니다 (실제 {n})")
if not got['sampled']:
    fail('솎아냈는데 sampled가 거짓입니다 — 화면이 전수로 그립니다')
if got['n'] > A.SEALED_SCORED_MAX:
    fail(f"상한 {A.SEALED_SCORED_MAX}을 넘겨 {got['n']}쌍을 보냅니다")
if got['n'] != len(got['pred']) or len(got['pred']) != len(got['actual']):
    fail('솎아낸 뒤 n과 실제 길이가 맞지 않습니다')
# A cap that thins to a handful is as broken as one that does not thin.
if got['n'] < A.SEALED_SCORED_MAX / 2:
    fail(f"상한의 절반도 못 채우고 {got['n']}쌍만 남겼습니다")

# ── the same response twice gives the same curve ──────────────────────
if A._sealed_scored(y, score) != got:
    fail('같은 입력이 두 번 다른 결과를 냅니다 — 곡선이 실행마다 움직입니다')

# ── the proportion survives a file that arrives grouped by outcome ────
n = A.SEALED_SCORED_MAX * 2
score = np.linspace(0, 1, n)
y = np.array([0] * (n // 2) + [1] * (n // 2))
got = A._sealed_scored(y, score)
rate = sum(got['actual']) / len(got['actual'])
if abs(rate - 0.5) > 0.02:
    fail(f'결과별로 묶인 파일에서 비율이 {rate:.3f}로 틀어졌습니다 (실제 0.5)')

# ── the shape the frontend's sealedScoredOf accepts ───────────────────
for key, kind in (('n', int), ('n_total', int), ('sampled', bool),
                  ('pred', list), ('actual', list)):
    if key not in got:
        fail(f'{key} 항목이 없습니다 — 프런트가 null로 읽고 그리지 않습니다')
    if not isinstance(got[key], kind):
        fail(f'{key}의 형이 {type(got[key]).__name__}입니다 ({kind.__name__}이어야 합니다)')
if not all(isinstance(v, float) for v in got['pred']):
    fail('pred에 float이 아닌 값이 섞였습니다 — JSON으로 나가지 못합니다')
if not all(isinstance(v, int) for v in got['actual']):
    fail('actual에 int가 아닌 값이 섞였습니다 — JSON으로 나가지 못합니다')

print('✅ 봉인 쌍 검사 통과')
