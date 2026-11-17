"""t-SNE · UMAP 「설정 바꿔 보기」(embedding_sweep.py) 확인.

iris 150개 · 라벨 있음, 큰 자료(3,000개 → 1,000개 표본), 작은 자료(12개 — 설정이 겹쳐 줄어듦)에서
  현재 설정이 정확히 한 번 표시되는지, 값이 실제 한도 안인지, 비율이 0~1 인지,
  표본 크기 · 이웃 번호 범위 · 점마다 안정성 길이가 맞는지 확인합니다.
실행: cd backend && python3 validation/validate_embedding_sweep.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from embedding_sweep import settings_sweep, sweep_values, neighbours_kept

P = F = 0
def chk(name, ok):
    global P, F
    if ok: P += 1
    else: F += 1; print('FAIL', name)

tsne = lambda Xs, v: TSNE(n_components=2, perplexity=v, random_state=42, init='pca', learning_rate='auto', max_iter=500).fit_transform(Xs)
um = lambda Xs, v: umap.UMAP(n_neighbors=int(v), random_state=42).fit_transform(Xs)

# sweep_values: 현재 값이 들어가고, 줄어든 값끼리 겹치지 않음
sv = sweep_values(20, ceiling=49.0, floor=5.0)
chk('sweep_values has current', any(r == 20 for r, _ in sv))
chk('sweep_values 4 values', len(sv) == 4)
chk('sweep_values sorted by used', [u for _, u in sv] == sorted(u for _, u in sv))
sv = sweep_values(30, ceiling=5.0, floor=5.0)
chk('sweep_values collapses to one, keeps current', sv == [(30.0, 5.0)])

X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
names = np.array(['setosa', 'versicolor', 'virginica'])[y]
rng = np.random.default_rng(0)
big = np.vstack([rng.normal(0, 1, (1500, 4)), rng.normal(4, 1, (1500, 4))])

for label, Xd, lab, emb, cur, ceil, floor, n_exp in [
    ('tsne iris', Xs, names, tsne, 30, 49.0, 5.0, 150),
    ('umap iris', Xs, names, um, 15, 149.0, 2.0, 150),
    ('tsne big', big, None, tsne, 30, 333.0, 5.0, 1000),
    ('tsne tiny', Xs[:12], names[:12], tsne, 30, 5.0, 5.0, 12),
]:
    d = settings_sweep(Xd, lab, emb, cur, ceiling=ceil, floor=floor)
    chk(f'{label} n_sample', d['n_sample'] == n_exp)
    chk(f'{label} one current', sum(r['current'] for r in d['runs']) == 1)
    chk(f'{label} values within limits', all(floor <= r['value_used'] <= ceil for r in d['runs']))
    chk(f'{label} keep in [0,1]', all(0 <= r['keep'] <= 1 for r in d['runs']))
    chk(f'{label} xy normalised', all(abs(c) <= 1.0001 for r in d['runs'] for p in r['xy'] for c in p))
    chk(f'{label} xy length', all(len(r['xy']) == n_exp for r in d['runs']))
    chk(f'{label} stability length', len(d['stability']) == n_exp)
    chk(f'{label} stable_share in [0,1]', 0 <= d['stable_share'] <= 1)
    chk(f'{label} orig_nn indices', all(0 <= j < n_exp and j != i for i, row in enumerate(d['orig_nn']) for j in row))
    chk(f'{label} labels length', (d['labels'] is None) == (lab is None) and (lab is None or len(d['labels']) == n_exp))
    if label in ('tsne iris', 'umap iris'):
        chk(f'{label} 4 runs', len(d['runs']) == 4)
        chk(f'{label} silhouette present', all(r['silhouette'] is not None for r in d['runs']))
        chk(f'{label} iris mostly stable', d['stable_share'] >= 0.8)

# neighbours_kept: 같은 자료면 1, 무작위 그림이면 낮음
chk('neighbours_kept identity = 1', abs(neighbours_kept(Xs, Xs) - 1.0) < 1e-9)
chk('neighbours_kept random low', neighbours_kept(Xs, rng.normal(size=(150, 2))) < 0.3)

print(f'\n==== embedding sweep: {P} PASS, {F} FAIL ====')
sys.exit(1 if F else 0)
