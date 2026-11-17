"""t-SNE · UMAP 「설정 바꿔 보기」 — 같은 자료 · 같은 시드로 설정값만 바꿔 4번 그리고, 그림마다
원래 이웃이 얼마나 남았는지와 점마다 설정이 바뀌어도 같은 이웃과 붙어 있는지를 잽니다.

결과 화면의 탐색 시뮬레이션 탭이 따로 부릅니다(payload 에 sweep: true). 본 분석은 느려지지 않습니다.
보고서 본문에는 neighbours_kept() 한 숫자만 들어갑니다.

  neighbours_kept(X, E, k)      원래 공간의 가까운 k 개 중 그림에서도 가까운 k 개에 든 비율(평균, 점마다)
  settings_sweep(...)           4개 설정값의 그림 · 이웃 유지율 · 라벨 분리도 · 점마다 안정성
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

K_NEIGHBOURS = 10
SWEEP_MAX_N = 1000
# 「안정적인 점」 — 네 그림 모두에서, 원래 가까운 이웃 10개 중 절반 이상이 그 점 주변(그림에서 가장
# 가까운 5% · 적어도 15개 안)에 남아 있는 점. 딱 10개끼리 맞추면 큰 덩어리 안의 자리 바뀜까지
# 「불안정」으로 세어, 무리가 분명히 유지되는 자료도 낮게 나옵니다.
STABLE_SHARE = 0.5
STAB_WINDOW_SHARE = 0.05
STAB_WINDOW_MIN = 15
BASE_VALUES = [5, 15, 30, 50]


def _knn(M, k, metric='euclidean'):
    k = min(k, len(M) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm='auto' if metric == 'euclidean' else 'brute')
    idx = nn.fit(M).kneighbors(M, return_distance=False)
    # 똑같은 행이 있으면 자기 자신이 첫 칸이 아닐 수 있습니다 — 첫 칸을 버리지 말고 자기 번호를 빼고 k 개.
    return np.array([[j for j in row if j != i][:k] for i, row in enumerate(idx)])


def neighbours_kept(X, E, k=K_NEIGHBOURS, max_n=5000, seed=42, metric='euclidean'):
    """원래 이웃 유지율(0~1). 점이 많으면 max_n 개를 뽑아서 잽니다. 점이 3개 미만이면 None."""
    X = np.asarray(X, dtype=float); E = np.asarray(E, dtype=float)
    n = len(X)
    if n < 3:
        return None
    if n > max_n:
        idx = np.random.default_rng(seed).choice(n, max_n, replace=False)
        X, E = X[idx], E[idx]
    a, b = _knn(X, k, metric), _knn(E, k)
    kk = a.shape[1]
    return float(np.mean([len(set(a[i]) & set(b[i])) / kk for i in range(len(a))]))


def sweep_values(current, ceiling, floor):
    """5·15·30·50 에서 현재 값에 가장 가까운 칸을 현재 값으로 바꾸고, 표본 크기 한도로 줄입니다.
    줄여서 겹치면 하나만 남깁니다 — (요청값, 실제값) 목록."""
    base = list(BASE_VALUES)
    cur = float(current)
    if cur not in base:
        j = int(np.argmin([abs(b - cur) for b in base]))
        base[j] = cur
    clamp = lambda v: float(min(max(v, floor), ceiling))
    cur_used = clamp(cur)
    out, seen = [(cur, cur_used)], {cur_used}   # 현재 값이 먼저 자리를 잡습니다(줄여서 겹쳐도 남게).
    for v in base:
        used = clamp(v)
        if used in seen:
            continue
        seen.add(used)
        out.append((v, used))
    return sorted(out, key=lambda t: t[1])


def settings_sweep(X, labels, embed, current, ceiling, floor, seed=42, max_n=SWEEP_MAX_N, metric='euclidean'):
    """embed(Xs, value) -> 2차원 좌표. labels 는 X 와 같은 길이(없으면 None).

    돌려주는 것(JSON 그대로):
      n_total, n_sample, sample_index   뽑은 행(원래 순서 기준 번호)
      labels                            뽑은 행의 라벨(문자열) 또는 None
      orig_nn                           뽑은 표본 안에서 원래 공간의 가까운 10개(번호)
      runs[]                            value_requested · value_used · current · xy(-1~1) · keep · silhouette
      stability[]                       점마다, 원래 이웃 10개 중 그림 주변(window)에 남은 비율 — 네 그림 중 가장 낮은 값
      stable_share / window             stability >= 0.5 인 점의 비율 / 「주변」으로 본 이웃 수
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))
    else:
        idx = np.arange(n)
    Xs = X[idx]
    lab = None
    if labels is not None:
        lab = [None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in np.asarray(labels, dtype=object)[idx]]

    orig = _knn(Xs, K_NEIGHBOURS, metric)
    kk = orig.shape[1]
    runs, neigh = [], []
    window = int(min(len(Xs) - 1, max(STAB_WINDOW_MIN, round(STAB_WINDOW_SHARE * len(Xs)))))
    for req, used in sweep_values(current, ceiling, floor):
        E = np.asarray(embed(Xs, used), dtype=float)[:, :2]
        nb = _knn(E, K_NEIGHBOURS)
        neigh.append(_knn(E, window))
        keep = float(np.mean([len(set(orig[i]) & set(nb[i])) / kk for i in range(len(Xs))]))
        sil = None
        if lab is not None:
            ok = [i for i, v in enumerate(lab) if v is not None]
            if len(set(lab[i] for i in ok)) >= 2 and len(ok) > len(set(lab[i] for i in ok)):
                try:
                    sil = float(silhouette_score(E[ok], [lab[i] for i in ok]))
                except Exception:
                    sil = None
        C = E - E.mean(0)
        m = float(np.abs(C).max()) or 1.0
        runs.append({
            'value_requested': float(req),
            'value_used': used,
            'current': float(req) == float(current),
            'xy': np.round(C / m, 3).tolist(),
            'keep': keep,
            'silhouette': sil,
        })

    # 점마다 가장 나쁜 그림 기준: 원래 이웃 중 그 그림에서 주변(window)에 남은 비율의 최솟값.
    stab = [min(len(set(orig[i]) & set(nb[i])) / kk for nb in neigh) for i in range(len(Xs))]
    stab = np.round(stab, 2)
    return {
        'n_total': int(n),
        'n_sample': int(len(Xs)),
        'sample_index': idx.tolist(),
        'labels': lab,
        'orig_nn': orig.tolist(),
        'k': int(kk),
        'runs': runs,
        'stability': stab.tolist(),
        'stable_share': float(np.mean(stab >= STABLE_SHARE)),
        'stable_threshold': STABLE_SHARE,
        'window': window,
    }
