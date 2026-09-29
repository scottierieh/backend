# Auto Compare 클래스 불균형 보정 — 백엔드 (완료)

리더보드 기준을 PR-AUC로 옮긴 이유는 정확도가 "적은 쪽 클래스를 무시하는 것"에
상을 주기 때문입니다. 그런데 그 리더보드에 오르는 모델들은 클래스가 균등한 것처럼
학습되고 있었습니다. 프론트는 처음부터 모든 분류 경로에 `class_weight: "balanced"`를
보내고 있었지만, **어떤 스크립트도 그 값을 읽지 않았습니다.**

이 문서는 그 교정의 기록이자, `scripts/check_class_weighting.py`가 검사하는 표입니다.

## 왜 한 줄로 끝나지 않는가

`class_weight='balanced'`를 14개 추정기에 그대로 넘기면 절반쯤에서 `TypeError`가
납니다. 라이브러리마다 같은 개념을 다른 이름으로 받거나, 아예 받지 않습니다.
그래서 난이도는 전부 **추정기별 경로**에 있습니다. 공유 헬퍼
`analysis_common.py`의 `balanced_weighting(estimator, y_train, requested)`가
그 분기를 한곳에 모으고 `(estimator, fit_kwargs, report)`를 돌려줍니다.

선호 순서:

1. **`class_weight='balanced'`** — 추정기가 매 fit마다 스스로 가중치를 도출합니다.
   교차검증이 정직해지는 건 이 경로뿐입니다.
2. **`scale_pos_weight` / `auto_class_weights`** — 같은 개념의 라이브러리 고유 이름.
3. **`fit(sample_weight=...)`** — 여기서 `y_train`만으로 계산해 넘깁니다.
4. **`priors=uniform`** — 판별분석 전용. 단, 호출자가 `priors`를 직접 준 경우는
   건드리지 않습니다(아래 참조).

## 추정기별 경로

| 스크립트 | 추정기 | 경로 |
| --- | --- | --- |
| `random_forest_analysis.py` | RandomForestClassifier | `class_weight='balanced'` |
| `decision_tree_analysis.py` | DecisionTreeClassifier | `class_weight='balanced'` |
| `lightgbm_analysis.py` | LGBMClassifier | `class_weight='balanced'` |
| `svm_analysis.py` | SVC | `class_weight='balanced'` |
| `xgboost_analysis.py` | XGBClassifier | `scale_pos_weight=<다수/소수 비>` |
| `catboost_analysis.py` | CatBoostClassifier | `auto_class_weights='Balanced'` |
| `adaboost_analysis.py` | AdaBoostClassifier | `fit(sample_weight=...)` |
| `gbm_analysis.py` | GradientBoostingClassifier | `fit(sample_weight=...)` |
| `naive_bayes_analysis.py` | GaussianNB 등 | `fit(sample_weight=...)` |
| `discriminant_analysis.py` | LDA / QDA | `priors=uniform` |
| `knn_analysis.py` | KNeighborsClassifier | **없음** |
| `mlp_analysis.py` | MLPClassifier | **없음** |
| `ensemble_stacking_analysis.py` | Voting / Stacking | **멤버별** (아래) |
| `elastic_net_regression_analysis.py` | ElasticNet | 해당 없음 (회귀 전용) |

### 경로가 없는 둘

`KNeighborsClassifier`는 `class_weight`를 받지 않고 `fit`도 `sample_weight`를
받지 않습니다 — `weights='distance'`는 거리 가중이지 클래스 가중이 아닙니다.
sklearn의 `MLPClassifier.fit`도 `sample_weight`를 받지 않습니다. 리샘플링으로
우회할 수는 있지만 그건 각 CV fold **안에서** 일어나야 하므로 별도 설계입니다.

중요한 건 응답이 **그렇다고 말한다**는 점입니다. 가중된 10개와 가중되지 않은 2개가
같은 PR-AUC 리더보드에 섞이면 그건 두 조건에서의 비교이고, 화면은 들어야만 표시할
수 있습니다. Compare 화면은 `class_weighting.applied === false`인 행에 ‡를 답니다.

### 판별분석: 호출자의 사전확률이 우선

판별분석의 경로인 사전확률(priors)은 호출자가 **의도적으로** 설정할 수 있는
값이기도 합니다. 호출자가 준 `priors`를 균등 사전확률로 말없이 바꾸는 건 묻지
않은 질문에 답하는 것이므로, `estimator.get_params()['priors']`가 이미 설정돼
있으면 그대로 두고 `applied: False`로 사유를 적습니다.

### 앙상블: 부분 가중이 가능하다

앙상블은 `sample_weight`를 하나만 받아 **모든** 멤버에게 같은 배열을 넘깁니다.
그래서 `fit(sample_weight=)`가 유일한 경로인 멤버(GBM)는 혼자만 가중될 수 없습니다 —
앙상블 수준에서 넘기면 이미 `class_weight='balanced'`를 든 멤버들에게도 닿아
보정이 두 번 계산됩니다. 그런 멤버는 가중 없이 두고 그 사실을 적습니다.

- 파라미터 경로가 있는 멤버만 가중합니다.
- 스태킹의 메타 학습자도 **같은** 불균형 타깃에 대해 학습하므로 함께 가중합니다.
- `applied`는 **모든** 멤버가 가중됐을 때만 `true`입니다. 한 멤버라도 빠진 블렌드는
  완전히 가중된 모델과 같은 조건이 아니며, 화면의 각주는 그걸 말하려고 있습니다.
- `class_weighting.members`에 멤버별 `{applied, method, reason}`이 그대로 실립니다.

## 교차검증

`class_weight` / `scale_pos_weight` / `auto_class_weights` / `priors`는
**추정기에** 설정되므로 각 fold가 자기 fold의 가중치를 다시 도출합니다.
`sample_weight` 경로만 여기서 계산한 배열을 `run_cv(..., sample_weight=...)`로
넘기며, 추정기가 Pipeline일 때는 `f'{steps[-1][0]}__sample_weight'`로 접두사를
붙여야 합니다(붙이지 않으면 `Pipeline.fit`이 `ValueError`로 전 fold를 떨굽니다).

## 끄는 법

요청에 `class_weight: null`을 보내면 가중 없이 학습하고, 응답의
`class_weighting`은 `{applied: false, reason: '...turned off in the request'}`가
됩니다. 조용히 무시하지 않습니다.

## 측정 (900행, 소수 클래스 약 13%)

가중 적용 후 소수 클래스 recall:

| 추정기 | 가중 후 recall |
| --- | --- |
| catboost | 0.667 |
| svm | 0.625 |
| adaboost | 0.625 |
| naive_bayes | 0.500 |
| lightgbm | 0.417 |
| xgboost | 0.417 |
| discriminant (LDA/QDA) | 0.708 |
| random_forest | 0.083 |

`random_forest`가 낮은 건 가중이 안 걸려서가 아니라(걸렸습니다), 이 데이터에서
균형 가중만으로는 다수 클래스 쪽 분할 선호를 뒤집지 못하기 때문입니다. 결정
임계값 조정(`_choose_threshold`, `scripts/check_decision_threshold.py`)이 같은
문제의 다른 축이고, 그쪽에서 recall 0.188 → 0.438이 나왔습니다.

## 검사

```
python scripts/check_class_weighting.py     # 몇 초
```

39개 어서션. 추정기별 경로, CV가 가중을 fold 안에서 다시 도출하는지, 경로 없는
둘이 사유를 말하는지, 판별분석이 호출자의 priors를 남겨두는지, 앙상블이 부분
가중을 `applied: false`로 보고하는지, 그리고 `/train`이 저장하는 아티팩트의
추정기가 보고서뿐 아니라 실제로 가중을 들고 있는지까지 봅니다.
