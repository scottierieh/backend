# AutoML 전처리 누수 교정 — 백엔드 (완료)

원 요청: `automl-preprocessing-leakage.md` (프론트에서 전달, 2026-09-27). 아래는
그 요청에 대한 백엔드(Model Lab AutoML 경로) 쪽 조치 기록입니다.

## 대상

Model Lab의 Auto Compare가 실제로 호출하는 14개 알고리즘 스크립트
(`src/lib/model-lab/predictive-registry.ts`의 `autoCompare.route` 전부):

random_forest · xgboost · lightgbm · decision_tree · catboost · adaboost ·
naive_bayes · discriminant(lda/qda) · svm · knn · mlp · ensemble_stacking ·
gbm · elastic_net_regression

## 무엇을 고쳤나 (요청의 4개 항목 전부)

새 공유 헬퍼 `analysis_common.py`의 `leak_safe_prepare` / `leak_safe_prepare_onehot`
가 모든 스크립트에 같은 순서를 강제합니다:

```
전체 데이터 → 타깃 결측 제거 → train/holdout 분할
→ train 만으로 대치·인코딩·스케일링 통계 계산 → holdout 은 그 통계로 transform
```

- **요청 1 (파이프라인 순서)** — 대치(중앙값/최빈값), 원-핫/순서형 인코딩,
  표준화(StandardScaler) 전부 train 분할 이후, train 데이터만으로 fit 하도록
  옮겼습니다. 원-핫은 `OneHotEncoder(handle_unknown='ignore')`로 바꿔, 홀드아웃에만
  있는 새 범주는 지시변수가 전부 0인 상태로 안전하게 들어갑니다(요청이 명시한
  "unknown" 처리). LabelEncoder 한 열짜리 인코딩을 쓰던 스크립트(RF·XGBoost·
  LightGBM·DecisionTree·LDA/QDA)는 `OrdinalEncoder(handle_unknown='use_encoded_value',
  unknown_value=-1)`로, 마찬가지로 unseen 범주가 -1로 안전하게 인코딩됩니다.
  CatBoost는 범주형을 원문자열 그대로 네이티브로 다루므로(다른 스크립트와 설계가
  다름) 별도 처리: 수치형 결측만 train 중앙값으로 대치하고, 범주형 결측은
  `'missing'`이라는 명시적 범주로 채웁니다(기존 코드는 순서가 뒤바뀌어 이 처리가
  한 번도 실행되지 않는 버그가 있었고, 함께 고쳤습니다).
- **요청 2 (교차검증 내부도 동일 원칙)** — 모든 스크립트의 CV 호출을
  train 분할 이후의 `X_train_raw`(미가공)로 바꾸고, `cv_pipeline(model)`이 매 호출마다
  새 전처리기를 만들어 `Pipeline([prep, model])`으로 감쌉니다. `cross_val_score`가
  이 Pipeline을 fold마다 새로 fit 하므로, 각 fold의 검증 성능은 그 fold의 train
  쪽에서만 계산된 통계로 평가됩니다. (이전에는 CV가 outer holdout까지 포함한 전체
  데이터에서 도는 경우도 있었습니다 — 그 leak도 함께 닫았습니다.) Naive Bayes의
  multinomial/bernoulli 전용 변환(shift, 이진화 임계값)도 `_NBTypeTransform`이라는
  sklearn Transformer로 만들어 Pipeline에 끼워 넣어, fold마다 다시 계산되게
  했습니다.
- **요청 3 (`valid_mask` 제거)** — 입력변수 결측으로 행을 통째로 버리던
  `valid_mask = ~(X.isna().any(axis=1) | y.isna())` 패턴을 전부 제거했습니다.
  타깃 결측 행만 분할 전에 제외하고, 입력변수 결측은 수치형 중앙값 / 범주형
  최빈값(또는 CatBoost는 `'missing'`)으로 대치합니다.
- **요청 4 (응답에 행 수 포함)** — 모든 스크립트의 응답에 `row_counts`
  (`n_input`, `n_target_missing_dropped`, `n_train`, `n_holdout`, `n_train_used`,
  `n_holdout_used`)를 추가했습니다. 지금은 입력변수 결측을 더 이상 버리지 않으므로
  `n_train == n_train_used`, `n_holdout == n_holdout_used`가 항상 성립합니다 — 어긋나면
  회귀입니다.

## 검증

- 14개 스크립트 전부 실제 subprocess 실행(결측 있는 수치형/범주형 열, 타깃 결측,
  홀드아웃에만 나타나는 범주 포함)으로 확인 — 크래시 없음, `row_counts`가
  기대한 대로 나옴(입력변수 결측 행은 더 이상 사라지지 않고, 타깃 결측 행만
  정확히 제외됨).
- `leak_safe_prepare`/`leak_safe_prepare_onehot` 자체를 별도로 단위 테스트 —
  홀드아웃에만 있는 범주가 원-핫에서 전부-0, 순서형에서 -1로 안전하게 인코딩되는
  것을 확인.
- 분류/회귀 두 경로 모두 대표 스크립트(RF·DecisionTree·SVM·GBM)에서 별도 확인.

## 관련이지만 이번에 손대지 않은 것

- **Ridge/Lasso 단독 회귀 페이지** — AutoML 레지스트리에 없어(Auto Compare가
  호출하지 않음) 범위 밖으로 뒀습니다. `elastic_net_regression_analysis.py`와
  거의 같은 구조라 같은 패턴으로 고칠 수 있습니다.
- **`cross_validation_analysis.py` / `hyperparameter_tuning_analysis.py`** —
  범용 유틸리티 스크립트로, Auto Compare 경로가 아니라 이번 범위에서 뺐습니다.
- **Naive Bayes의 `class_prior_` 버그** — multinomial/bernoulli 모드에서
  `model.class_prior_`(Gaussian NB에만 있는 속성)를 참조해 크래시하는 기존 버그를
  발견했습니다. 이번 누수 수정과는 무관한 별개의 사전 존재 버그라 손대지
  않았습니다(원본 코드에서도 재현 확인).
