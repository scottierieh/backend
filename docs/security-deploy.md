# 인증·보안 설정과 배포 순서 (statistica-api)

2026-11 보안 수정(인증 미들웨어, 수식 평가기, 모델 아티팩트 서명, 자원 제한)과
함께 들어온 환경 변수와 배포 순서를 정리합니다. 검증은 다음 명령으로 합니다:

```
pip install httpx                    # TestClient용, 개발 환경에서만 필요
python scripts/check_security.py     # 204 ok, 0 failure(s)
python scripts/check_startup.py      # 22 ok, 0 failure(s)
```

## 환경 변수

| 변수 | 필수 | 설명 |
| --- | --- | --- |
| `SKARI_FIREBASE_PROJECT` | 운영 필수 | Firebase 프로젝트 ID(ID 토큰의 `aud`). 인증이 강제됩니다. **비어 있으면 `/`·`/health` 말고 모든 경로가 503입니다(2026-11 감사 M2, fail closed)** — 서비스 키 호출만 그대로 됩니다. |
| `SKARI_AUTH_DISABLED` | 로컬 개발만 | `1`이고 `SKARI_FIREBASE_PROJECT`가 비어 있을 때만 인증 없이 모든 요청을 허용합니다(경고 로그). 프로젝트가 설정돼 있으면 무시됩니다. 운영에 넣지 마세요. 인증이 꺼져 있어도 소유자가 기록된 모델은 그 소유자(또는 서비스 키)만 쓰고 지울 수 있습니다. |
| `SKARI_BACKEND_KEY` | 운영 필수 | Next.js 서버가 `X-Skari-Service-Key`로 보내는 비밀값. 비어 있으면 서비스 키 인증은 꺼집니다. Secret Manager로 주입하세요. |
| `SKARI_MODEL_SIGNING_KEY` | 권장 | 모델 아티팩트 HMAC 서명 키. 없으면 `SKARI_BACKEND_KEY`를 쓰고 경고를 한 번 남깁니다(Next.js 서버도 그 키를 가지고 있으므로 따로 두세요). 둘 다 없으면 프로세스마다 임의 키(재시작하면 이전 모델이 로드되지 않음). **바꾸면 기존 모델 전부가 403** — 바꿀 때는 `scripts/sign_legacy_models.py`로 재서명. |
| `SKARI_USAGE_TICKET_KEY` | 선택 | 사용권 서명 키. 없으면 `SKARI_BACKEND_KEY`(지금까지와 같음). **앱(`src/lib/usage-ticket.ts`)과 R 서버(`usage_ticket.R`)가 같은 변수를 읽게 바뀐 뒤에만, 세 곳에 함께** 넣으세요. 한 곳만 넣으면 모든 사용권이 서명 불일치로 거절됩니다. |
| `SKARI_USAGE_TICKETS` | 선택 (기본 off) | `off` / `monitor` / `on`. `on`이면 AutoGluon 실행(`/api/analysis/autogluon`)과 AutoGluon 모델 저장(`/api/models/{id}/train`, `engine: "autogluon"`)에 사용권이 필요합니다. 저장은 실행의 `automl` 사용권을 만료 후 1시간까지, 사용권 하나당 3번 받습니다. 키가 비어 있으면 `on`에서는 거절합니다. |
| `SKARI_SCRIPT_TIMEOUT_SECS` | 선택 (기본 900) | 분석 서브프로세스 하나의 최대 실행 시간. 넘으면 프로세스 그룹째 종료하고 504. AutoGluon "꼼꼼히"(600초 예산)+재학습 여유분. |
| `SKARI_MAX_BODY_MB` | 선택 (기본 25) | 요청 본문 상한. 넘으면 413. (Cloud Run 자체 상한은 HTTP/1 32 MiB) |
| `SKARI_RL_SERVICE_MAX` | 선택 (기본 600) | 서비스 키 호출 전체가 공유하는 버킷의 분당 상한. 사용자(uid)·IP 버킷은 분당 60. |
| `SKARI_RL_IP_PER_MIN` | 선택 (기본 300) | IP 하나의 분당 상한(R 서버와 같음). 인증 전에 세므로 401도 셉니다. 서비스 키는 제외. |
| `SKARI_RL_UID_PER_DAY` | 선택 (기본 5000) | 계정 하나의 하루 요청 상한(R 서버와 같음). |
| `SKARI_MAX_INFLIGHT_PER_UID` | 선택 (기본 6) | 계정 하나가 동시에 돌릴 수 있는 요청 수. 넘으면 429. 서비스 키는 `SKARI_MAX_INFLIGHT_SERVICE`(기본 16). |
| `SKARI_MODEL_CACHE_MAX` | 선택 (기본 16) | 인스턴스가 메모리에 들고 있는 모델 수(LRU). 밀려난 AutoGluon 모델의 `ag_serve_*` 디렉터리도 지웁니다. |
| `SKARI_MAX_TRAIN_ROWS` / `SKARI_MAX_FEATURES` | 선택 (300000 / 300) | `/train` 한 번의 행(data+holdout)·특성 수 상한. |
| `SKARI_MAX_PREDICT_ROWS` / `SKARI_MAX_EXPLAIN_ROWS` | 선택 (50000 / 20) | `/predict` 한 번의 행 수, `explain: true`일 때의 행 수 상한. |
| `SKARI_API_DOCS` | 로컬 개발만 | `1`이면 `/docs`·`/redoc`·`/openapi.json`을 엽니다. 기본은 닫힘(404). |
| `MODEL_ARTIFACTS_BUCKET` | 기존 | 변경 없음. |

## 요청 계약

- 브라우저: `Authorization: Bearer <Firebase ID token>` (`user.getIdToken()`).
- Next.js 서버: `X-Skari-Service-Key: <SKARI_BACKEND_KEY>`.
- `/`, `/health`, `OPTIONS`(CORS preflight)는 인증 없이 응답합니다. 그 밖은 401 `{"detail": "Authentication required."}`.
- CORS: origin은 그대로 `*`, 허용 헤더 `Authorization`, `Content-Type`, `X-Skari-Service-Key`, credentials 없음.
- 모델: `/api/models/{id}/predict`·`DELETE`는 요청 경로의 `{id}`와 `artifactUri`의
  `models/{id}/<ts>/pipeline.joblib`가 일치하고, 이 서비스가 학습 때 GCS 메타데이터에
  남긴 HMAC 서명이 맞아야만 엽니다. **`predict`와 `DELETE` 모두 학습한 uid 본인 또는
  서비스 키만 가능합니다**(2026-11 감사 H3 — 그 전에는 `modelId`+`artifactUri`만 알면
  어느 계정이든 점수를 내고 `explain`으로 학습 행을 볼 수 있었습니다). 소유자 없이 서명된
  옛 모델은 서비스 키만 씁니다. 서비스 키(공개 폼 경로) 응답의 `shapImportance`에는
  점마다의 `value`(학습 배경 행의 값)가 빠집니다.
- `/api/models/{id}/train`의 `params`는 `tune_analysis.py`가 탐색하는 하이퍼파라미터만,
  범위 안에서만 받습니다(`algorithm_registry.TUNABLE_PARAMS`). 그 밖은 400(감사 H2).
- `POST /api/models/owner-purge` `{"uid": "..."}` — 서비스 키 전용. 그 uid가 소유자로
  서명된 모델 아티팩트를 모두 지우고 `{"deleted": n, "failed": n}`을 돌려줍니다. 계정
  삭제 절차에서 부릅니다(감사 E-06). 아티팩트에는 학습 행 최대 50개(SHAP 배경, 특성 열만)와
  학습된 모델 자체(k-NN·SVM·AutoGluon 일부는 학습 데이터를 품음)가 들어 있습니다.
- 오류: 분석 스크립트가 남긴 `{"error": ...}` 문장은 그대로 400으로 갑니다. 그 밖의 실패
  (트레이스백, JSON이 아닌 출력, 처리하지 못한 예외)는 로그에 남기고 응답에는
  `(ref <id>)`만 줍니다 — 로그에서 그 id로 찾으세요.
- 분석 서브프로세스는 비밀이 빠진 환경 변수(PATH·LANG·라이브러리 설정 등)와 실행마다 새
  작업 디렉터리로 돕니다.

## 배포 순서 (이 순서를 지키세요)

1. **프론트엔드 먼저.** 모든 백엔드 호출에 ID 토큰(또는 서버 경로라면 서비스 키)을
   붙이는 프론트를 배포합니다. 이 단계에서 백엔드는 아직 강제하지 않으므로 헤더가
   있어도 없어도 동작합니다.
2. **비밀값 생성** (한 번):
   ```
   openssl rand -base64 48 | gcloud secrets create skari-backend-key --data-file=-
   openssl rand -base64 48 | gcloud secrets create skari-model-signing-key --data-file=-
   ```
   Cloud Run 서비스 계정에 `roles/secretmanager.secretAccessor`. 같은
   `skari-backend-key`를 Next.js 서버 환경 변수 `SKARI_BACKEND_KEY`에도 넣습니다.
3. **기존 모델 재서명** (서명 이전에 저장된 아티팩트는 predict가 403이 됩니다):
   ```
   SKARI_MODEL_SIGNING_KEY=<2의 값> python scripts/sign_legacy_models.py           # dry run
   SKARI_MODEL_SIGNING_KEY=<2의 값> python scripts/sign_legacy_models.py --apply
   ```
4. **백엔드 배포 — 아직 `SKARI_FIREBASE_PROJECT` 없이**, 키와 제한만 (2026-11부터는 이
   단계에서 `SKARI_AUTH_DISABLED=1`도 함께 넣어야 합니다 — 없으면 모든 경로가 503. 이미
   인증을 켠 서비스라면 이 단계는 건너뜁니다):
   ```
   gcloud run deploy statistica-api \
     --image gcr.io/restart2-98207181-3e3a5/statistica-api:YYYYMMDD \
     --region us-central1 --allow-unauthenticated \
     --memory 4Gi --cpu 4 --cpu-boost \
     --timeout 960 --max-instances 10 \
     --set-secrets SKARI_BACKEND_KEY=skari-backend-key:latest,SKARI_MODEL_SIGNING_KEY=skari-model-signing-key:latest
   ```
   - `--timeout`은 `SKARI_SCRIPT_TIMEOUT_SECS`(900)보다 커야 합니다. 작으면 Cloud Run이
     요청을 먼저 끊고, 서브프로세스는 900초까지 계속 CPU를 씁니다.
   - `--max-instances`: rate limit이 인스턴스별이므로 총 상한 = 인스턴스 수 × 분당 60.
     비용 폭주의 상한이기도 합니다.
   - `--allow-unauthenticated`는 유지합니다(인증은 앱이 합니다 — 브라우저가 직접 부르므로 IAM 인증은 쓸 수 없음).
5. 로그에서 401이 없는지, 프론트 요청에 헤더가 붙는지 확인한 뒤 **인증을 켭니다**:
   ```
   gcloud run services update statistica-api --region us-central1 \
     --update-env-vars SKARI_FIREBASE_PROJECT=restart2-98207181-3e3a5
   ```
   (값은 프론트의 `NEXT_PUBLIC_FIREBASE_PROJECT_ID`와 같아야 합니다.)
6. 확인: `curl -s -o /dev/null -w '%{http_code}' -X POST <url>/api/analysis/nps` → `401`,
   `<url>/health` → `200`.

되돌리기: 2026-11부터 `SKARI_FIREBASE_PROJECT`를 빼면 인증이 꺼지지 않고 서비스가 503으로
닫힙니다. 정말 인증을 꺼야 하면 `SKARI_AUTH_DISABLED=1`도 함께 넣어야 합니다(운영 비권장).

## 2026-11 감사 수정 배포 (영역 C, Python)

- `--set-env-vars`는 변수 전체를 바꿉니다. 이번 이미지를 올릴 때 **`SKARI_FIREBASE_PROJECT`가
  빠지지 않았는지** 먼저 확인하세요(빠지면 이제 503).
- 이미지는 비루트 사용자(uid 10001)로 돕니다. `/app`은 읽기 전용, 쓰기는 `/tmp`뿐입니다.
- AutoGluon 저장에 사용권을 요구하는 것은 `SKARI_USAGE_TICKETS=on`일 때뿐입니다. `on`으로
  바꾸기 전에 프론트엔드가 저장(`/train`) 요청에 `automl` 사용권을 붙이고, Auto Lab의
  재학습이 사용 횟수를 세도록 바뀌어 있어야 합니다(그 전에는 저장이 403).
- 사용권 사용 횟수는 인스턴스별입니다: 사용권 하나는 인스턴스마다 최대 3번 실행에 쓰일 수
  있고 재시작하면 잊습니다. `--max-instances`로 그 배수를 묶으세요.
