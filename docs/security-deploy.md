# 인증·보안 설정과 배포 순서 (statistica-api)

2026-11 보안 수정(인증 미들웨어, 수식 평가기, 모델 아티팩트 서명, 자원 제한)과
함께 들어온 환경 변수와 배포 순서를 정리합니다. 검증은 다음 명령으로 합니다:

```
pip install httpx                    # TestClient용, 개발 환경에서만 필요
python scripts/check_security.py     # 109 ok, 0 failure(s)
python scripts/check_startup.py      # 22 ok, 0 failure(s)
```

## 환경 변수

| 변수 | 필수 | 설명 |
| --- | --- | --- |
| `SKARI_FIREBASE_PROJECT` | 운영 필수 | Firebase 프로젝트 ID(ID 토큰의 `aud`). **설정하면 인증이 강제됩니다.** 비어 있으면 경고를 한 번 남기고 모든 요청을 허용합니다(로컬 개발용). |
| `SKARI_BACKEND_KEY` | 운영 필수 | Next.js 서버가 `X-Skari-Service-Key`로 보내는 비밀값. 비어 있으면 서비스 키 인증은 꺼집니다. Secret Manager로 주입하세요. |
| `SKARI_MODEL_SIGNING_KEY` | 권장 | 모델 아티팩트 HMAC 서명 키. 없으면 `SKARI_BACKEND_KEY`를 씁니다. 둘 다 없으면 프로세스마다 임의 키(재시작하면 이전 모델이 로드되지 않음). **바꾸면 기존 모델 전부가 403** — 바꿀 때는 `scripts/sign_legacy_models.py`로 재서명. |
| `SKARI_SCRIPT_TIMEOUT_SECS` | 선택 (기본 900) | 분석 서브프로세스 하나의 최대 실행 시간. 넘으면 프로세스 그룹째 종료하고 504. AutoGluon "꼼꼼히"(600초 예산)+재학습 여유분. |
| `SKARI_MAX_BODY_MB` | 선택 (기본 25) | 요청 본문 상한. 넘으면 413. (Cloud Run 자체 상한은 HTTP/1 32 MiB) |
| `SKARI_RL_SERVICE_MAX` | 선택 (기본 600) | 서비스 키 호출 전체가 공유하는 버킷의 분당 상한. 사용자(uid)·IP 버킷은 분당 60. |
| `MODEL_ARTIFACTS_BUCKET` | 기존 | 변경 없음. |

## 요청 계약

- 브라우저: `Authorization: Bearer <Firebase ID token>` (`user.getIdToken()`).
- Next.js 서버: `X-Skari-Service-Key: <SKARI_BACKEND_KEY>`.
- `/`, `/health`, `OPTIONS`(CORS preflight)는 인증 없이 응답합니다. 그 밖은 401 `{"detail": "Authentication required."}`.
- CORS: origin은 그대로 `*`, 허용 헤더 `Authorization`, `Content-Type`, `X-Skari-Service-Key`, credentials 없음.
- 모델: `/api/models/{id}/predict`·`DELETE`는 요청 경로의 `{id}`와 `artifactUri`의
  `models/{id}/<ts>/pipeline.joblib`가 일치하고, 이 서비스가 학습 때 GCS 메타데이터에
  남긴 HMAC 서명이 맞아야만 엽니다. `DELETE`는 학습한 uid 본인 또는 서비스 키만 가능합니다.
  요청·응답 모양은 바뀌지 않았습니다.

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
4. **백엔드 배포 — 아직 `SKARI_FIREBASE_PROJECT` 없이**, 키와 제한만:
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

되돌리기: `--remove-env-vars SKARI_FIREBASE_PROJECT`로 즉시 인증만 끌 수 있습니다.
