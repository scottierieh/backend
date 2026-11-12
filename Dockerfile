# Python FastAPI backend (legacy analyses not yet migrated to r-backend/) -> Cloud Run.

FROM python:3.11-slim

WORKDIR /app

# build-essential is only needed while pip compiles the few packages without a
# manylinux wheel, so it is installed, used and purged inside ONE layer —
# leaving it in its own layer keeps ~400 MB in the final image forever, since
# a later `apt-get purge` cannot shrink an earlier layer. libgomp1 stays: the
# boosting libraries link against it at runtime.
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY . .

# Runtime configuration (see docs/security-deploy.md for values and rollout order):
#   SKARI_FIREBASE_PROJECT   Firebase project id; when set, every route except
#                            / and /health requires a Firebase ID token or the
#                            service key. Unset = auth disabled (local dev only).
#   SKARI_BACKEND_KEY        X-Skari-Service-Key secret (Secret Manager).
#   SKARI_MODEL_SIGNING_KEY  HMAC key for model artifacts (falls back to SKARI_BACKEND_KEY).
#   SKARI_SCRIPT_TIMEOUT_SECS (900)  per-analysis subprocess limit -> deploy with --timeout > this (960).
#   SKARI_MAX_BODY_MB (25), SKARI_RL_SERVICE_MAX (600)
# Deploy with --max-instances (e.g. 10): the rate limit is per instance.
# No secrets are baked into the image; inject them at deploy time.

# Cloud Run injects PORT (default 8080); uvicorn reads it via the shell-form CMD below.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
