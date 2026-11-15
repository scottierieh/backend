# Python FastAPI backend (legacy analyses not yet migrated to r-backend/) -> Cloud Run.

# Pinned by digest (the multi-arch index of python:3.11.18-slim-trixie, read
# from Docker Hub on 2026-11-15; it is what the floating python:3.11-slim tag
# resolved to that day). A floating tag gives a different base on every
# build. To update: pick the new tag, read its index digest
# (`docker buildx imagetools inspect python:<tag>`), change both here.
FROM python:3.11.18-slim-trixie@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

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

# Run as an unprivileged user (2026-11 audit L3/H2). /app stays owned by root
# and read-only for it, so nothing a request does can replace the code the
# service imports. Writable places: /tmp (per-run work dirs, AutoGluon, model
# serving) and the caches pointed there below.
RUN useradd --system --uid 10001 --home-dir /home/app --create-home --shell /usr/sbin/nologin app
ENV HOME=/home/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONDONTWRITEBYTECODE=1
USER 10001

# Runtime configuration (see docs/security-deploy.md for values and rollout order):
#   SKARI_FIREBASE_PROJECT   Firebase project id; every route except / and
#                            /health requires a Firebase ID token or the
#                            service key. Unset = every other route answers 503
#                            (fail closed) unless SKARI_AUTH_DISABLED=1 (local dev only).
#   SKARI_BACKEND_KEY        X-Skari-Service-Key secret (Secret Manager).
#   SKARI_MODEL_SIGNING_KEY  HMAC key for model artifacts (falls back to SKARI_BACKEND_KEY, logged).
#   SKARI_SCRIPT_TIMEOUT_SECS (900)  per-analysis subprocess limit -> deploy with --timeout > this (960).
#   SKARI_MAX_BODY_MB (25), SKARI_RL_SERVICE_MAX (600), SKARI_RL_IP_PER_MIN (300),
#   SKARI_RL_UID_PER_DAY (5000), SKARI_MAX_INFLIGHT_PER_UID (6) -- docs/security-deploy.md
# Deploy with --max-instances (e.g. 10): the rate limit is per instance.
# No secrets are baked into the image; inject them at deploy time.

# Cloud Run injects PORT (default 8080); uvicorn reads it via the shell-form CMD below.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
