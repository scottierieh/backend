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

# Cloud Run injects PORT (default 8080); uvicorn reads it via the shell-form CMD below.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
