# Python FastAPI backend (legacy analyses not yet migrated to r-backend/) -> Cloud Run.

FROM python:3.11-slim

WORKDIR /app

# build tools needed by catboost/lightgbm/tensorflow wheels on some platforms
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects PORT (default 8080); uvicorn reads it via the shell-form CMD below.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
