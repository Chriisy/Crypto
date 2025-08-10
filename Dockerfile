# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt && pip install -r requirements-dev.txt || true

COPY . .

RUN useradd -m runner && chown -R runner:runner /app
USER runner

EXPOSE 8000
CMD sh -c 'uvicorn app.main:app --host 0.0.0.0 --port 8000 || uvicorn app:app --host 0.0.0.0 --port 8000'
