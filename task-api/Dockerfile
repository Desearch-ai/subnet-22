# Build from the repository root: docker build -f task-api/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY task-api/requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 10001 crawler \
    && mkdir -p /data \
    && chown crawler /data

COPY desearch/ desearch/
COPY engine/__init__.py engine/chunking.py engine/
COPY task-api/ .

USER crawler

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
