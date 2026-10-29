FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY web_server.py benchmark_rbac.py ./
RUN mkdir -p /app/workspace/uploads

RUN pip install --upgrade pip && pip install .

EXPOSE 8090

HEALTHCHECK --interval=10s --timeout=5s --retries=6 \
    CMD curl -fsSL http://localhost:8090/api/health || exit 1

CMD ["python", "web_server.py"]
