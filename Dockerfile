FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY app ./app
# scripts/ 一并打入镜像，容器内可直接运行线上自检：
#   docker compose exec agent-memory python -m scripts.run_compliance_e2e.py
COPY scripts ./scripts
RUN mkdir -p /app/data/model_cache

EXPOSE 8000

# 单进程避免在轻量主机上为每个 worker 重复加载一份向量模型。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
