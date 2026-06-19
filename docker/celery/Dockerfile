FROM ghcr.io/astral-sh/uv:0.6.14-python3.10-bookworm

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev pkg-config \
    libjpeg-dev \
    gcc unzip zip \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY src/requirements.txt .
RUN uv pip install -r requirements.txt --system

# ── Application source ────────────────────────────────────────────────────────
COPY src/ .

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/celery/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# ── Default command: run both processing and indexing queues ──────────────────
# Override in docker-compose to run dedicated workers per queue for better isolation.
CMD ["celery", "-A", "celery_app.celery_config:celery_app", "worker", \
     "--loglevel=info", \
     "--queues=processing,indexing,default", \
     "--concurrency=2", \
     "--hostname=worker@%h"]
