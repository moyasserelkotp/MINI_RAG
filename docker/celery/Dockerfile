FROM ghcr.io/astral-sh/uv:0.6.14-python3.10-bookworm

WORKDIR /app

# System dependencies 
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

# Python dependencies 
COPY src/requirements.txt .
ENV UV_HTTP_TIMEOUT=120
ENV UV_COMPILE_BYTECODE=1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt --system
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flower==2.0.1 --system

# Application source — copy AFTER deps so src/ edits don't bust the pip cache
COPY src/ .

# Entrypoint 
COPY docker/celery/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

#  Default command: run both processing and indexing queues 
# Override in docker-compose to run dedicated workers per queue for better isolation.
CMD ["celery", "-A", "celery_app.celery_config:celery_app", "worker", \
        "--loglevel=info", \
        "--queues=processing,indexing,default", \
        "--concurrency=2", \
        "--hostname=worker@%h"]
