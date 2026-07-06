FROM ghcr.io/astral-sh/uv:0.6.14-python3.10-bookworm

WORKDIR /app

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .

ENV UV_HTTP_TIMEOUT=120
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt --system

COPY frontend/ /app/frontend/

# Expose default Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
